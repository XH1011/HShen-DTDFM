import random
import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import torch.nn as nn
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import f1_score



import utils
import model_base
from train_utils import InitTrain


class MixFeature(nn.Module):
    
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        batch_size = x.size(0)

        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        sigma = (var + self.eps).sqrt()
        mu, sigma = mu.detach(), sigma.detach()
        x_normed = (x - mu) / sigma

        interpolation = self.beta.sample((batch_size, 1, 1))
        interpolation = interpolation.to(x.device)

        # split into two halves and swap the order
        perm = torch.arange(batch_size - 1, -1, -1)  # inverse index
        perm_b, perm_a = perm.chunk(2)
        perm_b = perm_b[torch.randperm(batch_size // 2)]
        perm_a = perm_a[torch.randperm(batch_size // 2)]
        perm = torch.cat([perm_b, perm_a], 0)

        mu_perm, sigma_perm = mu[perm], sigma[perm]
        mu_mix = mu * interpolation + mu_perm * (1 - interpolation)
        sigma_mix = sigma * interpolation + sigma_perm * (1 - interpolation)

        return x_normed * sigma_mix + mu_mix


class MixFeatureLayer(nn.Module):

    def __init__(self,
                 in_channel=1, kernel_size=8, stride=1, padding=4,
                 mp_kernel_size=2, mp_stride=2, dropout=0.):
        super(MixFeatureLayer, self).__init__()

        layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 4, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer2 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer3 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer4 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer5 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
            nn.Flatten())
        
        dp = nn.Dropout(dropout)
        mixfeature = MixFeature()
        
        self.fs = nn.Sequential(
            layer1,
            mixfeature,
            layer2,
            mixfeature,
            layer3,
            layer4,
            dp,
            layer5)

    def forward(self, tar, x=None, y=None):
        h = self.fs(tar)
        
        return h


class Trainset(InitTrain):

    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 1280
        self.G = model_base.FeatureExtractor(in_channel=1, block=MixFeatureLayer, dropout=args.dropout).to(self.device)
        self.C = model_base.ClassifierMLP(input_size=output_size, output_size=args.num_classes,
                                          dropout=args.dropout, last=None).to(self.device)
        self._init_data(concat_src=True)

    def save_model(self):
        torch.save({
            'G': self.G.state_dict(),
            'C': self.C.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))

    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.G.load_state_dict(ckpt['G'])
        self.C.load_state_dict(ckpt['C'])

    def train(self):
        args = self.args

        # 打印网络结构
        print("Feature Extractor Structure:")
        print(self.G)
        print("\nClassifier Structure:")
        print(self.C)

        self.optimizer = self._get_optimizer([self.G, self.C])
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        best_acc = 0.0
        best_epoch = 0
        best_f1 = 0.0

        # 用于累积 val 数据
        val_acc_list = []
        val_f1_list = []

        for epoch in range(1, args.max_epoch+1):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-'*5)

            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            # Each epoch has a training and val phase
            epoch_acc = defaultdict(float)
            epoch_f1 = defaultdict(float)

            # Set model to train mode or evaluate mode
            self.G.train()
            self.C.train()
            epoch_loss = defaultdict(float)

            num_iter = len(self.dataloaders['train'])

            for i in tqdm(range(num_iter), ascii=True):
                source_data, source_labels = utils.get_next_batch(self.dataloaders,
            						     self.iters, 'concat_source', self.device)
                # forward
                self.optimizer.zero_grad()
                f = self.G(source_data)
                pred = self.C(f)

                loss = F.cross_entropy(pred, source_labels)
                epoch_acc['Source Data']  += utils.get_accuracy(pred, source_labels)
                epoch_acc['Source Data']  +=utils.get_f1_score(source_labels,pred)


                epoch_loss['Source Classifier'] += loss

                # backward
                loss.backward()
                self.optimizer.step()

            # Print the train and val information via each epoch
            for key in epoch_loss.keys():
                logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/num_iter))
            for key in epoch_acc.keys():
                logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key]/num_iter))
            for key in epoch_f1.keys():
                logging.info('Train-F1 {}: {:.4f}'.format(key, epoch_f1[key]/num_iter))


            # 验证阶段
            new_acc, new_f1 = self.test()

            # 累积验证结果
            val_acc_list.append(new_acc)
            val_f1_list.append(new_f1)

            new_acc, new_f1 = self.test()  # 获取准确率和 F1 分数
            if new_acc >= best_acc:
                best_acc = new_acc
                new_f1 >= best_f1
                best_f1 = new_f1  # 可选，记录最佳 F1 分数
                best_epoch = epoch
            # logging.info("The best model epoch {}, val-acc {:.4f}, val-f1 {:.4f}".format(best_epoch, best_acc, best_f1))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        # 训练完成后计算总的平均值
        avg_val_acc = sum(val_acc_list) / len(val_acc_list)
        avg_val_f1 = sum(val_f1_list) / len(val_f1_list)

        # 输出总的平均值
        logging.info("Training completed.")
        logging.info("Average Val-Acc Target Data: {:.4f}".format(avg_val_acc))
        logging.info("Average Val-F1 Target Data: {:.4f}".format(avg_val_f1))
        logging.info("Best Epoch: {}, Best Val-Acc: {:.4f}, Best Val-F1: {:.4f}".format(
            best_epoch, best_acc, best_f1
        ))





    def test(self):
        self.G.eval()
        self.C.eval()
        acc = 0.0
        f1 =0.0

        iters = iter(self.dataloaders['val'])
        num_iter = len(iters)
        with torch.no_grad():
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels, _ = next(iters)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                pred = self.C(self.G(target_data))

                acc += utils.get_accuracy(pred, target_labels)
                f1 += utils.get_f1_score(target_labels, pred)
        acc /= num_iter
        f1 /= num_iter



        logging.info('Val-Acc Target Data: {:.4f}'.format(acc))
        logging.info('Val-F1 Target Data: {:.4f}'.format(f1))
        return acc, f1