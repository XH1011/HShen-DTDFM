import os
import importlib
import pandas as pd
from scipy.io import loadmat

import aug
import data_utils
import load_methods


def get_files(root, dataset, faults, signal_size, condition=2):
    data, labels = [], []
    data_load = getattr(load_methods, dataset)

    for index, name in enumerate(faults):
        data_dir = os.path.join(root, 'condition_%d' % condition, name)

        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            signal = data_load(item_path)

            if signal is None or len(signal) == 0:
                continue

            # 确保 signal_size 合理，避免切割超过信号长度
            start = 0
            while start + signal_size <= len(signal):  # 修改条件，确保切割不会超出范围
                end = start + signal_size
                data.append(signal[start:end])
                labels.append(index)
                start += signal_size

    print(f"Loaded {len(data)} samples with {len(labels)} labels.")
    return data, labels


def data_transforms(normlize_type="-1-1"):
    transforms = {
        'train': aug.Compose([
            aug.Reshape(),
            aug.Normalize(normlize_type),
            aug.Retype()

        ]),
        'val': aug.Compose([
            aug.Reshape(),
            aug.Normalize(normlize_type),
            aug.Retype()
        ])
    }
    return transforms


class dataset(object):

    def __init__(self, data_dir, dataset, faults, signal_size, normlizetype, condition=2,
                 balance_data=False, test_size=0.2):
        self.balance_data = balance_data
        self.test_size = test_size
        self.num_classes = len(faults)
        self.data, self.labels = get_files(root=data_dir, dataset=dataset, faults=faults, signal_size=signal_size, condition=condition)
        self.transform = data_transforms(normlizetype)

    def data_preprare(self, source_label=None, is_src=False, random_state=1):
        data_pd = pd.DataFrame({"data": self.data, "labels": self.labels})
        data_pd = data_utils.balance_data(data_pd) if self.balance_data else data_pd
        print(f"Prepared data size: {len(data_pd)}")  # 打印准备后的数据大小
        if is_src:
            train_dataset = data_utils.dataset(list_data=data_pd, source_label=source_label,
                                               transform=self.transform['train'])
            print(f"Train dataset size: {len(train_dataset)}")  # 打印训练集大小
            return train_dataset
        else:
            train_pd, val_pd = data_utils.train_test_split_(data_pd, test_size=self.test_size,
                                                            num_classes=self.num_classes, random_state=random_state)
            print(f"Train set size: {len(train_pd)}, Validation set size: {len(val_pd)}")  # 打印训练集和验证集的大小

            # 假设 0 是正常类，提取正常类数据
            normal_class = train_pd[train_pd['labels'] == 0]
            # 提取故障类数据（非正常类数据）
            fault_classes = train_pd[train_pd['labels'] != 0]
            # 对故障类数据进行截取，只保留前 10 个样本（每类故障各 10 个）
            fault_classes = fault_classes.groupby('labels').head(10)
            # 将筛选后的正常类和故障类数据合并
            train_pd = pd.concat([normal_class, fault_classes], ignore_index=True)
            train_dataset = data_utils.dataset(list_data=train_pd, source_label=source_label,
                                               transform=self.transform['train'])

            val_dataset = data_utils.dataset(list_data=val_pd, source_label=source_label,
                                             transform=self.transform['val'])

            print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")  # 打印数据集大小
            return train_dataset, val_dataset