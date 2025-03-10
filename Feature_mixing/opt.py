import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='From github.com/XH1011/HShen-DTFM')
 
    # basic parameters
    parser.add_argument('--model_name', type=str, default='MixFeature',
                        help='Name of the model (in ./models directory)')
    parser.add_argument('--source', type=str, default='GS_1,GS_3,GS_4',
                        help='Source data, separated by "," (select specific conditions of the dataset with name_number, such as GS_1)')
    parser.add_argument('--target', type=str, default='GS_2',
                        help='Target data (select specific conditions of the dataset with name_number, such as GS_1)')
    parser.add_argument('--data_dir', type=str, default="./datasets",
                        help='Directory of the datasets')
    parser.add_argument('--train_mode', type=str, default='multi_source',
                        choices=['single_source', 'source_combine', 'multi_source'],
                        help='Training mode (select correctly before training)')
    parser.add_argument('--cuda_device', type=str, default='0',
                        help='Allocate the device to use only one GPU ('' means using cpu)')
    parser.add_argument('--save_dir', type=str, default='./ckpt',
                        help='Directory to save logs and model checkpoints')
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--signal_size', type=int, default=1024,
                        help='Signal length split by sliding window')
    parser.add_argument('--random_state', type=int, default=1,
                        help='Random state for the entire training')

    # optimization information
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '-1-1', 'mean-std'], default='-1-1',
                        help='Data normalization methods')
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for sgd')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Betas for adam')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for both sgd and adam')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='stepLR',
                        help='Type of learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='Parameter for the learning rate scheduler (except "fix")')
    parser.add_argument('--steps', type=str, default='10',
                        help='Step of learning rate decay for "step" and "stepLR"')
    parser.add_argument('--tradeoff', type=list, default=['exp', 'exp', 'exp'],
                        help='Trade-off coefficients for the sum of losses, integer or "exp" ("exp" represents an increase from 0 to 1)')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout layer coefficient')
    
    # save and load
    parser.add_argument('--save', type=bool, default=True, help='Save logs and trained model checkpoints')
    parser.add_argument('--load_path', type=str, default='',
                        help='Load trained model checkpoints from this path (for testing, not for resuming training)')
    args = parser.parse_args()
    return args
    
