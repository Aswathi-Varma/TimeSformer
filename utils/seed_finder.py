import os
import argparse
from statistics import stdev
import math
import sys
from sklearn.model_selection import KFold, train_test_split
sys.path.append("/root/seg_framework/MS-Mamba")
import numpy as np
import torch
from torch.backends import cudnn
from environment_setup import PROJECT_ROOT_DIR
from read_configs import bootstrap
from utils import misc, lr_sched
from dataset.DatasetSeg import DatasetSeg
from dataset.dataset_utils import Phase
import torchio as tio
from utils.metric import precision, recall, dice_score    
import wandb
sys.stdout = open("/root/seg_framework/MS-Mamba/sys_out", "w")

def get_args_parser():
    parser = argparse.ArgumentParser('MSLesSeg training', add_help=False)

    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model parameters
    parser.add_argument('--model', default='SegFormer3D', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of channels in the input')
   
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR',  # earlier 0
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--dist_on_itp', action='store_true')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    # Distributed
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--embed_dim', default=1, type=int,
                        help='number of embeddings')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Dataset parameters
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--data_dir', default='/root/MSLesSeg24/data', type=str,
                        help='path to dataset')
    parser.add_argument('--datalist', default=None, type=list,
                        help='datalist')
    parser.add_argument('--modalities', default='T1,T2,FLAIR,T1ce', type=str,
                        help='modalities')
    parser.add_argument('--preprocess', default=False, type=bool,
                        help='preprocess')

    return parser

def main(args):

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))), force = True)
    print("{}".format(args).replace(', ', ',\n'), force = True)
    
    device = torch.device(args.device)
    print("device ",device, force = True)
    
    #args initialization
    args = bootstrap(args=args, key='MSLESSEG')

    # Hard-coding the in channels
    args.in_channels = 2
    args.nb_classes = 1
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prepare train test splits
    datafolders = np.load('/root/seg_framework/MS-Mamba/dataset/patients.npy')

    def evaluate_seed(seed):
        kfold_splits = KFold(n_splits=5, random_state=seed, shuffle=True)
        train_count, val_count, test_count = [], [], []

        for idx, (train_ids, test_ids) in enumerate(kfold_splits.split(datafolders)):
            # Split the train_ids into train and validation
            train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42, shuffle=True)

            # Get the train, val, and test sets
            train_set = [datafolders[i] for i in train_ids]
            val_set = [datafolders[i] for i in val_ids]
            test_set = [datafolders[i] for i in test_ids]

            train_dataset = DatasetSeg(data_dir=args.data_dir, datalist=train_set, batch_size=args.batch_size, 
                               phase=Phase.TRAIN, transforms=ModuleNotFoundError, preprocess=args.preprocess)
            val_dataset = DatasetSeg(data_dir=args.data_dir, datalist=val_set, batch_size=args.batch_size,
                                 phase=Phase.VAL, transforms=ModuleNotFoundError, preprocess=args.preprocess)
            test_dataset = DatasetSeg(data_dir=args.data_dir, datalist=test_set, batch_size=args.batch_size,
                                    phase=Phase.TEST, transforms=ModuleNotFoundError, preprocess=args.preprocess)

            trian_dirs = train_dataset.data_dir_paths
            val_dirs = val_dataset.data_dir_paths
            test_dirs = test_dataset.data_dir_paths  

            # Prepare data
            train_n = len(trian_dirs)
            val_n = len(val_dirs)
            test_n = len(test_dirs)

            train_count.append(train_n)
            val_count.append(val_n)
            test_count.append(test_n)

        # Calculate the standard deviation of counts
        train_std = stdev(train_count)
        val_std = stdev(val_count)
        test_std = stdev(test_count)

        return train_std + val_std + test_std, train_count, val_count, test_count

    # Iterate over a range of seeds to find the best one
    best_seed = None
    best_std = float('inf')
    best_train, best_val, best_test = None, None, None
    for seed in range(10000):  # You can adjust the range as needed
        current_std, train_count, val_count, test_count = evaluate_seed(seed)
        if current_std < best_std:
            best_std = current_std
            best_seed = seed
            best_train = train_count
            best_val = val_count
            best_test = test_count

    print(f"Best seed: {best_seed} with standard deviation: {best_std}")
    print("Train count: ", best_train)
    print("Val count: ", best_val)
    print("Test count: ", best_test)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    wandb.finish()