# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import math
import os
from configparser import ConfigParser

from dataset.dataset_factory import get_dataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
from pathlib import Path
from typing import Iterable
import datetime
import torch.nn as nn
import numpy as np
import torch
from torch.backends import cudnn

from Vivim.read_configs import bootstrap
from utils import misc, lr_sched

from model.model_factory import get_models

from environment_setup import PROJECT_ROOT_DIR

from utils.misc import NativeScalerWithGradNormCount as NativeScaler

from torch.utils.tensorboard import SummaryWriter
import timm.optim.optim_factory as optim_factory
import time
import json

import torchio as tio


# def train_one_stage_epoch(model: torch.nn.Module,
#                           data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                           device: torch.device, epoch: int, loss_scaler,
#                           log_writer=None,
#                           args=None):
#     model.train(True)
#     metric_logger = misc.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 20
#     criterion = torch.nn.CosineSimilarity(dim=1).to(device)
#     accum_iter = args.accum_iter

#     optimizer.zero_grad()
#     if log_writer is not None:
#         print('log_dir: {}'.format(log_writer.log_dir))

#     for data_iter_step, original_volume in enumerate(
#             metric_logger.log_every(data_loader, print_freq, header)):

#         # we use a per iteration (instead of per epoch) lr scheduler
#         if data_iter_step % accum_iter == 0:
#             lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

#         original_volume = original_volume.to(device, non_blocking=True)
#         sys.stdout.flush()
#         with torch.cuda.amp.autocast(enabled=False):
#              pred = model(x=original_volume)

#         # This is based on our modification for weighted loss
#         reconstruction_loss = loss
#         loss_value = loss.item()
#         metric_logger.update(reconstruction_loss=reconstruction_loss)
#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             sys.exit(1)

#         loss /= accum_iter
#         loss_scaler(loss, optimizer, parameters=model.parameters(),
#                     update_grad=(data_iter_step + 1) % accum_iter == 0)
#         if (data_iter_step + 1) % accum_iter == 0:
#             optimizer.zero_grad()

#         torch.cuda.synchronize()

#         metric_logger.update(loss=loss_value)

#         lr = optimizer.param_groups[0]["lr"]
#         metric_logger.update(lr=lr)

#         loss_value_reduce = misc.all_reduce_mean(loss_value)
#         #  Our extra logs
#         reconstruction_loss_value_reduce = misc.all_reduce_mean(reconstruction_loss)
#         if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
#             """ We use epoch_1000x as the x-axis in tensorboard.
#             This calibrates different curves when batch size changes.
#             """
#             epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
#             log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
#             log_writer.add_scalar('lr', lr, epoch_1000x)

#             log_writer.add_scalar('reconstruction_loss', reconstruction_loss_value_reduce, epoch_1000x)
#         # Deleting the variables. Perhaps this allows us to use memory better
#         del original_volume
#         torch.cuda.empty_cache()

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_stage_epoch(model: torch.nn.Module,
                          data_loader: Iterable, optimizer: torch.optim.Optimizer, criterion, 
                          device: torch.device, epoch: int, loss_scaler,
                          log_writer=None,
                          args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
            print('log_dir: {}'.format(log_writer.log_dir))

    # we use a per iteration (instead of per epoch) lr scheduler
    for data_iter_step, (input_image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    # Adjust the learning rate per iteration if needed
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        input_image = input_image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(input_image)
            loss = criterion(output, target)

        # Check for non-finite loss (e.g., NaN or Inf)
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}        



def compute_contrastive_loss(args, criterion, p1, p2, z1, z2):
    return args.contr_weight * (-(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='contr_mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of channels in the input')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR',  # earlier 0
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--dist_on_itp', action='store_true')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    transforms = [
        tio.RandomAffine(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    transformations = tio.Compose(transforms)

    args = bootstrap(args=args, key='SETUP')
    dataset_train = get_dataset(dataset_name=args.dataset, mode=args.mode, args=args, transforms=transformations,
                                use_z_score=args.use_z_score)
    print(dataset_train)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # if global_rank == 0 and args.log_dir is not None:
    if args.log_dir is not None:
        log_dir = os.path.join(PROJECT_ROOT_DIR, args.log_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = get_models(model_name='autoenc', args=args)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    args.output_dir = os.path.join(PROJECT_ROOT_DIR, args.output_dir, 'checkpoints')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    min_loss = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        # loss weighting for the edge maps
        edge_map_weight = 0.01 * (1 - epoch / args.epochs)
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_stage_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            edge_map_weight=edge_map_weight
        )

        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        if train_stats['loss'] < min_loss:
            min_loss = train_stats['loss']
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch="min_loss")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
