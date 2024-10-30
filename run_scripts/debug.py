import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4,6,7"
import argparse
import json
import pickle
import random
import time
import math
import datetime
import sys
import nibabel
from scipy.ndimage import rotate
from sklearn.model_selection import KFold, train_test_split
import tqdm
sys.path.append("/root/seg_framework/MS-Mamba")
import numpy as np
import torch
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from environment_setup import PROJECT_ROOT_DIR
from read_configs import bootstrap
from utils import misc, lr_sched
from dataset.DatasetSeg2D import DatasetSeg2D
from dataset.dataset_utils import Phase
import torchio as tio
from model.Vivim2D import Vivim2D
from utils.metric import precision, recall, dice_score    
from utils.vivim_loss import dual_loss, structure_loss
import wandb
from model.SimpleModel import SimpleSegmentationModel
from torch import nn, optim

sys.stdout = open("/root/seg_framework/MS-Mamba/sys_out_1", "w")

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
    parser.add_argument('--mask_mode', default='concatenate to image',
                        help='mask mode')

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
    parser.add_argument('--preprocess', default=False, type=bool,
                        help='preprocess')
    parser.add_argument('--dim', default=2, type=int,
                        help='dimension of the data')
    parser.add_argument('--loss', default='mask tp1 tp2', type=str,
                        help='resume from checkpoint')

    return parser

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
                    max_norm=0, log_writer=None, args=None, mix_up_fn=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    for data_iter_step, (samples, targets_tp1, targets_tp2) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets_tp2 = targets_tp2.to(device, non_blocking=True)
        targets_tp1 = targets_tp1.to(device, non_blocking=True)

        # Check for NaN and all zeros in inputs
        if torch.isnan(samples).any():
            print(f"NaN detected in samples at step {data_iter_step}")
        if (samples == 0).all():
            print(f"All zeros detected in samples at step {data_iter_step}")

        if torch.isnan(targets_tp1).any():
            print(f"NaN detected in targets_tp1 at step {data_iter_step}")
        if (targets_tp1 == 0).all():
            print(f"All zeros detected in targets_tp1 at step {data_iter_step}")

        if torch.isnan(targets_tp2).any():
            print(f"NaN detected in targets_tp2 at step {data_iter_step}")
        if (targets_tp2 == 0).all():
            print(f"All zeros detected in targets_tp2 at step {data_iter_step}")


        if mix_up_fn is not None:
            samples, targets_tp2 = mix_up_fn(samples, targets_tp2)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets_tp1, targets_tp2)

        loss_value = loss.item()

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

def get_avg_seg_volume(output_dict, data_shape):
    axis_volumes = torch.zeros([3, *data_shape])
    for i in range(len(output_dict)):
        axis_volume = torch.squeeze(output_dict[i])
        if i == 1:
            rotated_axis_volume = axis_volume.permute(1, 0, 2)
        elif i == 2:
            rotated_axis_volume = axis_volume.permute(1, 2, 0)
        else:
            rotated_axis_volume = axis_volume
        axis_volumes[i] = rotated_axis_volume

    # Some explanations for the following line:
    # for axis_volumes we only used the predictions for the 1 label. By building the mean over all values up and rounding this we get the value 1
    # for those where the label 1 has the majority in softmax space, else 0. This exactly corresponds to our prediction as we would have taken argmax.
    return torch.unsqueeze(axis_volumes.mean(dim=0), dim=0)

@torch.no_grad()
def evaluate_test(data_loader, model, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    metric_fns = [precision, recall, dice_score]
    res = 218

    total_metrics = torch.zeros(len(metric_fns))
    patient_metrics = torch.zeros(len(metric_fns))
    all_patient_metrics = []  # To store metrics for each patient

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        # setup
        n_samples = 0
        axis = 0  # max 2
        c = 0
        data_shape = [res, res, res]

        output_agg = torch.zeros([3, *data_shape]).to(device)
        avg_seg_volume = None
        target_agg = torch.zeros(data_shape).to(device)

        for batch in metric_logger.log_every(data_loader, 10, header):
            data, mask, target = batch
            data, mask, target = data.to(device), mask.to(device), target.to(device)
            output = model(data)

            # output to probability
            output=output[:,-1,:,:,:]
            output = torch.sigmoid(output)

            for slice_output, slice_target in zip(output, target):
                output_agg[axis][c % res] = torch.unsqueeze(slice_output.float(), dim=0)

                if axis == 0:
                    target_agg[c % res] = slice_target.float()

                c += 1

                if not c % res and c > 0:
                    axis += 1
                    if not axis % 3 and axis > 0:
                        path = os.path.join(PROJECT_ROOT_DIR, 'output_dir', 'mslesseg', 'test')
                        os.makedirs(path, exist_ok=True)

                        if avg_seg_volume is None:
                            avg_seg_volume = get_avg_seg_volume(output_agg, data_shape)
                        else:
                            avg_seg_volume = torch.cat([avg_seg_volume, get_avg_seg_volume(output_agg, data_shape)], dim=0)

                        axis = 0
                        avg_seg_volume = avg_seg_volume.mean(dim=0)
                        n_samples += 1

                        # Evaluate 3D volume for the patient
                        evaluate_3D(avg_seg_volume, target_agg, metric_fns, path, n_samples, patient_metrics, total_metrics, device)

                        # Store patient metrics
                        all_patient_metrics.append(patient_metrics.clone())

                        # Logging patient metrics
                        print('---------------------------------')
                        print(f'Patient {int(n_samples)}:')
                        for i, met in enumerate(metric_fns):
                            print(f'      {met.__name__}: {patient_metrics[i].item() / 1}')
                        patient_metrics = torch.zeros(len(metric_fns))

                        avg_seg_volume = None

    # Log overall metrics averaged across all patients
    print('================================')
    print('Averaged over all patients:')
    for i, met in enumerate(metric_fns):
        mean_metric = total_metrics[i].item() / n_samples
        # Calculate standard deviation across patients for this metric
        patient_metrics_np = np.array([patient[i].item() for patient in all_patient_metrics])
        std_metric = np.std(patient_metrics_np)
        print(f'      {met.__name__}: {mean_metric:.4f} ± {std_metric:.4f}')


@torch.no_grad()
def evaluate_3D(avg_seg_volume, target_agg, metric_fns, path, patient, patient_metrics, total_metrics, device):
    prefix = f'patient_{int(patient)}'
    seg_volume = torch.round(avg_seg_volume).int().cpu().detach().numpy()
    rotated_seg_volume = rotate(rotate(seg_volume, 90, axes=(0, 1)), -90, axes=(1, 2))
    cropped_seg_volume = rotated_seg_volume[18:-18, :, 18:-18]
    nibabel.save(nibabel.Nifti1Image(cropped_seg_volume, np.eye(4)), os.path.join(path, f'{prefix}_seg.nii'))

    target_volume = torch.squeeze(target_agg).int().cpu().detach().numpy()
    rotated_target_volume = rotate(rotate(target_volume, 90, axes=(0, 1)), -90, axes=(1, 2))
    cropped_target_volume = rotated_target_volume[18:-18, :, 18:-18]
    nibabel.save(nibabel.Nifti1Image(cropped_target_volume, np.eye(4)), os.path.join(path, f'{prefix}_target.nii.gz'))

    #Convert to tensors
    cropped_seg_volume = torch.tensor(cropped_seg_volume).to(device)
    cropped_target_volume = torch.tensor(cropped_target_volume).to(device)

    # computing loss, metrics on test set
    for i, metric in enumerate(metric_fns):
        current_metric = metric(cropped_seg_volume, cropped_target_volume)
        patient_metrics[i] += current_metric
        total_metrics[i] += current_metric

@torch.no_grad()
def evaluate(data_loader, model, device):
    # Weights for breast_tumor = 2:1 majority being label 0
    # Since evaluation is always hard target and not SoftTarget
    #criterion = torch.nn.CrossEntropyLoss().to(device)
    criterion = dual_loss

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    count = 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target_tp1, target_tp2 = batch
        images = images.to(device, non_blocking=True)
        target_tp2 = target_tp2.to(device, non_blocking=True)
        target_tp1 = target_tp1.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            #target = target.reshape(bz*nf,nc,h,w,d)
            loss = criterion(output, target_tp1, target_tp2)

        outPRED = torch.cat((outPRED, output), 0)
        outGT = torch.cat((outGT, target_tp2), 0)

        metric_logger.update(loss=loss.item())
        count += 1

    prec, rec, dice = precision(outPRED, outGT), recall(outPRED, outGT), dice_score(outPRED, outGT)
    metric_logger.update(prec=prec)
    metric_logger.update(rec=rec)
    metric_logger.update(dice=dice)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'Random seed set as {seed}')

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
    
    set_seed(42)

    criterion = structure_loss
    model = SimpleSegmentationModel()
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("number of params:", n_parameters)
    print("model:", model)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    param_groups = model.parameters()
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr,)

    dummy_input = torch.randn(2, 5, 2, 128, 128).to(device)  # Generate random input for model
    target = torch.randn(5, 1, 128, 128).to(device)  # Generate random target for model

    train_model(dummy_input, target, criterion, optimizer, model)

def select_best_model(args, epoch, loss_scaler, max_val, model, model_without_ddp, optimizer, cur_val,
                      model_name='best_dice_model', output_dir=None):
    if cur_val > max_val:
        print(f"saving {model_name} @ epoch {epoch}")
        max_val = cur_val
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=model_name, output_dir=output_dir)  # A little hack for saving model with preferred name
    return max_val

def select_min_loss_model(args, epoch, loss_scaler, min_val, model, model_without_ddp, optimizer, cur_val,
                      model_name='', output_dir=None):
    if cur_val < min_val:
        print(f"saving {model_name} @ epoch {epoch}")
        min_val = cur_val
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=model_name, output_dir=output_dir)  # A little hack for saving model with preferred name
    return min_val 

def train_model(dummy_input, target, criterion, optimizer, model, num_epochs=10, num_runs=3):
    
    for run in range(1):
        print(f"\n--- Training Run {run + 1} ---")
        for epoch in range(num_epochs):
            model.train()           
            
            optimizer.zero_grad()
            output, _ = model(dummy_input)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    wandb.finish()
