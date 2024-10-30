import numpy as np
import torch
from sklearn.metrics import f1_score
from utils import metric_utils

def precision(output, target):
    with torch.no_grad():
        epsilon, tp, _, fp, _ = metric_utils.eps_tp_tn_fp_fn(output, target)
        return tp / (tp + fp + epsilon)


def recall(output, target):
    with torch.no_grad():
        epsilon, tp, _, _, fn = metric_utils.eps_tp_tn_fp_fn(output, target)
        return tp / (tp + fn + epsilon)

def dice_loss(output, target):
    smooth = 1e-7
    with torch.no_grad():
        # Flatten the tensors and move them to CPU
        target = metric_utils.flatten(target).cpu().detach().float()
        output = metric_utils.flatten(output).cpu().detach().float()

        if len(output.shape) == 2:
            target = np.argmax(target, axis=0)
            output = np.argmax(output, axis=0)
        # output = (output > 0.5).float()        

        intersection = (output * target).sum(-1)
        union = output.sum(-1) + target.sum(-1)
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
            
        return dice_loss


def dice_score(output, target):
    smooth = 1e-7
    with torch.no_grad():
        # Flatten the tensors and move them to CPU
        if len(output.shape) == 3: # 3D test images (H, W, D)
            target = target.flatten().cpu().detach().float()
            output = output.flatten().cpu().detach().float()
        else: # 2D slices during training (N, C, H, W)
            target = metric_utils.flatten(target).cpu().detach().float()
            output = metric_utils.flatten(output).cpu().detach().float()

        output = (output > 0.5).float()

        intersection = (output * target).sum(-1)
        union = output.sum(-1) + target.sum(-1)
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_score = dice.mean()

        return dice_score 

def asymmetric_loss(output, target):
    with torch.no_grad():
        return metric_utils.asymmetric_loss(2, output, target)
