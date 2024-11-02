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

# def dice_score(output_logits, target_mask, threshold=0.5, eps=1e-6):
#     """
#     Calculate the Dice score for each time slice and average over the time dimension.

#     Args:
#         output_logits (torch.Tensor): The model output logits of shape (n, t, 1, h, w).
#         target_mask (torch.Tensor): The ground truth binary mask of shape (n, t, 1, h, w).
#         threshold (float): Threshold for binarizing probabilities.
#         eps (float): Small constant to avoid division by zero.

#     Returns:
#         torch.Tensor: The mean Dice score across the time dimension.
#     """
#     # Apply sigmoid to convert logits to probabilities
#     probs = torch.sigmoid(output_logits)
    
#     # Binarize probabilities based on the threshold
#     binary_preds = (probs > threshold).float()
    
#     # Flatten the tensors for Dice calculation
#     binary_preds = binary_preds.view(binary_preds.size(0), binary_preds.size(1), -1)  # (n, t, h*w)
#     target_mask = target_mask.view(target_mask.size(0), target_mask.size(1), -1)      # (n, t, h*w)

#     # Calculate Dice score for each time slice
#     intersection = (binary_preds * target_mask).sum(-1)
#     union = binary_preds.sum(-1) + target_mask.sum(-1)
#     dice = (2.0 * intersection + eps) / (union + eps)

#     # Average Dice score over the time dimension
#     return dice.mean()

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
