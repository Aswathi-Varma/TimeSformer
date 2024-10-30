import math
import torch
import torch.nn.functional as F


def compute_loss(pred, mask, epsilon=1e-6):
    # Compute weighted loss using average pool and absolute difference
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    
    # Binary cross-entropy with logits, weighted by weit
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2,3)) / weit.sum(dim=(2,3))

    # Sigmoid activation to get predicted probabilities
    pred = torch.sigmoid(pred)
    
    # Compute intersection and union for IoU calculation
    inter = ((pred * mask) * weit).sum(dim=(2,3))
    union = ((pred + mask) * weit).sum(dim=(2,3))
    
    # Add epsilon to the denominator to avoid division by zero
    wiou = 1 - (inter + 1) / (union - inter + 1 + epsilon)

    # Return the combined loss
    return (wbce + wiou).mean()


def dual_loss(pred, mask_tp1, mask_tp2):

    if torch.isnan(pred).any() or torch.isnan(mask_tp1).any() or torch.isnan(mask_tp2).any():
        print("NaN detected in inputs")

    pred_tp1 = pred[:, 0, :, :, :]
    pred_tp2 = pred[:, 1, :, :, :]

    # Timepoint 1 loss
    loss_tp1 = compute_loss(pred_tp1, mask_tp1)
    
    # Timepoint 2 loss
    loss_tp2 = compute_loss(pred_tp2, mask_tp2)

    # Difference mask loss (TP2 - TP1)
    diff_pred = torch.abs(pred_tp2 - pred_tp1)
    diff_gt = torch.abs(mask_tp2 - mask_tp1)
    loss_diff = compute_loss(diff_pred, diff_gt)
    
    # Equal weight to all three loss terms
    total_loss = (loss_tp1 + loss_tp2 + loss_diff) / 3
    return total_loss



def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def mask_iou(pred, target, averaged=True):
    """
    param: pred of size [N x H x W]
    param: target of size [N x H x W]
    """

    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)

    inter = torch.min(pred, target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    iou = inter / union

    if averaged:
        iou = torch.mean(iou)

    return iou

def binary_entropy_loss(pred, target, num_object, eps=0.001, ref=None):

    ce = - 1.0 * target * torch.log(pred + eps) - (1 - target) * torch.log(1 - pred + eps)

    loss = torch.mean(ce)

    # TODO: training with bootstrapping

    return loss

def cross_entropy_loss(pred, mask, num_object, bootstrap=0.4, ref=None):

    # pred: [N x K x H x W]
    # mask: [N x K x H x W] one-hot encoded
    N, _, H, W = mask.shape

    pred = -1 * torch.log(pred)
    # loss = torch.sum(pred[:, :num_object+1] * mask[:, :num_object+1])
    # loss = loss / (H * W * N)

    # bootstrap
    num = int(H * W * bootstrap)
    ce = pred[:, :num_object+1] * mask[:, :num_object+1]
    if ref is not None:
        valid = torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0
        valid = valid.float().unsqueeze(2).unsqueeze(3)
        ce *= valid

    loss = torch.sum(ce, dim=1).view(N, -1)
    mloss, _ = torch.sort(loss, dim=-1, descending=True)
    loss = torch.mean(mloss[:, :num])

    return loss

def mask_iou_loss(pred, mask, num_object, ref=None):

    N, K, H, W = mask.shape
    loss = torch.zeros(1).to(pred.device)
    start = 0 if K == num_object else 1

    if ref is not None:
        valid = torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0

    for i in range(N):
        obj_loss = (1.0 - mask_iou(pred[i, start:num_object+start], mask[i, start:num_object+start], averaged=False))
        if ref is not None:
            obj_loss = obj_loss[valid[i, start:]]

        loss += torch.mean(obj_loss)

    loss = loss / N
    return loss

def smooth_l1_loss(pred, target, gamma=0.075):

    diff = torch.abs(pred-target)
    diff[diff>gamma] -= gamma / 2
    diff[diff<=gamma] *= diff[diff<=gamma] / (2 * gamma)

    return torch.mean(diff)





