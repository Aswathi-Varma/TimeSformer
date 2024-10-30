import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.CrossEntropyLoss()

    def dice_loss(self, outputs, targets):
        smooth = 1e-7
        # Flatten the tensor along batch and class dimensions
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (outputs * targets).sum()
        dice_coeff = (2. * intersection + smooth) / (outputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice_coeff
        
        return dice_loss

    def forward(self, outputs, targets):
        # Cross Entropy Loss expects the targets to be in the form of class indices, so we get the argmax
        ce_loss = self.ce_loss(outputs, targets)

        # Convert outputs to predicted class labels
        outputs = torch.argmax(outputs, dim=1)
        targets = torch.argmax(targets, dim=1)
        
        # Calculate Dice Loss
        dice_loss = self.dice_loss(outputs, targets)
        
        # Combine the losses with specified weights
        combined_loss = self.weight_ce * ce_loss + self.weight_dice * dice_loss
        
        return combined_loss
