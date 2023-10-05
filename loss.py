import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

def dice_loss(pred, target):
    smooth = 1e-5
    
    # flatten predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice  

def dice_loss_fn(pred,target,n_classes=3):
  smooth = 0.001
  pred = F.softmax(pred,dim=1).float().flatten(0,1) # (96,128,128)-> 3 * 32
  target = F.one_hot(target.to(torch.int64), n_classes).squeeze(1).permute(0, 3, 1, 2).float().flatten(0,1) # (96,128,128) -> 3 * 32
  assert pred.size() == pred.size(), "sizes do not match"

  intersection = 2 * (pred * target).sum(dim=(-1, -2)) # 96
  union = pred.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) #96
  #sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

  dice = (intersection + smooth) / ( union + smooth)

  return 1 - dice.mean()

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1e-5
        pred = pred.float()
        target = target.float()
        pred.requires_grad = True
        target.requires_grad = True

        # Define constants for classes
        FOREGROUND = 1
        BACKGROUND = 2
        NOT_CLASSIFIED = 3

        # Convert trimaps to binary masks for foreground
        mask_pred_foreground = (pred == FOREGROUND).float()  # Binary mask: 1 for foreground, 0 otherwise
        mask_target_foreground = (target == FOREGROUND).float()  # Binary mask: 1 for foreground, 0 otherwise

        mask_pred_foreground.requires_grad = True
        mask_target_foreground.requires_grad = True

        # Calculate intersection and union for the foreground
        intersection = torch.sum(mask_pred_foreground * mask_target_foreground, dim=(1, 2, 3))
        union = torch.sum(mask_pred_foreground + mask_target_foreground, dim=(1, 2, 3))  # Union = A + B - Intersection

        
        # Flatten predictions and targets
        # pred = pred.view(-1)
        # target = target.view(-1)
        
        # intersection = (pred * target).sum()
        # union = pred.sum() + target.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        
        return 1 - dice