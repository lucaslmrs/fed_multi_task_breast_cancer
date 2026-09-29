"""Binary Dice + BCE with explicit scientific weights."""
import math
import torch
from monai.losses import DiceLoss


class DiceBCELoss(torch.nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        weights = (dice_weight, bce_weight)
        if any(isinstance(w, bool) or not isinstance(w, (int, float)) or
               not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("DiceBCE weights must be finite, non-negative and have positive sum")
        self.dice_weight, self.bce_weight = weights
        self.dice = DiceLoss(include_background=True, sigmoid=True, smooth_dr=1,
                             smooth_nr=1, squared_pred=True)
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        if logits.ndim < 3 or logits.shape[1] != 1 or logits.shape != target.shape:
            raise ValueError("DiceBCE requires matching binary Bx1x... logits and targets")
        # Float32 reductions remain stable under BF16 autocast; gradients flow to the model.
        logits, target = logits.float(), target.float()
        return self.dice_weight * self.dice(logits, target) + self.bce_weight * self.bce(logits, target)
