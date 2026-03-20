from __future__ import annotations

import torch
from torch import nn


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = torch.sum(probs * target, dims)
        denominator = torch.sum(probs, dims) + torch.sum(target, dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return bce + dice_loss


def dice_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return float(dice.mean().item())


def iou_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1.0) / (union + 1.0)
    return float(iou.mean().item())
