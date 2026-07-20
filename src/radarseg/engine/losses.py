from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, targets.float())
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class CrossEntropyDiceLoss(nn.Module):
    """Cross-entropy plus foreground Dice for two-class semantic segmentation."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_long = targets.long()
        ce = F.cross_entropy(logits, target_long)
        foreground_logits = logits[:, 1:2]
        dice = self.dice(foreground_logits, targets.float())
        return self.ce_weight * ce + self.dice_weight * dice


def build_semantic_loss(loss_cfg: dict, model_name: str) -> nn.Module:
    name = str(loss_cfg.get("name", "bce_dice"))
    if name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
            dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        )
    if name == "cross_entropy_dice":
        return CrossEntropyDiceLoss(
            ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
            dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        )
    raise ValueError(f"Unsupported semantic loss '{name}' for model '{model_name}'")
