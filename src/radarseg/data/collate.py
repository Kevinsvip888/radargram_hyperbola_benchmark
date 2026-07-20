from __future__ import annotations

import torch


def detection_collate(batch):
    """Collate function for torchvision-style detection models."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def semantic_collate(batch):
    """Collate function for semantic segmentation models."""
    images, masks = zip(*batch)
    return torch.stack(list(images), dim=0), torch.stack(list(masks), dim=0)


def mask2former_collate(batch):
    """Collate function for Mask2Former instance training."""
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)
