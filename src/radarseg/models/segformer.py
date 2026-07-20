from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SegFormerSemantic(nn.Module):
    """Hugging Face SegFormer wrapper for binary semantic segmentation."""

    def __init__(self, pretrained_model_name: str = "nvidia/mit-b0", num_classes: int = 2) -> None:
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError("Install transformers to use SegFormer: pip install transformers") from exc

        id2label = {0: "background", 1: "hyperbola"}
        label2id = {v: k for k, v in id2label.items()}

        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
        except Exception:
            config = SegformerConfig.from_pretrained(
                pretrained_model_name,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
            )
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
