from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Mask2FormerInstance(nn.Module):
    """Hugging Face Mask2Former wrapper for instance segmentation.

    The forward method accepts normalized image tensors and optional instance targets.
    Targets should use the same format as `RadargramInstanceDataset`.
    """

    def __init__(self, pretrained_model_name: str = "facebook/mask2former-swin-tiny-coco-instance", num_classes: int = 2) -> None:
        super().__init__()
        try:
            from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
        except ImportError as exc:
            raise ImportError("Install transformers to use Mask2Former: pip install transformers") from exc

        id2label = {0: "background", 1: "hyperbola"}
        label2id = {v: k for k, v in id2label.items()}

        try:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                pretrained_model_name,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
        except Exception:
            config = Mask2FormerConfig.from_pretrained(
                pretrained_model_name,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
            )
            self.model = Mask2FormerForUniversalSegmentation(config)

    @staticmethod
    def _targets_to_hf(targets: list[dict[str, torch.Tensor | str]], device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        mask_labels: list[torch.Tensor] = []
        class_labels: list[torch.Tensor] = []
        for target in targets:
            masks = target["masks"].to(device=device, dtype=torch.float32)  # type: ignore[index]
            labels = target["labels"].to(device=device, dtype=torch.long)  # type: ignore[index]
            mask_labels.append(masks)
            class_labels.append(labels)
        return mask_labels, class_labels

    def forward(self, x: torch.Tensor, targets: list[dict[str, torch.Tensor | str]] | None = None):
        if targets is None:
            return self.model(pixel_values=x)
        mask_labels, class_labels = self._targets_to_hf(targets, x.device)
        return self.model(pixel_values=x, mask_labels=mask_labels, class_labels=class_labels)


@torch.no_grad()
def mask2former_outputs_to_instances(
    outputs,
    image_size: tuple[int, int],
    threshold: float = 0.5,
    min_area: int = 20,
    batch_index: int = 0,
) -> list[dict]:
    """Convert raw Mask2Former outputs for one batch item to instance masks.

    This uses query classification logits and predicted masks directly to avoid
    depending on processor post-processing details.
    """
    class_queries_logits = outputs.class_queries_logits[batch_index].detach().cpu()
    masks_queries_logits = outputs.masks_queries_logits[batch_index].detach().cpu()

    class_probs = class_queries_logits.softmax(dim=-1)
    # Class 1 is hyperbola. Last class can be the no-object class for some configs.
    scores = class_probs[:, 1] if class_probs.shape[-1] > 1 else class_probs[:, 0]

    masks = torch.sigmoid(masks_queries_logits[:, None, :, :])
    masks = F.interpolate(masks, size=image_size, mode="bilinear", align_corners=False)[:, 0]

    instances: list[dict] = []
    for idx in torch.argsort(scores, descending=True).tolist():
        score = float(scores[idx].item())
        if score < threshold:
            continue
        mask = (masks[idx].numpy() >= threshold).astype(np.uint8)
        area = int(mask.sum())
        if area < min_area:
            continue
        ys, xs = np.where(mask > 0)
        bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())] if xs.size else [0.0, 0.0, 0.0, 0.0]
        instances.append({"score": score, "bbox": bbox, "mask": mask, "area": area})
    return instances
