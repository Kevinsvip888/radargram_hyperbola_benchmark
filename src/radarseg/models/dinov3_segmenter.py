from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class DINOv3SemanticSegmenter(nn.Module):
    """DINOv3 feature extractor with a lightweight segmentation head.

    The model uses a DINOv3/ViT-style backbone from Hugging Face Transformers
    and trains a small convolutional decoder for binary semantic segmentation.
    This is intended as a strong feature-based radargram benchmark. Instance
    masks are obtained later by the existing connected-component post-processing
    used for U-Net and SegFormer.

    Notes
    -----
    * The input height and width should be divisible by the backbone patch size.
    * The backbone can be frozen for small datasets or fine-tuned for larger ones.
    * DINO-style backbones normally expect RGB ImageNet-normalized inputs. The
      wrapper repeats grayscale inputs to RGB and applies ImageNet normalization.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        out_channels: int = 1,
        decoder_channels: int = 256,
        freeze_backbone: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "DINOv3SemanticSegmenter requires transformers. Install optional dependencies with: "
                "pip install transformers accelerate"
            ) from exc

        self.backbone = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=trust_remote_code)
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = _get_hidden_size(self.backbone.config)
        self.patch_size = _get_patch_size(self.backbone.config)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_size, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, out_channels, kernel_size=1),
        )

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, c, h, w = images.shape
        if c == 1:
            images = images.repeat(1, 3, 1, 1)
        elif c != 3:
            raise ValueError(f"DINOv3SemanticSegmenter expects 1 or 3 channels, got {c}")

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input size {(h, w)} must be divisible by DINOv3 patch_size={self.patch_size}. "
                "Change input.image_size in the YAML config."
            )

        images = (images - self.mean.to(images.device, images.dtype)) / self.std.to(images.device, images.dtype)
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            outputs = self.backbone(pixel_values=images)

        patch_tokens = _extract_patch_tokens(outputs, expected_tokens=(h // self.patch_size) * (w // self.patch_size))
        features = patch_tokens.transpose(1, 2).contiguous().view(b, -1, h // self.patch_size, w // self.patch_size)
        logits = self.decoder(features)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


def _get_hidden_size(config: Any) -> int:
    for name in ("hidden_size", "embed_dim", "hidden_dim"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer DINOv3 hidden size from model config.")


def _get_patch_size(config: Any) -> int:
    patch_size = getattr(config, "patch_size", None)
    if isinstance(patch_size, (tuple, list)):
        patch_size = patch_size[0]
    if patch_size is None:
        # Most ViT DINO checkpoints use 14 or 16. Using 16 as a safe default
        # keeps the wrapper usable for configs that omit this field.
        patch_size = 16
    return int(patch_size)


def _extract_patch_tokens(outputs: Any, expected_tokens: int) -> torch.Tensor:
    """Return patch tokens from common Hugging Face ViT/DINO output formats."""
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        tokens = outputs.last_hidden_state
    elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
        tokens = outputs["last_hidden_state"]
    elif isinstance(outputs, (tuple, list)) and outputs:
        tokens = outputs[0]
    else:
        raise ValueError("Could not find last_hidden_state in DINOv3 outputs.")

    if tokens.ndim != 3:
        raise ValueError(f"Expected DINOv3 tokens with shape B x N x C, got {tuple(tokens.shape)}")

    # ViT backbones often prepend a CLS token. Some variants may also include
    # register tokens. Keep the final expected number of patch tokens because
    # they correspond to the spatial patch grid.
    if tokens.shape[1] == expected_tokens:
        return tokens
    if tokens.shape[1] > expected_tokens:
        return tokens[:, -expected_tokens:, :]
    raise ValueError(f"DINOv3 returned {tokens.shape[1]} tokens but expected at least {expected_tokens} patch tokens.")
