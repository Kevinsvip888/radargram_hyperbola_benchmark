from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from .unet import UNet
from .unetpp import UNetPlusPlus


def build_model(cfg: Mapping[str, Any]) -> nn.Module:
    model_cfg = cfg["model"]
    name = model_cfg["name"]

    if name == "unet":
        return UNet(
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("num_classes", 1)),
            base_channels=int(model_cfg.get("base_channels", 32)),
        )

    if name == "unetpp":
        return UNetPlusPlus(
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("num_classes", 1)),
            base_channels=int(model_cfg.get("base_channels", 32)),
            deep_supervision=bool(model_cfg.get("deep_supervision", False)),
        )

    if name == "dinov3":
        from .dinov3_segmenter import DINOv3SemanticSegmenter

        return DINOv3SemanticSegmenter(
            pretrained_model_name=str(model_cfg.get("pretrained_model_name", "facebook/dinov3-vits16-pretrain-lvd1689m")),
            out_channels=int(model_cfg.get("num_classes", 1)),
            decoder_channels=int(model_cfg.get("decoder_channels", 256)),
            freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )

    if name == "segformer":
        from .segformer import SegFormerSemantic

        return SegFormerSemantic(
            pretrained_model_name=str(model_cfg.get("pretrained_model_name", "nvidia/mit-b0")),
            num_classes=int(model_cfg.get("num_classes", 2)),
        )

    if name == "mask_rcnn":
        from .mask_rcnn import build_mask_rcnn

        return build_mask_rcnn(
            num_classes=int(model_cfg.get("num_classes", 2)),
            pretrained=bool(model_cfg.get("pretrained", True)),
            trainable_backbone_layers=int(model_cfg.get("trainable_backbone_layers", 3)),
            score_threshold=float(model_cfg.get("score_threshold", 0.5)),
            nms_threshold=float(model_cfg.get("nms_threshold", 0.5)),
        )

    if name == "mask2former":
        from .mask2former import Mask2FormerInstance

        return Mask2FormerInstance(
            pretrained_model_name=str(model_cfg.get("pretrained_model_name", "facebook/mask2former-swin-tiny-coco-instance")),
            num_classes=int(model_cfg.get("num_classes", 2)),
        )

    raise ValueError(f"Unsupported model: {name}")
