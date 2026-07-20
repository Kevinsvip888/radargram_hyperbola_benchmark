from __future__ import annotations

import torch
from torch import nn


def build_mask_rcnn(
    num_classes: int = 2,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    score_threshold: float = 0.5,
    nms_threshold: float = 0.5,
) -> nn.Module:
    """Build a torchvision Mask R-CNN model.

    `num_classes` includes background. For this project: background + hyperbola = 2.
    """
    try:
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    except ImportError as exc:
        raise ImportError("Install torchvision to use Mask R-CNN") from exc

    if pretrained:
        try:
            weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights=weights,
                trainable_backbone_layers=trainable_backbone_layers,
            )
        except Exception:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained=True,
                trainable_backbone_layers=trainable_backbone_layers,
            )
    else:
        try:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                trainable_backbone_layers=trainable_backbone_layers,
            )
        except TypeError:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained=False,
                pretrained_backbone=False,
                trainable_backbone_layers=trainable_backbone_layers,
            )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    model.roi_heads.score_thresh = float(score_threshold)
    model.roi_heads.nms_thresh = float(nms_threshold)
    return model


@torch.no_grad()
def mask_rcnn_predictions_to_instances(
    prediction: dict[str, torch.Tensor],
    threshold: float = 0.5,
    min_area: int = 20,
) -> list[dict]:
    """Convert a torchvision prediction dictionary to binary instance masks."""
    import numpy as np

    scores = prediction.get("scores", torch.empty(0)).detach().cpu()
    boxes = prediction.get("boxes", torch.empty((0, 4))).detach().cpu()
    masks = prediction.get("masks", torch.empty((0, 1, 0, 0))).detach().cpu()

    instances: list[dict] = []
    for idx in range(scores.shape[0]):
        score = float(scores[idx].item())
        if score < threshold:
            continue
        mask = (masks[idx, 0].numpy() >= threshold).astype(np.uint8)
        area = int(mask.sum())
        if area < min_area:
            continue
        instances.append({
            "score": score,
            "bbox": boxes[idx].numpy().tolist(),
            "mask": mask,
            "area": area,
        })
    return instances
