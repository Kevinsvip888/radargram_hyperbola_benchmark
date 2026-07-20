from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from radarseg.models.mask_rcnn import mask_rcnn_predictions_to_instances
from radarseg.models.mask2former import mask2former_outputs_to_instances
from radarseg.utils.masks import binary_metrics, mask_iou, semantic_to_instance_masks, union_masks


@torch.no_grad()
def evaluate_semantic_model(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device,
    threshold: float = 0.5,
    min_area: int = 20,
    model_name: str = "unet",
) -> dict[str, float]:
    model.eval()
    metrics_accum: dict[str, list[float]] = {"dice": [], "iou": [], "precision": [], "recall": [], "accuracy": []}
    instance_counts: list[float] = []

    for images, masks in tqdm(dataloader, desc="Evaluating semantic", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)

        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits[:, 0])
        else:
            probs = torch.softmax(logits, dim=1)[:, 1]

        preds = (probs >= threshold).detach().cpu().numpy().astype(np.uint8)
        targets = masks.detach().cpu().numpy().astype(np.uint8)

        for pred, target in zip(preds, targets):
            pred = np.squeeze(pred)
            target = np.squeeze(target)
            m = binary_metrics(pred, target)
            for key, value in m.items():
                metrics_accum[key].append(value)
            instance_counts.append(float(len(semantic_to_instance_masks(pred, min_area=min_area))))

    return {key: float(np.mean(values)) if values else 0.0 for key, values in metrics_accum.items()} | {
        "predicted_instances_per_image": float(np.mean(instance_counts)) if instance_counts else 0.0
    }


def _match_instances(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], iou_threshold: float = 0.5) -> dict[str, float]:
    if not pred_masks and not gt_masks:
        return {"precision": 1.0, "recall": 1.0, "mean_matched_iou": 1.0, "ap50": 1.0}
    if not pred_masks:
        return {"precision": 0.0, "recall": 0.0, "mean_matched_iou": 0.0, "ap50": 0.0}
    if not gt_masks:
        return {"precision": 0.0, "recall": 0.0, "mean_matched_iou": 0.0, "ap50": 0.0}

    ious = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for i, pred in enumerate(pred_masks):
        for j, gt in enumerate(gt_masks):
            ious[i, j] = mask_iou(pred, gt)

    matched_gt: set[int] = set()
    matched_ious: list[float] = []
    tp = 0
    for pred_idx in range(len(pred_masks)):
        gt_idx = int(np.argmax(ious[pred_idx]))
        best_iou = float(ious[pred_idx, gt_idx])
        if best_iou >= iou_threshold and gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(gt_idx)
            matched_ious.append(best_iou)

    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "ap50": float(precision if recall > 0 else 0.0),
    }


@torch.no_grad()
def evaluate_instance_model(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device,
    model_name: str,
    threshold: float = 0.5,
    min_area: int = 20,
) -> dict[str, float]:
    model.eval()
    semantic_accum: dict[str, list[float]] = {"dice": [], "iou": [], "precision": [], "recall": [], "accuracy": []}
    instance_accum: dict[str, list[float]] = {"precision": [], "recall": [], "mean_matched_iou": [], "ap50": []}
    pred_counts: list[float] = []
    gt_counts: list[float] = []

    for images, targets in tqdm(dataloader, desc="Evaluating instance", leave=False):
        images = [image.to(device) for image in images] if isinstance(images, list) else images.to(device)

        if model_name == "mask_rcnn":
            predictions = model(images)
            batch_instances = [
                mask_rcnn_predictions_to_instances(pred, threshold=threshold, min_area=min_area)
                for pred in predictions
            ]
        elif model_name == "mask2former":
            outputs = model(images)
            image_size = tuple(images.shape[-2:])
            batch_instances = [
                mask2former_outputs_to_instances(
                    outputs,
                    image_size=image_size,
                    threshold=threshold,
                    min_area=min_area,
                    batch_index=i,
                )
                for i in range(images.shape[0])
            ]
        else:
            raise ValueError(f"Unsupported instance model for evaluation: {model_name}")

        for idx, target in enumerate(targets):
            pred_masks = [item["mask"] for item in batch_instances[idx]] if idx < len(batch_instances) else []
            gt_masks = [m.detach().cpu().numpy().astype(np.uint8) for m in target["masks"]]

            pred_union = union_masks(pred_masks, shape=gt_masks[0].shape if gt_masks else (images[0].shape[-2], images[0].shape[-1]))
            gt_union = union_masks(gt_masks, shape=pred_union.shape)

            sm = binary_metrics(pred_union, gt_union)
            for key, value in sm.items():
                semantic_accum[key].append(value)

            im = _match_instances(pred_masks, gt_masks, iou_threshold=0.5)
            for key, value in im.items():
                instance_accum[key].append(value)

            pred_counts.append(float(len(pred_masks)))
            gt_counts.append(float(len(gt_masks)))

    out = {f"semantic_{key}": float(np.mean(values)) if values else 0.0 for key, values in semantic_accum.items()}
    out.update({f"instance_{key}": float(np.mean(values)) if values else 0.0 for key, values in instance_accum.items()})
    out["predicted_instances_per_image"] = float(np.mean(pred_counts)) if pred_counts else 0.0
    out["gt_instances_per_image"] = float(np.mean(gt_counts)) if gt_counts else 0.0
    return out
