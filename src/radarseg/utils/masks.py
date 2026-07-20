from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


UINT8_MAX = 255


def read_binary_mask(path: str | Path) -> np.ndarray:
    """Read a mask image and return a binary uint8 array with values {0, 1}."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask file not found: {path}")
    mask = np.array(Image.open(path).convert("L"))
    return (mask > 0).astype(np.uint8)


def save_binary_mask(mask: np.ndarray, path: str | Path) -> None:
    """Save a binary mask as 0/255 PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = ensure_binary(mask)
    Image.fromarray((mask * UINT8_MAX).astype(np.uint8)).save(path)


def ensure_binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}")
    return (mask > 0).astype(np.uint8)


def union_masks(masks: list[np.ndarray], shape: tuple[int, int] | None = None) -> np.ndarray:
    """Return the binary union of several masks."""
    if not masks:
        if shape is None:
            raise ValueError("Cannot create mask union from an empty list without a shape")
        return np.zeros(shape, dtype=np.uint8)

    out = np.zeros(masks[0].shape if shape is None else shape, dtype=np.uint8)
    for mask in masks:
        if mask.shape != out.shape:
            raise ValueError(f"Mask shape mismatch: expected {out.shape}, got {mask.shape}")
        out[mask > 0] = 1
    return out


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """Compute [x_min, y_min, x_max, y_max] boxes from binary instance masks.

    Parameters
    ----------
    masks:
        Array with shape (N, H, W) and values {0, 1}.
    """
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape (N, H, W), got {masks.shape}")

    boxes: list[list[float]] = []
    for mask in masks:
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            boxes.append([0.0, 0.0, 0.0, 0.0])
            continue
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    return np.asarray(boxes, dtype=np.float32)


def remove_small_components(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """Remove connected components smaller than `min_area`."""
    mask = ensure_binary(mask)
    if min_area <= 1:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_idx] = 1
    return cleaned


def semantic_to_instance_masks(mask: np.ndarray, min_area: int = 20) -> list[np.ndarray]:
    """Split a binary semantic mask into connected-component instance masks."""
    mask = remove_small_components(mask, min_area=min_area)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    instances: list[np.ndarray] = []
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        instances.append((labels == label_idx).astype(np.uint8))
    return instances


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    a = ensure_binary(mask_a).astype(bool)
    b = ensure_binary(mask_b).astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def binary_metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> dict[str, float]:
    """Compute standard binary segmentation metrics."""
    pred_bool = ensure_binary(pred).astype(bool)
    target_bool = ensure_binary(target).astype(bool)

    tp = np.logical_and(pred_bool, target_bool).sum()
    fp = np.logical_and(pred_bool, ~target_bool).sum()
    fn = np.logical_and(~pred_bool, target_bool).sum()
    tn = np.logical_and(~pred_bool, ~target_bool).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }


def mask_to_polygons(mask: np.ndarray, min_area: int = 1) -> list[list[float]]:
    """Convert a binary mask to COCO-style polygon lists using OpenCV contours."""
    mask = ensure_binary(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        contour = contour.reshape(-1, 2)
        if contour.shape[0] < 3:
            continue
        polygons.append(contour.astype(float).reshape(-1).tolist())
    return polygons
