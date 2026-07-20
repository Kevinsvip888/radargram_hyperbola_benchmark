from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from PIL import Image

from radarseg.utils.masks import masks_to_boxes


def build_sam2_predictor(cfg: Mapping[str, Any]) -> Any:
    """Build a SAM 2 image predictor from the official SAM 2 package.

    Supported config styles:
    1. Hugging Face checkpoint:
       model.pretrained_model_name: facebook/sam2-hiera-large
    2. Local official SAM 2 checkpoint/config:
       model.checkpoint: checkpoints/sam2.1_hiera_large.pt
       model.model_cfg: configs/sam2.1/sam2.1_hiera_l.yaml
    """
    model_cfg = cfg["model"]
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "SAM 2 requires the official sam2 package. Install it from "
            "https://github.com/facebookresearch/sam2 before using this integration."
        ) from exc

    pretrained_name = model_cfg.get("pretrained_model_name")
    if pretrained_name:
        return SAM2ImagePredictor.from_pretrained(str(pretrained_name))

    checkpoint = model_cfg.get("checkpoint")
    sam_model_cfg = model_cfg.get("model_cfg")
    if not checkpoint or not sam_model_cfg:
        raise ValueError(
            "SAM 2 config needs either model.pretrained_model_name or both "
            "model.checkpoint and model.model_cfg."
        )

    try:
        from sam2.build_sam import build_sam2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Could not import sam2.build_sam. Check your SAM 2 installation.") from exc

    sam_model = build_sam2(str(sam_model_cfg), str(checkpoint))
    return SAM2ImagePredictor(sam_model)


def predict_sam2_with_boxes(
    predictor: Any,
    image_path: str | Path,
    boxes: np.ndarray,
    *,
    min_area: int = 20,
    box_expansion: float = 0.0,
) -> list[dict[str, Any]]:
    """Predict one SAM 2 mask per prompt box.

    SAM 2 is promptable. For fair radargram benchmarking, this function is best
    used either with boxes from a separately trained proposal model or with
    ground-truth boxes only to measure mask-refinement quality.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    height, width = image_np.shape[:2]
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes = _expand_boxes(boxes, width=width, height=height, expansion=box_expansion)

    predictor.set_image(image_np)
    instances: list[dict[str, Any]] = []
    for box in boxes:
        masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=False)
        mask = _first_mask(masks, output_shape=(height, width))
        area = int(mask.sum())
        if area < min_area:
            continue
        score = float(np.asarray(scores).reshape(-1)[0]) if scores is not None and np.asarray(scores).size else 1.0
        instances.append({
            "score": score,
            "bbox": [float(v) for v in box.tolist()],
            "mask": mask,
            "area": area,
        })
    return instances


def boxes_from_instance_masks(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 4), dtype=np.float32)
    return masks_to_boxes(np.stack(masks, axis=0).astype(np.uint8))


def _first_mask(masks: Any, output_shape: tuple[int, int]) -> np.ndarray:
    arr = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape != output_shape:
        arr = cv2.resize(arr.astype(np.float32), dsize=(output_shape[1], output_shape[0]), interpolation=cv2.INTER_LINEAR)
    return (arr > 0).astype(np.uint8)


def _expand_boxes(boxes: np.ndarray, *, width: int, height: int, expansion: float) -> np.ndarray:
    if expansion <= 0:
        return boxes.copy()
    out = boxes.copy().astype(np.float32)
    bw = out[:, 2] - out[:, 0]
    bh = out[:, 3] - out[:, 1]
    out[:, 0] -= bw * expansion
    out[:, 2] += bw * expansion
    out[:, 1] -= bh * expansion
    out[:, 3] += bh * expansion
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, max(width - 1, 0))
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, max(height - 1, 0))
    return out
