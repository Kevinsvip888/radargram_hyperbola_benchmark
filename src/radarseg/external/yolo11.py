from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from radarseg.data.dataset import load_manifest
from radarseg.engine.evaluator import _match_instances
from radarseg.utils.io import ensure_dir, read_split_file, save_json
from radarseg.utils.masks import binary_metrics, mask_to_polygons, read_binary_mask, union_masks


YOLO_CLASS_ID = 0
YOLO_CLASS_NAME = "hyperbola"


def export_yolo11_seg_dataset(
    processed_root: str | Path,
    splits_dir: str | Path,
    output_root: str | Path,
    *,
    min_area: int = 1,
    overwrite: bool = False,
) -> Path:
    """Export the processed radargram dataset to Ultralytics YOLO segmentation format.

    YOLO segmentation labels are text files where each row contains the class id
    followed by normalized polygon coordinates. Because the user's primary
    annotation is an object mask, polygons are derived automatically from each
    mask contour. The largest valid contour is used when one object mask contains
    multiple disconnected parts.

    Returns
    -------
    Path
        Path to the generated ``data.yaml`` file.
    """
    processed_root = Path(processed_root)
    splits_dir = Path(splits_dir)
    output_root = Path(output_root)

    if overwrite and output_root.exists():
        shutil.rmtree(output_root)

    for split in ("train", "val", "test"):
        ensure_dir(output_root / "images" / split)
        ensure_dir(output_root / "labels" / split)

    manifest = load_manifest(processed_root)
    manifest["image_id"] = manifest["image_id"].astype(str)

    for split in ("train", "val", "test"):
        split_file = splits_dir / f"{split}.txt"
        if not split_file.is_file():
            continue
        split_ids = set(read_split_file(split_file))
        split_df = manifest[manifest["image_id"].isin(split_ids)].copy()
        _export_yolo_split(split_df, processed_root, output_root, split, min_area=min_area)

    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {YOLO_CLASS_ID: YOLO_CLASS_NAME},
    }
    data_yaml_path = output_root / "data.yaml"
    data_yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return data_yaml_path


def _export_yolo_split(
    split_df: pd.DataFrame,
    processed_root: Path,
    output_root: Path,
    split: str,
    *,
    min_area: int,
) -> None:
    for _, row in split_df.iterrows():
        image_id = str(row["image_id"])
        src_image = processed_root / str(row["image_path"])
        if not src_image.is_file():
            raise FileNotFoundError(f"Image listed in manifest is missing: {src_image}")

        suffix = src_image.suffix.lower() or ".png"
        dst_image = output_root / "images" / split / f"{image_id}{suffix}"
        shutil.copy2(src_image, dst_image)

        label_path = output_root / "labels" / split / f"{image_id}.txt"
        instance_dir = processed_root / str(row["instance_mask_dir"])
        image_width = int(row["width"])
        image_height = int(row["height"])
        lines: list[str] = []

        for mask_path in sorted(instance_dir.glob("*.png")):
            mask = read_binary_mask(mask_path)
            if int(mask.sum()) < min_area:
                continue
            polygon = _mask_to_largest_yolo_polygon(mask, image_width=image_width, image_height=image_height, min_area=min_area)
            if polygon:
                lines.append(" ".join([str(YOLO_CLASS_ID), *[f"{value:.6f}" for value in polygon]]))

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _mask_to_largest_yolo_polygon(
    mask: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    min_area: int,
) -> list[float]:
    polygons = mask_to_polygons(mask, min_area=min_area)
    if not polygons:
        return []

    def polygon_area(poly: list[float]) -> float:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        return float(abs(cv2.contourArea(pts)))

    polygon = max(polygons, key=polygon_area)
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        return []
    points[:, 0] = np.clip(points[:, 0] / max(image_width - 1, 1), 0.0, 1.0)
    points[:, 1] = np.clip(points[:, 1] / max(image_height - 1, 1), 0.0, 1.0)
    return points.reshape(-1).tolist()


def train_yolo11_seg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Train YOLO11-seg using Ultralytics while preserving the project config style."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("YOLO11-seg requires ultralytics. Install it with: pip install ultralytics") from exc

    paths = cfg["paths"]
    training = cfg["training"]
    model_cfg = cfg["model"]
    yolo_cfg = cfg.get("yolo", {})

    yolo_dataset_dir = Path(paths.get("yolo_dataset_dir", Path(paths["processed_root"]).parent / "yolo11_seg"))
    data_yaml = export_yolo11_seg_dataset(
        paths["processed_root"],
        paths["splits_dir"],
        yolo_dataset_dir,
        min_area=int(cfg.get("postprocessing", {}).get("min_area", 1)),
        overwrite=bool(yolo_cfg.get("overwrite_export", False)),
    )

    model = YOLO(str(model_cfg.get("pretrained_model", "yolo11n-seg.pt")))
    output_dir = Path(paths["output_dir"])
    project_dir = output_dir.parent
    run_name = output_dir.name

    # Ultralytics has its own augmentation pipeline. The values below map the
    # project augmentation intent to YOLO's train arguments but remain
    # conservative for radargrams.
    aug = cfg.get("augmentation", {}) or {}
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": int(training.get("epochs", 100)),
        "batch": int(training.get("batch_size", 4)),
        "imgsz": int(yolo_cfg.get("imgsz", max(cfg["input"].get("image_size", [640, 640])))),
        "device": str(yolo_cfg.get("device", "0" if torch.cuda.is_available() else "cpu")),
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
        "patience": int(training.get("patience", 20)),
        "lr0": float(training.get("lr", 1e-3)),
        "weight_decay": float(training.get("weight_decay", 5e-4)),
        "seed": int(training.get("seed", 42)),
        "workers": int(training.get("num_workers", 4)),
        "task": "segment",
        "fliplr": float(aug.get("horizontal_flip_p", 0.0)) if aug.get("enabled", False) else 0.0,
        "flipud": 0.0,
        "translate": float(aug.get("shift_limit", 0.0)) if aug.get("enabled", False) else 0.0,
        "scale": float(aug.get("scale_limit", 0.0)) if aug.get("enabled", False) else 0.0,
        "degrees": 0.0,
        "perspective": 0.0,
        "mosaic": float(yolo_cfg.get("mosaic", 0.0)),
        "mixup": float(yolo_cfg.get("mixup", 0.0)),
        "copy_paste": float(yolo_cfg.get("copy_paste", 0.0)),
    }

    results = model.train(**train_kwargs)
    summary = {
        "data_yaml": str(data_yaml),
        "output_dir": str(output_dir),
        "weights": str(output_dir / "weights" / "best.pt"),
        "results": str(results),
    }
    save_json(summary, output_dir / "training_summary.json")
    return summary


def yolo11_result_to_instances(result: Any, *, threshold: float, min_area: int, output_shape: tuple[int, int]) -> list[dict[str, Any]]:
    """Convert one Ultralytics result object to the project's instance format."""
    instances: list[dict[str, Any]] = []
    if getattr(result, "masks", None) is None or result.masks is None:
        return instances

    masks = result.masks.data.detach().cpu().numpy().astype(np.float32)
    boxes = result.boxes.xyxy.detach().cpu().numpy() if getattr(result, "boxes", None) is not None else np.zeros((len(masks), 4))
    scores = result.boxes.conf.detach().cpu().numpy() if getattr(result, "boxes", None) is not None else np.ones((len(masks),), dtype=np.float32)

    height, width = output_shape
    for mask, box, score in zip(masks, boxes, scores):
        if float(score) < threshold:
            continue
        if mask.shape != (height, width):
            mask = cv2.resize(mask, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        binary = (mask >= threshold).astype(np.uint8)
        area = int(binary.sum())
        if area < min_area:
            continue
        instances.append({
            "score": float(score),
            "bbox": [float(v) for v in box.tolist()],
            "mask": binary,
            "area": area,
        })
    return instances


def predict_yolo11_on_image(
    checkpoint: str | Path,
    image_path: str | Path,
    *,
    threshold: float,
    min_area: int,
    imgsz: int | None = None,
) -> list[dict[str, Any]]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("YOLO11-seg prediction requires ultralytics. Install it with: pip install ultralytics") from exc

    image_path = Path(image_path)
    image = Image.open(image_path)
    model = YOLO(str(checkpoint))
    kwargs = {"verbose": False, "conf": threshold}
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    result = model(str(image_path), **kwargs)[0]
    return yolo11_result_to_instances(result, threshold=threshold, min_area=min_area, output_shape=(image.height, image.width))


def evaluate_yolo11_seg(
    checkpoint: str | Path,
    processed_root: str | Path,
    splits_dir: str | Path,
    split: str,
    *,
    threshold: float,
    min_area: int,
    imgsz: int | None = None,
) -> dict[str, float]:
    """Evaluate YOLO11-seg predictions with the same project metrics."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("YOLO11-seg evaluation requires ultralytics. Install it with: pip install ultralytics") from exc

    processed_root = Path(processed_root)
    manifest = load_manifest(processed_root)
    ids = set(read_split_file(Path(splits_dir) / f"{split}.txt"))
    manifest = manifest[manifest["image_id"].astype(str).isin(ids)].copy()

    model = YOLO(str(checkpoint))
    semantic_accum = {"dice": [], "iou": [], "precision": [], "recall": [], "accuracy": []}
    instance_accum = {"precision": [], "recall": [], "mean_matched_iou": [], "ap50": []}
    pred_counts: list[float] = []
    gt_counts: list[float] = []

    for _, row in manifest.iterrows():
        image_path = processed_root / str(row["image_path"])
        gt_masks = [read_binary_mask(path) for path in sorted((processed_root / str(row["instance_mask_dir"])).glob("*.png"))]
        if not gt_masks:
            continue
        kwargs = {"verbose": False, "conf": threshold}
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        result = model(str(image_path), **kwargs)[0]
        pred_instances = yolo11_result_to_instances(
            result,
            threshold=threshold,
            min_area=min_area,
            output_shape=gt_masks[0].shape,
        )
        pred_masks = [item["mask"] for item in pred_instances]
        pred_union = union_masks(pred_masks, shape=gt_masks[0].shape)
        gt_union = union_masks(gt_masks, shape=gt_masks[0].shape)

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
