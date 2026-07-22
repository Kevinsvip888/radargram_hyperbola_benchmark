#!/usr/bin/env python
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch

from radarseg.config import load_config
from radarseg.data.dataset import load_raw_image_for_prediction
from radarseg.external.yolo11 import predict_yolo11_on_image
from radarseg.models.factory import build_model
from radarseg.models.mask_rcnn import mask_rcnn_predictions_to_instances
from radarseg.models.mask2former import mask2former_outputs_to_instances
from radarseg.postprocessing.semantic_to_instances import split_semantic_prediction
from radarseg.utils.checkpoint import load_checkpoint
from radarseg.utils.masks import masks_to_boxes
from radarseg.utils.prediction_io import save_prediction_outputs
from radarseg.utils.seed import get_device



def make_checkpoint_loading_config(cfg: dict) -> dict:
    """Return a config suitable for loading an already trained checkpoint.

    Torchvision Mask R-CNN does not need COCO pretrained weights during
    evaluation or prediction. Disabling them here avoids unnecessary downloads
    when the user already supplies --checkpoint.
    """
    cfg = deepcopy(cfg)
    model_cfg = cfg.get("model", {})
    if model_cfg.get("name") == "mask_rcnn":
        model_cfg["pretrained"] = False
    return cfg

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict hyperbola masks and export pixel coordinates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True, help="Folder containing images, or one image file.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--extensions", nargs="+", default=[".png", ".jpg", ".jpeg", ".tif", ".tiff"])
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help=(
            "Optional dataset split to predict. When used, image stems under --input-root "
            "are filtered using <splits-dir>/<split>.txt. Use this with processed images, "
            "for example --input-root dataset/processed/images --split test."
        ),
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("dataset/splits"),
        help="Folder containing train.txt, val.txt, and test.txt.",
    )
    parser.add_argument(
        "--strict-split",
        action="store_true",
        help="Raise an error if any ID in the requested split is missing under --input-root.",
    )
    return parser.parse_args()


def list_images(input_root: Path, extensions: list[str]) -> list[Path]:
    if input_root.is_file():
        return [input_root]
    exts = {e.lower() for e in extensions}
    return sorted([p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def read_split_ids(splits_dir: Path, split: str) -> list[str]:
    """Read image IDs from a split file.

    Split files contain one processed image ID per line, without file extension.
    For example, a line named ``sim_0001`` matches ``sim_0001.png``.
    """
    split_file = splits_dir / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"Split file is empty: {split_file}")
    return ids


def filter_images_by_split(
    image_paths: list[Path],
    split_ids: list[str],
    *,
    strict: bool = False,
) -> list[Path]:
    """Keep only images whose filename stem is listed in the requested split.

    The processed dataset stores images as ``<image_id>.png``. The split files
    store the corresponding ``image_id`` values, so filename stems are used for
    matching.
    """
    images_by_id = {path.stem: path for path in image_paths}
    selected: list[Path] = []
    missing: list[str] = []

    for image_id in split_ids:
        image_path = images_by_id.get(image_id)
        if image_path is None:
            missing.append(image_id)
        else:
            selected.append(image_path)

    if missing:
        message = (
            f"{len(missing)} image ID(s) from the split were not found under --input-root. "
            f"Examples: {missing[:5]}"
        )
        if strict:
            raise FileNotFoundError(message)
        print(f"Warning: {message}")

    if not selected:
        raise FileNotFoundError(
            "No images matched the requested split. Make sure --input-root points to "
            "dataset/processed/images, not the raw dataset root."
        )
    return selected


def predict_semantic(model: torch.nn.Module, image_tensor: torch.Tensor, threshold: float, min_area: int) -> list[dict]:
    logits = model(image_tensor[None])
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits[:, 0])[0]
    else:
        probs = torch.softmax(logits, dim=1)[:, 1][0]
    pred = (probs.detach().cpu().numpy() >= threshold).astype(np.uint8)
    masks = split_semantic_prediction(pred, min_area=min_area)
    instances = []
    if masks:
        boxes = masks_to_boxes(np.stack(masks, axis=0))
        for mask, box in zip(masks, boxes):
            instances.append({"score": 1.0, "bbox": box.tolist(), "mask": mask, "area": int(mask.sum())})
    return instances


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    num_threads = cfg["training"].get("num_threads")
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    device = get_device()

    image_size = cfg["input"].get("image_size")
    resize_mode = cfg["input"].get("resize_mode", "resize")
    pad_value = int(cfg["input"].get("pad_value", 0))
    grayscale = bool(cfg["input"].get("grayscale", False))
    model_name = cfg["model"]["name"]
    task = cfg["model"]["task"]
    threshold = float(cfg["postprocessing"].get("threshold", cfg["model"].get("score_threshold", 0.5)))
    min_area = int(cfg["postprocessing"].get("min_area", 20))

    images = list_images(args.input_root, args.extensions)
    if not images:
        raise FileNotFoundError(f"No input images found under: {args.input_root}")

    if args.split is not None:
        split_ids = read_split_ids(args.splits_dir, args.split)
        images = filter_images_by_split(images, split_ids, strict=args.strict_split)
        print(f"Predicting {len(images)} image(s) from split '{args.split}'.")
    else:
        print(f"Predicting {len(images)} image(s) from --input-root.")

    if model_name == "yolo11_seg":
        for image_path in images:
            instances = predict_yolo11_on_image(
                args.checkpoint,
                image_path,
                threshold=threshold,
                min_area=min_area,
                imgsz=cfg.get("yolo", {}).get("imgsz"),
            )
            out_dir = args.output_root / image_path.stem
            save_prediction_outputs(image_path, instances, out_dir)
            print(f"Saved YOLO11-seg prediction for {image_path.name} -> {out_dir}")
        return

    if model_name == "sam2":
        raise NotImplementedError(
            "Use scripts/predict_sam2_prompted.py for SAM 2 because SAM 2 requires prompt boxes. "
            "The generic predict.py command is reserved for models that directly produce masks from images."
        )

    checkpoint_cfg = make_checkpoint_loading_config(cfg)
    model = build_model(checkpoint_cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    with torch.no_grad():
        for image_path in images:
            loaded = load_raw_image_for_prediction(
                image_path,
                image_size=image_size,
                grayscale=grayscale,
                resize_mode=resize_mode,
                pad_value=pad_value,
                return_meta=True,
            )
            image_tensor, transform_meta = loaded
            image_tensor = image_tensor.to(device)

            if task == "semantic":
                instances = predict_semantic(model, image_tensor, threshold=threshold, min_area=min_area)
            elif model_name == "mask_rcnn":
                prediction = model([image_tensor])[0]
                instances = mask_rcnn_predictions_to_instances(prediction, threshold=threshold, min_area=min_area)
            elif model_name == "mask2former":
                outputs = model(image_tensor[None])
                instances = mask2former_outputs_to_instances(outputs, image_size=tuple(image_tensor.shape[-2:]), threshold=threshold, min_area=min_area)
            else:
                raise ValueError(f"Unsupported prediction combination: task={task}, model={model_name}")

            out_dir = args.output_root / image_path.stem
            save_prediction_outputs(image_path, instances, out_dir, transform_meta=transform_meta)
            print(f"Saved prediction for {image_path.name} -> {out_dir}")


if __name__ == "__main__":
    main()
