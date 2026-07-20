#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from PIL import Image

from radarseg.config import load_config
from radarseg.data.dataset import load_raw_image_for_prediction
from radarseg.models.factory import build_model
from radarseg.models.mask_rcnn import mask_rcnn_predictions_to_instances
from radarseg.models.mask2former import mask2former_outputs_to_instances
from radarseg.postprocessing.semantic_to_instances import split_semantic_prediction
from radarseg.utils.checkpoint import load_checkpoint
from radarseg.utils.coordinates import save_pixel_coordinates_csv
from radarseg.utils.io import ensure_dir, save_json
from radarseg.utils.masks import masks_to_boxes, save_binary_mask, union_masks
from radarseg.utils.seed import get_device
from radarseg.utils.visualization import overlay_masks_on_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict hyperbola masks and export pixel coordinates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True, help="Folder containing images, or one image file.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--extensions", nargs="+", default=[".png", ".jpg", ".jpeg", ".tif", ".tiff"])
    return parser.parse_args()


def list_images(input_root: Path, extensions: list[str]) -> list[Path]:
    if input_root.is_file():
        return [input_root]
    exts = {e.lower() for e in extensions}
    return sorted([p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


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


def save_prediction_outputs(image_path: Path, instances: list[dict], output_dir: Path) -> None:
    ensure_dir(output_dir)
    objects_dir = ensure_dir(output_dir / "objects")
    coords_dir = ensure_dir(output_dir / "coordinates")

    masks = [item["mask"] for item in instances]
    if masks:
        mask_all = union_masks(masks, shape=masks[0].shape)
    else:
        image = Image.open(image_path)
        mask_all = np.zeros((image.height, image.width), dtype=np.uint8)

    save_binary_mask(mask_all, output_dir / "mask_all_pred.png")
    overlay = overlay_masks_on_image(Image.open(image_path).convert("RGB").resize((mask_all.shape[1], mask_all.shape[0])), masks)
    overlay.save(output_dir / "overlay.png")

    payload = {"image_id": image_path.stem, "instances": []}
    for idx, item in enumerate(instances, start=1):
        mask_name = f"object_{idx:03d}_pred.png"
        coord_name = f"object_{idx:03d}_pixels.csv"
        mask_path = objects_dir / mask_name
        coord_path = coords_dir / coord_name
        save_binary_mask(item["mask"], mask_path)
        num_pixels = save_pixel_coordinates_csv(item["mask"], coord_path)
        payload["instances"].append({
            "instance_id": idx,
            "score": float(item.get("score", 1.0)),
            "bbox": [float(v) for v in item.get("bbox", [0, 0, 0, 0])],
            "num_pixels": num_pixels,
            "mask_path": str(Path("objects") / mask_name),
            "pixel_coordinates_path": str(Path("coordinates") / coord_name),
        })
    save_json(payload, output_dir / "prediction.json")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    num_threads = cfg["training"].get("num_threads")
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    device = get_device()
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    image_size = cfg["input"].get("image_size")
    grayscale = bool(cfg["input"].get("grayscale", False))
    model_name = cfg["model"]["name"]
    task = cfg["model"]["task"]
    threshold = float(cfg["postprocessing"].get("threshold", cfg["model"].get("score_threshold", 0.5)))
    min_area = int(cfg["postprocessing"].get("min_area", 20))

    images = list_images(args.input_root, args.extensions)
    if not images:
        raise FileNotFoundError(f"No input images found under: {args.input_root}")

    with torch.no_grad():
        for image_path in images:
            image_tensor = load_raw_image_for_prediction(image_path, image_size=image_size, grayscale=grayscale).to(device)

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
            save_prediction_outputs(image_path, instances, out_dir)
            print(f"Saved prediction for {image_path.name} -> {out_dir}")


if __name__ == "__main__":
    main()
