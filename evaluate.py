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

import torch
from torch.utils.data import DataLoader

from radarseg.config import load_config
from radarseg.data.collate import detection_collate, mask2former_collate, semantic_collate
from radarseg.data.dataset import RadargramInstanceDataset, RadargramSemanticDataset
from radarseg.engine.evaluator import evaluate_instance_model, evaluate_semantic_model
from radarseg.external.yolo11 import evaluate_yolo11_seg
from radarseg.models.factory import build_model
from radarseg.utils.checkpoint import load_checkpoint
from radarseg.utils.io import save_json
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
    parser = argparse.ArgumentParser(description="Evaluate a trained radargram segmentation model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    num_threads = cfg["training"].get("num_threads")
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    device = get_device()

    processed_root = cfg["paths"]["processed_root"]
    splits_dir = cfg["paths"]["splits_dir"]
    image_size = cfg["input"].get("image_size")
    resize_mode = cfg["input"].get("resize_mode", "resize")
    pad_value = int(cfg["input"].get("pad_value", 0))
    allow_upscale = bool(cfg["input"].get("allow_upscale", True))
    grayscale = bool(cfg["input"].get("grayscale", False))
    batch_size = int(cfg["training"].get("batch_size", 1))
    num_workers = int(cfg["training"].get("num_workers", 4))
    task = cfg["model"]["task"]
    model_name = cfg["model"]["name"]

    threshold = float(cfg["postprocessing"].get("threshold", cfg["model"].get("score_threshold", 0.5)))
    min_area = int(cfg["postprocessing"].get("min_area", 20))

    if model_name == "yolo11_seg":
        metrics = evaluate_yolo11_seg(
            args.checkpoint,
            processed_root,
            splits_dir,
            args.split,
            threshold=threshold,
            min_area=min_area,
            imgsz=cfg.get("yolo", {}).get("imgsz"),
        )
        print(metrics)
        output = args.output or Path(cfg["paths"]["output_dir"]) / f"metrics_{args.split}.json"
        save_json(metrics, output)
        print(f"Saved metrics to: {output}")
        return

    if model_name == "sam2":
        raise NotImplementedError(
            "Use scripts/evaluate_sam2_prompted.py for SAM 2. SAM 2 requires prompt boxes, "
            "so it is evaluated through the prompted SAM 2 utility rather than the generic checkpoint loader."
        )

    checkpoint_cfg = make_checkpoint_loading_config(cfg)
    model = build_model(checkpoint_cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    if task == "semantic":
        ds = RadargramSemanticDataset(
            processed_root,
            splits_dir,
            args.split,
            image_size=image_size,
            grayscale=grayscale,
            resize_mode=resize_mode,
            pad_value=pad_value,
            allow_upscale=allow_upscale,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=semantic_collate)
        metrics = evaluate_semantic_model(model, loader, device, threshold=threshold, min_area=min_area, model_name=model_name)
    else:
        ds = RadargramInstanceDataset(
            processed_root,
            splits_dir,
            args.split,
            image_size=image_size,
            grayscale=grayscale,
            resize_mode=resize_mode,
            pad_value=pad_value,
            allow_upscale=allow_upscale,
        )
        collate_fn = mask2former_collate if model_name == "mask2former" else detection_collate
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        metrics = evaluate_instance_model(model, loader, device, model_name=model_name, threshold=threshold, min_area=min_area)

    print(metrics)
    output = args.output or Path(cfg["paths"]["output_dir"]) / f"metrics_{args.split}.json"
    save_json(metrics, output)
    print(f"Saved metrics to: {output}")


if __name__ == "__main__":
    main()
