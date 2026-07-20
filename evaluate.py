#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from radarseg.models.factory import build_model
from radarseg.utils.checkpoint import load_checkpoint
from radarseg.utils.io import save_json
from radarseg.utils.seed import get_device


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
    grayscale = bool(cfg["input"].get("grayscale", False))
    batch_size = int(cfg["training"].get("batch_size", 1))
    num_workers = int(cfg["training"].get("num_workers", 4))
    task = cfg["model"]["task"]
    model_name = cfg["model"]["name"]

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    threshold = float(cfg["postprocessing"].get("threshold", cfg["model"].get("score_threshold", 0.5)))
    min_area = int(cfg["postprocessing"].get("min_area", 20))

    if task == "semantic":
        ds = RadargramSemanticDataset(processed_root, splits_dir, args.split, image_size=image_size, grayscale=grayscale)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=semantic_collate)
        metrics = evaluate_semantic_model(model, loader, device, threshold=threshold, min_area=min_area, model_name=model_name)
    else:
        ds = RadargramInstanceDataset(processed_root, splits_dir, args.split, image_size=image_size, grayscale=grayscale)
        collate_fn = mask2former_collate if model_name == "mask2former" else detection_collate
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        metrics = evaluate_instance_model(model, loader, device, model_name=model_name, threshold=threshold, min_area=min_area)

    print(metrics)
    output = args.output or Path(cfg["paths"]["output_dir"]) / f"metrics_{args.split}.json"
    save_json(metrics, output)
    print(f"Saved metrics to: {output}")


if __name__ == "__main__":
    main()
