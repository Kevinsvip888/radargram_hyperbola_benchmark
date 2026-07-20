#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import torch

from radarseg.config import load_config
from radarseg.external.sam2_prompted import boxes_from_instance_masks, build_sam2_predictor, predict_sam2_with_boxes
from radarseg.utils.io import read_split_file
from radarseg.utils.masks import read_binary_mask
from radarseg.utils.prediction_io import save_prediction_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM 2 prompted prediction on a processed dataset split.")
    parser.add_argument("--config", type=Path, default=Path("configs/sam2.yaml"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--prompt-source",
        choices=["gt"],
        default="gt",
        help="Currently 'gt' uses ground-truth boxes. Use this as a mask-refinement baseline.",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    processed_root = Path(cfg["paths"]["processed_root"])
    splits_dir = Path(cfg["paths"]["splits_dir"])
    output_root = args.output_root or Path(cfg["paths"]["output_dir"]) / f"prompted_predictions_{args.split}"
    min_area = int(cfg["postprocessing"].get("min_area", 20))
    box_expansion = float(cfg.get("sam2", {}).get("box_expansion", 0.0))

    predictor = build_sam2_predictor(cfg)
    manifest = pd.read_csv(processed_root / "annotations" / "manifest.csv")
    ids = set(read_split_file(splits_dir / f"{args.split}.txt"))
    manifest = manifest[manifest["image_id"].astype(str).isin(ids)].copy()

    for _, row in manifest.iterrows():
        image_id = str(row["image_id"])
        image_path = processed_root / str(row["image_path"])
        gt_masks = [read_binary_mask(path) for path in sorted((processed_root / str(row["instance_mask_dir"])).glob("*.png"))]
        boxes = boxes_from_instance_masks(gt_masks)
        instances = predict_sam2_with_boxes(
            predictor,
            image_path,
            boxes,
            min_area=min_area,
            box_expansion=box_expansion,
        )
        out_dir = output_root / image_id
        save_prediction_outputs(image_path, instances, out_dir)
        print(f"Saved SAM 2 prompted prediction for {image_id} -> {out_dir}")


if __name__ == "__main__":
    main()
