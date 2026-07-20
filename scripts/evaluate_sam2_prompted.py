#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch

from radarseg.config import load_config
from radarseg.engine.evaluator import _match_instances
from radarseg.external.sam2_prompted import boxes_from_instance_masks, build_sam2_predictor, predict_sam2_with_boxes
from radarseg.utils.io import read_split_file, save_json
from radarseg.utils.masks import binary_metrics, read_binary_mask, union_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM 2 using prompt boxes on a processed dataset split.")
    parser.add_argument("--config", type=Path, default=Path("configs/sam2.yaml"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--prompt-source",
        choices=["gt"],
        default="gt",
        help="Currently 'gt' uses ground-truth boxes to measure SAM2 mask-refinement quality.",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    processed_root = Path(cfg["paths"]["processed_root"])
    splits_dir = Path(cfg["paths"]["splits_dir"])
    min_area = int(cfg["postprocessing"].get("min_area", 20))
    box_expansion = float(cfg.get("sam2", {}).get("box_expansion", 0.0))

    predictor = build_sam2_predictor(cfg)
    manifest = pd.read_csv(processed_root / "annotations" / "manifest.csv")
    ids = set(read_split_file(splits_dir / f"{args.split}.txt"))
    manifest = manifest[manifest["image_id"].astype(str).isin(ids)].copy()

    semantic_accum = {"dice": [], "iou": [], "precision": [], "recall": [], "accuracy": []}
    instance_accum = {"precision": [], "recall": [], "mean_matched_iou": [], "ap50": []}
    pred_counts: list[float] = []
    gt_counts: list[float] = []

    for _, row in manifest.iterrows():
        image_path = processed_root / str(row["image_path"])
        gt_masks = [read_binary_mask(path) for path in sorted((processed_root / str(row["instance_mask_dir"])).glob("*.png"))]
        if not gt_masks:
            continue
        boxes = boxes_from_instance_masks(gt_masks)
        pred_instances = predict_sam2_with_boxes(
            predictor,
            image_path,
            boxes,
            min_area=min_area,
            box_expansion=box_expansion,
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

    metrics = {f"semantic_{key}": float(np.mean(values)) if values else 0.0 for key, values in semantic_accum.items()}
    metrics.update({f"instance_{key}": float(np.mean(values)) if values else 0.0 for key, values in instance_accum.items()})
    metrics["predicted_instances_per_image"] = float(np.mean(pred_counts)) if pred_counts else 0.0
    metrics["gt_instances_per_image"] = float(np.mean(gt_counts)) if gt_counts else 0.0
    metrics["prompt_source"] = args.prompt_source
    metrics["note"] = "SAM 2 evaluated with ground-truth boxes measures mask refinement, not automatic object detection."

    output = args.output or Path(cfg["paths"]["output_dir"]) / f"sam2_prompted_metrics_{args.split}.json"
    save_json(metrics, output)
    print(metrics)
    print(f"Saved metrics to: {output}")


if __name__ == "__main__":
    main()
