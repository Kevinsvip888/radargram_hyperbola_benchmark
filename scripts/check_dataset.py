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
from PIL import Image

from radarseg.utils.masks import read_binary_mask, union_masks
from radarseg.utils.visualization import save_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check processed radargram segmentation dataset.")
    parser.add_argument("--processed-root", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--save-overlays", type=Path, default=None, help="Optional folder for image/mask overlays.")
    parser.add_argument("--semantic-tolerance", type=float, default=0.01, help="Allowed semantic-vs-union mismatch ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.processed_root / "annotations" / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []

    if args.save_overlays:
        args.save_overlays.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        image_path = args.processed_root / str(row["image_path"])
        semantic_path = args.processed_root / str(row["semantic_mask_path"])
        instance_dir = args.processed_root / str(row["instance_mask_dir"])

        if not image_path.is_file():
            errors.append(f"{image_id}: missing image {image_path}")
            continue
        if not semantic_path.is_file():
            errors.append(f"{image_id}: missing semantic mask {semantic_path}")
            continue
        if not instance_dir.is_dir():
            errors.append(f"{image_id}: missing instance mask directory {instance_dir}")
            continue

        image = Image.open(image_path)
        width, height = image.size
        semantic = read_binary_mask(semantic_path)
        if semantic.shape != (height, width):
            errors.append(f"{image_id}: semantic mask shape {semantic.shape} != image shape {(height, width)}")
            continue

        object_masks = []
        for mask_path in sorted(instance_dir.glob("*.png")):
            mask = read_binary_mask(mask_path)
            if mask.shape != (height, width):
                errors.append(f"{image_id}: object mask {mask_path.name} shape {mask.shape} != image shape {(height, width)}")
                continue
            if int(mask.sum()) == 0:
                errors.append(f"{image_id}: empty object mask {mask_path.name}")
                continue
            object_masks.append(mask)

        if not object_masks:
            errors.append(f"{image_id}: no valid object masks")
            continue

        union = union_masks(object_masks, shape=(height, width))
        mismatch = np.logical_xor(semantic > 0, union > 0).sum() / max(int(union.size), 1)
        if mismatch > args.semantic_tolerance:
            warnings.append(f"{image_id}: semantic mask differs from union by {mismatch:.4%}")

        expected_count = int(row["num_instances"])
        if expected_count != len(object_masks):
            warnings.append(f"{image_id}: manifest num_instances={expected_count}, found={len(object_masks)}")

        if args.save_overlays:
            save_overlay(image_path, object_masks, args.save_overlays / f"{image_id}_overlay.png")

    print(f"Checked {len(df)} records.")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    for item in errors[:50]:
        print(f"[ERROR] {item}")
    for item in warnings[:50]:
        print(f"[WARN] {item}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
