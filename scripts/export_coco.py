#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from radarseg.utils.io import read_split_file
from radarseg.utils.masks import mask_to_polygons, masks_to_boxes, read_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export processed dataset split to COCO-style JSON.")
    parser.add_argument("--processed-root", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("dataset/splits"))
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-area", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.processed_root / "annotations" / "manifest.csv")
    if args.split != "all":
        ids = set(read_split_file(args.splits_dir / f"{args.split}.txt"))
        manifest = manifest[manifest["image_id"].astype(str).isin(ids)].copy()

    images = []
    annotations = []
    ann_id = 1
    for image_idx, (_, row) in enumerate(manifest.iterrows(), start=1):
        images.append({
            "id": image_idx,
            "file_name": str(row["image_path"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
        })
        instance_dir = args.processed_root / str(row["instance_mask_dir"])
        for mask_path in sorted(instance_dir.glob("*.png")):
            mask = read_binary_mask(mask_path)
            area = int(mask.sum())
            if area < args.min_area:
                continue
            box = masks_to_boxes(mask[None])[0].tolist()
            x1, y1, x2, y2 = box
            annotations.append({
                "id": ann_id,
                "image_id": image_idx,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1 + 1.0, y2 - y1 + 1.0],
                "area": area,
                "iscrowd": 0,
                "segmentation": mask_to_polygons(mask, min_area=args.min_area),
                "mask_path": str(Path(row["instance_mask_dir"]) / mask_path.name).replace("\\", "/"),
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "hyperbola", "supercategory": "gpr"}],
    }
    output = args.output or args.processed_root / "annotations" / f"instances_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"Saved COCO-style annotations to: {output}")


if __name__ == "__main__":
    main()
