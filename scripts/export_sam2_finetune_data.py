#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from radarseg.utils.io import read_split_file
from radarseg.utils.masks import mask_to_polygons, masks_to_boxes, read_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the processed dataset to a compact COCO-style bundle that can be adapted "
            "for official SAM 2 fine-tuning experiments."
        )
    )
    parser.add_argument("--processed-root", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("dataset/splits"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset/sam2_finetune"))
    parser.add_argument("--copy-images", action="store_true", help="Copy images into the SAM2 bundle instead of referencing processed paths.")
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    (args.output_root / "annotations").mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        (args.output_root / "images").mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.processed_root / "annotations" / "manifest.csv")
    manifest["image_id"] = manifest["image_id"].astype(str)

    for split in ("train", "val", "test"):
        split_file = args.splits_dir / f"{split}.txt"
        if not split_file.is_file():
            continue
        ids = set(read_split_file(split_file))
        split_df = manifest[manifest["image_id"].isin(ids)].copy()
        coco = _build_coco(split_df, args.processed_root, args.output_root, copy_images=args.copy_images, min_area=args.min_area)
        out_path = args.output_root / "annotations" / f"instances_{split}.json"
        out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        print(f"Saved {split} annotations: {out_path}")

    readme = args.output_root / "README_SAM2_EXPORT.md"
    readme.write_text(
        "# SAM 2 fine-tuning export\n\n"
        "This folder contains COCO-style image/mask annotations exported from the radargram project.\n"
        "Use it as the dataset input when adapting the official facebookresearch/sam2 training configs.\n\n"
        "Important: SAM 2 fine-tuning is not run by this project because SAM 2 uses its own training code,\n"
        "configuration system, checkpoint format, and environment. This export keeps your prepared masks,\n"
        "boxes, areas, polygons, and split definitions in a standard format.\n",
        encoding="utf-8",
    )
    print(f"Saved SAM 2 fine-tuning bundle to: {args.output_root}")


def _build_coco(split_df: pd.DataFrame, processed_root: Path, output_root: Path, *, copy_images: bool, min_area: int) -> dict:
    images = []
    annotations = []
    ann_id = 1
    for image_idx, (_, row) in enumerate(split_df.iterrows(), start=1):
        image_id = str(row["image_id"])
        src_image = processed_root / str(row["image_path"])
        file_name = str(row["image_path"]).replace("\\", "/")
        if copy_images:
            dst_image = output_root / "images" / f"{image_id}{src_image.suffix}"
            shutil.copy2(src_image, dst_image)
            file_name = str(Path("images") / dst_image.name).replace("\\", "/")

        images.append({
            "id": image_idx,
            "file_name": file_name,
            "width": int(row["width"]),
            "height": int(row["height"]),
        })

        instance_dir = processed_root / str(row["instance_mask_dir"])
        for mask_path in sorted(instance_dir.glob("*.png")):
            mask = read_binary_mask(mask_path)
            area = int(mask.sum())
            if area < min_area:
                continue
            x1, y1, x2, y2 = masks_to_boxes(mask[None])[0].tolist()
            annotations.append({
                "id": ann_id,
                "image_id": image_idx,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1 + 1.0, y2 - y1 + 1.0],
                "area": area,
                "iscrowd": 0,
                "segmentation": mask_to_polygons(mask, min_area=min_area),
                "mask_path": str(Path(row["instance_mask_dir"]) / mask_path.name).replace("\\", "/"),
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "hyperbola", "supercategory": "gpr"}],
    }


if __name__ == "__main__":
    main()
