#!/usr/bin/env python
"""Prepare radargram hyperbola masks for semantic and instance segmentation.

The script converts raw simulation folders such as

    DATASET_ROOT/sim_0001/radargram_raw_Ez.png
    DATASET_ROOT/sim_0001/MASK/radargram_raw_Ez_mask_semantic.png
    DATASET_ROOT/sim_0001/MASK/objects/*_mask.png

into a single processed dataset used by the training scripts.

Important multi-source behavior
-------------------------------
The default behavior is append-safe. You may run this script several times with
several raw roots and the manifest will be merged instead of overwritten. The
processed image IDs include a source prefix, so two roots that both contain
``sim_0001`` do not overwrite each other.

For example:

    python scripts/prepare_dataset.py --raw-root data/source_a --processed-root dataset/processed
    python scripts/prepare_dataset.py --raw-root data/source_b --processed-root dataset/processed

will create image IDs similar to:

    source_a_1a2b3c4d__sim_0001
    source_b_9e8f7a6b__sim_0001

Use ``--write-mode replace`` or ``--reset`` when you intentionally want to start
from a clean processed dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from PIL import Image

from radarseg.utils.io import ensure_dir
from radarseg.utils.masks import mask_to_polygons, masks_to_boxes, read_binary_mask, save_binary_mask, union_masks

MANIFEST_COLUMNS = [
    "image_id",
    "source_prefix",
    "source_root",
    "original_sim_id",
    "image_path",
    "semantic_mask_path",
    "instance_mask_dir",
    "width",
    "height",
    "num_instances",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare radargram dataset for segmentation training.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        nargs="+",
        required=True,
        help="One or more root folders containing sim_XXXX folders.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("dataset/processed"),
        help="Output processed dataset folder.",
    )
    parser.add_argument("--image-name", default="radargram_raw_Ez.png", help="Image filename inside each simulation folder.")
    parser.add_argument("--mask-dir-name", default="MASK", help="Mask directory name inside each simulation folder.")
    parser.add_argument("--objects-dir-name", default="objects", help="Object mask directory name inside MASK.")
    parser.add_argument(
        "--semantic-mask-name",
        default="radargram_raw_Ez_mask_semantic.png",
        help="Semantic mask filename inside MASK.",
    )
    parser.add_argument(
        "--create-semantic-if-missing",
        action="store_true",
        help="Create semantic masks as union of object masks if missing.",
    )
    parser.add_argument(
        "--repair-semantic",
        action="store_true",
        help="Replace semantic masks by union of object masks.",
    )
    parser.add_argument("--min-area", type=int, default=1, help="Ignore object masks smaller than this pixel area.")
    parser.add_argument(
        "--write-mode",
        choices=["append", "replace"],
        default="append",
        help=(
            "append merges new records with an existing manifest; replace writes a new manifest "
            "containing only the current raw-root input(s). Default: append."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the processed-root before writing. Use only when you want a fully clean processed dataset.",
    )
    parser.add_argument(
        "--source-prefix",
        default=None,
        help=(
            "Optional readable prefix for image IDs. Use only with one --raw-root. "
            "By default, the prefix is built from the raw-root folder name plus a short path hash."
        ),
    )
    parser.add_argument(
        "--no-source-prefix",
        action="store_true",
        help=(
            "Use the original sim folder name as image_id. This reproduces the old behavior, "
            "but it is unsafe when combining multiple raw roots that contain the same sim_XXXX names."
        ),
    )
    return parser.parse_args()


def sanitize_identifier(value: str) -> str:
    """Return a filesystem- and CSV-friendly identifier."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "source"


def short_path_hash(path: Path, length: int = 8) -> str:
    """Create a stable short hash from an absolute path."""
    resolved = str(path.expanduser().resolve())
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:length]


def make_source_prefix(raw_root: Path, explicit_prefix: str | None) -> str:
    """Create a source prefix that keeps IDs unique across several raw roots."""
    if explicit_prefix is not None:
        return sanitize_identifier(explicit_prefix)
    root_name = sanitize_identifier(raw_root.name)
    return f"{root_name}_{short_path_hash(raw_root)}"


def make_image_id(sim_dir: Path, source_prefix: str, no_source_prefix: bool) -> str:
    """Create the processed image ID for one simulation folder."""
    sim_id = sanitize_identifier(sim_dir.name)
    return sim_id if no_source_prefix else f"{source_prefix}__{sim_id}"


def find_sim_dirs(raw_root: Path) -> list[Path]:
    if not raw_root.is_dir():
        raise NotADirectoryError(f"Raw root not found: {raw_root}")
    return sorted([p for p in raw_root.iterdir() if p.is_dir()])


def load_existing_manifest(processed_root: Path) -> list[dict]:
    """Load an existing manifest if present.

    Older manifests from previous versions may not contain source metadata. Missing
    columns are added with empty values so the file remains usable.
    """
    manifest_path = processed_root / "annotations" / "manifest.csv"
    if not manifest_path.is_file():
        return []

    df = pd.read_csv(manifest_path)
    for column in MANIFEST_COLUMNS:
        if column not in df.columns:
            df[column] = "" if column not in {"width", "height", "num_instances"} else 0
    return df[MANIFEST_COLUMNS].to_dict(orient="records")


def merge_records(existing_records: list[dict], new_records: list[dict]) -> list[dict]:
    """Merge records by image_id, replacing duplicate IDs with the newest record."""
    merged: dict[str, dict] = {}
    for record in existing_records + new_records:
        image_id = str(record["image_id"])
        if image_id in merged:
            print(f"[INFO] Replacing existing manifest record for image_id={image_id}")
        merged[image_id] = record
    return [merged[key] for key in sorted(merged)]


def process_one_sim(
    sim_dir: Path,
    raw_root: Path,
    source_prefix: str,
    args: argparse.Namespace,
) -> dict | None:
    """Copy one simulation folder into processed format.

    Object masks are the primary annotation. Bounding boxes are not required as
    input; they are derived later from the saved object masks.
    """
    image_path = sim_dir / args.image_name
    mask_dir = sim_dir / args.mask_dir_name
    semantic_path = mask_dir / args.semantic_mask_name
    objects_dir = mask_dir / args.objects_dir_name

    if not image_path.is_file():
        print(f"[SKIP] Missing image: {image_path}")
        return None
    if not objects_dir.is_dir():
        print(f"[SKIP] Missing objects folder: {objects_dir}")
        return None

    object_paths = sorted(objects_dir.glob("*.png"))
    if not object_paths:
        print(f"[SKIP] No object masks found: {objects_dir}")
        return None

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image_id = make_image_id(sim_dir=sim_dir, source_prefix=source_prefix, no_source_prefix=args.no_source_prefix)

    out_image_rel = Path("images") / f"{image_id}.png"
    out_semantic_rel = Path("semantic_masks") / f"{image_id}.png"
    out_instances_rel = Path("instance_masks") / image_id

    out_image_path = args.processed_root / out_image_rel
    out_semantic_path = args.processed_root / out_semantic_rel
    out_instance_dir = args.processed_root / out_instances_rel

    ensure_dir(out_image_path.parent)
    ensure_dir(out_semantic_path.parent)

    # Clear the instance directory before writing. This avoids stale object_*.png
    # files when a simulation is reprocessed with fewer masks than before.
    if out_instance_dir.exists():
        shutil.rmtree(out_instance_dir)
    ensure_dir(out_instance_dir)

    shutil.copy2(image_path, out_image_path)

    object_masks = []
    saved_object_count = 0
    for obj_path in object_paths:
        mask = read_binary_mask(obj_path)
        if mask.shape != (height, width):
            raise ValueError(f"Mask size mismatch for {obj_path}: mask={mask.shape}, image={(height, width)}")

        area = int(mask.sum())
        if area < args.min_area:
            continue

        object_masks.append(mask)
        saved_object_count += 1
        save_binary_mask(mask, out_instance_dir / f"object_{saved_object_count:03d}.png")

    if saved_object_count == 0:
        print(f"[SKIP] All object masks below min_area for {sim_dir}")
        return None

    semantic_union = union_masks(object_masks, shape=(height, width))
    if args.repair_semantic or not semantic_path.is_file():
        if semantic_path.is_file() or args.create_semantic_if_missing:
            save_binary_mask(semantic_union, out_semantic_path)
        else:
            raise FileNotFoundError(
                f"Missing semantic mask: {semantic_path}. Use --create-semantic-if-missing to build it from object masks."
            )
    else:
        semantic = read_binary_mask(semantic_path)
        if semantic.shape != (height, width):
            raise ValueError(f"Semantic mask size mismatch for {semantic_path}: mask={semantic.shape}, image={(height, width)}")
        save_binary_mask(semantic, out_semantic_path)

    return {
        "image_id": image_id,
        "source_prefix": source_prefix if not args.no_source_prefix else "",
        "source_root": str(raw_root.expanduser().resolve()),
        "original_sim_id": sim_dir.name,
        "image_path": str(out_image_rel).replace("\\", "/"),
        "semantic_mask_path": str(out_semantic_rel).replace("\\", "/"),
        "instance_mask_dir": str(out_instances_rel).replace("\\", "/"),
        "width": width,
        "height": height,
        "num_instances": saved_object_count,
    }


def create_instances_json(processed_root: Path, records: list[dict], min_area: int = 1) -> None:
    """Create one COCO-style JSON file for the full processed manifest."""
    images = []
    annotations = []
    ann_id = 1

    for image_idx, record in enumerate(records, start=1):
        images.append({
            "id": image_idx,
            "file_name": record["image_path"],
            "width": int(record["width"]),
            "height": int(record["height"]),
        })

        instance_dir = processed_root / str(record["instance_mask_dir"])
        if not instance_dir.is_dir():
            print(f"[WARN] Instance mask directory missing while exporting JSON: {instance_dir}")
            continue

        for mask_path in sorted(instance_dir.glob("*.png")):
            mask = read_binary_mask(mask_path)
            if int(mask.sum()) < min_area:
                continue

            box = masks_to_boxes(mask[None, :, :])[0].tolist()
            x1, y1, x2, y2 = box
            bbox_xywh = [x1, y1, max(0.0, x2 - x1 + 1.0), max(0.0, y2 - y1 + 1.0)]
            annotations.append({
                "id": ann_id,
                "image_id": image_idx,
                "category_id": 1,
                "bbox": bbox_xywh,
                "area": int(mask.sum()),
                "iscrowd": 0,
                "segmentation": mask_to_polygons(mask, min_area=min_area),
                "mask_path": str(Path(record["instance_mask_dir"]) / mask_path.name).replace("\\", "/"),
            })
            ann_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "hyperbola", "supercategory": "gpr"}],
    }
    out_path = processed_root / "annotations" / "instances_all.json"
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_manifest(processed_root: Path, records: list[dict]) -> Path:
    """Save records using a stable, explicit column order."""
    manifest_path = processed_root / "annotations" / "manifest.csv"
    ensure_dir(manifest_path.parent)
    df = pd.DataFrame(records)
    for column in MANIFEST_COLUMNS:
        if column not in df.columns:
            df[column] = "" if column not in {"width", "height", "num_instances"} else 0
    df = df[MANIFEST_COLUMNS]
    df.to_csv(manifest_path, index=False)
    return manifest_path


def main() -> None:
    args = parse_args()

    if args.source_prefix is not None and len(args.raw_root) != 1:
        raise ValueError("--source-prefix can only be used when exactly one --raw-root is provided.")
    if args.no_source_prefix and len(args.raw_root) > 1:
        raise ValueError("--no-source-prefix is unsafe with multiple --raw-root values and is therefore not allowed.")
    if args.source_prefix is not None and args.no_source_prefix:
        raise ValueError("Use either --source-prefix or --no-source-prefix, not both.")
    if args.reset and args.processed_root.exists():
        print(f"[INFO] Removing processed root: {args.processed_root}")
        shutil.rmtree(args.processed_root)

    ensure_dir(args.processed_root)
    ensure_dir(args.processed_root / "images")
    ensure_dir(args.processed_root / "semantic_masks")
    ensure_dir(args.processed_root / "instance_masks")
    ensure_dir(args.processed_root / "annotations")

    existing_records = [] if args.write_mode == "replace" else load_existing_manifest(args.processed_root)
    new_records: list[dict] = []

    for raw_root in args.raw_root:
        source_prefix = make_source_prefix(raw_root, args.source_prefix)
        print(f"[INFO] Processing raw root: {raw_root}")
        if not args.no_source_prefix:
            print(f"[INFO] Source prefix: {source_prefix}")

        for sim_dir in find_sim_dirs(raw_root):
            record = process_one_sim(
                sim_dir=sim_dir,
                raw_root=raw_root,
                source_prefix=source_prefix,
                args=args,
            )
            if record is not None:
                new_records.append(record)

    if not new_records:
        raise RuntimeError("No valid simulation folders were processed.")

    records = merge_records(existing_records, new_records)
    manifest_path = save_manifest(args.processed_root, records)
    create_instances_json(args.processed_root, records, min_area=args.min_area)

    print(f"Processed {len(new_records)} new/updated images.")
    print(f"Total images in manifest: {len(records)}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"COCO-style full annotation saved to: {args.processed_root / 'annotations' / 'instances_all.json'}")


if __name__ == "__main__":
    main()
