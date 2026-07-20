#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from radarseg.external.yolo11 import export_yolo11_seg_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export processed masks to Ultralytics YOLO11-seg format.")
    parser.add_argument("--processed-root", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("dataset/splits"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset/yolo11_seg"))
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = export_yolo11_seg_dataset(
        args.processed_root,
        args.splits_dir,
        args.output_root,
        min_area=args.min_area,
        overwrite=args.overwrite,
    )
    print(f"Saved YOLO11-seg dataset YAML to: {data_yaml}")


if __name__ == "__main__":
    main()
