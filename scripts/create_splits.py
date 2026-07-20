#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import math
import random

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create image-level train/val/test splits.")
    parser.add_argument("--manifest", type=Path, default=Path("dataset/processed/annotations/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/splits"))
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_split(ids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def compute_split_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Compute robust train/val/test counts, including small datasets."""
    if n < 3:
        raise ValueError("Need at least 3 images to create train/val/test splits.")

    raw = [ratio * n for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remaining = n - sum(counts)
    fractions = sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in fractions[:remaining]:
        counts[i] += 1

    # Keep each nonzero-ratio split non-empty when possible.
    for i, ratio in enumerate(ratios):
        if ratio > 0 and counts[i] == 0:
            donor = max(range(3), key=lambda j: counts[j])
            if counts[donor] <= 1:
                raise ValueError(f"Cannot make all requested splits non-empty with n={n}.")
            counts[donor] -= 1
            counts[i] += 1

    if sum(counts) != n or any(c < 0 for c in counts):
        raise RuntimeError(f"Invalid split counts computed: {counts}")
    return counts[0], counts[1], counts[2]


def main() -> None:
    args = parse_args()
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    df = pd.read_csv(args.manifest)
    image_ids = df["image_id"].astype(str).tolist()
    n_train, n_val, n_test = compute_split_counts(len(image_ids), (args.train, args.val, args.test))

    rng = random.Random(args.seed)
    rng.shuffle(image_ids)
    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:n_train + n_val + n_test]

    save_split(sorted(train_ids), args.output_dir / "train.txt")
    save_split(sorted(val_ids), args.output_dir / "val.txt")
    save_split(sorted(test_ids), args.output_dir / "test.txt")

    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")
    print(f"Saved splits to: {args.output_dir}")


if __name__ == "__main__":
    main()
