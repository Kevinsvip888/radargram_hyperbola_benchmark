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
from radarseg.data.transforms import AugmentationConfig
from radarseg.engine.trainer import train_mask2former, train_mask_rcnn, train_semantic
from radarseg.models.factory import build_model
from radarseg.utils.seed import get_device, make_torch_generator, seed_worker, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a radargram hyperbola segmentation model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    return parser.parse_args()


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    collate_fn,
    device: torch.device,
    seed: int,
) -> DataLoader:
    """Create a DataLoader with deterministic worker seeding."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_torch_generator(seed),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["training"].get("seed", 42))
    set_seed(seed)

    num_threads = cfg["training"].get("num_threads")
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))

    device = get_device()
    print(f"Using device: {device}")

    processed_root = cfg["paths"]["processed_root"]
    splits_dir = cfg["paths"]["splits_dir"]
    image_size = cfg["input"].get("image_size")
    grayscale = bool(cfg["input"].get("grayscale", False))
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"].get("num_workers", 4))

    model_name = cfg["model"]["name"]
    task = cfg["model"]["task"]
    train_augmentation = AugmentationConfig.from_mapping(cfg.get("augmentation"))
    val_augmentation = None

    if train_augmentation.enabled:
        print("Training augmentation: enabled")
    else:
        print("Training augmentation: disabled")

    if task == "semantic":
        train_ds = RadargramSemanticDataset(
            processed_root,
            splits_dir,
            "train",
            image_size=image_size,
            grayscale=grayscale,
            augmentation=train_augmentation,
        )
        val_ds = RadargramSemanticDataset(
            processed_root,
            splits_dir,
            "val",
            image_size=image_size,
            grayscale=grayscale,
            augmentation=val_augmentation,
        )
        train_loader = _make_loader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=semantic_collate,
            device=device,
            seed=seed,
        )
        val_loader = _make_loader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=semantic_collate,
            device=device,
            seed=seed + 1,
        )
    else:
        train_ds = RadargramInstanceDataset(
            processed_root,
            splits_dir,
            "train",
            image_size=image_size,
            grayscale=grayscale,
            augmentation=train_augmentation,
        )
        val_ds = RadargramInstanceDataset(
            processed_root,
            splits_dir,
            "val",
            image_size=image_size,
            grayscale=grayscale,
            augmentation=val_augmentation,
        )
        collate_fn = mask2former_collate if model_name == "mask2former" else detection_collate
        train_loader = _make_loader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            device=device,
            seed=seed,
        )
        val_loader = _make_loader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            device=device,
            seed=seed + 1,
        )

    model = build_model(cfg).to(device)

    if task == "semantic":
        result = train_semantic(model, train_loader, val_loader, cfg, device)
    elif model_name == "mask_rcnn":
        result = train_mask_rcnn(model, train_loader, val_loader, cfg, device)
    elif model_name == "mask2former":
        result = train_mask2former(model, train_loader, val_loader, cfg, device)
    else:
        raise ValueError(f"Unsupported training combination: task={task}, model={model_name}")

    print("Training complete:", result)


if __name__ == "__main__":
    main()
