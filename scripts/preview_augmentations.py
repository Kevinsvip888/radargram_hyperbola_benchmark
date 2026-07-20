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
import torch
from PIL import Image

from radarseg.config import load_config
from radarseg.data.dataset import RadargramInstanceDataset
from radarseg.data.transforms import AugmentationConfig
from radarseg.utils.io import ensure_dir
from radarseg.utils.seed import set_seed
from radarseg.utils.visualization import overlay_masks_on_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save visual examples of training-time data augmentation.")
    parser.add_argument("--config", type=Path, required=True, help="Training config with dataset and augmentation settings.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where preview PNG files are saved.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to sample from.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of augmented examples to save.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed overriding training.seed for deterministic previews.")
    return parser.parse_args()


def tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
    """Convert a CxHxW float tensor in [0, 1] to a PIL RGB image."""
    image = image.detach().cpu().clamp(0, 1)
    if image.shape[0] == 1:
        arr = (image[0].numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")
    arr = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["training"].get("seed", 42))
    set_seed(seed)

    augmentation = AugmentationConfig.from_mapping(cfg.get("augmentation"))
    if not augmentation.enabled:
        raise ValueError("augmentation.enabled is false in the config. Enable it before previewing augmentations.")

    dataset = RadargramInstanceDataset(
        cfg["paths"]["processed_root"],
        cfg["paths"]["splits_dir"],
        args.split,
        image_size=cfg["input"].get("image_size"),
        grayscale=bool(cfg["input"].get("grayscale", False)),
        augmentation=augmentation,
    )

    output_dir = ensure_dir(args.output_dir)
    n = min(int(args.num_samples), len(dataset))
    for i in range(n):
        image, target = dataset[i]
        masks = [mask.detach().cpu().numpy().astype(np.uint8) for mask in target["masks"]]
        overlay = overlay_masks_on_image(tensor_to_pil_image(image), masks, draw_boxes=True, draw_labels=True)
        image_name = str(target.get("image_name", f"sample_{i:03d}"))
        overlay.save(output_dir / f"{i:03d}_{image_name}_aug_overlay.png")

    print(f"Saved {n} augmentation previews to: {output_dir}")


if __name__ == "__main__":
    main()
