from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from radarseg.data.transforms import (
    AugmentationConfig,
    ResizeConfig,
    apply_paired_augmentation,
    image_to_tensor,
    mask_to_tensor,
    resize_image,
    resize_mask,
)
from radarseg.utils.io import read_split_file
from radarseg.utils.masks import masks_to_boxes, read_binary_mask


@dataclass(frozen=True)
class SampleRecord:
    image_id: str
    image_path: Path
    semantic_mask_path: Path
    instance_mask_dir: Path
    width: int
    height: int
    num_instances: int


def load_manifest(processed_root: str | Path) -> pd.DataFrame:
    """Load the processed-dataset manifest created by prepare_dataset.py."""
    manifest_path = Path(processed_root) / "annotations" / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run scripts/prepare_dataset.py first.")
    df = pd.read_csv(manifest_path)
    required = {"image_id", "image_path", "semantic_mask_path", "instance_mask_dir", "width", "height", "num_instances"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return df


def filter_manifest_by_split(df: pd.DataFrame, splits_dir: str | Path, split: str) -> pd.DataFrame:
    """Keep only the records belonging to one split."""
    split_path = Path(splits_dir) / f"{split}.txt"
    image_ids = set(read_split_file(split_path))
    filtered = df[df["image_id"].isin(image_ids)].copy()
    if filtered.empty:
        raise ValueError(f"Split '{split}' is empty or does not match manifest IDs: {split_path}")
    return filtered.reset_index(drop=True)


def row_to_record(row: pd.Series, processed_root: str | Path) -> SampleRecord:
    processed_root = Path(processed_root)
    return SampleRecord(
        image_id=str(row["image_id"]),
        image_path=processed_root / str(row["image_path"]),
        semantic_mask_path=processed_root / str(row["semantic_mask_path"]),
        instance_mask_dir=processed_root / str(row["instance_mask_dir"]),
        width=int(row["width"]),
        height=int(row["height"]),
        num_instances=int(row["num_instances"]),
    )


class RadargramSemanticDataset(Dataset):
    """Dataset for U-Net and SegFormer binary semantic segmentation.

    Augmentation is optional and should normally be enabled only for the train
    split. The same geometric transforms are applied to the image and semantic
    mask, while intensity transforms affect only the radargram image.
    """

    def __init__(
        self,
        processed_root: str | Path,
        splits_dir: str | Path,
        split: str,
        image_size: Sequence[int] | None = None,
        grayscale: bool = False,
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.split = split
        self.resize = ResizeConfig.from_sequence(image_size)
        self.grayscale = grayscale
        self.augmentation = augmentation
        df = filter_manifest_by_split(load_manifest(self.processed_root), splits_dir, split)
        self.records = [row_to_record(row, self.processed_root) for _, row in df.iterrows()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        image = Image.open(record.image_path).convert("L" if self.grayscale else "RGB")
        mask = Image.open(record.semantic_mask_path).convert("L")

        # Resize before augmentation so all random geometric transforms operate
        # on the actual model input size.
        image = resize_image(image, self.resize)
        mask = resize_mask(mask, self.resize)
        image, masks = apply_paired_augmentation(image, [mask], self.augmentation)
        mask = masks[0]

        image_tensor = image_to_tensor(image, grayscale=self.grayscale)
        mask_tensor = mask_to_tensor(mask)
        return image_tensor, mask_tensor


class RadargramInstanceDataset(Dataset):
    """Dataset for Mask R-CNN and Mask2Former instance segmentation.

    The raw annotation is one binary mask per hyperbola. Bounding boxes, areas,
    and valid-object filtering are computed automatically from the masks after
    resizing and augmentation.
    """

    def __init__(
        self,
        processed_root: str | Path,
        splits_dir: str | Path,
        split: str,
        image_size: Sequence[int] | None = None,
        grayscale: bool = False,
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.split = split
        self.resize = ResizeConfig.from_sequence(image_size)
        self.grayscale = grayscale
        self.augmentation = augmentation
        df = filter_manifest_by_split(load_manifest(self.processed_root), splits_dir, split)
        self.records = [row_to_record(row, self.processed_root) for _, row in df.iterrows()]

    def __len__(self) -> int:
        return len(self.records)

    def _load_instance_masks(self, record: SampleRecord) -> list[Image.Image]:
        mask_paths = sorted(record.instance_mask_dir.glob("*.png"))
        masks: list[Image.Image] = []
        for path in mask_paths:
            mask = Image.open(path).convert("L")
            masks.append(resize_mask(mask, self.resize))
        return masks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        record = self.records[idx]
        image = Image.open(record.image_path).convert("L" if self.grayscale else "RGB")
        image = resize_image(image, self.resize)

        mask_images = self._load_instance_masks(record)
        image, mask_images = apply_paired_augmentation(image, mask_images, self.augmentation)
        image_tensor = image_to_tensor(image, grayscale=self.grayscale)

        mask_arrays = []
        for mask_img in mask_images:
            mask = np.asarray(mask_img.convert("L"), dtype=np.uint8)
            mask = (mask > 0).astype(np.uint8)
            if mask.sum() > 0:
                mask_arrays.append(mask)

        if mask_arrays:
            masks_np = np.stack(mask_arrays, axis=0).astype(np.uint8)
            boxes_np = masks_to_boxes(masks_np)
            valid = (boxes_np[:, 2] > boxes_np[:, 0]) & (boxes_np[:, 3] > boxes_np[:, 1])
            masks_np = masks_np[valid]
            boxes_np = boxes_np[valid]
        else:
            h, w = image_tensor.shape[-2:]
            masks_np = np.zeros((0, h, w), dtype=np.uint8)
            boxes_np = np.zeros((0, 4), dtype=np.float32)

        masks = torch.as_tensor(masks_np, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
        labels = torch.ones((masks.shape[0],), dtype=torch.int64)
        area = torch.as_tensor(masks_np.reshape(masks_np.shape[0], -1).sum(axis=1), dtype=torch.float32)
        iscrowd = torch.zeros((masks.shape[0],), dtype=torch.int64)

        target: dict[str, torch.Tensor | str] = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
            "image_name": record.image_id,
        }
        return image_tensor, target


def load_raw_image_for_prediction(path: str | Path, image_size: Sequence[int] | None = None, grayscale: bool = False) -> torch.Tensor:
    """Load a single image for inference. No augmentation is applied."""
    resize = ResizeConfig.from_sequence(image_size)
    image = Image.open(path).convert("L" if grayscale else "RGB")
    image = resize_image(image, resize)
    return image_to_tensor(image, grayscale=grayscale)


def read_instance_masks_from_record(record: SampleRecord) -> list[np.ndarray]:
    mask_paths = sorted(record.instance_mask_dir.glob("*.png"))
    return [read_binary_mask(path) for path in mask_paths]
