from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .masks import ensure_binary


def mask_to_coordinates(mask: np.ndarray) -> np.ndarray:
    """Return mask pixel coordinates as an array with columns [x, y]."""
    mask = ensure_binary(mask)
    ys, xs = np.where(mask > 0)
    return np.stack([xs, ys], axis=1).astype(np.int32) if xs.size else np.empty((0, 2), dtype=np.int32)


def save_pixel_coordinates_csv(mask: np.ndarray, path: str | Path) -> int:
    """Save all positive mask pixel coordinates to CSV.

    Returns
    -------
    int
        Number of saved pixels.
    """
    coords = mask_to_coordinates(mask)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(coords.tolist())
    return int(coords.shape[0])
