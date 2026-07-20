from __future__ import annotations

import cv2
import numpy as np

from radarseg.utils.masks import ensure_binary, remove_small_components


def clean_binary_mask(mask: np.ndarray, min_area: int = 20, fill_holes: bool = False) -> np.ndarray:
    """Clean a predicted binary mask."""
    mask = remove_small_components(mask, min_area=min_area)
    if not fill_holes:
        return mask

    mask_uint8 = ensure_binary(mask).astype(np.uint8)
    h, w = mask_uint8.shape
    flood = mask_uint8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = 1 - flood
    return np.clip(mask_uint8 + holes, 0, 1).astype(np.uint8)
