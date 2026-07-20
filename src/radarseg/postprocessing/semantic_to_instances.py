from __future__ import annotations

import numpy as np

from radarseg.utils.masks import semantic_to_instance_masks


def split_semantic_prediction(pred_mask: np.ndarray, min_area: int = 20) -> list[np.ndarray]:
    """Split a merged semantic mask into individual hyperbola masks."""
    return semantic_to_instance_masks(pred_mask, min_area=min_area)
