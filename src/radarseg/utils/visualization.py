from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .masks import ensure_binary, masks_to_boxes


def load_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def overlay_masks_on_image(
    image: Image.Image | np.ndarray,
    masks: list[np.ndarray],
    alpha: float = 0.45,
    draw_boxes: bool = True,
    draw_labels: bool = True,
) -> Image.Image:
    """Overlay instance masks on an RGB image.

    The colors are deterministic and chosen from a small palette for visual inspection.
    """
    if isinstance(image, np.ndarray):
        base = Image.fromarray(image).convert("RGB")
    else:
        base = image.convert("RGB")

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    palette = [
        (255, 0, 0, int(255 * alpha)),
        (0, 255, 0, int(255 * alpha)),
        (0, 0, 255, int(255 * alpha)),
        (255, 255, 0, int(255 * alpha)),
        (255, 0, 255, int(255 * alpha)),
        (0, 255, 255, int(255 * alpha)),
        (255, 128, 0, int(255 * alpha)),
        (128, 0, 255, int(255 * alpha)),
    ]

    binary_masks = [ensure_binary(mask) for mask in masks]
    for idx, mask in enumerate(binary_masks):
        color = palette[idx % len(palette)]
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba[mask > 0] = color
        mask_layer = Image.fromarray(rgba, mode="RGBA")
        overlay.alpha_composite(mask_layer)

    result = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

    if draw_boxes and binary_masks:
        draw_result = ImageDraw.Draw(result)
        boxes = masks_to_boxes(np.stack(binary_masks, axis=0))
        for idx, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box.tolist()
            draw_result.rectangle([x1, y1, x2, y2], outline="red", width=2)
            if draw_labels:
                draw_result.text((x1 + 2, y1 + 2), f"H{idx}", fill="yellow")

    return result


def save_overlay(image_path: str | Path, masks: list[np.ndarray], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_masks_on_image(load_rgb_image(image_path), masks)
    overlay.save(output_path)
