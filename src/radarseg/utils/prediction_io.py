from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from radarseg.utils.coordinates import save_pixel_coordinates_csv
from radarseg.utils.io import ensure_dir, save_json
from radarseg.utils.masks import save_binary_mask, union_masks
from radarseg.utils.visualization import overlay_masks_on_image


def save_prediction_outputs(image_path: str | Path, instances: list[dict[str, Any]], output_dir: str | Path) -> None:
    """Save predicted masks, per-instance pixel coordinates, overlay, and JSON.

    The function is shared by all model families so the output format remains
    identical for U-Net, SegFormer, Mask R-CNN, Mask2Former, YOLO11-seg, SAM 2,
    DINOv3, and future models.
    """
    image_path = Path(image_path)
    output_dir = ensure_dir(output_dir)
    objects_dir = ensure_dir(output_dir / "objects")
    coords_dir = ensure_dir(output_dir / "coordinates")

    masks = [item["mask"] for item in instances]
    if masks:
        mask_all = union_masks(masks, shape=masks[0].shape)
    else:
        image = Image.open(image_path)
        mask_all = np.zeros((image.height, image.width), dtype=np.uint8)

    save_binary_mask(mask_all, output_dir / "mask_all_pred.png")
    overlay_image = Image.open(image_path).convert("RGB").resize((mask_all.shape[1], mask_all.shape[0]))
    overlay = overlay_masks_on_image(overlay_image, masks)
    overlay.save(output_dir / "overlay.png")

    payload: dict[str, Any] = {"image_id": image_path.stem, "instances": []}
    for idx, item in enumerate(instances, start=1):
        mask_name = f"object_{idx:03d}_pred.png"
        coord_name = f"object_{idx:03d}_pixels.csv"
        mask_path = objects_dir / mask_name
        coord_path = coords_dir / coord_name
        save_binary_mask(item["mask"], mask_path)
        num_pixels = save_pixel_coordinates_csv(item["mask"], coord_path)
        payload["instances"].append({
            "instance_id": idx,
            "score": float(item.get("score", 1.0)),
            "bbox": [float(v) for v in item.get("bbox", [0, 0, 0, 0])],
            "num_pixels": num_pixels,
            "mask_path": str(Path("objects") / mask_name),
            "pixel_coordinates_path": str(Path("coordinates") / coord_name),
        })

    save_json(payload, output_dir / "prediction.json")
