from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from radarseg.data.transforms import SpatialTransformMeta, map_instances_to_original
from radarseg.utils.coordinates import save_pixel_coordinates_csv
from radarseg.utils.io import ensure_dir, save_json
from radarseg.utils.masks import save_binary_mask, union_masks
from radarseg.utils.visualization import overlay_masks_on_image


def save_prediction_outputs(
    image_path: str | Path,
    instances: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    transform_meta: SpatialTransformMeta | None = None,
) -> None:
    """Save predicted masks, per-instance pixel coordinates, overlay, and JSON.

    Parameters
    ----------
    image_path:
        Path to the original input image.
    instances:
        Prediction dictionaries. Their masks may be in model-input coordinates.
    output_dir:
        Folder where outputs for one image are written.
    transform_meta:
        Spatial preprocessing metadata returned by the prediction image loader.
        When provided, masks and boxes are converted back to the original image
        coordinate system before saving. Therefore CSV coordinates always match
        the original radargram pixels.
    """
    image_path = Path(image_path)
    output_dir = ensure_dir(output_dir)
    objects_dir = ensure_dir(output_dir / "objects")
    coords_dir = ensure_dir(output_dir / "coordinates")

    original_image = Image.open(image_path).convert("RGB")
    original_shape = (original_image.height, original_image.width)

    instances_to_save = map_instances_to_original(instances, transform_meta) if transform_meta is not None else instances
    masks = [np.asarray(item["mask"] > 0, dtype=np.uint8) for item in instances_to_save if "mask" in item]

    if masks:
        mask_all = union_masks(masks, shape=masks[0].shape)
    else:
        # With transform metadata, empty predictions should still produce an
        # empty mask in the original image coordinate system.
        if transform_meta is not None:
            mask_all = np.zeros((transform_meta.original_height, transform_meta.original_width), dtype=np.uint8)
        else:
            mask_all = np.zeros(original_shape, dtype=np.uint8)

    save_binary_mask(mask_all, output_dir / "mask_all_pred.png")

    overlay_base = original_image
    if (overlay_base.height, overlay_base.width) != mask_all.shape:
        overlay_base = overlay_base.resize((mask_all.shape[1], mask_all.shape[0]))
    overlay = overlay_masks_on_image(overlay_base, masks)
    overlay.save(output_dir / "overlay.png")

    payload: dict[str, Any] = {
        "image_id": image_path.stem,
        "coordinate_system": "original_image",
        "original_image_size": {"height": original_image.height, "width": original_image.width},
        "saved_mask_size": {"height": int(mask_all.shape[0]), "width": int(mask_all.shape[1])},
        "preprocessing": transform_meta.to_dict() if transform_meta is not None else None,
        "instances": [],
    }

    for idx, item in enumerate(instances_to_save, start=1):
        mask = np.asarray(item["mask"] > 0, dtype=np.uint8)
        mask_name = f"object_{idx:03d}_pred.png"
        coord_name = f"object_{idx:03d}_pixels.csv"
        mask_path = objects_dir / mask_name
        coord_path = coords_dir / coord_name
        save_binary_mask(mask, mask_path)
        num_pixels = save_pixel_coordinates_csv(mask, coord_path)
        payload["instances"].append({
            "instance_id": idx,
            "score": float(item.get("score", 1.0)),
            "bbox": [float(v) for v in item.get("bbox", [0, 0, 0, 0])],
            "num_pixels": num_pixels,
            "mask_path": str(Path("objects") / mask_name),
            "pixel_coordinates_path": str(Path("coordinates") / coord_name),
        })

    save_json(payload, output_dir / "prediction.json")
