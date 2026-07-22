from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image


VALID_RESIZE_MODES = {"resize", "letterbox"}


@dataclass(frozen=True)
class ResizeConfig:
    """Spatial preprocessing settings for model inputs.

    The YAML convention in this project is ``image_size: [height, width]``.
    PIL/OpenCV use ``(width, height)``, so all conversions are kept here to
    avoid repeated shape logic throughout the codebase.

    ``mode='resize'`` stretches every image/mask directly to ``image_size``.
    ``mode='letterbox'`` preserves the original aspect ratio and pads the
    remaining area to ``image_size``. When ``allow_upscale=False``, smaller
    images are padded without enlargement; only downscaling is allowed.
    """

    height: int
    width: int
    mode: str = "resize"
    pad_value: int = 0
    allow_upscale: bool = True

    @classmethod
    def from_sequence(cls, value: Sequence[int] | None) -> "ResizeConfig | None":
        """Backward-compatible constructor using direct resize."""
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError(f"image_size must contain [height, width], got: {value}")
        return cls(height=int(value[0]), width=int(value[1]), mode="resize", pad_value=0, allow_upscale=True)

    @classmethod
    def from_settings(
        cls,
        image_size: Sequence[int] | None,
        resize_mode: str = "resize",
        pad_value: int = 0,
        allow_upscale: bool = True,
    ) -> "ResizeConfig | None":
        """Create preprocessing settings from config values."""
        if image_size is None:
            return None
        if len(image_size) != 2:
            raise ValueError(f"image_size must contain [height, width], got: {image_size}")
        cfg = cls(
            height=int(image_size[0]),
            width=int(image_size[1]),
            mode=str(resize_mode),
            pad_value=int(pad_value),
            allow_upscale=bool(allow_upscale),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"image_size values must be positive, got: [{self.height}, {self.width}]")
        if self.mode not in VALID_RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {sorted(VALID_RESIZE_MODES)}, got: {self.mode!r}")
        if not 0 <= int(self.pad_value) <= 255:
            raise ValueError(f"pad_value must be in [0, 255], got: {self.pad_value}")


@dataclass(frozen=True)
class SpatialTransformMeta:
    """Metadata needed to map model-space masks back to the original image.

    Coordinates produced by neural networks are in the preprocessed model-input
    system. This metadata records exactly how the original image was resized or
    letterboxed so predicted masks can be mapped back to the original pixel
    coordinate system before saving CSV files.
    """

    resize_mode: str
    original_height: int
    original_width: int
    processed_height: int
    processed_width: int
    resized_height: int
    resized_width: int
    pad_top: int = 0
    pad_left: int = 0
    pad_bottom: int = 0
    pad_right: int = 0
    scale_x: float = 1.0
    scale_y: float = 1.0
    allow_upscale: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AugmentationConfig:
    """Safe radargram augmentations.

    Geometric transforms are applied to the radargram image and every mask with
    exactly the same parameters. Intensity transforms are applied only to the
    radargram image. The defaults are intentionally conservative because the
    hyperbola shape and vertical axis have physical meaning in GPR radargrams.
    """

    enabled: bool = False

    horizontal_flip_p: float = 0.0

    shift_scale_p: float = 0.0
    shift_limit: float = 0.03
    scale_limit: float = 0.05

    brightness_contrast_p: float = 0.0
    brightness_limit: float = 0.10
    contrast_limit: float = 0.10

    gamma_p: float = 0.0
    gamma_limit: tuple[float, float] = (0.90, 1.10)

    gaussian_noise_p: float = 0.0
    noise_std_limit: tuple[float, float] = (0.005, 0.030)

    gaussian_blur_p: float = 0.0
    blur_kernel_size: int = 3

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "AugmentationConfig":
        """Build augmentation settings from a YAML mapping.

        Missing keys fall back to safe defaults. The parser accepts lists for
        ranges, for example ``gamma_limit: [0.9, 1.1]``.
        """
        if not cfg:
            return cls()

        def _range_pair(name: str, default: tuple[float, float]) -> tuple[float, float]:
            value = cfg.get(name, default)
            if len(value) != 2:
                raise ValueError(f"augmentation.{name} must contain two values, got: {value}")
            lo, hi = float(value[0]), float(value[1])
            if lo > hi:
                raise ValueError(f"augmentation.{name} lower value must be <= upper value, got: {value}")
            return lo, hi

        config = cls(
            enabled=bool(cfg.get("enabled", False)),
            horizontal_flip_p=float(cfg.get("horizontal_flip_p", 0.0)),
            shift_scale_p=float(cfg.get("shift_scale_p", 0.0)),
            shift_limit=float(cfg.get("shift_limit", 0.03)),
            scale_limit=float(cfg.get("scale_limit", 0.05)),
            brightness_contrast_p=float(cfg.get("brightness_contrast_p", 0.0)),
            brightness_limit=float(cfg.get("brightness_limit", 0.10)),
            contrast_limit=float(cfg.get("contrast_limit", 0.10)),
            gamma_p=float(cfg.get("gamma_p", 0.0)),
            gamma_limit=_range_pair("gamma_limit", (0.90, 1.10)),
            gaussian_noise_p=float(cfg.get("gaussian_noise_p", 0.0)),
            noise_std_limit=_range_pair("noise_std_limit", (0.005, 0.030)),
            gaussian_blur_p=float(cfg.get("gaussian_blur_p", 0.0)),
            blur_kernel_size=int(cfg.get("blur_kernel_size", 3)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate probabilities and limits early so errors are easy to trace."""
        probability_fields = {
            "horizontal_flip_p": self.horizontal_flip_p,
            "shift_scale_p": self.shift_scale_p,
            "brightness_contrast_p": self.brightness_contrast_p,
            "gamma_p": self.gamma_p,
            "gaussian_noise_p": self.gaussian_noise_p,
            "gaussian_blur_p": self.gaussian_blur_p,
        }
        for name, value in probability_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"augmentation.{name} must be in [0, 1], got: {value}")
        if self.shift_limit < 0:
            raise ValueError("augmentation.shift_limit must be non-negative")
        if self.scale_limit < 0:
            raise ValueError("augmentation.scale_limit must be non-negative")
        if self.blur_kernel_size < 1:
            raise ValueError("augmentation.blur_kernel_size must be >= 1")


def resize_image(image: Image.Image, resize: ResizeConfig | None) -> Image.Image:
    """Backward-compatible direct resizing helper."""
    if resize is None:
        return image
    return image.resize((resize.width, resize.height), resample=Image.BILINEAR)


def resize_mask(mask: Image.Image, resize: ResizeConfig | None) -> Image.Image:
    """Backward-compatible direct mask resizing helper."""
    if resize is None:
        return mask
    return mask.resize((resize.width, resize.height), resample=Image.NEAREST)


def apply_spatial_preprocessing(
    image: Image.Image,
    masks: list[Image.Image] | None,
    resize: ResizeConfig | None,
) -> tuple[Image.Image, list[Image.Image], SpatialTransformMeta]:
    """Resize or letterbox an image and matching masks.

    The returned metadata is the single source of truth for converting
    predictions from model-input coordinates back to original-image coordinates.
    """
    masks = [] if masks is None else masks
    original_width, original_height = image.size

    if resize is None:
        meta = SpatialTransformMeta(
            resize_mode="none",
            original_height=original_height,
            original_width=original_width,
            processed_height=original_height,
            processed_width=original_width,
            resized_height=original_height,
            resized_width=original_width,
            allow_upscale=True,
        )
        return image, masks, meta

    resize.validate()
    if resize.mode == "resize":
        out_image = image.resize((resize.width, resize.height), resample=Image.BILINEAR)
        out_masks = [mask.resize((resize.width, resize.height), resample=Image.NEAREST) for mask in masks]
        meta = SpatialTransformMeta(
            resize_mode="resize",
            original_height=original_height,
            original_width=original_width,
            processed_height=resize.height,
            processed_width=resize.width,
            resized_height=resize.height,
            resized_width=resize.width,
            scale_x=resize.width / max(original_width, 1),
            scale_y=resize.height / max(original_height, 1),
            allow_upscale=resize.allow_upscale,
        )
        return out_image, out_masks, meta

    # Letterbox: preserve aspect ratio and pad to the requested model size.
    # When allow_upscale=False, smaller images are not enlarged; they are
    # centered on the target canvas with padding only.
    scale = min(resize.width / max(original_width, 1), resize.height / max(original_height, 1))
    if not resize.allow_upscale:
        scale = min(scale, 1.0)
    resized_width = max(1, min(resize.width, int(round(original_width * scale))))
    resized_height = max(1, min(resize.height, int(round(original_height * scale))))
    pad_left = (resize.width - resized_width) // 2
    pad_top = (resize.height - resized_height) // 2
    pad_right = resize.width - resized_width - pad_left
    pad_bottom = resize.height - resized_height - pad_top

    resized_image = image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    padded_image = _new_padded_image(image.mode, resize.width, resize.height, resize.pad_value)
    padded_image.paste(resized_image, (pad_left, pad_top))

    out_masks: list[Image.Image] = []
    for mask in masks:
        resized_mask = mask.resize((resized_width, resized_height), resample=Image.NEAREST)
        padded_mask = Image.new("L", (resize.width, resize.height), color=0)
        padded_mask.paste(resized_mask, (pad_left, pad_top))
        out_masks.append(padded_mask)

    meta = SpatialTransformMeta(
        resize_mode="letterbox",
        original_height=original_height,
        original_width=original_width,
        processed_height=resize.height,
        processed_width=resize.width,
        resized_height=resized_height,
        resized_width=resized_width,
        pad_top=pad_top,
        pad_left=pad_left,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
        scale_x=resized_width / max(original_width, 1),
        scale_y=resized_height / max(original_height, 1),
        allow_upscale=resize.allow_upscale,
    )
    return padded_image, out_masks, meta


def _new_padded_image(mode: str, width: int, height: int, pad_value: int) -> Image.Image:
    if mode == "L":
        return Image.new("L", (width, height), color=int(pad_value))
    # Normalize less common modes to RGB before padding. The dataset loaders call
    # convert("RGB") or convert("L") before this function, so this mainly guards
    # against direct utility use.
    return Image.new("RGB", (width, height), color=(int(pad_value), int(pad_value), int(pad_value)))


def image_to_tensor(image: Image.Image, grayscale: bool = False) -> torch.Tensor:
    """Convert a PIL image to a float tensor in [0, 1], shape C x H x W."""
    if grayscale:
        image = image.convert("L")
        arr = np.asarray(image, dtype=np.float32)[None, :, :] / 255.0
    else:
        image = image.convert("RGB")
        arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr.copy())


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """Convert a PIL mask to a binary float tensor, shape H x W."""
    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    arr = (arr > 0).astype(np.float32)
    return torch.from_numpy(arr.copy())


def normalize_image(
    image: torch.Tensor,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> torch.Tensor:
    """Normalize an image tensor. Defaults to no normalization."""
    if mean is None or std is None:
        return image
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    return (image - mean_t) / std_t.clamp_min(1e-12)


def apply_paired_augmentation(
    image: Image.Image,
    masks: list[Image.Image],
    cfg: AugmentationConfig | None,
) -> tuple[Image.Image, list[Image.Image]]:
    """Apply augmentation to one image and its masks.

    Parameters
    ----------
    image:
        Radargram image after spatial preprocessing.
    masks:
        Semantic mask plus/or instance masks. All masks must match image size.
    cfg:
        Augmentation configuration. When disabled, the input is returned without
        modification.

    Returns
    -------
    tuple
        Augmented image and masks. Masks remain binary PIL ``L`` images.
    """
    if cfg is None or not cfg.enabled:
        return image, masks
    cfg.validate()

    image_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask_arrays = [(np.asarray(mask.convert("L"), dtype=np.uint8) > 0).astype(np.uint8) for mask in masks]

    height, width = image_np.shape[:2]
    for mask in mask_arrays:
        if mask.shape != (height, width):
            raise ValueError(f"Mask/image size mismatch during augmentation: mask={mask.shape}, image={(height, width)}")

    # Horizontal flipping is physically safe for most radargram simulations:
    # it mirrors lateral position but does not invert the time/depth axis.
    if np.random.random() < cfg.horizontal_flip_p:
        image_np = np.ascontiguousarray(image_np[:, ::-1])
        mask_arrays = [np.ascontiguousarray(mask[:, ::-1]) for mask in mask_arrays]

    # Small translation/zoom. No rotation is used because rotation can create
    # physically unrealistic hyperbola orientations.
    if np.random.random() < cfg.shift_scale_p:
        tx = float(np.random.uniform(-cfg.shift_limit, cfg.shift_limit) * width)
        ty = float(np.random.uniform(-cfg.shift_limit, cfg.shift_limit) * height)
        scale = float(1.0 + np.random.uniform(-cfg.scale_limit, cfg.scale_limit))
        matrix = np.array(
            [
                [scale, 0.0, (1.0 - scale) * width / 2.0 + tx],
                [0.0, scale, (1.0 - scale) * height / 2.0 + ty],
            ],
            dtype=np.float32,
        )
        image_np = cv2.warpAffine(
            image_np,
            matrix,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mask_arrays = [
            cv2.warpAffine(
                mask,
                matrix,
                dsize=(width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(np.uint8)
            for mask in mask_arrays
        ]

    image_float = image_np.astype(np.float32) / 255.0

    if np.random.random() < cfg.brightness_contrast_p:
        brightness = float(np.random.uniform(-cfg.brightness_limit, cfg.brightness_limit))
        contrast = float(1.0 + np.random.uniform(-cfg.contrast_limit, cfg.contrast_limit))
        image_float = image_float * contrast + brightness

    if np.random.random() < cfg.gamma_p:
        gamma = float(np.random.uniform(cfg.gamma_limit[0], cfg.gamma_limit[1]))
        image_float = np.power(np.clip(image_float, 0.0, 1.0), gamma)

    if np.random.random() < cfg.gaussian_noise_p:
        std = float(np.random.uniform(cfg.noise_std_limit[0], cfg.noise_std_limit[1]))
        noise = np.random.normal(loc=0.0, scale=std, size=image_float.shape).astype(np.float32)
        image_float = image_float + noise

    image_float = np.clip(image_float, 0.0, 1.0)
    image_np = (image_float * 255.0).round().astype(np.uint8)

    if np.random.random() < cfg.gaussian_blur_p:
        kernel = _odd_kernel_size(cfg.blur_kernel_size)
        image_np = cv2.GaussianBlur(image_np, ksize=(kernel, kernel), sigmaX=0.0)

    out_image = Image.fromarray(image_np, mode="RGB")
    if image.mode == "L":
        out_image = out_image.convert("L")

    out_masks = [Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L") for mask in mask_arrays]
    return out_image, out_masks


def map_mask_to_original(mask: np.ndarray, meta: SpatialTransformMeta | None) -> np.ndarray:
    """Map one model-space binary mask back to the original image shape."""
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if meta is None:
        return mask

    expected_shape = (meta.processed_height, meta.processed_width)
    if mask.shape != expected_shape:
        # Some external libraries may already return original-size masks. If so,
        # keep them unchanged; otherwise resize to the processed canvas before
        # applying the inverse transform.
        original_shape = (meta.original_height, meta.original_width)
        if mask.shape == original_shape:
            return mask
        mask = cv2.resize(mask, dsize=(meta.processed_width, meta.processed_height), interpolation=cv2.INTER_NEAREST)

    if meta.resize_mode == "none":
        return mask

    if meta.resize_mode == "resize":
        out = cv2.resize(mask, dsize=(meta.original_width, meta.original_height), interpolation=cv2.INTER_NEAREST)
        return (out > 0).astype(np.uint8)

    if meta.resize_mode == "letterbox":
        y0 = int(meta.pad_top)
        y1 = int(meta.pad_top + meta.resized_height)
        x0 = int(meta.pad_left)
        x1 = int(meta.pad_left + meta.resized_width)
        cropped = mask[y0:y1, x0:x1]
        if cropped.size == 0:
            return np.zeros((meta.original_height, meta.original_width), dtype=np.uint8)
        out = cv2.resize(cropped, dsize=(meta.original_width, meta.original_height), interpolation=cv2.INTER_NEAREST)
        return (out > 0).astype(np.uint8)

    raise ValueError(f"Unsupported resize mode in transform metadata: {meta.resize_mode!r}")


def map_box_to_original(box: Sequence[float], meta: SpatialTransformMeta | None) -> list[float]:
    """Map one [x_min, y_min, x_max, y_max] box to original image coordinates."""
    x0, y0, x1, y1 = [float(v) for v in box]
    if meta is None or meta.resize_mode == "none":
        return _clip_box([x0, y0, x1, y1], meta)

    if meta.resize_mode == "resize":
        sx = meta.scale_x if meta.scale_x != 0 else 1.0
        sy = meta.scale_y if meta.scale_y != 0 else 1.0
        mapped = [x0 / sx, y0 / sy, x1 / sx, y1 / sy]
        return _clip_box(mapped, meta)

    if meta.resize_mode == "letterbox":
        sx = meta.scale_x if meta.scale_x != 0 else 1.0
        sy = meta.scale_y if meta.scale_y != 0 else 1.0
        mapped = [(x0 - meta.pad_left) / sx, (y0 - meta.pad_top) / sy, (x1 - meta.pad_left) / sx, (y1 - meta.pad_top) / sy]
        return _clip_box(mapped, meta)

    raise ValueError(f"Unsupported resize mode in transform metadata: {meta.resize_mode!r}")


def map_instances_to_original(instances: list[dict[str, Any]], meta: SpatialTransformMeta | None) -> list[dict[str, Any]]:
    """Convert instance masks, boxes, and areas to the original image system."""
    if meta is None:
        return instances

    mapped_instances: list[dict[str, Any]] = []
    for item in instances:
        if "mask" not in item:
            continue
        mapped_mask = map_mask_to_original(item["mask"], meta)
        mapped = dict(item)
        mapped["mask"] = mapped_mask
        mapped["area"] = int(mapped_mask.sum())
        mapped["bbox"] = _mask_to_box(mapped_mask)
        mapped_instances.append(mapped)
    return mapped_instances


def _mask_to_box(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _clip_box(box: Sequence[float], meta: SpatialTransformMeta | None) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    if meta is None:
        return [x0, y0, x1, y1]
    x0 = min(max(x0, 0.0), max(float(meta.original_width - 1), 0.0))
    x1 = min(max(x1, 0.0), max(float(meta.original_width - 1), 0.0))
    y0 = min(max(y0, 0.0), max(float(meta.original_height - 1), 0.0))
    y1 = min(max(y1, 0.0), max(float(meta.original_height - 1), 0.0))
    return [x0, y0, x1, y1]


def _odd_kernel_size(value: int) -> int:
    """Return a positive odd Gaussian-kernel size."""
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1
