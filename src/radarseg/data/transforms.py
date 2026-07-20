from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class ResizeConfig:
    """Target image size used by the model.

    The project uses the common segmentation convention [height, width] in YAML
    files, while PIL expects (width, height). Keeping the conversion in one
    place avoids shape mistakes.
    """

    height: int
    width: int

    @classmethod
    def from_sequence(cls, value: Sequence[int] | None) -> "ResizeConfig | None":
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError(f"image_size must contain [height, width], got: {value}")
        return cls(height=int(value[0]), width=int(value[1]))


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

        return cls(
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
    if resize is None:
        return image
    return image.resize((resize.width, resize.height), resample=Image.BILINEAR)


def resize_mask(mask: Image.Image, resize: ResizeConfig | None) -> Image.Image:
    if resize is None:
        return mask
    return mask.resize((resize.width, resize.height), resample=Image.NEAREST)


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
        Radargram image after optional resizing.
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


def _odd_kernel_size(value: int) -> int:
    """Return a positive odd Gaussian-kernel size."""
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1
