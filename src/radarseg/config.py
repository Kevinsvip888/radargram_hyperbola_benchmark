from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a YAML configuration is missing required fields."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a dictionary."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ConfigError(f"Config must be a YAML mapping: {path}")

    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    required_top = ["model", "paths", "input", "training", "postprocessing"]
    for key in required_top:
        if key not in cfg:
            raise ConfigError(f"Missing required config section: {key}")

    model_name = cfg["model"].get("name")
    task = cfg["model"].get("task")
    if model_name not in {"unet", "segformer", "mask_rcnn", "mask2former"}:
        raise ConfigError(f"Unsupported model.name: {model_name}")
    if task not in {"semantic", "instance"}:
        raise ConfigError(f"Unsupported model.task: {task}")


def get_path(cfg: Mapping[str, Any], key: str) -> Path:
    value = cfg.get("paths", {}).get(key)
    if value is None:
        raise ConfigError(f"Missing paths.{key} in config")
    return Path(value)
