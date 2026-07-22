from __future__ import annotations

from importlib.resources import path
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=False)
    return checkpoint
