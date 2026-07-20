from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from tqdm import tqdm

from radarseg.engine.evaluator import evaluate_instance_model, evaluate_semantic_model
from radarseg.engine.losses import build_semantic_loss
from radarseg.utils.checkpoint import save_checkpoint
from radarseg.utils.io import ensure_dir, save_json


class EarlyStopping:
    def __init__(self, patience: int = 20, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.num_bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience




def make_grad_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler | torch.cuda.amp.GradScaler:
    use_amp = bool(enabled) and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(device: torch.device, enabled: bool):
    use_amp = bool(enabled) and device.type == "cuda"
    try:
        return torch.amp.autocast(device_type=device.type, enabled=use_amp)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=use_amp)


def move_targets_to_device(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    moved = []
    for target in targets:
        moved_target = {}
        for key, value in target.items():
            moved_target[key] = value.to(device) if torch.is_tensor(value) else value
        moved.append(moved_target)
    return moved


def train_semantic(
    model: torch.nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    cfg: dict,
    device: torch.device,
) -> dict[str, float]:
    training_cfg = cfg["training"]
    model_name = cfg["model"]["name"]
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    loss_fn = build_semantic_loss(cfg.get("loss", {}), model_name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    scaler = make_grad_scaler(device, bool(training_cfg.get("amp", True)))
    early = EarlyStopping(patience=int(training_cfg.get("patience", 20)), mode="max")

    best_metric = -1.0
    history: list[dict[str, float]] = []
    epochs = int(training_cfg["epochs"])
    threshold = float(training_cfg.get("threshold", cfg["postprocessing"].get("threshold", 0.5)))
    min_area = int(cfg["postprocessing"].get("min_area", 20))

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} train", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=scaler.is_enabled()):
                logits = model(images)
                loss = loss_fn(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        val_metrics = evaluate_semantic_model(
            model,
            val_loader,
            device=device,
            threshold=threshold,
            min_area=min_area,
            model_name=model_name,
        )
        train_loss = float(sum(losses) / max(len(losses), 1))
        val_score = float(val_metrics.get("dice", 0.0))
        row = {"epoch": float(epoch), "train_loss": train_loss, **val_metrics}
        history.append(row)
        save_json(history, output_dir / "history.json")

        print(f"Epoch {epoch:03d} | loss={train_loss:.5f} | val_dice={val_score:.5f} | val_iou={val_metrics.get('iou', 0.0):.5f}")

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, val_metrics)
        if val_score > best_metric:
            best_metric = val_score
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, val_metrics)

        if early.step(val_score):
            print(f"Early stopping after {epoch} epochs. Best validation Dice: {best_metric:.5f}")
            break

    return {"best_val_dice": best_metric}


def train_mask_rcnn(
    model: torch.nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    cfg: dict,
    device: torch.device,
) -> dict[str, float]:
    training_cfg = cfg["training"]
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    early = EarlyStopping(patience=int(training_cfg.get("patience", 20)), mode="max")
    epochs = int(training_cfg["epochs"])
    threshold = float(cfg["postprocessing"].get("threshold", cfg["model"].get("score_threshold", 0.5)))
    min_area = int(cfg["postprocessing"].get("min_area", 20))

    best_metric = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} train", leave=False):
            images = [image.to(device) for image in images]
            targets = move_targets_to_device(targets, device)
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        val_metrics = evaluate_instance_model(
            model,
            val_loader,
            device=device,
            model_name="mask_rcnn",
            threshold=threshold,
            min_area=min_area,
        )
        train_loss = float(sum(losses) / max(len(losses), 1))
        val_score = float(val_metrics.get("instance_mean_matched_iou", 0.0))
        row = {"epoch": float(epoch), "train_loss": train_loss, **val_metrics}
        history.append(row)
        save_json(history, output_dir / "history.json")

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.5f} | "
            f"inst_mIoU={val_score:.5f} | sem_dice={val_metrics.get('semantic_dice', 0.0):.5f}"
        )

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, val_metrics)
        if val_score > best_metric:
            best_metric = val_score
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, val_metrics)

        if early.step(val_score):
            print(f"Early stopping after {epoch} epochs. Best validation instance mIoU: {best_metric:.5f}")
            break

    return {"best_val_instance_miou": best_metric}


def train_mask2former(
    model: torch.nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    cfg: dict,
    device: torch.device,
) -> dict[str, float]:
    training_cfg = cfg["training"]
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    scaler = make_grad_scaler(device, bool(training_cfg.get("amp", True)))
    early = EarlyStopping(patience=int(training_cfg.get("patience", 20)), mode="max")
    epochs = int(training_cfg["epochs"])
    threshold = float(cfg["postprocessing"].get("threshold", cfg["model"].get("score_threshold", 0.5)))
    min_area = int(cfg["postprocessing"].get("min_area", 20))

    best_metric = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} train", leave=False):
            images = images.to(device)
            targets = move_targets_to_device(targets, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=scaler.is_enabled()):
                outputs = model(images, targets=targets)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        val_metrics = evaluate_instance_model(
            model,
            val_loader,
            device=device,
            model_name="mask2former",
            threshold=threshold,
            min_area=min_area,
        )
        train_loss = float(sum(losses) / max(len(losses), 1))
        val_score = float(val_metrics.get("instance_mean_matched_iou", 0.0))
        row = {"epoch": float(epoch), "train_loss": train_loss, **val_metrics}
        history.append(row)
        save_json(history, output_dir / "history.json")

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.5f} | "
            f"inst_mIoU={val_score:.5f} | sem_dice={val_metrics.get('semantic_dice', 0.0):.5f}"
        )

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, val_metrics)
        if val_score > best_metric:
            best_metric = val_score
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, val_metrics)

        if early.step(val_score):
            print(f"Early stopping after {epoch} epochs. Best validation instance mIoU: {best_metric:.5f}")
            break

    return {"best_val_instance_miou": best_metric}
