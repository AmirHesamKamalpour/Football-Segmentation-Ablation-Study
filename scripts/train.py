from __future__ import annotations

import copy
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.data_loader import build_dataloaders
from models.model import build_model
from utils.losses import FocalLoss
from utils.metrics import (
    confusion_matrix_from_logits,
    mean_iou_from_confusion_matrix,
    pixel_accuracy_from_confusion_matrix,
)
from utils.visualization import plot_loss_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str = "auto") -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def build_criterion(cfg: dict) -> nn.Module:
    loss_name = cfg["training"].get("loss", "cross_entropy")
    ignore_index = cfg["data"].get("ignore_index", 255)

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)

    if loss_name == "focal":
        return FocalLoss(
            gamma=cfg["training"].get("focal_gamma", 2.0),
            alpha=cfg["training"].get("focal_alpha"),
            ignore_index=ignore_index,
        )

    raise ValueError(f"Unsupported loss: {loss_name}")


def train_one_epoch(model, loader, criterion, optimizer, device, num_classes: int, ignore_index: int) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        cm += confusion_matrix_from_logits(logits.detach().cpu(), masks.detach().cpu(), num_classes, ignore_index)

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    miou = mean_iou_from_confusion_matrix(cm)
    acc = pixel_accuracy_from_confusion_matrix(cm)
    return epoch_loss, miou, acc


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes: int, ignore_index: int) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)
        cm += confusion_matrix_from_logits(logits.detach().cpu(), masks.detach().cpu(), num_classes, ignore_index)

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    miou = mean_iou_from_confusion_matrix(cm)
    acc = pixel_accuracy_from_confusion_matrix(cm)
    return epoch_loss, miou, acc


def apply_experiment_to_cfg(base_cfg: dict, experiment: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for key in ["up_mode", "use_skip_connections", "use_batch_norm"]:
        if key in experiment:
            cfg["model"][key] = experiment[key]
    for key in ["loss"]:
        if key in experiment:
            cfg["training"][key] = experiment[key]
    cfg["experiment_name"] = experiment["name"]
    return cfg


def run_training(cfg: dict) -> dict:
    set_seed(cfg["project"].get("seed", 42))
    device = resolve_device(cfg["project"].get("device", "auto"))

    train_loader, val_loader, train_samples, val_samples = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"].get("learning_rate", 1e-4),
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )

    num_classes = cfg["model"].get("num_classes", 9)
    ignore_index = cfg["data"].get("ignore_index", 255)
    save_dir = PROJECT_ROOT / cfg["training"].get("save_dir", "models/saved_models")
    log_dir = PROJECT_ROOT / cfg["training"].get("log_dir", "outputs/logs")
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    exp_name = cfg.get("experiment_name", "experiment")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_miou": [],
        "val_miou": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_miou = -1.0

    print(f"Running {exp_name} on {device}")
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    for epoch in range(1, cfg["training"].get("epochs", 30) + 1):
        train_loss, train_miou, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, num_classes, ignore_index
        )
        val_loss, val_miou, val_acc = validate(
            model, val_loader, criterion, device, num_classes, ignore_index
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_miou"].append(train_miou)
        history["val_miou"].append(val_miou)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"[{exp_name}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_mIoU={train_miou:.4f}, val_mIoU={val_miou:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "history": history,
            "best_miou": max(best_miou, val_miou),
        }

        torch.save(checkpoint, save_dir / f"{exp_name}_last.pt")
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(checkpoint, save_dir / f"{exp_name}_best.pt")

    with open(log_dir / f"{exp_name}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(log_dir / f"{exp_name}_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *history.keys()])
        for i in range(len(history["train_loss"])):
            writer.writerow([i + 1, *[history[k][i] for k in history]])

    plot_loss_curves(history, PROJECT_ROOT / "outputs" / "figures" / f"{exp_name}_loss.png")
    return {"experiment": exp_name, "best_val_miou": best_miou, "history": history}


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    if cfg.get("experiments"):
        results = []
        for exp in cfg["experiments"]:
            results.append(run_training(apply_experiment_to_cfg(cfg, exp)))
        print(json.dumps(results, indent=2))
    else:
        run_training(cfg)
