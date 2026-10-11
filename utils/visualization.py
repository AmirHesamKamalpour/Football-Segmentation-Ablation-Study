from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.data_loader import UNIFIED_CLASS_NAMES, UNIFIED_ID_TO_COLOR


def decode_mask(mask: np.ndarray | torch.Tensor, id_to_color: Dict[int, tuple[int, int, int]] | None = None) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    id_to_color = id_to_color or UNIFIED_ID_TO_COLOR
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in id_to_color.items():
        rgb[mask == class_id] = np.asarray(color, dtype=np.uint8)

    return rgb


def denormalize_image(image: torch.Tensor, mean=None, std=None) -> np.ndarray:
    image = image.detach().cpu()
    if mean is not None and std is not None:
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        image = image * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


def plot_sample(image: torch.Tensor, gt_mask: torch.Tensor, pred_mask: torch.Tensor | None = None, save_path: str | Path | None = None) -> None:
    ncols = 3 if pred_mask is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    axes[0].imshow(denormalize_image(image))
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(decode_mask(gt_mask))
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    if pred_mask is not None:
        axes[2].imshow(decode_mask(pred_mask))
        axes[2].set_title("Prediction")
        axes[2].axis("off")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(history: dict, save_path: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="train_loss")
    ax.plot(history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(distribution: dict, save_path: str | Path | None = None) -> None:
    names = distribution.get("class_names", UNIFIED_CLASS_NAMES)
    percentages = distribution["percentages"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, percentages)
    ax.set_ylabel("Pixel percentage")
    ax.set_title("Class distribution")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_train_val_class_distribution_grouped(
    train_distribution: dict,
    val_distribution: dict,
    save_path: str | Path | None = None,
    title: str = "Train vs Validation Class Distribution",
    use_percentages: bool = True,
    annotate: bool = True,
) -> None:
    class_names = train_distribution["class_names"]

    if use_percentages:
        train_values = np.array(train_distribution["percentages"], dtype=float)
        val_values = np.array(val_distribution["percentages"], dtype=float)
        ylabel = "Pixel percentage (%)"
        value_fmt = "%.2f"
    else:
        train_values = np.array(train_distribution["counts"], dtype=float)
        val_values = np.array(val_distribution["counts"], dtype=float)
        ylabel = "Pixel count"
        value_fmt = "%.0f"

    x = np.arange(len(class_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 7))

    train_bars = ax.bar(
        x - width / 2,
        train_values,
        width,
        label="Train",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )

    val_bars = ax.bar(
        x + width / 2,
        val_values,
        width,
        label="Validation",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right")

    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Add a little headroom for labels.
    max_value = max(train_values.max(), val_values.max())
    ax.set_ylim(0, max_value * 1.18 if max_value > 0 else 1)

    if annotate:
        ax.bar_label(train_bars, fmt=value_fmt, padding=3, fontsize=8, rotation=90)
        ax.bar_label(val_bars, fmt=value_fmt, padding=3, fontsize=8, rotation=90)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

def plot_prediction_comparison(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    save_path: str | Path | None = None,
) -> None:
    ncols = 2 + len(predictions)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    axes[0].imshow(denormalize_image(image))
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(decode_mask(gt_mask))
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    for ax, (name, pred) in zip(axes[2:], predictions.items()):
        ax.imshow(decode_mask(pred))
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)