from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List
import argparse

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.data_loader import build_dataloaders, UNIFIED_CLASS_NAMES
from models.model import build_model
from utils.metrics import (
    confusion_matrix_from_logits,
    iou_from_confusion_matrix,
    mean_iou_from_confusion_matrix,
    pixel_accuracy_from_confusion_matrix,
)
from utils.visualization import plot_prediction_comparison


def resolve_device(device_cfg: str = "auto") -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg, checkpoint


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: str | Path, cfg: dict | None = None) -> dict:
    device = resolve_device((cfg or {}).get("project", {}).get("device", "auto"))
    model, checkpoint_cfg, _ = load_checkpoint_model(checkpoint_path, device)
    cfg = cfg or checkpoint_cfg

    _, val_loader, _, _ = build_dataloaders(cfg)
    num_classes = cfg["model"].get("num_classes", 9)
    ignore_index = cfg["data"].get("ignore_index", 255)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for batch in tqdm(val_loader, desc=f"eval {Path(checkpoint_path).name}"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        cm += confusion_matrix_from_logits(logits.cpu(), masks.cpu(), num_classes, ignore_index)

    per_class_iou = iou_from_confusion_matrix(cm)
    return {
        "checkpoint": str(checkpoint_path),
        "mIoU": mean_iou_from_confusion_matrix(cm),
        "pixel_accuracy": pixel_accuracy_from_confusion_matrix(cm),
        "per_class_iou": {
            UNIFIED_CLASS_NAMES[i]: float(per_class_iou[i]) for i in range(num_classes)
        },
    }


@torch.no_grad()
def plot_model_comparison(checkpoint_paths: Dict[str, str | Path], cfg_path: str | Path, save_path: str = "four", num_samples: int = 3) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg["project"].get("device", "auto"))
    _, val_loader, _, _ = build_dataloaders(cfg)

    models = {}
    for name, path in checkpoint_paths.items():
        model, _, _ = load_checkpoint_model(path, device)
        models[name] = model

    if save_path == "all":
        output_dir = PROJECT_ROOT / "outputs" / "figures" / "comparison_all"
    else:
        output_dir = PROJECT_ROOT / "outputs" / "figures" / "comparison_four"

    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for batch in val_loader:
        image = batch["image"][0]
        gt_mask = batch["mask"][0]
        image_device = image.unsqueeze(0).to(device)

        predictions = {}
        for name, model in models.items():
            logits = model(image_device)
            predictions[name] = logits.argmax(dim=1)[0].cpu()

        plot_prediction_comparison(
            image,
            gt_mask,
            predictions,
            save_path=output_dir / f"sample_{count:02d}.png",
        )

        count += 1
        if count >= num_samples:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q15",
        action="store_true",
        help="Run the four-model comparison plots.",
    )
    parser.add_argument(
        "--q_final",
        action="store_true",
        help="Run all model comparison plots.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of validation samples to plot.",
    )
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    ckpt_dir = PROJECT_ROOT / "models" / "saved_models"
    checkpoints = sorted(ckpt_dir.glob("*_best.pt"))

    results = [evaluate_checkpoint(path) for path in checkpoints]
    print(json.dumps(results, indent=2))

    logs_dir = PROJECT_ROOT / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(logs_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if args.q15:
        checkpoint_paths = {
            "skip + bilinear": ckpt_dir / "skip_bilinear_best.pt",
            "no skip + bilinear": ckpt_dir / "noskip_bilinear_best.pt",
            "skip + transposed": ckpt_dir / "skip_transposed_best.pt",
            "no skip + transposed": ckpt_dir / "noskip_transposed_best.pt",
        }

        missing = [str(path) for path in checkpoint_paths.values() if not path.exists()]

        if missing:
            print("Skipping model comparison because these checkpoints are missing:")
            for path in missing:
                print(f"  - {path}")
        else:
            plot_model_comparison(
                checkpoint_paths=checkpoint_paths,
                cfg_path=cfg_path,
                save_path="four",
                num_samples=args.num_samples,
            )
            print("Saved comparison plots to outputs/figures/comparison_four/")

    elif args.q_final:
        checkpoint_paths = {
            "skip + bilinear": ckpt_dir / "skip_bilinear_best.pt",
            "no skip + bilinear": ckpt_dir / "noskip_bilinear_best.pt",
            "skip + transposed": ckpt_dir / "skip_transposed_best.pt",
            "no skip + transposed": ckpt_dir / "noskip_transposed_best.pt",
            "best(skip + bilinear) + batch ": ckpt_dir / "best_with_batch_norm_best.pt",
            "best(skip + bilinear) + batch + focal ": ckpt_dir / "best_with_batch_norm_focal_loss_best.pt",
        }

        missing = [str(path) for path in checkpoint_paths.values() if not path.exists()]

        if missing:
            print("Skipping model comparison because these checkpoints are missing:")
            for path in missing:
                print(f"  - {path}")
        else:
            plot_model_comparison(
                checkpoint_paths=checkpoint_paths,
                cfg_path=cfg_path,
                save_path="all",
                num_samples=args.num_samples,
            )
            print("Saved comparison plots to outputs/figures/comparison_all/")
