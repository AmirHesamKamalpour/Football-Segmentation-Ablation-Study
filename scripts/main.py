from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.data_loader import FootballSegmentationDataset, build_dataloaders, compute_class_distribution
from scripts.train import apply_experiment_to_cfg, run_training
from utils.visualization import plot_class_distribution


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")

    # First validate data loading and analyze class distribution.
    train_loader, val_loader, train_samples, val_samples = build_dataloaders(cfg)
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Validation datasets: {sorted(set(s.dataset_name for s in val_samples))}")

    train_dist = compute_class_distribution(train_loader.dataset, cfg["classes"]["num_classes"])
    val_dist = compute_class_distribution(val_loader.dataset, cfg["classes"]["num_classes"])

    out_log = PROJECT_ROOT / "outputs" / "logs"
    out_log.mkdir(parents=True, exist_ok=True)
    with open(out_log / "class_distribution.json", "w", encoding="utf-8") as f:
        json.dump({"train": train_dist, "validation": val_dist}, f, indent=2)

    plot_class_distribution(train_dist, PROJECT_ROOT / "outputs" / "figures" / "train_class_distribution.png")
    plot_class_distribution(val_dist, PROJECT_ROOT / "outputs" / "figures" / "val_class_distribution.png")

    # Train all four baseline configurations.
    results = []
    for experiment in cfg.get("experiments", []):
        exp_cfg = apply_experiment_to_cfg(cfg, experiment)
        results.append(run_training(exp_cfg))

    with open(out_log / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
