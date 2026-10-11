from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.data_loader import build_dataloaders, compute_class_distribution
from utils.visualization import plot_class_distribution, plot_train_val_class_distribution_grouped


def main() -> None:
    with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_loader, val_loader, train_samples, val_samples = build_dataloaders(cfg)

    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print("Validation dataset counts:")
    for name in sorted(set(s.dataset_name for s in val_samples)):
        print(f"  {name}: {sum(s.dataset_name == name for s in val_samples)}")

    train_dist = compute_class_distribution(train_loader.dataset, cfg["classes"]["num_classes"])
    val_dist = compute_class_distribution(val_loader.dataset, cfg["classes"]["num_classes"])

    output_dir = PROJECT_ROOT / "outputs"
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    with open(output_dir / "logs" / "class_distribution.json", "w", encoding="utf-8") as f:
        json.dump({"train": train_dist, "validation": val_dist}, f, indent=2)

    plot_class_distribution(train_dist, output_dir / "figures" / "train_class_distribution.png")
    plot_class_distribution(val_dist, output_dir / "figures" / "val_class_distribution.png")

    plot_train_val_class_distribution_grouped(
    train_dist,
    val_dist,
    save_path=output_dir / "figures" / "train_val_class_distribution_grouped.png",
    title="Train vs Validation Class Distribution",
    use_percentages=True,
    annotate=True,
    )

    plot_train_val_class_distribution_grouped(
        train_dist,
        val_dist,
        save_path=output_dir / "figures" / "train_val_class_distribution_grouped_counts.png",
        title="Train vs Validation Class Distribution Counts",
        use_percentages=False,
        annotate=False,
    )

    print(json.dumps({"train": train_dist, "validation": val_dist}, indent=2))


if __name__ == "__main__":
    main()
