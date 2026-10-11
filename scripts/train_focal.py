from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.train import run_training


def main() -> None:
    with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Set to the best model found after adding Batch Normalization.
    cfg = copy.deepcopy(cfg)
    cfg["experiment_name"] = "best_with_batch_norm_focal_loss"
    cfg["model"]["up_mode"] = "bilinear"
    cfg["model"]["use_skip_connections"] = True
    cfg["model"]["use_batch_norm"] = True
    cfg["training"]["loss"] = "focal"
    cfg["training"]["focal_gamma"] = 2.0

    result = run_training(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
