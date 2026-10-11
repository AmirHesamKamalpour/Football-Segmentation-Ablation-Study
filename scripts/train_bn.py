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

    # After Q1-4, set these to the best configuration you observed.
    best_base = {
        "name": "best_with_batch_norm",
        "up_mode": "bilinear",
        "use_skip_connections": True,
        "use_batch_norm": True,
        "loss": "cross_entropy",
    }

    cfg = copy.deepcopy(cfg)
    cfg["experiment_name"] = best_base["name"]
    cfg["model"]["up_mode"] = best_base["up_mode"]
    cfg["model"]["use_skip_connections"] = best_base["use_skip_connections"]
    cfg["model"]["use_batch_norm"] = best_base["use_batch_norm"]
    cfg["training"]["loss"] = best_base["loss"]

    result = run_training(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
