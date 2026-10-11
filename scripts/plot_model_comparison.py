# scripts/plot_model_comparison.py

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "evaluation_results.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_PATTERNS = {
    "Baseline\nskip_bilinear": "skip_bilinear_best",
    "BN": "best_with_batch_norm_best",
    "BN +\nFocal Loss": "best_with_batch_norm_focal_loss_best",
}

METRICS = [
    "mIoU",
    "pixel_accuracy",
    "ball",
    "referee",
    "goalkeeper",
]

DISPLAY_METRICS = [
    "mIoU",
    "Pixel Acc.",
    "Ball IoU",
    "Referee IoU",
    "Goalkeeper IoU",
]


def load_evaluation_results(log_path: Path) -> list[dict]:
    if not log_path.exists():
        raise FileNotFoundError(
            f"Could not find {log_path}. Run `python scripts/evaluate.py` first."
        )

    with open(log_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not isinstance(results, list):
        raise ValueError(
            "Expected evaluation_results.json to contain a list of checkpoint results."
        )

    return results


def find_result(results: list[dict], checkpoint_pattern: str) -> dict:
    matches = [
        result for result in results
        if Path(result["checkpoint"]).stem == checkpoint_pattern
    ]

    if not matches:
        available = "\n".join(
            f"- {Path(result['checkpoint']).stem}" for result in results
        )
        raise ValueError(
            f"Could not find exact checkpoint name: {checkpoint_pattern}\n\n"
            f"Available checkpoints:\n{available}"
        )

    if len(matches) > 1:
        print(
            f"Warning: multiple exact matches for {checkpoint_pattern}. "
            f"Using the first one: {matches[0]['checkpoint']}"
        )

    return matches[0]


def extract_metric(result: dict, metric: str) -> float:
    if metric == "mIoU":
        return float(result["mIoU"])

    if metric == "pixel_accuracy":
        return float(result["pixel_accuracy"])

    return float(result["per_class_iou"][metric])


def build_plot_matrix(results: list[dict]) -> tuple[list[str], np.ndarray]:
    model_names = []
    rows = []

    for display_name, checkpoint_pattern in MODEL_PATTERNS.items():
        result = find_result(results, checkpoint_pattern)
        model_names.append(display_name)

        row = [extract_metric(result, metric) for metric in METRICS]
        rows.append(row)

    return model_names, np.array(rows, dtype=float)


def grouped_bar_chart(model_names: list[str], values: np.ndarray) -> None:
    x = np.arange(len(DISPLAY_METRICS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 7))

    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for i, model_name in enumerate(model_names):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            values[i],
            width,
            label=model_name.replace("\n", " "),
            color=colors[i],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.92,
        )

        ax.bar_label(
            bars,
            labels=[f"{v:.2f}" for v in values[i]],
            padding=3,
            fontsize=9,
        )

    ax.set_title(
        "Model Comparison Across Segmentation Metrics",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )
    ax.set_ylabel("Score", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_METRICS, fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=11, loc="upper center", ncol=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    png_path = OUTPUT_DIR / "model_comparison_grouped_bar.png"
    pdf_path = OUTPUT_DIR / "model_comparison_grouped_bar.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def improvement_from_baseline_chart(model_names: list[str], values: np.ndarray) -> None:
    baseline = values[0]
    improvements = values[1:] - baseline

    x = np.arange(len(DISPLAY_METRICS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 7))

    colors = ["#55A868", "#C44E52"]
    labels = [name.replace("\n", " ") for name in model_names[1:]]

    for i, label in enumerate(labels):
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            improvements[i],
            width,
            label=label,
            color=colors[i],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.92,
        )

        ax.bar_label(
            bars,
            labels=[f"{v:+.2f}" for v in improvements[i]],
            padding=3,
            fontsize=9,
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(
        "Improvement Over Baseline skip_bilinear",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )
    ax.set_ylabel("Score Change", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_METRICS, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=11, loc="upper center", ncol=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    png_path = OUTPUT_DIR / "model_improvement_over_baseline.png"
    pdf_path = OUTPUT_DIR / "model_improvement_over_baseline.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def small_object_focus_chart(model_names: list[str], values: np.ndarray) -> None:
    small_metric_names = ["Ball IoU", "Referee IoU", "Goalkeeper IoU"]
    small_values = values[:, [2, 3, 4]]

    x = np.arange(len(model_names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 7))

    colors = ["#8172B3", "#CCB974", "#64B5CD"]

    for i, metric_name in enumerate(small_metric_names):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            small_values[:, i],
            width,
            label=metric_name,
            color=colors[i],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.92,
        )

        ax.bar_label(
            bars,
            labels=[f"{v:.2f}" for v in small_values[:, i]],
            padding=3,
            fontsize=9,
        )

    ax.set_title(
        "Small / Difficult Object IoU Comparison",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )
    ax.set_ylabel("IoU", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylim(0, max(0.6, float(small_values.max()) + 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=11, loc="upper left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    png_path = OUTPUT_DIR / "small_object_iou_comparison.png"
    pdf_path = OUTPUT_DIR / "small_object_iou_comparison.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    results = load_evaluation_results(LOG_PATH)
    model_names, values = build_plot_matrix(results)

    print("Loaded values from evaluation_results.json:")
    for name, row in zip(model_names, values):
        formatted = ", ".join(
            f"{metric}={value:.4f}"
            for metric, value in zip(DISPLAY_METRICS, row)
        )
        print(f"{name.replace(chr(10), ' ')}: {formatted}")

    grouped_bar_chart(model_names, values)
    improvement_from_baseline_chart(model_names, values)
    small_object_focus_chart(model_names, values)


if __name__ == "__main__":
    main()