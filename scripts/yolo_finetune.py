from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "config" / "ultralytics"))

from ultralytics import YOLO, settings  # noqa: E402


# Project-owned paths.
DATASETS_DIR = PROJECT_ROOT / "data" / "dataset" / "coco"
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"
YOLO_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo"
PRED_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo_predictions"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

RUN_NAME = "yolov8n_coco8_freeze10"


def ensure_dirs() -> None:
    for path in [DATASETS_DIR, PRETRAINED_DIR, YOLO_OUTPUT_DIR, PRED_OUTPUT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def configure_ultralytics_paths() -> None:
    """
    Force Ultralytics-managed artifacts into the homework project structure.

    datasets_dir -> data/dataset
    weights_dir  -> models/pretrained
    runs_dir     -> outputs/yolo
    """
    settings.update(
        {
            "datasets_dir": str(DATASETS_DIR),
            "weights_dir": str(PRETRAINED_DIR),
            "runs_dir": str(YOLO_OUTPUT_DIR),
        }
    )


def load_pretrained_model() -> YOLO:
    pretrained_path = PRETRAINED_DIR / "yolov8n.pt"

    if pretrained_path.exists():
        return YOLO(str(pretrained_path))

    old_cwd = Path.cwd()
    try:
        os.chdir(PRETRAINED_DIR)
        model = YOLO("yolov8n.pt")
    finally:
        os.chdir(old_cwd)

    # In most versions this will download to PRETRAINED_DIR/yolov8n.pt.
    if pretrained_path.exists():
        return YOLO(str(pretrained_path))

    # Fallback: use the loaded model object directly.
    return model


def find_coco8_val_image() -> Path | None:
    """
    Search only project-controlled dataset locations first.
    """
    candidates = [
        DATASETS_DIR / "coco8" / "images" / "val",
        PROJECT_ROOT / "coco8" / "images" / "val",
    ]

    for directory in candidates:
        if directory.exists():
            images = sorted(
                p for p in directory.glob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if images:
                return images[0]

    for image in sorted(DATASETS_DIR.rglob("*")):
        if image.suffix.lower() in {".jpg", ".jpeg", ".png"} and "val" in image.parts:
            return image

    return None


def copy_report_assets(run_dir: Path, pred_dir: Path | None) -> None:

    report_dir = PROJECT_ROOT / "outputs" / "report_assets"
    report_dir.mkdir(parents=True, exist_ok=True)

    results_png = run_dir / "results.png"
    if results_png.exists():
        shutil.copy2(results_png, report_dir / "yolo_results.png")

    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, PROJECT_ROOT / "models" / "saved_models" / "yolov8n_coco8_best.pt")

    if pred_dir is not None and pred_dir.exists():
        predicted_images = sorted(
            p for p in pred_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if predicted_images:
            shutil.copy2(predicted_images[0], report_dir / "yolo_validation_prediction.jpg")


def main() -> None:
    ensure_dirs()
    configure_ultralytics_paths()

    model = load_pretrained_model()

    train_results = model.train(
        data="coco8.yaml",
        epochs=50,
        imgsz=640,
        freeze=10,
        project=str(YOLO_OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,
    )

    run_dir = Path(model.trainer.save_dir).resolve()
    best_model_path = run_dir / "weights" / "best.pt"
    results_plot = run_dir / "results.png"

    print(f"Training run directory: {run_dir}")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Results plot: {results_plot}")

    val_image = find_coco8_val_image()
    pred_dir = None

    if val_image is not None and best_model_path.exists():
        trained_model = YOLO(str(best_model_path))

        pred_results = trained_model.predict(
            source=str(val_image),
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(PRED_OUTPUT_DIR),
            name="val_prediction",
            exist_ok=True,
            conf=0.25,
        )
        
        pred_dir = Path(trained_model.predictor.save_dir).resolve()

        print(f"Validation image: {val_image}")
        print(f"Prediction output directory: {pred_dir}")

    else:
        print("Could not auto-locate a COCO8 validation image or best.pt was not found.")

    copy_report_assets(run_dir, pred_dir)

    summary = {
        "dataset_dir": str(DATASETS_DIR),
        "pretrained_dir": str(PRETRAINED_DIR),
        "training_run_dir": str(run_dir),
        "best_model": str(best_model_path),
        "last_model": str(run_dir / "weights" / "last.pt"),
        "results_plot": str(results_plot),
        "validation_image": str(val_image) if val_image else None,
        "prediction_dir": str(pred_dir) if pred_dir else None,
        "report_assets_dir": str(PROJECT_ROOT / "outputs" / "report_assets"),
    }

    summary_path = LOG_DIR / "yolo_finetune_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()