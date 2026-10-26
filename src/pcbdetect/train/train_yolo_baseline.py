"""Training pipeline for Baseline 2: YOLOv8 Nano transfer learning.

Trains a fast baseline on DeepPCB (group-disjoint split) to establish
the initial deep learning reference score before advanced fine-tuning in M3.
"""

import argparse
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from pcbdetect.constants import PROJECT_ROOT


def train_yolo_baseline(
    data_yaml: Path = PROJECT_ROOT / "data" / "processed" / "yolo" / "data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 5,
    imgsz: int = 640,
    batch_size: int = 16,
    project_dir: Path = PROJECT_ROOT / "runs" / "baseline",
    experiment_name: str = "yolov8n_baseline",
) -> dict[str, Any]:
    """Train YOLOv8n baseline detector.

    Args:
        data_yaml: Path to data.yaml dataset descriptor.
        model_name: Pretrained model weights identifier (default: yolov8n.pt).
        epochs: Number of training epochs (baseline establishes initial reference).
        imgsz: Square input image resolution (default: 640).
        batch_size: Training batch size.
        project_dir: Directory where runs and weights are saved.
        experiment_name: Name of the run subdirectory.

    Returns:
        Dict containing training metrics and best checkpoint path.
    """
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml}. Run M1 data preparation first!"
        )

    print(f"\n[YOLO Baseline] Loading pretrained weights: {model_name}...")
    model = YOLO(model_name)

    print(
        f"[YOLO Baseline] Starting training: epochs={epochs}, imgsz={imgsz}, batch={batch_size}..."
    )
    start_time = time.perf_counter()

    train_results = model.train(
        data=str(data_yaml).replace("\\", "/"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=str(project_dir).replace("\\", "/"),
        name=experiment_name,
        exist_ok=True,
        workers=2,
        device="cpu",  # portable across CPU and GPU environments
        verbose=True,
        plots=True,
    )

    elapsed_sec = time.perf_counter() - start_time
    save_dir = (
        Path(train_results.save_dir)
        if hasattr(train_results, "save_dir")
        else project_dir / experiment_name
    )
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = save_dir / "weights" / "last.pt"

    # Copy best weights to models/ directory for persistent access
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target_weights = models_dir / "yolov8n_baseline.pt"
    if best_pt.exists():
        import shutil

        shutil.copy2(best_pt, target_weights)
        print(f"[YOLO Baseline] Exported baseline checkpoint to: {target_weights}")

    # Validate model on test set to capture baseline metrics
    print("\n[YOLO Baseline] Evaluating baseline on test set...")
    val_results = model.val(
        data=str(data_yaml).replace("\\", "/"),
        split="test",
        device="cpu",
    )

    metrics_dict = {
        "model_name": f"YOLOv8 Nano ({model_name})",
        "epochs": epochs,
        "training_time_sec": round(elapsed_sec, 2),
        "weights_path": str(target_weights),
        "mAP50": round(float(val_results.box.map50) * 100.0, 2),
        "mAP50_95": round(float(val_results.box.map) * 100.0, 2),
        "precision": round(float(val_results.box.mp) * 100.0, 2),
        "recall": round(float(val_results.box.mr) * 100.0, 2),
    }

    # Per-class recall
    if hasattr(val_results.box, "r") and len(val_results.box.r) > 0:
        from pcbdetect.constants import YOLO_ID_TO_NAME

        class_recalls = {}
        for idx, rec in enumerate(val_results.box.r):
            cls_name = YOLO_ID_TO_NAME.get(idx, f"class_{idx}")
            class_recalls[cls_name] = round(float(rec) * 100.0, 2)
        metrics_dict["class_recalls_pct"] = class_recalls

    print(
        f"\n[YOLO Baseline] Results: mAP@0.5={metrics_dict['mAP50']}%, Recall={metrics_dict['recall']}%, Precision={metrics_dict['precision']}%"
    )
    return metrics_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n Baseline on DeepPCB")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of baseline epochs (default: 5)"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    train_yolo_baseline(epochs=args.epochs, batch_size=args.batch_size, imgsz=args.imgsz)
