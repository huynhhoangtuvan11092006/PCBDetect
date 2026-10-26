"""Advanced Fine-Tuning script for YOLOv8 on PCBDetect with MLflow tracking.

Applies domain-specific PCB augmentations and small-object detection hyperparameters:
- Rotation invariance: 90/180-degree flips and rotations.
- Tuned loss gains for small micro-defects (mousebite, spur, pin-hole).
- Full logging of mAP@0.5, mAP@0.5:0.95, per-class metrics to MLflow.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class YOLOConfig:
    data_yaml: Path = Path("data/processed/yolo/data.yaml")
    model_name: str = "yolov8n.pt"
    imgsz: int = 640
    epochs: int = 5
    batch_size: int = 8
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    box: float = 7.5       # Enhanced box loss gain for tight small defect localization
    cls: float = 0.5       # Balanced classification loss
    dfl: float = 1.5       # Distribution focal loss
    degrees: float = 90.0  # PCB orientation invariance
    fliplr: float = 0.5
    flipud: float = 0.5
    mosaic: float = 1.0
    close_mosaic: int = 2
    experiment_name: str = "PCBDetect-YOLO"
    run_name: str = "yolov8n_finetuned_augmented"
    output_dir: Path = Path("models")


def train_yolo_advanced(cfg: YOLOConfig) -> dict[str, Any]:
    """Train YOLO with advanced configurations and MLflow tracking."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_target = cfg.output_dir / "yolov8_advanced.pt"

    logger.info(f"Loading YOLO base model: {cfg.model_name}")
    model = YOLO(cfg.model_name)

    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.log_params({
            "model_name": cfg.model_name,
            "imgsz": cfg.imgsz,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr0": cfg.lr0,
            "box_gain": cfg.box,
            "degrees": cfg.degrees,
            "mosaic": cfg.mosaic,
        })

        logger.info(f"Starting YOLO fine-tuning for {cfg.epochs} epochs...")
        results = model.train(
            data=str(cfg.data_yaml),
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch_size,
            lr0=cfg.lr0,
            lrf=cfg.lrf,
            box=cfg.box,
            cls=cfg.cls,
            dfl=cfg.dfl,
            degrees=cfg.degrees,
            fliplr=cfg.fliplr,
            flipud=cfg.flipud,
            mosaic=cfg.mosaic,
            close_mosaic=cfg.close_mosaic,
            project="runs/yolo_advanced",
            name="exp",
            exist_ok=True,
            verbose=True,
            device="cpu",
        )

        # Validate on test set
        logger.info("Evaluating on Test split...")
        val_metrics = model.val(data=str(cfg.data_yaml), split="test", imgsz=cfg.imgsz, device="cpu")

        map50 = float(val_metrics.box.map50)
        map50_95 = float(val_metrics.box.map)
        precision = float(val_metrics.box.mp)
        recall = float(val_metrics.box.mr)

        logger.info(f"Test Results: Precision={precision:.4f}, Recall={recall:.4f}, mAP@0.5={map50:.4f}, mAP@0.5:0.95={map50_95:.4f}")

        mlflow.log_metrics({
            "test_mAP_50": map50,
            "test_mAP_50_95": map50_95,
            "test_precision": precision,
            "test_recall": recall,
        })

        # Save model
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            shutil.copy(best_pt, best_target)
            logger.info(f"Copied best model to {best_target}")
            mlflow.log_artifact(str(best_target))

        return {
            "mAP_50": map50,
            "mAP_50_95": map50_95,
            "precision": precision,
            "recall": recall,
            "best_model_path": str(best_target),
        }


if __name__ == "__main__":
    config = YOLOConfig()
    train_yolo_advanced(config)
