"""Fine-Tuning script for YOLOv8-Small (yolov8s.pt) on PCBDetect.

Upgrades architecture from YOLOv8n (3.1M params) to YOLOv8s (11.2M params)
to significantly improve feature resolution and reduce confusion on micro-defects.
Logs all metrics, saves best weights to models/yolov8s_advanced.pt.
"""

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class YOLOSmallConfig:
    data_yaml: Path = Path("data/processed/yolo/data.yaml")
    model_name: str = "c:/PCBDetect/yolov8s.pt"
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
    output_dir: Path = Path("models")
    metrics_output: Path = Path("docs/yolov8s_metrics.json")


def train_yolo_small(cfg: YOLOSmallConfig) -> dict[str, Any]:
    """Train YOLOv8s and evaluate on test set."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_target = cfg.output_dir / "yolov8s_advanced.pt"

    logger.info(f"Loading YOLOv8s base model: {cfg.model_name}")
    model = YOLO(cfg.model_name)

    start_train_time = time.time()
    logger.info(f"Starting YOLOv8s fine-tuning for {cfg.epochs} epochs on CPU...")
    results = model.train(
        data=str(cfg.data_yaml.resolve()).replace("\\", "/"),
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
        project="runs/yolo_small",
        name="exp",
        exist_ok=True,
        verbose=True,
        device="cpu",
    )
    train_duration = time.time() - start_train_time
    logger.info(f"Training completed in {train_duration:.1f}s.")

    # Validate on test set
    logger.info("Evaluating YOLOv8s on independent Test split (234 images)...")
    val_metrics = model.val(
        data=str(cfg.data_yaml.resolve()).replace("\\", "/"),
        split="test",
        imgsz=cfg.imgsz,
        device="cpu",
        plots=True,
    )

    map50 = float(val_metrics.box.map50)
    map50_95 = float(val_metrics.box.map)
    precision = float(val_metrics.box.mp)
    recall = float(val_metrics.box.mr)
    f1 = float(2 * precision * recall / (precision + recall + 1e-9))

    # Per-class metrics
    class_names = ["open", "short", "mousebite", "spur", "copper", "pin_hole"]
    per_class = {}
    for idx, name in enumerate(class_names):
        p_c = float(val_metrics.box.p[idx]) if idx < len(val_metrics.box.p) else 0.0
        r_c = float(val_metrics.box.r[idx]) if idx < len(val_metrics.box.r) else 0.0
        ap50_c = float(val_metrics.box.ap50[idx]) if idx < len(val_metrics.box.ap50) else 0.0
        f1_c = float(2 * p_c * r_c / (p_c + r_c + 1e-9))
        per_class[name] = {
            "precision": p_c,
            "recall": r_c,
            "f1": f1_c,
            "ap50": ap50_c,
        }

    logger.info("=== YOLOv8s TEST EVALUATION RESULTS ===")
    logger.info(f"mAP@0.5: {map50 * 100:.2f}%")
    logger.info(f"mAP@0.5:0.95: {map50_95 * 100:.2f}%")
    logger.info(f"Precision: {precision * 100:.2f}%")
    logger.info(f"Recall: {recall * 100:.2f}%")
    logger.info(f"F1-Score: {f1 * 100:.2f}%")

    # Save model
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy(best_pt, best_target)
        logger.info(f"Copied best YOLOv8s weights to {best_target}")

    report = {
        "model_architecture": "YOLOv8s (Small, 11.2M parameters)",
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "train_duration_seconds": train_duration,
        "overall_metrics": {
            "mAP_50": map50,
            "mAP_50_95": map50_95,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "per_class_metrics": per_class,
        "weights_path": str(best_target),
    }

    cfg.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.metrics_output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Metrics saved to {cfg.metrics_output}")

    return report


if __name__ == "__main__":
    config = YOLOSmallConfig()
    train_yolo_small(config)
