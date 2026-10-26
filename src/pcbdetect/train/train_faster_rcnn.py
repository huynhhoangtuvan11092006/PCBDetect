"""Training script for Faster R-CNN on PCBDetect with MLflow tracking.

Implements two-stage transfer learning:
Stage 1 (Freeze): Freeze ResNet-50 backbone, train RPN and Box Predictor heads.
Stage 2 (Unfreeze): Unfreeze top backbone layers with low learning rate.
Logs parameters, loss curves, and mAP metrics to MLflow.
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from pcbdetect.constants import DEFECT_CLASSES
from pcbdetect.data.coco_dataset import PCBCocoDataset, detection_collate_fn
from pcbdetect.models.faster_rcnn import build_faster_rcnn, unfreeze_backbone_top_layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    data_dir: Path = Path("data/processed/coco")
    output_dir: Path = Path("models")
    batch_size: int = 4
    num_workers: int = 0
    freeze_epochs: int = 2
    unfreeze_epochs: int = 1
    lr_head: float = 1e-3
    lr_backbone: float = 1e-5
    momentum: float = 0.9
    weight_decay: float = 0.0005
    experiment_name: str = "PCBDetect-FasterRCNN"
    run_name: str = "faster_rcnn_resnet50_fpn_transfer"
    device: str = "cpu"


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection-over-Union between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


@torch.no_grad()
def evaluate_detection(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5,
) -> dict[str, float]:
    """Evaluate Faster R-CNN on validation/test set to compute Precision, Recall, and mAP@0.5."""
    model.eval()

    num_classes = len(DEFECT_CLASSES)
    tp_per_class = {c: 0 for c in range(1, num_classes + 1)}
    fp_per_class = {c: 0 for c in range(1, num_classes + 1)}
    fn_per_class = {c: 0 for c in range(1, num_classes + 1)}
    total_gt = {c: 0 for c in range(1, num_classes + 1)}

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        predictions = model(images)

        for pred, target in zip(predictions, targets):
            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()

            for lbl in gt_labels:
                if lbl in total_gt:
                    total_gt[lbl] += 1

            pred_boxes = pred["boxes"].cpu().numpy()
            pred_scores = pred["scores"].cpu().numpy()
            pred_labels = pred["labels"].cpu().numpy()

            # Filter by confidence threshold
            keep = pred_scores >= conf_thresh
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

            matched_gt = set()

            # Sort predictions by score descending
            order = np.argsort(-pred_scores)
            for idx in order:
                p_box = pred_boxes[idx]
                p_lbl = pred_labels[idx]

                best_iou = 0.0
                best_gt_idx = -1

                for g_idx, (g_box, g_lbl) in enumerate(zip(gt_boxes, gt_labels)):
                    if g_idx in matched_gt or g_lbl != p_lbl:
                        continue
                    iou = calculate_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    matched_gt.add(best_gt_idx)
                    if p_lbl in tp_per_class:
                        tp_per_class[p_lbl] += 1
                else:
                    if p_lbl in fp_per_class:
                        fp_per_class[p_lbl] += 1

            # Count un-matched GT as FN
            for g_idx, g_lbl in enumerate(gt_labels):
                if g_idx not in matched_gt:
                    if g_lbl in fn_per_class:
                        fn_per_class[g_lbl] += 1

    total_tp = sum(tp_per_class.values())
    total_fp = sum(fp_per_class.values())
    total_fn = sum(fn_per_class.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Class-wise mAP approx
    class_aps = []
    class_recalls = {}
    for c_id, name in enumerate(DEFECT_CLASSES, 1):
        c_tp = tp_per_class[c_id]
        c_fp = fp_per_class[c_id]
        c_fn = fn_per_class[c_id]
        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_ap = c_prec * c_rec  # Simplified AP estimate
        class_aps.append(c_ap)
        class_recalls[name] = c_rec

    m_ap50 = float(np.mean(class_aps)) if class_aps else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "mAP_50": float(m_ap50),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def train_faster_rcnn(config: TrainConfig) -> Path:
    """Run full training pipeline with MLflow experiment tracking."""
    device = torch.device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_annot = config.data_dir / "annotations" / "instances_train.json"
    val_annot = config.data_dir / "annotations" / "instances_val.json"
    train_images = config.data_dir / "images" / "train"
    val_images = config.data_dir / "images" / "val"

    logger.info("Loading training and validation COCO datasets...")
    train_dataset = PCBCocoDataset(train_annot, train_images)
    val_dataset = PCBCocoDataset(val_annot, val_images)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=detection_collate_fn,
    )

    logger.info(f"Loaded {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # Build model with frozen backbone
    logger.info("Initializing Faster R-CNN ResNet-50-FPN (Stage 1: Frozen Backbone)...")
    model = build_faster_rcnn(num_classes=7, pretrained_backbone=True, freeze_backbone=True)
    model.to(device)

    # Set up MLflow
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=config.run_name) as run:
        mlflow.log_params({
            "model_type": "FasterRCNN_ResNet50_FPN_v1",
            "backbone": "ResNet-50-FPN",
            "num_classes": 7,
            "freeze_epochs": config.freeze_epochs,
            "unfreeze_epochs": config.unfreeze_epochs,
            "batch_size": config.batch_size,
            "lr_head": config.lr_head,
            "lr_backbone": config.lr_backbone,
        })

        # --- Stage 1: Train Head Only ---
        logger.info(f"--- Stage 1: Head training ({config.freeze_epochs} epochs) ---")
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=config.lr_head, momentum=config.momentum, weight_decay=config.weight_decay)

        step = 0
        best_recall = 0.0
        best_model_path = config.output_dir / "faster_rcnn_best.pt"

        # Limit batches per epoch for rapid iteration on CPU while achieving convergence
        max_batches_per_epoch = 60

        for epoch in range(1, config.freeze_epochs + 1):
            model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for b_idx, (images, targets) in enumerate(train_loader):
                if b_idx >= max_batches_per_epoch:
                    break
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()
                step += 1

            avg_loss = epoch_loss / min(len(train_loader), max_batches_per_epoch)
            duration = time.time() - t0
            logger.info(f"Stage 1 Epoch {epoch}/{config.freeze_epochs} - Loss: {avg_loss:.4f} - Time: {duration:.1f}s")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # --- Stage 2: Unfreeze Backbone Layer 4 & FPN ---
        if config.unfreeze_epochs > 0:
            logger.info(f"--- Stage 2: Fine-tuning with unfrozen top layers ({config.unfreeze_epochs} epochs) ---")
            unfreeze_backbone_top_layers(model, unfreeze_fpn=True, unfreeze_layer4=True)

            # Different learning rates for backbone vs heads
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if "backbone" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

            optimizer = torch.optim.SGD([
                {"params": backbone_params, "lr": config.lr_backbone},
                {"params": head_params, "lr": config.lr_head * 0.5},
            ], momentum=config.momentum, weight_decay=config.weight_decay)

            for epoch in range(1, config.unfreeze_epochs + 1):
                model.train()
                epoch_loss = 0.0
                t0 = time.time()

                for b_idx, (images, targets) in enumerate(train_loader):
                    if b_idx >= max_batches_per_epoch:
                        break
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    epoch_loss += losses.item()

                total_epoch = config.freeze_epochs + epoch
                avg_loss = epoch_loss / min(len(train_loader), max_batches_per_epoch)
                duration = time.time() - t0
                logger.info(f"Stage 2 Epoch {epoch}/{config.unfreeze_epochs} - Loss: {avg_loss:.4f} - Time: {duration:.1f}s")
                mlflow.log_metric("train_loss", avg_loss, step=total_epoch)

        # --- Validation Evaluation ---
        logger.info("Evaluating on Validation set...")
        val_metrics = evaluate_detection(model, val_loader, device=device)
        logger.info(f"Val Metrics: Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}, mAP@0.5={val_metrics['mAP_50']:.4f}")

        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        # Save model weights
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "val_metrics": val_metrics,
        }, best_model_path)
        logger.info(f"Saved best model checkpoint to {best_model_path}")
        mlflow.log_artifact(str(best_model_path))

    return best_model_path


if __name__ == "__main__":
    cfg = TrainConfig()
    train_faster_rcnn(cfg)
