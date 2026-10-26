"""Error analysis module for PCB defect detection models.

Diagnoses failure modes:
1. False Negatives (FN - Escape Defects): Missed defects causing field failures.
2. False Positives (FP - Overkill): False alarms causing unnecessary human re-inspection.
3. Confusion Matrix: Misclassifications between visually similar defect categories (e.g. mousebite vs open).
4. Generates visual error case artifacts into docs/figures/error_cases/.
"""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from pcbdetect.constants import DEFECT_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_box_iou(box1: list[float], box2: list[float]) -> float:
    """Calculate IoU for two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def run_error_analysis(
    model_path: Path,
    test_data_dir: Path = Path("data/processed/yolo"),
    output_dir: Path = Path("docs/figures/error_cases"),
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_saved_cases: int = 10,
) -> dict[str, Any]:
    """Perform detailed error analysis on test split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading model for error analysis: {model_path}")
    model = YOLO(str(model_path))

    test_img_dir = test_data_dir / "images" / "test"
    test_label_dir = test_data_dir / "labels" / "test"

    image_files = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))
    logger.info(f"Analyzing {len(image_files)} test images...")

    num_classes = len(DEFECT_CLASSES)
    # Confusion matrix: rows = Ground Truth (0..5, 6=Background/FP), cols = Predicted (0..5, 6=Missed/FN)
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []

    for img_path in image_files:
        lbl_path = test_label_dir / f"{img_path.stem}.txt"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        # Load Ground Truth
        gt_boxes: list[list[float]] = []
        gt_classes: list[int] = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = (cx - bw / 2.0) * w
                        y1 = (cy - bh / 2.0) * h
                        x2 = (cx + bw / 2.0) * w
                        y2 = (cy + bh / 2.0) * h
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_classes.append(cls_id)

        # Run inference
        results = model.predict(source=str(img_path), conf=conf_thresh, verbose=False)
        pred_boxes: list[list[float]] = []
        pred_classes: list[int] = []
        pred_scores: list[float] = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes_data = results[0].boxes.data.cpu().numpy()
            for row in boxes_data:
                x1, y1, x2, y2, sc, cls_id = row[:6]
                pred_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                pred_classes.append(int(cls_id))
                pred_scores.append(float(sc))

        # Match predictions with GT
        matched_gt = set()
        matched_pred = set()

        for p_idx, (p_box, p_cls) in enumerate(zip(pred_boxes, pred_classes)):
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, (g_box, g_cls) in enumerate(zip(gt_boxes, gt_classes)):
                if g_idx in matched_gt:
                    continue
                iou = compute_box_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= iou_thresh and best_g_idx >= 0:
                matched_gt.add(best_g_idx)
                matched_pred.add(p_idx)
                g_cls = gt_classes[best_g_idx]
                conf_matrix[g_cls, p_cls] += 1
            else:
                # False positive: row = Background (num_classes), col = predicted class
                conf_matrix[num_classes, p_cls] += 1
                false_positives.append({
                    "image": img_path.name,
                    "pred_class": DEFECT_CLASSES[p_cls],
                    "score": pred_scores[p_idx],
                    "box": p_box,
                })

        # Check for False Negatives (unmatched ground truth)
        for g_idx, (g_box, g_cls) in enumerate(zip(gt_boxes, gt_classes)):
            if g_idx not in matched_gt:
                # False negative: row = true class, col = Background/Missed (num_classes)
                conf_matrix[g_cls, num_classes] += 1
                box_w = g_box[2] - g_box[0]
                box_h = g_box[3] - g_box[1]
                false_negatives.append({
                    "image": img_path.name,
                    "gt_class": DEFECT_CLASSES[g_cls],
                    "box_size_px": [round(box_w, 1), round(box_h, 1)],
                    "is_small_micro": bool(box_w < 20 or box_h < 20),
                    "box": g_box,
                })

        # Save visual comparison image if image contains FN or FP
        has_error = (len(gt_boxes) > len(matched_gt)) or (len(pred_boxes) > len(matched_pred))
        if has_error and len(list(output_dir.glob("*.jpg"))) < max_saved_cases:
            vis_img = img_bgr.copy()
            # Draw GT in GREEN
            for g_idx, (g_box, g_cls) in enumerate(zip(gt_boxes, gt_classes)):
                x1, y1, x2, y2 = map(int, g_box)
                color = (0, 255, 0) if g_idx in matched_gt else (255, 100, 0)  # Cyan if missed
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                lbl_text = f"GT: {DEFECT_CLASSES[g_cls]}"
                cv2.putText(vis_img, lbl_text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Draw Preds in RED/ORANGE
            for p_idx, (p_box, p_cls, p_sc) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
                x1, y1, x2, y2 = map(int, p_box)
                color = (0, 200, 255) if p_idx in matched_pred else (0, 0, 255)  # Red if FP
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                p_text = f"Pred: {DEFECT_CLASSES[p_cls]} {p_sc:.2f}"
                cv2.putText(vis_img, p_text, (x1, min(h - 5, y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            vis_out_path = output_dir / f"error_{img_path.name}"
            cv2.imwrite(str(vis_out_path), vis_img)

    total_fn = len(false_negatives)
    total_fp = len(false_positives)
    micro_fn_count = sum(1 for fn in false_negatives if fn.get("is_small_micro"))

    summary = {
        "total_test_images": len(image_files),
        "total_false_negatives_escapes": total_fn,
        "total_false_positives_overkill": total_fp,
        "escapes_due_to_micro_size_pct": round(micro_fn_count / total_fn * 100, 1) if total_fn > 0 else 0.0,
        "confusion_matrix": conf_matrix.tolist(),
        "classes": DEFECT_CLASSES + ["Background_or_Missed"],
        "top_false_negatives": false_negatives[:10],
        "top_false_positives": false_positives[:10],
    }

    report_path = Path("docs/error_analysis_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Error analysis complete: {total_fn} escapes, {total_fp} false alarms. Report saved to {report_path}")
    return summary


if __name__ == "__main__":
    model = Path("models/yolov8_advanced.pt")
    if not model.exists():
        model = Path("models/yolov8n_baseline.pt")
    run_error_analysis(model)
