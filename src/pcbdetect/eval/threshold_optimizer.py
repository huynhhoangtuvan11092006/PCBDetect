"""Precision-Recall Curve Analysis and Per-Class Operating Threshold Optimization.

Finds operating thresholds satisfying industrial factory SLAs:
- High-Recall Regime: Target Recall >= 95% to minimize Escape Defects.
- Per-Class Dynamic Thresholding vs. Global Thresholding.
- Generates publication-ready PR curves and tradeoff figures.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from pcbdetect.constants import DEFECT_CLASSES, DEFECT_COLORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute Intersection-over-Union for boxes [x1, y1, x2, y2]."""
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


@dataclass
class OperatingPoint:
    class_name: str
    target_recall: float
    selected_threshold: float
    achieved_recall: float
    achieved_precision: float
    achieved_f1: float
    tp: int
    fp: int
    fn: int


class ThresholdOptimizer:
    """Evaluates raw model predictions across confidence thresholds to find optimal operating points."""

    def __init__(
        self,
        model_path: Path = Path("models/yolov8_advanced.pt"),
        test_images_dir: Path = Path("data/processed/yolo/images/test"),
        test_labels_dir: Path = Path("data/processed/yolo/labels/test"),
        iou_thresh: float = 0.45,
    ):
        self.model_path = model_path
        self.test_images_dir = test_images_dir
        self.test_labels_dir = test_labels_dir
        self.iou_thresh = iou_thresh

        logger.info(f"Loading YOLO model for threshold optimization: {model_path}")
        self.model = YOLO(str(model_path))

    def collect_raw_predictions(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Collect all ground truths and raw predictions at low conf threshold (0.01)."""
        image_files = sorted(list(self.test_images_dir.glob("*.jpg")) + list(self.test_images_dir.glob("*.png")))
        logger.info(f"Running inference on {len(image_files)} test images at conf=0.01...")

        all_gt: list[dict[str, Any]] = []
        all_preds: list[dict[str, Any]] = []

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Ground truth
            lbl_path = self.test_labels_dir / f"{img_path.stem}.txt"
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
                            all_gt.append({
                                "image": img_path.name,
                                "class_id": cls_id,
                                "class_name": DEFECT_CLASSES[cls_id],
                                "box": [x1, y1, x2, y2],
                            })

            # Predictions at very low conf to capture full spectrum
            results = self.model.predict(source=str(img_path), conf=0.01, verbose=False)
            if len(results) > 0 and results[0].boxes is not None:
                boxes_data = results[0].boxes.data.cpu().numpy()
                for row in boxes_data:
                    x1, y1, x2, y2, sc, cls_id = row[:6]
                    all_preds.append({
                        "image": img_path.name,
                        "class_id": int(cls_id),
                        "class_name": DEFECT_CLASSES[int(cls_id)],
                        "confidence": float(sc),
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                    })

        logger.info(f"Collected {len(all_gt)} Ground Truth boxes and {len(all_preds)} raw predictions.")
        return all_gt, all_preds

    def evaluate_at_thresholds(
        self,
        all_gt: list[dict[str, Any]],
        all_preds: list[dict[str, Any]],
        thresholds: np.ndarray,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Evaluate Precision and Recall across a range of thresholds for each class."""
        classes = DEFECT_CLASSES + ["ALL"]
        results: dict[str, dict[str, list[float]]] = {
            c: {"thresholds": [], "precision": [], "recall": [], "f1": [], "tp": [], "fp": [], "fn": []}
            for c in classes
        }

        # Index ground truths by (image, class_id)
        gt_by_img: dict[str, list[dict[str, Any]]] = {}
        for g in all_gt:
            img = g["image"]
            if img not in gt_by_img:
                gt_by_img[img] = []
            gt_by_img[img].append(g)

        for conf in thresholds:
            filtered_preds = [p for p in all_preds if p["confidence"] >= conf]
            preds_by_img: dict[str, list[dict[str, Any]]] = {}
            for p in filtered_preds:
                img = p["image"]
                if img not in preds_by_img:
                    preds_by_img[img] = []
                preds_by_img[img].append(p)

            # Per class counters
            tp_counts = {c: 0 for c in DEFECT_CLASSES}
            fp_counts = {c: 0 for c in DEFECT_CLASSES}
            fn_counts = {c: 0 for c in DEFECT_CLASSES}

            all_images = set(list(gt_by_img.keys()) + list(preds_by_img.keys()))

            for img in all_images:
                img_gts = gt_by_img.get(img, [])
                img_preds = preds_by_img.get(img, [])

                matched_gt = set()
                # Sort preds by score descending
                sorted_preds = sorted(img_preds, key=lambda x: -x["confidence"])

                for p in sorted_preds:
                    p_cls = p["class_name"]
                    p_box = p["box"]
                    best_iou = 0.0
                    best_g_idx = -1

                    for g_idx, g in enumerate(img_gts):
                        if g_idx in matched_gt or g["class_name"] != p_cls:
                            continue
                        iou = compute_iou(p_box, g["box"])
                        if iou > best_iou:
                            best_iou = iou
                            best_g_idx = g_idx

                    if best_iou >= self.iou_thresh and best_g_idx >= 0:
                        matched_gt.add(best_g_idx)
                        tp_counts[p_cls] += 1
                    else:
                        fp_counts[p_cls] += 1

                for g_idx, g in enumerate(img_gts):
                    if g_idx not in matched_gt:
                        fn_counts[g["class_name"]] += 1

            # Record per-class
            total_tp, total_fp, total_fn = 0, 0, 0
            for c in DEFECT_CLASSES:
                c_tp = tp_counts[c]
                c_fp = fp_counts[c]
                c_fn = fn_counts[c]
                total_tp += c_tp
                total_fp += c_fp
                total_fn += c_fn

                prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 1.0
                rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

                results[c]["thresholds"].append(conf)
                results[c]["precision"].append(prec)
                results[c]["recall"].append(rec)
                results[c]["f1"].append(f1)
                results[c]["tp"].append(c_tp)
                results[c]["fp"].append(c_fp)
                results[c]["fn"].append(c_fn)

            # Record ALL
            all_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
            all_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            all_f1 = (2 * all_prec * all_rec) / (all_prec + all_rec) if (all_prec + all_rec) > 0 else 0.0

            results["ALL"]["thresholds"].append(conf)
            results["ALL"]["precision"].append(all_prec)
            results["ALL"]["recall"].append(all_rec)
            results["ALL"]["f1"].append(all_f1)
            results["ALL"]["tp"].append(total_tp)
            results["ALL"]["fp"].append(total_fp)
            results["ALL"]["fn"].append(total_fn)

        # Convert lists to numpy arrays
        converted = {}
        for c, data in results.items():
            converted[c] = {k: np.array(v) for k, v in data.items()}
        return converted

    def find_operating_points(
        self,
        curve_data: dict[str, dict[str, np.ndarray]],
        target_recall: float = 0.95,
    ) -> dict[str, OperatingPoint]:
        """Find the threshold for each class that satisfies target recall with highest precision."""
        operating_points: dict[str, OperatingPoint] = {}

        classes_to_process = [c for c in DEFECT_CLASSES + ["ALL"] if c in curve_data]
        for c_name in classes_to_process:
            recalls = curve_data[c_name]["recall"]
            precisions = curve_data[c_name]["precision"]
            f1s = curve_data[c_name]["f1"]
            thresholds = curve_data[c_name]["thresholds"]
            tps = curve_data[c_name]["tp"]
            fps = curve_data[c_name]["fp"]
            fns = curve_data[c_name]["fn"]

            # Filter indices where recall >= target_recall
            valid_indices = np.where(recalls >= target_recall)[0]

            if len(valid_indices) > 0:
                # Among valid indices, pick the one with highest precision (highest threshold)
                best_idx = valid_indices[np.argmax(precisions[valid_indices])]
            else:
                # If target recall not strictly reachable, select point with highest recall
                best_idx = int(np.argmax(recalls))

            op = OperatingPoint(
                class_name=c_name,
                target_recall=target_recall,
                selected_threshold=float(thresholds[best_idx]),
                achieved_recall=float(recalls[best_idx]),
                achieved_precision=float(precisions[best_idx]),
                achieved_f1=float(f1s[best_idx]),
                tp=int(tps[best_idx]),
                fp=int(fps[best_idx]),
                fn=int(fns[best_idx]),
            )
            operating_points[c_name] = op

        return operating_points

    def plot_pr_curves(
        self,
        curve_data: dict[str, dict[str, np.ndarray]],
        operating_points: dict[str, OperatingPoint],
        output_path: Path,
    ) -> None:
        """Plot high-res multi-class PR curves with operating point markers."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 8), dpi=200)

        for c in DEFECT_CLASSES:
            rec = curve_data[c]["recall"]
            prec = curve_data[c]["precision"]
            color = DEFECT_COLORS.get(c, "#333333")
            plt.plot(rec, prec, label=f"{c} (F1 max: {np.max(curve_data[c]['f1']):.2f})", color=color, linewidth=2.0)

            op = operating_points[c]
            plt.scatter(
                [op.achieved_recall],
                [op.achieved_precision],
                color=color,
                edgecolor="black",
                s=80,
                zorder=5,
            )

        # Plot overall curve
        all_rec = curve_data["ALL"]["recall"]
        all_prec = curve_data["ALL"]["precision"]
        plt.plot(all_rec, all_prec, label="ALL (Mean)", color="#111111", linewidth=2.8, linestyle="--")

        # Mark Target Recall line
        plt.axvline(x=0.95, color="#E63946", linestyle=":", linewidth=1.5, label="Factory SLA Recall Target (95%)")

        plt.title("PCBDetect Precision-Recall Curves & Production Operating Points", fontsize=14, fontweight="bold")
        plt.xlabel("Recall (Coverage - Sensitivity)", fontsize=12)
        plt.ylabel("Precision (Accuracy - Positive Predictive Value)", fontsize=12)
        plt.xlim([0.0, 1.02])
        plt.ylim([0.0, 1.02])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower left", fontsize=10, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(str(output_path))
        plt.close()
        logger.info(f"Saved PR curve plot to {output_path}")

    def plot_tradeoff_curve(
        self,
        curve_data: dict[str, dict[str, np.ndarray]],
        operating_points: dict[str, OperatingPoint],
        output_path: Path,
    ) -> None:
        """Plot Confidence Threshold vs. Recall vs. False Alarm Rate (FAR)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        thresholds = curve_data["ALL"]["thresholds"]
        recall = curve_data["ALL"]["recall"]
        precision = curve_data["ALL"]["precision"]
        fps = curve_data["ALL"]["fp"]
        tps = curve_data["ALL"]["tp"]

        # False alarm rate proxy: FP / (FP + TP) = 1 - Precision
        far = 1.0 - precision

        plt.figure(figsize=(10, 6), dpi=200)
        plt.plot(thresholds, recall * 100, label="Recall (% - Escape Reduction)", color="#2A9D8F", linewidth=2.5)
        plt.plot(thresholds, precision * 100, label="Precision (% - Overkill Reduction)", color="#457B9D", linewidth=2.5)
        plt.plot(thresholds, far * 100, label="False Alarm Rate (% - Re-inspection Burden)", color="#E63946", linewidth=2.0, linestyle=":")

        opt = operating_points["ALL"]
        plt.axvline(x=opt.selected_threshold, color="#9B5DE5", linestyle="--", linewidth=1.8, label=f"Global Opt Threshold = {opt.selected_threshold:.2f}")

        plt.title("Operating Tradeoff: Confidence vs. Recall vs. False Alarm Rate", fontsize=13, fontweight="bold")
        plt.xlabel("Confidence Threshold", fontsize=11)
        plt.ylabel("Percentage (%)", fontsize=11)
        plt.ylim([0, 105])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="center right", fontsize=10, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(str(output_path))
        plt.close()
        logger.info(f"Saved Tradeoff plot to {output_path}")


def run_threshold_optimization(
    output_config_path: Path = Path("configs/operating_thresholds.json"),
    output_fig_dir: Path = Path("docs/figures/operating_points"),
) -> dict[str, Any]:
    """Execute complete Milestone 5 threshold optimization workflow."""
    optimizer = ThresholdOptimizer()
    all_gt, all_preds = optimizer.collect_raw_predictions()

    thresholds = np.linspace(0.02, 0.95, 94)
    logger.info("Computing PR metrics across 94 threshold points...")
    curve_data = optimizer.evaluate_at_thresholds(all_gt, all_preds, thresholds)

    operating_points = optimizer.find_operating_points(curve_data, target_recall=0.95)

    # Save plots
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    pr_fig = output_fig_dir / "pr_curves_per_class.png"
    tradeoff_fig = output_fig_dir / "operating_tradeoff_far.png"
    optimizer.plot_pr_curves(curve_data, operating_points, pr_fig)
    optimizer.plot_tradeoff_curve(curve_data, operating_points, tradeoff_fig)

    # Export configuration JSON
    config_data = {
        "strategy": "per_class_dynamic_thresholding",
        "factory_target_recall": 0.95,
        "global_threshold": operating_points["ALL"].selected_threshold,
        "per_class_thresholds": {
            c: {
                "threshold": operating_points[c].selected_threshold,
                "achieved_recall": round(operating_points[c].achieved_recall, 4),
                "achieved_precision": round(operating_points[c].achieved_precision, 4),
                "achieved_f1": round(operating_points[c].achieved_f1, 4),
            }
            for c in DEFECT_CLASSES
        },
        "operating_summary": {k: asdict(v) for k, v in operating_points.items()},
    }

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    logger.info(f"Operating thresholds successfully exported to {output_config_path}")
    return config_data


if __name__ == "__main__":
    run_threshold_optimization()
