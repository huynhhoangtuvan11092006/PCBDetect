"""Classical OpenCV baseline detector for PCB defect inspection.

Uses template difference, binarization, morphological filtering, and contour extraction.
Provides benchmark metrics (Precision, Recall, F1, Latency) following the DeepPCB standard (IoU >= 0.33).
"""

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from pcbdetect.data.align import compute_difference_image
from pcbdetect.data.dataset import DeepPCBSample


@dataclass
class ClassicalDetection:
    """Represents a predicted bounding box from classical template matching."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    area: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two boxes (x1, y1, x2, y2)."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


class ClassicalDetector:
    """Detects PCB defects by subtracting golden template and extracting defect contours."""

    def __init__(
        self,
        diff_threshold: int = 40,
        min_area: int = 15,
        max_area: int = 50000,
        morph_kernel_size: int = 3,
        align_first: bool = False,
    ):
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.morph_kernel_size = morph_kernel_size
        self.align_first = align_first

    def predict(
        self,
        test_img: np.ndarray,
        template_img: np.ndarray,
    ) -> tuple[list[ClassicalDetection], np.ndarray]:
        """Detect defect candidates.

        Returns:
            Tuple of (list of ClassicalDetection, clean_binary_mask).
        """
        _, mask = compute_difference_image(
            test_img=test_img,
            template_img=template_img,
            align_first=self.align_first,
            diff_threshold=self.diff_threshold,
            morph_kernel_size=self.morph_kernel_size,
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[ClassicalDetection] = []
        img_h, img_w = test_img.shape[:2]

        for cnt in contours:
            c_area = cv2.contourArea(cnt)
            if c_area < self.min_area or c_area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Pad bounding box slightly to capture subtle trace borders
            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_w, x + w + pad)
            y2 = min(img_h, y + h + pad)

            # Intensity-based confidence: mean pixel intensity inside mask defect region
            crop_mask = mask[y : y + h, x : x + w]
            score = float(np.mean(crop_mask) / 255.0) if crop_mask.size > 0 else 0.5

            detections.append(
                ClassicalDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=score,
                    area=int(c_area),
                )
            )

        return detections, mask


def evaluate_classical_detector(
    samples: list[DeepPCBSample],
    iou_threshold: float = 0.33,  # DeepPCB benchmark standard
    diff_threshold: int = 40,
    min_area: int = 15,
) -> dict[str, Any]:
    """Run full evaluation of Classical Baseline on given dataset samples."""
    detector = ClassicalDetector(diff_threshold=diff_threshold, min_area=min_area)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    class_tp: dict[str, int] = {}
    class_total: dict[str, int] = {}

    total_inference_time = 0.0

    for sample in samples:
        test_img = sample.load_test_image()
        temp_img = sample.load_template_image()

        t0 = time.perf_counter()
        detections, _ = detector.predict(test_img, temp_img)
        t1 = time.perf_counter()
        total_inference_time += t1 - t0

        # Match predictions to GT boxes
        gt_boxes = sample.boxes
        matched_gt_indices = set()
        matched_det_indices = set()

        # Count total GT per class
        for gt in gt_boxes:
            class_total[gt.class_name] = class_total.get(gt.class_name, 0) + 1

        for d_idx, det in enumerate(detections):
            det_box = (det.x1, det.y1, det.x2, det.y2)
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, gt in enumerate(gt_boxes):
                if g_idx in matched_gt_indices:
                    continue
                iou = compute_iou(det_box, (gt.x1, gt.y1, gt.x2, gt.y2))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gt_indices.add(best_gt_idx)
                matched_det_indices.add(d_idx)
                gt_cls = gt_boxes[best_gt_idx].class_name
                class_tp[gt_cls] = class_tp.get(gt_cls, 0) + 1

        tp = len(matched_gt_indices)
        fp = len(detections) - len(matched_det_indices)
        fn = len(gt_boxes) - len(matched_gt_indices)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_latency_ms = (total_inference_time / len(samples)) * 1000.0 if samples else 0.0

    class_recalls = {}
    for cls_name, total in class_total.items():
        tp = class_tp.get(cls_name, 0)
        class_recalls[cls_name] = round(tp / total * 100.0, 2) if total > 0 else 0.0

    return {
        "model_name": "Classical OpenCV Baseline (Template Diff)",
        "num_test_samples": len(samples),
        "total_ground_truth_defects": total_tp + total_fn,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision * 100.0, 2),
        "recall": round(recall * 100.0, 2),
        "f1_score": round(f1 * 100.0, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "throughput_fps": round(1000.0 / avg_latency_ms, 1) if avg_latency_ms > 0 else 0.0,
        "class_recalls_pct": class_recalls,
    }
