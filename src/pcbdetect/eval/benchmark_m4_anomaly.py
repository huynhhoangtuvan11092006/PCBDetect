"""Milestone 4 Benchmark & Evaluation Module for Unsupervised Anomaly Detection (PaDiM).

Evaluates:
- Image-level AUROC (separating defect-free Golden Templates from defective Tested PCBs).
- Pixel-level AUROC (defect localization against Ground Truth binary masks).
- Latency and Throughput (FPS on CPU).
- Exports visual anomaly heatmaps and comparison artifacts into docs/figures/anomaly/.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import mlflow
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from pcbdetect.data.split import create_group_disjoint_split
from pcbdetect.models.padim import PaDiM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_ground_truth_mask(
    boxes: list[tuple[float, float, float, float]],
    shape: tuple[int, int],
) -> np.ndarray:
    """Create binary mask (H, W) where defect pixels = 1, normal = 0."""
    mask = np.zeros(shape, dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)
    return mask


def run_m4_benchmark(
    model_path: Path = Path("models/padim_resnet18.pt"),
    output_fig_dir: Path = Path("docs/figures/anomaly"),
    output_metrics_json: Path = Path("docs/m4_anomaly_metrics.json"),
    img_size: int = 256,
    num_eval_samples: int = 100,
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute M4 Anomaly Detection Benchmark."""
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading PaDiM model from {model_path}...")
    padim = PaDiM(d_reduced=100, device=device)
    padim.load(model_path)

    splits = create_group_disjoint_split()
    test_samples = splits.test

    if num_eval_samples and len(test_samples) > num_eval_samples:
        test_samples = test_samples[:num_eval_samples]

    logger.info(f"Evaluating Anomaly Detection on {len(test_samples)} pairs (Normal vs Defective)...")

    image_labels: list[int] = []
    image_scores: list[float] = []

    pixel_labels_flat: list[np.ndarray] = []
    pixel_scores_flat: list[np.ndarray] = []

    latencies_ms: list[float] = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    saved_figures_count = 0

    for idx, sample in enumerate(test_samples):
        # 1. Evaluate Normal Golden Template (Label = 0)
        temp_bgr = cv2.imread(str(sample.template_image_path))
        if temp_bgr is not None:
            temp_rgb = cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2RGB)
            temp_resized = cv2.resize(temp_rgb, (img_size, img_size))
            temp_tensor = torch.from_numpy(temp_resized).permute(2, 0, 1).float() / 255.0
            temp_tensor = ((temp_tensor - mean) / std).unsqueeze(0)

            t0 = time.time()
            score_norm, hm_norm = padim.predict(temp_tensor)
            latencies_ms.append((time.time() - t0) * 1000.0)

            image_labels.append(0)
            image_scores.append(float(score_norm[0]))

        # 2. Evaluate Defective Tested PCB (Label = 1)
        test_bgr = cv2.imread(str(sample.test_image_path))
        if test_bgr is not None:
            test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)
            test_resized = cv2.resize(test_rgb, (img_size, img_size))
            test_tensor = torch.from_numpy(test_resized).permute(2, 0, 1).float() / 255.0
            test_tensor = ((test_tensor - mean) / std).unsqueeze(0)

            score_anom, hm_anom = padim.predict(test_tensor)
            image_labels.append(1)
            image_scores.append(float(score_anom[0]))

            # Ground truth defect mask
            h_orig, w_orig = test_bgr.shape[:2]
            scaled_boxes = []
            for b in sample.boxes:
                # Scale coordinates to img_size
                sx = img_size / w_orig
                sy = img_size / h_orig
                scaled_boxes.append((b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy))

            gt_mask = create_ground_truth_mask(scaled_boxes, (img_size, img_size))

            # Collect pixel scores (subsample every 2 pixels to keep memory small)
            pixel_labels_flat.append(gt_mask[::2, ::2].flatten())
            pixel_scores_flat.append(hm_anom[0, ::2, ::2].flatten())

            # Save qualitative visual artifact
            if saved_figures_count < 6 and temp_bgr is not None:
                # Generate JET colormap overlay
                hm = hm_anom[0]
                hm_norm_vis = cv2.normalize(hm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                heatmap_color = cv2.applyColorMap(hm_norm_vis, cv2.COLORMAP_JET)

                overlay = cv2.addWeighted(test_resized, 0.6, cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), 0.4, 0)

                # Draw GT boxes on overlay
                for box in scaled_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Difference image with template
                diff = cv2.absdiff(temp_resized, test_resized)
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
                _, diff_thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
                diff_vis = cv2.cvtColor(diff_thresh, cv2.COLOR_GRAY2RGB)

                # Horizontal collage: [Template, Test PCB, Difference, Anomaly Heatmap Overlay]
                row = np.hstack([temp_resized, test_resized, diff_vis, overlay])
                fig_path = output_fig_dir / f"anomaly_{sample.test_image_path.stem}.jpg"
                cv2.imwrite(str(fig_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
                saved_figures_count += 1

    # Compute Image-level AUROC
    image_auroc = float(roc_auc_score(image_labels, image_scores))

    # Compute Pixel-level AUROC
    all_pix_labels = np.concatenate(pixel_labels_flat)
    all_pix_scores = np.concatenate(pixel_scores_flat)
    pixel_auroc = float(roc_auc_score(all_pix_labels, all_pix_scores))

    avg_latency = float(np.mean(latencies_ms))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    model_size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0.0

    metrics = {
        "model_name": "PaDiM (ResNet-18)",
        "backbone": "ResNet-18",
        "image_level_auroc": round(image_auroc, 4),
        "pixel_level_auroc": round(pixel_auroc, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "throughput_fps": round(fps, 1),
        "model_size_mb": round(model_size_mb, 2),
        "num_normal_test_evaluated": sum(1 for y in image_labels if y == 0),
        "num_anomalous_test_evaluated": sum(1 for y in image_labels if y == 1),
        "saved_heatmaps_dir": str(output_fig_dir),
    }

    with open(output_metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"M4 Benchmark Results: Image AUROC={image_auroc:.4f}, Pixel AUROC={pixel_auroc:.4f}, Latency={avg_latency:.2f}ms ({fps:.1f} FPS)")

    mlflow.set_experiment("PCBDetect-Anomaly-PaDiM")
    with mlflow.start_run(run_name="padim_benchmark_evaluation"):
        mlflow.log_metrics({
            "image_auroc": image_auroc,
            "pixel_auroc": pixel_auroc,
            "avg_latency_ms": avg_latency,
            "throughput_fps": fps,
        })
        mlflow.log_artifact(str(output_metrics_json))

    return metrics


if __name__ == "__main__":
    run_m4_benchmark()
