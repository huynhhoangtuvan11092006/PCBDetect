"""Unit tests for Milestone 2 (M2) Baselines:

- IoU calculation logic
- ClassicalDetector prediction and contour extraction
- Classical detector evaluation metrics
- YOLOv8 model loading and inference capability
"""

from pathlib import Path

import numpy as np
import pytest

from pcbdetect.constants import RAW_DATA_DIR
from pcbdetect.data.dataset import BoundingBox, DeepPCBSample
from pcbdetect.models.classical_baseline import (
    ClassicalDetector,
    compute_iou,
    evaluate_classical_detector,
)


def test_compute_iou():
    """Verify IoU calculation on known bounding box coordinates."""
    box1 = (0, 0, 100, 100)
    box2 = (0, 0, 100, 100)
    assert compute_iou(box1, box2) == 1.0

    # Disjoint boxes
    box3 = (200, 200, 300, 300)
    assert compute_iou(box1, box3) == 0.0

    # 50% width overlap: (0, 0, 100, 100) and (50, 0, 150, 100)
    # inter_area = 50 * 100 = 5000, union_area = 10000 + 10000 - 5000 = 15000 -> 1/3
    box4 = (50, 0, 150, 100)
    assert pytest.approx(compute_iou(box1, box4), 1e-4) == 5000.0 / 15000.0


def test_classical_detector_on_synthetic_defect():
    """Verify ClassicalDetector detects synthetic short circuit defect."""
    h, w = 200, 200
    template = np.full((h, w, 3), 255, dtype=np.uint8)
    template[90:110, :] = 0  # black horizontal trace

    test_img = template.copy()
    # Add vertical bridging defect (short circuit)
    test_img[60:140, 95:105] = 0

    detector = ClassicalDetector(diff_threshold=30, min_area=20)
    detections, _mask = detector.predict(test_img, template)

    assert len(detections) >= 1
    det = detections[0]
    assert det.width > 0
    assert det.height > 0
    assert det.score > 0.0
    assert 50 <= det.y1 and det.y2 <= 150


def test_classical_evaluator_metrics_schema(tmp_path: Path):
    """Verify evaluation metric dictionary structure."""
    mock_img = tmp_path / "mock.jpg"
    mock_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    # Synthetic sample
    sample = DeepPCBSample(
        sample_id="test001",
        group_id="group001",
        board_id="001",
        test_image_path=mock_img,
        template_image_path=mock_img,
        annotation_path=tmp_path / "annot.txt",
        split="test",
        boxes=[BoundingBox(x1=50, y1=50, x2=100, y2=100, class_id=1, class_name="open")],
    )

    # Monkeypatch image loaders to return valid numpy arrays
    sample.load_test_image = lambda: np.zeros((200, 200, 3), dtype=np.uint8)
    sample.load_template_image = lambda: np.zeros((200, 200, 3), dtype=np.uint8)

    metrics = evaluate_classical_detector([sample], diff_threshold=30, min_area=10)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "avg_latency_ms" in metrics
    assert "class_recalls_pct" in metrics


@pytest.mark.skipif(not RAW_DATA_DIR.exists(), reason="DeepPCB dataset required")
def test_yolo_model_inference():
    """Verify YOLO model loads and runs forward pass on sample image."""
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.predict(dummy_img, device="cpu", verbose=False)
    assert len(results) == 1
    assert hasattr(results[0], "boxes")
