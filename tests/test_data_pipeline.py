"""Unit tests for PCBDetect data loading and parsing pipeline."""

from pathlib import Path

import pytest

from pcbdetect.constants import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFECT_ID_TO_NAME,
    RAW_DATA_DIR,
    YOLO_ID_TO_NAME,
)
from pcbdetect.data.dataset import (
    BoundingBox,
    load_deeppcb_dataset,
    parse_annotation_file,
)


def test_constants_mapping():
    """Verify class ID mappings for DeepPCB and YOLO."""
    assert len(DEFECT_ID_TO_NAME) == 6
    assert DEFECT_ID_TO_NAME[1] == "open"
    assert DEFECT_ID_TO_NAME[2] == "short"
    assert DEFECT_ID_TO_NAME[3] == "mousebite"
    assert DEFECT_ID_TO_NAME[4] == "spur"
    assert DEFECT_ID_TO_NAME[5] == "copper"
    assert DEFECT_ID_TO_NAME[6] == "pin_hole"

    # Verify YOLO 0-indexed mapping
    assert len(YOLO_ID_TO_NAME) == 6
    for i in range(6):
        assert YOLO_ID_TO_NAME[i] == DEFECT_ID_TO_NAME[i + 1]


def test_bounding_box_properties():
    """Test geometric calculations on BoundingBox."""
    box = BoundingBox(x1=100, y1=150, x2=140, y2=200, class_id=1, class_name="open")
    assert box.width == 40
    assert box.height == 50
    assert box.area == 2000
    assert box.aspect_ratio == 40 / 50
    assert box.center == (120.0, 175.0)

    # Test YOLO normalization
    cls_idx, x_c, y_c, w_norm, h_norm = box.to_yolo_format(640, 640)
    assert cls_idx == 0
    assert pytest.approx(x_c, 1e-4) == 120.0 / 640.0
    assert pytest.approx(y_c, 1e-4) == 175.0 / 640.0
    assert pytest.approx(w_norm, 1e-4) == 40.0 / 640.0
    assert pytest.approx(h_norm, 1e-4) == 50.0 / 640.0


def test_parse_annotation_file(tmp_path: Path):
    """Test parsing standard space-separated DeepPCB annotation text."""
    annot_file = tmp_path / "sample.txt"
    annot_file.write_text("100 200 150 240 1\n300 400 320 450 2\n", encoding="utf-8")

    boxes = parse_annotation_file(annot_file)
    assert len(boxes) == 2

    assert boxes[0].class_id == 1
    assert boxes[0].class_name == "open"
    assert boxes[0].x1 == 100
    assert boxes[0].y1 == 200
    assert boxes[0].x2 == 150
    assert boxes[0].y2 == 240

    assert boxes[1].class_id == 2
    assert boxes[1].class_name == "short"


@pytest.mark.skipif(not RAW_DATA_DIR.exists(), reason="DeepPCB raw dataset not present")
def test_load_deeppcb_dataset_splits():
    """Test loading actual DeepPCB dataset from disk."""
    trainval_samples = load_deeppcb_dataset(split="trainval")
    test_samples = load_deeppcb_dataset(split="test")

    assert len(trainval_samples) == 1000
    assert len(test_samples) == 500

    # Validate first sample
    s0 = trainval_samples[0]
    assert s0.test_image_path.exists(), f"Image missing: {s0.test_image_path}"
    assert s0.template_image_path.exists(), f"Template missing: {s0.template_image_path}"
    assert s0.annotation_path.exists(), f"Annotation missing: {s0.annotation_path}"
    assert s0.num_defects > 0

    # Verify bounding boxes are within image boundaries
    for box in s0.boxes:
        assert 0 <= box.x1 <= DEFAULT_IMAGE_WIDTH
        assert 0 <= box.x2 <= DEFAULT_IMAGE_WIDTH
        assert 0 <= box.y1 <= DEFAULT_IMAGE_HEIGHT
        assert 0 <= box.y2 <= DEFAULT_IMAGE_HEIGHT
        assert box.width > 0
        assert box.height > 0
