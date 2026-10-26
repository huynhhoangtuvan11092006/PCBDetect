"""Unit tests for Milestone 1 (M1) Data Pipeline:

- Standard and Group-Disjoint Splits
- Template Alignment & Difference Imaging
- Tiling & Sliding Window Coordinate Translation
- PCB Data Augmentations
- YOLO and COCO Export Schema Validation
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pcbdetect.constants import RAW_DATA_DIR
from pcbdetect.data.align import align_images_orb, compute_difference_image
from pcbdetect.data.augment import flip_horizontal, flip_vertical, rotate_90, rotate_180
from pcbdetect.data.convert import export_coco_dataset, export_yolo_dataset
from pcbdetect.data.dataset import BoundingBox, DeepPCBSample
from pcbdetect.data.split import (
    DatasetSplits,
    create_group_disjoint_split,
    create_standard_split,
)
from pcbdetect.data.tiling import remap_predictions_to_global, slice_image_with_boxes


@pytest.mark.skipif(not RAW_DATA_DIR.exists(), reason="DeepPCB raw dataset not present")
def test_split_standard_strategy():
    """Verify standard benchmark split (1000 trainval -> 800 train, 200 val; 500 test)."""
    splits = create_standard_split(val_ratio=0.2, random_seed=42)
    assert len(splits.train) == 800
    assert len(splits.val) == 200
    assert len(splits.test) == 500
    assert splits.total_count == 1500
    assert splits.mode == "standard"


@pytest.mark.skipif(not RAW_DATA_DIR.exists(), reason="DeepPCB raw dataset not present")
def test_split_group_disjoint_zero_leakage():
    """Verify strict board-disjoint split guarantees 100% zero group leakage."""
    splits = create_group_disjoint_split()
    summary = splits.summary()

    train_groups = set(summary["train_groups"])
    val_groups = set(summary["val_groups"])
    test_groups = set(summary["test_groups"])

    # Disjointness checks
    assert len(train_groups.intersection(test_groups)) == 0, "Data leakage between train and test!"
    assert len(train_groups.intersection(val_groups)) == 0, "Data leakage between train and val!"
    assert len(val_groups.intersection(test_groups)) == 0, "Data leakage between val and test!"

    # Total samples must sum to all 1500 images
    assert splits.total_count == 1500
    assert len(splits.train) > 900
    assert len(splits.test) > 200


def test_align_and_difference_computation():
    """Test template alignment and classical difference generation."""
    # Synthetic PCB image: white canvas with black traces
    template = np.full((100, 100, 3), 255, dtype=np.uint8)
    template[40:60, :] = 0  # Black trace across center

    test_img = template.copy()
    # Introduce a short-circuit defect bridging top and bottom
    test_img[30:70, 45:55] = 0

    aligned, _, _success = align_images_orb(test_img, template)
    assert aligned.shape == template.shape

    enhanced_diff, mask = compute_difference_image(
        test_img, template, align_first=False, diff_threshold=30
    )
    assert enhanced_diff.shape == template.shape
    assert mask.shape == (100, 100)

    # Defect region must be highlighted in binary mask
    assert np.any(mask > 0)


def test_tiling_coordinate_translation():
    """Test slicing an image and remapping bounding boxes to local tile coordinates."""
    large_img = np.zeros((1200, 1200, 3), dtype=np.uint8)
    boxes = [
        BoundingBox(x1=100, y1=100, x2=150, y2=150, class_id=1, class_name="open"),
        BoundingBox(x1=700, y1=700, x2=750, y2=750, class_id=2, class_name="short"),
    ]

    tiles = slice_image_with_boxes(large_img, boxes, tile_size=640, overlap_ratio=0.2)
    assert len(tiles) > 1

    # Check tile 0 (starts at 0, 0) contains the first box
    tile_0 = tiles[0]
    assert tile_0.x_offset == 0 and tile_0.y_offset == 0
    assert len(tile_0.boxes) == 1
    assert tile_0.boxes[0].x1 == 100
    assert tile_0.boxes[0].y1 == 100
    assert tile_0.boxes[0].class_name == "open"

    # Test global coordinate remapping
    local_preds = [(100, 100, 150, 150, 0.95, 0)]
    remapped = remap_predictions_to_global(local_preds, x_offset=500, y_offset=500)
    assert remapped[0][:4] == (600, 600, 650, 650)


def test_pcb_augmentations():
    """Verify that geometric augmentations analytically transform bounding boxes within bounds."""
    w, h = 640, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    boxes = [BoundingBox(x1=100, y1=120, x2=160, y2=180, class_id=1, class_name="open")]

    # 1. Flip horizontal: new x1 = 640 - 160 = 480, new x2 = 640 - 100 = 540
    _, flip_h_boxes = flip_horizontal(img, boxes)
    assert flip_h_boxes[0].x1 == 480
    assert flip_h_boxes[0].x2 == 540
    assert flip_h_boxes[0].width == 60

    # 2. Flip vertical: new y1 = 640 - 180 = 460, new y2 = 640 - 120 = 520
    _, flip_v_boxes = flip_vertical(img, boxes)
    assert flip_v_boxes[0].y1 == 460
    assert flip_v_boxes[0].y2 == 520
    assert flip_v_boxes[0].height == 60

    # 3. Rotate 90: width and height swap
    _, rot90_boxes = rotate_90(img, boxes)
    assert rot90_boxes[0].width == 60
    assert rot90_boxes[0].height == 60
    assert 0 <= rot90_boxes[0].x1 < rot90_boxes[0].x2 <= w
    assert 0 <= rot90_boxes[0].y1 < rot90_boxes[0].y2 <= h

    # 4. Rotate 180
    _, rot180_boxes = rotate_180(img, boxes)
    assert rot180_boxes[0].width == 60
    assert rot180_boxes[0].height == 60


def test_yolo_and_coco_export(tmp_path: Path):
    """Test exporting YOLO and COCO formats on mock samples and validate schemas."""
    # Create mock sample
    mock_test_img = tmp_path / "mock_test.jpg"
    mock_test_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal jpeg-like header

    mock_sample = DeepPCBSample(
        sample_id="00041000",
        group_id="group00041",
        board_id="00041",
        test_image_path=mock_test_img,
        template_image_path=mock_test_img,
        annotation_path=tmp_path / "mock.txt",
        split="train",
        boxes=[
            BoundingBox(x1=100, y1=150, x2=200, y2=250, class_id=1, class_name="open"),
            BoundingBox(x1=300, y1=350, x2=380, y2=420, class_id=2, class_name="short"),
        ],
    )

    splits = DatasetSplits(
        train=[mock_sample],
        val=[mock_sample],
        test=[mock_sample],
        mode="test_split",
    )

    out_dir = tmp_path / "processed"

    # 1. Test YOLO Export
    yaml_path = export_yolo_dataset(splits, out_dir)
    assert yaml_path.exists()

    yolo_label_file = out_dir / "yolo" / "labels" / "train" / "group00041_00041000_test.txt"
    assert yolo_label_file.exists()

    with open(yolo_label_file, "r") as f:
        lines = f.readlines()
    assert len(lines) == 2

    for line in lines:
        parts = line.strip().split()
        assert len(parts) == 5
        cls_idx = int(parts[0])
        assert 0 <= cls_idx <= 5
        x_c, y_c, w_norm, h_norm = map(float, parts[1:])
        assert 0.0 <= x_c <= 1.0
        assert 0.0 <= y_c <= 1.0
        assert 0.0 < w_norm <= 1.0
        assert 0.0 < h_norm <= 1.0

    # 2. Test COCO Export
    coco_paths = export_coco_dataset(splits, out_dir)
    train_json = coco_paths["train"]
    assert train_json.exists()

    with open(train_json, "r") as f:
        coco_data = json.load(f)

    assert "images" in coco_data
    assert "annotations" in coco_data
    assert "categories" in coco_data
    assert len(coco_data["images"]) == 1
    assert len(coco_data["annotations"]) == 2
    assert len(coco_data["categories"]) == 6

    # Verify COCO bbox is [x, y, w, h]
    annot_0 = coco_data["annotations"][0]
    assert annot_0["bbox"] == [100, 150, 100, 100]
    assert annot_0["area"] == 10000
    assert annot_0["category_id"] == 1
