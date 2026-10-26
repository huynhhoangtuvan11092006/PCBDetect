"""Dataset parsing and representation for DeepPCB."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pcbdetect.constants import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFECT_ID_TO_NAME,
    RAW_DATA_DIR,
)


@dataclass
class BoundingBox:
    """Represents an axis-aligned bounding box for a PCB defect."""

    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_yolo_format(
        self,
        img_w: int = DEFAULT_IMAGE_WIDTH,
        img_h: int = DEFAULT_IMAGE_HEIGHT,
    ) -> tuple[int, float, float, float, float]:
        """Convert to YOLO format: (class_idx_0_indexed, x_center_norm, y_center_norm, w_norm, h_norm)."""
        x_center = (self.x1 + self.x2) / (2.0 * img_w)
        y_center = (self.y1 + self.y2) / (2.0 * img_h)
        norm_w = self.width / img_w
        norm_h = self.height / img_h
        yolo_class_idx = self.class_id - 1
        return (yolo_class_idx, x_center, y_center, norm_w, norm_h)


@dataclass
class DeepPCBSample:
    """Represents a paired DeepPCB test image, template image, and defect annotations."""

    sample_id: str
    group_id: str
    board_id: str
    test_image_path: Path
    template_image_path: Path
    annotation_path: Path
    split: str
    boxes: list[BoundingBox]

    @property
    def num_defects(self) -> int:
        return len(self.boxes)

    @property
    def defect_types(self) -> list[str]:
        return [b.class_name for b in self.boxes]

    def load_test_image(self) -> np.ndarray:
        """Load the defective test image (BGR)."""
        img = cv2.imread(str(self.test_image_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image at {self.test_image_path}")
        return img

    def load_template_image(self) -> np.ndarray:
        """Load the golden defect-free template image (BGR)."""
        img = cv2.imread(str(self.template_image_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image at {self.template_image_path}")
        return img

    def render_overlay(self, thickness: int = 2) -> np.ndarray:
        """Render test image with ground truth bounding boxes and class labels."""
        img = self.load_test_image()
        from pcbdetect.constants import DEFECT_RGB

        for box in self.boxes:
            # OpenCV uses BGR order
            rgb = DEFECT_RGB.get(box.class_name, (0, 255, 0))
            bgr = (rgb[2], rgb[1], rgb[0])
            cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), bgr, thickness)
            label = box.class_name
            cv2.putText(
                img,
                label,
                (box.x1, max(12, box.y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                bgr,
                1,
                cv2.LINE_AA,
            )
        return img


def parse_annotation_file(file_path: Path) -> list[BoundingBox]:
    """Parse a single DeepPCB text annotation file.

    Format per line: x1 y1 x2 y2 type
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {file_path}")

    boxes: list[BoundingBox] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                x1, y1, x2, y2, cls_id = map(int, parts[:5])
                # Ensure ordered coordinates
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                cls_name = DEFECT_ID_TO_NAME.get(cls_id, f"unknown_{cls_id}")
                boxes.append(
                    BoundingBox(
                        x1=x_min,
                        y1=y_min,
                        x2=x_max,
                        y2=y_max,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )
            except ValueError:
                continue
    return boxes


def load_deeppcb_dataset(
    data_dir: Path | None = None,
    split: str = "all",
) -> list[DeepPCBSample]:
    """Load and parse DeepPCB dataset splits.

    Args:
        data_dir: Path to PCBData directory. Defaults to RAW_DATA_DIR.
        split: 'trainval', 'test', or 'all'.

    Returns:
        List of DeepPCBSample objects.
    """
    base_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
    if not base_dir.exists():
        raise FileNotFoundError(f"DeepPCB data directory does not exist: {base_dir}")

    splits_to_load = []
    if split in ("trainval", "all"):
        splits_to_load.append(("trainval", base_dir / "trainval.txt"))
    if split in ("test", "all"):
        splits_to_load.append(("test", base_dir / "test.txt"))

    samples: list[DeepPCBSample] = []

    for split_name, split_file in splits_to_load:
        if not split_file.exists():
            continue
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                img_rel_path, annot_rel_path = parts[0], parts[1]

                # Expected structure: groupXXXXX/YYYYY/YYYYYZZZ.jpg
                # Notice DeepPCB images have _test.jpg and _temp.jpg suffix
                img_rel = Path(img_rel_path)
                annot_rel = Path(annot_rel_path)

                group_id = img_rel.parts[0]  # e.g., 'group20085'
                board_id = img_rel.parts[1]  # e.g., '20085'
                sample_stem = img_rel.stem  # e.g., '20085000'

                test_img_path = base_dir / group_id / board_id / f"{sample_stem}_test.jpg"
                temp_img_path = base_dir / group_id / board_id / f"{sample_stem}_temp.jpg"
                annot_path = base_dir / annot_rel

                # Fallback check if test/temp named directly as img_rel
                if not test_img_path.exists():
                    test_candidate = base_dir / img_rel
                    if test_candidate.exists():
                        test_img_path = test_candidate

                boxes = parse_annotation_file(annot_path) if annot_path.exists() else []

                sample = DeepPCBSample(
                    sample_id=sample_stem,
                    group_id=group_id,
                    board_id=board_id,
                    test_image_path=test_img_path,
                    template_image_path=temp_img_path,
                    annotation_path=annot_path,
                    split=split_name,
                    boxes=boxes,
                )
                samples.append(sample)

    return samples
