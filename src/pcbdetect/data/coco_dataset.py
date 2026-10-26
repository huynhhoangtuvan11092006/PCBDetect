"""PyTorch Dataset implementation for MS COCO format on PCBDetect.

Compatible with torchvision object detection models (Faster R-CNN, RetinaNet, SSD).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PCBCocoDataset(Dataset):
    """PyTorch Dataset for PCB defect detection using COCO JSON annotations."""

    def __init__(
        self,
        annotations_json: Path,
        images_dir: Path,
        transforms: Any = None,
    ):
        self.annotations_json = Path(annotations_json)
        self.images_dir = Path(images_dir)
        self.transforms = transforms

        if not self.annotations_json.exists():
            raise FileNotFoundError(f"COCO annotations file not found: {self.annotations_json}")

        with open(self.annotations_json, "r", encoding="utf-8") as f:
            coco_dict = json.load(f)

        self.images = coco_dict.get("images", [])
        self.categories = {cat["id"]: cat["name"] for cat in coco_dict.get("categories", [])}

        # Index annotations by image_id
        self.img_to_annots: Dict[int, List[dict]] = {}
        for ann in coco_dict.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.img_to_annots:
                self.img_to_annots[img_id] = []
            self.img_to_annots[img_id].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = self.images_dir / img_info["file_name"]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Convert to Tensor (C, H, W) in [0.0, 1.0]
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        annots = self.img_to_annots.get(img_id, [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annots:
            # COCO bbox: [x_min, y_min, width, height]
            x, y, w, h = ann["bbox"]
            # Convert to [x1, y1, x2, y2]
            x1, y1 = float(x), float(y)
            x2, y2 = float(x + w), float(y + h)

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(int(ann["category_id"]))
                areas.append(float(ann["area"]))
                iscrowd.append(int(ann.get("iscrowd", 0)))

        if len(boxes) == 0:
            # Empty tensor for images with no defects
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            areas_tensor = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        return img_tensor, target


def detection_collate_fn(batch: List[Tuple[torch.Tensor, dict]]) -> Tuple[List[torch.Tensor], List[dict]]:
    """Collate function for object detection batching."""
    return tuple(zip(*batch))
