"""Data modules for PCBDetect.

Includes parsing, exploratory data analysis, dataset splitting,
template alignment, tiling, augmentation, and YOLO/COCO export pipelines.
"""

from pcbdetect.data.align import align_images_orb, compute_difference_image
from pcbdetect.data.augment import (
    apply_pcb_augmentation,
    flip_horizontal,
    flip_vertical,
    rotate_90,
    rotate_180,
)
from pcbdetect.data.convert import export_coco_dataset, export_yolo_dataset, run_pipeline
from pcbdetect.data.dataset import (
    BoundingBox,
    DeepPCBSample,
    load_deeppcb_dataset,
    parse_annotation_file,
)
from pcbdetect.data.split import (
    DatasetSplits,
    create_group_disjoint_split,
    create_standard_split,
    get_dataset_splits,
)
from pcbdetect.data.tiling import (
    ImageTile,
    remap_predictions_to_global,
    slice_image_with_boxes,
)

__all__ = [
    "BoundingBox",
    "DeepPCBSample",
    "load_deeppcb_dataset",
    "parse_annotation_file",
    "DatasetSplits",
    "create_group_disjoint_split",
    "create_standard_split",
    "get_dataset_splits",
    "align_images_orb",
    "compute_difference_image",
    "ImageTile",
    "slice_image_with_boxes",
    "remap_predictions_to_global",
    "apply_pcb_augmentation",
    "flip_horizontal",
    "flip_vertical",
    "rotate_90",
    "rotate_180",
    "export_yolo_dataset",
    "export_coco_dataset",
    "run_pipeline",
]
