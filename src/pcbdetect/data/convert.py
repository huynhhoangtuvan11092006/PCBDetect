"""Unified conversion pipeline for DeepPCB dataset.

Generates both:
1. Ultralytics YOLO format with data.yaml
2. MS COCO format with JSON instances files

Supports both 'standard' and 'group_disjoint' split strategies.
Can be executed in a single command.
"""

import argparse
import json
from pathlib import Path
import shutil
from typing import Dict, List, Literal
import yaml
from tqdm import tqdm

from pcbdetect.constants import (
    DEFECT_ID_TO_NAME,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    PROCESSED_DATA_DIR,
    YOLO_ID_TO_NAME,
)
from pcbdetect.data.dataset import DeepPCBSample
from pcbdetect.data.split import DatasetSplits, get_dataset_splits


def export_yolo_dataset(
    splits: DatasetSplits,
    output_dir: Path,
) -> Path:
    """Export dataset splits to Ultralytics YOLO format.

    Args:
        splits: DatasetSplits containing train, val, and test samples.
        output_dir: Destination directory (e.g. data/processed/yolo).

    Returns:
        Path to generated data.yaml.
    """
    yolo_dir = output_dir / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[YOLO Export] Writing YOLO format to {yolo_dir} (Split mode: {splits.mode})...")

    split_map = {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }

    for split_name, samples in split_map.items():
        img_dest_dir = yolo_dir / "images" / split_name
        lbl_dest_dir = yolo_dir / "labels" / split_name
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        lbl_dest_dir.mkdir(parents=True, exist_ok=True)

        for s in tqdm(samples, desc=f"Exporting YOLO {split_name} ({len(samples)} imgs)"):
            # Copy test image
            dst_img_name = f"{s.group_id}_{s.sample_id}_test.jpg"
            dst_img_path = img_dest_dir / dst_img_name
            shutil.copy2(s.test_image_path, dst_img_path)

            # Write YOLO label file: class_idx_0_indexed x_c y_c w_norm h_norm
            dst_lbl_name = f"{s.group_id}_{s.sample_id}_test.txt"
            dst_lbl_path = lbl_dest_dir / dst_lbl_name

            with open(dst_lbl_path, "w", encoding="utf-8") as f:
                for b in s.boxes:
                    cls_idx, x_c, y_c, w_norm, h_norm = b.to_yolo_format(
                        DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT
                    )
                    f.write(f"{cls_idx} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    # Generate data.yaml with absolute and relative paths
    data_yaml_content = {
        "path": str(yolo_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": YOLO_ID_TO_NAME,
        "nc": len(YOLO_ID_TO_NAME),
        "split_strategy": splits.mode,
    }

    yaml_path = yolo_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"[YOLO Export] Finished! Config file created at: {yaml_path}")
    return yaml_path


def export_coco_dataset(
    splits: DatasetSplits,
    output_dir: Path,
) -> Dict[str, Path]:
    """Export dataset splits to standard MS COCO format for Faster R-CNN.

    Args:
        splits: DatasetSplits containing train, val, and test samples.
        output_dir: Destination directory (e.g. data/processed/coco).

    Returns:
        Dict mapping split names to generated JSON paths.
    """
    coco_dir = output_dir / "coco"
    annot_dir = coco_dir / "annotations"
    annot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[COCO Export] Writing COCO format to {coco_dir} (Split mode: {splits.mode})...")

    categories = [
        {"id": cls_id, "name": cls_name, "supercategory": "pcb_defect"}
        for cls_id, cls_name in DEFECT_ID_TO_NAME.items()
    ]

    split_map = {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }

    json_paths = {}

    for split_name, samples in split_map.items():
        img_dest_dir = coco_dir / "images" / split_name
        img_dest_dir.mkdir(parents=True, exist_ok=True)

        images_list: List[dict] = []
        annotations_list: List[dict] = []
        annot_id = 1

        for img_id, s in enumerate(tqdm(samples, desc=f"Exporting COCO {split_name} ({len(samples)} imgs)"), start=1):
            dst_img_name = f"{s.group_id}_{s.sample_id}_test.jpg"
            dst_img_path = img_dest_dir / dst_img_name
            shutil.copy2(s.test_image_path, dst_img_path)

            images_list.append({
                "id": img_id,
                "file_name": dst_img_name,
                "width": DEFAULT_IMAGE_WIDTH,
                "height": DEFAULT_IMAGE_HEIGHT,
                "group_id": s.group_id,
                "board_id": s.board_id,
            })

            for b in s.boxes:
                # COCO bbox format: [x_top_left, y_top_left, width, height]
                coco_bbox = [b.x1, b.y1, b.width, b.height]
                annotations_list.append({
                    "id": annot_id,
                    "image_id": img_id,
                    "category_id": b.class_id,
                    "bbox": coco_bbox,
                    "area": b.area,
                    "segmentation": [],
                    "iscrowd": 0,
                })
                annot_id += 1

        coco_data = {
            "info": {
                "description": "DeepPCB Defect Dataset in MS COCO Format",
                "version": "1.0",
                "year": 2026,
                "contributor": "PCBDetect Engineering Team",
                "split_strategy": splits.mode,
            },
            "licenses": [{"id": 1, "name": "MIT License"}],
            "images": images_list,
            "annotations": annotations_list,
            "categories": categories,
        }

        json_file_path = annot_dir / f"instances_{split_name}.json"
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)

        json_paths[split_name] = json_file_path
        print(f"[COCO Export] Generated {json_file_path.name}: {len(images_list)} images, {len(annotations_list)} annotations")

    return json_paths


def run_pipeline(
    strategy: Literal["standard", "group_disjoint"] = "group_disjoint",
    target_format: Literal["both", "yolo", "coco"] = "both",
    output_dir: Path = PROCESSED_DATA_DIR,
    val_ratio: float = 0.2,
) -> dict:
    """Execute complete end-to-end dataset preparation and conversion."""
    print("=" * 65)
    print(f"PCBDetect Data Pipeline: Strategy='{strategy}', Format='{target_format}'")
    print("=" * 65)

    splits = get_dataset_splits(strategy=strategy, val_ratio=val_ratio)
    split_info = splits.summary()
    print("\nDataset Partitioning Summary:")
    print(f"  - Strategy:    {split_info['mode']}")
    print(f"  - Train Count: {split_info['train_count']} images ({len(split_info['train_groups'])} groups)")
    print(f"  - Val Count:   {split_info['val_count']} images ({len(split_info['val_groups'])} groups)")
    print(f"  - Test Count:  {split_info['test_count']} images ({len(split_info['test_groups'])} groups)")
    print(f"  - Total Count: {split_info['total_count']} images")

    results = {"split_summary": split_info}

    if target_format in ("yolo", "both"):
        yaml_path = export_yolo_dataset(splits, output_dir)
        results["yolo_data_yaml"] = str(yaml_path)

    if target_format in ("coco", "both"):
        coco_paths = export_coco_dataset(splits, output_dir)
        results["coco_annotations"] = {k: str(v) for k, v in coco_paths.items()}

    # Save summary metadata
    meta_path = output_dir / f"pipeline_run_{strategy}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("M1 Data Pipeline Conversion Completed Successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 65)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DeepPCB dataset to YOLO/COCO format")
    parser.add_argument(
        "--strategy",
        choices=["standard", "group_disjoint"],
        default="group_disjoint",
        help="Split strategy: 'group_disjoint' (zero leakage) or 'standard' (paper benchmark)",
    )
    parser.add_argument(
        "--format",
        choices=["both", "yolo", "coco"],
        default="both",
        help="Target format to export: 'yolo', 'coco', or 'both'",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Destination directory for processed dataset",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio for standard split (default: 0.2)",
    )

    args = parser.parse_args()
    run_pipeline(
        strategy=args.strategy,
        target_format=args.format,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
    )
