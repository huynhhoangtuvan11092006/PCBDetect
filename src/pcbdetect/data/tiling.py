"""Tiling and sliding window processing for high-resolution PCB images.

Enables processing large PCB scans (e.g. 2Kx2K, 16Kx16K) by slicing into
standard-sized overlapping patches while accurately remapping bounding box coordinates.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from pcbdetect.data.dataset import BoundingBox


@dataclass
class ImageTile:
    """Represents a sliced patch from a larger image canvas."""
    tile_index: int
    image: np.ndarray
    x_offset: int
    y_offset: int
    width: int
    height: int
    boxes: List[BoundingBox]


def slice_image_with_boxes(
    image: np.ndarray,
    boxes: List[BoundingBox],
    tile_size: int = 640,
    overlap_ratio: float = 0.2,
    min_box_area_ratio: float = 0.35,
) -> List[ImageTile]:
    """Slice an image and its bounding boxes into overlapping tiles.

    Args:
        image: Original input image array (H, W, C).
        boxes: List of BoundingBox objects in original image coordinates.
        tile_size: Square tile dimension (default 640).
        overlap_ratio: Overlap fraction between adjacent tiles (e.g. 0.2 = 20%).
        min_box_area_ratio: Minimum remaining area fraction to retain a clipped box.

    Returns:
        List of ImageTile objects with translated local coordinates.
    """
    img_h, img_w = image.shape[:2]
    step = int(tile_size * (1.0 - overlap_ratio))

    x_starts = list(range(0, max(1, img_w - tile_size + 1), step))
    if x_starts[-1] + tile_size < img_w:
        x_starts.append(img_w - tile_size)

    y_starts = list(range(0, max(1, img_h - tile_size + 1), step))
    if y_starts[-1] + tile_size < img_h:
        y_starts.append(img_h - tile_size)

    tiles: List[ImageTile] = []
    tile_idx = 0

    for y in y_starts:
        for x in x_starts:
            tile_img = image[y : y + tile_size, x : x + tile_size].copy()
            tile_boxes: List[BoundingBox] = []

            for b in boxes:
                # Calculate intersection between box and tile window
                inter_x1 = max(x, b.x1)
                inter_y1 = max(y, b.y1)
                inter_x2 = min(x + tile_size, b.x2)
                inter_y2 = min(y + tile_size, b.y2)

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    orig_area = b.area

                    if orig_area > 0 and (inter_area / orig_area) >= min_box_area_ratio:
                        # Translate to local tile coordinates
                        local_box = BoundingBox(
                            x1=inter_x1 - x,
                            y1=inter_y1 - y,
                            x2=inter_x2 - x,
                            y2=inter_y2 - y,
                            class_id=b.class_id,
                            class_name=b.class_name,
                        )
                        tile_boxes.append(local_box)

            tiles.append(
                ImageTile(
                    tile_index=tile_idx,
                    image=tile_img,
                    x_offset=x,
                    y_offset=y,
                    width=tile_size,
                    height=tile_size,
                    boxes=tile_boxes,
                )
            )
            tile_idx += 1

    return tiles


def remap_predictions_to_global(
    tile_predictions: List[Tuple[int, int, int, int, float, int]],  # (x1, y1, x2, y2, score, class_id)
    x_offset: int,
    y_offset: int,
) -> List[Tuple[int, int, int, int, float, int]]:
    """Remap predictions made on a local tile back to global PCB canvas coordinates."""
    remapped = []
    for x1, y1, x2, y2, score, cls_id in tile_predictions:
        remapped.append(
            (
                x1 + x_offset,
                y1 + y_offset,
                x2 + x_offset,
                y2 + y_offset,
                score,
                cls_id,
            )
        )
    return remapped
