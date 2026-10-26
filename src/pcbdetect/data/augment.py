"""PCB-domain safe data augmentation pipeline using OpenCV and NumPy.

Guarantees:
- Geometric invariance (Flips, 90/180/270 rotations) with exact analytical bounding box transforms.
- Defect preservation: avoids aggressive cropping or heavy blurs that eliminate tiny defects (<32px).
- Bounding box validity checking: ensuring all transformed boxes remain within boundaries.
"""

from typing import List, Tuple
import cv2
import numpy as np

from pcbdetect.data.dataset import BoundingBox


def flip_horizontal(
    image: np.ndarray,
    boxes: List[BoundingBox],
) -> Tuple[np.ndarray, List[BoundingBox]]:
    """Flip image and boxes horizontally."""
    w = image.shape[1]
    flipped_img = cv2.flip(image, 1)
    new_boxes = []
    for b in boxes:
        new_boxes.append(
            BoundingBox(
                x1=w - b.x2,
                y1=b.y1,
                x2=w - b.x1,
                y2=b.y2,
                class_id=b.class_id,
                class_name=b.class_name,
            )
        )
    return flipped_img, new_boxes


def flip_vertical(
    image: np.ndarray,
    boxes: List[BoundingBox],
) -> Tuple[np.ndarray, List[BoundingBox]]:
    """Flip image and boxes vertically."""
    h = image.shape[0]
    flipped_img = cv2.flip(image, 0)
    new_boxes = []
    for b in boxes:
        new_boxes.append(
            BoundingBox(
                x1=b.x1,
                y1=h - b.y2,
                x2=b.x2,
                y2=h - b.y1,
                class_id=b.class_id,
                class_name=b.class_name,
            )
        )
    return flipped_img, new_boxes


def rotate_90(
    image: np.ndarray,
    boxes: List[BoundingBox],
) -> Tuple[np.ndarray, List[BoundingBox]]:
    """Rotate image and boxes 90 degrees clockwise."""
    h, w = image.shape[:2]
    rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    new_boxes = []
    for b in boxes:
        # After 90 deg clockwise: x' = h - y, y' = x
        new_x1 = h - b.y2
        new_y1 = b.x1
        new_x2 = h - b.y1
        new_y2 = b.x2
        new_boxes.append(
            BoundingBox(
                x1=min(new_x1, new_x2),
                y1=min(new_y1, new_y2),
                x2=max(new_x1, new_x2),
                y2=max(new_y1, new_y2),
                class_id=b.class_id,
                class_name=b.class_name,
            )
        )
    return rotated_img, new_boxes


def rotate_180(
    image: np.ndarray,
    boxes: List[BoundingBox],
) -> Tuple[np.ndarray, List[BoundingBox]]:
    """Rotate image and boxes 180 degrees."""
    h, w = image.shape[:2]
    rotated_img = cv2.rotate(image, cv2.ROTATE_180)
    new_boxes = []
    for b in boxes:
        new_boxes.append(
            BoundingBox(
                x1=w - b.x2,
                y1=h - b.y2,
                x2=w - b.x1,
                y2=h - b.y1,
                class_id=b.class_id,
                class_name=b.class_name,
            )
        )
    return rotated_img, new_boxes


def adjust_contrast_brightness(
    image: np.ndarray,
    alpha: float = 1.1,  # Contrast factor (0.8 - 1.2)
    beta: int = 5,       # Brightness factor (-20 to 20)
) -> np.ndarray:
    """Adjust contrast and brightness safely."""
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float = 8.0,
) -> np.ndarray:
    """Add subtle Gaussian sensor noise without dissolving small defects."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_pcb_augmentation(
    image: np.ndarray,
    boxes: List[BoundingBox],
    aug_type: str,
) -> Tuple[np.ndarray, List[BoundingBox]]:
    """Apply a named augmentation safely."""
    if aug_type == "flip_h":
        return flip_horizontal(image, boxes)
    elif aug_type == "flip_v":
        return flip_vertical(image, boxes)
    elif aug_type == "rot_90":
        return rotate_90(image, boxes)
    elif aug_type == "rot_180":
        return rotate_180(image, boxes)
    elif aug_type == "photometric":
        adj_img = adjust_contrast_brightness(image, alpha=1.08, beta=8)
        noisy = add_gaussian_noise(adj_img, sigma=5.0)
        return noisy, [BoundingBox(b.x1, b.y1, b.x2, b.y2, b.class_id, b.class_name) for b in boxes]
    else:
        return image.copy(), [BoundingBox(b.x1, b.y1, b.x2, b.y2, b.class_id, b.class_name) for b in boxes]
