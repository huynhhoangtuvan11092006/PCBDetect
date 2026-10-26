"""Image alignment and classical difference computation using OpenCV.

Provides registration algorithms to compensate for micro-mechanical vibrations
and illumination shifts, generating classical difference residuals.
"""

from typing import Tuple
import cv2
import numpy as np


def align_images_orb(
    test_img: np.ndarray,
    template_img: np.ndarray,
    max_features: int = 1500,
    good_match_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Align test image to golden template using ORB feature detector and Homography.

    Args:
        test_img: Tested defective PCB image (BGR).
        template_img: Defect-free golden template image (BGR).
        max_features: Number of ORB keypoints to extract.
        good_match_ratio: Fraction of best matches to retain for homography estimation.

    Returns:
        Tuple of (aligned_test_img, homography_matrix, is_aligned_successfully).
    """
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) if test_img.ndim == 3 else test_img
    gray_temp = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY) if template_img.ndim == 3 else template_img

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(gray_test, None)
    kp2, des2 = orb.detectAndCompute(gray_temp, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        # Fallback: return unaligned image if keypoints are insufficient
        return test_img.copy(), np.eye(3, dtype=np.float32), False

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    num_good_matches = max(4, int(len(matches) * good_match_ratio))
    good_matches = matches[:num_good_matches]

    pts_test = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_temp = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    h_matrix, inliers = cv2.findHomography(pts_test, pts_temp, cv2.RANSAC, 5.0)

    if h_matrix is None:
        return test_img.copy(), np.eye(3, dtype=np.float32), False

    height, width = template_img.shape[:2]
    aligned_test = cv2.warpPerspective(test_img, h_matrix, (width, height))
    return aligned_test, h_matrix, True


def compute_difference_image(
    test_img: np.ndarray,
    template_img: np.ndarray,
    align_first: bool = False,
    diff_threshold: int = 40,
    morph_kernel_size: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute difference residual and defect candidate binary mask.

    Args:
        test_img: Tested defective PCB image.
        template_img: Golden defect-free template image.
        align_first: If True, executes ORB registration before subtraction.
        diff_threshold: Pixel intensity difference threshold (0 to 255).
        morph_kernel_size: Size of morphological structuring element.

    Returns:
        Tuple of (enhanced_difference_img_bgr, clean_binary_mask).
    """
    if align_first:
        curr_test, _, _ = align_images_orb(test_img, template_img)
    else:
        curr_test = test_img

    # Absolute difference
    diff = cv2.absdiff(curr_test, template_img)

    # Convert to grayscale if 3-channel
    if diff.ndim == 3:
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    else:
        diff_gray = diff

    # Binarize difference image
    _, binary_mask = cv2.threshold(diff_gray, diff_threshold, 255, cv2.THRESH_BINARY)

    # Morphological noise filtering: Opening to remove salt-and-pepper noise, Closing to bridge micro-gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # Enhanced visual representation
    enhanced_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    return enhanced_diff, cleaned_mask
