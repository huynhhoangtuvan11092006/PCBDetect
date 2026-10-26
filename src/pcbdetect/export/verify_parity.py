"""Numerical Parity Verification: Verifies PyTorch vs. ONNX Runtime consistency.

Ensures that mathematical outputs, class scores, and bounding box coordinates
match within tight numerical tolerances (Cosine Similarity >= 0.9999).
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def preprocess_image(image_path: Path, imgsz: int = 640) -> np.ndarray:
    """Preprocess image to standard YOLO tensor shape (1, 3, 640, 640) in [0, 1]."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image at {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (imgsz, imgsz))
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))  # (3, H, W)
    tensor = np.expand_dims(tensor, axis=0)   # (1, 3, H, W)
    return tensor


def verify_pytorch_onnx_parity(
    model_pt_path: Path = Path("models/yolov8_advanced.pt"),
    model_onnx_path: Path = Path("models/yolov8_advanced.onnx"),
    sample_image_path: Path = Path("data/processed/yolo/images/test"),
    imgsz: int = 640,
    tolerance_cosine: float = 0.9999,
) -> dict[str, Any]:
    """Verify numerical parity between PyTorch and ONNX Runtime predictions."""
    if not model_pt_path.exists():
        model_pt_path = Path("models/yolov8n_baseline.pt")
    if not model_onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {model_onnx_path}. Run export_onnx.py first.")

    # Find a sample test image
    if sample_image_path.is_dir():
        imgs = list(sample_image_path.glob("*.jpg")) + list(sample_image_path.glob("*.png"))
        if not imgs:
            raise FileNotFoundError(f"No test images found in {sample_image_path}")
        sample_img = imgs[0]
    else:
        sample_img = sample_image_path

    logger.info(f"Verifying numerical parity on sample image: {sample_img.name}")
    input_tensor_np = preprocess_image(sample_img, imgsz=imgsz)
    input_tensor_torch = torch.from_numpy(input_tensor_np)

    # 1. Run PyTorch inference
    yolo_model = YOLO(str(model_pt_path))
    torch_model = yolo_model.model
    torch_model.eval()

    with torch.no_grad():
        preds_torch = torch_model(input_tensor_torch)
        if isinstance(preds_torch, (tuple, list)):
            preds_torch = preds_torch[0]
        preds_torch_np = preds_torch.cpu().numpy()

    # 2. Run ONNX Runtime inference
    session = ort.InferenceSession(str(model_onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    preds_onnx = session.run([output_name], {input_name: input_tensor_np})[0]

    # 3. Compute Numerical Discrepancy
    diff = np.abs(preds_torch_np - preds_onnx)
    max_abs_error = float(np.max(diff))
    mean_abs_error = float(np.mean(diff))

    # Cosine similarity
    vec1 = preds_torch_np.flatten()
    vec2 = preds_onnx.flatten()
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cosine_sim = float(np.dot(vec1, vec2) / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 1.0

    passed = bool(cosine_sim >= tolerance_cosine and max_abs_error < 0.05)

    result = {
        "sample_image": sample_img.name,
        "pytorch_output_shape": list(preds_torch_np.shape),
        "onnx_output_shape": list(preds_onnx.shape),
        "max_absolute_error": round(max_abs_error, 6),
        "mean_absolute_error": round(mean_abs_error, 6),
        "cosine_similarity": round(cosine_sim, 6),
        "parity_verified": passed,
        "tolerance_cosine": tolerance_cosine,
    }

    logger.info(f"Parity Results: Max Abs Error={max_abs_error:.6f}, Cosine Sim={cosine_sim:.6f}, Passed={passed}")
    return result


if __name__ == "__main__":
    verify_pytorch_onnx_parity()
