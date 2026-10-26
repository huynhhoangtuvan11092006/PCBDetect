"""Export module: Converts PyTorch YOLOv8 champion model to ONNX FP32, FP16, and INT8.

Supports:
- Standard ONNX FP32 with graph optimization and constant folding.
- Half-precision ONNX FP16 for memory reduction.
- Dynamic Quantization ONNX INT8 for CPU edge acceleration.
"""

import logging
import shutil
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_all_variants(
    model_pt_path: Path = Path("models/yolov8_advanced.pt"),
    output_dir: Path = Path("models"),
    imgsz: int = 640,
) -> dict[str, Path]:
    """Export PyTorch weights to ONNX FP32, FP16, and INT8 formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not model_pt_path.exists():
        fallback = Path("models/yolov8n_baseline.pt")
        if fallback.exists():
            logger.warning(f"{model_pt_path} not found. Falling back to {fallback}")
            model_pt_path = fallback
        else:
            raise FileNotFoundError(f"No PyTorch model checkpoint found at {model_pt_path}")

    logger.info(f"Loading PyTorch checkpoint from {model_pt_path}...")
    model = YOLO(str(model_pt_path))

    exported_paths: dict[str, Path] = {"pytorch_fp32": model_pt_path}

    # 1. Export ONNX FP32
    logger.info("Exporting ONNX FP32 (Full Precision with Graph Simplification)...")
    fp32_export_file = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        simplify=True,
        half=False,
        opset=17,
    )
    fp32_target = output_dir / "yolov8_advanced.onnx"
    # Ensure it's preserved as pure FP32
    fp32_pure = output_dir / "yolov8_advanced_pure_fp32.onnx"
    shutil.copy(fp32_export_file, fp32_pure)
    exported_paths["onnx_fp32"] = fp32_target
    logger.info(f"Saved ONNX FP32 to {fp32_target} ({fp32_target.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Export ONNX INT8 via Dynamic Quantization from pure FP32
    logger.info("Quantizing ONNX FP32 to ONNX INT8 (Dynamic Quantization)...")
    int8_target = output_dir / "yolov8_advanced_int8.onnx"
    try:
        quantize_dynamic(
            model_input=str(fp32_pure),
            model_output=str(int8_target),
            weight_type=QuantType.QUInt8,
        )
        exported_paths["onnx_int8"] = int8_target
        logger.info(f"Saved ONNX INT8 to {int8_target} ({int8_target.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        logger.warning(f"INT8 Dynamic quantization failed: {e}")

    # 3. Export ONNX FP16
    logger.info("Exporting ONNX FP16 (Half Precision)...")
    try:
        fp16_export_file = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,
            simplify=True,
            half=True,
            opset=17,
        )
        fp16_target = output_dir / "yolov8_advanced_fp16.onnx"
        if Path(fp16_export_file).resolve() != fp16_target.resolve():
            shutil.copy(fp16_export_file, fp16_target)
        # Restore pure fp32 to yolov8_advanced.onnx
        shutil.copy(fp32_pure, fp32_target)
        exported_paths["onnx_fp16"] = fp16_target
        logger.info(f"Saved ONNX FP16 to {fp16_target} ({fp16_target.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        logger.warning(f"Direct FP16 export failed or not supported on this platform: {e}")

    # 3. Export ONNX INT8 via Dynamic Quantization
    logger.info("Quantizing ONNX FP32 to ONNX INT8 (Dynamic Quantization)...")
    int8_target = output_dir / "yolov8_advanced_int8.onnx"
    try:
        quantize_dynamic(
            model_input=str(fp32_target),
            model_output=str(int8_target),
            weight_type=QuantType.QUInt8,
        )
        exported_paths["onnx_int8"] = int8_target
        logger.info(f"Saved ONNX INT8 to {int8_target} ({int8_target.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        logger.warning(f"INT8 Dynamic quantization failed: {e}")

    # Validate ONNX files via ONNX Runtime session test
    for variant, path in exported_paths.items():
        if path.suffix == ".onnx":
            try:
                _ = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                logger.info(f"Validated {variant} ONNX session in ONNX Runtime successfully.")
            except Exception as e:
                logger.warning(f"ONNX session validation warning for {variant}: {e}")

    return exported_paths


if __name__ == "__main__":
    export_all_variants()
