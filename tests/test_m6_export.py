"""Unit tests for Milestone 6 (M6): ONNX Export, Parity, and Edge Benchmarking."""

from pathlib import Path

import numpy as np
import pytest

from pcbdetect.benchmark.benchmark_inference import BenchmarkResult, InferenceBenchmark
from pcbdetect.export.verify_parity import preprocess_image


def test_preprocess_image():
    """Verify image preprocessing produces standard (1, 3, 640, 640) float32 tensor."""
    test_img = Path("data/processed/yolo/images/test")
    if not test_img.exists():
        pytest.skip("Test images not found")

    sample_files = list(test_img.glob("*.jpg")) + list(test_img.glob("*.png"))
    if not sample_files:
        pytest.skip("No sample images in test dir")

    tensor = preprocess_image(sample_files[0], imgsz=640)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_benchmark_result_schema():
    """Verify BenchmarkResult dataclass fields and non-negative metric calculations."""
    res = BenchmarkResult(
        variant_name="Test Variant",
        runtime_engine="ONNX Runtime",
        precision="FP32",
        model_size_mb=6.5,
        latency_mean_ms=45.0,
        latency_p50_ms=44.2,
        latency_p95_ms=48.5,
        latency_std_ms=2.1,
        throughput_fps=22.2,
        peak_ram_mb=350.0,
        test_map50=0.8184,
    )
    assert res.latency_p95_ms >= res.latency_p50_ms
    assert res.throughput_fps > 0
    assert res.model_size_mb > 0
    assert res.peak_ram_mb > 0


def test_inference_benchmark_warmup():
    """Verify InferenceBenchmark helper methods."""
    bench = InferenceBenchmark(imgsz=640, warmup_runs=2, benchmark_runs=5)
    mem = bench.get_memory_mb()
    assert mem > 0
