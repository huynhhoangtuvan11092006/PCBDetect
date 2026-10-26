"""Unit and integration tests for Milestone 7: Serving and Web API."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pcbdetect.serving.app import app


@pytest.fixture(scope="module")
def client():
    """Create FastAPI TestClient within lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    """Verify GET / returns service descriptor."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "PCBDetect" in data["service"]
    assert "docs_url" in data
    assert "predict_url" in data


def test_health_endpoint(client: TestClient):
    """Verify GET /health returns healthy status and model telemetry."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_precision"] in ["FP16", "FP32", "INT8"]
    assert "open" in data["per_class_thresholds"]
    assert data["uptime_seconds"] >= 0
    assert data["ram_usage_mb"] > 0


def test_predict_valid_image(client: TestClient):
    """Verify POST /predict succeeds on real PCB image and returns expected schema."""
    test_dir = Path("data/processed/yolo/images/test")
    test_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if not test_files:
        pytest.skip("Test images not available")

    with open(test_files[0], "rb") as f:
        img_bytes = f.read()

    files = {"file": ("test_pcb.jpg", img_bytes, "image/jpeg")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] in ["PASS", "FAIL"]
    assert isinstance(data["is_defective"], bool)
    assert isinstance(data["defect_count"], int)
    assert "latency" in data
    assert data["latency"]["total_ms"] > 0
    assert "overlay_base64" in data
    assert data["overlay_base64"] is not None


def test_predict_corrupt_file(client: TestClient):
    """Verify POST /predict properly rejects non-image corrupted files with HTTP 400."""
    fake_bytes = b"not an image file data content"
    files = {"file": ("corrupt.jpg", fake_bytes, "image/jpeg")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400


def test_feedback_endpoint(client: TestClient):
    """Verify POST /feedback records QC inspector annotations."""
    payload = {
        "image_id": "test_board_001.jpg",
        "predicted_defect_count": 2,
        "is_correct": True,
        "inspector_notes": "All bounding boxes verified by QC line inspector",
    }
    resp = client.post("/feedback", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "feedback_id" in data
    assert "recorded_at" in data
