"""FastAPI Application for PCBDetect Automated Optical Inspection (AOI) Service."""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import psutil
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from pcbdetect.serving.inference_service import InferenceService
from pcbdetect.serving.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictionResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pcbdetect.api")

FEEDBACK_DIR = Path("data/feedback")
FEEDBACK_FILE = FEEDBACK_DIR / "qc_feedback.jsonl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler: load ONNX model and prepare resources on startup."""
    logger.info("Initializing PCBDetect Inference Service...")
    app.state.startup_time = time.time()
    try:
        app.state.service = InferenceService()
        logger.info(f"Service loaded model: {app.state.service.model_path.name}")
    except Exception as e:
        logger.error(f"Failed to load inference service: {e}")
        app.state.service = None

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Shutting down PCBDetect Inference Service...")


app = FastAPI(
    title="PCBDetect AOI Quality Inspection Service",
    description="Automated Optical Inspection API for PCB surface defect detection using ONNX Runtime.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_latency_middleware(request: Request, call_next):
    """Log latency of every incoming HTTP request."""
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    if request.url.path not in ["/health"]:
        logger.info(f"{request.method} {request.url.path} - {response.status_code} ({duration_ms:.1f}ms)")
    return response


@app.get("/", summary="Root status check")
async def root():
    """Root info endpoint."""
    return {
        "service": "PCBDetect AOI Quality Inspection API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
    }


@app.get("/health", response_model=HealthResponse, summary="Service Health & Model Status")
async def health(request: Request):
    """Return health state, active model version, memory footprint, and thresholds."""
    service: InferenceService = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is not loaded or degraded",
        )

    uptime = time.time() - request.app.state.startup_time
    mem_info = psutil.Process(os.getpid()).memory_info()
    ram_mb = mem_info.rss / (1024 * 1024)

    return HealthResponse(
        status="healthy",
        model_name=service.model_path.name,
        model_precision=service.precision,
        runtime_engine="ONNX Runtime CPU",
        uptime_seconds=round(uptime, 1),
        ram_usage_mb=round(ram_mb, 1),
        per_class_thresholds=service.per_class_thresholds,
    )


@app.post("/predict", response_model=PredictionResponse, summary="Inspect single PCB image")
async def predict(
    request: Request,
    file: UploadFile = File(..., description="PCB test image file (JPEG or PNG)"),
    return_overlay: bool = True,
):
    """Perform surface defect inspection on uploaded PCB image."""
    service: InferenceService = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not ready",
        )

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty filename provided")

    # Read image bytes
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    try:
        response = service.predict(contents, return_overlay=return_overlay)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Inference error on {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {e!s}")


@app.post("/feedback", response_model=FeedbackResponse, summary="Submit QC Inspector feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Record QC inspector feedback for Human-in-the-Loop active learning."""
    feedback_id = str(uuid.uuid4())
    recorded_at = datetime.now(UTC).isoformat()

    record = {
        "feedback_id": feedback_id,
        "recorded_at": recorded_at,
        "image_id": feedback.image_id,
        "predicted_defect_count": feedback.predicted_defect_count,
        "is_correct": feedback.is_correct,
        "inspector_notes": feedback.inspector_notes,
        "actual_defects": feedback.actual_defects,
    }

    try:
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Recorded feedback {feedback_id}: correct={feedback.is_correct}")
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save feedback")

    return FeedbackResponse(
        success=True,
        message="Feedback successfully recorded for model retraining loop.",
        feedback_id=feedback_id,
        recorded_at=recorded_at,
    )
