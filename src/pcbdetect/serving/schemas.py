"""Pydantic schemas for PCBDetect FastAPI Serving endpoints."""


from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    """Detected defect bounding box and classification."""

    class_id: int = Field(..., description="Defect class integer ID (0-5)")
    class_name: str = Field(..., description="Canonical defect class name")
    score: float = Field(..., description="Confidence score [0.0 - 1.0]")
    bbox: list[float] = Field(
        ...,
        description="Bounding box [xmin, ymin, xmax, ymax] in pixel coordinates",
        min_length=4,
        max_length=4,
    )
    severity: str = Field("MAJOR", description="Defect severity: CRITICAL, MAJOR, or MINOR")


class LatencyBreakdown(BaseModel):
    """End-to-end inference latency breakdown in milliseconds."""

    preprocess_ms: float = Field(..., description="Image preprocessing and tensor conversion time")
    inference_ms: float = Field(..., description="ONNX Runtime model execution time")
    postprocess_ms: float = Field(..., description="NMS and threshold filtering time")
    total_ms: float = Field(..., description="Total wall-clock request processing time")


class PredictionResponse(BaseModel):
    """Response payload for POST /predict endpoint."""

    status: str = Field(..., description="Inspection outcome: 'PASS' (clean) or 'FAIL' (defective)")
    is_defective: bool = Field(..., description="True if any defect was confirmed above operating threshold")
    defect_count: int = Field(..., description="Total count of confirmed defects")
    detections: list[DetectionBox] = Field(default_factory=list, description="List of detected defect instances")
    class_breakdown: dict[str, int] = Field(default_factory=dict, description="Count of defects grouped by class")
    latency: LatencyBreakdown = Field(..., description="Detailed latency metrics")
    model_version: str = Field(..., description="Active model variant and format")
    overlay_base64: str | None = Field(None, description="Optional PNG overlay image encoded in base64")


class HealthResponse(BaseModel):
    """Response payload for GET /health endpoint."""

    status: str = Field("healthy", description="Service health state: 'healthy' or 'degraded'")
    model_name: str = Field(..., description="Active model identifier")
    model_precision: str = Field(..., description="Weight precision: FP16, FP32, or INT8")
    runtime_engine: str = Field(..., description="Inference engine: ONNX Runtime CPU")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    ram_usage_mb: float = Field(..., description="Peak process memory consumption in MB")
    per_class_thresholds: dict[str, float] = Field(..., description="Active operating thresholds per class")


class FeedbackRequest(BaseModel):
    """Inspector feedback request payload for POST /feedback."""

    image_id: str | None = Field(None, description="Identifier of the inspected image")
    predicted_defect_count: int = Field(..., description="Count predicted by AI model")
    is_correct: bool = Field(..., description="Inspector verification: True if AI was correct, False otherwise")
    inspector_notes: str | None = Field(None, description="Optional notes or remarks from QC inspector")
    actual_defects: list[str] | None = Field(default=None, description="Actual defect types confirmed by inspector")


class FeedbackResponse(BaseModel):
    """Response payload for POST /feedback."""

    success: bool = Field(True, description="True if feedback was successfully recorded")
    message: str = Field(..., description="Status message")
    feedback_id: str = Field(..., description="Unique UUID for recorded feedback entry")
    recorded_at: str = Field(..., description="ISO 8601 UTC timestamp of record creation")
