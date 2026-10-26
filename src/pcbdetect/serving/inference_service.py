"""Inference Service for PCBDetect using ONNX Runtime with Per-Class Dynamic Thresholds."""

import base64
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from pcbdetect.constants import DEFECT_CLASSES, DEFECT_COLORS_RGB, YOLO_ID_TO_NAME
from pcbdetect.serving.schemas import DetectionBox, LatencyBreakdown, PredictionResponse

logger = logging.getLogger("pcbdetect.serving")


class InferenceService:
    """Production inference engine wrapping ONNX Runtime with M5 operating points."""

    def __init__(
        self,
        model_path: str | None = None,
        thresholds_path: str | None = None,
    ):
        self.model_path = self._resolve_model_path(model_path)
        self.thresholds_path = (
            Path(thresholds_path)
            if thresholds_path
            else Path("configs/operating_thresholds.json")
        )
        self.per_class_thresholds = self._load_thresholds()

        logger.info(f"Loading ONNX session from {self.model_path}...")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_dtype = self.session.get_inputs()[0].type
        self.imgsz = 640

        # Determine precision tag
        if "fp16" in self.model_path.name.lower():
            self.precision = "FP16"
        elif "int8" in self.model_path.name.lower():
            self.precision = "INT8"
        else:
            self.precision = "FP32"

        logger.info(
            f"ONNX Session ready: {self.model_path.name} ({self.precision}), "
            f"input: {self.input_name} {self.input_shape} ({self.input_dtype})"
        )

    def _resolve_model_path(self, model_path: str | None) -> Path:
        """Find best available ONNX model variant."""
        if model_path and Path(model_path).exists():
            return Path(model_path)

        candidates = [
            Path("models/yolov8_advanced_fp16.onnx"),
            Path("models/yolov8_advanced.onnx"),
            Path("models/yolov8_advanced_pure_fp32.onnx"),
            Path("models/yolov8_advanced_int8.onnx"),
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError("No valid ONNX model found in models/ directory!")

    def _load_thresholds(self) -> dict[str, float]:
        """Load optimized per-class thresholds from M5 config."""
        default_thresholds = {
            "open": 0.02,
            "short": 0.02,
            "mousebite": 0.08,
            "spur": 0.02,
            "copper": 0.03,
            "pin_hole": 0.35,
        }
        if self.thresholds_path.exists():
            try:
                with open(self.thresholds_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    pct = data.get("per_class_thresholds", {})
                    loaded = {}
                    for cls_name, info in pct.items():
                        if isinstance(info, dict) and "threshold" in info:
                            loaded[cls_name] = float(info["threshold"])
                        elif isinstance(info, (int, float)):
                            loaded[cls_name] = float(info)
                    if loaded:
                        return loaded
            except Exception as e:
                logger.warning(f"Failed to load operating thresholds: {e}, using defaults.")
        return default_thresholds

    def preprocess(self, img_bytes: bytes) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Decode raw image bytes and prepare normalized tensor."""
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Corrupt or invalid image format")

        orig_h, orig_w = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = img_rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        if "float16" in self.input_dtype:
            tensor = tensor.astype(np.float16)

        scale_x = orig_w / float(self.imgsz)
        scale_y = orig_h / float(self.imgsz)
        return tensor, img_bgr, scale_x, scale_y

    def predict(
        self,
        img_bytes: bytes,
        custom_thresholds: dict[str, float] | None = None,
        iou_threshold: float = 0.45,
        return_overlay: bool = True,
    ) -> PredictionResponse:
        """Run complete detection pipeline: preprocess -> onnx infer -> nms -> overlay."""
        t0 = time.perf_counter()

        # 1. Preprocess
        tensor, orig_img_bgr, scale_x, scale_y = self.preprocess(img_bytes)
        t1 = time.perf_counter()

        # 2. ONNX Inference
        outputs = self.session.run(None, {self.input_name: tensor})
        raw_preds = outputs[0]  # Shape: (1, 4 + num_classes, num_anchors)
        t2 = time.perf_counter()

        # 3. Postprocess (NMS with Per-Class Thresholds)
        preds = np.squeeze(raw_preds, axis=0)  # (10, 8400)
        preds = np.transpose(preds, (1, 0))  # (8400, 10)

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]  # (8400, 6)

        active_thresholds = dict(self.per_class_thresholds)
        if custom_thresholds:
            active_thresholds.update(custom_thresholds)

        candidate_boxes = []
        candidate_scores = []
        candidate_class_ids = []

        for anchor_idx in range(len(preds)):
            scores = class_scores[anchor_idx]
            cls_id = int(np.argmax(scores))
            score = float(scores[cls_id])
            cls_name = YOLO_ID_TO_NAME.get(cls_id, f"class_{cls_id}")
            thresh = active_thresholds.get(cls_name, 0.02)

            if score >= thresh:
                cx, cy, w, h = boxes_xywh[anchor_idx]
                xmin = (cx - w / 2.0) * scale_x
                ymin = (cy - h / 2.0) * scale_y
                xmax = (cx + w / 2.0) * scale_x
                ymax = (cy + h / 2.0) * scale_y

                candidate_boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                candidate_scores.append(score)
                candidate_class_ids.append(cls_id)

        # Non-Maximum Suppression (NMS)
        detections: list[DetectionBox] = []
        class_breakdown: dict[str, int] = {c: 0 for c in DEFECT_CLASSES}

        if candidate_boxes:
            indices = cv2.dnn.NMSBoxes(
                bboxes=candidate_boxes,
                scores=candidate_scores,
                score_threshold=0.01,
                nms_threshold=iou_threshold,
            )
            if len(indices) > 0:
                for idx in indices.flatten():
                    x, y, w, h = candidate_boxes[idx]
                    cls_id = candidate_class_ids[idx]
                    cls_name = YOLO_ID_TO_NAME.get(cls_id, f"class_{cls_id}")
                    score = float(candidate_scores[idx])

                    # Determine severity
                    if cls_name in ["open", "short"]:
                        severity = "CRITICAL"
                    elif cls_name in ["mousebite", "spur", "copper"]:
                        severity = "MAJOR"
                    else:
                        severity = "MINOR"

                    det = DetectionBox(
                        class_id=cls_id,
                        class_name=cls_name,
                        score=round(score, 4),
                        bbox=[round(x, 1), round(y, 1), round(x + w, 1), round(y + h, 1)],
                        severity=severity,
                    )
                    detections.append(det)
                    class_breakdown[cls_name] = class_breakdown.get(cls_name, 0) + 1

        t3 = time.perf_counter()

        # 4. Optional Overlay Generation
        overlay_b64 = None
        if return_overlay:
            overlay_b64 = self._render_overlay(orig_img_bgr, detections)

        total_ms = (time.perf_counter() - t0) * 1000.0
        pre_ms = (t1 - t0) * 1000.0
        infer_ms = (t2 - t1) * 1000.0
        post_ms = (t3 - t2) * 1000.0

        is_defective = len(detections) > 0
        status = "FAIL" if is_defective else "PASS"

        return PredictionResponse(
            status=status,
            is_defective=is_defective,
            defect_count=len(detections),
            detections=detections,
            class_breakdown=class_breakdown,
            latency=LatencyBreakdown(
                preprocess_ms=round(pre_ms, 2),
                inference_ms=round(infer_ms, 2),
                postprocess_ms=round(post_ms, 2),
                total_ms=round(total_ms, 2),
            ),
            model_version=f"{self.model_path.name} ({self.precision})",
            overlay_base64=overlay_b64,
        )

    def _render_overlay(self, img_bgr: np.ndarray, detections: list[DetectionBox]) -> str:
        """Render industrial bounding boxes with labels and return base64 PNG."""
        canvas = img_bgr.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            rgb = DEFECT_COLORS_RGB.get(det.class_name, (0, 255, 0))
            bgr = (rgb[2], rgb[1], rgb[0])

            # Draw box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr, 2)

            # Label banner
            label = f"{det.class_name} {det.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_label_top = max(0, y1 - th - 6)
            cv2.rectangle(
                canvas,
                (x1, y_label_top),
                (x1 + tw + 6, y1),
                bgr,
                -1,
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Encode to PNG base64
        success, encoded = cv2.imencode(".png", canvas)
        if not success:
            return ""
        return base64.b64encode(encoded.tobytes()).decode("utf-8")
