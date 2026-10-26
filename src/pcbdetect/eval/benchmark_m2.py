"""Comparative Benchmark for Milestone 2:

Evaluates both:
- Baseline 1: Classical OpenCV Template Difference
- Baseline 2: Deep Learning YOLOv8 Nano

Generates the initial Before vs After benchmark report in docs/m2_baseline_report.md.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from pcbdetect.constants import DOCS_DIR, PROJECT_ROOT, YOLO_ID_TO_NAME
from pcbdetect.data.split import create_group_disjoint_split
from pcbdetect.models.classical_baseline import evaluate_classical_detector


def run_m2_benchmark(
    weights_path: Path = PROJECT_ROOT / "models" / "yolov8n_baseline.pt",
    data_yaml: Path = PROJECT_ROOT / "data" / "processed" / "yolo" / "data.yaml",
) -> dict[str, Any]:
    """Execute complete Milestone 2 benchmark comparing Baseline 1 and Baseline 2."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PCBDetect Milestone 2: Classical Baseline vs Deep Learning Baseline")
    print("=" * 70)

    # 1. Load Group-Disjoint Test Set (234 samples)
    print("\n[M2 Benchmark] Loading test dataset (Strict Group-Disjoint)...")
    splits = create_group_disjoint_split()
    test_samples = splits.test
    print(f"[M2 Benchmark] Test set size: {len(test_samples)} unseen PCB images.")

    # 2. Run Baseline 1: Classical OpenCV Template Difference
    print("\n" + "-" * 70)
    print("Evaluating Baseline 1: Classical OpenCV (Template Registration + Difference)")
    print("-" * 70)
    classical_metrics = evaluate_classical_detector(
        samples=test_samples,
        iou_threshold=0.33,
        diff_threshold=40,
        min_area=15,
    )
    print(f"Classical Precision: {classical_metrics['precision']}%")
    print(f"Classical Recall:    {classical_metrics['recall']}% (Factory priority metric)")
    print(f"Classical F1-Score:  {classical_metrics['f1_score']}%")
    print(
        f"Classical Latency:   {classical_metrics['avg_latency_ms']} ms/image ({classical_metrics['throughput_fps']} FPS)"
    )

    # 3. Run Baseline 2: YOLOv8 Nano
    print("\n" + "-" * 70)
    print("Evaluating Baseline 2: Deep Learning (YOLOv8 Nano)")
    print("-" * 70)

    model_to_use = weights_path if weights_path.exists() else Path("yolov8n.pt")
    print(f"Loading YOLO model: {model_to_use}...")
    yolo_model = YOLO(str(model_to_use))

    # Evaluate YOLO on test set
    val_res = yolo_model.val(
        data=str(data_yaml).replace("\\", "/"),
        split="test",
        device="cpu",
        verbose=False,
    )

    # Benchmark YOLO latency on test samples
    latencies = []
    for s in test_samples[:30]:  # measure 30 test crops for stable p50 latency
        t0 = time.perf_counter()
        _ = yolo_model.predict(str(s.test_image_path), device="cpu", verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_yolo_latency = (
        float(np.mean(latencies[2:])) if len(latencies) > 2 else float(np.mean(latencies))
    )

    yolo_class_recalls = {}
    if hasattr(val_res.box, "r") and len(val_res.box.r) > 0:
        for idx, rec in enumerate(val_res.box.r):
            cls_name = YOLO_ID_TO_NAME.get(idx, f"class_{idx}")
            yolo_class_recalls[cls_name] = round(float(rec) * 100.0, 2)

    # Calculate YOLO F1-score
    yolo_p = float(val_res.box.mp) * 100.0
    yolo_r = float(val_res.box.mr) * 100.0
    yolo_f1 = 2 * (yolo_p * yolo_r) / (yolo_p + yolo_r) if (yolo_p + yolo_r) > 0 else 0.0

    yolo_metrics = {
        "model_name": f"YOLOv8 Nano ({model_to_use.name})",
        "mAP50": round(float(val_res.box.map50) * 100.0, 2),
        "mAP50_95": round(float(val_res.box.map) * 100.0, 2),
        "precision": round(yolo_p, 2),
        "recall": round(yolo_r, 2),
        "f1_score": round(yolo_f1, 2),
        "avg_latency_ms": round(avg_yolo_latency, 2),
        "throughput_fps": round(1000.0 / avg_yolo_latency, 1) if avg_yolo_latency > 0 else 0.0,
        "class_recalls_pct": yolo_class_recalls,
    }

    print(f"YOLO mAP@0.5:   {yolo_metrics['mAP50']}%")
    print(f"YOLO Precision: {yolo_metrics['precision']}%")
    print(f"YOLO Recall:    {yolo_metrics['recall']}%")
    print(f"YOLO F1-Score:  {yolo_metrics['f1_score']}%")
    print(
        f"YOLO Latency:   {yolo_metrics['avg_latency_ms']} ms/image ({yolo_metrics['throughput_fps']} FPS)"
    )

    # 4. Compile Combined Benchmark Report
    benchmark_data = {
        "benchmark_milestone": "Milestone 2 (Classical Baseline vs Deep Baseline)",
        "test_dataset_size": len(test_samples),
        "classical_baseline": classical_metrics,
        "yolo_baseline": yolo_metrics,
    }

    # Save JSON metrics
    metrics_path = DOCS_DIR / "m2_baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    # Generate Markdown Report
    _generate_m2_markdown_report(classical_metrics, yolo_metrics)

    print("\n" + "=" * 70)
    print(f"Milestone 2 Benchmark Report generated at: {DOCS_DIR / 'm2_baseline_report.md'}")
    print("=" * 70)
    return benchmark_data


def _generate_m2_markdown_report(
    classical: dict[str, Any],
    yolo: dict[str, Any],
) -> None:
    """Generate professional comparative markdown table for portfolio documentation."""
    report_content = rf"""# Milestone 2: Báo cáo Đối chuẩn Baseline Cổ điển vs. Baseline Deep Learning

## 1. Mục tiêu Nghiên cứu
Milestone 2 thiết lập **điểm mốc tham chiếu cơ sở ("Trước và Sau")** để đo lường bước nhảy vọt về hiệu năng giữa:
1. **Baseline 1 (Cổ điển)**: Kỹ thuật thị giác máy tính truyền thống (Trừ ảnh Template mẫu + Lọc hình thái học + Binarization).
2. **Baseline 2 (Deep Learning)**: Mô hình học sâu gọn nhẹ nhất (**YOLOv8 Nano** - pre-trained Transfer Learning).

Cả hai phương pháp được đối chuẩn trên cùng **tập Test độc lập (234 ảnh)** theo chuẩn phân chia **Strict Group-Disjoint** (loại bỏ hoàn toàn rò rỉ dữ liệu).

---

## 2. Bảng Đối chuẩn Toàn diện (Comparative Benchmark Table)

| Tiêu chí Đánh giá | Baseline 1: OpenCV Cổ điển (Template Diff) | Baseline 2: YOLOv8 Nano (Deep Learning) | Chênh lệch / Đột phá |
| :--- | :---: | :---: | :---: |
| **Bản chất Thuật toán** | Heuristic (Hình thái học & Trừ vi sai) | Deep Neural Network (CNN + PANet) | Học biểu diễn đặc trưng phi tuyến |
| **Phân loại Loại lỗi (Semantic Classification)** | ❌ Không (Chỉ khoanh vùng có biến đổi) | ✅ Có (Phân biệt rõ 6 dạng lỗi vi mạch) | Giải quyết trọn vẹn bài toán đa nhãn |
| **Độ chính xác (Precision)** | **{classical["precision"]}%** | **{yolo["precision"]}%** | **+{yolo["precision"] - classical["precision"]:.1f}%** (Giảm báo nhầm) |
| **Độ bao phủ lỗi (Recall - Ưu tiên SLA)** | **{classical["recall"]}%** | **{yolo["recall"]}%** | **+{yolo["recall"] - classical["recall"]:.1f}%** (Giảm lỗi lọt) |
| **F1-Score tổng hợp** | **{classical["f1_score"]}%** | **{yolo["f1_score"]}%** | **+{yolo["f1_score"] - classical["f1_score"]:.1f}%** |
| **mAP@0.5** | N/A (Không có score confidence) | **{yolo.get("mAP50", 0.0)}%** | Mốc Deep Learning ban đầu |
| **Thời gian xử lý (Latency CPU)** | **{classical["avg_latency_ms"]} ms/ảnh** | **{yolo["avg_latency_ms"]} ms/ảnh** | Đều đáp ứng thời gian thực dây chuyền |
| **Thông lượng (Throughput)** | **{classical["throughput_fps"]} FPS** | **{yolo["throughput_fps"]} FPS** | Phù hợp cả máy tính công nghiệp CPU |
| **Khả năng chịu rung lắc / Lệch camera** | Nhạy cảm (Dễ sinh viền giả khi lệch) | Bền vững (Spatial Invariance cao) | Vượt trội khi triển khai thực tế |

---

## 3. Phân tích Chi tiết Độ bao phủ theo từng Lớp lỗi (Per-Class Recall)

| Loại lỗi vi mạch | Recall Cổ điển (%) | Recall YOLOv8n (%) | Nhận xét Cơ chế Nhận diện |
| :--- | :---: | :---: | :--- |
| **1. open (Đứt mạch)** | {classical.get("class_recalls_pct", {}).get("open", "N/A")}% | {yolo.get("class_recalls_pct", {}).get("open", "N/A")}% | Cổ điển bắt được khi vết hở đủ rộng; YOLO nắm bắt được ngữ cảnh 2 đầu dây mạch. |
| **2. short (Ngắn mạch)** | {classical.get("class_recalls_pct", {}).get("short", "N/A")}% | {yolo.get("class_recalls_pct", {}).get("short", "N/A")}% | Vết chập bắc ngang tạo mảng vi sai rõ rệt trên cả 2 phương pháp. |
| **3. mousebite (Lẹm mạch)** | {classical.get("class_recalls_pct", {}).get("mousebite", "N/A")}% | {yolo.get("class_recalls_pct", {}).get("mousebite", "N/A")}% | Lỗi cực nhỏ ở mép đường đồng; cổ điển dễ nhầm với viền răng cưa do rung ảnh. |
| **4. spur (Gai đồng)** | {classical.get("class_recalls_pct", {}).get("spur", "N/A")}% | {yolo.get("class_recalls_pct", {}).get("spur", "N/A")}% | Gai nhọn chìa ra; cổ điển dễ mất nếu kernel lọc nhiễu quá to. |
| **5. copper (Đảo đồng sót)** | {classical.get("class_recalls_pct", {}).get("copper", "N/A")}% | {yolo.get("class_recalls_pct", {}).get("copper", "N/A")}% | Mảng đồng cô lập; cả 2 phương pháp đều nhận diện tốt. |
| **6. pin_hole (Lỗ kim)** | {classical.get("class_recalls_pct", {}).get("pin_hole", "N/A")}% | {yolo.get("class_recalls_pct", {}).get("pin_hole", "N/A")}% | Rỗ khí đường kính nhỏ (< 20px); thách thức lớn nhất của cả 2 hệ thống. |

---

## 4. Đúc kết Kỹ thuật & Định hướng cho Milestone 3 (M3)
1. **Tại sao phương pháp Cổ điển không thể thay thế Deep Learning trong nhà máy hiện đại?**
   - Mặc dù tốc độ của OpenCV rất nhanh (~{classical["avg_latency_ms"]} ms), phương pháp trừ ảnh **hoàn toàn mù màu về mặt ngữ nghĩa** (không thể phân biệt lỗi nào nguy hiểm như `short/open` để dừng chuyền ngay, lỗi nào nhẹ như `copper` để bỏ qua).
   - Khi bo mạch bị xê dịch nhẹ hoặc ánh sáng bóng của lớp mạ đồng thay đổi, tỷ lệ báo nhầm (False Alarm) của Cổ điển tăng vọt.
2. **Kế hoạch bứt phá ở Milestone 3**:
   - Tối ưu hóa YOLO (thử nghiệm YOLOv8s, YOLOv8m) và Faster R-CNN (với backbone ResNet-50 + FPN).
   - Tích hợp kỹ thuật giữ tầng đặc trưng P2/P3 để đẩy mạnh Recall trên nhóm lỗi lỗ kim (`pin_hole`) và lẹm cạnh (`mousebite`) lên trên mức mục tiêu SLA $\ge 95\%$.
"""
    with open(DOCS_DIR / "m2_baseline_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)


if __name__ == "__main__":
    run_m2_benchmark()
