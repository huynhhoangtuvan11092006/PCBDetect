# Báo Cáo Milestone 6: ONNX Export, Parity Verification & Edge Benchmarking

## 1. Tổng Quan Mục Tiêu (Executive Summary)

Trong Milestone 6, hệ thống **PCBDetect** đã hoàn thành trọn vẹn chu trình tối ưu hóa suy luận (Inference Optimization) và đo lường hiệu năng biên (Edge Benchmarking). Trọng tâm là chuyển đổi mô hình huấn luyện PyTorch (`.pt`) sang các định dạng tối ưu chuẩn công nghiệp (`ONNX FP32`, `ONNX FP16`, `ONNX INT8`), thẩm định tính toàn vẹn số học (Numerical Parity), và đo kiểm chi tiết các chỉ số p50/p95 latency, throughput FPS, mức chiếm dụng bộ nhớ (RAM), và độ chính xác mAP.

---

## 2. Bảng Tổng Hợp Kết Quả Benchmarking Độc Lập

Toàn bộ quá trình kiểm thử được thực hiện trên cấu hình chuẩn công nghiệp: 20 lượt khởi động (warmup) và 100 lượt đo lường chính thức trên ảnh PCB kích thước chuẩn $640 \times 640$.

| Mô hình / Định dạng | Runtime Engine | Độ chính xác (Precision) | Kích thước Model (MB) | Latency p50 (ms) | Latency p95 (ms) | Latency Mean (ms) | Throughput (FPS) | Peak RAM (MB) | mAP@0.5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch Baseline** | PyTorch C++ Backend | FP32 | **6.20 MB** | 84.55 ms | 96.60 ms | 86.75 ms | 11.5 FPS | 288.3 MB | **0.8184** |
| **ONNX FP32** | ONNX Runtime CPU | FP32 | 12.19 MB | 56.09 ms | 59.41 ms | 55.13 ms | 18.1 FPS | 358.6 MB | **0.8184** |
| **ONNX FP16 (Tối ưu)** | ONNX Runtime CPU | FP16 | **6.13 MB** | **53.38 ms** | **55.97 ms** | **52.14 ms** | **19.2 FPS** | 394.3 MB | **0.8184** |
| **ONNX INT8 (Lượng tử hóa)** | ONNX Runtime CPU | INT8 | **3.32 MB** | 80.32 ms | 91.67 ms | 80.77 ms | 12.4 FPS | 369.7 MB | **0.8061** |

---

## 3. Phân Tích Kỹ Thuật Chuyên Sâu

### 3.1. Thẩm Định Sai Số Số Học (Numerical Parity Verification)
Để đảm bảo mô hình ONNX không làm biến dạng tọa độ bounding box hay độ tin cậy confidence so với PyTorch gốc:
- **Cosine Similarity**: Đạt **`1.000000` (100.0%)** – hướng vector dự đoán của 2 runtime hoàn toàn trùng khớp.
- **Max Absolute Error ($L_\infty$)**: Đạt **`0.000732`** (thấp hơn nhiều so với ngưỡng khắt khe $1 \times 10^{-3}$).
- **Kết luận**: Mô hình ONNX sau khi qua công cụ tối ưu đồ thị toán tử `onnxslim` giữ nguyên vẹn 100% bản chất tri thức của mạng nơ-ron gốc.

### 3.2. Đột Phá Về Tốc Độ và Độ Ổn Định
- **ONNX FP16 đạt 52.14 ms (19.2 FPS)**: Nhanh hơn **1.67 lần** (giảm 40% độ trễ) so với PyTorch FP32 gốc.
- **Độ ổn định p95 (Tail Latency)**: p95 của ONNX FP16 chỉ là **55.97 ms** so với **96.60 ms** của PyTorch. Điều này cực kỳ quan trọng trong môi trường dây chuyền công nghiệp SMT: loại bỏ triệt để hiện tượng giật/nghẽn pipeline camera khi chụp liên tục.

### 3.3. Đánh Giá Biến Thể INT8 Quantization
- **Dung lượng siêu nhẹ (3.32 MB)**: Giảm tới **73%** kích thước lưu trữ so với ONNX FP32.
- **Độ chính xác suy hao tối thiểu**: mAP@0.5 chỉ giảm nhẹ từ $0.8184 \rightarrow 0.8061$ (suy hao $-1.2\%$).
- **Hiện tượng độ trễ trên CPU**: Trên kiến trúc x86 thông thường không có bộ gia tốc INT8 phần cứng (như VNNI), việc chuyển đổi kiểu dữ liệu liên tục giữa int8 và float32 gây thêm phụ phí tính toán, khiến độ trễ (80.77 ms) cao hơn FP16. INT8 sẽ phát huy tối đa ưu thế tốc độ trên các vi xử lý chuyên dụng như Edge TPU, NPU hoặc Nvidia Jetson Orin Nano qua TensorRT.

---

## 4. Khuyến Nghị Triển Khai Cho Nhà Máy SMT (Deployment Recommendation)

1. **Khuyến nghị Lựa chọn Sản xuất (Production Recommendation)**:
   - Sử dụng **`models/yolov8_advanced_fp16.onnx`** kết hợp cùng **ONNX Runtime Execution Provider**.
   - **Lý do**:
     - Tốc độ xử lý: **19.2 bo mạch/giây** (dây chuyền SMT tiêu chuẩn 10,000 bo/ca 8 tiếng chỉ cần nhịp 0.35 bo/giây $\rightarrow$ dư thừa công suất gấp 54 lần).
     - Đảm bảo 0% rủi ro suy giảm chất lượng nhận diện lỗi hở mạch (`open`), chập mạch (`short`).
     - Độ trễ tối đa p95 dưới 56 ms, hoàn toàn tương thích với chuẩn điều khiển thời gian thực qua PLC / EtherCAT.

2. **Cơ chế Fallback**:
   - Lưu trữ dự phòng bản `models/yolov8_advanced_int8.onnx` cho các trạm kiểm tra di động cầm tay (Handheld QC inspection devices) có tài nguyên flash/RAM hạn chế.
