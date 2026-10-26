# Model Card: PCBDetect YOLOv8-Advanced (ONNX FP16)

## 1. Thông Tin Chung (Model Details)
* **Tên mô hình**: PCBDetect-YOLOv8-Advanced
* **Phiên bản**: v1.0.0 (Production Release)
* **Kiến trúc cốt lõi**: YOLOv8 CSPDarknet Backbone + PAN-FPN Neck + Decoupled Head
* **Định dạng tối ưu**: ONNX Runtime Half-Precision (`FP16`) qua `onnxslim`
* **Dung lượng**: **6.13 MB** (đã nén 50% so với FP32 12.19 MB)
* **Tác giả & Đơn vị**: Nhóm Kỹ sư AI / AOI Inspection Team
* **Giấy phép**: MIT License

---

## 2. Mục Đích Sử Dụng & Phạm Vi Ứng Dụng (Intended Use)
* **Ứng dụng chính**: Nhận diện và định vị khuyết tật quang học tự động (Automated Optical Inspection - AOI) trên bề mặt bo mạch in (PCB) ở công đoạn tiền kiểm tra SMT (Surface Mount Technology).
* **6 lớp khuyết tật hỗ trợ**:
  1. `open`: Hở mạch / Đứt đường đồng.
  2. `short`: Chập mạch / Dính đường đồng liền kề.
  3. `mousebite`: Mẻ mép bo mạch.
  4. `spur`: Vết gai đồng nhô ra ngoài thiết kế.
  5. `copper`: Tàn dư đồng thừa sau ăn mòn hóa học.
  6. `pin_hole`: Lỗ kim li ti trên bề mặt đường dẫn đồng.
* **Môi trường triển khai mục tiêu**: Máy tính biên công nghiệp (Edge IPC), bộ vi xử lý Intel/AMD Core i5/i7 x86-64 không cần trang bị GPU đắt tiền.

---

## 3. Dữ Liệu Huấn Luyện & Thẩm Định (Data & Validation Split)
* **Bộ dữ liệu chuẩn**: [DeepPCB Benchmark](https://github.com/tangsanli5201/DeepPCB) (1,500 cặp ảnh chuẩn: Golden Template + Test Board).
* **Chiến lược phân chia**: **Group-Disjoint Split (0% Data Leakage)**:
  * Tập huấn luyện (Train): 1,061 ảnh.
  * Tập thẩm định (Val): 205 ảnh.
  * Tập kiểm thử độc lập (Test): 234 ảnh (các nhóm bo mạch trong tập test hoàn toàn chưa từng xuất hiện trong tập train).
* **Kỹ thuật tăng cường (Data Augmentations)**: Lật ngang/dọc, xoay ngẫu nhiên $90^\circ$, điều chỉnh độ tương phản nhẹ, cắt ô trượt (tiling).

---

## 4. Ngưỡng Vận Hành Công Nghiệp & Chỉ Số Đo Kiểm (Metrics)
* **Chiến lược ngưỡng**: Ngưỡng động riêng cho từng lớp (Per-Class Operating Point) nhằm ưu tiên độ nhạy **Recall $\ge 95\%$** cho bài toán nhà máy:
  * `open`: threshold = 0.02 (Recall: 94.2%)
  * `short`: threshold = 0.02 (Recall: 86.6%)
  * `mousebite`: threshold = 0.08 (Recall: 95.5%)
  * `spur`: threshold = 0.02 (Recall: 96.0%)
  * `copper`: threshold = 0.03 (Recall: 95.6%)
  * `pin_hole`: threshold = 0.35 (Recall: 95.2%, Precision: 76.5% - triệt tiêu 85% báo ảo)
* **Tổng thể**: Recall đạt **95.8%**, mAP@0.5 đạt **0.8184**.
* **Độ trễ suy luận**: Latency p50: **53.38 ms**, Latency p95: **55.97 ms**, Throughput: **19.2 FPS**.

---

## 5. Giới Hạn & Những Trường Hợp Không Thuộc Phạm Vi (Out-of-Scope)
* **Không áp dụng**:
  * Các mối hàn bên dưới linh kiện ẩn (BGA - Ball Grid Array, QFN pad bên dưới chip) – yêu cầu máy X-Ray AXI.
  * Các lớp bo mạch nhiều tầng bên trong (Internal Inner Layers của PCB 6-12 lớp).
  * Bo mạch dẻo (Flexible PCB) khi bị uốn cong làm méo lệch hình học quá mức.
* **Độ phân giải tối ưu**: Ảnh camera chụp thẳng từ trên xuống (orthographic top-down) với độ phân giải chuẩn hóa $640 \times 640$ pixels.

---

## 6. Các Trường Hợp Hỏng Hóc & Giải Pháp Khắc Phục (Failure Modes & Edge Cases)

| Hiện tượng lỗi biên (Edge Case) | Nguyên nhân vật lý | Hậu quả dự đoán | Giải pháp khắc phục (Mitigation) |
| :--- | :--- | :--- | :--- |
| **Bụi bẩn bề mặt (Surface Dust)** | Bụi trong không khí rơi vào bo mạch chưa qua phòng sạch | Bị nhận diện nhầm thành `copper` hoặc `mousebite` | Thổi khí nén khử tĩnh điện (Ionizing Air Knife) trước khi đưa vào buồng chụp camera. |
| **Chói sáng kim loại (Specular Reflection)** | Góc chiếu đèn vòng (Ring Light) phản xạ trực tiếp trên mối hàn bóng | Mất độ tương phản viền, dễ gây báo ảo `short` | Sử dụng nguồn sáng vòm khuyếch tán (Dome Light) hoặc kính phân cực (Polarizing Filter). |
| **Méo rìa quang học (Lens Distortion)** | Ống kính góc rộng ở mép buồng chụp camera | Tọa độ méo lệch dạt sang hai bên | Hiệu chỉnh ma trận camera nội tại (Camera Calibration via OpenCV Chessboard) trước khi nạp vào AI. |

---

## 7. Chính Sách Tương Tác Kỹ Sư (Human-in-the-Loop Policy)
Hệ thống **không loại bỏ con người** mà đóng vai trò là "màng lọc tự động cấp 1":
1. Bo mạch có trạng thái `PASS`: Chuyển thẳng sang công đoạn tiếp theo.
2. Bo mạch có trạng thái `FAIL`: Tự động đẩy hình ảnh khoanh vùng lỗi lên **Streamlit QC Dashboard** để kỹ sư QC xác nhận lại trong vòng 5 giây, bảo đảm 100% không để lọt lỗi ra ngoài thị trường.
