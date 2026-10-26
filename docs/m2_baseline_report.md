# Milestone 2: Báo cáo Đối chuẩn Baseline Cổ điển vs. Baseline Deep Learning

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
| **Độ chính xác (Precision)** | **6.61%** | **58.22%** | **+51.6%** (Giảm báo nhầm) |
| **Độ bao phủ lỗi (Recall - Ưu tiên SLA)** | **17.19%** | **63.28%** | **+46.1%** (Giảm lỗi lọt) |
| **F1-Score tổng hợp** | **9.55%** | **60.65%** | **+51.1%** |
| **mAP@0.5** | N/A (Không có score confidence) | **61.48%** | Mốc Deep Learning ban đầu |
| **Thời gian xử lý (Latency CPU)** | **2.32 ms/ảnh** | **49.71 ms/ảnh** | Đều đáp ứng thời gian thực dây chuyền |
| **Thông lượng (Throughput)** | **430.6 FPS** | **20.1 FPS** | Phù hợp cả máy tính công nghiệp CPU |
| **Khả năng chịu rung lắc / Lệch camera** | Nhạy cảm (Dễ sinh viền giả khi lệch) | Bền vững (Spatial Invariance cao) | Vượt trội khi triển khai thực tế |

---

## 3. Phân tích Chi tiết Độ bao phủ theo từng Lớp lỗi (Per-Class Recall)

| Loại lỗi vi mạch | Recall Cổ điển (%) | Recall YOLOv8n (%) | Nhận xét Cơ chế Nhận diện |
| :--- | :---: | :---: | :--- |
| **1. open (Đứt mạch)** | 12.26% | 52.14% | Cổ điển bắt được khi vết hở đủ rộng; YOLO nắm bắt được ngữ cảnh 2 đầu dây mạch. |
| **2. short (Ngắn mạch)** | 13.45% | 41.18% | Vết chập bắc ngang tạo mảng vi sai rõ rệt trên cả 2 phương pháp. |
| **3. mousebite (Lẹm mạch)** | 4.92% | 78.03% | Lỗi cực nhỏ ở mép đường đồng; cổ điển dễ nhầm với viền răng cưa do rung ảnh. |
| **4. spur (Gai đồng)** | 3.96% | 52.86% | Gai nhọn chìa ra; cổ điển dễ mất nếu kernel lọc nhiễu quá to. |
| **5. copper (Đảo đồng sót)** | 54.63% | 77.0% | Mảng đồng cô lập; cả 2 phương pháp đều nhận diện tốt. |
| **6. pin_hole (Lỗ kim)** | 17.9% | 78.46% | Rỗ khí đường kính nhỏ (< 20px); thách thức lớn nhất của cả 2 hệ thống. |

---

## 4. Đúc kết Kỹ thuật & Định hướng cho Milestone 3 (M3)
1. **Tại sao phương pháp Cổ điển không thể thay thế Deep Learning trong nhà máy hiện đại?**
   - Mặc dù tốc độ của OpenCV rất nhanh (~2.32 ms), phương pháp trừ ảnh **hoàn toàn mù màu về mặt ngữ nghĩa** (không thể phân biệt lỗi nào nguy hiểm như `short/open` để dừng chuyền ngay, lỗi nào nhẹ như `copper` để bỏ qua).
   - Khi bo mạch bị xê dịch nhẹ hoặc ánh sáng bóng của lớp mạ đồng thay đổi, tỷ lệ báo nhầm (False Alarm) của Cổ điển tăng vọt.
2. **Kế hoạch bứt phá ở Milestone 3**:
   - Tối ưu hóa YOLO (thử nghiệm YOLOv8s, YOLOv8m) và Faster R-CNN (với backbone ResNet-50 + FPN).
   - Tích hợp kỹ thuật giữ tầng đặc trưng P2/P3 để đẩy mạnh Recall trên nhóm lỗi lỗ kim (`pin_hole`) và lẹm cạnh (`mousebite`) lên trên mức mục tiêu SLA $\ge 95\%$.
