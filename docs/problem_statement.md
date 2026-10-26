# Tuyên bố bài toán: Hệ thống Kiểm định Lỗi Bo mạch PCB trong Công nghiệp (PCBDetect)

## 1. Bối cảnh Sản xuất & Tổng quan Dự án
Bo mạch in (Printed Circuit Board - PCB) là linh hồn và nền tảng của mọi thiết bị điện tử hiện đại—từ thiết bị tiêu dùng, bộ điều khiển khả trình PLC trong nhà máy, cho đến các hệ thống nhúng quan trọng như ECU ô tô và thiết bị hàng không vũ trụ. Trong các dây chuyền chế tạo mạch in thô (bare-board) và công nghệ dán bề mặt (SMT - Surface Mount Technology), các lỗi phát sinh trong quá trình ăn mòn hóa học (etching), khoan lỗ (drilling) và mạ đồng (plating) bắt buộc phải được phát hiện với tỷ lệ lọt lỗi gần như bằng 0.

Các hệ thống kiểm tra quang học tự động truyền thống (Automated Optical Inspection - AOI) thường dựa trên phương pháp so khớp mẫu (template matching) cứng nhắc và ngưỡng hình thái học (morphological thresholding). Trong sản xuất vi mạch mật độ cao hiện đại (High-Density Interconnect - HDI), các phương pháp cổ điển này bộc lộ những hạn chế nghiêm trọng:
1. **Tỷ lệ báo nhầm quá cao (Overkill / False Alarm > 20–30%)**: Sai lệch vi căn chỉnh cơ học (micro-alignment offset), dao động ánh sáng nhẹ hoặc lớp oxy hóa bề mặt dễ dàng kích hoạt các bộ lọc luật cứng, khiến bảng mạch hoàn toàn bình thường bị đánh trượt.
2. **Bỏ lọt các lỗi vi mô tinh vi**: Các vết đứt mạch siêu nhỏ (hairline opens), ngắn mạch chân tóc (micro-shorts) và lẹm cạnh mạch (mouse-bites) rất dễ thoát khỏi các kernel lọc thủ công.
3. **Gây quá tải cho trạm QC và nguy cơ mỏi mắt của công nhân**: Tỷ lệ báo nhầm cao làm nghẽn trạm kiểm định thủ công bằng kính hiển vi của công nhân QC, gián tiếp gia tăng xác suất con người mệt mỏi và vô tình bấm "PASS" cho một bảng mạch thực sự bị lỗi.

**PCBDetect** giải quyết bài toán này bằng cách kết hợp mô hình phát hiện vật thể có giám sát (YOLO / Faster R-CNN) làm nhánh chính, song song với mô hình phát hiện bất thường (PatchCore / PaDiM) làm nhánh đối sánh, nhằm đem lại khả năng kiểm định tự động với độ tin cậy cao và ưu tiên Recall tối đa.

---

## 2. Kinh tế học Chất lượng: Lỗi lọt (False Negative) vs. Báo nhầm (False Positive)

Trong sản xuất công nghiệp quy mô lớn, chi phí của các loại sai số kiểm định mang **tính chất bất đối xứng tuyệt đối**:

| Loại sai số | Thuật ngữ nhà máy | Hậu quả tức thì | Tác động dây chuyền hạ nguồn | Chi phí thiệt hại tương đối |
| :--- | :--- | :--- | :--- | :--- |
| **False Negative (FN)** | **Lỗi lọt (Escape Defect)** | Bảng mạch hỏng bị dán nhãn "PASS" và chuyển sang công đoạn tiếp theo | Được hàn thêm hàng trăm linh kiện/chip đắt tiền; gây chập cháy khi test xuất xưởng hoặc hỏng hóc trong tay khách hàng; thu hồi sản phẩm hàng loạt; kiện tụng bảo hành. | **100x – 1000x** ($50 – $10.000+) |
| **False Positive (FP)** | **Báo nhầm / Báo oan (Overkill / False Alarm)** | Bảng mạch chuẩn bị gắn cờ nghi ngờ lỗi | Kỹ thuật viên QC mất 5–10 giây soi lại dưới kính hiển vi tại trạm kiểm định; tốn một lượng nhỏ chu kỳ thời gian (cycle time). | **1x** ($0.10 – $0.50) |

```
                     MA TRẬN QUYẾT ĐỊNH KIỂM ĐỊNH & CHI PHÍ NHÀ MÁY
               
                                      TRẠNG THÁI THỰC TẾ
                               CÓ LỖI (+)              BÌNH THƯỜNG (-)
                         ┌──────────────────────┬──────────────────────┐
            DỰ ĐOÁN      │    True Positive     │    False Positive    │
            CÓ LỖI (+)   │   (Bắt trúng lỗi)    │      (Báo nhầm)      │
                         │ Chi phí: Quy trình QC│  Chi phí: ~$0.20 QC  │
      QUYẾT              ├──────────────────────┼──────────────────────┤
      ĐỊNH               │    False Negative    │    True Negative     │
      MÔ HÌNH            │      (LỌT LỖI!)      │      (Đạt chuẩn)     │
            DỰ ĐOÁN      │ Chi phí: $50-$10.000+│ Chi phí: Luồng chuẩn │
            SẠCH (-)     │  [RỦI RO TỬ HUYỆT]   │                      │
                         └──────────────────────┴──────────────────────┘
```

### Kết luận cho Thiết kế Kỹ thuật
- **Recall là ưu tiên số 1**: Mục tiêu tối thượng của mô hình là đạt **$\text{Recall} \ge 95\%$ (kỳ vọng $\ge 98\%$)** trên các lỗi chí mạng (đặc biệt là **Ngắn mạch / Short** và **Đứt mạch / Open**).
- **Precision quyết định năng suất dây chuyền**: Precision kiểm soát độ dài hàng đợi của trạm QC con người. Nếu Precision rơi xuống dưới $80\%$, lượng bo mạch chờ kiểm tra lại sẽ làm tắc nghẽn băng chuyền sản xuất.
- **Lựa chọn ngưỡng vận hành (Operating Point)**: Không bao giờ chọn ngưỡng confidence mặc định $0.50$ một cách mù quáng. Ngưỡng tự tin phải được quét và tinh chỉnh trực tiếp trên đường cong Precision-Recall (PR Curve) để thỏa mãn điều kiện Recall trước, sau đó tối ưu hóa Precision.

---

## 3. Phân loại 6 dạng lỗi trong DeepPCB

Bộ chuẩn DeepPCB phân loại 6 dạng khuyết tật phổ biến và nguy hiểm nhất trên bo mạch:

```
┌─────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Loại khuyết tật │ Cơ chế vật lý & Mối nguy hại đối với độ tin cậy bo mạch                │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. Đứt mạch     │ Đường đồng bị đứt gãy do ăn mòn quá mức hoặc vết xước cơ học. Làm hở   │
│    (Open)       │ mạch tín hiệu/nguồn; gây mất hoàn toàn chức năng của phân vùng mạch.   │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. Ngắn mạch    │ Cầu đồng ngoài ý muốn bắc ngang nối liền hai đường mạch độc lập. Nguy  │
│    (Short)      │ cơ quá dòng, sụt áp, khóa mạch nguồn (latch-up), chập cháy nổ linh kiện│
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. Lẹm mạch     │ Vết cắn/khuyết đồng dọc theo mép đường dẫn, làm thắt hẹp tiết diện.   │
│    (Mousebite)  │ Làm tăng điện trở cục bộ, tích nhiệt và dễ đứt ngầm do mỏi nhiệt.      │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 4. Gai đồng     │ Nhánh đồng nhọn chìa ra từ cạnh đường mạch nhưng chưa chạm mạch bên.   │
│    (Spur)       │ Làm suy yếu khe hở cách điện; nguy cơ phóng điện hồ quang cao áp.      │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 5. Đảo đồng sót │ Mảng đồng cô lập còn sót lại trên vùng cách điện do rửa hóa chất kém.  │
│    (Copper)     │ Gây điện dung ký sinh không mong muốn và rủi ro dính thiếc hàn khi SMT.│
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 6. Lỗ kim       │ Lỗ châm kim hoặc vết rỗ khí nhỏ trên đường mạch hoặc mặt phẳng mass.   │
│    (Pin Hole)   │ Làm giảm độ bám dính của lớp mạ và suy giảm công suất tải dòng điện.   │
└─────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Các thách thức kỹ thuật cốt lõi & Giải pháp thiết kế

1. **Thách thức phát hiện vật thể siêu nhỏ (COCO Small Regime)**:
   - Qua bước phân tích EDA, hơn **32% số lỗi thuộc nhóm Small** (diện tích $< 32 \times 32 = 1.024\text{ px}^2$) và **68% thuộc nhóm Medium**. Diện tích lỗi trung bình chỉ chiếm **0.35%** diện tích tấm ảnh $640 \times 640$.
   - Các mạng phát hiện vật thể tiêu chuẩn khi giảm mẫu quá sâu ($32\times$ downsampling) sẽ làm tiêu biến hoàn toàn tín hiệu của các lỗi 15–20 pixel trên feature map tầng P5.
   - *Giải pháp thiết kế*: Bảo toàn các tầng đặc trưng có độ phân giải cao (stride P2/P3), tích hợp Feature Pyramid Network (PANet/FPN) và áp dụng cơ chế Tiling / Slicing Window khi ảnh lớn.

2. **Chống rò rỉ dữ liệu qua phân chia theo cụm bảng mạch (Group-Disjoint Splits)**:
   - Các bo mạch trong nhà máy được in thành từng panel chứa nhiều mạch cùng kiểu dáng (board layout). Nếu cắt ảnh và chia train/val/test ngẫu nhiên theo ảnh, các mẩu ảnh con của cùng một bo mạch vật lý sẽ lọt vào cả tập train và test. Mô hình sẽ "học vẹt" đường nét vi mạch thay vì học bản chất của lỗi.
   - *Giải pháp thiết kế*: Thực hiện chia tập nghiêm ngặt theo nhóm bảng mạch vật lý (group/board ID). Đánh giá kiểm thử bắt buộc phải diễn ra trên các layout bo mạch mà mô hình chưa từng nhìn thấy trong lúc train.

3. **Vòng phản hồi có sự tham gia của con người (Human-in-the-loop Feedback)**:
   - Trong dây chuyền thực tế, nhân viên QC sẽ nhấn nút xác nhận lỗi hoặc hủy bỏ cảnh báo sai của hệ thống.
   - *Giải pháp thiết kế*: Kiến trúc serving (FastAPI + Streamlit/Gradio) phải tích hợp sẵn cơ chế ghi nhận feedback (đúng/sai) vào cơ sở dữ liệu để phục vụ đào tạo chủ động (Active Learning) và hard-example mining.

---

## 5. Các chỉ số hiệu năng mục tiêu (KPIs)

| Chỉ số (Metric) | Mức chấp nhận tối thiểu | Mức tối ưu kỳ vọng | Ý nghĩa vận hành công nghiệp |
| :--- | :--- | :--- | :--- |
| **Recall (Lỗi Open & Short)** | $\ge 96.0\%$ | $\ge 99.0\%$ | Tuyệt đối không để lỗi nguy hiểm lọt khỏi cổng nhà máy. |
| **mAP@0.5 toàn diện** | $\ge 93.0\%$ | $\ge 97.0\%$ | Độ chính xác định vị và phân loại đồng đều trên cả 6 lớp lỗi. |
| **Tỷ lệ báo nhầm (FAR / Overkill)** | $\le 8.0\%$ | $\le 3.0\%$ | Giữ bàn kiểm định của kỹ thuật viên QC thông thoáng, không gây ùn ứ băng chuyền. |
| **Độ trễ suy luận (Inference Latency)**| $\le 100\text{ ms}$ (CPU) | $\le 20\text{ ms}$ (Edge GPU) | Đảm bảo tốc độ quét đồng bộ theo thời gian thực (10–30 fps). |
| **Định dạng triển khai** | ONNX Runtime | ONNX FP16 / TensorRT | Chạy ổn định trên máy tính nhúng công nghiệp (Industrial IPC) không cần cài Python nặng nề. |
