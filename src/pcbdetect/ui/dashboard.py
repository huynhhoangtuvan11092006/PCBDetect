"""Streamlit QC Inspection Dashboard for PCBDetect AOI System."""

import base64
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="PCBDetect AOI - Giám Sát Chất Lượng SMT",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
    }
    .status-pass {
        background-color: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        font-size: 1.5rem;
        padding: 8px 16px;
        border-radius: 6px;
        text-align: center;
        border: 1px solid #86EFAC;
    }
    .status-fail {
        background-color: #FEE2E2;
        color: #B91C1C;
        font-weight: 700;
        font-size: 1.5rem;
        padding: 8px 16px;
        border-radius: 6px;
        text-align: center;
        border: 1px solid #FCA5A5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
st.sidebar.title("⚙️ Cấu Hình Hệ Thống")

api_url = st.sidebar.text_input("FastAPI Server URL", value="http://localhost:8000")

# Check API Health
health_data = None
try:
    resp = requests.get(f"{api_url}/health", timeout=1.5)
    if resp.status_code == 200:
        health_data = resp.json()
        st.sidebar.success("🟢 Kết nối API: Sẵn sàng")
        st.sidebar.markdown(f"**Model:** `{health_data['model_name']}`")
        st.sidebar.markdown(f"**Precision:** `{health_data['model_precision']}`")
        st.sidebar.markdown(f"**Engine:** `{health_data['runtime_engine']}`")
        st.sidebar.markdown(f"**RAM:** `{health_data['ram_usage_mb']} MB`")
    else:
        st.sidebar.warning("🟡 API phản hồi bất thường")
except Exception:
    st.sidebar.error("🔴 Không kết nối được FastAPI (Chế độ Fallback nạp model cục bộ)")

# Defect Color Legend
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Bảng Màu Khuyết Tật")
st.sidebar.markdown(
    """
    - <span style='color:#E63946; font-weight:bold;'>■ Hở mạch (open)</span>
    - <span style='color:#457B9D; font-weight:bold;'>■ Chập mạch (short)</span>
    - <span style='color:#F4A261; font-weight:bold;'>■ Vết cắn mép (mousebite)</span>
    - <span style='color:#2A9D8F; font-weight:bold;'>■ Vết gai đồng (spur)</span>
    - <span style='color:#9B5DE5; font-weight:bold;'>■ Đồng thừa (copper)</span>
    - <span style='color:#F15BB5; font-weight:bold;'>■ Lỗ kim (pin_hole)</span>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("<div class='main-header'>🔬 PCBDetect – Hệ Thống AOI Giám Sát Chất Lượng Bo Mạch SMT</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Hệ thống phát hiện khuyết tật bề mặt PCB thời gian thực dựa trên ONNX Runtime FP16 & Per-Class Dynamic Thresholds.</div>",
    unsafe_allow_html=True,
)

# Sample Images Discovery
sample_dir = Path("data/processed/yolo/images/test")
sample_files = []
if sample_dir.exists():
    sample_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))

# Input Selection
col_input1, col_input2 = st.columns([1, 1])

with col_input1:
    uploaded_file = st.file_uploader("📤 Tải ảnh bo mạch cần soi (JPG / PNG):", type=["jpg", "jpeg", "png"])

with col_input2:
    selected_sample = None
    if sample_files:
        sample_options = ["(Chọn ảnh mẫu thử nghiệm)"] + [f.name for f in sample_files[:15]]
        chosen = st.selectbox("🚀 Hoặc chọn nhanh ảnh mẫu có sẵn từ tập Test:", options=sample_options)
        if chosen != "(Chọn ảnh mẫu thử nghiệm)":
            selected_sample = sample_dir / chosen

# Read Image Bytes
img_bytes = None
image_name = "test_pcb.jpg"

if uploaded_file is not None:
    img_bytes = uploaded_file.getvalue()
    image_name = uploaded_file.name
elif selected_sample is not None and selected_sample.exists():
    with open(selected_sample, "rb") as f:
        img_bytes = f.read()
    image_name = selected_sample.name

# Inference Execution
if img_bytes is not None:
    pred_result = None
    error_msg = None

    with st.spinner("Đang chạy mô hình phân tích AOI..."):
        if health_data:
            # Call FastAPI
            try:
                files = {"file": (image_name, img_bytes, "image/jpeg")}
                res = requests.post(f"{api_url}/predict", files=files, timeout=10.0)
                if res.status_code == 200:
                    pred_result = res.json()
                else:
                    error_msg = f"API Error {res.status_code}: {res.text}"
            except Exception as e:
                error_msg = f"Lỗi gọi API: {e}"
        else:
            # Fallback directly to InferenceService
            try:
                from pcbdetect.serving.inference_service import InferenceService

                if "local_service" not in st.session_state:
                    st.session_state.local_service = InferenceService()
                svc = st.session_state.local_service
                pred_obj = svc.predict(img_bytes)
                pred_result = pred_obj.model_dump()
            except Exception as e:
                error_msg = f"Lỗi nạp mô hình cục bộ: {e}"

    if error_msg:
        st.error(error_msg)
    elif pred_result:
        # Display Results
        st.markdown("---")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            if pred_result["status"] == "PASS":
                st.markdown("<div class='status-pass'>✅ PASS (ĐẠT)</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='status-fail'>❌ FAIL (LỖI)</div>", unsafe_allow_html=True)

        with kpi_col2:
            st.metric("Tổng Số Khuyết Tật", f"{pred_result['defect_count']} lỗi")

        with kpi_col3:
            total_lat = pred_result["latency"]["total_ms"]
            st.metric("Thời Gian Xử Lý", f"{total_lat:.1f} ms")

        with kpi_col4:
            fps = 1000.0 / total_lat if total_lat > 0 else 0
            st.metric("Tốc Độ (Throughput)", f"{fps:.1f} FPS")

        # Side-by-side Images
        st.markdown("### 🖼️ Đối Soát Trực Quan Bo Mạch")
        img_col1, img_col2 = st.columns(2)

        with img_col1:
            st.subheader("Ảnh Gốc Đầu Vào")
            orig_pil = Image.open(BytesIO(img_bytes))
            st.image(orig_pil, use_container_width=True)

        with img_col2:
            st.subheader("Kết Quả Phân Tích Lỗi AI")
            overlay_b64 = pred_result.get("overlay_base64")
            if overlay_b64:
                overlay_bytes = base64.b64decode(overlay_b64)
                overlay_pil = Image.open(BytesIO(overlay_bytes))
                st.image(overlay_pil, use_container_width=True)
            else:
                st.image(orig_pil, use_container_width=True)

        # Defect Breakdown Details
        st.markdown("### 📊 Phân Rã Chi Tiết Khuyết Tật")
        det_col1, det_col2 = st.columns([1, 2])

        with det_col1:
            st.markdown("**Số Lượng Từng Lớp Lỗi:**")
            breakdown = pred_result.get("class_breakdown", {})
            for cls_name, count in breakdown.items():
                if count > 0:
                    st.write(f"- **{cls_name}**: {count} lỗi")
            if pred_result["defect_count"] == 0:
                st.write("✨ Không phát hiện bất kỳ khuyết tật nào trên bề mặt bo mạch.")

        with det_col2:
            detections = pred_result.get("detections", [])
            if detections:
                st.markdown("**Danh Sách Tọa Độ Bounding Box:**")
                table_data = []
                for idx, d in enumerate(detections, 1):
                    table_data.append(
                        {
                            "STT": idx,
                            "Loại Khuyết Tật": d["class_name"],
                            "Độ Tin Cậy": f"{d['score'] * 100:.1f}%",
                            "Mức Độ": d["severity"],
                            "Tọa Độ [xmin, ymin, xmax, ymax]": str(d["bbox"]),
                        }
                    )
                st.dataframe(table_data, use_container_width=True)

        # Human-in-the-Loop QC Feedback
        st.markdown("---")
        st.markdown("### 👨‍🏭 Đánh Giá Của Kỹ Sư QC Ca Trực (Human-in-the-Loop)")

        with st.form("qc_feedback_form"):
            eval_choice = st.radio(
                "Đánh giá độ chính xác của AI trên bo mạch này:",
                options=["✅ AI Dự đoán Hoàn toàn Đúng", "❌ Báo Ảo / Dự đoán Nhầm"],
                horizontal=True,
            )
            inspector_notes = st.text_input("Ghi chú của Kỹ sư QC (tùy chọn):", placeholder="Ví dụ: Lỗi pin_hole bị nhiễu do bụi bề mặt...")
            submit_btn = st.form_submit_button("📩 Gửi Phản Hồi Về Hệ Thống")

            if submit_btn:
                is_correct = "Đúng" in eval_choice
                payload = {
                    "image_id": image_name,
                    "predicted_defect_count": pred_result["defect_count"],
                    "is_correct": is_correct,
                    "inspector_notes": inspector_notes,
                }
                if health_data:
                    try:
                        fb_resp = requests.post(f"{api_url}/feedback", json=payload, timeout=3.0)
                        if fb_resp.status_code == 200:
                            st.success("✅ Đã ghi nhận phản hồi QC thành công! Dữ liệu sẽ được lưu trữ để huấn luyện lại mô hình.")
                        else:
                            st.warning(f"Lỗi gửi phản hồi: {fb_resp.text}")
                    except Exception as e:
                        st.error(f"Không thể gửi phản hồi tới server: {e}")
                else:
                    st.info("Chế độ Offline: Đã ghi nhận tương tác cục bộ.")
else:
    st.info("👆 Vui lòng tải lên một ảnh bo mạch PCB hoặc chọn ảnh mẫu từ danh sách trên để bắt đầu soi lỗi.")
