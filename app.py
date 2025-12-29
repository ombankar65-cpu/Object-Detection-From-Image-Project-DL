import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Page config
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🔍",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# Custom CSS for beauty
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #4CAF50;
}
.subtitle {
    text-align: center;
    color: #BBBBBB;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🔍 Object Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by YOLO & Streamlit</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)

uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    col1, col2 = st.columns(2)

    # Load image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

    # Run detection
    results = model.predict(image_np, conf=confidence)
    annotated_frame = results[0].plot()

    with col2:
        st.subheader("🎯 Detection Result")
        st.image(annotated_frame, use_container_width=True)

    # Detection summary
    st.markdown("### 🧾 Detection Summary")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        st.write(f"**{label}** — Confidence: `{conf:.2f}`")

else:
    st.info("👆 Upload an image to start detection")
