import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLO Object Detection", layout="centered")

st.title("📦 YOLO Object Detection App")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    img_array = np.array(image)
    results = model(img_array)

    st.image(results[0].plot(), caption="Detected Objects", use_column_width=True)










import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="📦",
    layout="centered"
)

st.title("📦 YOLO Object Detection App")
st.write("Upload an image and detect objects using YOLO")

# ---------------- Load YOLO Model ----------------
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# ---------------- Image Upload ----------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Convert image to numpy array
    img_array = np.array(image)

    # YOLO Detection
    results = model(img_array)

    # Display Result
    st.subheader("Detected Objects")
    detected_img = results[0].plot()
    st.image(detected_img, use_column_width=True)
