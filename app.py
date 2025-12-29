import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Page configuration
st.set_page_config(page_title="YOLO Object Detection", page_icon="📦", layout="wide")

# Custom CSS for attractive interface
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #ffecd2, #fcb69f);
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #d35400;
        text-align: center;
        margin-bottom: 20px;
    }
    .upload-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stButton>button {
        background-color: #e67e22;
        color: white;
        font-size: 1rem;
        border-radius: 8px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📦 YOLO Object Detection App</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Upload an image and detect objects using YOLOv5</p>', unsafe_allow_html=True)

# Image upload
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

# Detect objects
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # YOLO inference
    results = model(np.array(image))
    result_img = np.squeeze(results.render())
    
    st.image(result_img, caption='Detected Objects', use_column_width=True)
