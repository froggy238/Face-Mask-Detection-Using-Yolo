import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("😷 Face Mask Detection System (Image + Video)")

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

# -----------------------------
# PROCESS FILE
# -----------------------------
if uploaded_file is not None:

    file_type = uploaded_file.type

    # -------------------------
    # IMAGE PROCESSING
    # -------------------------
    if "image" in file_type:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_container_width=True)

        img = np.array(image)

        results = model(img)
        result_img = results[0].plot()

        st.image(result_img, caption="Detection Result", use_container_width=True)

    # -------------------------
    # VIDEO PROCESSING
    # -------------------------
    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        st.info("Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Video processing completed ✅")