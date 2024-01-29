import streamlit as st
from PIL import Image
import io
import torch

st.set_page_config(layout="wide")

# Confidence threshold for model prediction
confidence_threshold = 0.25

# Function to load the YOLOv8 model
@st.experimental_singleton
def load_model():
    # Replace the next line with the path to your YOLOv8 model
    model_path = 'models/last.pt'
    model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path, force_reload=True)
    model.conf = confidence_threshold
    return model

# Function to perform inference on an image
def infer_image(image, model):
    results = model(image)
    results.render()  # Update results.imgs with boxes and labels
    for img in results.imgs:
        img_byte_arr = io.BytesIO()
        im = Image.fromarray(img)
        im.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()

# Load the YOLOv8 model
model = load_model()

# Title of the application
st.title("YOLOv8 Object Detection")

# Sidebar for uploading images
st.sidebar.title("Upload Image")

# File uploader allows user to add their own image
uploaded_file = st.sidebar.file_uploader("Drag or Upload Image", type=['png', 'jpg', 'jpeg'])

# Main panel for displaying images
col1, col2 = st.columns(2)
with col1:
    st.header("Original Image")
    if uploaded_file is not None:
        # Display the original image
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, use_column_width=True)

with col2:
    st.header("Processed Image")
    if uploaded_file is not None and st.sidebar.button("Process Image"):
        # Perform inference
        results = infer_image(uploaded_image, model)
        # Display the processed image
        st.image(results, use_column_width=True, caption="Processed Image with YOLOv8")
