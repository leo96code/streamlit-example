import streamlit as st
from roboflow import Roboflow
import requests
from PIL import Image
from io import BytesIO
import base64
# Initialize Roboflow
rf = Roboflow(api_key="AtAN7fsWbxIN9Moql1gJ")
project = rf.workspace().project("mb-yellow-mosaic")
model = project.version("3").model

st.title('Image Processing with Roboflow')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def display_segmentation(original, mask, class_map):
    # This function is a placeholder.
    # You need to implement the actual overlay or side-by-side display logic.
    st.image(original, caption='Original Image', use_column_width=True)
    st.image(mask, caption='Segmentation Mask', use_column_width=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Process Image'):
        try:
            img_bytes = uploaded_file.getvalue()

            # Perform inference
            response = model.predict(img_bytes, confidence=40).json()
            
            # Process the response
            segmentation_mask = response['segmentation_mask']
            decoded = base64.b64decode(segmentation_mask)
            mask_image = Image.open(BytesIO(decoded))

            class_map = response['class_map']
            # Here, you can map class IDs to class names using class_map if needed

            # Display the original image and segmentation mask
            display_segmentation(image, mask_image, class_map)

        except Exception as e:
            st.error(f"An error occurred: {e}")
