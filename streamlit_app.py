import streamlit as st
from roboflow import Roboflow
import requests
from PIL import Image
from io import BytesIO

# Initialize Roboflow
rf = Roboflow(api_key="AtAN7fsWbxIN9Moql1gJ")
project = rf.workspace().project("mb-yellow-mosaic")
model = project.version("3").model

st.title('Image Processing with Roboflow')

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Process Image'):
        try:
            img_bytes = uploaded_file.getvalue()

            # Perform inference (adjust according to the actual API requirements)
            response = model.predict(img_bytes, confidence=40).json()
            
            # Assuming the response contains a URL to the processed image
            processed_image_url = response['processed_image_url']
            response = requests.get(processed_image_url)
            processed_image = Image.open(BytesIO(response.content))

            st.image(processed_image, caption='Processed Image', use_column_width=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
