import streamlit as st
from roboflow import Roboflow
from PIL import Image
import requests
from io import BytesIO

# Initialize Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace().project("YOUR_PROJECT_NAME")
model = project.version("YOUR_VERSION").model

st.title('Image Processing with Roboflow')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Process Image'):
        # TODO: Modify this part to suit the Roboflow model prediction
        result = model.predict(uploaded_file, confidence=40, overlap=30).json()
        
        # Assuming the result contains a URL to the processed image
        response = requests.get(result['processed_image_url'])
        processed_image = Image.open(BytesIO(response.content))

        st.image(processed_image, caption='Processed Image', use_column_width=True)
