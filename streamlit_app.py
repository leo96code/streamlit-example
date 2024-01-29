import streamlit as st
from PIL import Image
import io
import torch
from torch import hub
import requests
import tempfile

# Function to download and load model from a direct URL
@st.cache(allow_output_mutation=True)
def load_model(model_url):
    # GitHub raw content URL for direct download
    # Replace 'blob' with 'raw' in the provided GitHub URL
    raw_model_url = model_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')

    # Download the model file from the URL
    r = requests.get(raw_model_url, allow_redirects=True)
    if r.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model_file:
            tmp_model_file.write(r.content)
            tmp_model_file.flush()
            # Load the model using PyTorch
            model = torch.hub.load('ultralytics/yolov8', 'custom', path_or_model=tmp_model_file.name)
            return model
    else:
        raise Exception(f"Failed to download the model, status code: {r.status_code}")

# Set up the main layout
col1, col2 = st.columns(2)

with col1:
    # Create a file uploader to upload images
    uploaded_file = st.file_uploader("Drag or Upload Image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # Use BytesIO to handle the uploaded file as a file-like object
        image = Image.open(io.BytesIO(bytes_data))
        # Display the uploaded image
        st.image(image, caption='Uploaded Image.')

        # When the user clicks the 'Process' button
        if st.button('Process'):
            # Load the model
            model = load_model('https://github.com/leo96code/streamlit-example/blob/57caeea14280d4a0404931480d158bd82d0830ad/models/last.pt')
            # Convert PIL image to RGB if not already in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Perform inference
            results = model(image)
            # Convert results to image
            result_image = results.render()[0]
            # Save the result image to a variable accessible outside of this block
            st.session_state['result_image'] = Image.fromarray(result_image)

with col2:
    if 'result_image' in st.session_state:
        # Display the image with detections
        st.image(st.session_state['result_image'], caption='Processed Image.')

# To clear the session state if needed
if st.button('Clear Image'):
    if 'result_image' in st.session_state:
        del st.session_state['result_image']
