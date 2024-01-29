import streamlit as st
from PIL import Image
import io
import torch
import requests
from torchvision import transforms

# Function to load model from the given URL
@st.cache(allow_output_mutation=True)
def load_model(model_url):
    # Assuming the model is in PyTorch format
    model = torch.hub.load('ultralytics/yolov8', 'custom', path_or_model='https://github.com/leo96code/streamlit-example/blob/master/models/last.pt')
    return model

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
            model = load_model('https://github.com/leo96code/streamlit-example/blob/master/models/last.pt')
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
