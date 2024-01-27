import streamlit as st
import tempfile
import os
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from roboflow import Roboflow



rf = Roboflow(api_key="AtAN7fsWbxIN9Moql1gJ")
project = rf.workspace().project("mb-yellow-mosaic")
model = project.version("3").model

st.title('Image Processing with Roboflow')

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)

    # Convert image to RGBA to process transparency
    # This will be used later for overlaying the mask
    image_rgba = image.convert("RGBA")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # When the button is pressed
    if st.button('Process Image'):
        # Save the uploaded file to a temporary file without changing its mode
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            # Directly write the original image data without converting to RGBA
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Proceed with processing
        try:
            # Call the Roboflow API (replace this with the correct call as per your Roboflow API)
            response = model.predict(tmp_file_path, confidence=40).json()
            
            # Remove the temporary file
            os.unlink(tmp_file_path)
            
            # Decode the base64-encoded segmentation mask
            segmentation_mask_data = base64.b64decode(response['predictions']['segmentation_mask'])
            segmentation_mask = Image.open(BytesIO(segmentation_mask_data)).convert("L")
            
            # Prepare the overlay
            overlay_image = Image.new("RGBA", image_rgba.size)
            overlay_image.putalpha(segmentation_mask)
            
            # Overlay the mask on the original RGBA image
            combined_image = Image.alpha_composite(image_rgba, overlay_image)
            
            # Display the combined image
            st.image(combined_image, caption='Processed Image', use_column_width=True)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
