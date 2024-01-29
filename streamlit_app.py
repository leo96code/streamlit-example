import streamlit as st
from PIL import Image
import io

# Set up the main layout
col1, col2 = st.columns(2)

with col1:
    # Create a file uploader to upload images
    uploaded_file = st.file_uploader("Drag or Upload Image", type=['png', 'jpg', 'jpeg'])

    # When the user clicks the 'Process' button
    if st.button('Process'):
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            # Use BytesIO to handle the uploaded file as a file-like object
            image = Image.open(io.BytesIO(bytes_data))
            # Save the uploaded image to a variable accessible outside of this block
            st.session_state['image'] = image

with col2:
    if 'image' in st.session_state:
        # Display the image
        st.image(st.session_state['image'], caption='Uploaded Image.')

# To clear the session state if needed
if st.button('Clear Image'):
    if 'image' in st.session_state:
        del st.session_state['image']
