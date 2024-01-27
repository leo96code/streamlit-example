import streamlit as st
from PIL import Image

# Title
st.title("Image Upload and Model Selection")

# Sidebar for model selection
model = st.sidebar.selectbox(
    "Select a Model",
    ["MB", "Seg", "YoloV8"]
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display uploaded image and process button
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Process button
    if st.button('Process Image'):
        st.write("Model selected:", model)
        # Implement model processing here
        # For demonstration, just display the image again
        st.image(image, caption='Processed Image', use_column_width=True)
else:
    st.write("Please upload an image...")

