import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Function to load and process the image
def load_image(image_file):
    return Image.open(image_file)

# Title
st.title("YOLO Model Image Prediction")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load model
model = YOLO('models/last.pt')  # Adjust the path to the model file if necessary

# Display uploaded image and process button
if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process button
    if st.button('Process Image'):
        # Perform prediction
        results = model(image)
        
        # Display results
        for r in results:
            masks = r.masks.cpu().numpy()  # Getting masks
            # Display each mask
            for mask in masks:
                st.image(mask, caption='Mask', use_column_width=True)
else:
    st.write("Please upload an image...")

