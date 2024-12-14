import streamlit as st
import torch
from PIL import Image
import numpy as np
import io
from internal.detection import detect_disease  # Updated import
from internal.image_processing import process_image  # Updated import
from internal.rendering import draw_boxes  # Updated import

# Load models
xray_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/xray.pt', force_reload=True)
camera_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/camera.pt', force_reload=True)

# Streamlit app layout
st.title("Dental Disease Detection App")
st.write("Upload an X-ray or Camera image to detect dental diseases.")

# Image upload section
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process the image and display it
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated line

    # Select the model type
    model_type = st.selectbox("Select Model", ("xray", "camera"))

    # Convert image to a format suitable for the model
    image_array = np.array(image)

    if st.button("Detect Disease"):
        # Select the model based on user input
        model = xray_model if model_type == 'xray' else camera_model

        # Perform disease detection
        detections = detect_disease(image_array, model)

        # Draw bounding boxes on the image
        image_with_boxes = draw_boxes(image_array, detections, model.names)

        # Convert the output image to PIL format
        output_image = Image.fromarray(image_with_boxes)

        # Display the result
        st.image(output_image, caption="Detection Results", use_container_width=True)  # Updated line
