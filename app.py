import streamlit as st
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np

# Load the GFPGAN model
model = torch.hub.load('TencentARC/GFPGANv1', 'GFPGANv1')

# Define a function to enhance an image
def enhance_image(image, enhancement_level):
    # Convert the image to a tensor and normalize its values
    image_tensor = ToTensor()(image).unsqueeze(0) / 255.0

    # Enhance the image using the GFPGAN model
    with torch.no_grad():
        enhanced_tensor = model(image_tensor, enhancement_level)

    # Convert the enhanced tensor back to an image
    enhanced_image = ToPILImage()(enhanced_tensor.squeeze(0).cpu())

    return enhanced_image

# Set up the Streamlit app
st.title("GFPGAN Image Enhancer")
st.write("Use the slider below to adjust the enhancement level")

enhancement_level = st.slider("Enhancement level", 0.0, 1.0, 0.5, 0.1)

# Set up the camera
camera = st.camera_input()

if not camera:
    st.error("Could not access camera")

while camera:
    # Convert the camera input to an image
    image = Image.fromarray(camera)

    # Enhance the image
    enhanced_image = enhance_image(image, enhancement_level)

    # Convert the enhanced image back to an array
    enhanced_frame = np.array(enhanced_image)

    # Display the original and enhanced frames side by side
    st.image(np.hstack([camera, enhanced_frame]), channels="RGB")

    # Check if the user wants to stop the app
    if st.button("Stop"):
        break

    # Update the camera input
    camera = st.camera_input()

# Release the camera
st.stop()  # To avoid an error when stopping the streamlit app
