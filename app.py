import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load image stylization module.
@st.cache(allow_output_mutation=True)
def load_model():
  return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

style_transfer_model = load_model()
def enhance_image(image, enhancement_level):
    # Convert the image to a tensor and normalize its values
    image_tensor = ToTensor()(image).unsqueeze(0) / 255.0

    # Enhance the image using the GFPGAN model
    with torch.no_grad():
        enhanced_tensor = style_transfer_model(image_tensor, enhancement_level)

    # Convert the enhanced tensor back to an image
    enhanced_image = ToPILImage()(enhanced_tensor.squeeze(0).cpu())

    return enhanced_image
# Upload content and style images.

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
st.stop()  
