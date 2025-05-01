import base64
import streamlit as st
from PIL import Image
import numpy as np

def set_background(image_file):
    """
    Set a custom image background in the Streamlit app.

    Args:
        image_file (str): Path to the image file.
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def predict(img, model, class_names):
    """
    Predict the class of an image using a trained model.

    Args:
        img (np.array): Preprocessed image of shape (1, 224, 224, 3)
        model (tf.keras.Model): Trained model to use for prediction
        class_names (list): List of class names

    Returns:
        Tuple: (predicted class name, confidence score)
    """
    img = img / 255.0  # Normalize pixel values between 0 and 1

    prediction = model.predict(img)  # Shape: (1, 1) for binary classification

    threshold = 0.5
    binary_prediction = (prediction > threshold).astype(int)  # Convert to 0 or 1
    class_index = binary_prediction[0][0]                     # Extract predicted index
    class_name = class_names[class_index]
    confidence_score = prediction[0][0]                       # Raw confidence score (between 0 and 1)

    return class_name, confidence_score
