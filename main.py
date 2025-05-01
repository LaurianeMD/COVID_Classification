import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np

from util import set_background, predict

# Set the custom background for the Streamlit app
set_background('./bgs/ld.jpg')

# Title and header
st.title('COVID-19 Chest X-ray Classification')
st.header('Please upload a chest X-ray image')

# Upload file input (image uploader)
file = st.file_uploader('Upload an X-ray image', type=['jpeg', 'jpg', 'png'], label_visibility='collapsed')

# Load pre-trained classification model
model = load_model('./models/Covid_class.h5', compile=False)

# Load class names from labels file
with open('./models/labels.txt', 'r') as f:
    lines = f.readlines()
    class_names = [a.strip().split(' ')[1] for a in lines]  # Extract class name from each line

# If a file has been uploaded
if file is not None:
    # Load the uploaded image and resize it to match model input
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)               # Convert to array format
    img_array = np.expand_dims(img_array, axis=0)     # Add batch dimension → shape: (1, 224, 224, 3)

    # Display the image in the app
    to_display = Image.open(file).convert('RGB')
    st.image(to_display, use_container_width=True)

    # Make prediction using the model
    class_name, conf_score = predict(img_array, model, class_names)

    # Display the result and confidence
    st.write(f"## Predicted Class: {class_name}")
    st.metric(label="Confidence", value=f"{conf_score * 100:.1f}%")
