import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
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
    img = np.expand_dims(img, axis=0)  # Ajoute une dimension de lot (batch)
    # Normalisation de l'image (si nécessaire)
    img = img / 255.0  # Vous pouvez adapter la normalisation en fonction de celle utilisée pendant l'entraînement
    
    # Prédiction de la classe de l'image
    prediction = model.predict(img)
    
    # Convertir le score prédit en étiquette binaire
    seuil = 0.5  # Vous pouvez ajuster le seuil au besoin
    binary_prediction = (prediction > seuil).astype(int)
    class_index = binary_prediction[0][0]

    class_name = class_names[class_index]
    confidence_score = prediction[0][0]

    return class_name, confidence_score