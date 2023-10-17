import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import base64
from PIL import ImageOps, Image
from keras.models import load_model





# Charger le modèle
model = load_model('./models/covid_classifier.h5')

class_names = {
    0: "COVID",
    1: "Normal"
}

st.title("COVID Classification")

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

set_background('./bgs/bd.jpg')


# Charger l'image depuis l'ordinateur local
file_to_predict = st.file_uploader("Télécharger une image", type=["jpg", "png", "jpeg"])
if file_to_predict is not None:
    image_bytes = file_to_predict.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image_to_predict = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image_to_predict is not None:
        st.image(image_to_predict, caption="Image chargée", use_column_width=True)
        
        # Redimensionnez l'image à la taille attendue par le modèle (224x224)
        target_size = (224, 224)
        image_to_predict = cv2.resize(image_to_predict, target_size)
        
        # Normalisez l'image si nécessaire
        image_to_predict = image_to_predict / 255.0
        
        # Utilisez le modèle pour obtenir les prédictions
        img_to_predict = np.expand_dims(image_to_predict, axis=0)
        predictions = model.predict(img_to_predict)
        
        # Obtenez la classe prédite (indice de la classe)
        predicted_class = np.argmax(predictions)
        predicted_class_name = class_names.get(predicted_class, "Classe inconnue")
        
        # Affichez la classe prédite
        st.write(f"Predicted Class: {predicted_class} - {predicted_class_name}")
        
        # Affichez les probabilités pour chaque classe
        st.write("Probabilités :", predictions)
    else:
        st.write("Erreur de chargement de l'image.")

