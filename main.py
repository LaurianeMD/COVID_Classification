import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


from util import classify, set_background


set_background('C:/Users/USER/Documents/Nouveau dossier/Master IA/Computer vision/Projet_Covid_CV/bgs/imcovid.jpg')

# set title
st.title('Covid classification')

# set header
st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('C:/Users/USER/Documents/Nouveau dossier/Master IA/Computer vision/Projet_Covid_CV/model/covid_classifier.h5',compile=False)

# load class names
with open('C:/Users/USER/Documents/Nouveau dossier/Master IA/Computer vision/Projet_Covid_CV/model/labels.txt', 'r') as f:
    lines = f.readlines()
#    print("Lines:", lines)
#    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    class_names = [line.strip().split(' ')[1] for line in lines if line.strip() and len(line.strip().split(' ')) > 1]
#    print("Class Names:", class_names)
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
