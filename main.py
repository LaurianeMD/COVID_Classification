import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np

from util import set_background, predict

set_background('./bgs/ld.jpg')

# set title
st.title('Covid classification')

# set header
st.header('Please upload a X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./models/Covid_class.h5', compile=False)

# load class names
with open('./models/labels.txt', 'r') as f:
    lines = f.readlines()
    class_names = [a[:-1].split(' ')[1] for a in lines]
    f.close()

# display image
if file is not None:
    img = image.load_img(file, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = np.reshape(img, (224, 224, 3))

    # display image 
    to_display = Image.open(file).convert('RGB')
    st.image(to_display, use_column_width=True)

    # classify image
    class_name, conf_score = predict(img, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
