from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import streamlit as st
from keras.preprocessing.image import img_to_array
from skimage import transform
from keras.models import model_from_json
import tensorflow as tf


@st.cache(allow_output_mutation=True)
def get_model():
    json_file = open('classifier.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("weights.hdf5")
    return loaded_model


def predict(image_data):
    loaded_model = get_model()
    img = img_to_array(image_data)
    np_image = transform.resize(img, (64, 64, 3))

    image4 = np.expand_dims(np_image, axis=0)
    predictions = loaded_model.predict(image4)
    score = tf.nn.softmax(predictions[0])
    return score

