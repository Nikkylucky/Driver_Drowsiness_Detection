# streamlit app for driver drowsiness detection using a pre-trained model

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageOps
from playsound import playsound


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

model = load_model('./drowiness_new7.h5')

def classify_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = face.astype("float")
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    face = preprocess_input(face)
    pred = model.predict(face)
    pred = np.argmax(pred, axis=1)
    return pred

def main():

    st.set_page_config(page_title="Drowsiness Detection", page_icon="mailbox_with_mail", layout="wide", initial_sidebar_state="expanded")

    option = option_menu(
        menu_title = None,
        options = ["Home", "Detector"],
        icons = ["house", "gear"],
        menu_icon = 'cast',
        default_index = 0,
        orientation = "horizontal"
    )

    if option == "Home":
        st.markdown("<h1 style='text-align: center; color: #08e08a;'>Drowsiness Detector</h1>", unsafe_allow_html=True)
        lottie_hello = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_mDnmhAgZkb.json")

# rgb color code for 
        st_lottie(
            lottie_hello,
            speed=1,
            reverse=False,
            loop=True,
            quality="High",
            #renderer="svg",
            height=400,
            width=-900,
            key=None,
        )

        st.markdown("<h4 style='text-align: center; color: #dae5e0;'>Upload your image and let the model detect the Drowsiness</h4>", unsafe_allow_html=True)

    elif option == "Detector":
        st.markdown("<h1 style='text-align: center; color: #08e08a;'>Drowsiness Detector</h1>", unsafe_allow_html=True)
        img = st.camera_input("Camera", key="camera")
        if img is not None:
            image = Image.open(img)
            image = np.array(image)
            res = classify_face(image)
            #yawn_detection_dict = ('Closed','Open','no_yawn','yawn')
            st.write(res[0])
            if res[0] == 0:
                st.write("Drowsiness Detected")
                # play sound here
                playsound('./alarm.mp3')
            if res[0] == 1:
                st.write("Drowsiness not Detected")
            if res[0] == 2:
                st.write("No Yawn Detected")
            if res[0] == 3:
                st.write("Yawn Detected")

if __name__ == "__main__":
    main()