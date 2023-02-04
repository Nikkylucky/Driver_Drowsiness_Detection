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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import time


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

model = load_model('./drowiness_new7.h5')

emotion_dict = {0:'Drowsiness Detected', 1 :'No Drowsiness', 2: 'No Yawning', 3:'Yawning'}


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    left = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    right = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
except Exception:
    st.write("Error loading cascade classifiers")


def play_sound():
    html_string="""
                            <audio autoplay loop>
                                <source src="https://www.orangefreesounds.com/wp-content/uploads/2022/04/Small-bell-ringing-short-sound-effect.mp3" type="audio/mp3">
                            </audio>
                            """
    sound = st.empty()
    sound.markdown(html_string, unsafe_allow_html=True)
    time.sleep(10)
    sound.empty()

def classify_face(face):
    resu = []
    face1 = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(image=face, scaleFactor=1.3, minNeighbors=5)
    leye = left.detectMultiScale(image=face, scaleFactor=1.3, minNeighbors=5)
    reye = right.detectMultiScale(image=face, scaleFactor=1.3, minNeighbors=5)
    '''for (x, y, w, h) in faces:
        face1 = face1[y:y + h, x:x + w]
        face1 = cv2.resize(face1, (224, 224), interpolation=cv2.INTER_AREA)
        if np.sum([face1]) != 0:
            face1 = face1.astype("float") / 255.0
            face1= img_to_array(face1)
            face1 = np.expand_dims(face1, axis=0)
            pred = model.predict(face1)
            pred1 = np.argmax(pred, axis=1)
            resu.append(pred1)'''
    
    for (x1, y1, w1, h1) in leye:
        face = face[y1:y1 + h1, x1:x1 + w1]
        face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
        if np.sum([face]) != 0:
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            pred = model.predict(face)
            pred2 = np.argmax(pred, axis=1)
            resu.append(pred2)
        
    for (x2, y2, w2, h2) in reye:
        face = face[y2:y2 + h2, x2:x2 + w2]
        face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
        if np.sum([face]) != 0:
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            pred = model.predict(face)
            pred3 = np.argmax(pred, axis=1)
            resu.append(pred3)

    return resu


class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(image=img_color, scaleFactor=1.3, minNeighbors=5)
        leye = left.detectMultiScale(image=img_color, scaleFactor=1.3, minNeighbors=5)
        reye = right.detectMultiScale(image=img_color, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_color[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = model.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                final_pred = emotion_dict[maxindex]
                output = str(final_pred)

            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for (x1, y1, w1, h1) in leye:
            cv2.rectangle(img=img, pt1=(x1, y1), pt2=(
                x1 + w1, y1 + h1), color=(255, 0, 0), thickness=2)
            roi_gray = img_color[y1:y1 + h1, x1:x1 + w1]
            roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = model.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                final_pred = emotion_dict[maxindex]
                output = str(final_pred)

            label_position = (x1, y1)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for (x2, y2, w2, h2) in reye:
            cv2.rectangle(img=img, pt1=(x2, y2), pt2=(
                x2 + w2, y2 + h2), color=(255, 0, 0), thickness=2)
            roi_gray = img_color[y2:y2 + h2, x2:x2 + w2]
            roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = model.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                final_pred = emotion_dict[maxindex]
                output = str(final_pred)

            label_position = (x2, y2)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

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
        options = ["Camera", "Image"]
        choice = st.selectbox("Select Source", options)
        if choice == "Image":
            img = st.camera_input("Camera", key="camera")
            if img is not None:
                image = Image.open(img)
                image = np.array(image)
                res = classify_face(image)
                #yawn_detection_dict = ('Closed','Open','no_yawn','yawn')
                #st.write(res)

                if len(res) > 1 and len(res) < 3: # if 1 eye is detected then 2^1 = 2 combinations
                    if res[1] == 0:
                        st.error("Drowsiness Detected in Left Eye")
                        play_sound()
                    else:
                        st.success("No Drowsiness Detected in Left Eye")

                if len(res) > 2: # if 2 eyes are detected then 2^2 = 4 combinations
                    if res[1] & res[2] == 0:
                        st.error("Drowsiness Detected in Both Eyes")
                        play_sound()
                    elif res[1] & res[2] == 1:
                        st.success("No Drowsiness Detected in Both Eyes")
                    elif res[1] == 1 & res[2] == 0:
                        st.success("Drowsiness Detected in Right Eye")
                        play_sound()
                    elif res[1] == 0 & res[2] == 1:
                        st.error("Drowsiness Detected in left Eye")
                        play_sound()
            else:
                st.error("No image found")
                
                    
        elif choice == "Camera":
            st.write("Use your webcam to detect Drowsiness")
            st.error("No sound alerting for this mode")
            st.success("Switch to Image mode for sound alerting")
            webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_transformer_factory=Faceemotion)


if __name__ == "__main__":
    main()
