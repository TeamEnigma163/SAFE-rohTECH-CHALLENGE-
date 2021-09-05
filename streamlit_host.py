import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("saved_model/enigma.hdf5")
uploaded_file = st.file_uploader("Choose a image file", type="jpeg")



if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(200,200))
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape,batch_size=10)
        #st.title(prediction)
        st.title('The major reason of accident is: ')
        if prediction.max()==prediction[0][0]:
            st.title('Crash due to negligence')
        elif prediction.max()==prediction[0][1]:
            st.title('Fog')
        elif prediction.max()==prediction[0][2]:
            st.title('No seat belt')
        elif prediction.max()==prediction[0][3]:
            st.title('Potholes')
        elif prediction.max()==prediction[0][4]:
            st.title('Rain')
        else:
            st.title('More information required!')