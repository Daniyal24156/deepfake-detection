import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Constants
MODEL_PATH = 'models/deepfake_model.h5'
IMG_SIZE = 128

# Load model once
model = load_model(MODEL_PATH)

# Image preprocessing
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.title("🧠 Deepfake Detection App")
st.write("Upload a face image and I’ll tell you if it’s REAL or FAKE!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = preprocess_image(image)
    prediction = model.predict(img)[0][0]

    st.subheader("Prediction:")
    if prediction < 0.5:
        st.success("✅ REAL FACE (Label = 0)")
    else:
        st.error("❌ FAKE FACE (Label = 1)")

    st.caption(f"Confidence Score: {prediction:.4f}")
