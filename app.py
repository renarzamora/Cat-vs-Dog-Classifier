import streamlit as st
import os
from predict import predict_image

st.set_page_config(page_title = "Cat vs Dog Classifier", page_icon = "🐱🐶")

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog, and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image", type = ["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # preview
    st.image(uploaded_file, caption="Uploaded image", width = 300)

    # temporary copy
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    model_path = "./model/cat_dog_cnn.pth"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please train the model first by running `train.py`.")
    else:
        label, prob = predict_image(temp_path, model_path = model_path)
        st.success(f"Prediction: **{label}** (Confidence: {prob:.2f})")
