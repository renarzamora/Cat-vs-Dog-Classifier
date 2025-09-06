# Cat-vs-Dog-Classifier
Cat and dog image classifier using **PyTorch** and **Streamlit**. Train a **simple CNN** and display an interactive demo to predict whether an image is of a cat or a dog.
=======
# 🐱🐶 Cat vs Dog Classifier

Cat and dog image classifier using **PyTorch** and **Streamlit**.
Train a **simple CNN** and display an interactive demo to predict whether an image is of a cat or a dog.

---

## 🚀 Demo

![demo](docs/demo.gif)

---

## 📂 Project structure
cat-dog-classifier/
├── app.py # Interfaz Streamlit
├── model.py # Definición de la CNN
├── train.py # Entrenamiento del modelo
├── predict.py # Predicciones con el modelo
├── split_val.py # Divide train/val
├── requirements.txt # Dependencias
├── README.md
└── model/ # Pesos guardados (cat_dog_cnn.pth)

---

📊 Dataset

Use [the Dogs vs. Cats dataset from Kaggle.](https://www.kaggle.com/c/dogs-vs-cats/data)

Organize it like this:

data/
├── train/
│   ├── Cats/
│   └── Dogs/
└── val/
    ├── Cats/
    └── Dogs/

---

In my case I downloaded all the images to train/, and used split_val.py to move the 20% to the val/ folder:

python split_val.py

---

🏋️‍♂️ Training

Run:
python train.py

---

This trains the model and saves the weights to:
./model/cat_dog_cnn.pth

Prediction
Example from Python:

from predict import predict_image
label, prob = predict_image("example.jpg")
print(label, prob)

---

🌐 Web interface (Streamlit)

Run:
streamlit run app.py

---

📈 Results

With 5 training epochs, the model achieves approximately 79% accuracy in validation.

---

🛠️ Technologies used

PyTorch

TorchVision

Streamlit

Pillow

---

Licence MIT


---

# 📄 app.py

```python
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


---

## 🚀 Demo

![demo](docs/demo.gif)

---

by Renar zamora Data Scientist / AI engineer
[Profile: ](https://www.linkedin.com/in/renar-arnoldo-zamora-54bb9024/)
