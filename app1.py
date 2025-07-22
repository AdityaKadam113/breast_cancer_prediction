import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load the scaler and model
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = tf.keras.models.load_model('breast_cancer_model.h5')

st.title("Breast Cancer Prediction App")
st.write("Enter the patient's test results to predict whether the tumor is malignant or benign.")

# Input fields for 30 features
features = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

user_input = []
for feature in features:
    val = st.number_input(f"{feature}", step=0.01)
    user_input.append(val)

if st.button("Predict"):
    input_np = np.array(user_input).reshape(1, -1)
    input_std = scaler.transform(input_np)
    prediction = model.predict(input_std)
    pred_label = np.argmax(prediction)

    if pred_label == 0:
        st.error("The model predicts: Malignant tumor")
    else:
        st.success("The model predicts: Benign tumor")
