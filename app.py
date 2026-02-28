import streamlit as st
import numpy as np
import pickle

model, scaler = pickle.load(open("diabetes_model.pkl", "rb"))


st.title("Diabetes Prediction App")

preg = st.number_input("Pregnancies (0 – 17)", min_value=0, max_value=17, value=0)
glucose = st.number_input("Glucose (70 – 180)", min_value=70, max_value=180, value=0)
bp = st.number_input("Blood Pressure (60 – 120)", min_value=60, max_value=120, value=0)
skin = st.number_input("Skin Thickness (10 – 50)", min_value=10, max_value=50, value=0)
insulin = st.number_input("Insulin (15 – 276)", min_value=15, max_value=276, value=0)
bmi = st.number_input("BMI (18.5 – 40)", min_value=18.5, max_value=40.0, value=0.0)
dpf = st.number_input("Diabetes Pedigree Function (0.1 – 2.5)", min_value=0.1, max_value=2.5, value=0.0)
age = st.number_input("Age (21 – 60)", min_value=21, max_value=60, value=21)


if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Non-Diabetic")


