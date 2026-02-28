import streamlit as st
import numpy as np
import pickle

model, scaler = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("Diabetes Prediction App")

preg = st.number_input("Pregnancies (0-17)")
glucose = st.number_input("Glucose (70 – 180)")
bp = st.number_input("Blood Pressure (60 – 120)")
skin = st.number_input("Skin Thickness (10 – 50)")
insulin = st.number_input("Insulin (15 – 276)")
bmi = st.number_input("BMI (18.5 – 40)")
dpf = st.number_input("Diabetes Pedigree Function (0.1 – 2.5)")
age = st.number_input("Age")

if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Non-Diabetic")



