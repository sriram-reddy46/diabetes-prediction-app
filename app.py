import streamlit as st
import numpy as np
import pickle

model, scaler = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("Diabetes Prediction App")


st.subheader("Reference Ranges")

st.markdown("""
- **Pregnancies:** 0 – 17  
- **Glucose:** 70 – 180  
- **Blood Pressure:** 60 – 120  
- **Skin Thickness:** 10 – 50  
- **Insulin:** 15 – 276  
- **BMI:** 18.5 – 40  
- **DPF:** 0.1 – 2.5  
- **Age:** 21 – 60  
""")


preg = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Non-Diabetic")

