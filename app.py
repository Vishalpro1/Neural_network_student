import streamlit as st
import numpy as np
import joblib
import os

# Load model & scaler
try:
    model = joblib.load(r"student_performance_model.pkl")
    scaler = joblib.load(r"student_performance_scaler.pkl")
except FileNotFoundError as e:
    st.error(f"Error: Could not find model files. {str(e)}")
    st.stop()

st.title("-----Student Performance Prediction App----")

st.write("Enter student details to predict performance:")

# Inputs
hours = st.number_input("Hours Studied", 0, 12, 6)
previous_score = st.number_input("Previous Score", 0, 100, 70)
extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
sleep = st.number_input("Sleep Hours", 0, 12, 7)
papers = st.number_input("Sample Question Papers Practiced", 0, 10, 3)

# Encoding
extra_val = 1 if extra == "Yes" else 0

if st.button("Predict Performance"):
    input_data = np.array([[hours, previous_score, extra_val, sleep, papers]])
    input_scaled = scaler.transform(input_data)

    prediction_value = float(model.predict(input_scaled)[0][0])

    st.success(f"Predicted Performance Index: {prediction_value:.2f}")

    if prediction_value <= 40:
        st.error("🥲 Result is Poor.Please study more!!!")
        st.snow()
    elif prediction_value < 70:
        st.warning("😊 Average performance .You can do Better!!")
        st.snow()
    else:
        st.success("✨ Excellent performance! Keep it up!!!")
        st.balloons()