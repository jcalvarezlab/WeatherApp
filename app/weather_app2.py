import streamlit as st
import joblib
import pandas as pd
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load('models/model_pipeline5.pkl')

pipeline = load_model()

st.title("☔ Weather Rain Prediction App")

st.sidebar.header('Input Features')

# Request ONLY most important 5 input features
humidity_3pm = st.sidebar.number_input("Humidity 3pm (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
pressure_3pm = st.sidebar.number_input("Pressure 3pm (hPa)", min_value=800.0, max_value=1100.0, value=1010.0, step=0.1)
sunshine = st.sidebar.number_input("Sunshine (hours)", min_value=0.0, max_value=24.0, value=8.0, step=0.1)
temperature_3pm = st.sidebar.number_input("Temperature 3pm (°C)", min_value=-10.0, max_value=50.0, value=22.0, step=0.1)
wind_gust_speed = st.sidebar.number_input("Wind Gust Speed (km/h)", min_value=0.0, max_value=150.0, value=30.0, step=0.1)

input_df = pd.DataFrame({
    'Humidity3pm': [humidity_3pm],
    'Pressure3pm': [pressure_3pm],
    'Sunshine': [sunshine],
    'Temperature3pm': [temperature_3pm],
    'WindGustSpeed': [wind_gust_speed]
})

if st.button('Predict'):
    prediction = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][prediction]
    result = 'Yes' if prediction == 1 else 'No'
    st.write(f"### 🌧️ Will it rain tomorrow? **{result}** (Confidence: {prob:.2f})")
