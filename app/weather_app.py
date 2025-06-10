import streamlit as st
import joblib
import pandas as pd
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load('model_pipeline2.pkl')

pipeline = load_model()

st.title("☔ Weather Rain Prediction App")

# Example input
st.sidebar.header('Input Features')
Evapotation = st.sidebar.number_input("Evapotation", value=0)
Sunshine = st.sidebar.number_input("Sunshine", value=0)
humidity_9am = st.sidebar.number_input("Humidity9am (%)", value=80.0)
humidity_3pm = st.sidebar.number_input("Humidity3pm (%)", value=80.0)
wind_speed_9am = st.sidebar.number_input("Wind 9am Speed (km/h)", value=30.0)
wind_speed_3pm = st.sidebar.number_input("Wind 3pm Speed (km/h)", value=30.0)
wind_gust_speed = st.sidebar.number_input("Wind gust Speed (km/h)", value=30.0)
pressure_3pm = st.sidebar.number_input("Pressure 3pm", value=1007.0)
pressure_9am = st.sidebar.number_input("Pressure 9am", value=1008.0)
cloud_9am = st.sidebar.number_input("Cloud 9am", value=8.0)
cloud_3pm =  st.sidebar.number_input("Cloud 3 pm", value=7.0)
temperature_9am = st.sidebar.number_input("Temperature 9am", value=25.0)
temperature_3pm = st.sidebar.number_input("Temperature 3pm", value=30.0)
min_temp = st.sidebar.number_input("Min Temp", value=15.0)
max_temp = st.sidebar.number_input("Max Temp", value=35.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", value=999.0)

#Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday,RainTomorrow
#KeyError: "None of [Index(['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainYesterday',\n 'Season'],\n dtype='object')] are in the [columns]"

#wind_gust_dir = st.sidebar.selectbox("Wind Gust Direction", ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
#wind_dir_9am = st.sidebar.selectbox("Wind Direction 9am", ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
#wind_dir_3pm = st.sidebar.selectbox("Wind Direction 3pm", ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

#input_df = pd.DataFrame({
#    'MinTemp': [min_temp],
#    'MaxTemp': [max_temp],
#    'Rainfall': [rainfall],
#    'WindGustSpeed': [wind_gust_speed],
#    'WindGustDir': [wind_gust_dir],
#    'WindDir9am': [wind_dir_9am],
#    'WindDir3pm': [wind_dir_3pm],
#})

input_df = pd.DataFrame({'Evaporation': [Evapotation], 'Sunshine': [Sunshine], 'WindSpeed9am': [wind_speed_9am], 'WindSpeed3pm': [wind_speed_3pm], 'Humidity9am': [humidity_9am], 'Humidity3pm': [humidity_3pm], 'Pressure9am': [pressure_9am], 'Pressure3pm': [pressure_3pm], 'Cloud9am': [cloud_9am], 'Cloud3pm': [cloud_3pm], 'Temp9am': [temperature_9am], 'Temp3pm': [temperature_3pm], 'MinTemp': [min_temp], 'MaxTemp':[max_temp], 'Rainfall': [rainfall], 'WindGustSpeed':[wind_gust_speed],})

if st.button('Predict'):
    prediction = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][prediction]
    result = 'Yes' if prediction == 1 else 'No'
    st.write(f"### 🌧️ Will it rain tomorrow? **{result}** (Confidence: {prob:.2f})")

