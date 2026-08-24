
import streamlit as st
from models import cardio
import pandas as pd
import requests

import matplotlib.pyplot as plt
import seaborn as sns

## function implementation
features, scaler, model, Y_pred, cr, cm = cardio()

st.header('Cardiovascular Disease Predictiton')
st.subheader('Using Logistic Regression')

API_URL = 'http://127.0.0.1:8000/predict-logistic-cardio'
# age = st.text_input('Age', placeholder='Enter your age')

st.sidebar.header(
    'Cardio Features'
)

age = st.sidebar.slider(
    'Age', min_value=28, max_value=65, value=35, step=1
)

# 1-> Female, 2-> Male
gender_dict = {1: 'Female', 2: 'Male'}
gender = st.sidebar.radio(
    'Gender',
    options = list(gender_dict.keys()),
    format_func = lambda x: gender_dict.get(x)
)

height = st.sidebar.slider(
    'Height', min_value=125, max_value=200, value=140, step=1

)

weight = st.sidebar.slider(
    'Weight', min_value=40, max_value=120, value=60, step=1

)

ap_hi = st.sidebar.slider(
    'Systolic Pressure', min_value=100, max_value=200, value=120, step=1

)

ap_lo = st.sidebar.slider(
    'Di-Systolic Pressure', min_value=50, max_value=90, value=65, step=1

)

cholesterol_dict = {1: 'Low Cholesterol', 2: 'Mid Cholesterol', 3: 'High Cholesterol'} # type: ignore

cholesterol = st.sidebar.radio(
    'Cholesterol',
    options=list(cholesterol_dict.keys()),
    format_func = lambda x: cholesterol_dict.get(x)
)

gluc_dict = {1: 'Low Glucose', 2: 'Mid Glucose', 3: 'High Glucose'} # type: ignore

gluc = st.sidebar.radio(
    'Glucose',
    options=list(gluc_dict.keys()),
    format_func = lambda x: gluc_dict.get(x)
)

Smoke_dict = {0: 'Doesnot smoke', 1: 'Does Smoke'}
smoke = st.sidebar.radio(
    'Smoke',
    options=list(Smoke_dict.keys()),
    format_func= lambda x: Smoke_dict.get(x)
)

alco_dict = {0: 'Doesnot Drink Alcohl', 1: 'Does Drink Alcohol'}
alco = st.sidebar.radio(
    'Alcohol',
    options=list(alco_dict.keys()),
    format_func= lambda x: alco_dict.get(x)
)

active_dict = {0: 'Doesnot smoke', 1: 'Does Smoke'}
active = st.sidebar.radio(
    'Pyysical Activities (PA)',
    options=list(active_dict.keys()),
    format_func= lambda x: active_dict.get(x)
)

## Create prediction button
'''
if st.button('Predict Cardio'):
    #
    st.toast('Button Clicked.', icon="😊")
    data = pd.DataFrame([[
        age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
    ]], columns=features)

    data_scale = scaler.transform(data)
    prediction = model.predict(data_scale)[0]

    if prediction == 0:
        st.success('No cardiovascular deseaso found.')
    else:
        st.warning('Cardiovascular disease found.')
'''

if st.button('predict Cardio'):
    payload = {
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol':cholesterol,
        'gluc':gluc,
        'smoke': smoke,
        'alco' : alco,
        'active': active
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()

            if result['Prediction Status'] == 0:
                st.write('Likely to be Healtyty')
                st.success('No Cardiovascular disease found')
            else:
                st.write('Likely to be Unhealtyty')
                st.success('Cardiovascular disease found')
        else:
            st.error(f'API Status code Error: {response.status_code}')
    except requests.exceptions.RequestException as e:
        st.error(f'API Error: {e}')


        
## Visualizations

st.subheader('Visualizations')
fig, axes = plt.subplots(figsize=(6,4))

sns.heatmap(cm, annot=True, fmt='1.0f', xticklabels=['Predicted Healthy[0]', 'Predicted Unhealthy[1]'],
           yticklabels = ['Actual Healthy[0]', 'Actual Unhealthy[1]'])
plt.title('Actual Cardio vs. Predicted Cardio')
st.pyplot(fig)


## Classification Report
st.subheader('Classification Report')
data = pd.DataFrame(cr).transpose()
st.dataframe(data.style.format(precision=2))