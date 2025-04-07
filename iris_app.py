import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

def load_data():
    df = pd.read_csv('Iris.csv')
    df['Species'] = df['Species'].str.replace('Iris-', '')
    return df

def train_model(df):
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    le = LabelEncoder()
    y = le.fit_transform(df['Species'])
    model = LinearRegression()
    model.fit(X, y)
    return model, le

df = load_data()
model, label_encoder = train_model(df)

st.title('Iris Species Prediction')
st.write('Enter flower measurements in centimeters:')

col1, col2 = st.columns(2)
with col1:
    sl = st.number_input('Sepal Length', min_value=0.0, value=5.1, step=0.1)
    sw = st.number_input('Sepal Width', min_value=0.0, value=3.5, step=0.1)
with col2:
    pl = st.number_input('Petal Length', min_value=0.0, value=1.4, step=0.1)
    pw = st.number_input('Petal Width', min_value=0.0, value=0.2, step=0.1)

if st.button('Predict Species'):
    input_data = [[sl, sw, pl, pw]]
    prediction = model.predict(input_data)[0]
    final_pred = int(np.round(np.clip(prediction, 0, 2)))
    species = label_encoder.inverse_transform([final_pred])[0].capitalize()
    st.success(f'Predicted Species: {species}')
    
