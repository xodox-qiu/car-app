import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

@st.cache_resource
def load_model():
    df = pd.read_csv('Data/CarPrice_Assignment.csv')

    df['transmission'] = df['CarName'].apply(lambda x: 'Manual' if 'manual' in x.lower() else 'Automatic')
    df['transmission_val'] = df['transmission'].map({'Manual': 0, 'Automatic': 1})

    np.random.seed(42)
    df['year'] = np.random.randint(1990, 2025, size=len(df))
    df['kilometer'] = np.random.randint(10000, 200000, size=len(df))

    X = df[['year', 'kilometer', 'transmission_val']]
    y = df['price']

    model = LinearRegression()
    model.fit(X, y)

    with open('model_prediksi_harga_mobil.sav', 'wb') as f:
        pickle.dump(model, f)

    return model, df

model, df_mobil = load_model()

st.title("Prediksi Harga Mobil")

st.header('Datasets')
st.dataframe(df_mobil.head())

st.title("Visualisasi Data Mobil")

st.subheader("Line Chart: Highway MPG")
st.line_chart(df_mobil['highwaympg'])

st.subheader("Bar Chart: Curb Weight")
st.bar_chart(df_mobil['curbweight'])

st.subheader("Area Chart: Horsepower")
st.area_chart(df_mobil['horsepower'])

st.write("Masukkan spesifikasi mobil untuk memprediksi harganya")

tahun = st.number_input("Tahun Mobil", min_value=1990, max_value=2025, step=1)
kilometer = st.number_input("Kilometer Tempuh", min_value=0, max_value=1000000, step=1000)
transmisi = st.selectbox("Transmisi", ['Manual', 'Automatic'])

transmisi_val = 0 if transmisi == 'Manual' else 1

if st.button("Prediksi Harga"):
    data_prediksi = np.array([[tahun, kilometer, transmisi_val]])
    harga_prediksi = model.predict(data_prediksi)

    st.success(f"Prediksi harga mobil adalah: Rp {int(harga_prediksi[0]):,}")