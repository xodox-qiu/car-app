import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

model = LinearRegression()
pickle.dump(model, open('Data/model_prediksi_harga_mobil.sav', 'wb'))

st.title("Prediksi Harga Mobil")

st.header('Datasets')
df_mobil = pd.read_csv('Data/CarPrice_Assignment.csv')
df_mobil

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

if transmisi == 'Manual':
    transmisi_val = 0
else:
    transmisi_val = 1

if st.button("Prediksi Harga"):
    data_prediksi = np.array([[tahun, kilometer, transmisi_val]])
    harga_prediksi = model.predict(data_prediksi)

    st.success(f"Prediksi harga mobil adalah: Rp {int(harga_prediksi[0]):,}")

#new
