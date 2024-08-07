import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Mengimpor data
url = 'https://raw.githubusercontent.com/ervanervan/dataset-skripsi/main/laporan_iklim_harian_tanjungpinang_ff_x.csv'
data = pd.read_csv(url)
data_ff_x_tpi = data.to_json(orient='records', indent=4)

# Mengonversi kolom 'Tanggal' ke tipe datetime
data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d-%m-%Y')

# Mengatur 'Tanggal' sebagai indeks
data.set_index('Tanggal', inplace=True)

# Menggunakan MinMaxScaler untuk menskalakan data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Modifikasi fungsi create_dataset
def create_dataset(dataset, timeseries=1):
    X, Y = [], []
    for i in range(len(dataset)-timeseries):
        end_ix = i + timeseries
        # if end_ix + timeseries > len(dataset):  # Pastikan masih ada data untuk input 5 hari sebelumnya
        #     break
        a = dataset[i:end_ix, 0]
        b= dataset[end_ix, 0]
        X.append(a)
        Y.append(b)
    return np.array(X), np.array(Y)

timeseries = 5

# Mengubah bentuk input menjadi [samples, time steps, features]
X, Y = create_dataset(data_scaled, timeseries)
# X= np.reshape(Y, (Y.shape[0], timeseries, 1))
X = np.reshape(X, (X.shape[0], timeseries, 1))
X_data_ff_x_tpi = X

# Membagi data menjadi 70% training dan 30% testing
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

modelFileName= 'BiGRUFFXTPI.keras'
model = tf.keras.models.load_model(modelFileName)


def predict_ff_x_tpi(input_data):
    prediction = model.predict(input_data)
    result = scaler.inverse_transform(prediction).flatten()
    data = []
    for i in range (len(result)):
        data.append(float(result[i]))
    return data

def predict_wind_speed_90days(model, scaler, input_data):
    input_data_reshaped = np.array(input_data).reshape(timeseries, 1)
    input_scaled = scaler.transform(input_data_reshaped)
    input_reshaped = np.reshape(input_scaled, (1, timeseries, 1))
    prediction_scaled = model.predict(input_reshaped)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0, 0]

Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))
timeseries_input = scaler.inverse_transform(Y_test[-(timeseries):])


def predict_forcasting_ff_x_tpi():
    timeseries_speeds = []
    for i in range (len(timeseries_input)):
        timeseries_speeds.append(timeseries_input[i][0])

    forecasted = []
    for i in range(90):

        input_x = np.array(timeseries_speeds).reshape(-1, 1)
        prediction = predict_wind_speed_90days(model, scaler, input_x)
        forecasted.append(float(prediction))
        
        timeseries_speeds.pop(0)
        timeseries_speeds.append(float(prediction))


def predict_forcasting_ff_x_tpi_by_input(input_data):
    input_timeseries = np.array(input_data).reshape(1, -1)
    timeseries_input = scaler.inverse_transform(input_timeseries)

    timeseries_speeds = []
    for i in range(len(timeseries_input[0])):
        timeseries_speeds.append(timeseries_input[0][i])

    forecasted = []
    for i in range(90):
        input_x = np.array(timeseries_speeds).reshape(-1, 1)
        prediction = predict_wind_speed_90days(model, scaler, input_x)
        forecasted.append(float(prediction))
        
        timeseries_speeds.pop(0)
        timeseries_speeds.append(float(prediction))
    
    return forecasted