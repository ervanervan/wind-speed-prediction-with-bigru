import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Bidirectional
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Mengimpor data
url = 'https://raw.githubusercontent.com/ervanervan/dataset-skripsi/main/laporan_iklim_anambas_ff_x.csv'
data = pd.read_csv(url)

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

# Membagi data menjadi 70% training dan 30% testing
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Modifikasi arsitektur model

def createModel():
    model = Sequential()
    model.add(Bidirectional(GRU(50, activation='tanh', return_sequences=False), input_shape=(timeseries, 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

model = createModel()

# Melatih model
def trainingModel(model):
    model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1, validation_split=0.2)
    return model

history = trainingModel(model)

# Evaluasi model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
# print(X_test)

# [[[0,2,2,3]]]
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))
actual_train = scaler.inverse_transform(Y_train)
actual_test = scaler.inverse_transform(Y_test)
mse = mean_squared_error(actual_test, test_predict)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_test, test_predict)
mape = np.mean(np.abs((actual_test - test_predict) / actual_test)) * 100
akurasi = 100 - mape
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.2f}%')
print(f'Akurasi: {akurasi:.2f}%')

model_name = "Bidirectional_GRU_FF_X_ANAMBAS"
model_performance = {
    'MAPE' : mape,
    'AKURASI' : akurasi,
    'model_name' : model_name
}

# Menyimpan variabel ke file JSON
with open(model_name+'.json', 'w') as json_file:
    json.dump(model_performance, json_file, indent=4)


# Simpan model
model_filename = "BiGRUFFXANB.keras"
model.save(model_filename)

# Plot data aktual vs prediksi
plt.figure(figsize=(10,6))
plt.plot(data.index[:train_size], actual_train.flatten(), label='Data Aktual - Training')
plt.plot(data.index[:train_size], train_predict.flatten(), label='Prediksi - Training')
plt.plot(data.index[train_size:train_size+len(actual_test)], actual_test.flatten(), label='Data Aktual - Testing')
plt.plot(data.index[train_size:train_size+len(test_predict)], test_predict.flatten(), label='Prediksi - Testing')
plt.xlabel('Tanggal')
plt.ylabel('Kecepatan Angin maksimum')
plt.title('Prediksi vs Data Aktual Kecepatan Angin maksimum')
plt.legend()
plt.savefig("hasil_prediksi_5daysinput_"+model_name+".jpeg", format='jpeg', dpi=1000)
plt.show()



def predict_wind_speed_5days(model, scaler, input_data):
    # Menambah dimensi agar sesuai dengan format yang diharapkan oleh scaler.transform()
    input_data_reshaped = np.array(input_data).reshape(timeseries, 1)  # Ubah dimensi input menjadi (5, 1)
    # Menskalakan input data
    input_scaled = scaler.transform(input_data_reshaped)
    # Menambah dimensi untuk sesuai dengan bentuk input model
    input_reshaped = np.reshape(input_scaled, (1, timeseries, 1))
    # Melakukan prediksi
    # print(input_reshaped)
    prediction_scaled = model.predict(input_reshaped)
    # Membalikkan skala hasil prediksi ke skala aslinya
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0, 0]

# Contoh penggunaan: memprediksi kecepatan angin maksimum untuk hari berikutnya berdasarkan 5 hari sebelumnya
inputan_kecepatan = scaler.inverse_transform(Y_test[-(timeseries):])
kecepatan_sebelumnya = []
for i in range (len(inputan_kecepatan)):
    kecepatan_sebelumnya.append(inputan_kecepatan[i][0])
# print(kecepatan_sebelumnya)

forecasted = []
for i in range(90):
    input_x = np.array(kecepatan_sebelumnya).reshape(-1, 1)
    prediction = predict_wind_speed_5days(model, scaler, input_x)
    forecasted.append(prediction)
    kecepatan_sebelumnya.pop(0)
    kecepatan_sebelumnya.append(prediction)


# Plot data aktual vs prediksi
plt.figure(figsize=(10,12))

# Plot data aktual
plt.plot(data.index[train_size:train_size+len(test_predict)], test_predict.flatten(), label='Data Predicition - Testing')

# Plot prediksi kecepatan angin
forecasted_dates = pd.date_range(start=data.index[train_size+len(test_predict)], periods=len(forecasted))
plt.plot(forecasted_dates, forecasted, label='Prediksi Kecepatan Angin')

# Atur label dan judul plot
plt.xlabel('Tanggal')
plt.ylabel('Kecepatan Angin')
plt.title('Prediksi Kecepatan Angin Maksimum Selama 5 Hari')

# Tampilkan legenda dan grid
plt.legend()
plt.grid(True)
plt.savefig("forecasting_"+model_name+'.jpeg', format='jpeg', dpi=1000)

# Tampilkan plot
plt.show()
