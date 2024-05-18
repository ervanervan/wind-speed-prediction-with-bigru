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
# Anambas X
url = 'https://raw.githubusercontent.com/ervanervan/dataset-skripsi/main/laporan_iklim_harian_tanjungpinang_ff_x.csv'
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
        a = dataset[i:end_ix, 0]
        b = dataset[end_ix, 0]
        X.append(a)
        Y.append(b)
    return np.array(X), np.array(Y)

timeseries = 5
X, Y = create_dataset(data_scaled, timeseries)
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
    history = model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1, validation_split=0.2)
    return history

history = trainingModel(model)

model_name = "Bidirectional_GRU_FF_X_TANJUNGPINANG"

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig("loss_plot_"+model_name+".jpeg", format='jpeg', dpi=1000)
plt.show()

# Evaluasi model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

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

model_performance = {
    'MAPE': mape,
    'AKURASI': akurasi,
    'model_name': model_name
}

with open(model_name+'.json', 'w') as json_file:
    json.dump(model_performance, json_file, indent=4)

model_filename = "BiGRUFFXTPI.keras"
model.save(model_filename)

plt.figure(figsize=(10, 6))
plt.plot(data.index[:train_size], actual_train.flatten(), label='Data Aktual - Training')
plt.plot(data.index[:train_size], train_predict.flatten(), label='Prediksi - Training')
plt.plot(data.index[train_size:train_size+len(actual_test)], actual_test.flatten(), label='Data Aktual - Testing')
plt.plot(data.index[train_size:train_size+len(test_predict)], test_predict.flatten(), label='Prediksi - Testing')
plt.xlabel('Tanggal')
plt.ylabel('Kecepatan Angin Maksimum')
plt.title('Prediksi vs Data Aktual Kecepatan Angin Maksimum')
plt.legend()
plt.savefig("hasil_prediksi_90days_"+model_name+".jpeg", format='jpeg', dpi=1000)
plt.show()

def predict_wind_speed_90days(model, scaler, input_data):
    input_data_reshaped = np.array(input_data).reshape(timeseries, 1)
    input_scaled = scaler.transform(input_data_reshaped)
    input_reshaped = np.reshape(input_scaled, (1, timeseries, 1))
    prediction_scaled = model.predict(input_reshaped)
    prediction = scaler.inverse_transform(prediction_scaled)
    return prediction[0, 0]

inputan_kecepatan = scaler.inverse_transform(Y_test[-(timeseries):])
kecepatan_sebelumnya = [inputan_kecepatan[i][0] for i in range(len(inputan_kecepatan))]

forecasted = []
for i in range(90):
    input_x = np.array(kecepatan_sebelumnya).reshape(-1, 1)
    prediction = predict_wind_speed_90days(model, scaler, input_x)
    forecasted.append(prediction)
    kecepatan_sebelumnya.pop(0)
    kecepatan_sebelumnya.append(prediction)

plt.figure(figsize=(10, 12))
plt.plot(data.index[train_size:train_size+len(test_predict)], test_predict.flatten(), label='Data Predicition - Testing')
forecasted_dates = pd.date_range(start=data.index[train_size+len(test_predict)], periods=len(forecasted))
plt.plot(forecasted_dates, forecasted, label='Prediksi Kecepatan Angin')
plt.xlabel('Tanggal')
plt.ylabel('Kecepatan Angin')
plt.title('Prediksi Kecepatan Angin Maksimum Selama 90 Hari')
plt.legend()
plt.grid(True)
plt.savefig("forecasting_"+model_name+'.jpeg', format='jpeg', dpi=1000)
plt.show()
