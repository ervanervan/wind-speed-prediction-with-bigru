import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
print(tf.__version__)

# Load data from URL
url = 'https://raw.githubusercontent.com/ervanervan/dataset-skripsi/main/laporan_iklim_anambas_ff_avg_1.csv'
df = pd.read_csv(url)

# Convert 'Tanggal' column to datetime index
df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d-%m-%Y')
df.set_index('Tanggal', inplace=True)

# Plot time series of daily wind speed
# plt.figure(figsize=(12, 6))
# plt.plot(df.index, df['ff_avg'], color='black')
# plt.title('Time Series of Daily Wind Speed')
# plt.xlabel('Date')
# plt.ylabel('Wind Speed (m/s)')
# plt.grid(True)
# plt.show()

# Check for missing values
# print('Total num of missing values:')
# print(df.ff_avg.isna().sum())
# print('')

# Locate the missing value
# df_missing_date = df.loc[df.ff_avg.isna() == True]
# print('The date of missing value:')
# print(df_missing_date.index)

# Replace missing value with interpolation
# df.ff_avg.interpolate(inplace=True)

# Plot time series of daily wind speed after handling missing values
# plt.figure(figsize=(12, 6))
# plt.plot(df.index, df['ff_avg'], color='black')
# plt.title('Time Series of Daily Wind Speed (Interpolated)')
# plt.xlabel('Date')
# plt.ylabel('Wind Speed (m/s)')
# plt.grid(True)
# plt.show()

# Split train data and test data (70% train, 30% test)
train_size = int(len(df) * 0.7)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Print the size of train and test data
# print(f'Size of train data: {len(train_data)}')
# print(f'Size of test data: {len(test_data)}')


# Scale data using Min-Max Scaler
scaler = MinMaxScaler().fit(train_data)
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

# Print scaled train and test data
# print(train_scaled)
# print(test_scaled)

# Assuming train_scaled and test_scaled are already defined
look_back = 5  # Number of timesteps to look back

def create_dataset(X, look_back=1):
    Xs, ys = [], []
    for i in range(len(X)-look_back):
        v = X[i:i+look_back]
        Xs.append(v)
        ys.append(X[i+look_back])
    return np.array(Xs), np.array(ys)

X_train, y_train = create_dataset(train_scaled, look_back)
X_test, y_test = create_dataset(test_scaled, look_back)

# print('X_train.shape: ', X_train.shape)
# print('y_train.shape: ', y_train.shape)
# print('X_test.shape: ', X_test.shape) 
# print('y_test.shape: ', y_test.shape)

# Get shape of subset of test data
# subset_shape = X_test[:33].shape
# print('Shape of subset:', subset_shape)

# Create Bidirectional GRU model
def create_bidirectional_gru(units):
    model = Sequential()
    model.add(Bidirectional(GRU(units=units, activation='tanh', return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(units=units, return_sequences=False)))
    # model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse')
    return model

model_bidirectional_gru = create_bidirectional_gru(75)

# Fit Bidirectional GRU model
def fit_model(model):
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32, shuffle=False, callbacks=[early_stop])
    return history

history_bidirectional_gru = fit_model(model_bidirectional_gru)

# Transform data back to original data space
y_test = scaler.inverse_transform(y_test)
y_train = scaler.inverse_transform(y_train)

def plot_loss (history, model_name):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation Loss for ' + model_name)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
    plt.grid(True)
    plt.show()
 
plot_loss (history_bidirectional_gru, 'Bidirectional GRU')

# Make prediction
def prediction(model):
    prediction = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction
prediction_bigru = prediction(model_bidirectional_gru)

# Plot test data vs prediction
def plot_future(prediction, model_name, y_test):
    plt.figure(figsize=(10, 6))
    range_future = len(prediction)
    plt.plot(np.arange(range_future), np.array(y_test), 
             label='Test data')
    plt.plot(np.arange(range_future), 
             np.array(prediction),label='Prediction')
    plt.title('Test data vs prediction for ' + model_name)
    plt.legend(loc='upper left')
    plt.xlabel('Time (day)')
    plt.ylabel('Wind Speed (m/s)')
    plt.grid(True)
    plt.show()

plot_future(prediction_bigru, 'Bidirectional GRU', y_test)


def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    accuracy = 1 - np.mean(np.abs(predictions - actual) / actual)

    print(model_name + ':')
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    print('Mean Absolute Percentage Error: {:.4f}%'.format(mape))  # Format persentase
    print('Accuracy: {:.4f}%'.format(accuracy * 100))
    print('')

evaluate_prediction(prediction_bigru, y_test, 'Bidirectional GRU')


# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import pandas as pd
# import json
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, GRU, Bidirectional
# from tensorflow.keras.callbacks import Callback
# from sklearn.metrics import mean_squared_error

# # Mengimpor data
# # Anambas AVG
# url = 'https://raw.githubusercontent.com/ervanervan/dataset-skripsi/main/laporan_iklim_anambas_ff_avg_1.csv'
# data = pd.read_csv(url)

# # Mengonversi kolom 'Tanggal' ke tipe datetime
# data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d-%m-%Y')

# # Mengatur 'Tanggal' sebagai indeks
# data.set_index('Tanggal', inplace=True)

# # Menggunakan MinMaxScaler untuk menskalakan data
# scaler = MinMaxScaler(feature_range=(0, 1))
# data_scaled = scaler.fit_transform(data)

# # Modifikasi fungsi create_dataset
# def create_dataset(dataset, timeseries=1):
#     X, Y = [], []
#     for i in range(len(dataset)-timeseries):
#         end_ix = i + timeseries
#         a = dataset[i:end_ix, 0]
#         b = dataset[end_ix, 0]
#         X.append(a)
#         Y.append(b)
#     return np.array(X), np.array(Y)

# timeseries = 5
# X, Y = create_dataset(data_scaled, timeseries)
# X = np.reshape(X, (X.shape[0], timeseries, 1))

# # Membagi data menjadi 70% training dan 30% testing
# train_size = int(len(X) * 0.7)
# test_size = len(X) - train_size
# X_train, X_test = X[:train_size], X[train_size:]
# Y_train, Y_test = Y[:train_size], Y[train_size:]

# class PerformanceHistory(Callback):
#     def __init__(self):
#         self.rmse_train = []
#         self.rmse_val = []
#         self.mape_train = []
#         self.mape_val = []
    
#     def on_epoch_end(self, epoch, logs=None):
#         # Menghitung RMSE dan MAPE dari loss dan mae
#         train_rmse = np.sqrt(logs['loss'])
#         val_rmse = np.sqrt(logs['val_loss'])
#         train_mape = logs['mae'] * 100
#         val_mape = logs['val_mae'] * 100
        
#         # Menyimpan nilai RMSE dan MAPE di list
#         self.rmse_train.append(train_rmse)
#         self.rmse_val.append(val_rmse)
#         self.mape_train.append(train_mape)
#         self.mape_val.append(val_mape)

# # Modifikasi arsitektur model
# def createModel():
#     model = Sequential()
#     model.add(Bidirectional(GRU(75, activation='tanh', return_sequences=False), input_shape=(timeseries, 1)))
#     model.add(Dense(1))
#     model.summary()
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
#     return model

# model = createModel()

# # Melatih model
# def trainingModel(model):
#     callback_performance = PerformanceHistory()
#     history = model.fit(X_train, Y_train, epochs=80, batch_size=64, validation_split=0.2, callbacks=[callback_performance])
#     return history, callback_performance

# history, callback_performance = trainingModel(model)

# model_name = "Bidirectional_GRU_FF_AVG_ANAMBAS1"

# # Plot training & validation loss values
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.savefig("loss_plot_"+model_name+".jpeg", format='jpeg', dpi=1000)
# plt.show()

# # Evaluasi model
# train_predict = model.predict(X_train)
# test_predict = model.predict(X_test)

# # Inverse transform predictions and actual values to original scale
# train_predict = scaler.inverse_transform(train_predict)
# test_predict = scaler.inverse_transform(test_predict)
# Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
# Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))
# actual_train = scaler.inverse_transform(Y_train)
# actual_test = scaler.inverse_transform(Y_test)

# # Fungsi untuk menghitung metrik evaluasi
# def calculate_metrics(actual, predicted):
#     mse = mean_squared_error(actual, predicted)
#     rmse = np.sqrt(mse)
#     mape = np.mean(np.abs((actual - predicted) / actual)) * 100
#     accuracy = 100 - mape
#     return rmse, mape, accuracy

# # Kalkulasi metrik untuk data pelatihan
# rmse_train, mape_train, akurasi_train = calculate_metrics(actual_train, train_predict)
# print('\nTraining Data:')
# print(f'Train RMSE: {rmse_train:.4f}')
# print(f'Train MAPE: {mape_train:.2f}%')
# print(f'Train Accuracy: {akurasi_train:.2f}%')

# # Kalkulasi metrik untuk data pengujian
# rmse_test, mape_test, akurasi_test = calculate_metrics(actual_test, test_predict)
# print('\nTesting Data:')
# print(f'Test RMSE: {rmse_test:.4f}')
# print(f'Test MAPE: {mape_test:.2f}%')
# print(f'Test Accuracy: {akurasi_test:.2f}%')
# print('\n')

# BiGRU_Layer = model.layers[0]
# BiGRU_Weight = BiGRU_Layer.get_weights()
# input_weight = BiGRU_Weight[0][0]
# bias = BiGRU_Weight[2][0]

# def convert_to_serializable(obj):
#     if isinstance(obj, np.float32):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# # Membuat dictionary untuk menyimpan kinerja model
# model_performance = {
#     'Training Performance': {
#         'RMSE': float(rmse_train),
#         'MAPE': float(mape_train),
#         'Accuracy': float(akurasi_train)
#     },
#     'Testing Performance': {
#         'RMSE': float(rmse_test),
#         'MAPE': float(mape_test),
#         'Accuracy': float(akurasi_test)
#     },
#     'Model Name': model_name,
#     'Bobot': list(map(float, input_weight)),
#     'Bias': list(map(float, bias))
# }

# # Menyimpan kinerja model ke dalam file JSON
# with open(model_name + '.json', 'w') as json_file:
#     json.dump(model_performance, json_file, indent=4, default=convert_to_serializable)

# model_filename = "BiGRUFFAVGANB1.keras"
# model.save(model_filename)

# # Plot RMSE dan MAPE
# plt.figure(figsize=(14, 7))

# # Plotting RMSE
# plt.subplot(1, 2, 1)
# plt.plot(callback_performance.rmse_train, label='Training RMSE')
# plt.plot(callback_performance.rmse_val, label='Validation RMSE')
# plt.title('Training and Validation RMSE')
# plt.xlabel('Epoch')
# plt.ylabel('RMSE')
# plt.legend()

# # Plotting MAPE
# plt.subplot(1, 2, 2)
# plt.plot(callback_performance.mape_train, label='Training MAPE')
# plt.plot(callback_performance.mape_val, label='Validation MAPE')
# plt.title('Training and Validation MAPE')
# plt.xlabel('Epoch')
# plt.ylabel('MAPE (%)')
# plt.legend()

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(data.index[:train_size], actual_train.flatten(), label='Data Aktual - Training')
# plt.plot(data.index[:train_size], train_predict.flatten(), label='Prediksi - Training')
# plt.plot(data.index[train_size:train_size+len(actual_test)], actual_test.flatten(), label='Data Aktual - Testing')
# plt.plot(data.index[train_size:train_size+len(test_predict)], test_predict.flatten(), label='Prediksi - Testing')
# plt.xlabel('Tanggal')
# plt.ylabel('Kecepatan Angin Rata-Rata')
# plt.title('Prediksi vs Data Aktual Kecepatan Angin Rata-Rata')
# plt.legend()
# plt.savefig("hasil_prediksi_90days_"+model_name+".jpeg", format='jpeg', dpi=1000)
# plt.show()

# def predict_wind_speed_90days(model, scaler, input_data):
#     input_data_reshaped = np.array(input_data).reshape(timeseries, 1)
#     input_scaled = scaler.transform(input_data_reshaped)
#     input_reshaped = np.reshape(input_scaled, (1, timeseries, 1))
#     prediction_scaled = model.predict(input_reshaped)
#     prediction = scaler.inverse_transform(prediction_scaled)
#     return prediction[0, 0]

# inputan_kecepatan = scaler.inverse_transform(Y_test[-(timeseries):])
# kecepatan_sebelumnya = [inputan_kecepatan[i][0] for i in range(len(inputan_kecepatan))]

# forecasted = []
# for i in range(90):
#     input_x = np.array(kecepatan_sebelumnya).reshape(-1, 1)
#     prediction = predict_wind_speed_90days(model, scaler, input_x)
#     forecasted.append(prediction)
#     kecepatan_sebelumnya.pop(0)
#     kecepatan_sebelumnya.append(prediction)

# plt.figure(figsize=(10, 12))
# plt.plot(data.index[train_size:train_size+len(test_predict)], test_predict.flatten(), label='Data Predicition - Testing')
# forecasted_dates = pd.date_range(start=data.index[train_size+len(test_predict)], periods=len(forecasted))
# plt.plot(forecasted_dates, forecasted, label='Prediksi Kecepatan Angin')
# plt.xlabel('Tanggal')
# plt.ylabel('Kecepatan Angin')
# plt.title('Prediksi Kecepatan Angin Rata-Rata Selama 90 Hari')
# plt.legend()
# plt.grid(True)
# plt.savefig("forecasting_"+model_name+'.jpeg', format='jpeg', dpi=1000)
# plt.show()





# import os
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, GRU, Bidirectional
# from tensorflow.keras.callbacks import Callback
# from sklearn.metrics import mean_squared_error
# import json

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Mengimpor data
# url = 'https://raw.githubusercontent.com/ervanervan/dataset-skripsi/main/laporan_iklim_anambas_ff_avg_1.csv'
# data = pd.read_csv(url)

# # Mengonversi kolom 'Tanggal' ke tipe datetime dan mengatur sebagai indeks
# data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d-%m-%Y')
# data.set_index('Tanggal', inplace=True)

# # Membagi data menjadi training dan testing sebelum scaling
# train_size = int(len(data) * 0.7)
# test_size = len(data) - train_size
# train_data, test_data = data[:train_size], data[train_size:]

# # Scaling data
# scaler = MinMaxScaler(feature_range=(0, 1))
# train_scaled = scaler.fit_transform(train_data)
# test_scaled = scaler.transform(test_data)

# # Fungsi untuk membuat dataset
# def create_dataset(dataset, timeseries=1):
#     X, Y = [], []
#     for i in range(len(dataset)-timeseries):
#         end_ix = i + timeseries
#         a = dataset[i:end_ix, 0]
#         b = dataset[end_ix, 0]
#         X.append(a)
#         Y.append(b)
#     return np.array(X), np.array(Y)

# timeseries = 5
# X_train, Y_train = create_dataset(train_scaled, timeseries)
# X_test, Y_test = create_dataset(test_scaled, timeseries)

# # Reshaping untuk model GRU
# X_train = X_train.reshape((X_train.shape[0], timeseries, 1))
# X_test = X_test.reshape((X_test.shape[0], timeseries, 1))

# # Kelas Callback untuk melacak performance
# class PerformanceHistory(Callback):
#     def __init__(self):
#         self.rmse_train = []
#         self.rmse_val = []
#         self.mape_train = []
#         self.mape_val = []

#     def on_epoch_end(self, epoch, logs=None):
#         train_rmse = np.sqrt(logs['loss'])
#         val_rmse = np.sqrt(logs['val_loss'])
#         train_mape = logs['mae'] * 100
#         val_mape = logs['val_mae'] * 100
#         self.rmse_train.append(train_rmse)
#         self.rmse_val.append(val_rmse)
#         self.mape_train.append(train_mape)
#         self.mape_val.append(val_mape)

# # Fungsi untuk membangun model GRU
# def create_model():
#     model = Sequential([
#         Bidirectional(GRU(75, activation='tanh', return_sequences=False), input_shape=(timeseries, 1)),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
#     return model

# model = create_model()
# model.summary()
# model_name = "Bidirectional_GRU_FF_AVG_ANAMBAS"

# # Fungsi untuk melatih model
# def train_model(model, X, Y):
#     callback = PerformanceHistory()
#     history = model.fit(X, Y, epochs=80, batch_size=64, validation_split=0.2, callbacks=[callback])
#     return history, callback

# history, callback = train_model(model, X_train, Y_train)

# # Plot training & validation loss values
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.savefig("loss" + model_name + '.jpeg', format='jpeg', dpi=1000)
# plt.show()

# # Evaluasi model menggunakan data test
# test_predict = model.predict(X_test)
# train_predict = model.predict(X_train)

# # Inverse transform predictions and actual values to original scale
# train_predict = scaler.inverse_transform(train_predict)
# test_predict = scaler.inverse_transform(test_predict)
# Y_train_inv = scaler.inverse_transform(Y_train.reshape(-1, 1))
# Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

# # Fungsi untuk menghitung metrik evaluasi
# def calculate_metrics(actual, predicted):
#     mse = mean_squared_error(actual, predicted)
#     rmse = np.sqrt(mse)
#     mape = np.mean(np.abs((actual - predicted) / actual)) * 100
#     accuracy = 100 - mape
#     return rmse, mape, accuracy

# # Kalkulasi metrik untuk data pelatihan dan pengujian
# rmse_train, mape_train, accuracy_train = calculate_metrics(Y_train_inv, train_predict)
# rmse_test, mape_test, accuracy_test = calculate_metrics(Y_test_inv, test_predict)

# print('Training Data:')
# print(f'Train RMSE: {rmse_train:.4f}')
# print(f'Train MAPE: {mape_train:.2f}%')
# print(f'Train Accuracy: {accuracy_train:.2f}%')

# print('Testing Data:')
# print(f'Test RMSE: {rmse_test:.4f}')
# print(f'Test MAPE: {mape_test:.2f}%')
# print(f'Test Accuracy: {accuracy_test:.2f}%')

# # Menyimpan model
# model_path = model_name + ".keras"
# model.save(model_path)
# print(f"Model saved as {model_path}")

# BiGRU_Layer = model.layers[0]
# BiGRU_Weight = BiGRU_Layer.get_weights()
# input_weight = BiGRU_Weight[0][0]
# bias = BiGRU_Weight[2][0]

# def convert_to_serializable(obj):
#     if isinstance(obj, np.float32):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# # Membuat dictionary untuk menyimpan kinerja model
# model_performance = {
#     'Training Performance': {
#         'RMSE': float(rmse_train),
#         'MAPE': float(mape_train),
#         'Accuracy': float(accuracy_train)
#     },
#     'Testing Performance': {
#         'RMSE': float(rmse_test),
#         'MAPE': float(mape_test),
#         'Accuracy': float(accuracy_test)
#     },
#     'Model Name': model_name,
#     'Bobot': list(map(float, input_weight)),
#     'Bias': list(map(float, bias))
# }

# # Menyimpan kinerja model ke dalam file JSON
# with open(model_name + '.json', 'w') as json_file:
#     json.dump(model_performance, json_file, indent=4, default=convert_to_serializable)


# # Plot RMSE dan MAPE
# plt.figure(figsize=(14, 7))
# plt.subplot(1, 2, 1)
# plt.plot(callback.rmse_train, label='Training RMSE')
# plt.plot(callback.rmse_val, label='Validation RMSE')
# plt.title('Training and Validation RMSE')
# plt.xlabel('Epoch')
# plt.ylabel('RMSE')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(callback.mape_train, label='Training MAPE')
# plt.plot(callback.mape_val, label='Validation MAPE')
# plt.title('Training and Validation MAPE')
# plt.xlabel('Epoch')
# plt.ylabel('MAPE (%)')
# plt.legend()
# plt.tight_layout()
# plt.savefig("rmse_and_mape" + model_name + '.jpeg', format='jpeg', dpi=1000)
# plt.show()

# # Plot prediksi dan data aktual
# plt.figure(figsize=(10, 6))
# plt.plot(data.index[:len(Y_train_inv)], Y_train_inv.flatten(), label='Data Aktual - Training')
# plt.plot(data.index[:len(train_predict)], train_predict.flatten(), label='Prediksi - Training')
# plt.plot(data.index[len(Y_train_inv):len(Y_train_inv) + len(Y_test_inv)], Y_test_inv.flatten(), label='Data Aktual - Testing')
# plt.plot(data.index[len(train_predict):len(train_predict) + len(test_predict)], test_predict.flatten(), label='Prediksi - Testing')
# plt.xlabel('Tanggal')
# plt.ylabel('Kecepatan Angin Rata-Rata')
# plt.title('Prediksi vs Data Aktual Kecepatan Angin Rata-Rata')
# plt.legend()
# plt.grid(True)
# plt.savefig("actual_and_prediction" + model_name + '.jpeg', format='jpeg', dpi=1000)
# plt.show()

# # Fungsi untuk memprediksi kecepatan angin selama 90 hari
# def predict_forecasting(model, scaler, input_data):
#     input_data_reshaped = np.array(input_data).reshape(-1, 1)  # Reshape data input
#     input_scaled = scaler.transform(input_data_reshaped)  # Scale the input
#     input_reshaped = np.reshape(input_scaled, (1, timeseries, 1))  # Reshape for the model
#     prediction_scaled = model.predict(input_reshaped)  # Predict
#     prediction = scaler.inverse_transform(prediction_scaled)  # Inverse scaling
#     return prediction[0, 0]

# # Mengambil 5 data terakhir dari testing set sebagai starting point
# timeseries_speeds_input = scaler.inverse_transform(Y_test[-timeseries:].reshape(-1, 1))
# timeseries_speeds = [timeseries_speeds_input[i][0] for i in range(len(timeseries_speeds_input))]

# forecasted = []
# for i in range(90):
#     input_x = np.array(timeseries_speeds[-timeseries:])  # Get the last 'timeseries' values
#     prediction = predict_forecasting(model, scaler, input_x)
#     forecasted.append(prediction)
#     timeseries_speeds.append(prediction)  # Append the new prediction to the list

# # Visualisasi prediksi kecepatan angin untuk 90 hari
# plt.figure(figsize=(10, 6))
# plt.plot(data.index[train_size:train_size+len(test_predict)], test_predict.flatten(), label='Data Prediksi - Testing')
# forecasted_dates = pd.date_range(start=data.index[train_size+len(test_predict)], periods=90)
# plt.plot(forecasted_dates, forecasted, label='Prediksi Kecepatan Angin 90 Hari')
# plt.xlabel('Tanggal')
# plt.ylabel('Kecepatan Angin')
# plt.title('Prediksi Kecepatan Angin Rata-Rata Selama 90 Hari')
# plt.legend()
# plt.grid(True)
# plt.savefig("forecasting_" + model_name + '.jpeg', format='jpeg', dpi=1000)
# plt.show()