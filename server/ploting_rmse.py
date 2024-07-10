import matplotlib.pyplot as plt


# RMSE TRAINING
# rmse_train = [0.6474, 0.6443, 0.6423, 0.6413, 0.6441, 0.6508, 0.6455, 0.6463, 
#               0.6448, 0.6447, 0.6471, 0.6464, 0.6575, 0.6479, 0.6522, 0.6530, 
#               0.6455, 0.6464, 0.6471, 0.6472]
# rmse_train = [1.7415, 1.7395, 1.7131, 1.7087, 1.7324, 1.7461, 1.7391, 1.7321, 
#               1.7208, 1.7176, 1.7456, 1.7435, 1.7658, 1.7658, 1.7421, 1.8079, 
#               1.7431, 1.7470, 1.7401, 1.7374]
# rmse_train = [0.7144, 0.7001, 0.6908, 0.6560, 0.6050, 0.7216, 0.7131, 0.7011, 
#               0.6950, 0.6970, 0.7337, 0.7237, 0.7158, 0.7387, 0.7003, 0.7312, 
#               0.7239, 0.7217, 0.7163, 0.7054]
rmse_train = [1.9747, 1.9657, 1.9642, 1.9615, 1.9235, 1.9843, 1.9749, 1.9669, 
              2.0252, 1.9567, 2.0169, 1.9812, 1.9783, 1.9677, 1.9689, 2.0717, 
              2.0117, 1.9817, 1.9795, 1.9705]
data_rmse_train = [i+1 for i in range(len(rmse_train))]
model_name = "FFXTPI"

# Plot RMSE Training dengan garis tebal dan grid horizontal
plt.figure(figsize=(14, 7))
plt.plot(data_rmse_train, rmse_train, linewidth=2)  # Set garis lebih tebal
plt.title('RMSE Data Training')
plt.ylabel('RMSE')
plt.xlabel('Percobaan')
plt.legend(['RMSE'], loc='upper right')
plt.grid(True)  # Tampilkan hanya grid horizontal
plt.savefig("RMSE_Train"+model_name+".jpeg", format='jpeg', dpi=1000)
plt.show()


# MAPE TRAINING
# mape_train = [31.35, 31.14, 31.00, 31.14, 32.16, 32.66, 31.05, 31.01, 31.14, 31.34, 
#               31.39, 31.05, 29.95, 30.97, 32.79, 31.31, 31.30, 31.02, 30.83, 31.64]
# mape_train = [22.75, 22.21, 22.81, 22.78, 21.86, 23.12, 22.99, 22.27, 23.81, 
#               22.94, 22.86, 23.54, 21.89, 21.74, 22.35, 24.95, 23.67, 22.60, 
#               23.14, 22.96]
# mape_train = [18.98, 18.97, 18.55, 17.64, 15.63, 19.09, 19.18, 18.83, 18.92, 
#               18.95, 20.21, 19.00, 19.03, 19.83, 19.08, 18.83, 19.28, 19.15, 
#               19.09, 18.94]
mape_train = [28.37, 27.83, 26.85, 26.37, 27.26, 29.75, 28.00, 27.35, 24.75, 
              28.08, 29.14, 27.49, 29.55, 28.77, 26.92, 30.20, 29.16, 27.90, 
              27.16, 27.87]
data_mape_train = [i+1 for i in range(len(mape_train))]

# Plot MAPE Training dengan garis tebal dan grid horizontal
plt.figure(figsize=(14, 7))
plt.plot(data_mape_train, mape_train, linewidth=2)  # Set garis lebih tebal
plt.title('MAPE Data Training')
plt.ylabel('MAPE (%)')
plt.xlabel('Percobaan')
plt.legend(['MAPE'], loc='upper right')
plt.grid(True)  # Tampilkan hanya grid horizontal
plt.savefig("MAPE_Train"+model_name+".jpeg", format='jpeg', dpi=1000)
plt.show()


# Data Akurasi dari gambar
# accuracy_train = [68.65, 68.86, 69.00, 68.86, 67.84, 67.34, 68.95, 68.99, 68.86, 68.66, 
#                   68.61, 68.95, 70.05, 69.03, 67.21, 68.69, 68.70, 68.98, 69.17, 68.36]
# accuracy_train = [77.25, 77.79, 77.19, 77.22, 78.14, 76.88, 77.01, 77.73, 76.19, 
#                   77.06, 77.14, 76.46, 78.11, 78.26, 77.65, 75.05, 76.33, 77.40, 
#                   76.86, 77.04]
# accuracy_train = [81.02, 81.03, 81.45, 82.36, 84.37, 80.91, 80.82, 81.17, 81.08, 
#                   81.05, 79.79, 81.00, 80.97, 80.17, 80.92, 81.17, 80.72, 80.85, 
#                   80.91, 81.06]
accuracy_train = [71.63, 72.17, 73.15, 73.63, 72.74, 70.25, 72.00, 72.65, 75.25, 
                  71.92, 70.86, 72.51, 70.45, 71.23, 73.08, 69.80, 70.84, 72.10, 
                  72.84, 72.13]
data_accuracy_train = [i+1 for i in range(len(accuracy_train))]

# Plot Akurasi Training dengan garis tebal dan grid horizontal
plt.figure(figsize=(14, 7))
plt.plot(data_accuracy_train, accuracy_train, linewidth=2)  # Set garis lebih tebal
plt.title('Akurasi Data Training')
plt.ylabel('Akurasi (%)')
plt.xlabel('Percobaan')
plt.legend(['Akurasi'], loc='upper right')
plt.grid(True)  # Tampilkan hanya grid horizontal
plt.savefig("Akurasi_Train_"+model_name+".jpeg", format='jpeg', dpi=1000)
plt.show()