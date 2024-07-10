import matplotlib.pyplot as plt

# mape = [19.31, 20, 23.25, 19.31, 20, 23.25, 19.31, 20, 23.25]
# data_percobaan = []

# for i in range (len(mape)) :
#     data_percobaan.append(i+1)
# # Plot training & validation loss values
# plt.figure(figsize=(10, 6))
# plt.plot(data_percobaan,mape, label='Training Loss')
# plt.title('Model mape')
# plt.ylabel('MAPE')
# plt.xlabel('Percobaan')
# plt.legend(loc='upper right')
# plt.grid(True)
# # plt.savefig("Loss_Plot_"+model_name+".jpeg", format='jpeg', dpi=1000)
# plt.show()


# # RMSE TRAINING

# rmse_train = [1,3,3,2,4,3,6,3,4,1,1,2]
# data_rmse_train = []

# for i in range (len(rmse_train)) :
#     data_rmse_train.append(i+1)
# # Plot RMSE Training
# plt.figure(figsize=(14, 7))
# plt.plot(data_rmse_train,rmse_train)
# plt.title('RMSE Data Training')
# plt.ylabel('RMSE')
# plt.xlabel('Percobaan')
# plt.legend(loc='upper right')
# plt.grid(True)
# # plt.savefig("RMSE_Train"+model_name+".jpeg", format='jpeg', dpi=1000)
# plt.show()

# RMSE TRAINING
rmse_train = [0.6474, 0.6443, 0.6423, 0.6413, 0.6441, 0.6508, 0.6455, 0.6463, 
              0.6448, 0.6447, 0.6471, 0.6464, 0.6575, 0.6479, 0.6522, 0.6530, 
              0.6455, 0.6464, 0.6471, 0.6472]
data_rmse_train = [i+1 for i in range(len(rmse_train))]
model_name = "FFAVGTPI"

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
mape_train = [31.35, 31.14, 31.00, 31.14, 32.16, 32.66, 31.05, 31.01, 31.14, 31.34, 
              31.39, 31.05, 29.95, 30.97, 32.79, 31.31, 31.30, 31.02, 30.83, 31.64]
data_mape_train = [i+1 for i in range(len(mape_train))]

# Plot MAPE Training dengan garis tebal dan grid horizontal
plt.figure(figsize=(14, 7))
plt.plot(data_mape_train, mape_train, linewidth=2)  # Set garis lebih tebal
plt.title('MAPE Data Training')
plt.ylabel('MAPE')
plt.xlabel('Percobaan')
plt.legend(['MAPE'], loc='upper right')
plt.grid(True)  # Tampilkan hanya grid horizontal
plt.savefig("MAPE_Train"+model_name+".jpeg", format='jpeg', dpi=1000)
plt.show()


