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

rmse_train = [1,3,3,2,4,3,6,3,4,1,1,2]
data_rmse_train = [i+1 for i in range(len(rmse_train))]

# Plot RMSE Training dengan garis tebal dan grid horizontal
plt.figure(figsize=(14, 7))
plt.plot(data_rmse_train, rmse_train, linewidth=2)  # Set garis lebih tebal
plt.title('RMSE Data Training')
plt.ylabel('RMSE')
plt.xlabel('Percobaan')
plt.legend(['RMSE'], loc='upper right')
plt.grid(True)  # Tampilkan hanya grid horizontal
# plt.savefig("RMSE_Train"+model_name+".jpeg", format='jpeg', dpi=1000)
plt.show()


