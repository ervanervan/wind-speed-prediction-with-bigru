import matplotlib.pyplot as plt



mape = [19.31, 20, 23.25, 19.31, 20, 23.25, 19.31, 20, 23.25]
data_percobaan = []

for i in range (len(mape)) :
    data_percobaan.append(i+1)
# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(data_percobaan,mape, label='Training Loss')
plt.title('Model mape')
plt.ylabel('MAPE')
plt.xlabel('Percobaan')
plt.legend(loc='upper right')
plt.grid(True)
# plt.savefig("Loss_Plot_"+model_name+".jpeg", format='jpeg', dpi=1000)
plt.show()