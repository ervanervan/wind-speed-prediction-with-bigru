import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Initialize weight matrices with correct dimensions
        self.Wz = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wr = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wh = np.random.randn(hidden_size, input_size + hidden_size)
        self.bz = np.zeros(hidden_size)
        self.br = np.zeros(hidden_size)
        self.bh = np.zeros(hidden_size)

    def forward(self, x, h_prev):
        # Ensure x and h_prev are reshaped correctly
        x = x.reshape(-1, self.input_size)  # should be 1 x input_size
        h_prev = h_prev.reshape(-1, self.hidden_size)  # should be 1 x hidden_size
        concat = np.concatenate((x, h_prev), axis=1)  # Results in 1 x (input_size + hidden_size)
        z = self.sigmoid(np.dot(concat, self.Wz.T) + self.bz)
        r = self.sigmoid(np.dot(concat, self.Wr.T) + self.br)
        h_hat = np.tanh(np.dot(np.concatenate((x, r * h_prev), axis=1), self.Wh.T) + self.bh)
        h_next = (1 - z) * h_prev + z * h_hat
        return h_next.squeeze()  # Return to shape (hidden_size,)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class BidirectionalGRU:
    def __init__(self, input_size, hidden_size):
        self.forward_gru = GRUCell(input_size, hidden_size)
        self.backward_gru = GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.Wy = np.random.randn(2 * hidden_size, 1)
        self.by = np.zeros(1)

    def fit(self, X, y, epochs, learning_rate=0.01):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            fwd_states = np.zeros((len(X), self.hidden_size))
            bwd_states = np.zeros((len(X), self.hidden_size))
            for t in range(len(X)):
                fwd_states_t = np.zeros(self.hidden_size)
                bwd_states_t = np.zeros(self.hidden_size)
                for i in range(X.shape[1]):
                    fwd_states_t = self.forward_gru.forward(X[t, i], fwd_states_t)
                    bwd_states_t = self.backward_gru.forward(X[t, X.shape[1] - i - 1], bwd_states_t)
                fwd_states[t] = fwd_states_t
                bwd_states[len(X) - t - 1] = bwd_states_t
                combined_state = np.concatenate([fwd_states[t], bwd_states[t]])
                y_pred = np.dot(combined_state, self.Wy) + self.by
                error = y_pred - y[t]
                total_loss += error**2
                gradient = learning_rate * np.outer(combined_state, error)
                self.Wy -= gradient
                self.by -= learning_rate * error
            mean_loss = total_loss / len(X)
            losses.append(mean_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {mean_loss}")

    def predict(self, X):
        fwd_states = np.zeros((len(X), self.hidden_size))
        bwd_states = np.zeros((len(X), self.hidden_size))
        predictions = []
        for t in range(len(X)):
            fwd_states_t = np.zeros(self.hidden_size)
            bwd_states_t = np.zeros(self.hidden_size)
            for i in range(X.shape[1]):
                fwd_states_t = self.forward_gru.forward(X[t, i], fwd_states_t)
                bwd_states_t = self.backward_gru.forward(X[t, X.shape[1] - i - 1], bwd_states_t)
            fwd_states[t] = fwd_states_t
            bwd_states[len(X) - t - 1] = bwd_states_t
            combined_state = np.concatenate([fwd_states[t], bwd_states[t]])
            predictions.append(np.dot(combined_state, self.Wy) + self.by)
        return np.array(predictions).flatten()

# Load data from URL
url = "https://raw.githubusercontent.com/ervanervan/dataset-skripsi/main/laporan_iklim_anambas_ff_x.csv"
data = pd.read_csv(url)

# Convert 'Tanggal' column to datetime
data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d-%m-%Y')

# Set 'Tanggal' as index
data.set_index('Tanggal', inplace=True)

# Use MinMaxScaler to scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create dataset function with time series window
def create_dataset(dataset, timeseries=1):
    X, Y = [], []
    for i in range(len(dataset)-timeseries):
        end_ix = i + timeseries
        X.append(dataset[i:end_ix, 0])
        Y.append(dataset[end_ix, 0])
    return np.array(X), np.array(Y)

timeseries = 5
X, Y = create_dataset(data_scaled, timeseries)
X = np.reshape(X, (X.shape[0], timeseries, 1))
print("x shape >> ", X.shape)
print("y shape >> ", Y.shape)

# Split data into training and testing sets
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = Y[:train_size], Y[train_size:]

# Instantiate and train the model
model = BidirectionalGRU(input_size=1, hidden_size=10)
model.fit(X_train, y_train, epochs=1000)

# Predictions and evaluation
y_pred = model.predict(X_test)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape}%")

# Convert predictions from list to numpy array for easier handling
y_pred = np.array(y_pred).flatten()

# Plotting the actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Wind Speed')
plt.plot(y_pred, label='Predicted Wind Speed')
plt.title('Comparison of Actual and Predicted Wind Speeds')
plt.xlabel('Time Steps')
plt.ylabel('Wind Speed')
plt.legend()
plt.grid(True)
plt.show()
