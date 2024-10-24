import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load data
data = pd.read_csv("/content/drive/My Drive/7Train1.csv", header=None)

# Use the `add_big_drop_label_train` function for both train and test datasets
def add_big_drop_label_train(bandwidth, threshold=5):
    big_drop = []
    for i in range(0, len(bandwidth)-1):
        diff = abs(bandwidth[i+1] - bandwidth[i])
        big_drop.append(1 if diff >= threshold else 0)
    return big_drop + [0]




bandwidth = data.iloc[:, 0].values  # Convert to NumPy array
length = len(bandwidth)
mean = np.mean(bandwidth)
len_train = math.floor(length * 0.8)
for i in range(length):
    if bandwidth[i] > 40:
        bandwidth[i] = 0

window_size = 5
predict_size = 1
batch_size = 4

# Add 'big drop' label to bandwidth data
big_drop_labels = torch.tensor(add_big_drop_label_train(bandwidth), dtype=torch.float32)

scaler = MinMaxScaler(feature_range=(-1, 1))
# Normalize bandwidth as before
bandwidth_normalized = scaler.fit_transform(bandwidth.reshape(-1, 1))

# Combine the bandwidth and big drop labels into a single input feature set
input_features = np.column_stack((bandwidth_normalized, big_drop_labels))

# Convert to PyTorch tensor
data_tensor = torch.FloatTensor(input_features)

# Function to create in-out sequences for bandwidth and big drop labels
def create_inout_sequences_with_big_drop(input_data, big_drop_labels, window_size, predict_size):
    inout_seq = []
    L = len(input_data)
    for i in range(L - window_size - predict_size + 1):
        train_seq = input_data[i:i + window_size, :]
        train_label_bandwidth = input_data[i + window_size:i + window_size + predict_size, 0]  # Predict only bandwidth
        train_label_big_drop = big_drop_labels[i + window_size:i + window_size + predict_size]
        inout_seq.append((train_seq, (train_label_bandwidth, train_label_big_drop)))
    return inout_seq

# Update sequences creation
train_inout_seq = create_inout_sequences_with_big_drop(data_tensor[:len_train], big_drop_labels[:len_train], window_size, predict_size)
test_inout_seq = create_inout_sequences_with_big_drop(data_tensor[len_train:], big_drop_labels[len_train:], window_size, predict_size)

# Custom dataset class remains the same
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# Create DataLoader for training
train_dataset = TimeSeriesDataset(train_inout_seq)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Define LSTM model to take both bandwidth and big drop as input
class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=128, output_size_bandwidth=1, output_size_classification=1, num_layers=2, dropout=0.5):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, dropout=dropout, batch_first=True)
        self.linear_bandwidth = nn.Linear(hidden_layer_size, output_size_bandwidth)
        self.linear_classification = nn.Linear(hidden_layer_size, output_size_classification)
          # For binary classification

    def forward(self, input_seq, hidden_state):
        lstm_out, hidden_state = self.lstm(input_seq, hidden_state)
        lstm_out_last = lstm_out[:, -1, :]  # Select the last time step output
        bandwidth_prediction = self.linear_bandwidth(lstm_out_last)  # Apply linear layer for bandwidth
        big_drop_prediction = self.linear_classification(lstm_out_last) # Apply linear layer and sigmoid for big drop classification
        return bandwidth_prediction, big_drop_prediction, hidden_state

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_layer_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_layer_size))

# Initialize the model, loss function, and optimizer
model = LSTM(input_size=2, dropout=0.2, num_layers=2)  # Adjust the dropout rate and number of layers as needed
mse_loss_function = nn.MSELoss()
bce_loss_function = nn.MSELoss()  # For big drop classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs =15
train_losses = []

for epoch in range(epochs):
    epoch_train_loss = 0
    for seq, (labels_bandwidth, labels_big_drop) in train_loader:
        batch_size = seq.size(0)  # Get the batch size dynamically
        hidden_state = model.init_hidden(batch_size)  # Initialize hidden state with correct batch size

        optimizer.zero_grad()

        # Reshape seq to have the correct dimensions for LSTM input
        seq = seq.view(batch_size, window_size, -1)  # (batch_size, window_size, features)
        labels_bandwidth = labels_bandwidth.view(batch_size, -1)  # (batch_size, 1)
        labels_big_drop = labels_big_drop.view(batch_size, -1)    # (batch_size, 1)

        # Forward pass
        y_pred_bandwidth, y_pred_big_drop, hidden_state = model(seq, hidden_state)

        # Calculate the losses for both bandwidth and classification
        bandwidth_loss = mse_loss_function(y_pred_bandwidth, labels_bandwidth)
        classification_loss = bce_loss_function(y_pred_big_drop, labels_big_drop)
        total_loss = bandwidth_loss + classification_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        epoch_train_loss += total_loss.item()

    train_losses.append(epoch_train_loss / len(train_loader))

    if epoch % 5 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Total Loss: {epoch_train_loss:.8f}')

# Testing loop
model.eval()
hidden_state = model.init_hidden(1)

predictions = []
big_drop_predictions = []
for seq, _ in test_inout_seq:
    seq = seq.view(1, window_size, 2)  # Adjust input size to include bandwidth and big drop

    with torch.no_grad():
        y_pred_bandwidth, y_pred_big_drop, hidden_state = model(seq, hidden_state)
        predictions.append(y_pred_bandwidth.item())
        big_drop_predictions.append(y_pred_big_drop.item())

# Convert predictions back to original scale for bandwidth
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Calculate actual values
actual_values = bandwidth[len_train + window_size + predict_size - 1:]

# Calculate MAE and RMSE for all test data
mae = mean_absolute_error(actual_values, predictions)
rmse = np.sqrt(mean_squared_error(actual_values, predictions))
import matplotlib.pyplot as plt

# Mean Absolute Error (MAE) and RMSE Calculation
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
error_ratio_rmse = (rmse / np.mean(actual_values)) * 100
error_ratio_mae = (mae / np.mean(actual_values)) * 100
print(f'Error Ratio RMSE: {error_ratio_rmse:.4f}%')
print(f'Error Ratio MAE: {error_ratio_mae:.4f}%')

# Slice for 200 values for plotting
plot_length = 200
predictions_200 = predictions[:plot_length]
actual_values_200 = actual_values[:plot_length]

# Now we find the corresponding "big drop" labels for both actual and predicted values
big_drop_actual_200 = big_drop_labels[len_train + window_size:len_train + window_size + plot_length]
big_drop_pred_200 = big_drop_predictions[:plot_length]

# Plot the actual and predicted values
plt.figure(figsize=(12, 6))
x_range = range(len_train + window_size, len_train + window_size + plot_length)

# Plot the actual bandwidth values
plt.plot(x_range, actual_values_200, label='Actual Data', color='blue', linewidth=1.5)

# Plot the predicted bandwidth values
plt.plot(x_range, predictions_200, label='Predicted Data', color='orange', linestyle='--', linewidth=1.5)

# Add blue dots for actual big drops
for i in range(plot_length):
    if big_drop_actual_200[i] == 1:
        plt.scatter(x_range[i], actual_values_200[i], color='blue', s=100, marker='o', label='Actual Big Drop' if i == 0 else "")

# Add red dots for predicted big drops
for i in range(plot_length):
    if big_drop_pred_200[i] == 1:
        plt.scatter(x_range[i], predictions_200[i], color='red', s=100, marker='o', label='Predicted Big Drop' if i == 0 else "")

# Add labels and legend
plt.legend(loc='upper right')
plt.xlabel("Index")
plt.ylabel("Bandwidth")
plt.title("Actual vs Predicted Bandwidth with Big Drops (200 Samples)")
plt.show()
