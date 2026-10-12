###################################################################
### ---- Long Short-term Memory - Recurrent Neural Network ---- ###
###################################################################

##### =========================== Import Libraries =========================== #####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


### use Nvidia GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
if device.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))



##### =========================== Dataset Preprocessing =========================== #####

### reproduceability
np.random.seed(42)
torch.manual_seed(42)


dataset = pd.read_csv('apsrht_datetime.csv', index_col='date_time', parse_dates=True).asfreq('15min')['temp_C']


### dataset splitting
split_1 = int(len(dataset) * 0.80)
split_2 = int(len(dataset) * 0.90)
train_set = dataset.iloc[:split_1]
validation_set = dataset.iloc[split_1:split_2]
test_set = dataset.iloc[split_2:]


### Min-Max Scaling; scaler is only fit on the training data, otherwise data leakage would occur
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set.values.reshape(-1, 1))
validation_scaled = scaler.transform(validation_set.values.reshape(-1, 1))
test_scaled = scaler.transform(test_set.values.reshape(-1, 1))


def get_lookback_window(dataset, window_size=60):
    """
    Creates prediction datasets from time series data by splitting into
    "lookback" windows. It looks back 60 measurements to predict the next
    measurement.
    """
    X, y = [], []

    for i in range(len(dataset) - window_size):
        X.append(dataset[i:i + window_size])
        y.append(dataset[i + window_size])

    return np.array(X), np.array(y)

### X, y "lookback" sets for train/val/test
X_train, y_train = get_lookback_window(train_scaled)
X_val, y_val = get_lookback_window(validation_scaled)
X_test, y_test = get_lookback_window(test_scaled)


### convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


### final time-series prepared datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)


### torch dataloading
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



##### =========================== Neural Network =========================== #####

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        
        # output layer
        self.fc = nn.Linear(in_features=64, out_features=1)

    def forward(self, X):  # input shape: (batch_size, sequence_length, input_size)
    
        # passes input data into the LSTM
        output, (hidden, cell) = self.lstm(X)
        
        # takes the output from the final time step
        output = output[:, -1, :]
        
        # pass final pattern through linear layer to get predicted temp
        output = self.fc(output)
        
        return output


model = LSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



##### =========================== Training Loop =========================== #####

num_epochs = 50
epoch_losses = []
val_losses = []

for epoch in range(num_epochs):
    
    ### training ###
    model.train()  # set model to train mode
    train_running_loss = 0.0
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()  # clear gradients from previous iteration
        predictions = model(X)  # generate model predictions
        loss = criterion(predictions, y)  # compute loss on predictions
        loss.backward()  # back propagation; calculate gradients
        optimizer.step()  # adjust model parameters based on gradients
        train_running_loss += loss.item()
    
    epoch_loss = train_running_loss / len(train_loader)
    epoch_losses.append(epoch_loss)
    

    ### validation ###
    model.eval()
    val_running_loss = 0.0
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            loss = criterion(predictions, y)
            val_running_loss += loss.item()

    val_loss = val_running_loss / len(val_loader)
    val_losses.append(val_loss)
    
    print(f'=== Epoch {epoch + 1}/{num_epochs} ===')
    print(f'Training Loss: {epoch_loss:.6f} --- Validation Loss: {val_loss:.6f}\n')



##### =========================== Model Evaluation =========================== #####

### testing ###
model.eval()
test_running_loss = 0.0
all_test_labels = []
all_test_predictions = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        predictions = model(X)
        loss = criterion(predictions, y)
        test_running_loss += loss.item()
        all_test_predictions.extend(predictions.cpu().numpy().flatten())
        all_test_labels.extend(y.cpu().numpy().flatten())


test_loss = test_running_loss / len(test_loader)

### convert scaled predictions and labels back to original units
all_test_predictions = scaler.inverse_transform(np.array(all_test_predictions).reshape(-1, 1)).flatten()
all_test_labels = scaler.inverse_transform(np.array(all_test_labels).reshape(-1, 1)).flatten()

### calculate regression metrics
test_rmse = np.sqrt(mean_squared_error(all_test_labels, all_test_predictions))
test_mae = mean_absolute_error(all_test_labels, all_test_predictions)

print('-------------------------------------------------------------')
print('=== Test Results ===')
print(f'Test MSE: {test_loss:.6f}    (scaled)')
print(f'Test RMSE: {test_rmse:.2f} degrees_C  (unscaled)')
print(f'Test MAE: {test_mae:.2f} degrees_C   (unscaled)')



##### =========================== Visualizations =========================== #####

### training loss vs. validation loss to check for overfitting
plt.figure(figsize=(12, 5))
plt.plot(epoch_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('LSTM Training vs. Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('MSE Loss (Scaled Data)', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


### actual vs. predicted values on test set
test_dates = test_set.index[:len(all_test_predictions)]
plt.figure(figsize=(12, 5))
plt.plot(test_dates, all_test_labels, label='Actual Measured Temperature', linewidth=2)
plt.plot(test_dates, all_test_predictions, label='predicted Temperature', linewidth=2, alpha=0.8)
plt.title('Actual vs Predicted Temperatures', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.ylabel('Temperature (degrees C)', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

