import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the SSL Model
class SSLModel(nn.Module):
    def __init__(self):
        super(SSLModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.projection_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.projection_head(x)
        return x

# Define MAE Loss
def mae_loss(predicted, target):
    return torch.mean(torch.abs(predicted - target))

# Load and preprocess data
file_path = 'data/15minute_data_newyork.csv'
data = pd.read_csv(file_path)

data['local_15min'] = pd.to_datetime(data['local_15min'], errors='coerce')
data['car1'] = pd.to_numeric(data['car1'], errors='coerce')
data['car2'] = pd.to_numeric(data['car2'], errors='coerce')
data['date'] = data['local_15min'].dt.date

# Select specific HomeIDs
home_ids = [1222, 3000, 9053]

# Define a percentage of data points to mask
mask_percentage = 0.8

# Training parameters
epochs = 5
batch_size = 32
learning_rate = 0.001

# Function to train the SSL model
def train_ssl_model(train_data, mask_percentage, epochs, batch_size, learning_rate):
    ssl_data = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)

    num_mask = int(mask_percentage * ssl_data.size(0))
    mask = torch.randperm(ssl_data.size(0))[:num_mask]
    masked_data = ssl_data.clone()
    masked_data[mask] = 0  # Mask the selected data points

    model = SSLModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        perm = torch.randperm(ssl_data.size(0))
        epoch_loss = 0

        for i in range(0, ssl_data.size(0), batch_size):
            batch_indices = perm[i:i + batch_size]
            batch_data = masked_data[batch_indices]

            optimizer.zero_grad()

            # Forward pass
            output = model(batch_data)

            # SSL target
            target = ssl_data[batch_indices]  # Target is the original unmasked data

            # Calculate MAE Loss
            loss = mae_loss(output, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / ((ssl_data.size(0) // batch_size) + 1):.4f}')

    print('Training finished.')
    return model, masked_data

# Plot EV load consumption for each HomeID
fig, axs = plt.subplots(len(home_ids), 1, figsize=(15, 5 * len(home_ids)), constrained_layout=True)
fig.suptitle('EV Load Consumption for Selected HomeIDs', fontsize=16)

for i, homeid in enumerate(home_ids):
    subset = data[data['dataid'] == homeid]
    daily_sum = subset.groupby('date')[['car1', 'car2']].sum().reset_index()

    ev_load = (daily_sum['car1'] + daily_sum['car2']) / 4.0  # 4 is to convert to kWh

    # Plot EV load consumption
    ax = axs[i]
    ax.plot(daily_sum['date'], ev_load.values, linestyle='-', color='b', linewidth=2, label=f'HomeID: {homeid}')
    ax.set_title(f'EV Load Consumption for HomeID: {homeid}', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('EV Load (kWh)', fontsize=12)
    ax.grid(True)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=10)

os.makedirs('results', exist_ok=True)
fig.savefig('results/ev_loads_consumption.png', dpi=300, bbox_inches='tight')

plt.show()

# Plot original vs masked data and predictions
fig, axs = plt.subplots(len(home_ids), 1, figsize=(15, 5 * len(home_ids)), constrained_layout=True)
fig.suptitle('Original vs Masked EV Load Data for Selected HomeIDs', fontsize=16)

for i, homeid in enumerate(home_ids):
    subset = data[data['dataid'] == homeid]
    daily_sum = subset.groupby('date')[['car1', 'car2']].sum().reset_index()

    ev_load = (daily_sum['car1'] + daily_sum['car2']) / 4.0  # 4 is to convert to kWh

    # Split the data into training and testing sets
    split_idx = int(0.8 * len(ev_load))
    train_data = ev_load[:split_idx].values

    # Train the model
    model, masked_data = train_ssl_model(train_data, mask_percentage, epochs, batch_size, learning_rate)

    # Plot original vs masked data
    ax = axs[i]
    ax.plot(daily_sum['date'][:split_idx], train_data, linestyle='-', color='b', linewidth=2, label='Original')
    ax.plot(daily_sum['date'][:split_idx], masked_data.squeeze().numpy(), linestyle='--', color='g', linewidth=2, label='Masked')
    ax.set_title(f'HomeID: {homeid} - Original vs Masked Data', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('EV Load (kWh)', fontsize=12)
    ax.grid(True)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=10)

os.makedirs('results', exist_ok=True)
fig.savefig('results/ev_loads_original_vs_masked.png', dpi=300, bbox_inches='tight')

plt.show()

# Plot training, testing, and original data with predictions
fig, axs = plt.subplots(len(home_ids), 1, figsize=(15, 5 * len(home_ids)), constrained_layout=True)
fig.suptitle('Training, Testing, and Original Data for Selected HomeIDs', fontsize=16)

for i, homeid in enumerate(home_ids):
    subset = data[data['dataid'] == homeid]
    daily_sum = subset.groupby('date')[['car1', 'car2']].sum().reset_index()

    ev_load = (daily_sum['car1'] + daily_sum['car2']) / 4.0  # 4 is to convert to kWh

    # Split the data into training and testing sets
    split_idx = int(0.8 * len(ev_load))
    train_data = ev_load[:split_idx].values
    test_data = ev_load[split_idx:].values

    # Train the model
    model, masked_data = train_ssl_model(train_data, mask_percentage, epochs, batch_size, learning_rate)

    # Predict on the test data
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1) 
    predicted_test_data = model(test_data_tensor).detach().numpy()

    # Plot training, testing, and original data
    ax = axs[i]
    ax.plot(daily_sum['date'][:split_idx], train_data, linestyle='-', color='b', linewidth=2, label='Training Data')
    ax.plot(daily_sum['date'][split_idx:], test_data, linestyle='--', color='r', linewidth=2, label='Testing Data')
    ax.plot(daily_sum['date'][split_idx:], predicted_test_data, linestyle=':', color='green', linewidth=2, label='Predicted Test Data')
    ax.set_title(f'HomeID: {homeid} - Training, Testing, and Original Data', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('EV Load (kWh)', fontsize=12)
    ax.grid(True)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=10)

os.makedirs('results', exist_ok=True)
fig.savefig('results/ev_loads_training_testing_original.png', dpi=300, bbox_inches='tight')

plt.show()
