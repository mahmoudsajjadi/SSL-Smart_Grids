import torch
import torchvision.transforms as transforms
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the SSL Model
class SSLModel(nn.Module):
    def __init__(self):
        super(SSLModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.projection_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
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

# Prepare data for SSL
ssl_data = []
for homeid in home_ids:
    subset = data[data['dataid'] == homeid]
    daily_sum = subset.groupby('date')[['car1', 'car2']].sum().reset_index()
    ssl_data.append(daily_sum[['car1', 'car2']].values)

ssl_data = np.concatenate(ssl_data, axis=0)
ssl_data = torch.tensor(ssl_data, dtype=torch.float32)

# Define a percentage of data points to mask
mask_percentage = 0.2
num_mask = int(mask_percentage * ssl_data.size(0))

# Create masks
mask = torch.randperm(ssl_data.size(0))[:num_mask]
masked_data = ssl_data.clone()
masked_data[mask] = 0  # Mask the selected data points

# Create SSL Model instance
model = SSLModel()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the SSL Model
epochs = 100
batch_size = 32

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
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/((ssl_data.size(0)//batch_size)+1):.4f}')

print('Training finished.')

# Save the model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/ssl_model.pth')

# Plot EV load and predicted values for each HomeID
fig, axs = plt.subplots(len(home_ids), 1, figsize=(15, 5*len(home_ids)), constrained_layout=True)
fig.suptitle('Original and Predicted EV Load for Selected HomeIDs', fontsize=16)

for i, homeid in enumerate(home_ids):
    subset = data[data['dataid'] == homeid]
    daily_sum = subset.groupby('date')[['car1', 'car2']].sum().reset_index()
    
    original_values = daily_sum[['car1', 'car2']].values
    original_values = torch.tensor(original_values, dtype=torch.float32)
    predicted_values = model(original_values).detach().numpy()
    
    ax = axs[i]
    
    ax.plot(daily_sum['date'], (original_values[:, 0] + original_values[:, 1]) / 4, linestyle='-', color='b', label='Original')
    ax.plot(daily_sum['date'], (predicted_values[:, 0] + predicted_values[:, 1]) / 4, linestyle='--', color='r', label='Predicted')
    ax.set_title(f'HomeID: {homeid}', fontsize=12)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('EV Load (kWh)', fontsize=10)
    ax.grid(True)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=10)

os.makedirs('results', exist_ok=True)
fig.savefig('results/ev_loads_comparison.png', dpi=300, bbox_inches='tight')

plt.show()
