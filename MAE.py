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

# Prepare data for SSL
unique_homeids = data['dataid'].unique()
ssl_data = []

for homeid in unique_homeids:
    subset = data[data['dataid'] == homeid]
    daily_sum = subset.groupby('date')[['car1', 'car2']].sum().reset_index()
    ssl_data.append(daily_sum[['car1', 'car2']].values)

ssl_data = np.concatenate(ssl_data, axis=0)
ssl_data = torch.tensor(ssl_data, dtype=torch.float32)

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
        batch_data = ssl_data[batch_indices]
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch_data)
        
        # SSL target
        target = batch_data  # In self-supervised learning, target is the input itself
        
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

# Plot EV load (sum of car1 and car2) for each HomeID
num_plots = len(unique_homeids)
nrows = int(num_plots ** 0.5)
ncols = (num_plots + nrows - 1) // nrows

y_min, y_max = -2, 24

fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15), constrained_layout=True)
fig.suptitle('Daily Sum of EV Load for Each HomeID', fontsize=16)

for i, homeid in enumerate(unique_homeids):
    subset = data[data['dataid'] == homeid]
    daily_sum = subset.groupby('date')[['car1', 'car2']].sum()
    
    row = i // ncols
    col = i % ncols
    
    ax = axs[row, col] if num_plots > 1 else axs
    
    ax.plot(daily_sum.index, (daily_sum['car1'] + daily_sum['car2']) / 4, linestyle='-', color='b')
    ax.set_title(f'HomeID: {homeid}', fontsize=12)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('EV Load (kWh)', fontsize=10)
    ax.grid(True)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=10)

for j in range(i + 1, nrows * ncols):
    fig.delaxes(axs.flatten()[j])

os.makedirs('results', exist_ok=True)
fig.savefig('results/daily_ev_loads.png', dpi=300, bbox_inches='tight')

plt.show()
