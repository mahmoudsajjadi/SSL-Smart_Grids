import torch
import torchvision
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import os

# from lightly.loss import NTXentLoss
# from lightly.models.modules import SimCLRProjectionHead
# from lightly.transforms.simclr_transform import SimCLRTransform


file_path = 'data/15minute_data_newyork.csv'  
data = pd.read_csv(file_path)


data['local_15min'] = pd.to_datetime(data['local_15min'], errors='coerce')
data['car1'] = pd.to_numeric(data['car1'], errors='coerce')
data['car2'] = pd.to_numeric(data['car2'], errors='coerce')


data['date'] = data['local_15min'].dt.date

# Plot EV load (sum of car1 and car2) for each HomeID
unique_homeids = data['dataid'].unique()
num_plots = len(unique_homeids)

# Determine the number of rows and columns for subplots
nrows = int(num_plots**0.5)
ncols = (num_plots + nrows - 1) // nrows

# Calculate the global maximum value for setting a common y-axis limit
# global_max = (data[['car1', 'car2']].sum(axis=1).max() / 4) * 1.1  # 10% more than the maximum value
y_min, y_max = -2, 24

fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15), constrained_layout=True)
fig.suptitle('Daily Sum of EV Load for Each HomeID', fontsize=16)

for i, homeid in enumerate(unique_homeids):
    subset = data[data['dataid'] == homeid]
    
    # Group by date and sum car1 and car2
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

# Hide any empty subplots
for j in range(i + 1, nrows * ncols):
    fig.delaxes(axs.flatten()[j])

# Save the plot as an image file in the results folder
os.makedirs('results', exist_ok=True)
fig.savefig('results/daily_ev_loads.png', dpi=300, bbox_inches='tight')

plt.show()
