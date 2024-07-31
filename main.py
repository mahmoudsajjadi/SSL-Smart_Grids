import torch
import torchvision
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import os

# from lightly.loss import NTXentLoss
# from lightly.models.modules import SimCLRProjectionHead
# from lightly.transforms.simclr_transform import SimCLRTransform

# Read data from CSV file
file_path = 'data/15minute_data_newyork.csv'  # Update the path accordingly
data = pd.read_csv(file_path)

# Ensure the data types are correct and convert the timestamp to datetime format
data['local_15min'] = pd.to_datetime(data['local_15min'], errors='coerce')
data['car1'] = pd.to_numeric(data['car1'], errors='coerce')
data['car2'] = pd.to_numeric(data['car2'], errors='coerce')

# Extract the date part from the timestamp for grouping
data['date'] = data['local_15min'].dt.date

# Plot EV load (sum of car1 and car2) for each HomeID
unique_homeids = data['dataid'].unique()
num_plots = len(unique_homeids)

# Determine the number of rows and columns for subplots
nrows = int(num_plots**0.5)
ncols = (num_plots + nrows - 1) // nrows

fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15), constrained_layout=True)
fig.suptitle('Daily Sum of EV Load for Each HomeID', fontsize=16)

y_min, y_max = -5, data[['car1', 'car2']].max().sum() / 4 + 5  # Set a common y-axis range

for i, homeid in enumerate(unique_homeids):
    subset = data[data['dataid'] == homeid]
    
    # Group by date and sum car1 and car2
    daily_sum = subset.groupby('date')[['car1', 'car2']].sum()
    
    row = i // ncols
    col = i % ncols
    
    ax = axs[row, col] if num_plots > 1 else axs
    
    ax.plot(daily_sum.index, (daily_sum['car1'] + daily_sum['car2']) / 4, marker='o', linestyle='-', color='b')
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
