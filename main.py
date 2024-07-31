import torch
import torchvision
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt

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

for i, homeid in enumerate(unique_homeids):
    subset = data[data['dataid'] == homeid]
    
    # Group by date and sum car1 and car2
    daily_sum = subset.groupby('date')[['car1', 'car2']].sum()
    
    row = i // ncols
    col = i % ncols
    
    ax = axs[row, col] if num_plots > 1 else axs
    
    ax.plot(daily_sum.index, (daily_sum['car1'] + daily_sum['car2']) / 4, marker='o')
    ax.set_title(f'EV Load for HomeID: {homeid}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Sum of EV Load (kWh)')
    ax.grid(True)

# Hide any empty subplots
for j in range(i + 1, nrows * ncols):
    fig.delaxes(axs.flatten()[j])

plt.show()
