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

# Plot EV load (sum of car1 and car2) for each dataid
unique_dataids = data['dataid'].unique()

for dataid in unique_dataids:
    subset = data[data['dataid'] == dataid]
    
    # Group by date and sum car1 and car2
    daily_sum = subset.groupby('date')[['car1', 'car2']].sum()
    
    plt.figure(figsize=(10, 5))
    plt.plot(daily_sum.index, daily_sum['car1'] + daily_sum['car2'], marker='o')
    plt.title(f'EV Load for DataID: {dataid}')
    plt.xlabel('Date')
    plt.ylabel('Daily Sum of EV Load (car1 + car2)')
    plt.grid(True)
    plt.show()
