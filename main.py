import torch
import torchvision
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform

# Read data from CSV file
file_path = 'data/15minute_data_newyork.csv'  # Update the path accordingly
data = pd.read_csv(file_path)

# Ensure the data types are correct and convert the timestamp to datetime format
data['local_15min'] = pd.to_datetime(data['local_15min'], format='%m/%d/%Y %H:%M')
data['car1'] = pd.to_numeric(data['car1'], errors='coerce')

# Plot EV load for each dataid
unique_dataids = data['dataid'].unique()

for dataid in unique_dataids:
    subset = data[data['dataid'] == dataid]
    plt.figure(figsize=(10, 5))
    plt.plot(subset['local_15min'], subset['car1'], marker='o')
    plt.title(f'EV Load for DataID: {dataid}')
    plt.xlabel('Time')
    plt.ylabel('EV Load (car1)')
    plt.grid(True)
    plt.show()
    
    
    
