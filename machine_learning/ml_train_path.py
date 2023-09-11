import numpy as np
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DATASET_FOLDER = "dataset_path/"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create the dataset
inputs = []
outputs = []

# Iterate through each training file
for folder in os.listdir(DATASET_FOLDER):
    for file in os.listdir(DATASET_FOLDER + folder):
        # Load the data
        data = np.load(DATASET_FOLDER + folder + "/" + file)

        # Add the data to the dataset
        inputs.append(data["csi_amp_datapoints"][:-1])
        outputs.append(data["csi_amp_datapoints"][-1])

# Convert the dataset to tensors
inputs = torch.tensor(inputs, dtype=torch.float32)
outputs = torch.tensor(outputs, dtype=torch.float32)

print(f"Inputs shape: {inputs.shape}")
print(f"Outputs shape: {outputs.shape}")

# Split the dataset into training and testing sets
train_inputs = inputs[:int(len(inputs) * 0.8)]
train_outputs = outputs[:int(len(outputs) * 0.8)]
test_inputs = inputs[int(len(inputs) * 0.8):]
test_outputs = outputs[int(len(outputs) * 0.8):]

# Create the model
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.LeakyReLU(),
    nn.Linear(256, 256),
    nn.LeakyReLU(),
    nn.Linear(256, 256),
    nn.LeakyReLU(),
    
