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
        inputs.append(np.stack((data["csi_amp_datapoints"][:-1] * -1, data["csi_phase_datapoints"][:-1]), axis=0))
        outputs.append(data["csi_amp_datapoints"][-1][64] * -1)

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
# Input shape: (2, 10, 128)
# Output shape: (1, 128)
model = nn.Sequential(
    nn.Conv2d(2, 16, kernel_size=(2, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
    nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
    nn.Flatten(),
    nn.Linear(1920, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.ReLU(),
)
    
# Move everything to the GPU
model = model.to(device)
train_inputs = train_inputs.to(device)
train_outputs = train_outputs.to(device)
test_inputs = test_inputs.to(device)
test_outputs = test_outputs.to(device)

# Create the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train the model
training_losses = []
test_losses = []
for epoch in range(100):
    # Forward pass
    outputs = model(train_inputs)
    loss = loss_fn(outputs, train_outputs)
    training_losses.append(loss.item())

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Test the model
    test_outputs_pred = model(test_inputs)
    test_loss = loss_fn(test_outputs_pred, test_outputs)
    test_losses.append(test_loss.item())

    print(f"Epoch {epoch + 1}: Training loss: {loss.item()}, Test loss: {test_loss.item()}")

# Plot the losses
plt.plot(training_losses, label="Training loss")
plt.plot(test_losses, label="Test loss")
plt.legend()
plt.savefig("loss.png")

# Save the model
torch.save(model.state_dict(), "model.pt")

# Test the model with 5 random datapoints
for i in range(5):
    # Choose a random datapoint
    index = np.random.randint(0, len(test_inputs))

    # Get the input and output
    input = test_inputs[index]
    output = test_outputs[index]

    # Get the prediction
    prediction = model(input.unsqueeze(0)).item()

    # Print the results
    print(f"Input: {input}")
    print(f"Output: {output}")
    print(f"Prediction: {prediction}")
    print(f"Error: {abs(output - prediction)}")
    print()