import numpy as np
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DATASET_PATH = "dataset/"
inputs = []
outputs = []

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Go through each file in the dataset folder
for file in os.listdir(DATASET_PATH):
    # Load the data
    data = np.load(DATASET_PATH + file)

    # Add the data to the dataset
    inputs.append(np.concatenate((data["csi_amp"], data["csi_phase"])))
    # inputs.append((data["csi_amp"]))
    outputs.append((data["local_map"]).flatten())

# Graph a sample of the dataset
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(inputs[i])
    plt.title(outputs[i])
    plt.
plt.savefig("sample.png")

# Convert the dataset to tensors
inputs = torch.tensor(inputs, dtype=torch.float32)
outputs = torch.tensor(outputs, dtype=torch.float32)

# Add a dimension
inputs = inputs.reshape((inputs.shape[0], 1, inputs.shape[1]))

# Print the shapes of the inputs and outputs
print(f"Inputs shape: {inputs.shape}")
print(f"Outputs shape: {outputs.shape}")

# Split the dataset into training and testing sets
train_inputs = inputs[:int(len(inputs) * 0.8)]
train_outputs = outputs[:int(len(outputs) * 0.8)]
test_inputs = inputs[int(len(inputs) * 0.8):]
test_outputs = outputs[int(len(outputs) * 0.8):]

## Create the model
# model = nn.Sequential(
#     nn.Linear(128, 256),
#     nn.LeakyReLU(),
#     nn.Linear(256, 256),
#     nn.LeakyReLU(),
#     nn.Linear(256, 256),
#     nn.LeakyReLU(),
#     nn.Linear(256, 256),
#     nn.LeakyReLU(),
#     nn.Linear(256, 256),
#     nn.LeakyReLU(),
#     nn.Linear(256, 128),
#     nn.LeakyReLU(),
#     nn.Linear(128, 64),
#     nn.LeakyReLU(),
#     nn.Linear(64, 9),
#     nn.Flatten()
# )

model = nn.Sequential(
    nn.Conv1d(1, 16, 4, stride=2),
    nn.LeakyReLU(),
    nn.Conv1d(16, 32, 4, stride=2),
    nn.LeakyReLU(),
    nn.Conv1d(32, 64, 4, stride=2),
    nn.LeakyReLU(),
    nn.Conv1d(64, 32, 4, stride=2),
    nn.LeakyReLU(),
    nn.Conv1d(32, 16, 4, stride=2),
    nn.LeakyReLU(),
    nn.Flatten(),
    nn.Linear(32, 9),
    nn.LeakyReLU(),
    nn.Linear(9, 9),
)

# model = nn.Sequential(
#     nn.Conv1d(1, 16, 4, stride=2),
#     nn.LeakyReLU(),
#     nn.Conv1d(16, 32, 4, stride=2),
#     nn.LeakyReLU(),
#     nn.Conv1d(32, 64, 4, stride=2),
#     nn.LeakyReLU(),
#     nn.Conv1d(64, 32, 4, stride=2),
#     nn.LeakyReLU(),
#     nn.Conv1d(32, 16, 4, stride=2),
#     nn.Flatten(),
#     nn.Linear(32, 9)
# )

# Create the loss function
loss_fn = nn.MSELoss()

# Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Load the model
try:
    model.load_state_dict(torch.load("model.pt"))
    print("Loaded previous model")
except:
    print("Failed to load previous model")

# Move the model and data to the GPU
model.to(device)
train_inputs = train_inputs.to(device)
train_outputs = train_outputs.to(device)
test_inputs = test_inputs.to(device)
test_outputs = test_outputs.to(device)

# Train the model and plot the loss and accuracy
train_losses = []
test_losses = []
for epoch in range(1000):
    # Permuate the training data
    permutation = torch.randperm(train_inputs.size()[0])
    train_inputs = train_inputs[permutation]
    train_outputs = train_outputs[permutation]

    # Forward pass
    y_pred = model(train_inputs)

    # Compute loss
    loss = loss_fn(y_pred, train_outputs)
    train_losses.append(loss.item())

    # Zero gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # Compute accuracy
    with torch.no_grad():
        y_pred = model(test_inputs)
        accuracy = 1 - torch.mean(torch.abs(y_pred - test_outputs) / torch.abs(test_outputs))
        test_loss = loss_fn(y_pred, test_outputs)
        test_losses.append(test_loss.item())

    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}\t| Train: {loss.item()}\t| Test: {test_loss.item()}\t| Accuracy: {accuracy}")

# Plot the loss and accuracy (log scale)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.yscale("log")
plt.legend()
plt.savefig("loss.png")

# Save the model
torch.save(model.state_dict(), "model.pt")

# Evaluate the model with 5 random test cases
with torch.no_grad():
    for i in range(5):
        # Get a random test case
        index = np.random.randint(0, len(test_inputs))
        input = test_inputs[index]
        output = test_outputs[index]

        input = input.reshape((1, 1, 128))

        # Make a prediction
        prediction = model(input)

        # Print the prediction and actual output
        print(f"Input: {input}")
        print(f"Output: {output}")
        print(f"Prediction: {prediction}")
        print(f"Accuracy: {1 - torch.mean(torch.abs(prediction - output) / torch.abs(output))}")
        print()