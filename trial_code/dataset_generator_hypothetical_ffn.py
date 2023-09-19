import numpy as np
import matplotlib.pyplot as plt
from cprint import *
from scipy.stats import linregress

DATASET_LENGTH = 100
DATASET_NOISE = 0
TRAJECTORY_LENGTH = 8
FREQUENCY_BINS = 128
SIGNAL_NOISE = 1
SIGNAL_SLOPE_NOISE = .5

signal_increases = np.arange(TRAJECTORY_LENGTH) / 10
cprint.info(f'signal_increases {signal_increases.shape}')
signal_decreases = np.flip(signal_increases, 0)

print(f'signal_increases {signal_increases} shape: {signal_increases.shape}')
print(f'signal_decreases {signal_decreases}')


# plt.plot(np.arange(signal_increases.shape[0]), signal_increases, color='blue')
# plt.plot(np.arange(signal_increases.shape[0]), signal_decreases, color='orange')
# 
# plt.show()

signal_increases = np.tile(signal_increases, (FREQUENCY_BINS, 1)).T
cprint.info(f'signal_increases {signal_increases.shape}')
signal_increases = signal_increases[np.newaxis, ...]
cprint.info(f'signal_increases {signal_increases.shape}')
signal_increases = np.tile(signal_increases, (int(DATASET_LENGTH / 2), 1, 1, 1))
cprint.info(f'signal_increases {signal_increases.shape}')
# Add noise - second dimension: if magnitude only then 1. If phase included then 2.
noise_signal = np.random.normal(loc=0, scale=SIGNAL_NOISE, size=(int(DATASET_LENGTH / 2), 1, TRAJECTORY_LENGTH,FREQUENCY_BINS))
signal_increases = signal_increases + noise_signal
# Alter slope
slope = np.random.normal(loc=1, scale=SIGNAL_SLOPE_NOISE, size=(int(DATASET_LENGTH / 2), 1, 1, 1))
signal_increases = signal_increases * slope
cprint.info(f'signal_increases {signal_increases.shape}')
# for i in range(5):
#     rand = np.random.randint(int(DATASET_LENGTH / 2))
#     plt.plot(np.arange(8), np.squeeze(signal_increases[rand:rand+1,:1,:,:]))
#     plt.show()

signal_decreases = np.tile(signal_decreases, (FREQUENCY_BINS, 1)).T
signal_decreases = signal_decreases[np.newaxis, ...]
signal_decreases = np.tile(signal_decreases, (int(DATASET_LENGTH / 2), 1, 1, 1))
# Add noise - second dimension: if magnitude only then 1. If phase included then 2.
noise_signal = np.random.normal(loc=0, scale=SIGNAL_NOISE, size=(int(DATASET_LENGTH / 2), 1, TRAJECTORY_LENGTH,FREQUENCY_BINS))
signal_decreases = signal_decreases + noise_signal
# Alter slope
slope = np.random.normal(loc=1, scale=SIGNAL_SLOPE_NOISE, size=(int(DATASET_LENGTH / 2), 1, 1, 1))
signal_decreases = signal_decreases * slope
cprint.info(f'signal_increases {signal_decreases.shape}')
# for i in range(5):
#     rand = np.random.randint(int(DATASET_LENGTH / 2))
#     plt.plot(np.arange(8), np.squeeze(signal_decreases[rand:rand+1,:1,:,:]))
#     plt.show()

# Join increasing and decreasing signals
dataset = np.concatenate((signal_increases, signal_decreases))
rng = np.random.default_rng()
rng.shuffle(dataset, axis=0)

# Create labels as per slope of signal
labels = np.zeros(dataset.shape[0])
cprint.info(f'dataset {dataset.shape}')
for i in range(dataset.shape[0]):
    data = np.mean(np.squeeze(dataset[i]),axis=1)
    labels[i] = linregress(np.arange(data.shape[0]), data).slope
labels = np.where(labels >= 0, 1, 0)
cprint.info(labels)

# # Create labels according to last two datapoints
# labels = np.zeros(dataset.shape[0])
# cprint.info(f'dataset {dataset.shape}')
# last_two_points = np.mean(dataset[:,:,6:7,:]) - np.mean(dataset[:,:,7:8,:])
# cprint.err(last_two_points)
# labels = np.where(labels >= 0, 1, 0)
# cprint.info(labels)

# Introduce noise to dataset via labels
# print(labels.T)
rands = np.random.randint(labels.shape[0], size=int(DATASET_NOISE / 100 * labels.shape[0]))
labels[rands] = np.where(labels[rands] == 1, 0, 1)
print(labels)

import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

import torchvision.transforms as T
from PIL import Image

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=10_000)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cnn_csi",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    
    args = parser.parse_args()
    # args.batch_size = int(args.num_envs * args.num_steps)
    # args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1024, 128)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 16)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

class MyNeuralNetwork(nn.Module):
    def __init__(self, input_size = 1024):
        super(MyNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50,2)
    
    def forward(self, x):
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # define a transform to convert a tensor to PIL image
    transform = T.ToPILImage()

    # net = Deep()
    net = MyNeuralNetwork()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Convert the dataset to tensors
    inputs = torch.tensor(dataset, dtype=torch.float32)
    outputs = torch.tensor(labels, dtype=torch.float32)

    print(f"Inputs shape: {inputs.shape}")
    print(f"Outputs shape: {outputs.shape}")

    # Split the dataset into training and testing sets
    train_inputs = inputs[:int(len(inputs) * 0.8)]
    train_outputs = outputs[:int(len(outputs) * 0.8)]
    test_inputs = inputs[int(len(inputs) * 0.8):]
    test_outputs = outputs[int(len(outputs) * 0.8):]

    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                       shuffle=True, num_workers=2)
    cprint.ok('Entering training...')
    
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_inputs, 0):

            test = torch.mean(data[0], axis = 1)

            data = np.squeeze(data, 0)
            data = data.flatten()
            # get the inputs; data is a list of [inputs, labels]
            input = data.to(device)
            train_outputs = train_outputs.squeeze()
            label = train_outputs[i:i+1].type(torch.LongTensor).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("losses/running_loss", loss.item(), i)


            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # # Save model
    PATH = './rf_net.pth'
    torch.save(net.state_dict(), PATH)

    # Test model
    # net = Deep()
    net = MyNeuralNetwork()
    net.load_state_dict(torch.load(PATH))
    net.to(device)

    acc = 0
    samples = 100
    for i in range(samples):
        rand = torch.randint(0, test_inputs.shape[0], (1,))
        test_input = test_inputs[rand,:1,:8,:]
        test_input = test_input.squeeze()
        test_input = test_input.flatten()
        input = test_input.to(device)
        outputs = net(input)
        cprint.warn(f'outputs {outputs}')
        cprint.info(f'softmax {F.softmax(outputs, dim = 1)}')
        cprint.info(f'argmax {F.softmax(outputs, dim = 1).argmax()}')
        prediction = F.softmax(outputs, dim = 1).argmax()
        # test = torch.mean(test_input[0], axis = 1)
        # plt.plot(torch.arange(test.shape[0]), test)
        # plt.show()
        cprint.info(f'prediction {prediction} label {test_outputs[rand]}')
        if test_outputs[[rand]] == prediction.cpu():
            acc += 1
    acc = (acc / samples) * 100
    print('accuracy %', acc)
    # Test accuracy
    acc = 0
    for i, data in enumerate(test_inputs, 0):
        test = torch.mean(data[0], axis = 1)
        data = np.squeeze(data, 0)
        data = data.flatten()
        # get the inputs; data is a list of [inputs, labels]
        input = data.to(device)
        outputs = net(input)
        prediction = F.softmax(outputs, dim = 1).argmax()
        if test_outputs[[i]] == prediction.cpu():
            acc += 1
    acc = (acc / test_inputs.shape[0]) * 100
    print('accuracy %', acc)
    print(test_outputs)
    cprint.info(f'total positive slopes {test_outputs[test_outputs == 1].shape}')
    cprint.err(f'test_inputs length {test_inputs.shape}')
    writer.close()

    # NOTE Run "tensorboard --logdir runs" to see results