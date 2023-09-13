import argparse
from cprint import *
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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(2, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1134, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        # cprint.info('First conv')
        # cprint.warn(x.shape)
        x = F.relu(x)
        # cprint.info('First Relu')
        # cprint.warn(x.shape)
        x = self.pool(x)
        # cprint.info('First pool')
        # cprint.warn(x.shape)
        # x = self.conv2(x)
        # cprint.info('Second conv')
        # cprint.warn(x.shape)
        # x = F.relu(x)
        # cprint.info('Second Relu')
        # cprint.warn(x.shape)
        # x = self.pool(x)
        # cprint.info('Second pool')
        # cprint.warn(x.shape)
        # x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # cprint.info('flatten')
        # cprint.warn(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # cprint.info('second linear')
        # cprint.warn(x.shape)
        # x = F.relu(self.fc1(x))
        # cprint.info('third relu and fc1')
        # cprint.warn(x.shape)
        # x = F.relu(self.fc2(x))
        # cprint.info('fourth relu and fc2')
        # cprint.warn(x.shape)
        # x = self.fc3(x)
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

    net = Net()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Define the input tensor shape
    batch_size = 1
    channels = 2
    height = 9
    width = 128

    # DATASET_FOLDER = "../dataset_path/"
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
            # print(data["csi_amp_datapoints"][:8].shape)
            # print(data["csi_amp_datapoints"][-1][64])
            # Add the data to the dataset
            inputs.append(np.stack((data["csi_amp_datapoints"][:8] * -1, data["csi_phase_datapoints"][:8]), axis=0))

            # Get output
            # print(data["csi_amp_datapoints"][:8])
            # print(data["csi_amp_datapoints"][7])
            # print(data["csi_amp_datapoints"][8])
            balance = np.mean(data["csi_amp_datapoints"][8]) - np.mean(data["csi_amp_datapoints"][7])
            
            if balance > 0:
                label = np.array([[1., 0]]) # "forward"
            else:
                label = np.array([[0., 1.]]) # "backward"
            
            outputs.append(label)

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

    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                       shuffle=True, num_workers=2)
    cprint.ok('Entering training...')
    
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_inputs, 0):
        # for data in train_inputs:
            # cprint.warn(data[:,:9,:].shape)
            # cprint.warn(data[0,8,:])
            # cprint.warn(data[0,9,:])
            # cprint.warn(torch.mean(data[0,9,:]) - torch.mean(data[0,8,:]))
            data = data[np.newaxis, ...]
            # cprint.err(data)
            
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data[:,:9,:].to(device), data[1].to(device) 
            input = data[:,:1,:8,:].to(device)
            label = train_outputs[i].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(input)
            # cprint.info(output)
            # cprint.warn(label)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("losses/running_loss", running_loss, i)


            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # # Save model
    # PATH = './rf_net.pth'
    # torch.save(net.state_dict(), PATH)

    # # Test model
    # net = Net()
    # net.load_state_dict(torch.load(PATH))

    # outputs = net()

    writer.close()