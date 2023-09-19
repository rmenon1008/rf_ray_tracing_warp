import numpy as np
import matplotlib.pyplot as plt

DATASET_LENGTH = 1000

signal_power_increases = lambda a : a + 1
signal_power_decreases = lambda a : a - 1

signal_increases = np.arange(8) / 10
signal_decreases = np.flip(signal_increases, 0)

print(f'signal_increases {signal_increases} shape: {signal_increases.shape}')
print(f'signal_decreases {signal_decreases}')


# plt.plot(np.arange(signal_increases.shape[0]), signal_increases, color='blue')
# plt.plot(np.arange(signal_increases.shape[0]), signal_decreases, color='orange')
# 
# plt.show()

signal_increases = np.tile(signal_increases, (128, 1)).T
signal_increases = signal_increases[np.newaxis, ...]
signal_increases = np.tile(signal_increases, (int(DATASET_LENGTH / 2), 1, 1, 1))

signal_decreases = np.tile(signal_decreases, (128, 1)).T
signal_decreases = signal_decreases[np.newaxis, ...]
signal_decreases = np.tile(signal_decreases, (int(DATASET_LENGTH / 2), 1, 1, 1))

dataset = np.concatenate((signal_increases, signal_decreases))
rng = np.random.default_rng()
rng.shuffle(dataset, axis=0)
labels = np.round(dataset[:,:,7:8,127:128])
labels = np.squeeze(labels, axis=1)

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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.apply(self._init_weights)
        # 1 Channel
        # self.conv1 = nn.Conv2d(2, 6, kernel_size=(2, 3))
        # self.pool = nn.MaxPool2d(2, (1, 2))
        # self.conv2 = nn.Conv2d(6, 16, kernel_size=(2, 3))
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=(1, 2))
        # self.fc1 = nn.Linear(1344, 64)
        # # self.fc2 = nn.Linear(512, 64)
        # self.fc3 = nn.Linear(64, 1)
        # self.sig = nn.Sigmoid()

        # 2 Channels
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(2, 10))
        self.pool = nn.MaxPool2d(2, (2, 2))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(2, 10))
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=(1, 2))
        self.fc1 = nn.Linear(400, 64)
        # self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        # cprint.info('Second conv')
        # cprint.warn(x.shape)
        x = F.relu(x)
        # cprint.info('Second Relu')
        # cprint.warn(x.shape)
        x = self.pool(x)
        # cprint.info('Second pool')
        # cprint.warn(x.shape)
        # x = self.conv3(x)
        # cprint.info('Second conv')
        # cprint.warn(x.shape)
        # x = F.relu(x)
        # cprint.info('Third Relu')
        # cprint.warn(x.shape)
        # x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # cprint.info('flatten')
        # cprint.warn(x.shape)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc3(x))
        # x = self.sig(self.fc3(x))
        x = self.tanh(self.fc3(x))
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(128, 8)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(8, 128)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 8)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
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

    # net = Net()
    net = Deep()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # Define the input tensor shape
    batch_size = 1
    channels = 2
    height = 9
    width = 128

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
            # if torch.randint(0,10,(1,)) == 1:
            # print(data[0][:9])
            test = torch.mean(data[0], axis = 1)
            # cprint.info(test.shape)
            # cprint.info(test)

            # writer.add_figure('input', img)

            data = data[np.newaxis, ...]
            
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data[:,:9,:].to(device), data[1].to(device) 
            input = data[:,:1,:8,:].to(device)
            label = train_outputs[i].to(device)

            # plt.plot(torch.arange(test.shape[0]), test)
            # cprint.info(f'label: {label}')
            # cprint.ok(f'mean: {test}')
            # plt.show()

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
            writer.add_scalar("losses/running_loss", loss.item(), i)


            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # # Save model
    PATH = './rf_net.pth'
    torch.save(net.state_dict(), PATH)

    # Test model
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.to(device)

    acc = 0
    for i in range(100):
        rand = torch.randint(0, test_inputs.shape[0], (1,))
        # data = test_inputs[np.newaxis, ...]
        test_input = test_inputs[rand,:1,:8,:]
        input = test_input.to(device)
        outputs = net(input)
        print('outputs', outputs, 'test_outputs', test_outputs[[rand]])
        # test = torch.mean(test_input[0], axis = 1)
        # plt.plot(torch.arange(test.shape[0]), test)
        # plt.show()
        if test_outputs[[rand]] == 0 and outputs[[0]] < 0.5:
            acc += 1
        elif test_outputs[[rand]] == 1 and outputs[[0]] > 0.5:
            acc += 1
    acc = (acc / 20) * 100
    print('accuracy %', acc)
    # Test accuracy
    acc = 0
    for i in range(test_inputs.shape[0]):
        test_input = test_inputs[torch.tensor([i]),:1,:8,:]
        input = test_input.to(device)
        outputs = net(input)
        if test_outputs[[i]] == 0 and outputs[[0]] < 0.5:
            acc += 1
        elif test_outputs[[i]] == 1 and outputs[[0]] > 0.5:
            acc += 1
    acc = (acc / test_inputs.shape[0]) * 100
    print('accuracy %', acc)
        
    writer.close()

    # NOTE Run "tensorboard --logdir runs" to see results