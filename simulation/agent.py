import mesa
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from tracing.signal_processing import paths_to_csi, watts_to_dbm

class LunarAgent(mesa.Agent):
    ARCH = nn.Sequential(
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

    
    def __init__(self, unique_id, model, type="mobile"):
        super().__init__(unique_id, model)
        self.id = unique_id
        self.pos = None
        self.type = type
        self.csi_mag = None

        self.ml_model = self.ARCH
        self.ml_model.load_state_dict(torch.load("machine_learning/model.pt"))

    def move_2d(self, dx, dy):
        self.model.move_2d(self, dx, dy)
    
    def step(self):
        # plt.figure()
        # plt.plot(self.csi_mag)
        # plt.title("CSI Amplitude")
        # plt.xlabel("Subcarrier")
        # plt.ylabel("Amplitude (dBm)")
        # plt.savefig("csi_amp.png")
        # plt.close()
        if self.type == "mobile":
            paths = self.model.get_paths(self.unique_id, 0)
            mag, phase = paths_to_csi(paths, 1_000_000, 1)
            self.csi_mag = watts_to_dbm(mag)
            # paths = self.model.get_paths(self.unique_id, 0)
            # mag, phase = paths_to_csi(paths, 1_000_000, 1)
            # input = np.concatenate((mag, phase))
            # input = input.reshape((1, 1, input.shape[0]))
            # output = self.ml_model(torch.tensor(input, dtype=torch.float32))
            # output = output.detach().numpy()
            # output = output.reshape((3, 3))
            # output = np.argmax(output)
            # dx = output // 3 - 1
            # dy = output % 3 - 1
            self.move_2d(0, 0.1)

    def get_state(self):
        return {
            "id": self.unique_id,
            "type": self.type,
            "pos": self.pos,
            "csi_mag": self.csi_mag,
        }
