import json
import http.server

import warp as wp
import numpy as np
import trimesh as tm
import matplotlib.pyplot as plt

from tracer import Tracer
from viz.visualization import visualize

def to_dbm(power):
    return 10 * np.log10(power / 1e-3)

LIGHT_SPEED_MPS = 2.998e8   # 299,800,000 m/s
SAMPLE_RATE_HZ = 100e9     # 10 GHz
SAMPLE_WINDOW_S = 200.0e-9  # 500 ns
MAX_BOUNCES = 4
TX_NUM_RAYS = 5_000_000

mesh = tm.load_mesh("models/apollo_17_landing_site.stl") 
tx_pos = np.array([10, 0, 4.5])
rx_pos = np.array([-10.125, 0, 4.8])

# mesh = tm.load_mesh("models/almost_empty.stl")
# tx_pos = np.array([1, 0, 1])
# rx_pos = np.array([41, 0, 1])

# mesh = tm.load_mesh("models/room.stl")
# tx_pos = np.array([10, 0, 5])
# rx_pos = np.array([-10, 0, 5])

tx_power = 1
rx_radius = 0.1

tracer = Tracer(mesh, LIGHT_SPEED_MPS, SAMPLE_RATE_HZ, SAMPLE_WINDOW_S, MAX_BOUNCES, TX_NUM_RAYS)
paths, impulse_response = tracer.compute_cir(tx_pos, tx_power, rx_pos, rx_radius)

time = np.linspace(0, SAMPLE_WINDOW_S, impulse_response.shape[0])

plt.figure()
plt.plot(time, impulse_response)
plt.show()
# plt.savefig("impulse_response.png")

signal_tx = np.sin(2 * np.pi * 2.4e9 * time)
signal_rx = np.convolve(impulse_response, signal_tx, mode="same")
r = np.nonzero(signal_rx)[:10000]

signal_rx = signal_rx[r]
signal_tx = signal_tx[r]
time = time[r]

signal_rx_power = np.sum(signal_rx ** 2) / signal_rx.shape[0]
print(f"Signal RX power: {to_dbm(signal_rx_power)} dBm")

# plt.figure()
# plt.plot(time, signal_tx)
# plt.plot(time, signal_rx)
# plt.show()


# # Find the power of the impulse response
# impulse_response_power = np.sum(impulse_response ** 2)
# print(f"Impulse response power: {to_dbm(impulse_response_power)} dBm")

visualize(mesh, tx_pos, rx_pos, paths)
