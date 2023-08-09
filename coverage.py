import warp as wp
import numpy as np
import trimesh as tm
import matplotlib.pyplot as plt

from tracer import Tracer
from visualization import visualize

def to_dbm(power):
    return 10 * np.log10(power / 1e-3)

LIGHT_SPEED_MPS = 2.998e8   # 299,800,000 m/s
SAMPLE_RATE_HZ = 100e9     # 10 GHz
SAMPLE_WINDOW_S = 100.0e-9  # 500 ns
MAX_BOUNCES = 2
TX_NUM_RAYS = 1_000_000

# mesh = tm.load_mesh("models/room.stl")
# tx_pos = np.array([10, 0, 5])

# mesh = tm.load_mesh("models/apollo_17_landing_site.stl") 
# tx_pos = np.array([10, 0, 4.5])


tx_power = 1
rx_radius = 0.1

tracer = Tracer(mesh, LIGHT_SPEED_MPS, SAMPLE_RATE_HZ, SAMPLE_WINDOW_S, MAX_BOUNCES, TX_NUM_RAYS)
point_color_pairs = []

colors = plt.get_cmap("viridis")
def dbm_to_color(dbm):
    min = -130
    max = -70
    val = (dbm - min) / (max - min)
    return colors(val)

for x in range(-15, 16, 2):
    for y in range(-15, 16, 2):
        for z in range(0, 16, 2):
            print(f"Computing CIR for ({x}, {y}, {z})...")
            rx_pos = np.array([x, y, z])
            paths, impulse_response = tracer.compute_cir(tx_pos, tx_power, rx_pos, rx_radius)

            time = np.linspace(0, SAMPLE_WINDOW_S, impulse_response.shape[0])
            signal_tx = np.sin(2 * np.pi * 2.4e9 * time)
            signal_rx = np.convolve(impulse_response, signal_tx, mode="same")
            r = np.nonzero(signal_rx)[:10000]

            signal_rx = signal_rx[r]
            signal_tx = signal_tx[r]
            time = time[r]

            signal_rx_power = np.sum(signal_rx ** 2) / signal_rx.shape[0]
            print(f"Signal RX power: {to_dbm(signal_rx_power)} dBm")

            point_color_pairs.append((rx_pos, dbm_to_color(to_dbm(signal_rx_power))))

visualize(mesh, tx_pos=tx_pos, rx_pos=None, paths=None, point_color_pairs=point_color_pairs)
