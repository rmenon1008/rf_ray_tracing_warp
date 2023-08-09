import json
import http.server

import warp as wp
import numpy as np
import trimesh as tm
import matplotlib.pyplot as plt

from tracer import Tracer
from visualization import visualize

def to_dbm(power):
    return 10 * np.log10(power / 1e-3)

def place_on_ground(mesh, pos, offset=0):
    # Create a ray from the 2D position, pointing straight down
    ray_origin = np.array([pos[0], pos[1], 1000])
    ray_direction = np.array([0, 0, -1])

    # Find the intersection of the ray with the mesh
    ray = tm.ray.ray_triangle.RayMeshIntersector(mesh)
    intersections = ray.intersects_location(ray_origins=[ray_origin], ray_directions=[ray_direction])
    try:
        intersection = intersections[0][0]
        pos.append(intersection[2] + offset)
    except IndexError:
        print("Warning: could not find intersection with mesh")
        pos.append(0)

    return pos

LIGHT_SPEED_MPS = 2.998e8   # 299,800,000 m/s
SAMPLE_RATE_HZ = 100e9     # 10 GHz
SAMPLE_WINDOW_S = 200.0e-9  # 500 ns
MAX_BOUNCES = 4
TX_NUM_RAYS = 5_000_000

mesh = tm.load_mesh("models/quarry_9.stl") 

tx_pos = place_on_ground(mesh, [80, 100], offset=2)
rx_pos = place_on_ground(mesh, [80, 120], offset=10)

# mesh = tm.load_mesh("models/almost_empty.stl")
# tx_pos = np.array([1, 0, 1])
# rx_pos = np.array([41, 0, 1])

# mesh = tm.load_mesh("models/room.stl")
# tx_pos = np.array([10, 0, 5])
# rx_pos = np.array([-10, 0, 5])

tx_power = 1
rx_radius = 0.1

tracer = Tracer(mesh, LIGHT_SPEED_MPS, SAMPLE_RATE_HZ, SAMPLE_WINDOW_S, MAX_BOUNCES, TX_NUM_RAYS)
paths = tracer.trace_paths(tx_pos, tx_power, rx_pos, rx_radius)

# time = np.linspace(0, SAMPLE_WINDOW_S, impulse_response.shape[0])

# plt.figure()
# plt.plot(time, impulse_response)
# plt.show()
# # plt.savefig("impulse_response.png")

# signal_tx = np.sin(2 * np.pi * 2.4e9 * time)
# signal_rx = np.convolve(impulse_response, signal_tx, mode="same")
# r = np.nonzero(signal_rx)[:10000]

# signal_rx = signal_rx[r]
# signal_tx = signal_tx[r]
# time = time[r]

# signal_rx_power = np.sum(signal_rx ** 2) / signal_rx.shape[0]
# print(f"Signal RX power: {to_dbm(signal_rx_power)} dBm")

visualize(mesh, tx_pos, rx_pos, paths)
