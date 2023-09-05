import time
import os

import numpy as np
import trimesh as tm
import PIL

from tracing.tracer import *
from tracing.visualization import *
from tracing.mesh_operations import *
from tracing.signal_processing import *

MAX_BOUNCES = 4
TX_NUM_RAYS = 10_000_000
TX_POWER_WATTS = 1
RX_RADIUS_M = 0.0198832  # From wavelength**2 / (4*pi)

mesh = load_mesh("meshes/quarry_9.stl", scale=0.25)
print(f"Mesh: {mesh.bounding_box.bounds}")

tracer = Tracer(mesh, MAX_BOUNCES, TX_NUM_RAYS, RX_RADIUS_M)

def random_noise_amount():
    return np.random.uniform(-50, -90)

def rand_height():
    return np.random.uniform(0.4, 1.0)

# Save data with the following columns:
# csi_amp (64) | csi_phase (64) | tx_pos (3) | rx_pos (3) | local_map (9)

RUNS = 8000

# Find the largest run in the dataset folder
largest_run = 0
for file in os.listdir("dataset/"):
    if file.endswith(".npz"):
        run = int(file.split("_")[-1].split(".")[0])
        if run > largest_run:
            largest_run = run

for run in range(largest_run, largest_run + RUNS):
    paths = []
    rx_height = rand_height()
    while len(paths) == 0:
        tx_pos = random_pos_on_ground(mesh, offset=rand_height())
        rx_pos = random_pos_on_ground(mesh, offset=rx_height)
        paths = tracer.trace_paths(tx_pos, rx_pos)

    local_map = np.zeros((3, 3))
    for x in range(0, 3):
        for y in range(0, 3):
            pos = [tx_pos[0] + (x - 1), tx_pos[1] + (y - 1), tx_pos[2]]
            local_map[x, y] = watts_to_dbm(paths_to_phase_amp(tracer.trace_paths(pos, rx_pos), TX_NUM_RAYS, TX_POWER_WATTS, 2.401e9)[0])
    
    noise_dbm = random_noise_amount()
    csi_amp, csi_phase = paths_to_csi(paths, TX_NUM_RAYS, tx_power_watts=TX_POWER_WATTS, noise_dbm=noise_dbm)
    csi_amp_dbm = watts_to_dbm(csi_amp)

    # Save data
    np.savez(f"dataset/d_{run}.npz", csi_amp=csi_amp_dbm, csi_phase=csi_phase, tx_pos=tx_pos, rx_pos=rx_pos, local_map=local_map, noise_dbm=noise_dbm)

    center_amp = watts_to_dbm(csi_amp[32])
    print(f"Run {run + 1}/{RUNS + largest_run} | Amplitude: {center_amp:.2f} dBm | Noise: {noise_dbm:.2f} dBm")