import time
import os

import numpy as np
import trimesh as tm

from tracing.tracer import *
from tracing.visualization import *
from tracing.mesh_operations import *
from tracing.signal_processing import *

# Save data with the following columns:
# csi_amp (128x11) | csi_phase (128x11) | tx_pos (11) | rx_pos (11)

MAX_BOUNCES = 4
TX_NUM_RAYS = 5_000_000
TX_POWER_WATTS = 1
RX_RADIUS_M = 0.0198832  # From wavelength**2 / (4*pi)
HEIGHT_M = 0.5
NOISE_DBM = -85

PATH_LEN = 11
PATH_SPACING_M = 0.2

BATCH_SIZE = 100
RUNS_PER_BATCH = 100
FOLDER = "dataset_path"

# Load the mesh
mesh = load_mesh("meshes/quarry_9.stl", scale=0.25)
print(f"Mesh: {mesh.bounding_box.bounds}")
tracer = Tracer(mesh, MAX_BOUNCES, TX_NUM_RAYS, RX_RADIUS_M)

for batch in range(BATCH_SIZE):
    start = time.time()
    print(f"Batch {batch + 1}/{BATCH_SIZE}")

    batch_folder = f"{FOLDER}/batch_{batch}"
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)

    for run in range(RUNS_PER_BATCH):
        csi_amp_datapoints = np.zeros((PATH_LEN, 128))
        csi_phase_datapoints = np.zeros((PATH_LEN, 128))
        paths = []

        # First find a valid starting point that has at least one path
        while len(paths) == 0:
            tx_pos = random_pos_on_ground(mesh, offset=HEIGHT_M)
            rx_pos = random_pos_on_ground(mesh, offset=HEIGHT_M)
            paths = tracer.trace_paths(tx_pos, rx_pos)

            # Get the CSI
            csi_amp, csi_phase = paths_to_csi(paths, TX_NUM_RAYS, tx_power_watts=TX_POWER_WATTS, noise_dbm=NOISE_DBM)
            csi_amp_dbm = watts_to_dbm(csi_amp)

            # Add the CSI to the datapoints
            csi_amp_datapoints[0] = csi_amp_dbm
            csi_phase_datapoints[0] = csi_phase

        # Choose a random direction to move rx in
        direction = np.random.uniform(0, 2*np.pi)

        # Move in that direction until we have enough datapoints
        for i in range(PATH_LEN):
            # Move the rx
            rx_pos = [rx_pos[0] + PATH_SPACING_M * np.cos(direction), rx_pos[1] + PATH_SPACING_M * np.sin(direction), rx_pos[2]]
            paths = tracer.trace_paths(tx_pos, rx_pos)

            # Get the CSI
            csi_amp, csi_phase = paths_to_csi(paths, TX_NUM_RAYS, tx_power_watts=TX_POWER_WATTS, noise_dbm=NOISE_DBM)
            csi_amp_dbm = watts_to_dbm(csi_amp)

            # Add the CSI to the datapoints
            csi_amp_datapoints[i] = csi_amp_dbm
            csi_phase_datapoints[i] = csi_phase

        # Save data
        save_file = f"{batch_folder}/d_{run}.npz"
        np.savez(save_file, csi_amp_datapoints=csi_amp_datapoints, csi_phase_datapoints=csi_phase_datapoints, tx_pos=tx_pos, rx_pos=rx_pos)
        print(f"Run {run + 1}/{RUNS_PER_BATCH} | Amp: {csi_amp_datapoints[-1][64]} | Phase: {csi_phase_datapoints[-1][64]}")
    print(f"Batch {batch + 1}/{BATCH_SIZE} took {time.time() - start:.2f} seconds")
