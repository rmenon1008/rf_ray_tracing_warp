import numpy as np
import trimesh as tm
from matplotlib import pyplot as plt

from tracing.tracer import *
from tracing.visualization import *
from tracing.mesh_operations import *
from tracing.signal_processing import *

MAX_BOUNCES = 4
TX_NUM_RAYS = 20_000_000
# RX_RADIUS_M = 0.001224  # From wavelength**2 / (4*pi)
RX_RADIUS_M = 0.05

# mesh = load_mesh("meshes/quarry_9.stl", scale=0.25)
# tx_pos = place_on_ground(mesh, [20, 25], offset=.5)
# rx_pos = place_on_ground(mesh, [37.5, 30], offset=.5)

# tx_pos = place_on_ground(mesh, [20, 32.5], offset=.5)
# rx_pos = place_on_ground(mesh, [20, 45], offset=.5)

mesh = tm.load_mesh("meshes/room.stl")
tx_pos = [-10, 0, 6]
rx_pos = [10, 0.5, 6]

print(f"Mesh: {mesh.bounding_box.bounds}")
print(f"TX: {tx_pos}")
print(f"RX: {rx_pos}")

tracer = Tracer(mesh, MAX_BOUNCES, TX_NUM_RAYS, RX_RADIUS_M)
paths = tracer.trace_paths(tx_pos, rx_pos)

# amp, phase = paths_to_phase_amp(paths, TX_NUM_RAYS, tx_power_watts=1, freq_hz=2.4e9)
# print(f"Amplitude: {watts_to_dbm(amp)} dBm")
# print(f"Phase: {phase} rad")

csi_amp, csi_phase = paths_to_csi(paths, TX_NUM_RAYS, tx_power_watts=1, noise_dbm=-60)
csi_amp_dbm = watts_to_dbm(csi_amp)
csi_phase_deg = np.rad2deg(csi_phase)

print(f"Center amplitude: {watts_to_dbm(csi_amp[64])} dBm")

plt.figure()
plt.plot(csi_amp_dbm)
plt.title("CSI Amplitude")
plt.xlabel("Subcarrier")
plt.ylabel("Amplitude (dBm)")
plt.savefig("csi_amp.png")

plt.figure()
plt.plot(csi_phase_deg)
plt.title("CSI Phase")
plt.xlabel("Subcarrier")
plt.ylabel("Phase (deg)")   
plt.savefig("csi_phase.png")

viz_scene(mesh, tx_pos, rx_pos, paths)
