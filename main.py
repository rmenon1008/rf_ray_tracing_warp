import json
import http.server

import warp as wp
import numpy as np
import trimesh as tm
import matplotlib.pyplot as plt
from trimesh import viewer

from tracer import Tracer
from viz.visualization import visualize

mesh = tm.load_mesh("models/apollo_17_landing_site.stl") 
tx_pos = np.array([10, 0, 4.5])
tx_power = 1
rx_pos = np.array([-10, 0, 4.8])
rx_radius = 0.1

tracer = Tracer(mesh)
paths, impulse_response = tracer.compute_cir(tx_pos, tx_power, rx_pos, rx_radius)
visualize(paths, tx_pos, rx_pos, mesh)
