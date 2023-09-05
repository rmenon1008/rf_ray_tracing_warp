import json

import numpy as np
from trimesh.exchange.export import export_dict

from mesa.visualization.ModularVisualization import VisualizationElement

def serialize_recursively(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [serialize_recursively(item) for item in obj]
    elif isinstance(obj, tuple):
        return [serialize_recursively(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_recursively(value) for key, value in obj.items()}
    else:
        return obj

class LunarVis(VisualizationElement):
    local_includes = [
        "simulation/web/controls.js",
        "simulation/web/controls.css",
        # "simulation/web/three.js",
        # "simulation/web/orbit_controls.js",
        "simulation/web/lunar_vis.js",
        "simulation/web/lunar_vis.css",
        # "simulation/web/json_formatter.min.js",
        "simulation/web/Linefont[wdth,wght].woff2"
    ]

    def __init__(self, mesh):
        self.mesh = mesh
        new_element = "new LunarVis({})".format(export_dict(mesh))
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        agents = []
        for agent in model.schedule.agents:
            agents.append(agent.get_state())
        return serialize_recursively({
            "agents": agents,
            "paths": model.paths,
        })