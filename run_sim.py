import json

import mesa

from simulation.model import LunarModel
from simulation.vis import LunarVis
from tracing.mesh_operations import load_mesh

MESH_PATH = "meshes/quarry_9.stl"
NUM_AGENTS = 2
mesh = load_mesh(MESH_PATH, scale=0.25)

class ObjectOption(mesa.visualization.UserParam):
    def __init__(self, name="", value=None, choices=None, description=None):
        self.param_type = "object"
        self.name = name
        self._value = json.dumps(value)

    @property
    def value(self):
        return json.loads(self._value)

    @value.setter
    def value(self, value):
        self._value = value

vis = LunarVis(mesh)
server = mesa.visualization.ModularServer(
    LunarModel,
    [vis],
    "Model",
    {
        "mesh": mesh,
        "num_agents": NUM_AGENTS,
    },
    8000
)
server.settings["template_path"] = "simulation/web"
server.launch(open_browser=False)