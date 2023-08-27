import mesa

from mesh_operations import *

class MeshSpace:
    def __init__(self, mesh):
        self.mesh = mesh
        self.bounds = mesh.bounding_box.bounds

class LunarModel(mesa.Model):
    def __init__(self, mesh, num_agents = 1):
        self.num_agents = num_agents
        self.schedule = mesa.time.RandomActivation(self)

        for i in range(self.num_agents):
            a = LunarAgent(i, self)
            a.set_pos(random_pos_on_ground(self.mesh, offset=0.1))
            self.schedule.add(a)

    def step(self):
        self.schedule.step()

class LunarAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.pos = None

    def set_pos(self, pos):
        self.pos = pos

    def step(self):
        pass
