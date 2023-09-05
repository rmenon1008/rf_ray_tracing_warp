import mesa
from simulation.agent import LunarAgent
from simulation.space import MeshSpace
from tracing.tracer import Tracer
from tracing.signal_processing import paths_to_csi

class LunarModel(mesa.Model):
    MAX_BOUNCES = 3
    TX_NUM_RAYS = 1_000_000
    RX_RADIUS_M = 0.2  # From wavelength**2 / (4*pi)
    ANTENNA_HEIGHT_M = 0.6

    def __init__(self, mesh, num_agents = 2):
        self.mesh = mesh
        self.num_agents = num_agents
        self.schedule = mesa.time.RandomActivation(self)
        self.space = MeshSpace(mesh)
        self.tracer = Tracer(mesh, self.MAX_BOUNCES, self.TX_NUM_RAYS, self.RX_RADIUS_M)
        self.paths = {}

        a = LunarAgent(0, self, type="fixed")
        self.teleport_agent(a, self.space.random_pos_on_ground(offset=self.ANTENNA_HEIGHT_M))
        self.schedule.add(a)

        a = LunarAgent(1, self)
        self.teleport_agent(a, self.space.random_pos_on_ground())
        self.schedule.add(a)

        self.step()

    def step(self):
        self.schedule.step()
        self.__update_paths()

    def __update_paths(self):
        paths = {}
        # Get all pairs of agents
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                pos1 = self.schedule.agents[i].pos
                pos2 = self.schedule.agents[j].pos
                print(f"Tracing path from {pos1} to {pos2}")
                paths[f"{i}-{j}"] = self.tracer.trace_paths(pos1, pos2)
        self.paths = paths
        print(f"Paths: {len(paths)}")

    def teleport_agent(self, agent, pos):
        agent.pos = pos

    def move_2d(self, agent, dx, dy):
        new_pos = self.space.place_on_ground([agent.pos[0] + dx, agent.pos[1] + dy, agent.pos[2]], offset=self.ANTENNA_HEIGHT_M)
        if self.space.pos_valid(new_pos):
            agent.pos = new_pos
        else:
            print(f"Agent {agent.unique_id} moved illegally")

    def get_paths(self, agent_tx, agent_rx):
        def reverse_paths(paths):
            return [path[::-1] for path in paths]

        if f"{agent_tx}-{agent_rx}" in self.paths:
            return self.paths[f"{agent_tx}-{agent_rx}"]
        elif f"{agent_rx}-{agent_tx}" in self.paths:
            return reverse_paths(self.paths[f"{agent_rx}-{agent_tx}"])
        else:
            print(f"Path from {agent_tx} to {agent_rx} not found")
            return []
