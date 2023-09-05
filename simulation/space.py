import math

import trimesh as tm

from tracing.mesh_operations import *

class MeshSpace:
    def __init__(self, mesh):
        self.mesh = mesh
        self.bounds = mesh.bounding_box.bounds

    def place_on_ground(self, xy_pos, offset=0.0):
        return place_on_ground(self.mesh, xy_pos, offset).tolist()
    
    def random_pos_on_ground(self, offset=0.0):
        point = None
        while point is None or not self.pos_valid(point):
            point = random_pos_on_ground(self.mesh, offset)
        return point.tolist()
    
    def pos_valid(self, pos):
        # Check if the point is within the mesh's X and Y bounds
        if pos[0] < self.bounds[0][0] or pos[0] > self.bounds[1][0]:
            return False
        if pos[1] < self.bounds[0][1] or pos[1] > self.bounds[1][1]:
            return False
        
        # Check if the point is inside the mesh
        if self.mesh.contains([pos]):
            return False
        
        return True
    
    def distance_los(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)