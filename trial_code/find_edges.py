import trimesh
import numpy as np

# https://github.com/mikedh/trimesh/issues/113


##### or, if you want to find a wireframe consisting of edges along sharp angles
def find_edges(mesh, angle):
    sharp = mesh.face_adjacency_angles > np.radians(angles)
    edges = mesh.face_adjacency_edges[sharp]
    lines = mesh.vertices[edges]

    return lines