import numpy as np
import trimesh as tm

def load_mesh(path, scale=1):
    mesh = tm.load_mesh(path)
    tm.repair.fix_inversion(mesh)
    tm.repair.fix_winding(mesh)
    tm.repair.fix_normals(mesh)
    if scale != 1:
        mesh.apply_scale(scale)
    return mesh

def place_on_ground(mesh, pos, offset=0):
    # Create a ray from the 2D position, pointing straight down
    ray_origin = np.array([pos[0], pos[1], 1000])
    ray_direction = np.array([0, 0, -1])

    # Find the intersection of the ray with the mesh
    ray = tm.ray.ray_triangle.RayMeshIntersector(mesh)
    intersections = ray.intersects_location(ray_origins=[ray_origin], ray_directions=[ray_direction])
    try:
        # Find the highest intersection point
        pos = intersections[0][intersections[0][:, 2].argmax()]
        pos[2] += offset
    except:
        pos.append(0)

    return pos

def random_pos_on_ground(mesh, offset=0):
    bounds = mesh.bounding_box.bounds

    found_pos = False
    while not found_pos:
        x = np.random.uniform(bounds[0][0], bounds[1][0])
        y = np.random.uniform(bounds[0][1], bounds[1][1])
        pos = place_on_ground(mesh, [x, y], offset=offset)
        if pos[2] != 0:
            found_pos = True
        
    return pos

class LocationGrid:
    def __init__(self, bounds, divison_size):
        self.divison_size = divison_size
        self.bounds = bounds
        self.grid_shape = (int((self.bounds[1][0] - self.bounds[0][0]) / self.divison_size), int((self.bounds[1][1] - self.bounds[0][1]) / self.divison_size))

    def nearest_grid_point(self, pos):
        x = int((pos[0] - self.bounds[0][0]) / self.divison_size)
        y = int((pos[1] - self.bounds[0][1]) / self.divison_size)
        return x, y
    
    def random_grid_point(self):
        x = np.random.randint(0, (self.bounds[1][0] - self.bounds[0][0]) / self.divison_size)
        y = np.random.randint(0, (self.bounds[1][1] - self.bounds[0][1]) / self.divison_size)
        return x, y
    
    def pos_from_grid_point(self, x, y):
        return [self.bounds[0][0] + x * self.divison_size, self.bounds[0][1] + y * self.divison_size, 0]
    
    def random_pos(self):
        x, y = self.random_grid_point()
        return self.pos_from_grid_point(x, y)
    
    def all_points(self):
        positions = []
        for x in range(int((self.bounds[1][0] - self.bounds[0][0]) / self.divison_size)):
            for y in range(int((self.bounds[1][1] - self.bounds[0][1]) / self.divison_size)):
                positions.append(self.pos_from_grid_point(x, y))
        return positions
    
    def local_region_points(self, pos, radius=1):
        positions = []
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                positions.append(self.pos_from_grid_point(*self.nearest_grid_point(pos) + np.array([x, y])))
        return positions