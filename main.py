print("Importing...")

import warp as wp
import numpy as np
import trimesh as tm
import json
import http.server
import matplotlib.pyplot as plt

print("Done!", end="\n\n")

# SAMPLE_RATE = 1.0e10
# LIGHT_SPEED = 2.998e8

SAMPLE_RATE = 1.0e4
LIGHT_SPEED = 2.998e3

print("Initializing Warp...")
wp.init()
print("Done!", end="\n\n")

# Import the 3D model
print("Loading mesh...")
# mesh = tm.load_mesh("models/room.stl")
# mesh = tm.load_mesh("models/apollo_17_landing_site.stl") 
mesh = tm.load_mesh("models/almost_empty.stl")
print("Done!", end="\n\n")
# Create a ray emitter
tx_pos = np.array([20, 0, 4.5])
tx_power = 1
tx_num_rays = 80_000_000
max_bounces = 3

# Create a ray receiver
rx_pos = np.array([-20, 0, 4.8])
rx_radius = 0.1

# Check if max_bounces changed from the last run
try:
    with open("cached_settings.json", "r") as f:
        cached_settings = json.load(f)
        if cached_settings["max_bounces"] != max_bounces:
            print("Clearing kernel cache")
            wp.build.clear_kernel_cache()
except:
    print("Clearing kernel cache")
    wp.build.clear_kernel_cache()

with open("cached_settings.json", "w") as f:
    json.dump({"max_bounces": max_bounces}, f)

@wp.func
def reflect(v: wp.vec3, n: wp.vec3) -> wp.vec3:
    return v - 2.0 * wp.dot(v, n) * n

@wp.func
def amp_air_loss(distance: wp.float32) -> wp.float32:
    return 1.0
    # return 1.0 / (distance * distance)

@wp.func
def amp_bounce_loss(angle_between: wp.float32) -> wp.float32:
    return wp.abs((wp.sin(angle_between + 1.57) + 1.0) / 2.0)

@wp.func
def delay(distance: wp.float32) -> wp.float32:
    return (distance / LIGHT_SPEED) * SAMPLE_RATE

@wp.kernel
def ray_tracing_kernel(
    env_mesh: wp.uint64,
    tx_pos: wp.vec3,
    rx_mesh: wp.uint64,
    max_bounces: wp.uint32,
    hits: wp.array(dtype=wp.vec3),
    traced_paths: wp.array2d(dtype=wp.vec3),
    received_paths: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    # Choose a random direction for the ray
    state = wp.rand_init(tid)
    dir = wp.sample_unit_sphere_surface(state)
    pos = tx_pos

    traced_paths[tid, 0] = pos

    for bounce in range(max_bounces):
        ray_finished = False
        if not ray_finished:
            maybe_hit_rx = False
            maybe_hit_env = False

            t_rx = float(0.0)
            u_rx = float(0.0)
            v_rx = float(0.0)
            sign_rx = float(0.0)
            n_rx = wp.vec3()
            f_rx = int(0)

            # Check if the ray hit the reciever
            if wp.mesh_query_ray(rx_mesh, pos, dir, 1.0e6, t_rx, u_rx, v_rx, sign_rx, n_rx, f_rx):
                maybe_hit_rx = True

            t_env = float(0.0)
            u_env = float(0.0)
            v_env = float(0.0)
            sign_env = float(0.0)
            n_env = wp.vec3()
            f_env = int(0)

            # Check if the ray hit the environment
            if wp.mesh_query_ray(env_mesh, pos, dir, 1.0e6, t_env, u_env, v_env, sign_env, n_env, f_env):
                maybe_hit_env = True

            hit_recv = maybe_hit_rx and (not maybe_hit_env or t_env > t_rx)
            if hit_recv:
                pos = pos + dir * t_rx
                hits[tid] = pos
                traced_paths[tid, bounce + 1] = pos
                for i in range(wp.int32(max_bounces) + 1):
                    received_paths[tid, i] = traced_paths[tid, i]
                ray_finished = True
            elif maybe_hit_env:
                pos = pos + dir * t_env
                hits[tid] = pos
                traced_paths[tid, bounce + 1] = pos
                dir = reflect(dir, n_env)
            else:
                ray_finished = True

@wp.kernel
def impulse_calculation_kernel(
    received_paths: wp.array2d(dtype=wp.vec3),
    impulse_response: wp.array1d(dtype=wp.float32),
    tx_power: wp.float32
):
    tid = wp.tid()

    amplitude = tx_power / wp.float32(received_paths.shape[0])
    delay = wp.float32(0.0)

    for i in range(2, received_paths.shape[1]):
        seg1 = received_paths[tid, i] - received_paths[tid, i - 1]
        seg2 = received_paths[tid, i - 1] - received_paths[tid, i - 2]
        seg1_len = wp.length(seg1)
        angle_between = wp.acos(wp.dot(seg1, seg2) / (seg1_len * wp.length(seg2)))
        amplitude *= amp_air_loss(seg1_len) * amp_bounce_loss(angle_between)
        delay += delay(seg1_len)

    impulse_response[wp.int32(delay)] = impulse_response[wp.int32(delay)] + amplitude


env_v = wp.array(mesh.vertices, dtype=wp.vec3)
env_f = wp.array(mesh.faces.flatten(), dtype=wp.int32)
wp_env_mesh = wp.Mesh(points=env_v, velocities=None, indices=env_f)

rx_mesh = tm.primitives.Sphere(radius=rx_radius, center=rx_pos)
rx_v = wp.array(rx_mesh.vertices, dtype=wp.vec3)
rx_f = wp.array(rx_mesh.faces.flatten(), dtype=wp.int32)
wp_rx_mesh = wp.Mesh(points=rx_v, velocities=None, indices=rx_f)

hits = wp.zeros(tx_num_rays, dtype=wp.vec3)
recieved_paths = wp.zeros((tx_num_rays, max_bounces + 1), dtype=wp.vec3)
traced_paths = wp.zeros((tx_num_rays, max_bounces + 1), dtype=wp.vec3)

print("Tracing rays...")
wp.launch(
    ray_tracing_kernel,
    dim = (tx_num_rays, 1, 1),
    inputs = [wp_env_mesh.id, tx_pos, wp_rx_mesh.id, max_bounces, hits, traced_paths, recieved_paths],
)
wp.synchronize_device()
print("Done!", end="\n\n")

hits = hits.numpy()
paths = recieved_paths.numpy()

print("Calculating impulse response...")
impulse_response = wp.zeros((int(SAMPLE_RATE * 0.01)), dtype=wp.float32)
wp.launch(
    impulse_calculation_kernel,
    dim = (tx_num_rays, 1, 1),
    inputs = [recieved_paths, impulse_response, tx_power],
)
wp.synchronize_device()
print("Done!", end="\n\n")

impulse_response = impulse_response.numpy()
max_response = np.max(impulse_response)

# Find the power in dBm
power = 10 * np.log10(max_response / 1e-3)
print("Power: {:.2f} dBm".format(power))

# Plot the impulse response
plt.plot(impulse_response)
plt.savefig("impulse_response.png")

# Visualize the result
scene = tm.Scene()
mesh.visual.face_colors = [100, 100, 100, 255]
scene.add_geometry(mesh)

tx_vis = tm.primitives.Sphere(radius=0.5, center=tx_pos)
tx_vis.visual.face_colors = [255, 0, 0, 255]
scene.add_geometry(tx_vis)

rx_vis = tm.primitives.Sphere(radius=0.5, center=rx_pos)
rx_vis.visual.face_colors = [0, 255, 0, 255]
scene.add_geometry(rx_vis)

# scene.add_geometry(tm.points.PointCloud(hits, colors=np.tile([255, 255, 255], (hits.shape[0], 1))))

# Remove paths where all the points are at the origin
empty_path = np.zeros((max_bounces+1, 3))
paths = paths[np.any(paths != empty_path, axis=(1, 2))]

# Add the paths to the scene
print(f"Adding {paths.shape[0]} paths to the scene...")
for path in paths:
    path = path[~np.all(path == 0, axis=1)]

    if path.shape[0] < 2:
        continue
    path = tm.load_path(path)
    path.colors = np.tile([200, 200, 200], (path.entities.shape[0], 1))
    scene.add_geometry(path)
print("Done!", end="\n\n")

with open("web/scene.html", "w") as f:
    f.write(tm.viewer.scene_to_html(scene))

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/index.html' or self.path == '/':
            self.path = '/web/scene.html'
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
httpd = http.server.HTTPServer(('', 8000), Handler)
print("Serving visualization at localhost:8000")
httpd.serve_forever()
