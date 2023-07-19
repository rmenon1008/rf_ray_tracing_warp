import time

import warp as wp
import numpy as np
import trimesh as tm
from trimesh import viewer

import kernel

class Tracer:
    LIGHT_SPEED_MPS = 2.998e8   # 299,800,000 m/s
    SAMPLE_RATE_HZ = 50.0e9    # 50 GHz
    SAMPLE_WINDOW_S = 500.0e-9  # 500 ns
    MAX_BOUNCES = 3
    TX_NUM_RAYS = 1_000_000

    def __init__(self, environment_trimesh):
        wp.init()
        wp.build.clear_kernel_cache()

        env_vertices = wp.array(environment_trimesh.vertices, dtype=wp.vec3)
        env_faces = wp.array(environment_trimesh.faces.flatten(), dtype=wp.int32)
        self.wp_env = wp.Mesh(points=env_vertices, velocities=None, indices=env_faces)

    def _generate_rx_mesh(self, rx_pos, rx_radius):
        rx_trimesh = tm.primitives.Sphere(center=rx_pos, radius=rx_radius, subdivisions=1)
        rx_vertices = wp.array(rx_trimesh.vertices, dtype=wp.vec3)
        rx_faces = wp.array(rx_trimesh.faces.flatten(), dtype=wp.int32)
        return wp.Mesh(points=rx_vertices, velocities=None, indices=rx_faces)
    
    def compute_cir(self, tx_pos, tx_power, rx_pos, rx_radius):
        start_time = time.perf_counter()
        wp_rx_mesh = self._generate_rx_mesh(rx_pos, rx_radius)

        recieved_paths = wp.zeros((self.TX_NUM_RAYS, self.MAX_BOUNCES + 1), dtype=wp.vec3)
        traced_paths = wp.zeros((self.TX_NUM_RAYS, self.MAX_BOUNCES + 1), dtype=wp.vec3)
        impulse_response = wp.zeros((int(self.SAMPLE_RATE_HZ * self.SAMPLE_WINDOW_S)), dtype=wp.float32)

        wp.launch(
            kernel.trace_cir_kernel,
            dim=(self.TX_NUM_RAYS, 1, 1),
            inputs=[self.wp_env.id, tx_pos, tx_power, wp_rx_mesh.id, self.MAX_BOUNCES, traced_paths, recieved_paths, impulse_response],
        )
        wp.synchronize_device()

        paths = recieved_paths.numpy()
        empty_path = np.zeros((self.MAX_BOUNCES + 1, 3))
        paths = paths[np.any(paths != empty_path, axis=(1, 2))]

        impulse_response = impulse_response.numpy()

        end_time = time.perf_counter()
        print(f"Tracing took {end_time - start_time} seconds")

        return paths, impulse_response

        



