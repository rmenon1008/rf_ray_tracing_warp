import time
import math

import warp as wp
import numpy as np
import trimesh as tm
from trimesh import viewer

import kernel

class Tracer:
    def __init__(self, environment_trimesh, light_speed_mps, sample_rate_hz, sample_window_s, max_bounces, tx_num_rays):
        wp.init()
        wp.build.clear_kernel_cache()

        self.light_speed_mps = light_speed_mps
        self.sample_rate_hz = sample_rate_hz
        self.sample_window_s = sample_window_s
        self.max_bounces = max_bounces
        self.tx_num_rays = tx_num_rays

        env_vertices = wp.array(environment_trimesh.vertices, dtype=wp.vec3)
        env_faces = wp.array(environment_trimesh.faces.flatten(), dtype=wp.int32)
        self.wp_env = wp.Mesh(points=env_vertices, velocities=None, indices=env_faces)

    def _generate_rx_mesh(self, rx_pos, rx_radius):
        rx_trimesh = tm.primitives.Sphere(center=rx_pos, radius=rx_radius, subdivisions=1)
        rx_vertices = wp.array(rx_trimesh.vertices, dtype=wp.vec3)
        rx_faces = wp.array(rx_trimesh.faces.flatten(), dtype=wp.int32)
        return wp.Mesh(points=rx_vertices, velocities=None, indices=rx_faces)

    # Adapted from here:
    # https://en.wikipedia.org/wiki/Fresnel_equations#Power_(intensity)_reflection_and_transmission_coefficients
    def _bounce_amplitude(self, angle_between):
        if math.isnan(angle_between):
            print("Encountered NaN in _bounce_amplitude")
            return 0

        theta = (math.pi / 2) - (angle_between / 2)

        print(theta)

        n_1 = 5.0
        n_2 = 1.0
        theta_i = math.asin((n_2 * math.sin(theta)) / n_1)
        print(theta_i)

        num = n_2 * math.cos(theta_i) - n_1 * math.cos(theta)
        denom = n_2 * math.cos(theta_i) + n_1 * math.cos(theta)

        amp = -(num / denom)**2
        if amp < -1:
            amp = -1

        if math.isnan(amp):
            print("Encountered NaN in _bounce_amplitude")
            return 0
        
        print(amp)

        return -amp
    
    def compute_cir(self, tx_pos, tx_power, rx_pos, rx_radius):
        start_time = time.perf_counter()
        wp_rx_mesh = self._generate_rx_mesh(rx_pos, rx_radius)

        nan_array = np.empty((self.tx_num_rays, self.max_bounces + 1, 3))
        nan_array[:] = np.nan

        recieved_paths = wp.array(nan_array, dtype=wp.vec3)
        traced_paths = wp.array(nan_array, dtype=wp.vec3)
        row_mask = wp.array(np.zeros(self.tx_num_rays), dtype=wp.uint32)

        print("Launching kernel...")
        wp.launch(
            kernel.trace_paths_kernel,
            dim=(self.tx_num_rays, 1, 1),
            inputs=[self.wp_env.id, tx_pos, wp_rx_mesh.id, self.max_bounces, traced_paths, recieved_paths, row_mask],
        )
        wp.synchronize_device()
        print("Done")


        paths = recieved_paths.numpy()
        print("Converted to numpy")
        # paths = [path for path in paths if not np.isnan(path).all()]
        paths = paths[row_mask.numpy()!=0, :, :]

        print("Removed empty paths")
        cleaned_paths = []
        for path in paths:
            new_path = []
            for i in range(path.shape[0]):
                if np.isnan(path[i]).any():
                    break
                new_path.append(path[i])
            cleaned_paths.append(np.array(new_path))

        print("Cleaned NaNs")

        impulse_response = np.zeros(int(self.sample_window_s * self.sample_rate_hz))
        for path in cleaned_paths:
            amplitude = tx_power / self.tx_num_rays
            distance = 0.0

            for p1, p2, p3 in zip(path[:-2], path[1:-1], path[2:]):
                seg1 = p2 - p1
                seg2 = p3 - p2
                seg1_len = np.linalg.norm(seg1)
                angle_between = np.arccos(np.dot(seg1, seg2) / (seg1_len * np.linalg.norm(seg2)))
                amplitude *= self._bounce_amplitude(angle_between)
                distance += seg1_len
            distance += np.linalg.norm(path[-2] - path[-1])

            delay_samples = int((distance / self.light_speed_mps) * self.sample_rate_hz)
            if delay_samples < impulse_response.shape[0]:
                impulse_response[delay_samples] += amplitude

        print(f"Traced {len(cleaned_paths)} paths in {time.perf_counter() - start_time} seconds")

        return cleaned_paths, impulse_response

        



