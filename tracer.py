import time
import math

import warp as wp
import numpy as np
import trimesh as tm
from trimesh import viewer

# import kernel_new as kernel
import kernel

# class Tracer:
#     def __init__(self, environment_trimesh, light_speed_mps, sample_rate_hz, sample_window_s, max_bounces, tx_num_rays):
#         wp.init()
#         wp.build.clear_kernel_cache()

#         # self.light_speed_mps = light_speed_mps
#         # self.sample_rate_hz = sample_rate_hz
#         # self.sample_window_s = sample_window_s
#         self.max_bounces = max_bounces
#         self.tx_num_rays = tx_num_rays
#         env_vertices = wp.array(environment_trimesh.vertices, dtype=wp.vec3)
#         env_faces = wp.array(environment_trimesh.faces.flatten(), dtype=wp.int32)
#         self.wp_env = wp.Mesh(points=env_vertices, velocities=None, indices=env_faces)

#     # Adapted from here:
#     # https://en.wikipedia.org/wiki/Fresnel_equations#Power_(intensity)_reflection_and_transmission_coefficients
#     def _bounce_amplitude(self, angle_between):
#         if math.isnan(angle_between):
#             print("Encountered NaN in _bounce_amplitude")
#             return 0

#         theta = (math.pi / 2) - (angle_between / 2)

#         print(theta)

#         n_1 = 5.0
#         n_2 = 1.0
#         theta_i = math.asin((n_2 * math.sin(theta)) / n_1)
#         print(theta_i)

#         num = n_2 * math.cos(theta_i) - n_1 * math.cos(theta)
#         denom = n_2 * math.cos(theta_i) + n_1 * math.cos(theta)

#         amp = -(num / denom)**2
#         if amp < -1:
#             amp = -1

#         if math.isnan(amp):
#             print("Encountered NaN in _bounce_amplitude")
#             return 0

#         return -amp
    
#     def trace_paths(self, tx_pos, tx_power, rx_pos, rx_radius):
#         tx_pos = wp.vec3(tx_pos)
#         rx_pos = wp.vec3(rx_pos)

#         paths = wp.zeros(shape=(self.tx_num_rays, self.max_bounces + 1), dtype=kernel.PathPoint)
#         bounce_num = 0

#         wp.launch(
#             kernel.do_bounce,
#             dim=self.tx_num_rays,
#             inputs=[
#                 self.wp_env.id,
#                 tx_pos,
#                 rx_pos,
#                 bounce_num,
#                 paths,
#             ],
#         )
#         wp.synchronize_device()

#         paths = paths.numpy()

#         print(paths)
        
#         # Mask out rows that didn't hit the receiver
#         # cleaned_paths = []
#         # for path in paths:
#         #     cleaned_path = []
#         #     for point in path:
#         #         if not point[0] == 0:
#         #             cleaned_path.append(point)
#         #     if len(cleaned_path) > 2:
#         #         cleaned_paths.append(cleaned_path)

#         # return cleaned_paths



class Tracer:
    def __init__(self, environment_trimesh, light_speed_mps, sample_rate_hz, sample_window_s, max_bounces, tx_num_rays):
        wp.init()
        wp.build.clear_kernel_cache()

        # self.light_speed_mps = light_speed_mps
        # self.sample_rate_hz = sample_rate_hz
        # self.sample_window_s = sample_window_s
        self.max_bounces = max_bounces
        self.tx_num_rays = tx_num_rays
        env_vertices = wp.array(environment_trimesh.vertices, dtype=wp.vec3)
        env_faces = wp.array(environment_trimesh.faces.flatten(), dtype=wp.int32)
        self.wp_env = wp.Mesh(points=env_vertices, velocities=None, indices=env_faces)
    
    def trace_paths(self, tx_pos, tx_power, rx_pos, rx_radius):
        tx_pos = wp.vec3(tx_pos)
        rx_pos = wp.vec3(rx_pos)

        paths = wp.zeros(shape=(self.tx_num_rays, self.max_bounces + 1), dtype=wp.vec3)
        row_mask = wp.zeros(shape=self.tx_num_rays, dtype=wp.uint8)

        wp.launch(
            kernel.trace_paths_kernel,
            dim=self.tx_num_rays,
            inputs=[
                self.wp_env.id,
                tx_pos,
                rx_pos,
                self.max_bounces,
                paths,
                row_mask,
            ],
        )
        wp.synchronize_device()

        paths = paths.numpy()
        row_mask = row_mask.numpy()

        paths = paths[row_mask == 1]

        print(paths.shape)
        
        # Mask out rows that didn't hit the receiver
        cleaned_paths = []
        for path in paths:
            cleaned_path = []
            for point in path:
                if not point[0] == 0:
                    cleaned_path.append(point)
            if len(cleaned_path) > 2:
                cleaned_paths.append(cleaned_path)

        return cleaned_paths
