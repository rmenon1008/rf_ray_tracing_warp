import warp as wp

@wp.func
def reflect(v: wp.vec3, n: wp.vec3) -> wp.vec3:
    dot_vn = wp.dot(v, n)
    return v - 2.0 * dot_vn * n


@wp.kernel
def trace_paths_kernel(
    env_mesh: wp.uint64,
    tx_pos: wp.vec3,
    rx_pos: wp.vec3,
    rx_radius: wp.float32,
    max_bounces: wp.int32,
    traced_paths: wp.array2d(dtype=wp.vec3),
    row_mask: wp.array1d(dtype=wp.uint8),
    # points: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    # Choose a random direction for the ray
    state = wp.rand_init(tid)
    dir = wp.sample_unit_sphere_surface(state)
    pos = tx_pos

    traced_paths[tid, 0] = pos
    ray_finished = wp.uint8(0)

    for bounce in range(max_bounces):
        if ray_finished == 0:
            maybe_hit_rx = False
            maybe_hit_env = False
            
            t_rx = wp.dot(rx_pos - pos, dir)
            overlap_amt = 0.0
            if t_rx > 0:
                closest_point = pos + dir * t_rx                
                maybe_hit_rx = wp.length(closest_point - rx_pos) <= rx_radius

            t_env = float(0.0)
            u_env = float(0.0)
            v_env = float(0.0)
            sign_env = float(0.0)
            n_env = wp.vec3()
            f_env = int(0)

            # Check if the ray hit the environment
            if wp.mesh_query_ray(env_mesh, pos, dir, 1.0e6, t_env, u_env, v_env, sign_env, n_env, f_env):
                maybe_hit_env = True

            hit_recv = maybe_hit_rx and ((not maybe_hit_env) or (maybe_hit_env and t_env > t_rx))
            if hit_recv:
                # The ray hit the receiver
                traced_paths[tid, bounce + 1] = closest_point
                row_mask[tid] = wp.uint8(1)
                ray_finished = wp.uint8(1)
            elif maybe_hit_env:
                # The ray hit the environment
                pos = pos + dir * t_env
                traced_paths[tid, bounce + 1] = pos
                dir = reflect(dir, n_env)
                dir = wp.normalize(dir)
            else:
                # The ray did not hit anything
                ray_finished = wp.uint8(1)

class Tracer:
    def __init__(self, environment_trimesh, max_bounces, tx_num_rays, rx_radius):
        wp.init()
        wp.build.clear_kernel_cache()

        self.max_bounces = max_bounces
        self.tx_num_rays = tx_num_rays
        self.rx_radius = rx_radius
        env_vertices = wp.array(environment_trimesh.vertices, dtype=wp.vec3)
        env_faces = wp.array(environment_trimesh.faces.flatten(), dtype=wp.int32)
        self.wp_env = wp.Mesh(points=env_vertices, velocities=None, indices=env_faces)
    
    def trace_paths(self, tx_pos, rx_pos):
        tx_pos = wp.vec3(tx_pos)
        rx_pos = wp.vec3(rx_pos)

        paths = wp.zeros(shape=(self.tx_num_rays, self.max_bounces + 1), dtype=wp.vec3)
        row_mask = wp.zeros(shape=self.tx_num_rays, dtype=wp.uint8)

        wp.launch(
            trace_paths_kernel,
            dim=self.tx_num_rays,
            inputs=[
                self.wp_env.id,
                tx_pos,
                rx_pos,
                self.rx_radius,
                self.max_bounces,
                paths,
                row_mask,
            ],
        )
        wp.synchronize_device()

        paths = paths.numpy()
        row_mask = row_mask.numpy()
        paths = paths[row_mask == 1]

        # Mask out rows that didn't hit the receiver
        cleaned_paths = []
        for path in paths:
            cleaned_path = []
            for point in path:
                if not point[0] == 0:
                    cleaned_path.append(point)
            if len(cleaned_path) > 0:
                cleaned_paths.append(cleaned_path)

        return cleaned_paths
