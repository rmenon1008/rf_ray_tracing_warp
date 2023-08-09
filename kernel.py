import warp as wp

# LIGHT_SPEED_MPS = 2.998e8   # 299,800,000 m/s
# SAMPLE_RATE_HZ = 50.0e9    # 50 GHz

@wp.func
def reflect(v: wp.vec3, n: wp.vec3) -> wp.vec3:
    dot_vn = wp.dot(v, n)
    if dot_vn < 0:
        return v - 2.0 * dot_vn * n
    else:
        return v


@wp.func
def ray_hit_rx(pos: wp.vec3, dir: wp.vec3, rx_pos: wp.vec3, rx_radius: wp.float32, t_rx: wp.float32) -> bool:
    # Find the closest point on the ray to the receiver
    t_rx = wp.dot(rx_pos - pos, dir)
    closest_point = pos + dir * t_rx

    # Check if the closest point is within the receiver radius
    return wp.length(closest_point - rx_pos) <= rx_radius


@wp.kernel
def trace_paths_kernel(
    env_mesh: wp.uint64,
    tx_pos: wp.vec3,
    rx_pos: wp.vec3,
    max_bounces: wp.int32,
    traced_paths: wp.array2d(dtype=wp.vec3),
    row_mask: wp.array(dtype=wp.uint8),
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

            t_rx = wp.dot(rx_pos - pos, dir)
            closest_point = pos + dir * t_rx
            maybe_hit_rx = wp.length(closest_point - rx_pos) <= 0.1

            t_env = float(0.0)
            u_env = float(0.0)
            v_env = float(0.0)
            sign_env = float(0.0)
            n_env = wp.vec3()
            f_env = int(0)

            # Check if the ray hit the environment
            if wp.mesh_query_ray(env_mesh, pos, dir, 1.0e6, t_env, u_env, v_env, sign_env, n_env, f_env):
                maybe_hit_env = True

            hit_recv = maybe_hit_rx and (not maybe_hit_env or (maybe_hit_env and t_env > t_rx))
            if hit_recv:
                # The ray hit the receiver
                pos = pos + dir * t_rx
                traced_paths[tid, bounce + 1] = pos
                row_mask[tid] = wp.uint8(1)
                ray_finished = True
            elif maybe_hit_env:
                # The ray hit the environment
                pos = pos + dir * t_env
                traced_paths[tid, bounce + 1] = pos
                dir = reflect(dir, n_env)
                # print(wp.length(dir))
            else:
                # The ray did not hit anything
                ray_finished = True