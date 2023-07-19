import warp as wp

LIGHT_SPEED_MPS = 2.998e8   # 299,800,000 m/s
SAMPLE_RATE_HZ = 50.0e9    # 50 GHz

# @wp.func
# def sample_close_to_target(target: wp.vec3, phi: wp.float64, state: wp.state) -> wp.vec3:
#     while True:
#         sample = wp.sample_unit_sphere_surface(state)
#         angle_between = wp.acos(wp.dot(target, sample))
#         if angle_between < phi:
#             return sample

@wp.func
def reflect(v: wp.vec3, n: wp.vec3) -> wp.vec3:
    return v - 2.0 * wp.dot(v, n) * n

@wp.func
def amp_bounce_loss(angle_between: wp.float32) -> wp.float32:
    return wp.abs((wp.sin(angle_between + 1.57) + 1.0) / 2.0)

@wp.func
def delay(distance: wp.float32) -> wp.float32:
    return (distance / LIGHT_SPEED_MPS) * SAMPLE_RATE_HZ

@wp.func
def set_impulse_response(path: wp.array(dtype=wp.vec3), impulse_response: wp.array(dtype=wp.float32), ray_power: wp.float32):
    amplitude = ray_power
    delay = wp.float32(0.0)

    for i in range(2, path.shape[0]):
        seg1 = path[i - 1] - path[i - 2]
        seg2 = path[i] - path[i - 1]
        seg1_len = wp.length(seg1)
        angle_between = wp.acos(wp.dot(seg1, seg2) / (seg1_len * wp.length(seg2)))
        amplitude *= amp_bounce_loss(angle_between)
        delay += delay(seg1_len)
    delay_samples = wp.int32(delay)
    if delay_samples < impulse_response.shape[0]:
        impulse_response[delay_samples] = impulse_response[delay_samples] + amplitude

@wp.kernel
def trace_cir_kernel(
    env_mesh: wp.uint64,
    tx_pos: wp.vec3,
    tx_power: wp.float32,
    rx_mesh: wp.uint64,
    max_bounces: wp.int32,
    traced_paths: wp.array2d(dtype=wp.vec3),
    received_paths: wp.array2d(dtype=wp.vec3),
    impulse_response: wp.array(dtype=wp.float32)
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

            hit_recv = maybe_hit_rx and (not maybe_hit_env or (maybe_hit_env and t_env > t_rx))
            if hit_recv:
                pos = pos + dir * t_rx
                traced_paths[tid, bounce + 1] = pos
                for i in range(wp.int32(max_bounces) + 1):
                    received_paths[tid, i] = traced_paths[tid, i]
                set_impulse_response(received_paths[tid], impulse_response, tx_power / wp.float32(received_paths.shape[0]))
                ray_finished = True
            elif maybe_hit_env:
                pos = pos + dir * t_env
                traced_paths[tid, bounce + 1] = pos
                dir = reflect(dir, n_env)
            else:
                ray_finished = True