import warp as wp

# LIGHT_SPEED_MPS = 2.998e8   # 299,800,000 m/s
# SAMPLE_RATE_HZ = 50.0e9    # 50 GHz

POINT_TYPE_TX = 1
POINT_TYPE_NO_HIT = 2
POINT_TYPE_REFLECTION = 3
# POINT_TYPE_DIFFRACTION = 4  # Not supported yet
# POINT_TYPE_TRANSMISSION = 5  # Not supported yet
POINT_TYPE_RX = 6

@wp.struct
class PathPoint:
    pos: wp.vec3
    p_type: wp.int32
    face_id: wp.int32


@wp.func
def get_point_type(dir: wp.vec3, normal: wp.vec3) -> wp.int32:
    # if wp.dot(dir, normal) > 0:
    #     return POINT_TYPE_REFLECTION
    # else:
    #     return POINT_TYPE_DIFFRACTION
    return POINT_TYPE_REFLECTION

@wp.func
def reflect(v: wp.vec3, n: wp.vec3) -> wp.vec3:
    return v - 2.0 * wp.dot(v, n) * n

@wp.kernel
def do_bounce(
    env_mesh: wp.uint64,
    tx_pos: wp.vec3,
    rx_pos: wp.vec3,
    bounce_num: wp.int32,
    paths: wp.array2d(dtype=PathPoint),
):
    tid = wp.tid()

    if bounce_num == 0:
        paths[tid, 0].pos = tx_pos
        paths[tid, 0].p_type = POINT_TYPE_TX
        paths[tid, 0].face_id = 0

        print(paths[tid, 0].pos)

        bounce_num = 1
        rand_state = wp.rand_init(tid)

        bounce_type = POINT_TYPE_TX
        pos = tx_pos
        dir = wp.sample_unit_sphere_surface(rand_state)
    # else:
    #     bounce_type = paths[tid, bounce_num - 1].p_type
    #     pos = paths[tid, bounce_num - 1].pos
    #     dir = wp.normalize(paths[tid, bounce_num - 1].pos - paths[tid, bounce_num - 2].pos)
    
    # if bounce_type == POINT_TYPE_RX:
    #     print("Already hit the receiver, should not be here")
    #     return
    # elif bounce_type == POINT_TYPE_NO_HIT:
    #     print("Already missed env, should not be here")
    #     return
    # elif bounce_type == POINT_TYPE_TX:
    #     # print("Initial point should not be here")
    #     new_dir = wp.normalize(rx_pos - pos)
    #     # return
    # elif bounce_type == POINT_TYPE_REFLECTION:
    #     normal = wp.mesh_eval_face_normal(env_mesh, paths[tid, bounce_num - 1].face_id)
    #     new_dir = reflect(dir, normal)
    # else:
    #     print("Not supported yet")
    #     return

    # # Check if the ray hit the environment
    # maybe_hit_env = False
    # t_env = float(0.0)
    # u_env = float(0.0)
    # v_env = float(0.0)
    # sign_env = float(0.0)
    # n_env = wp.vec3()
    # f_env = int(0)

    # if wp.mesh_query_ray(env_mesh, pos, new_dir, 1.0e6, t_env, u_env, v_env, sign_env, n_env, f_env):
    #     maybe_hit_env = True

    # # Check if the ray hit the receiver
    # t_rx = wp.dot(rx_pos - pos, new_dir)
    # closest_point = pos + new_dir * t_rx
    # maybe_hit_rx = wp.length(closest_point - rx_pos) <= 0.2

    # if maybe_hit_rx and (not maybe_hit_env or (maybe_hit_env and t_env > t_rx)):
    #     # Hit the receiver
    #     paths[tid, bounce_num].pos = pos + new_dir * t_rx
    #     paths[tid, bounce_num].face_id = 0
    #     paths[tid, bounce_num].p_type = POINT_TYPE_RX

    # elif maybe_hit_env:
    #     # Hit the environment
    #     paths[tid, bounce_num].pos = pos + new_dir * t_env
    #     paths[tid, bounce_num].face_id = f_env
    #     paths[tid, bounce_num].p_type = get_point_type(new_dir, n_env)

    # else:
    #     # No hit
    #     paths[tid, bounce_num].pos = wp.vec3()
    #     paths[tid, bounce_num].face_id = 0
    #     paths[tid, bounce_num].p_type = POINT_TYPE_NO_HIT
