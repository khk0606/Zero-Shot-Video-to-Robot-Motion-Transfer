import jax.numpy as jnp
from brax import math
from dial_mpc.utils.function_utils import global_to_body_velocity, get_foot_step


def _get_reward_cfg(env):
    cfg = getattr(env, "_config", None)
    if cfg is None:
        return {}
    reward_cfg = getattr(cfg, "reward", None)
    return reward_cfg if isinstance(reward_cfg, dict) else {}


def _w(reward_cfg, key: str, default: float) -> float:
    try:
        return float(reward_cfg.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def compute_sds_reward(pipeline_state, state_info, env):
    """SDS reward with the original Dial-MPC Go2 scaffold preserved."""
    reward_cfg = _get_reward_cfg(env)

    torso_idx = env._torso_idx - 1
    x = pipeline_state.x
    xd = pipeline_state.xd
    ctrl = pipeline_state.ctrl

    # gait/contact scaffold (same structure as original env reward)
    z_feet = pipeline_state.site_xpos[env._feet_site_id][:, 2]
    duty_ratio, cadence, amplitude = env._gait_params[env._gait]
    phases = env._gait_phase[env._gait]
    z_feet_tar = get_foot_step(
        duty_ratio, cadence, amplitude, phases, state_info["step"] * env.dt
    )
    reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / 0.05) ** 2)

    foot_pos = pipeline_state.site_xpos[env._feet_site_id]
    foot_contact_z = foot_pos[:, 2] - env._foot_radius
    contact = foot_contact_z < 1e-3
    contact_filt_mm = contact | state_info["last_contact"]
    first_contact = (state_info["feet_air_time"] > 0) * contact_filt_mm
    reward_air_time = jnp.sum((state_info["feet_air_time"] - 0.1) * first_contact)

    # targets
    vel_tar = state_info["vel_tar"]
    ang_vel_tar = state_info["ang_vel_tar"]
    target_vx_fallback = _w(reward_cfg, "target_vx_fallback", 0.18)
    target_vx = jnp.where(jnp.abs(vel_tar[0]) < 1e-6, target_vx_fallback, vel_tar[0])
    target_base_height = _w(reward_cfg, "target_base_height", 0.30)

    # kinematics
    pos = x.pos[torso_idx]
    rot = x.rot[torso_idx]
    vb = global_to_body_velocity(xd.vel[torso_idx], rot)
    ab = global_to_body_velocity(xd.ang[torso_idx], rot)

    euler = math.quat_to_euler(rot)
    pitch = euler[1]
    pitch_ref = jnp.array(getattr(env, "_home_pitch", 0.0), dtype=pitch.dtype)
    # Use wrapped angular deviation from the home pose so we don't
    # penalize a model-specific static pitch offset.
    pitch_dev = jnp.abs(jnp.atan2(jnp.sin(pitch - pitch_ref), jnp.cos(pitch - pitch_ref)))

    # keep original reward blocks, adjust weights/targets for slow walk
    pos_tar = state_info["pos_tar"] + state_info["vel_tar"] * env.dt * state_info["step"]
    r_mat = math.quat_to_3x3(rot)
    head_vec = jnp.array([0.285, 0.0, 0.0])
    head_pos = pos + jnp.dot(r_mat, head_vec)
    reward_pos = -jnp.sum((head_pos - pos_tar) ** 2)

    vec_tar = jnp.array([0.0, 0.0, 1.0])
    vec = math.rotate(vec_tar, x.rot[0])
    reward_upright = -jnp.sum(jnp.square(vec - vec_tar))

    yaw_tar = state_info["yaw_tar"] + ang_vel_tar[2] * env.dt * state_info["step"]
    yaw = math.quat_to_euler(rot)[2]
    d_yaw = yaw - yaw_tar
    reward_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))

    reward_vel = -(jnp.square(vb[0] - target_vx) + 0.5 * jnp.square(vb[1]))
    reward_ang_vel = -jnp.square(ab[2] - ang_vel_tar[2])
    reward_height = -jnp.square(pos[2] - target_base_height)

    default_pose = env._default_pose if hasattr(env, "_default_pose") else jnp.zeros_like(pipeline_state.q[7:])
    penalty_joint_pose = jnp.sum(jnp.square(pipeline_state.q[7:] - default_pose))
    penalty_joint_vel = jnp.sum(jnp.square(pipeline_state.qvel[6:]))

    power = jnp.maximum(ctrl * pipeline_state.qvel[6:] / 160.0, 0.0)
    penalty_energy = jnp.sum(power ** 2)

    penalty_vertical_vel = jnp.square(vb[2])
    penalty_collapse = jnp.square(jnp.clip(0.24 - pos[2], 0.0, 1.0))
    penalty_rear_up = jnp.square(jnp.clip(pitch_dev - 0.18, 0.0, 1.0))
    penalty_standstill = jnp.square(jnp.clip(target_vx - vb[0], 0.0, 10.0))
    penalty_all_feet_air = jnp.where(jnp.min(foot_contact_z) > 0.01, 1.0, 0.0)

    reward = (
        reward_gaits * _w(reward_cfg, "reward_gaits_w", 0.10)
        + reward_air_time * _w(reward_cfg, "reward_air_time_w", 0.01)
        + reward_pos * _w(reward_cfg, "reward_pos_w", 0.00)
        + reward_upright * _w(reward_cfg, "reward_upright_w", 1.60)
        + reward_yaw * _w(reward_cfg, "reward_yaw_w", 0.60)
        + reward_vel * _w(reward_cfg, "reward_vel_x_w", 1.60)
        + reward_ang_vel * _w(reward_cfg, "reward_ang_vel_w", 0.70)
        + reward_height * _w(reward_cfg, "reward_height_w", 1.30)
        - penalty_joint_pose * _w(reward_cfg, "penalty_joint_pose_w", 0.20)
        - penalty_joint_vel * _w(reward_cfg, "penalty_joint_vel_w", 0.0012)
        - penalty_energy * _w(reward_cfg, "penalty_energy_w", 0.0005)
        - penalty_vertical_vel * _w(reward_cfg, "penalty_vertical_vel_w", 0.80)
        - penalty_collapse * _w(reward_cfg, "penalty_collapse_w", 8.00)
        - penalty_rear_up * _w(reward_cfg, "penalty_rear_up_w", 12.00)
        - penalty_standstill * _w(reward_cfg, "penalty_standstill_w", 0.35)
        - penalty_all_feet_air * _w(reward_cfg, "penalty_all_feet_air_w", 2.50)
    )
    return reward
