from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class MotionSpec:
    mode: str                 # sit_stand | slow_walk | walk | trot_pace
    gait: str
    target_speed_mps: float
    base_height_m: float
    notes: str = ""


def parse_speed_mps(text: str) -> Optional[float]:
    t = text.lower()

    # 1) "0.3 - 0.6 m/s"
    m = re.search(r'(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*m/s', t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (a + b) * 0.5

    # 2) "4 - 7 km/h"
    m = re.search(r'(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*km/h', t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return ((a + b) * 0.5) / 3.6

    # 3) "1.1 m/s"
    m = re.search(r'(\d+(?:\.\d+)?)\s*m/s', t)
    if m:
        return float(m.group(1))

    # 4) "7 km/h"
    m = re.search(r'(\d+(?:\.\d+)?)\s*km/h', t)
    if m:
        return float(m.group(1)) / 3.6

    return None


def heuristic_spec(report: str) -> MotionSpec:
    t = report.lower()
    speed = parse_speed_mps(t)
    if speed is None:
        speed = 0.4

    gait = "walk"
    if "pace" in t or "pacing" in t:
        gait = "pace"
    elif "trot" in t:
        gait = "trot"
    elif "sit" in t or "sitting" in t or "stand still" in t:
        gait = "sit"

    if gait == "sit":
        mode = "sit_stand"
        target_speed = 0.0
    elif gait in ("pace", "trot") or speed >= 1.2:
        mode = "trot_pace"
        target_speed = max(speed, 1.0)
    elif speed <= 0.7:
        mode = "slow_walk"
        target_speed = max(speed, 0.25)
    else:
        mode = "walk"
        target_speed = max(speed, 0.5)

    # SUS에서 흔한 값이 0.30m라 기본 유지
    base_h = 0.30

    return MotionSpec(
        mode=mode,
        gait=gait,
        target_speed_mps=round(target_speed, 3),
        base_height_m=base_h,
        notes="heuristic spec",
    )


def _extract_json_block(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    m = re.search(r'\{.*\}', s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def gemini_spec(report: str, model: str, api_key: str) -> Optional[MotionSpec]:
    try:
        import requests
    except ImportError:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    prompt = f"""
You are extracting motion-control metadata for quadruped reward generation.
Return JSON only.

Allowed mode values: "sit_stand", "slow_walk", "walk", "trot_pace"
JSON schema:
{{
  "mode": str,
  "gait": str,
  "target_speed_mps": number,
  "base_height_m": number,
  "notes": str
}}

Rules:
- If report says sit/stand still, mode=sit_stand and target_speed_mps=0.0
- Slow walk should be around 0.25~0.7 m/s
- base_height_m in range 0.26~0.34 for Go2
- Output ONLY JSON

Report:
{report}
"""

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512},
    }

    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        txt = data["candidates"][0]["content"]["parts"][0]["text"]
        obj = _extract_json_block(txt)
        if not obj:
            return None

        mode = str(obj.get("mode", "slow_walk"))
        if mode not in {"sit_stand", "slow_walk", "walk", "trot_pace"}:
            mode = "slow_walk"

        gait = str(obj.get("gait", "walk"))
        v = float(obj.get("target_speed_mps", 0.4))
        h = float(obj.get("base_height_m", 0.30))
        notes = str(obj.get("notes", "gemini spec"))

        # clamp
        v = max(0.0, min(v, 3.0))
        h = max(0.26, min(h, 0.34))

        if mode == "sit_stand":
            v = 0.0

        return MotionSpec(mode=mode, gait=gait, target_speed_mps=round(v, 3), base_height_m=round(h, 3), notes=notes)
    except Exception:
        return None


def build_reward_code(spec: MotionSpec) -> str:
    if spec.mode == "sit_stand":
        reward_vel_w = 0.6
        standstill_w = 0.0
        gait_w = 0.02
        air_w = 0.0
    elif spec.mode == "slow_walk":
        reward_vel_w = 1.6
        standstill_w = 0.35
        gait_w = 0.10
        air_w = 0.01
    elif spec.mode == "walk":
        reward_vel_w = 2.0
        standstill_w = 0.45
        gait_w = 0.12
        air_w = 0.015
    else:
        reward_vel_w = 2.4
        standstill_w = 0.55
        gait_w = 0.16
        air_w = 0.02

    code = f'''import jax.numpy as jnp
from brax import math
from dial_mpc.utils.function_utils import global_to_body_velocity, get_foot_step


GENERATED_MOTION_SPEC = {{
    "mode": "{spec.mode}",
    "gait": "{spec.gait}",
    "target_speed_mps": {spec.target_speed_mps:.3f},
    "base_height_m": {spec.base_height_m:.3f},
    "notes": {json.dumps(spec.notes, ensure_ascii=False)},
}}


def _get_reward_cfg(env):
    cfg = getattr(env, "_config", None)
    if cfg is None:
        return {{}}
    reward_cfg = getattr(cfg, "reward", None)
    return reward_cfg if isinstance(reward_cfg, dict) else {{}}


def _w(reward_cfg, key: str, default: float) -> float:
    try:
        return float(reward_cfg.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def compute_sds_reward(pipeline_state, state_info, env):
    """Auto-generated SDS reward that preserves the Dial-MPC Go2 scaffold."""
    reward_cfg = _get_reward_cfg(env)

    torso_idx = env._torso_idx - 1
    x = pipeline_state.x
    xd = pipeline_state.xd
    ctrl = pipeline_state.ctrl

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

    vel_tar = state_info["vel_tar"]
    ang_vel_tar = state_info["ang_vel_tar"]
    target_vx_fallback = _w(reward_cfg, "target_vx_fallback", {spec.target_speed_mps:.3f})
    target_vx = jnp.where(jnp.abs(vel_tar[0]) < 1e-6, target_vx_fallback, vel_tar[0])
    target_base_height = _w(reward_cfg, "target_base_height", {spec.base_height_m:.3f})

    pos = x.pos[torso_idx]
    rot = x.rot[torso_idx]
    vb = global_to_body_velocity(xd.vel[torso_idx], rot)
    ab = global_to_body_velocity(xd.ang[torso_idx], rot)

    euler = math.quat_to_euler(rot)
    pitch = euler[1]
    pitch_ref = jnp.array(getattr(env, "_home_pitch", 0.0), dtype=pitch.dtype)
    pitch_dev = jnp.abs(jnp.atan2(jnp.sin(pitch - pitch_ref), jnp.cos(pitch - pitch_ref)))

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
        reward_gaits * _w(reward_cfg, "reward_gaits_w", {gait_w:.3f})
        + reward_air_time * _w(reward_cfg, "reward_air_time_w", {air_w:.3f})
        + reward_pos * _w(reward_cfg, "reward_pos_w", 0.00)
        + reward_upright * _w(reward_cfg, "reward_upright_w", 1.60)
        + reward_yaw * _w(reward_cfg, "reward_yaw_w", 0.60)
        + reward_vel * _w(reward_cfg, "reward_vel_x_w", {reward_vel_w:.3f})
        + reward_ang_vel * _w(reward_cfg, "reward_ang_vel_w", 0.70)
        + reward_height * _w(reward_cfg, "reward_height_w", 1.30)
        - penalty_joint_pose * _w(reward_cfg, "penalty_joint_pose_w", 0.20)
        - penalty_joint_vel * _w(reward_cfg, "penalty_joint_vel_w", 0.0012)
        - penalty_energy * _w(reward_cfg, "penalty_energy_w", 0.0005)
        - penalty_vertical_vel * _w(reward_cfg, "penalty_vertical_vel_w", 0.80)
        - penalty_collapse * _w(reward_cfg, "penalty_collapse_w", 8.00)
        - penalty_rear_up * _w(reward_cfg, "penalty_rear_up_w", 12.00)
        - penalty_standstill * _w(reward_cfg, "penalty_standstill_w", {standstill_w:.3f})
        - penalty_all_feet_air * _w(reward_cfg, "penalty_all_feet_air_w", 2.50)
    )
    return reward
'''
    return code


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/final_sus_report.txt")
    parser.add_argument("--output", default="dial-mpc/dial_mpc/envs/sds_reward_function.py")
    parser.add_argument("--spec-out", default="output/motion_spec.json")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--no-api", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    in_path = (base / args.input).resolve() if not os.path.isabs(args.input) else Path(args.input)
    out_path = (base / args.output).resolve() if not os.path.isabs(args.output) else Path(args.output)
    spec_path = (base / args.spec_out).resolve() if not os.path.isabs(args.spec_out) else Path(args.spec_out)

    if not in_path.exists():
        raise FileNotFoundError(f"SUS report not found: {in_path}")

    report = read_text(in_path)

    spec = heuristic_spec(report)

    api_key = os.getenv("GEMINI_API_KEY", "")
    if (not args.no_api) and api_key:
        s = gemini_spec(report, model=args.model, api_key=api_key)
        if s is not None:
            spec = s

    code = build_reward_code(spec)
    write_text(out_path, code)
    write_text(spec_path, json.dumps(asdict(spec), ensure_ascii=False, indent=2))

    print(f"[OK] SUS input: {in_path}")
    print(f"[OK] Motion spec: {spec_path}")
    print(f"[OK] Reward code: {out_path}")
    print(f"[INFO] mode={spec.mode}, gait={spec.gait}, target_speed={spec.target_speed_mps} m/s")


if __name__ == "__main__":
    main()
