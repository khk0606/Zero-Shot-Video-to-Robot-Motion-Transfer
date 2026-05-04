import json
import os
import time
from dataclasses import dataclass
import importlib
from multiprocessing import shared_memory
from pathlib import Path
import sys

import argparse
import art
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import scienceplots
import yaml

from dial_mpc.config.base_env_config import BaseEnvConfig
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.examples import deploy_examples
from dial_mpc.utils.io_utils import (
    get_example_path,
    get_model_path,
    load_dataclass_from_dict,
)

plt.style.use(["science"])


@dataclass
class DialSimConfig:
    robot_name: str
    scene_name: str
    sim_leg_control: str
    plot: bool
    record: bool
    real_time_factor: float
    sim_dt: float
    sync_mode: bool
    draw_refs: bool = False
    headless: bool = False
    max_time_sec: float = 0.0
    max_steps: int = 0
    stop_on_fall: bool = False
    fall_height_threshold: float = 0.18
    target_base_height: float = 0.30
    metrics_filename: str = "metrics.json"


class DialSim:
    def __init__(
        self,
        sim_config: DialSimConfig,
        env_config: BaseEnvConfig,
        dial_config: DialConfig,
        config_dict: dict | None = None,
    ):
        self.plot = sim_config.plot
        self.record = sim_config.record
        self.data = []
        self.ctrl_dt = env_config.dt
        self.real_time_factor = sim_config.real_time_factor
        self.sim_dt = sim_config.sim_dt
        self.n_acts = dial_config.Hsample + 1
        self.n_frame = int(self.ctrl_dt / self.sim_dt)
        self.t = 0.0
        self.sync_mode = sim_config.sync_mode
        self.leg_control = sim_config.sim_leg_control
        self.draw_refs = bool(sim_config.draw_refs)
        self.headless = bool(sim_config.headless)

        self.max_time_sec = float(sim_config.max_time_sec)
        self.max_steps = int(sim_config.max_steps)
        self.stop_on_fall = bool(sim_config.stop_on_fall)
        self.fall_height_threshold = float(sim_config.fall_height_threshold)
        self.target_base_height = float(sim_config.target_base_height)
        self.metrics_filename = str(sim_config.metrics_filename)

        self.step_count = 0
        self.stop_reason = "running"
        self.sim_exception: str | None = None
        self.config_dict = dict(config_dict or {})
        self.expected_gait = str(self.config_dict.get("gait", "") or "").strip().lower()
        if self.expected_gait == "pacing":
            self.expected_gait = "pace"
        self.foot_radius = 0.0175
        self.foot_contact_eps = 1e-3
        self.foot_site_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        self.foot_geom_names = ("FL", "FR", "RL", "RR")

        self.mj_model = mujoco.MjModel.from_xml_path(
            get_model_path(sim_config.robot_name, sim_config.scene_name).as_posix()
        )
        self.mj_model.opt.timestep = self.sim_dt
        self.mj_data = mujoco.MjData(self.mj_model)

        self.q_history = np.zeros((self.n_acts, self.mj_model.nu))
        self.qref_history = np.zeros((self.n_acts, self.mj_model.nu))
        self.n_plot_joint = 4

        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.Nx = self.mj_model.nq + self.mj_model.nv
        self.Nu = self.mj_model.nu
        self.default_q = self.mj_model.keyframe("home").qpos
        self.default_u = self.mj_model.keyframe("home").ctrl
        self.home_roll = 0.0
        self.home_pitch = 0.0
        if self.default_q.shape[0] >= 7:
            home_quat = np.asarray(self.default_q[3:7], dtype=np.float64).reshape(1, 4)
            hr, hp = self._roll_pitch_from_quat(home_quat)
            self.home_roll = float(hr[0])
            self.home_pitch = float(hp[0])

        # position cmd smoothing (prevents start spike)
        self.enable_cmd_slew = True
        self.cmd_slew_step = 0.02
        self.prev_pos_cmd = self.default_q[7 : 7 + self.Nu].copy()

        self.ctrl_low = None
        self.ctrl_high = None
        if self.mj_model.actuator_ctrlrange.shape[0] == self.Nu:
            ctrl_low = self.mj_model.actuator_ctrlrange[:, 0].copy()
            ctrl_high = self.mj_model.actuator_ctrlrange[:, 1].copy()
            ctrl_span = np.abs(ctrl_high - ctrl_low)
            # Some models expose [0, 0] ctrlrange for all actuators when limits are not used.
            # In that case clipping would collapse all commands to zero.
            if np.any(ctrl_span > 1e-9):
                self.ctrl_low = ctrl_low
                self.ctrl_high = ctrl_high
        self.foot_site_ids = self._resolve_foot_site_ids()
        self.foot_geom_ids = self._resolve_foot_geom_ids()
        self.foot_contact_source = "none"
        if self.foot_site_ids.size == 4:
            self.foot_contact_source = "site"
        elif self.foot_geom_ids.size == 4:
            self.foot_contact_source = "geom"
        self.foot_contact_history = []

        # publisher
        self.time_shm = shared_memory.SharedMemory(name="time_shm", create=True, size=32)
        self.time_shared = np.ndarray(1, dtype=np.float32, buffer=self.time_shm.buf)
        self.time_shared[0] = 0.0

        self.state_shm = shared_memory.SharedMemory(name="state_shm", create=True, size=self.Nx * 32)
        self.state_shared = np.ndarray((self.Nx,), dtype=np.float32, buffer=self.state_shm.buf)

        # listener
        self.acts_shm = shared_memory.SharedMemory(name="acts_shm", create=True, size=self.n_acts * self.Nu * 32)
        self.acts_shared = np.ndarray((self.n_acts, self.mj_model.nu), dtype=np.float32, buffer=self.acts_shm.buf)
        self.acts_shared[:] = self.default_q[7 : 7 + self.Nu]

        self.refs_shm = shared_memory.SharedMemory(
            name="refs_shm", create=True, size=self.n_acts * self.Nu * 3 * 32
        )
        self.refs_shared = np.ndarray((self.n_acts, self.Nu, 3), dtype=np.float32, buffer=self.refs_shm.buf)
        self.refs_shared[:] = 0.0

        self.plan_time_shm = shared_memory.SharedMemory(name="plan_time_shm", create=True, size=32)
        self.plan_time_shared = np.ndarray(1, dtype=np.float32, buffer=self.plan_time_shm.buf)
        self.plan_time_shared[0] = -self.ctrl_dt

        self.tau_shm = shared_memory.SharedMemory(name="tau_shm", create=True, size=self.n_acts * self.Nu * 32)
        self.tau_shared = np.ndarray((self.n_acts, self.mj_model.nu), dtype=np.float32, buffer=self.tau_shm.buf)
        self.tau_shared[:] = 0.0

    def _resolve_foot_site_ids(self) -> np.ndarray:
        site_ids = []
        for name in self.foot_site_names:
            try:
                sid = int(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, name))
            except Exception:
                sid = -1
            if sid < 0:
                return np.zeros((0,), dtype=np.int32)
            site_ids.append(sid)
        return np.asarray(site_ids, dtype=np.int32)

    def _resolve_foot_geom_ids(self) -> np.ndarray:
        geom_ids = []
        for name in self.foot_geom_names:
            try:
                gid = int(mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM.value, name))
            except Exception:
                gid = -1
            if gid < 0:
                return np.zeros((0,), dtype=np.int32)
            geom_ids.append(gid)
        return np.asarray(geom_ids, dtype=np.int32)

    def _get_foot_z(self) -> np.ndarray | None:
        if self.foot_site_ids.size == 4:
            return np.asarray(self.mj_data.site_xpos[self.foot_site_ids, 2], dtype=np.float64)
        if self.foot_geom_ids.size == 4:
            return np.asarray(self.mj_data.geom_xpos[self.foot_geom_ids, 2], dtype=np.float64)
        return None

    def _safe_position_cmd(self, target):
        cmd = np.asarray(target, dtype=np.float32).copy()

        if self.ctrl_low is not None and self.ctrl_high is not None:
            cmd = np.clip(cmd, self.ctrl_low, self.ctrl_high)

        if self.enable_cmd_slew:
            delta = cmd - self.prev_pos_cmd
            delta = np.clip(delta, -self.cmd_slew_step, self.cmd_slew_step)
            cmd = self.prev_pos_cmd + delta
            self.prev_pos_cmd = cmd.copy()

        return cmd

    def _append_record(self):
        self.data.append(np.concatenate([[self.t], self.mj_data.qpos, self.mj_data.qvel, self.mj_data.ctrl]))
        foot_z = self._get_foot_z()
        if foot_z is not None:
            foot_contact = (foot_z - self.foot_radius) < self.foot_contact_eps
            self.foot_contact_history.append(foot_contact.astype(np.float64))

    def _should_stop(self) -> tuple[bool, str]:
        if self.max_steps > 0 and self.step_count >= self.max_steps:
            return True, "max_steps"
        if self.max_time_sec > 0.0 and self.t >= self.max_time_sec:
            return True, "max_time_sec"
        if self.stop_on_fall and float(self.mj_data.qpos[2]) < self.fall_height_threshold:
            return True, "fall_detected"
        return False, ""

    def _extract_targets(self) -> tuple[float, float, float]:
        target_vx = 0.0
        target_wz = 0.0
        target_height = self.target_base_height

        fixed_vel = self.config_dict.get("fixed_vel_tar")
        if isinstance(fixed_vel, list) and len(fixed_vel) >= 1:
            target_vx = float(fixed_vel[0])
        elif "default_vx" in self.config_dict:
            target_vx = float(self.config_dict.get("default_vx", 0.0))

        fixed_ang = self.config_dict.get("fixed_ang_vel_tar")
        if isinstance(fixed_ang, list) and len(fixed_ang) >= 3:
            target_wz = float(fixed_ang[2])
        elif "default_vyaw" in self.config_dict:
            target_wz = float(self.config_dict.get("default_vyaw", 0.0))

        reward_cfg = self.config_dict.get("reward")
        if isinstance(reward_cfg, dict) and "target_base_height" in reward_cfg:
            target_height = float(reward_cfg.get("target_base_height", target_height))
        elif "target_base_height" in self.config_dict:
            target_height = float(self.config_dict.get("target_base_height", target_height))

        return target_vx, target_wz, target_height

    @staticmethod
    def _roll_pitch_from_quat(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # MuJoCo qpos quaternion order: [w, x, y, z]
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        return roll, pitch

    @staticmethod
    def _angle_dev(angle: np.ndarray, ref: float) -> np.ndarray:
        diff = angle - ref
        return np.arctan2(np.sin(diff), np.cos(diff))

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return 0.0
        aa = np.asarray(a, dtype=np.float64) - float(np.mean(a))
        bb = np.asarray(b, dtype=np.float64) - float(np.mean(b))
        den = np.sqrt(float(np.dot(aa, aa)) * float(np.dot(bb, bb)))
        if den <= 1e-9:
            return 0.0
        corr = float(np.dot(aa, bb) / den)
        return float(np.clip(corr, -1.0, 1.0))

    @staticmethod
    def _body_forward_velocity(quat_wxyz: np.ndarray, vel_world_xyz: np.ndarray) -> np.ndarray:
        if quat_wxyz.ndim != 2 or vel_world_xyz.ndim != 2:
            return np.zeros((0,), dtype=np.float64)
        if quat_wxyz.shape[0] != vel_world_xyz.shape[0]:
            return np.zeros((0,), dtype=np.float64)
        if quat_wxyz.shape[1] < 4 or vel_world_xyz.shape[1] < 3:
            return np.zeros((0,), dtype=np.float64)

        q = np.asarray(quat_wxyz[:, :4], dtype=np.float64)
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q_norm = np.where(q_norm > 1e-9, q_norm, 1.0)
        q = q / q_norm
        w = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        # First column of body->world rotation matrix.
        r11 = 1.0 - 2.0 * (y * y + z * z)
        r21 = 2.0 * (x * y + w * z)
        r31 = 2.0 * (x * z - w * y)

        v = np.asarray(vel_world_xyz[:, :3], dtype=np.float64)
        return v[:, 0] * r11 + v[:, 1] * r21 + v[:, 2] * r31

    def _infer_gait_metrics(self) -> dict:
        default_score = 0.5 if self.expected_gait else 1.0
        default = {
            "predicted_gait": "unknown",
            "predicted_gait_confidence": 0.0,
            "gait_match_score": default_score,
            "gait_diag_sync": 0.0,
            "gait_lateral_sync": 0.0,
            "gait_front_sync": 0.0,
            "gait_hind_sync": 0.0,
            "gait_switches_per_sec": 0.0,
            "contact_match_score": default_score,
        }
        if len(self.foot_contact_history) < 8:
            return default

        contact = np.asarray(self.foot_contact_history, dtype=np.float64)
        if contact.ndim != 2 or contact.shape[1] != 4:
            return default

        fl = contact[:, 0]
        fr = contact[:, 1]
        rl = contact[:, 2]
        rr = contact[:, 3]

        diag_sync = 0.5 * (self._safe_corr(fl, rr) + self._safe_corr(fr, rl))
        lateral_sync = 0.5 * (self._safe_corr(fl, rl) + self._safe_corr(fr, rr))
        front_sync = self._safe_corr(fl, fr)
        hind_sync = self._safe_corr(rl, rr)

        switches = np.sum(np.abs(np.diff(contact, axis=0)) > 0.5, axis=0)
        duration = max(float(self.t), 1e-6)
        switches_per_sec = float(np.mean(switches) / duration)

        max_sync = max(diag_sync, lateral_sync, front_sync, hind_sync)
        if switches_per_sec < 0.2:
            predicted_gait = "stand"
            confidence = 0.9
        elif diag_sync > 0.30 and diag_sync > lateral_sync + 0.12:
            predicted_gait = "trot"
            confidence = float(np.clip(0.55 + 0.45 * (diag_sync - lateral_sync), 0.0, 1.0))
        elif lateral_sync > 0.30 and lateral_sync > diag_sync + 0.12:
            predicted_gait = "pace"
            confidence = float(np.clip(0.55 + 0.45 * (lateral_sync - diag_sync), 0.0, 1.0))
        elif switches_per_sec >= 0.5 and max_sync < 0.45:
            predicted_gait = "walk"
            confidence = float(np.clip(0.50 + 0.50 * (0.45 - max_sync), 0.0, 1.0))
        else:
            predicted_gait = "mixed"
            confidence = float(np.clip(0.45 + 0.25 * (1.0 - abs(diag_sync - lateral_sync)), 0.0, 1.0))

        expected = self.expected_gait
        activity_score = float(np.clip((switches_per_sec - 0.20) / 0.80, 0.0, 1.0))
        if expected == "walk":
            pattern_score = 1.0 - max(diag_sync, lateral_sync)
            gait_match_score = 0.5 * pattern_score + 0.5 * activity_score
        elif expected == "trot":
            pattern_score = 0.5 + 0.5 * (diag_sync - lateral_sync)
            gait_match_score = 0.7 * pattern_score + 0.3 * activity_score
        elif expected == "pace":
            pattern_score = 0.5 + 0.5 * (lateral_sync - diag_sync)
            gait_match_score = 0.7 * pattern_score + 0.3 * activity_score
        elif expected in {"stand", ""}:
            gait_match_score = 0.5 + 0.5 * confidence
        else:
            gait_match_score = 0.5
        gait_match_score = float(np.clip(gait_match_score, 0.0, 1.0))
        if expected in {"walk", "trot", "pace"} and predicted_gait == "stand":
            gait_match_score = min(gait_match_score, 0.2)
        if expected and predicted_gait == expected:
            gait_match_score = max(gait_match_score, float(np.clip(0.75 + 0.25 * confidence, 0.0, 1.0)))

        return {
            "predicted_gait": predicted_gait,
            "predicted_gait_confidence": confidence,
            "gait_match_score": gait_match_score,
            "gait_diag_sync": diag_sync,
            "gait_lateral_sync": lateral_sync,
            "gait_front_sync": front_sync,
            "gait_hind_sync": hind_sync,
            "gait_switches_per_sec": switches_per_sec,
            "contact_match_score": gait_match_score,
        }

    def compute_metrics(self) -> dict:
        if len(self.data) == 0:
            return {
                "success": False,
                "fail_reason": self.stop_reason if self.stop_reason != "running" else "no_data",
                "fall_time_sec": -1.0,
                "fall_count": 0,
                "mean_roll": 0.0,
                "mean_pitch": 0.0,
                "mean_height_error": 0.0,
                "mean_vel_error": 0.0,
                "yaw_rate_error": 0.0,
                "forward_progress_m": 0.0,
                "mean_forward_speed_mps": 0.0,
                "forward_motion_ratio": 0.0,
                "expected_gait": self.expected_gait,
                "foot_contact_source": self.foot_contact_source,
                "predicted_gait": "unknown",
                "predicted_gait_confidence": 0.0,
                "gait_match_score": 0.0,
                "gait_diag_sync": 0.0,
                "gait_lateral_sync": 0.0,
                "gait_front_sync": 0.0,
                "gait_hind_sync": 0.0,
                "gait_switches_per_sec": 0.0,
                "contact_match_score": 0.0,
                "energy": 0.0,
                "episode_return": -1e6,
                "steps": int(self.step_count),
                "sim_time_sec": float(self.t),
                "stop_reason": self.stop_reason,
            }

        data = np.array(self.data, dtype=np.float64)
        nq = self.mj_model.nq
        nv = self.mj_model.nv

        t_series = data[:, 0]
        qpos = data[:, 1 : 1 + nq]
        qvel = data[:, 1 + nq : 1 + nq + nv]
        ctrl = data[:, 1 + nq + nv :]

        base_z = qpos[:, 2] if nq > 2 else np.zeros(data.shape[0], dtype=np.float64)
        fall_mask = base_z < self.fall_height_threshold
        prev_mask = np.concatenate(([False], fall_mask[:-1]))
        fall_events = np.logical_and(fall_mask, np.logical_not(prev_mask))
        fall_count = int(fall_events.sum())
        fall_time_sec = float(t_series[np.argmax(fall_events)]) if fall_count > 0 else -1.0

        if nq >= 7:
            quat = qpos[:, 3:7]
            roll, pitch = self._roll_pitch_from_quat(quat)
            mean_roll = float(np.mean(np.abs(roll)))
            mean_pitch = float(np.mean(np.abs(pitch)))
            roll_dev = self._angle_dev(roll, self.home_roll)
            pitch_dev = self._angle_dev(pitch, self.home_pitch)
            mean_roll_dev = float(np.mean(np.abs(roll_dev)))
            mean_pitch_dev = float(np.mean(np.abs(pitch_dev)))
            max_pitch_dev = float(np.max(np.abs(pitch_dev)))
        else:
            mean_roll = 0.0
            mean_pitch = 0.0
            mean_roll_dev = 0.0
            mean_pitch_dev = 0.0
            max_pitch_dev = 0.0

        target_vx, target_wz, target_height = self._extract_targets()

        actual_vx_world = qvel[:, 0] if nv > 0 else np.zeros(data.shape[0], dtype=np.float64)
        actual_wz = qvel[:, 5] if nv > 5 else np.zeros(data.shape[0], dtype=np.float64)
        if nq >= 7 and nv >= 3:
            body_vx = self._body_forward_velocity(qpos[:, 3:7], qvel[:, :3])
        else:
            body_vx = actual_vx_world
        if body_vx.size != data.shape[0]:
            body_vx = actual_vx_world

        mean_vel_error = float(np.mean(np.abs(body_vx - target_vx)))
        yaw_rate_error = float(np.mean(np.abs(actual_wz - target_wz)))
        mean_height_error = float(np.mean(np.abs(base_z - target_height)))
        if t_series.size >= 2:
            forward_progress_m = float(np.trapz(body_vx, t_series))
        else:
            forward_progress_m = float(body_vx[0] * self.sim_dt) if body_vx.size > 0 else 0.0
        sim_span = max(float(t_series[-1] - t_series[0]), self.sim_dt)
        mean_forward_speed_mps = float(forward_progress_m / sim_span)
        if abs(target_vx) > 1e-6:
            dir_sign = np.sign(target_vx)
            speed_thr = max(0.02, 0.2 * abs(target_vx))
            forward_motion_ratio = float(np.mean((body_vx * dir_sign) > speed_thr))
        else:
            forward_motion_ratio = float(np.mean(np.abs(body_vx) < 0.05))
        gait_metrics = self._infer_gait_metrics()
        contact_match_score = float(gait_metrics.get("contact_match_score", 0.5))

        if nv > 6:
            joint_vel = qvel[:, 6 : 6 + self.Nu]
        else:
            joint_vel = np.zeros((data.shape[0], 0), dtype=np.float64)

        if joint_vel.shape[1] > 0 and ctrl.shape[1] > 0:
            m = min(joint_vel.shape[1], ctrl.shape[1])
            power = ctrl[:, :m] * joint_vel[:, :m]
            energy = float(np.mean(np.sum(np.square(power), axis=1)))
        else:
            energy = 0.0

        success = bool((fall_count == 0) and (self.sim_exception is None))
        fail_reason = ""
        if not success:
            if self.sim_exception is not None:
                fail_reason = "sim_exception"
            elif fall_count > 0:
                fail_reason = "fall_detected"
            else:
                fail_reason = self.stop_reason if self.stop_reason != "running" else "unknown"

        episode_return = float(
            -(
                5.0 * mean_vel_error
                + 2.0 * mean_height_error
                + 0.5 * (mean_roll + mean_pitch)
                - 0.5 * forward_progress_m
                - 0.5 * contact_match_score
                + 0.01 * energy
            )
        )

        return {
            "success": success,
            "fail_reason": fail_reason,
            "fall_time_sec": fall_time_sec,
            "fall_count": fall_count,
            "mean_roll": mean_roll,
            "mean_pitch": mean_pitch,
            "mean_roll_dev": mean_roll_dev,
            "mean_pitch_dev": mean_pitch_dev,
            "max_pitch_dev": max_pitch_dev,
            "mean_height_error": mean_height_error,
            "mean_vel_error": mean_vel_error,
            "yaw_rate_error": yaw_rate_error,
            "forward_progress_m": forward_progress_m,
            "mean_forward_speed_mps": mean_forward_speed_mps,
            "forward_motion_ratio": forward_motion_ratio,
            "expected_gait": self.expected_gait,
            "foot_contact_source": self.foot_contact_source,
            **gait_metrics,
            "contact_match_score": contact_match_score,
            "energy": energy,
            "episode_return": episode_return,
            "steps": int(self.step_count),
            "sim_time_sec": float(self.t),
            "stop_reason": self.stop_reason,
        }

    def main_loop(self):
        if self.plot:
            fig, axs = plt.subplots(self.n_plot_joint, 1, figsize=(12, 12))
            handles = []
            handles_ref = []
            colors = plt.cm.rainbow(np.linspace(0, 1, self.n_plot_joint))
            for i in range(self.n_plot_joint):
                handles.append(axs[i].plot(self.q_history[:, i], color=colors[i])[0])
                handles_ref.append(axs[i].plot(self.qref_history[:, i], color=colors[i], linestyle="--")[0])
                axs[i].set_ylim(-1.0 + self.default_q[i + 7], 1.0 + self.default_q[i + 7])
                axs[i].set_xlabel("Time (s)")
                axs[i].set_ylabel(f"Joint {i+1} Position")
            plt.show(block=False)

        viewer = None
        if not self.headless:
            viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
            )

            if self.draw_refs:
                cnt = 0
                viewer.user_scn.ngeom = 0
                for i in range(self.n_acts - 1):
                    for j in range(self.mj_model.nu):
                        color = np.array(
                            [1.0 * i / max(1, (self.n_acts - 1)), 1.0 * j / max(1, self.mj_model.nu), 0.0, 1.0]
                        )
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[cnt],
                            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                            size=np.zeros(3),
                            rgba=color,
                            pos=self.refs_shared[i, j, :],
                            mat=np.eye(3).flatten(),
                        )
                        cnt += 1
                viewer.user_scn.ngeom = cnt
            else:
                viewer.user_scn.ngeom = 0
            viewer.sync()

        while True:
            if self.plot:
                for j in range(self.n_plot_joint):
                    handles[j].set_ydata(self.acts_shared[:, j])
                    handles_ref[j].set_ydata(self.qref_history[:, j])
                plt.pause(0.001)

            if self.draw_refs and viewer is not None:
                for i in range(self.n_acts - 1):
                    for j in range(self.mj_model.nu):
                        r0 = self.refs_shared[i, j, :]
                        r1 = self.refs_shared[i + 1, j, :]
                        mujoco.mjv_connector(
                            viewer.user_scn.geoms[i * self.mj_model.nu + j],
                            mujoco.mjtGeom.mjGEOM_CAPSULE,
                            0.02,
                            r0,
                            r1,
                        )

            if self.sync_mode:
                q = self.mj_data.qpos
                while self.t <= (self.plan_time_shared[0] + self.ctrl_dt):
                    if self.leg_control == "position":
                        self.mj_data.ctrl = self._safe_position_cmd(self.acts_shared[0])
                    elif self.leg_control == "torque":
                        self.mj_data.ctrl = self.tau_shared[0]

                    if self.record:
                        self._append_record()

                    mujoco.mj_step(self.mj_model, self.mj_data)
                    self.t += self.sim_dt
                    self.step_count += 1

                    q = self.mj_data.qpos
                    qd = self.mj_data.qvel
                    state = np.concatenate([q, qd])
                    self.time_shared[:] = self.t
                    self.state_shared[:] = state

                    should_stop, reason = self._should_stop()
                    if should_stop:
                        self.stop_reason = reason
                        break

                self.q_history = np.roll(self.q_history, -1, axis=0)
                self.q_history[-1, :] = q[7:]
                self.qref_history = np.roll(self.qref_history, -1, axis=0)
                self.qref_history[-1, :] = self.mj_data.ctrl
                if viewer is not None:
                    viewer.sync()

                if self.stop_reason != "running":
                    break

            else:
                t0 = time.time()
                if self.plan_time_shared[0] < 0.0:
                    time.sleep(0.01)
                    continue

                delta_time = self.t - self.plan_time_shared[0]
                delta_step = int(delta_time / self.ctrl_dt)
                if delta_time > self.ctrl_dt / self.real_time_factor:
                    print(f"[WARN] Delayed by {delta_time * 1000.0:.1f} ms")
                if delta_step >= self.n_acts or delta_step < 0:
                    delta_step = self.n_acts - 1

                if self.leg_control == "position":
                    self.mj_data.ctrl = self._safe_position_cmd(self.acts_shared[delta_step])
                elif self.leg_control == "torque":
                    self.mj_data.ctrl = self.tau_shared[delta_step]

                if self.record:
                    self._append_record()

                mujoco.mj_step(self.mj_model, self.mj_data)
                self.t += self.sim_dt
                self.step_count += 1

                q = self.mj_data.qpos
                qd = self.mj_data.qvel
                state = np.concatenate([q, qd])

                self.time_shared[:] = self.t
                self.state_shared[:] = state

                self.q_history = np.roll(self.q_history, -1, axis=0)
                self.q_history[-1, :] = q[7:]
                self.qref_history = np.roll(self.qref_history, -1, axis=0)
                self.qref_history[-1, :] = self.mj_data.ctrl
                if viewer is not None:
                    viewer.sync()

                should_stop, reason = self._should_stop()
                if should_stop:
                    self.stop_reason = reason
                    break

                duration = time.time() - t0
                if duration < self.sim_dt / self.real_time_factor:
                    time.sleep(self.sim_dt / self.real_time_factor - duration)
                else:
                    print("[WARN] Sim loop overruns")

        if self.stop_reason == "running":
            self.stop_reason = "completed"

    @staticmethod
    def _safe_close_unlink(shm_obj):
        try:
            shm_obj.close()
        except Exception:
            pass
        try:
            shm_obj.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def close(self):
        self._safe_close_unlink(self.time_shm)
        self._safe_close_unlink(self.state_shm)
        self._safe_close_unlink(self.acts_shm)
        self._safe_close_unlink(self.plan_time_shm)
        self._safe_close_unlink(self.refs_shm)
        self._safe_close_unlink(self.tau_shm)


def main(args=None):
    art.tprint("LeCAR @ CMU\nDIAL-MPC\nSIMULATOR", font="big", chr_ignore=True)
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    group.add_argument("--example", type=str, default=None, help="Example to run")
    group.add_argument("--list-examples", action="store_true", help="List available examples")
    parser.add_argument("--custom-env", type=str, default=None, help="Custom environment to import dynamically")
    args = parser.parse_args(args)

    if args.custom_env is not None:
        sys.path.append(os.getcwd())
        importlib.import_module(args.custom_env)

    if args.list_examples:
        print("Available examples:")
        for example in deploy_examples:
            print(f"  - {example}")
        return

    if args.example is not None:
        if args.example not in deploy_examples:
            print(f"Example {args.example} not found.")
            return
        config_path = get_example_path(args.example + ".yaml")
    else:
        config_path = os.path.abspath(args.config)

    print(f"[CONFIG] dial_sim.py using: {config_path}")
    config_dict = yaml.safe_load(open(config_path, "r"))

    sim_config = load_dataclass_from_dict(DialSimConfig, config_dict)
    env_config = load_dataclass_from_dict(BaseEnvConfig, config_dict)
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    mujoco_env = DialSim(sim_config, env_config, dial_config, config_dict=config_dict)

    try:
        mujoco_env.main_loop()
    except KeyboardInterrupt:
        mujoco_env.stop_reason = "keyboard_interrupt"
    except Exception as exc:
        mujoco_env.sim_exception = str(exc)
        mujoco_env.stop_reason = "sim_exception"
        raise
    finally:
        base_output_dir = Path(dial_config.output_dir)
        base_output_dir.mkdir(parents=True, exist_ok=True)

        artifact_dir = None
        if mujoco_env.record and len(mujoco_env.data) > 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            data = np.array(mujoco_env.data)
            artifact_dir = base_output_dir / f"sim_{dial_config.env_name}_{env_config.task_name}_{timestamp}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            np.save(artifact_dir / "states.npy", data)

        metrics = mujoco_env.compute_metrics()
        if artifact_dir is not None:
            metrics["artifact_dir"] = str(artifact_dir)

        metrics_path = base_output_dir / sim_config.metrics_filename
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[METRICS] wrote {metrics_path}")

        mujoco_env.close()


if __name__ == "__main__":
    main()
