import os
import time
from multiprocessing import shared_memory
import importlib
import sys
import argparse

try:
    from multiprocessing import resource_tracker
except Exception:
    resource_tracker = None

import yaml
import numpy as np
import art

import jax
from jax import numpy as jnp

import brax.envs as brax_envs
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_core import DialConfig, MBDPI
from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
from dial_mpc.utils.io_utils import (
    load_dataclass_from_dict,
    get_example_path,
)
from dial_mpc.examples import deploy_examples


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags


def _unregister_shared_memory(shm_obj: shared_memory.SharedMemory) -> None:
    # Planner is not the owner of these shared memories (create=False).
    # Unregister from resource_tracker to avoid noisy "leaked shared_memory"
    # warnings when simulator already unlinked segments.
    if resource_tracker is None:
        return
    shm_name = getattr(shm_obj, "_name", None) or getattr(shm_obj, "name", None)
    if not shm_name:
        return
    try:
        resource_tracker.unregister(shm_name, "shared_memory")
    except Exception:
        pass


def _attach_shared_memory(name: str, size: int) -> shared_memory.SharedMemory:
    shm_obj = shared_memory.SharedMemory(name=name, create=False, size=size)
    _unregister_shared_memory(shm_obj)
    return shm_obj


class MBDPublisher:
    def __init__(self, env: BaseEnv, env_config: BaseEnvConfig, dial_config: DialConfig):
        self.dial_config = dial_config
        self.env = env
        self.env_config = env_config

        self.mbdpi = MBDPI(self.dial_config, self.env)
        self.rng = jax.random.PRNGKey(seed=self.dial_config.seed)
        self.pipeline_init_jit = jax.jit(self.env.pipeline_init)
        self.shift_vmap = jax.jit(jax.vmap(self.shift, in_axes=(1, None), out_axes=1))

        self.Y = jnp.zeros([self.dial_config.Hnode + 1, self.mbdpi.nu])
        self.ctrl_dt = env_config.dt

        self.n_acts = self.dial_config.Hsample + 1
        self.nx = self.env.sys.mj_model.nq + self.env.sys.mj_model.nv
        self.nu = self.env.sys.mj_model.nu
        self.default_q = self.env.sys.mj_model.keyframe("home").qpos

        # warm-start / anti-spike
        self.startup_hold_sec = float(getattr(self.env_config, "startup_hold_sec", 2.0))
        self._startup_t0 = None
        self._hold_release_t0 = None
        warmup_min_sec = float(getattr(self.env_config, "planner_warmup_min_sec", 4.0))
        warmup_scale = float(getattr(self.env_config, "planner_warmup_scale", 0.35))
        self.warmup_sec = max(
            warmup_min_sec, float(getattr(self.env_config, "ramp_up_time", 10.0)) * warmup_scale
        )
        self.enable_warmstart = bool(getattr(self.env_config, "planner_enable_warmstart", True))

        self._q_home = jnp.array(self.default_q[7:7 + self.nu])
        self.enable_joint_clamp = bool(getattr(self.env_config, "planner_enable_joint_clamp", True))
        self.max_joint_delta = float(getattr(self.env_config, "planner_max_joint_delta", 0.012))
        self.max_seq_delta = float(getattr(self.env_config, "planner_max_seq_delta", 0.010))
        self._jt_prev = jnp.array(self.default_q[7:7 + self.nu])

        self._debug_counter = 0
        self._was_in_hold = False

        # publisher
        self.acts_shm = _attach_shared_memory(name="acts_shm", size=self.n_acts * self.nu * 32)
        self.acts_shared = np.ndarray(
            (self.n_acts, self.nu), dtype=np.float32, buffer=self.acts_shm.buf
        )
        self.acts_shared[:] = self.default_q[7:7 + self.nu]

        self.refs_shm = _attach_shared_memory(
            name="refs_shm", size=self.n_acts * self.env.sys.nu * 3 * 32
        )
        self.refs_shared = np.ndarray(
            (self.n_acts, self.env.sys.nu, 3), dtype=np.float32, buffer=self.refs_shm.buf
        )
        self.refs_shared[:] = 1.0

        self.plan_time_shm = _attach_shared_memory(name="plan_time_shm", size=32)
        self.plan_time_shared = np.ndarray(1, dtype=np.float32, buffer=self.plan_time_shm.buf)
        self.plan_time_shared[0] = -0.02

        # listener
        self.time_shm = _attach_shared_memory(name="time_shm", size=32)
        self.time_shared = np.ndarray(1, dtype=np.float32, buffer=self.time_shm.buf)
        self.time_shared[0] = 0.0

        self.state_shm = _attach_shared_memory(name="state_shm", size=self.nx * 32)
        self.state_shared = np.ndarray((self.nx,), dtype=np.float32, buffer=self.state_shm.buf)
        self.state_shared[: self.default_q.shape[0]] = self.default_q

        self.tau_shm = _attach_shared_memory(name="tau_shm", size=self.n_acts * self.nu * 32)
        self.tau_shared = np.ndarray(
            (self.n_acts, self.nu), dtype=np.float32, buffer=self.tau_shm.buf
        )

    def shift(self, x, shift_time):
        spline = InterpolatedUnivariateSpline(self.mbdpi.step_nodes, x, k=2)
        return spline(self.mbdpi.step_nodes + shift_time)

    def init_mjx_state(self, q, qd, t):
        q = jnp.array(q)
        qd = jnp.array(qd)
        state = self.env.reset(jax.random.PRNGKey(0))
        pipeline_state = self.pipeline_init_jit(q, qd)
        obs = self.env._get_obs(pipeline_state, state.info)
        return state.replace(pipeline_state=pipeline_state, obs=obs)

    def update_mjx_state(self, state, q, qd, t):
        q = jnp.array(q)
        qd = jnp.array(qd)
        pipeline_state = state.pipeline_state.replace(qpos=q, qvel=qd)
        step = int(t / self.ctrl_dt)
        info = state.info
        info["step"] = step
        return state.replace(pipeline_state=pipeline_state, info=info)

    def _in_startup_hold(self) -> bool:
        if self.startup_hold_sec <= 1e-6:
            return False
        if self._startup_t0 is None:
            self._startup_t0 = time.time()
            return True
        return (time.time() - self._startup_t0) < self.startup_hold_sec

    def _compute_alpha(self) -> float:
        if self._hold_release_t0 is None:
            return 0.0
        elapsed = time.time() - self._hold_release_t0
        if self.warmup_sec <= 1e-6:
            return 1.0
        return float(max(0.0, min(1.0, elapsed / self.warmup_sec)))

    def _warmstart_blend_jt0(self, jt0: jax.Array, alpha: float) -> jax.Array:
        alpha_smooth = alpha * alpha
        return (1.0 - alpha_smooth) * self._q_home + alpha_smooth * jt0

    def _blend_seq_home(self, jt_seq: jax.Array, alpha: float) -> jax.Array:
        alpha_smooth = alpha * alpha
        return (1.0 - alpha_smooth) * self._q_home[None, :] + alpha_smooth * jt_seq

    def _clamp_jt0(self, jt0: jax.Array) -> jax.Array:
        if not self.enable_joint_clamp:
            return jt0
        delta = jt0 - self._jt_prev
        delta = jnp.clip(delta, -self.max_joint_delta, self.max_joint_delta)
        out = self._jt_prev + delta
        self._jt_prev = out
        return out

    def _clamp_seq_delta(self, jt_seq: jax.Array, max_step: float) -> jax.Array:
        if jt_seq.shape[0] <= 1:
            return jt_seq

        def _one(prev, curr):
            d = jnp.clip(curr - prev, -max_step, max_step)
            out = prev + d
            return out, out

        first = jt_seq[0]
        _, rest = jax.lax.scan(_one, first, jt_seq[1:])
        return jnp.concatenate([first[None, :], rest], axis=0)

    def _to_joint_seq(self, joint_targets: jax.Array) -> jax.Array:
        jt = joint_targets
        if not hasattr(jt, "ndim"):
            jt = jnp.array(jt)

        if jt.ndim == 1:
            seq = jnp.tile(jt[None, :], (self.n_acts, 1))
        elif jt.ndim == 2:
            if jt.shape[1] == self.nu:
                seq = jt
            elif jt.shape[0] == self.nu:
                seq = jt.T
            else:
                seq = jnp.tile(self._q_home[None, :], (self.n_acts, 1))
        elif jt.ndim == 3:
            cand = jt[0]
            if cand.ndim == 2 and cand.shape[1] == self.nu:
                seq = cand
            elif cand.ndim == 2 and cand.shape[0] == self.nu:
                seq = cand.T
            else:
                flat = cand.reshape(-1, cand.shape[-1])
                if flat.shape[1] == self.nu:
                    seq = flat
                elif flat.shape[0] == self.nu:
                    seq = flat.T
                else:
                    seq = jnp.tile(self._q_home[None, :], (self.n_acts, 1))
        else:
            seq = jnp.tile(self._q_home[None, :], (self.n_acts, 1))

        h = seq.shape[0]
        if h < self.n_acts:
            seq = jnp.concatenate([seq, jnp.repeat(seq[-1:, :], self.n_acts - h, axis=0)], axis=0)
        elif h > self.n_acts:
            seq = seq[: self.n_acts, :]
        return seq

    def _first_u(self, us: jax.Array) -> jax.Array:
        if not hasattr(us, "ndim"):
            return us
        if us.ndim == 1:
            return us
        if us.ndim == 2:
            return us[:, 0] if us.shape[0] == self.nu else us[0, :]
        return us[0, :, 0] if us.shape[1] == self.nu else us[0, 0, :]

    def main_loop(self):
        def reverse_scan(rng_Y0_state, factor):
            rng, Y0, state = rng_Y0_state
            rng, Y0, info = self.mbdpi.reverse_once(state, rng, Y0, factor)
            return (rng, Y0, state), info

        last_plan_time = self.time_shared[0]
        state = self.init_mjx_state(
            self.state_shared[: self.env.sys.mj_model.nq].copy(),
            self.state_shared[self.env.sys.mj_model.nq :].copy(),
            last_plan_time.copy(),
        )

        first_time = True
        while True:
            t0 = time.time()

            plan_time = self.time_shared[0]
            state = self.update_mjx_state(
                state,
                self.state_shared[: self.env.sys.mj_model.nq],
                self.state_shared[self.env.sys.mj_model.nq :],
                plan_time,
            )

            in_hold = self._in_startup_hold()
            if in_hold:
                self.Y = self.Y * 0.0
                jt_home = jnp.tile(self._q_home[None, :], (self.n_acts, 1))
                self.acts_shared[:, :] = np.asarray(jt_home, dtype=np.float32)
                self.tau_shared[:, :] = 0.0
                self.refs_shared[:, :, :] = 0.0
                self.plan_time_shared[0] = plan_time
                self._was_in_hold = True
                last_plan_time = plan_time
                continue

            if self._was_in_hold:
                # release edge: reset planner state to avoid impulse at hold-off
                self.Y = self.Y * 0.0
                self._jt_prev = jnp.array(self._q_home)
                self._hold_release_t0 = time.time()
                self._was_in_hold = False

            shift_time = plan_time - last_plan_time
            if shift_time > self.ctrl_dt + 1e-3:
                print(f"[WARN] sim overtime {(shift_time - self.ctrl_dt) * 1000:.1f} ms")
            if shift_time > self.ctrl_dt * self.n_acts:
                print(f"[WARN] long time unplanned {shift_time * 1000:.1f} ms, reset control")
                self.Y = self.Y * 0.0
            else:
                self.Y = self.shift_vmap(self.Y, shift_time)

            if first_time:
                print("Performing JIT on DIAL-MPC")
                factors = self.dial_config.traj_diffuse_factor ** (
                    jnp.arange(self.dial_config.Ndiffuse_init)
                )[:, None]
                (self.rng, self.Y, _), info = jax.lax.scan(
                    reverse_scan, (self.rng, self.Y, state), factors
                )
                first_time = False

            factors = self.dial_config.traj_diffuse_factor ** (
                jnp.arange(self.dial_config.Ndiffuse)
            )[:, None]
            (self.rng, self.Y, _), info = jax.lax.scan(
                reverse_scan, (self.rng, self.Y, state), factors
            )

            x_targets = info["xbar"][-1, :, 1:, :3]

            us = self.mbdpi.node2u_vmap(self.Y)
            us0 = self._first_u(us)

            joint_targets = self.env.act2joint(us)
            jt_seq = self._to_joint_seq(joint_targets)

            if hasattr(self.env, "joint_range"):
                jt_seq = jnp.clip(
                    jt_seq,
                    self.env.joint_range[:, 0],
                    self.env.joint_range[:, 1],
                )

            alpha = self._compute_alpha() if self.enable_warmstart else 1.0
            jt0 = self._warmstart_blend_jt0(jt_seq[0], alpha)
            jt0 = self._clamp_jt0(jt0)

            if alpha < 1.0:
                jt_seq = self._blend_seq_home(jt_seq, alpha)
                jt_seq = jt_seq.at[0].set(jt0)
            else:
                jt_seq = jt_seq.at[0].set(jt0)

            jt_seq = self._clamp_seq_delta(jt_seq, self.max_seq_delta)
            if hasattr(self.env, "joint_range"):
                jt_seq = jnp.clip(
                    jt_seq,
                    self.env.joint_range[:, 0],
                    self.env.joint_range[:, 1],
                )

            taus = self.env.act2tau(us0, state.pipeline_state)
            if alpha < 1.0:
                taus = taus * 0.0

            jt_seq_np = np.asarray(jt_seq, dtype=np.float32)
            self.acts_shared[:, :] = jt_seq_np

            tau_np = np.asarray(taus, dtype=np.float32).reshape(1, -1)
            self.tau_shared[:, :] = np.repeat(tau_np, self.n_acts, axis=0)

            self.plan_time_shared[0] = plan_time
            self.refs_shared[:, :, :] = x_targets[: self.refs_shared.shape[0], :, :]

            self._debug_counter += 1
            if self._debug_counter % 100 == 0:
                std0 = float(np.std(jt_seq_np[:, 0]))
                norm0 = float(np.linalg.norm(jt_seq_np[0] - np.asarray(self._q_home)))
                print(
                    f"[DBG] alpha={alpha:.2f}, jt_std0={std0:.5f}, "
                    f"jt0_from_home={norm0:.5f}, t={plan_time:.2f}"
                )

            last_plan_time = plan_time
            if time.time() - t0 > self.ctrl_dt:
                print(f"[WARN] real overtime {(time.time() - t0) * 1000:.1f} ms")

    @staticmethod
    def _safe_close(shm_obj):
        try:
            shm_obj.close()
        except Exception:
            pass

    def close(self):
        self._safe_close(self.acts_shm)
        self._safe_close(self.refs_shm)
        self._safe_close(self.plan_time_shm)
        self._safe_close(self.time_shm)
        self._safe_close(self.state_shm)
        self._safe_close(self.tau_shm)


def main(args=None):
    art.tprint("LeCAR @ CMU\nDIAL-MPC\nPLANNER", font="big", chr_ignore=True)
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

    print(f"[CONFIG] dial_plan.py using: {config_path}")
    config_dict = yaml.safe_load(open(config_path, "r"))

    print("Creating environment")
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(env_config_type, config_dict, convert_list_to_array=True)
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)

    mbd_publisher = MBDPublisher(env, env_config, dial_config)

    try:
        mbd_publisher.main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        mbd_publisher.close()


if __name__ == "__main__":
    main()
