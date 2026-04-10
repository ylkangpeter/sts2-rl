# -*- coding: utf-8 -*-
"""Thread-parallel VecEnv for IO-bound HTTP training environments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


class ThreadedVecEnv(DummyVecEnv):
    """A VecEnv that keeps envs in one process and steps them in parallel threads."""

    def __init__(self, env_fns: list):
        super().__init__(env_fns)
        self._executor = ThreadPoolExecutor(max_workers=self.num_envs, thread_name_prefix="st2rl-env")

    def _step_one(self, env_i: int) -> tuple[int, Any, float, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.envs[env_i].step(self.actions[env_i])
        done = bool(terminated or truncated)
        if done:
            info = dict(info)
            info["TimeLimit.truncated"] = bool(truncated and not terminated)
            info["terminal_observation"] = obs
            obs, self.reset_infos[env_i] = self.envs[env_i].reset()
        return env_i, obs, float(reward), done, info

    def _reset_one(self, env_i: int) -> tuple[int, Any, dict[str, Any]]:
        options = getattr(self, "_options", None)
        maybe_options = {"options": options[env_i]} if options and options[env_i] else {}
        obs, reset_info = self.envs[env_i].reset(seed=self._seeds[env_i], **maybe_options)
        return env_i, obs, reset_info

    def step_wait(self):
        futures = [self._executor.submit(self._step_one, env_i) for env_i in range(self.num_envs)]
        for future in futures:
            env_i, obs, reward, done, info = future.result()
            self.buf_rews[env_i] = reward
            self.buf_dones[env_i] = done
            self.buf_infos[env_i] = info
            self._save_obs(env_i, obs)
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), list(self.buf_infos)

    def reset(self):
        futures = [self._executor.submit(self._reset_one, env_i) for env_i in range(self.num_envs)]
        for future in futures:
            env_i, obs, reset_info = future.result()
            self.reset_infos[env_i] = reset_info
            self._save_obs(env_i, obs)
        self._reset_seeds()
        reset_options = getattr(self, "_reset_options", None)
        if callable(reset_options):
            reset_options()
        return self._obs_from_buf()

    def close(self) -> None:
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        finally:
            super().close()

    def get_images(self) -> Sequence[np.ndarray | None]:
        return [env.render() if getattr(env, "render_mode", None) == "rgb_array" else None for env in self.envs]
