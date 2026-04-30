# -*- coding: utf-8 -*-
"""Thin PPO wrapper around Stable-Baselines3 used by the project scripts."""

from __future__ import annotations

from typing import Any

from stable_baselines3 import PPO


SAFE_SAVE_EXCLUDES = (
    "lr_schedule",
    "clip_range",
    "clip_range_vf",
)


def safe_save_sb3_model(model: PPO, path: str) -> None:
    try:
        model.save(path)
    except IndexError as exc:
        if "tuple index out of range" not in str(exc):
            raise
        model.save(path, exclude=list(SAFE_SAVE_EXCLUDES))


class UnifiedPPOModel:
    """Small compatibility wrapper for the repo's trainer/runner APIs."""

    def __init__(self, env, **model_config: Any):
        self.env = env
        self.model_config = dict(model_config)
        policy = self.model_config.pop("policy", "MlpPolicy")
        self.model = PPO(policy, env, **self.model_config)

    def train(self, total_timesteps: int, callback=None) -> None:
        self.model.learn(total_timesteps=int(total_timesteps), callback=callback)

    def continue_training(self, additional_timesteps: int, callback=None) -> None:
        self.model.learn(
            total_timesteps=int(additional_timesteps),
            callback=callback,
            reset_num_timesteps=False,
        )

    def predict(self, observation, deterministic: bool = True):
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str) -> None:
        safe_save_sb3_model(self.model, path)

    @classmethod
    def load(cls, path: str, env=None) -> "UnifiedPPOModel":
        wrapper = cls.__new__(cls)
        wrapper.env = env
        wrapper.model_config = {}
        wrapper.model = PPO.load(path, env=env)
        if env is not None:
            wrapper.model.set_env(env)
        return wrapper
