# -*- coding: utf-8 -*-
"""Training callbacks"""

import os
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from st2rl.training.telemetry import TrainingControl, TrainingStatusWriter


class TrainingCallback(BaseCallback):
    """训练进度回调"""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """每步调用"""
        # 获取当前回合信息
        if self.locals.get('dones') is not None:
            dones = self.locals['dones']
            rewards = self.locals['rewards']

            for i, done in enumerate(dones):
                if done:
                    # 记录回合奖励
                    self.episode_rewards.append(self.locals['infos'][i].get('episode_reward', 0))
                    self.episode_lengths.append(self.locals['infos'][i].get('episode_steps', 0))

        # 定期输出统计
        if self.n_calls % 1000 == 0 and self.verbose > 0:
            if self.episode_rewards:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                print(f"Step {self.num_timesteps}: "
                      f"Mean Reward (last 100): {mean_reward:.2f}, "
                      f"Mean Length: {mean_length:.1f}")

        return True


class CheckpointCallback(BaseCallback):
    """检查点保存回调"""

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        model_name: str = "ppo_sts2",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.model_name = model_name

    def _on_step(self) -> bool:
        """每步调用"""
        if self.n_calls % self.save_freq == 0:
            # 构建保存路径
            path = os.path.join(
                self.save_path,
                f"{self.model_name}_step{self.num_timesteps}"
            )

            # 保存模型
            self.model.save(path)

            if self.verbose > 0:
                print(f"Checkpoint saved: {path}")

        return True


class TrainingStatusCallback(BaseCallback):
    """Persist compact training progress snapshots for the dashboard."""

    def __init__(
        self,
        status_writer: TrainingStatusWriter,
        total_timesteps: int,
        num_envs: int,
        vec_env: str,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.status_writer = status_writer
        self.total_timesteps = total_timesteps
        self.num_envs = num_envs
        self.vec_env = vec_env
        self.start_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.control = TrainingControl(str(status_writer.root))
        self.current_status = "running"

    def _on_training_start(self) -> None:
        self.control.clear()
        self.current_status = "running"
        self.status_writer.write(
            {
                "status": "running",
                "current_timesteps": 0,
                "total_timesteps": self.total_timesteps,
                "progress_pct": 0.0,
                "num_envs": self.num_envs,
                "vec_env": self.vec_env,
                "fps": 0.0,
                "episodes_finished": 0,
            }
        )

    def _write_status(self, *, status: str) -> None:
        elapsed = max(0.001, time.time() - self.start_time)
        current_timesteps = int(self.num_timesteps)
        recent_rewards = self.episode_rewards[-100:]
        recent_lengths = self.episode_lengths[-100:]
        self.current_status = status
        self.status_writer.write(
            {
                "status": status,
                "current_timesteps": current_timesteps,
                "total_timesteps": self.total_timesteps,
                "progress_pct": round((current_timesteps / self.total_timesteps) * 100, 2) if self.total_timesteps else 0.0,
                "num_envs": self.num_envs,
                "vec_env": self.vec_env,
                "fps": round(current_timesteps / elapsed, 2),
                "episodes_finished": len(self.episode_rewards),
                "mean_reward_100": round(float(np.mean(recent_rewards)), 2) if recent_rewards else 0.0,
                "mean_length_100": round(float(np.mean(recent_lengths)), 2) if recent_lengths else 0.0,
            }
        )

    def _handle_control(self) -> bool:
        control = self.control.read()
        command = str(control.get("command") or "").lower()
        if command == "stop":
            self._write_status(status="stopping")
            return False
        return True

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None:
            infos = []
        dones = self.locals.get("dones")
        if dones is None:
            dones = []
        for index, done in enumerate(dones):
            if done and index < len(infos):
                self.episode_rewards.append(infos[index].get("episode_reward", 0.0))
                self.episode_lengths.append(infos[index].get("episode_steps", 0))

        if not self._handle_control():
            return False

        if self.n_calls % 4 == 0:
            self._write_status(status="running")
        return True

    def _on_training_end(self) -> None:
        final_status = "stopped" if self.current_status == "stopping" else "finished"
        self._write_status(status=final_status)
        self.control.clear()
