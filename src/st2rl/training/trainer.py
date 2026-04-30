# -*- coding: utf-8 -*-
"""Unified trainer for headless and UI environments"""

import multiprocessing as mp
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from st2rl.environments.factory import EnvironmentFactory
from st2rl.training.callbacks import CheckpointCallback, TrainingCallback, TrainingStatusCallback
from st2rl.training.telemetry import TrainingStatusWriter
from st2rl.training.threaded_vec_env import ThreadedVecEnv


class UnifiedTrainer:
    """统一的训练器

    支持 headless 和 UI 两种训练模式，模型互通
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器

        Args:
            config: 配置字典
                - mode: "headless" 或 "ui"
                - environment: 环境配置
                - model: 模型参数
                - training: 训练参数
                - paths: 路径配置
        """
        self.config = config
        self.mode = config.get('mode', 'headless')
        self.env = None
        self.model = None

        # 路径配置
        paths_config = config.get('paths', {})
        self.base_save_dir = paths_config.get('save_dir', './models/unified')
        self.model_name = paths_config.get('model_name', 'ppo_sts2')
        self.experiment_name = paths_config.get('experiment_name', self.model_name)
        self.run_id = paths_config.get('run_id') or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(self.base_save_dir, self.experiment_name, self.run_id)
        os.makedirs(self.save_dir, exist_ok=True)
        self.dashboard_dir = os.path.join("logs", "dashboard_runs", self.experiment_name, self.run_id, "dashboard")
        os.makedirs(self.dashboard_dir, exist_ok=True)
        self.resume_model_path: str | None = None
        self.resume_load_status = "fresh"
        self.resume_failure_reason: str | None = None

    def _build_env_factory(self, mode: str, env_config: Dict[str, Any], rank: int):
        def _factory():
            config = dict(env_config)
            config.setdefault("seed_offset", rank)
            return EnvironmentFactory.create(mode, **config)

        return _factory

    def _configure_windows_subproc_python(self) -> None:
        if os.name != "nt":
            return
        pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        if os.path.isfile(pythonw):
            mp.set_executable(pythonw)

    def create_environment(self, mode: str = None):
        """
        创建环境

        Args:
            mode: 可选，覆盖配置中的 mode

        Returns:
            环境实例
        """
        mode = mode or self.mode
        env_config = dict(self.config.get('environment', {}))
        training_config = self.config.get('training', {})
        num_envs = max(1, int(training_config.get('num_envs', 1)))
        vec_env = str(training_config.get('vec_env', 'threaded')).lower()
        env_config.setdefault("telemetry_dir", self.dashboard_dir)
        env_config.setdefault("telemetry_model_run_dir", self.save_dir)

        print(f"Creating {mode} environment...")
        if num_envs == 1:
            self.env = EnvironmentFactory.create(mode, **env_config)
        else:
            env_fns = [self._build_env_factory(mode, env_config, rank) for rank in range(num_envs)]
            if vec_env == 'subproc':
                self._configure_windows_subproc_python()
                self.env = SubprocVecEnv(env_fns)
            elif vec_env == 'threaded':
                self.env = ThreadedVecEnv(env_fns)
            else:
                self.env = DummyVecEnv(env_fns)
            print(f"Using vectorized environment: type={vec_env} num_envs={num_envs}")

        return self.env

    def create_model(self, model_path: str = None):
        """
        创建或加载模型

        Args:
            model_path: 可选，加载已有模型继续训练

        Returns:
            模型实例
        """
        if self.env is None:
            raise ValueError("Environment not created. Call create_environment() first.")

        model_config = self.config.get('model', {})
        from st2rl.models.ppo_model import UnifiedPPOModel

        if model_path and os.path.exists(model_path + ".zip"):
            # 加载已有模型继续训练
            print(f"Loading model from {model_path} for continued training...")
            self.resume_model_path = model_path
            try:
                self.model = UnifiedPPOModel.load(model_path, env=self.env)
                self.resume_load_status = "resumed"
                self.resume_failure_reason = None
            except ValueError as exc:
                message = str(exc)
                if "Observation spaces do not match" in message or "Action spaces do not match" in message:
                    print("Model/environment space mismatch detected. Falling back to a fresh PPO model.")
                    print(f"Resume load failed: {message}")
                    self.model = UnifiedPPOModel(self.env, **model_config)
                    self.resume_load_status = "fallback_fresh"
                    self.resume_failure_reason = message
                else:
                    raise
        else:
            # 创建新模型
            print("Creating new PPO model...")
            self.model = UnifiedPPOModel(self.env, **model_config)
            self.resume_model_path = None
            self.resume_load_status = "fresh"
            self.resume_failure_reason = None

        return self.model

    def train(
        self,
        total_timesteps: int = None,
        save_path: str = None,
        callbacks: Optional[list] = None
    ):
        """
        训练模型

        Args:
            total_timesteps: 总训练步数（覆盖配置）
            save_path: 保存路径（覆盖配置）
            callbacks: 额外回调列表
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        # 获取训练参数
        training_config = self.config.get('training', {})
        timesteps = total_timesteps or training_config.get('total_timesteps', 100000)
        num_envs = max(1, int(training_config.get('num_envs', 1)))
        vec_env = str(training_config.get('vec_env', 'threaded')).lower()

        # 构建保存路径
        if save_path is None:
            save_path = os.path.join(self.save_dir, self.model_name)

        # 创建回调
        callback_list = []

        # 检查点回调
        save_freq = training_config.get('save_freq', 10000)
        if save_freq > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=self.save_dir,
                model_name=self.model_name
            )
            callback_list.append(checkpoint_callback)

        # 训练进度回调
        verbose = training_config.get('verbose', 1)
        if verbose > 0:
            training_callback = TrainingCallback(verbose=verbose)
            callback_list.append(training_callback)

        status_writer = TrainingStatusWriter(
            self.dashboard_dir,
            {
                "experiment_name": self.experiment_name,
                "run_id": self.run_id,
                "mode": self.mode,
                "save_dir": self.save_dir,
                "save_path": save_path,
                "model_name": self.model_name,
                "resume_model_path": self.resume_model_path,
                "resume_load_status": self.resume_load_status,
                "resume_failure_reason": self.resume_failure_reason,
            },
            model_run_dir=self.save_dir,
        )
        callback_list.append(
            TrainingStatusCallback(
                status_writer=status_writer,
                total_timesteps=timesteps,
                num_envs=num_envs,
                vec_env=vec_env,
                verbose=0,
            )
        )

        # 添加用户自定义回调
        if callbacks:
            callback_list.extend(callbacks)

        # 合并回调
        if callback_list:
            callback = CallbackList(callback_list)
        else:
            callback = None

        # 开始训练
        training_config = self.config.get('training', {})
        print(f"Starting training for {timesteps} timesteps...")
        print(f"Mode: {self.mode}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Run ID: {self.run_id}")
        print(f"Parallel envs: {num_envs} ({vec_env})")
        print(f"Save path: {save_path}")

        self.model.train(timesteps, callback=callback)

        # 保存最终模型
        self.model.save(save_path)
        print(f"\nTraining completed! Model saved to {save_path}")

    def continue_training(
        self,
        model_path: str,
        additional_timesteps: int,
        save_path: str = None
    ):
        """
        继续训练已有模型

        Args:
            model_path: 已有模型路径
            additional_timesteps: 额外训练步数
            save_path: 新的保存路径
        """
        # 加载模型
        self.create_model(model_path)

        # 继续训练
        if save_path is None:
            save_path = model_path + "_continued"

        self.model.continue_training(additional_timesteps)
        self.model.save(save_path)

        print(f"Continued training completed! Model saved to {save_path}")

    def save(self, path: str = None):
        """保存模型"""
        if self.model is None:
            raise ValueError("Model not created")

        if path is None:
            path = os.path.join(self.save_dir, self.model_name)

        self.model.save(path)

    def close(self) -> None:
        """Release vectorized environments and their remote game sessions."""
        env = self.env
        self.env = None
        if env is None:
            return
        try:
            env.close()
        except Exception:
            pass

    def load(self, path: str):
        """加载模型"""
        if self.env is None:
            raise ValueError("Environment not created")

        self.model = UnifiedPPOModel.load(path, env=self.env)
        return self.model

    def switch_mode(self, new_mode: str):
        """
        切换训练模式

        用于从 headless 切换到 UI 进行微调

        Args:
            new_mode: 新模式 "headless" 或 "ui"
        """
        if new_mode == self.mode:
            return

        print(f"Switching mode from {self.mode} to {new_mode}")
        self.mode = new_mode

        # 保存当前模型
        if self.model is not None:
            temp_path = os.path.join(self.save_dir, "temp_switch")
            self.model.save(temp_path)

            # 创建新环境
            self.create_environment(new_mode)

            # 加载模型到新环境
            self.model = UnifiedPPOModel.load(temp_path, env=self.env)

            # 删除临时文件
            import glob
            for f in glob.glob(temp_path + "*"):
                os.remove(f)
        else:
            # 只创建新环境
            self.create_environment(new_mode)
