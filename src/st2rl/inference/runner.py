# -*- coding: utf-8 -*-
"""Unified runner for deploying trained models"""

import time
from typing import Dict, Any, Generator, Optional, List
import numpy as np

from st2rl.environments.factory import EnvironmentFactory
from st2rl.models.ppo_model import UnifiedPPOModel


class UnifiedRunner:
    """统一的运行器

    用于在 headless 或 UI 模式下运行训练好的模型
    """

    def __init__(
        self,
        model_path: str,
        mode: str = "ui",
        env_config: Dict[str, Any] = None,
        deterministic: bool = True
    ):
        """
        初始化运行器

        Args:
            model_path: 模型路径（不含扩展名）
            mode: 运行模式 "headless" 或 "ui"
            env_config: 环境配置
            deterministic: 是否确定性运行
        """
        self.model_path = model_path
        self.mode = mode
        self.env_config = env_config or {}
        self.deterministic = deterministic

        self.env = None
        self.model = None

        # 统计
        self.episode_count = 0
        self.total_rewards = []
        self.episode_lengths = []

    def setup(self):
        """设置环境和模型"""
        # 创建环境
        print(f"Creating {self.mode} environment...")
        self.env = EnvironmentFactory.create(self.mode, **self.env_config)

        # 加载模型
        print(f"Loading model from {self.model_path}...")
        self.model = UnifiedPPOModel.load(self.model_path, env=self.env)

        print("Setup complete!")
        return self

    def run_episode(self, render: bool = False) -> Dict[str, Any]:
        """
        运行单局游戏

        Args:
            render: 是否渲染

        Returns:
            回合统计信息
        """
        if self.env is None or self.model is None:
            raise ValueError("Runner not set up. Call setup() first.")

        obs, info = self.env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # 预测动作
            action, _ = self.model.predict(obs, deterministic=self.deterministic)

            # 执行动作
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if render:
                self.env.render()
                time.sleep(0.1)

        # 记录统计
        self.episode_count += 1
        self.total_rewards.append(total_reward)
        self.episode_lengths.append(steps)

        result = {
            'episode': self.episode_count,
            'steps': steps,
            'total_reward': total_reward,
            'terminated': terminated,
            'truncated': truncated,
            'info': info
        }

        return result

    def run(
        self,
        num_episodes: int,
        render: bool = False,
        progress_interval: int = 1
    ) -> Generator[Dict[str, Any], None, None]:
        """
        运行多局游戏

        Args:
            num_episodes: 游戏局数
            render: 是否渲染
            progress_interval: 进度输出间隔

        Yields:
            每局游戏的结果
        """
        if self.env is None or self.model is None:
            self.setup()

        print(f"\nRunning {num_episodes} episodes in {self.mode} mode...")
        print("=" * 60)

        for episode in range(num_episodes):
            result = self.run_episode(render=render)

            # 输出进度
            if (episode + 1) % progress_interval == 0:
                print(f"Episode {episode + 1}/{num_episodes}: "
                      f"Reward={result['total_reward']:.2f}, "
                      f"Steps={result['steps']}, "
                      f"Terminated={result['terminated']}")

            yield result

        # 输出总结
        self._print_summary()

    def run_batch(
        self,
        num_episodes: int,
        render: bool = False
    ) -> List[Dict[str, Any]]:
        """
        批量运行游戏

        Args:
            num_episodes: 游戏局数
            render: 是否渲染

        Returns:
            所有结果列表
        """
        results = []
        for result in self.run(num_episodes, render):
            results.append(result)
        return results

    def _print_summary(self):
        """打印运行总结"""
        if not self.total_rewards:
            return

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total Episodes: {self.episode_count}")
        print(f"Mean Reward: {np.mean(self.total_rewards):.2f} "
              f"(±{np.std(self.total_rewards):.2f})")
        print(f"Mean Length: {np.mean(self.episode_lengths):.1f} "
              f"(±{np.std(self.episode_lengths):.1f})")
        print(f"Max Reward: {np.max(self.total_rewards):.2f}")
        print(f"Min Reward: {np.min(self.total_rewards):.2f}")
        print("=" * 60)

    def evaluate(
        self,
        num_episodes: int = 10
    ) -> Dict[str, float]:
        """
        评估模型性能

        Args:
            num_episodes: 评估局数

        Returns:
            评估指标
        """
        results = self.run_batch(num_episodes)

        rewards = [r['total_reward'] for r in results]
        lengths = [r['steps'] for r in results]
        victories = sum(1 for r in results if r['terminated'] and r['total_reward'] > 0)

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'victory_rate': victories / len(results),
            'num_episodes': len(results)
        }

    def close(self):
        """关闭运行器"""
        if self.env:
            self.env.close()
            self.env = None
