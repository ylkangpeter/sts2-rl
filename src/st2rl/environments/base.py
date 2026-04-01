# -*- coding: utf-8 -*-
"""Base environment for unified STS2 environment"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional
from abc import abstractmethod

from st2rl.core.state_adapter import StateAdapter, UnifiedState
from st2rl.core.spaces import UnifiedActionSpace, UnifiedObservationSpace


class UnifiedSTS2Env(gym.Env):
    """统一的 STS2 环境"""

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        mode: str,  # "headless" 或 "ui"
        character: str = "Ironclad",
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__()

        self.mode = mode
        self.character = character
        self.max_steps = max_steps
        self.current_step = 0

        # 统一的空间
        self.observation_space = UnifiedObservationSpace().create()
        self.action_space = spaces.Box(0, 20, shape=(3,), dtype=np.float32)

        # 动作解码器
        self.action_decoder = UnifiedActionSpace()

        # 当前状态
        self.current_state: Optional[UnifiedState] = None

        # 统计信息
        self.episode_reward = 0.0
        self.episode_steps = 0

        # 初始化客户端
        self._init_client(**kwargs)

    @abstractmethod
    def _init_client(self, **kwargs):
        """初始化客户端（由子类实现）"""
        pass

    @abstractmethod
    def _get_raw_state(self) -> Dict[str, Any]:
        """获取原始状态（由子类实现）"""
        pass

    @abstractmethod
    def _execute_action(self, action_type: str, params: Dict[str, Any]):
        """执行动作（由子类实现）"""
        pass

    @abstractmethod
    def _do_reset(self):
        """执行重置（由子类实现）"""
        pass

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_steps = 0

        # 执行具体重置逻辑
        self._do_reset()

        # 获取初始状态
        raw_state = self._get_raw_state()
        self.current_state = StateAdapter.adapt(raw_state, self.mode)

        return self.current_state.to_observation(), self.current_state.get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        self.current_step += 1
        self.episode_steps += 1

        # 解码动作
        action_type, params = self.action_decoder.decode(action)

        # 执行动作
        self._execute_action(action_type, params)

        # 获取新状态
        raw_state = self._get_raw_state()
        self.current_state = StateAdapter.adapt(raw_state, self.mode)

        # 计算奖励
        reward = self._calculate_reward()
        self.episode_reward += reward

        # 检查终止
        terminated = self.current_state.is_terminal()
        truncated = self.current_step >= self.max_steps

        info = self.current_state.get_info()
        info['episode_reward'] = self.episode_reward
        info['episode_steps'] = self.episode_steps

        return self.current_state.to_observation(), reward, terminated, truncated, info

    def _calculate_reward(self) -> float:
        """计算奖励（可在子类中覆盖）"""
        if self.current_state is None:
            return 0.0

        reward = 0.0

        # 基础奖励
        reward += self.current_state.floor * 0.1

        # HP 奖励
        hp_ratio = self.current_state.player_hp / max(self.current_state.player_max_hp, 1)
        reward += hp_ratio * 0.01

        # 结束奖励
        if self.current_state.is_terminal():
            if self.current_state.player_hp > 0:
                reward += 10.0  # 胜利
            else:
                reward -= 5.0   # 失败

        return reward

    def render(self, mode='human'):
        """渲染环境"""
        if self.current_state:
            print(f"Step {self.current_step}: "
                  f"Screen={self.current_state.screen}, "
                  f"HP={self.current_state.player_hp}/{self.current_state.player_max_hp}, "
                  f"Floor={self.current_state.floor}")

    def close(self):
        """关闭环境"""
        pass
