# -*- coding: utf-8 -*-
"""Unified action and observation spaces"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, List, Tuple


class UnifiedActionSpace:
    """统一的动作空间"""

    # 动作类型定义（与两种模式兼容）
    ACTION_TYPES = [
        "play_card",           # 0: 打出卡牌
        "end_turn",            # 1: 结束回合
        "use_potion",          # 2: 使用药水
        "select_map_node",     # 3: 选择地图节点
        "choose_option",       # 4: 选择事件选项
        "select_card_reward",  # 5: 选择卡牌奖励
        "skip_reward",         # 6: 跳过奖励
        "select_bundle",       # 7: 选择卡牌包
        "shop_purchase",       # 8: 商店购买
        "leave_room",          # 9: 离开房间
        "proceed",             # 10: 继续
    ]

    def __init__(self):
        # 离散动作空间
        self.action_type_space = spaces.Discrete(len(self.ACTION_TYPES))
        # 参数空间（用于 play_card, use_potion 等）
        self.param_space = spaces.Box(0, 20, shape=(3,), dtype=np.float32)

        self.space = spaces.Dict({
            'action_type': self.action_type_space,
            'params': self.param_space
        })

    def sample(self) -> np.ndarray:
        """采样动作"""
        action_type = self.action_type_space.sample()
        params = self.param_space.sample()
        return np.array([action_type, params[0], params[1]], dtype=np.float32)

    def decode(self, action: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """解码动作向量"""
        action_type_idx = int(action[0]) % len(self.ACTION_TYPES)
        action_type = self.ACTION_TYPES[action_type_idx]

        params = {
            'index': int(action[1]),
            'target': int(action[2]),
        }

        return action_type, params

    def encode(self, action_type: str, params: Dict[str, Any]) -> np.ndarray:
        """编码动作"""
        action_idx = self.ACTION_TYPES.index(action_type) if action_type in self.ACTION_TYPES else 0
        return np.array([
            action_idx,
            params.get('index', 0),
            params.get('target', 0)
        ], dtype=np.float32)

    def get_action_name(self, action_idx: int) -> str:
        """获取动作名称"""
        if 0 <= action_idx < len(self.ACTION_TYPES):
            return self.ACTION_TYPES[action_idx]
        return "unknown"


class UnifiedObservationSpace:
    """统一的观察空间"""

    def __init__(self):
        # 数值特征维度
        self.feature_dim = 30  # 根据 StateAdapter.to_observation 调整

        self.space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.feature_dim,),
            dtype=np.float32
        )

    def create(self) -> spaces.Box:
        return self.space

    def get_dimension(self) -> int:
        return self.feature_dim
