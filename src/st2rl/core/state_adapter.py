# -*- coding: utf-8 -*-
"""Unified state representation for headless and UI modes"""

from typing import Dict, Any, List, Optional
import numpy as np


class UnifiedState:
    """统一的游戏状态表示"""

    def __init__(self, raw_state: Dict[str, Any], source: str = "unknown"):
        """
        Args:
            raw_state: 原始状态（来自 headless 或 UI）
            source: 来源 ("headless" 或 "ui")
        """
        self.source = source
        self.raw = raw_state

        # 统一字段
        self.player_hp = 0
        self.player_max_hp = 0
        self.player_gold = 0
        self.player_energy = 0
        self.floor = 0
        self.act = 0
        self.screen = ""  # 统一屏幕类型
        self.hand = []
        self.enemies = []
        self.potions = []
        self.relics = []
        self.deck_size = 0

        # 解析原始状态
        self._parse()

    def _parse(self):
        """解析原始状态为统一格式"""
        if self.source == "headless":
            self._parse_headless()
        elif self.source == "ui":
            self._parse_ui()

    def _parse_headless(self):
        """解析 headless 状态"""
        player = self.raw.get("player", {})
        self.player_hp = player.get("hp", 0)
        self.player_max_hp = player.get("max_hp", 0)
        self.player_gold = player.get("gold", 0)
        self.player_energy = self.raw.get("energy", 0)
        self.floor = self.raw.get("floor", 0)
        self.act = self.raw.get("act", 0)
        self.screen = self._map_headless_screen(self.raw.get("decision", ""))
        self.hand = self.raw.get("hand", [])
        self.enemies = self.raw.get("enemies", [])
        self.potions = player.get("potions", [])
        self.relics = player.get("relics", [])
        self.deck_size = player.get("deck_size", 0)

    def _parse_ui(self):
        """解析 UI 状态"""
        run = self.raw.get("run", {})
        player = run.get("player", {})
        combat = self.raw.get("combat", {})

        self.player_hp = player.get("hp", 0)
        self.player_max_hp = player.get("max_hp", 0)
        self.player_gold = player.get("gold", 0)
        self.player_energy = combat.get("energy", 0) if combat else 0
        self.floor = run.get("floor", 0)
        self.act = run.get("act", 0)
        self.screen = self.raw.get("screen", "UNKNOWN")

        # 解析手牌
        hand_cards = combat.get("hand", []) if combat else []
        self.hand = [{"name": c.get("name", ""),
                      "cost": c.get("cost", 0),
                      "can_play": c.get("playable", False)}
                     for c in hand_cards]

        # 解析敌人
        enemies = combat.get("enemies", []) if combat else []
        self.enemies = [{"name": e.get("name", ""),
                         "hp": e.get("hp", 0),
                         "max_hp": e.get("max_hp", 0)}
                        for e in enemies]

        self.potions = player.get("potions", [])
        self.relics = player.get("relics", [])
        self.deck_size = len(run.get("deck", []))

    def _map_headless_screen(self, decision: str) -> str:
        """映射 headless decision 到统一 screen"""
        mapping = {
            "combat_play": "COMBAT",
            "map_select": "MAP",
            "event_choice": "EVENT",
            "rest_site": "REST",
            "card_reward": "CARD_REWARD",
            "shop": "SHOP",
            "game_over": "GAME_OVER",
        }
        return mapping.get(decision, decision.upper())

    def to_observation(self) -> np.ndarray:
        """转换为模型输入向量"""
        # 数值特征
        features = [
            self.player_hp / 100.0,
            self.player_max_hp / 100.0,
            self.player_gold / 500.0,
            self.player_energy / 10.0,
            self.floor / 50.0,
            self.act / 3.0,
            len(self.hand) / 10.0,
            len(self.enemies) / 3.0,
            self.deck_size / 50.0,
        ]

        # 手牌编码（简化版）
        for i in range(5):  # 最多 5 张手牌
            if i < len(self.hand):
                card = self.hand[i]
                features.append(card.get("cost", 0) / 5.0)
                features.append(1.0 if card.get("can_play") else 0.0)
            else:
                features.extend([0.0, 0.0])

        # 敌人编码（简化版）
        for i in range(3):  # 最多 3 个敌人
            if i < len(self.enemies):
                enemy = self.enemies[i]
                features.append(enemy.get("hp", 0) / 200.0)
                features.append(enemy.get("max_hp", 0) / 200.0)
            else:
                features.extend([0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def is_terminal(self) -> bool:
        """是否结束"""
        return self.screen == "GAME_OVER" or self.player_hp <= 0

    def get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            'source': self.source,
            'screen': self.screen,
            'floor': self.floor,
            'act': self.act,
            'hp': self.player_hp,
            'max_hp': self.player_max_hp,
            'gold': self.player_gold,
        }


class StateAdapter:
    """状态适配器工厂"""

    @staticmethod
    def adapt(state: Dict[str, Any], source: str) -> UnifiedState:
        return UnifiedState(state, source)
