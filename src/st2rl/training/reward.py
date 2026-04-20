# -*- coding: utf-8 -*-
"""Reward shaping for HTTP CLI reinforcement learning."""

from dataclasses import dataclass

from st2rl.gameplay.types import GameStateView


@dataclass(slots=True)
class RewardConfig:
    """Reward weights grouped by gameplay outcome."""

    hp_delta_weight: float = 0.1
    damage_penalty_weight: float = 0.5
    gold_delta_weight: float = 0.01
    combat_victory_reward: float = 1.0
    card_reward_seen_reward: float = 0.3
    elite_reward: float = 1.5
    shop_high_gold_reward: float = 0.35
    shop_low_gold_penalty: float = -0.4
    terminal_victory_reward: float = 100.0
    terminal_defeat_penalty: float = -50.0
    invalid_action_penalty: float = -0.1
    stuck_penalty: float = -5.0
    progress_floor_reward: float = 0.5
    act_progress_reward: float = 25.0
    boss_combat_victory_reward: float = 30.0


class RewardTracker:
    """Computes shaped rewards from consecutive canonical states."""

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self._previous_state: GameStateView | None = None

    def reset(self, state: GameStateView) -> None:
        self._previous_state = state

    def on_invalid_action(self) -> float:
        return self.config.invalid_action_penalty

    def on_stuck(self) -> float:
        return self.config.stuck_penalty

    @staticmethod
    def _safe_int(value: object, default: int = 0) -> int:
        try:
            if value is None or value == "":
                return default
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

    @classmethod
    def _context(cls, state: GameStateView) -> tuple[int, int, str]:
        context = state.raw.get("context") or {}
        act = cls._safe_int(context.get("act") or state.raw.get("act"), 0)
        floor = cls._safe_int(context.get("floor") or state.raw.get("floor"), 0)
        room_type = str(context.get("room_type") or state.raw.get("room_type") or "")
        return act, floor, room_type

    @staticmethod
    def _is_act1_boss_context(act: int, floor: int, room_type: str) -> bool:
        return act == 1 and (floor >= 17 or "boss" in room_type.lower())

    def compute(self, state: GameStateView) -> float:
        previous = self._previous_state
        self._previous_state = state
        if previous is None:
            return 0.0

        reward = 0.0
        hp_delta = state.hp - previous.hp
        if hp_delta >= 0:
            reward += hp_delta * self.config.hp_delta_weight
        else:
            reward += hp_delta * self.config.damage_penalty_weight

        reward += (state.gold - previous.gold) * self.config.gold_delta_weight

        prev_act, prev_floor, prev_room_type = self._context(previous)
        curr_act, curr_floor, curr_room_type = self._context(state)
        if curr_act > prev_act:
            reward += (curr_act - prev_act) * self.config.act_progress_reward

        if curr_floor > prev_floor:
            reward += (curr_floor - prev_floor) * self.config.progress_floor_reward
            room_type = curr_room_type.lower()
            if "elite" in room_type:
                reward += self.config.elite_reward
            elif "shop" in room_type:
                if state.gold >= 250:
                    reward += self.config.shop_high_gold_reward
                elif state.gold < 100:
                    reward += self.config.shop_low_gold_penalty

        if previous.decision == "combat_play" and state.decision == "card_reward":
            reward += self.config.combat_victory_reward
            if self._is_act1_boss_context(prev_act, prev_floor, prev_room_type):
                reward += self.config.boss_combat_victory_reward

        if prev_act == 1 and curr_act >= 2:
            reward += self.config.boss_combat_victory_reward

        if state.decision == "card_reward" and previous.decision != "card_reward":
            reward += self.config.card_reward_seen_reward

        if state.game_over:
            reward += self.config.terminal_victory_reward if state.victory else self.config.terminal_defeat_penalty

        return reward
