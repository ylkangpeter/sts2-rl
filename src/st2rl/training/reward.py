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
    act1_floor6_reward: float = 3.0
    act1_floor10_reward: float = 4.0
    act1_floor13_reward: float = 5.0
    act1_floor16_reward: float = 7.0
    act1_boss_reach_reward: float = 10.0
    act1_boss_clear_reward: float = 40.0
    act1_boss_prep_hp_reward: float = 8.0
    act1_boss_prep_block_density_reward: float = 5.0
    act1_boss_prep_potion_reward: float = 2.5
    act1_early_damage_penalty: float = -1.2
    act1_late_preboss_low_hp_penalty: float = -8.0
    act1_elite_risk_penalty: float = -4.0
    act1_deadly_greed_penalty: float = -6.0


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

    @staticmethod
    def _deck_cards(state: GameStateView) -> list[dict]:
        return [card for card in (state.player.get("deck") or []) if isinstance(card, dict)]

    @staticmethod
    def _card_type(card: dict) -> str:
        return str(card.get("type") or "").strip().lower()

    @classmethod
    def _deck_block_density(cls, state: GameStateView) -> float:
        deck = cls._deck_cards(state)
        if not deck:
            return 0.0
        block_cards = 0
        for card in deck:
            text = str(card.get("description") or "").lower()
            stats = card.get("stats") or {}
            if cls._safe_int(stats.get("block"), 0) > 0 or "block" in text or "格挡" in text or "防御" in text:
                block_cards += 1
        return float(block_cards) / max(1, len(deck))

    @classmethod
    def _potion_count(cls, state: GameStateView) -> int:
        return sum(1 for potion in (state.player.get("potions") or []) if isinstance(potion, dict))

    @classmethod
    def _current_global_floor(cls, state: GameStateView) -> int:
        act, floor, _ = cls._context(state)
        if act <= 1:
            return floor
        if act == 2:
            return 18 + floor
        if act == 3:
            return 35 + floor
        return 51 + floor + max(0, act - 4) * 17

    @staticmethod
    def _is_act1_room(room_type: str) -> bool:
        lowered = room_type.lower()
        return "boss" in lowered or "elite" in lowered or "monster" in lowered

    def _act1_progress_reward(self, previous: GameStateView, state: GameStateView) -> float:
        reward = 0.0
        prev_act, prev_floor, _ = self._context(previous)
        curr_act, curr_floor, curr_room_type = self._context(state)
        if prev_act != 1 or curr_act != 1:
            return reward
        for milestone_floor, milestone_reward in (
            (6, self.config.act1_floor6_reward),
            (10, self.config.act1_floor10_reward),
            (13, self.config.act1_floor13_reward),
            (16, self.config.act1_floor16_reward),
        ):
            if prev_floor < milestone_floor <= curr_floor:
                reward += milestone_reward
        if not self._is_act1_boss_context(prev_act, prev_floor, "") and self._is_act1_boss_context(curr_act, curr_floor, curr_room_type):
            reward += self.config.act1_boss_reach_reward
            hp_ratio = state.hp / max(1, state.max_hp)
            reward += hp_ratio * self.config.act1_boss_prep_hp_reward
            reward += min(1.0, self._deck_block_density(state) / 0.3) * self.config.act1_boss_prep_block_density_reward
            reward += min(2, self._potion_count(state)) * self.config.act1_boss_potion_reward
        return reward

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

        reward += self._act1_progress_reward(previous, state)

        if previous.decision == "combat_play" and state.decision == "card_reward":
            reward += self.config.combat_victory_reward
            if self._is_act1_boss_context(prev_act, prev_floor, prev_room_type):
                reward += self.config.boss_combat_victory_reward
                reward += self.config.act1_boss_clear_reward

        if prev_act == 1 and curr_act >= 2:
            reward += self.config.boss_combat_victory_reward
            reward += self.config.act1_boss_clear_reward

        if state.decision == "card_reward" and previous.decision != "card_reward":
            reward += self.config.card_reward_seen_reward

        prev_hp_ratio = previous.hp / max(1, previous.max_hp)
        curr_hp_ratio = state.hp / max(1, state.max_hp)
        if prev_act == 1 and curr_act == 1:
            if prev_floor <= 8 and hp_delta < -8:
                reward += self.config.act1_early_damage_penalty
            if prev_floor >= 13 and curr_floor >= 13 and curr_hp_ratio < 0.45:
                reward += self.config.act1_late_preboss_low_hp_penalty
            if "elite" in prev_room_type.lower() and prev_floor <= 12 and prev_hp_ratio < 0.7 and hp_delta < -10:
                reward += self.config.act1_elite_risk_penalty
            if previous.decision == "combat_play" and curr_hp_ratio < 0.3 and hp_delta < -12:
                reward += self.config.act1_deadly_greed_penalty

        if state.game_over:
            reward += self.config.terminal_victory_reward if state.victory else self.config.terminal_defeat_penalty

        return reward
