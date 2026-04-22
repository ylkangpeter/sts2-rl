# -*- coding: utf-8 -*-
"""Decision policy split by gameplay stage."""

import random
import re
from typing import Any

from st2rl.gameplay.config import FlowPolicyConfig
from st2rl.gameplay.heuristics import (
    best_shop_card,
    best_shop_potion,
    best_shop_relic,
    choose_card_reward,
    choose_event_option,
    choose_map_node_choice,
    choose_purge_target,
    choose_purge_targets,
    choose_upgrade_targets,
    is_shop_context,
    shop_purge_cost,
    should_prioritize_shop_purge,
)
from st2rl.gameplay.types import FlowAction, GameStateView, card_needs_enemy_target


class SimpleFlowPolicy:
    """Simple baseline flow policy split by game stage."""

    def __init__(self, config: FlowPolicyConfig | None = None):
        self.config = config or FlowPolicyConfig()

    def choose_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        if state.decision == "combat_play":
            return self._pick_combat_action(state, rng)
        if state.decision in ("map_node", "map_select"):
            return self._pick_map_action(state, rng)
        if state.decision == "card_reward":
            return self._pick_card_reward_action(state, rng)
        if state.decision == "shop":
            return self._pick_shop_action(state, rng)
        if state.decision in ("event", "event_choice"):
            return self._pick_event_action(state, rng)
        if state.decision == "rest_site":
            return self._pick_rest_action(state, rng)
        if state.decision == "card_select":
            return self._pick_card_select_action(state, rng)
        if state.decision == "bundle_select":
            return self._pick_bundle_action(state, rng)
        return FlowAction("proceed")

    def _pick_map_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        choices = list(state.choices)
        if not choices:
            for node in state.map_nodes:
                if node.get("current"):
                    choices = list(node.get("children") or [])
                    break
        if not choices:
            return FlowAction("proceed")
        pick = choose_map_node_choice(choices, state) or choices[0]
        return FlowAction("select_map_node", {"col": pick["col"], "row": pick["row"]})

    def _pick_combat_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        enemies = state.living_enemies()
        if not enemies:
            return FlowAction("proceed")

        potion_action = self._pick_combat_potion_action(state)
        if potion_action is not None:
            return potion_action

        playable = state.playable_cards()
        if not playable:
            return FlowAction("end_turn", {"_policy_allow_end_turn": True})

        incoming_damage = self._enemy_intended_damage(state)
        block_gap = max(0, incoming_damage - state.block)
        act, floor, room_type, is_boss_room = self._combat_context(state)
        is_boss_floor = floor >= 16 or is_boss_room
        is_act2_boss = act >= 2 and is_boss_floor
        is_ceremonial_beast = self._is_ceremonial_beast_boss(state)
        is_kin = self._is_kin_boss(state)
        is_targeted_boss = is_boss_floor or is_act2_boss or is_ceremonial_beast or is_kin
        if is_targeted_boss and len(playable) > 1:
            safer_playable = [card for card in playable if not self._should_avoid_hp_loss_energy_card(state, card, block_gap)]
            if safer_playable:
                playable = safer_playable

        if is_targeted_boss and (is_ceremonial_beast or is_kin):
            force_safe = [
                card
                for card in playable
                if not self._should_force_skip_hp_loss_energy_card(state, card, incoming_damage, block_gap)
            ]
            if force_safe:
                playable = force_safe
            elif any(self._is_hp_loss_energy_card_id(card) for card in playable):
                return FlowAction("end_turn", {"_policy_allow_end_turn": True})
            strict_safe = [card for card in playable if not self._should_avoid_hp_loss_energy_card(state, card, block_gap)]
            if strict_safe:
                playable = strict_safe
            elif all(self._card_hp_loss(card) > 0 and self._card_energy_gain(card) > 0 for card in playable):
                return FlowAction("end_turn", {"_policy_allow_end_turn": True})

        if is_targeted_boss and (is_ceremonial_beast or is_kin) and incoming_damage <= 0 and len(playable) > 1:
            aggressive_playable = [
                card
                for card in playable
                if self._card_damage(card) > 0
                or self._text(card.get("type")) == "power"
                or self._card_draw(card) > 0
                or self._card_energy_gain(card) > 0
            ]
            if aggressive_playable:
                playable = aggressive_playable

        if is_targeted_boss and block_gap >= max(10, int(state.hp * 0.45)):
            block_playable = [card for card in playable if self._card_block(card) > 0]
            if block_playable:
                block_scores = [
                    (
                        self._combat_card_score(state, card) + min(self._card_block(card), block_gap) * 2.5,
                        card,
                    )
                    for card in block_playable
                ]
                _block_score, block_card = max(block_scores, key=lambda item: item[0])
                args = {"card_index": block_card["index"]}
                if card_needs_enemy_target(block_card):
                    args["target_index"] = self._pick_combat_target(state, block_card, enemies).get("index", enemies[0]["index"])
                return FlowAction("play_card", args)

        if is_targeted_boss and (is_ceremonial_beast or is_kin):
            urgent_gap = max(6, int(state.hp * 0.28))
            if block_gap >= urgent_gap:
                block_playable = [card for card in playable if self._card_block(card) > 0]
                if block_playable:
                    block_scores = [
                        (
                            self._combat_card_score(state, card) + min(self._card_block(card), block_gap) * 3.2,
                            card,
                        )
                        for card in block_playable
                    ]
                    _block_score, block_card = max(block_scores, key=lambda item: item[0])
                    args = {"card_index": block_card["index"]}
                    if card_needs_enemy_target(block_card):
                        args["target_index"] = self._pick_combat_target(state, block_card, enemies).get("index", enemies[0]["index"])
                    return FlowAction("play_card", args)

        if is_act2_boss and block_gap >= max(8, int(state.hp * 0.33)):
            block_playable = [card for card in playable if self._card_block(card) > 0]
            if block_playable:
                block_scores = [
                    (
                        self._combat_card_score(state, card) + min(self._card_block(card), block_gap) * 3.3,
                        card,
                    )
                    for card in block_playable
                ]
                _block_score, block_card = max(block_scores, key=lambda item: item[0])
                args = {"card_index": block_card["index"]}
                if card_needs_enemy_target(block_card):
                    args["target_index"] = self._pick_combat_target(state, block_card, enemies).get("index", enemies[0]["index"])
                return FlowAction("play_card", args)

        card_scores = [(self._combat_card_score(state, item), item) for item in playable]
        best_score, card = max(card_scores, key=lambda item: item[0])
        if best_score < -5.0 and block_gap <= 0:
            if is_targeted_boss and is_ceremonial_beast:
                damage_playable = [item for item in playable if self._card_damage(item) > 0]
                if damage_playable:
                    card = max(damage_playable, key=lambda item: self._combat_card_score(state, item) + self._card_damage(item) * 3.5)
                else:
                    return FlowAction("end_turn", {"_policy_allow_end_turn": True})
            else:
                return FlowAction("end_turn", {"_policy_allow_end_turn": True})
        args = {"card_index": card["index"]}
        if card_needs_enemy_target(card):
            args["target_index"] = self._pick_combat_target(state, card, enemies).get("index", enemies[0]["index"])
        return FlowAction("play_card", args)

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            if value is None or value == "":
                return default
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default

    @staticmethod
    def _text(value: Any) -> str:
        return str(value or "").strip().lower()

    def _card_damage(self, card: dict[str, Any]) -> int:
        stats = card.get("stats") or {}
        return self._safe_int((stats or {}).get("damage"), 0)

    def _card_block(self, card: dict[str, Any]) -> int:
        stats = card.get("stats") or {}
        return self._safe_int((stats or {}).get("block"), 0)

    def _card_hp_loss(self, card: dict[str, Any]) -> int:
        stats = card.get("stats") or {}
        if not isinstance(stats, dict):
            return 0
        return max(
            self._safe_int(stats.get("hploss"), 0),
            self._safe_int(stats.get("hp_loss"), 0),
            self._safe_int(stats.get("HpLoss"), 0),
        )

    def _card_stat(self, card: dict[str, Any], *names: str) -> int:
        stats = card.get("stats") or {}
        if not isinstance(stats, dict):
            return 0
        lowered = {str(key).lower(): value for key, value in stats.items()}
        return max((self._safe_int(lowered.get(name.lower()), 0) for name in names), default=0)

    def _card_draw(self, card: dict[str, Any]) -> int:
        card_id = str(card.get("id") or "").strip().upper()
        if card_id in {"CARD.POMMEL_STRIKE", "CARD.SHRUG_IT_OFF", "CARD.BURNING_PACT", "CARD.BATTLE_TRANCE"}:
            return self._card_stat(card, "cards", "draw")
        description = self._text(card.get("description"))
        if "draw" in description or "抽" in description:
            return self._card_stat(card, "cards", "draw")
        return self._card_stat(card, "draw")

    def _card_energy_gain(self, card: dict[str, Any]) -> int:
        return self._card_stat(card, "energy")

    def _is_random_exhaust_risk(self, card: dict[str, Any]) -> bool:
        card_id = str(card.get("id") or "").strip().upper()
        if card_id == "CARD.TRUE_GRIT" and not bool(card.get("upgraded")):
            return True
        description = self._text(card.get("description"))
        return "random" in description and ("exhaust" in description or "消耗" in description)

    def _has_valuable_exhaust_target_in_hand(self, state: GameStateView, selected_card: dict[str, Any]) -> bool:
        selected_index = selected_card.get("index")
        for card in state.hand:
            if card.get("index") == selected_index:
                continue
            if self._card_damage(card) > 0 or self._card_draw(card) > 0 or self._card_energy_gain(card) > 0:
                return True
        return False

    def _playable_followup_pressure(self, state: GameStateView, card: dict[str, Any]) -> tuple[int, int]:
        selected_index = card.get("index")
        playable_after = [
            item
            for item in state.playable_cards()
            if item.get("index") != selected_index and self._text(item.get("type")) not in {"status", "curse"}
        ]
        spendable_cost = sum(max(0, self._safe_int(item.get("cost"), 0)) for item in playable_after)
        return len(playable_after), spendable_cost

    def _playable_followup_quality(self, state: GameStateView, card: dict[str, Any]) -> float:
        selected_index = card.get("index")
        quality = 0.0
        for item in state.playable_cards():
            if item.get("index") == selected_index:
                continue
            card_name = self._text(item.get("name") or item.get("card_id") or item.get("id"))
            damage = self._card_damage(item)
            block = self._card_block(item)
            draw = self._card_draw(item)
            energy = self._card_energy_gain(item)
            if damage > 0:
                quality += min(damage, 20) / 6.0
                if any(token in card_name for token in ("strike", "打击")):
                    quality -= 1.0
            if block > 0:
                quality += min(block, 16) / 8.0
                if any(token in card_name for token in ("defend", "防御")):
                    quality -= 0.8
            quality += draw * 2.0 + energy * 2.0
            if "bash" in card_name or "痛击" in card_name:
                quality += 1.5
        return quality

    def _drawn_card_play_value(
        self,
        state: GameStateView,
        card: dict[str, Any],
        remaining_energy: int,
        *,
        draw_locked: bool = False,
    ) -> float:
        card_type = self._text(card.get("type"))
        card_name = self._text(card.get("name") or card.get("card_id") or card.get("id"))
        cost = max(0, self._safe_int(card.get("cost"), 0))
        damage = self._card_damage(card)
        block = self._card_block(card)
        draw = 0 if draw_locked else self._card_draw(card)
        energy = self._card_energy_gain(card)
        if card_type in {"status", "curse"}:
            return -4.0
        if cost > remaining_energy and energy <= 0:
            return max(0.0, draw * 1.5 - cost)

        value = 0.0
        if damage > 0:
            lowest_enemy_ehp = min((self._enemy_effective_hp(enemy) for enemy in state.living_enemies()), default=999)
            value += min(damage, 24) / 3.5
            if damage >= lowest_enemy_ehp:
                value += 5.0
        if block > 0:
            incoming_damage = self._enemy_intended_damage(state)
            block_gap = max(0, incoming_damage - state.block)
            value += min(block, max(0, block_gap)) / 3.0
            if block_gap <= 0:
                value += min(block, 12) / 12.0
        value += draw * 4.0 + energy * 5.0
        if any(token in card_name for token in ("strike", "打击")) and draw <= 0 and energy <= 0:
            value -= 2.5
        if any(token in card_name for token in ("defend", "防御")) and draw <= 0 and energy <= 0:
            value -= 2.0
        if any(token in card_name for token in ("pommel", "剑柄", "headbutt", "shrug", "耸肩", "burning pact", "燃烧契约")):
            value += 4.0
        if draw_locked and ("battle_trance" in card_name or "战斗专注" in card_name):
            value -= 5.0
        if "bloodletting" in card_name or "放血" in card_name:
            value += 3.0 if state.hp / max(1, state.max_hp) >= 0.45 else -2.0
        return value

    def _guaranteed_draw_followup_bonus(self, state: GameStateView, card: dict[str, Any]) -> float:
        if self._text(card.get("type")) in {"status", "curse"}:
            return 0.0
        draw_amount = self._card_draw(card)
        if draw_amount <= 0:
            return 0.0

        draw_pile = [item for item in state.draw_pile if isinstance(item, dict)]
        draw_pile_count = self._safe_int(state.raw.get("draw_pile_count"), len(draw_pile))
        if draw_pile_count <= 0 or len(draw_pile) < draw_pile_count or draw_amount < draw_pile_count:
            return 0.0

        cost = max(0, self._safe_int(card.get("cost"), 0))
        remaining_energy = max(0, state.energy - cost + self._card_energy_gain(card))
        selected_index = card.get("index")
        current_followup_cost = sum(
            max(0, self._safe_int(item.get("cost"), 0))
            for item in state.playable_cards()
            if item.get("index") != selected_index and self._text(item.get("type")) not in {"status", "curse"}
        )
        spendable_after_current_hand = max(0, remaining_energy - current_followup_cost)
        if spendable_after_current_hand <= 0 and remaining_energy <= 0:
            return 0.0

        guaranteed = draw_pile[:draw_pile_count]
        selected_name = self._text(card.get("name") or card.get("card_id") or card.get("id"))
        draw_locked = "battle_trance" in selected_name or "战斗专注" in selected_name
        values = [
            self._drawn_card_play_value(
                state,
                item,
                max(0, spendable_after_current_hand),
                draw_locked=draw_locked,
            )
            for item in guaranteed
        ]
        positive_values = [value for value in values if value > 0]
        bad_values = [value for value in values if value < 0]
        if not positive_values:
            return max(-6.0, sum(bad_values) * 0.4)

        bonus = sum(positive_values) * 0.55 + min(len(positive_values), 3) * 1.0
        if draw_pile_count <= 2:
            bonus += 1.5
        return max(-4.0, min(10.0, bonus))

    def _enemy_attack_pressure(self, state: GameStateView) -> int:
        pressure = 0
        enemies = state.living_enemies()
        for enemy in enemies:
            if enemy.get("intends_attack"):
                pressure += 1
                continue
            intent_text = self._text(enemy.get("intent"))
            if any(token in intent_text for token in ("attack", "deathblow", "\u653b\u51fb")):
                pressure += 1
                continue
            for intent in enemy.get("intents") or []:
                if self._intent_is_attack(intent):
                    pressure += 1
                    break
        if pressure <= 0 and self._unknown_intent_damage_estimate(state, enemies) > 0:
            return max(1, len(enemies))
        return pressure

    def _enemy_has_known_intent(self, enemy: dict[str, Any]) -> bool:
        if enemy.get("intends_attack") is not None:
            return True
        if enemy.get("intents"):
            return True
        intent_text = self._text(enemy.get("intent"))
        return bool(intent_text and intent_text not in {"none", "null", "unknown"})

    def _intent_is_attack(self, intent: Any) -> bool:
        intent_type = self._text((intent or {}).get("type") if isinstance(intent, dict) else intent)
        return any(token in intent_type for token in ("attack", "deathblow"))

    def _intent_damage(self, intent: dict[str, Any]) -> int:
        hits = max(1, self._safe_int(intent.get("hits"), 1))
        damage = max(0, self._safe_int(intent.get("damage"), 0))
        return min(damage, 60) * hits

    def _enemy_threat(self, enemy: dict[str, Any]) -> int:
        threat = 0
        for intent in enemy.get("intents") or []:
            if isinstance(intent, dict) and self._intent_is_attack(intent):
                threat += self._intent_damage(intent)
        if enemy.get("intends_attack") and threat <= 0:
            threat = 1
        return threat

    def _enemy_intended_damage(self, state: GameStateView) -> int:
        total = 0
        enemies = state.living_enemies()
        for enemy in enemies:
            for intent in enemy.get("intents") or []:
                if not isinstance(intent, dict) or not self._intent_is_attack(intent):
                    continue
                total += self._intent_damage(intent)
        if total > 0:
            return total
        return self._unknown_intent_damage_estimate(state, enemies)

    def _unknown_intent_damage_estimate(self, state: GameStateView, enemies: list[dict[str, Any]]) -> int:
        if not enemies or any(self._enemy_has_known_intent(enemy) for enemy in enemies):
            return 0
        act, floor, room_type, is_boss_room = self._combat_context(state)
        if act <= 1:
            return 0
        if is_boss_room or floor >= 16:
            base = 52 if act == 2 else 58
            return base + max(0, len(enemies) - 1) * 8
        if "elite" in room_type:
            base = 34 if act == 2 else 38
            return base + max(0, len(enemies) - 1) * 5
        if act >= 2:
            return 20 + max(0, len(enemies) - 1) * 4
        return 0

    def _combat_context(self, state: GameStateView) -> tuple[int, int, str, bool]:
        context = state.raw.get("context") or {}
        act = self._safe_int(context.get("act") or state.raw.get("act"), 0)
        floor = self._safe_int(context.get("floor") or state.raw.get("floor"), 0)
        room_type = self._text(context.get("room_type"))
        is_boss_room = room_type == "boss" or "boss" in room_type
        return act, floor, room_type, is_boss_room

    def _should_avoid_hp_loss_energy_card(self, state: GameStateView, card: dict[str, Any], block_gap: int) -> bool:
        hp_loss = self._card_hp_loss(card)
        if hp_loss <= 0:
            return False
        energy_gain = self._card_energy_gain(card)
        if energy_gain <= 0:
            return False
        _, floor, _, _ = self._combat_context(state)
        is_ceremonial_beast = self._is_ceremonial_beast_boss(state)
        is_kin = self._is_kin_boss(state)
        if floor < 16 and not (is_ceremonial_beast or is_kin):
            return False
        hp_ratio = state.hp / max(1, state.max_hp)
        followup_count, spendable_cost = self._playable_followup_pressure(state, card)
        round_no = self._safe_int(state.round, 0)
        if is_ceremonial_beast or is_kin:
            if block_gap > 0:
                return True
            if state.hp <= max(28, hp_loss + 16):
                return True
            if hp_ratio < 0.76:
                return True
            if round_no >= 7:
                return True
            if followup_count < 3 or spendable_cost <= state.energy:
                return True
        if hp_ratio < 0.65:
            return True
        if state.hp <= max(24, hp_loss + 12):
            return True
        if block_gap > 0:
            return True
        return False

    def _is_hp_loss_energy_card_id(self, card: dict[str, Any]) -> bool:
        card_id = str(card.get("id") or "").strip().upper()
        return card_id in {"CARD.BLOODLETTING", "CARD.OFFERING"}

    def _hp_loss_energy_profile(self, card: dict[str, Any]) -> tuple[int, int]:
        card_id = str(card.get("id") or "").strip().upper()
        hp_loss = self._card_hp_loss(card)
        energy_gain = self._card_energy_gain(card)
        if card_id == "CARD.BLOODLETTING":
            hp_loss = max(hp_loss, 3)
            energy_gain = max(energy_gain, 3)
        elif card_id == "CARD.OFFERING":
            hp_loss = max(hp_loss, 6)
            energy_gain = max(energy_gain, 2)
        return hp_loss, energy_gain

    def _hp_loss_energy_followup_value(self, state: GameStateView, card: dict[str, Any]) -> tuple[int, int]:
        selected_index = card.get("index")
        _, energy_gain = self._hp_loss_energy_profile(card)
        available_energy = max(0, state.energy + energy_gain)
        unlockable_cards = [
            item
            for item in state.hand
            if isinstance(item, dict)
            and item.get("index") != selected_index
            and self._text(item.get("type")) not in {"status", "curse"}
            and max(0, self._safe_int(item.get("cost"), 0)) <= available_energy
        ]
        unlocked_block = sum(self._card_block(item) for item in unlockable_cards)
        unlocked_damage = sum(self._card_damage(item) for item in unlockable_cards)
        return unlocked_block, unlocked_damage

    def _should_force_skip_hp_loss_energy_card(
        self,
        state: GameStateView,
        card: dict[str, Any],
        incoming_damage: int,
        block_gap: int,
    ) -> bool:
        if not self._is_hp_loss_energy_card_id(card):
            return False
        if not (self._is_ceremonial_beast_boss(state) or self._is_kin_boss(state)):
            return False
        hp_loss, _energy_gain = self._hp_loss_energy_profile(card)
        hp_ratio = state.hp / max(1, state.max_hp)
        round_no = self._safe_int(state.round, 0)
        followup_count, spendable_cost = self._playable_followup_pressure(state, card)
        unlocked_block, unlocked_damage = self._hp_loss_energy_followup_value(state, card)
        lowest_enemy_ehp = min((self._enemy_effective_hp(enemy) for enemy in state.living_enemies()), default=999)
        projected_hp = state.hp - hp_loss + state.block
        can_convert_now = (
            projected_hp > 0
            and (
                unlocked_block >= max(8, min(block_gap, 16))
                or (incoming_damage <= 0 and unlocked_damage >= max(20, lowest_enemy_ehp))
                or (block_gap > 0 and unlocked_damage >= lowest_enemy_ehp and block_gap <= max(10, int(state.hp * 0.25)))
            )
        )
        if can_convert_now:
            return False
        if block_gap > 0:
            return True
        if state.hp <= max(34, int(state.max_hp * 0.62)):
            return True
        if hp_ratio < 0.78:
            return True
        if round_no >= 6:
            return True
        if incoming_damage <= 0 and round_no >= 4:
            return True
        if followup_count < 3 or spendable_cost <= state.energy:
            return True
        return False

    def _boss_enemy_key(self, state: GameStateView) -> str:
        return " ".join(
            self._text(enemy.get("id") or enemy.get("name") or "")
            for enemy in state.living_enemies()
            if isinstance(enemy, dict)
        )

    def _is_vantom_boss(self, state: GameStateView) -> bool:
        return "vantom" in self._boss_enemy_key(state)

    def _is_insatiable_boss(self, state: GameStateView) -> bool:
        key = self._boss_enemy_key(state)
        return "insatiable" in key or "sand" in key or "沙虫" in key

    def _is_ceremonial_beast_boss(self, state: GameStateView) -> bool:
        key = self._boss_enemy_key(state)
        return (
            "ceremonial_beast_boss" in key
            or "ceremonial_beast" in key
            or "ceremonial beast" in key
            or "仪式兽" in key
        )

    def _is_kin_boss(self, state: GameStateView) -> bool:
        key = self._boss_enemy_key(state)
        return "kin_priest" in key or "kin_follower" in key or "the_kin_boss" in key or "同族" in key

    def _sandpit_turns(self, state: GameStateView) -> int:
        for enemy in state.living_enemies():
            for power in enemy.get("powers") or []:
                if not isinstance(power, dict):
                    continue
                power_id = str(power.get("id") or power.get("name") or "").strip().upper()
                if "SANDPIT" not in power_id and "沙坑" not in power_id:
                    continue
                amount = self._safe_int(power.get("amount"), 0)
                if amount > 0:
                    return amount
        return 0

    def _potion_heal_amount(self, potion: dict[str, Any]) -> int:
        vars_payload = potion.get("vars") or {}
        if isinstance(vars_payload, dict):
            values = []
            for key, value in vars_payload.items():
                if any(token in str(key).lower() for token in ("heal", "health", "hp")):
                    parsed = self._safe_int(value, None)
                    if parsed is not None and parsed > 0:
                        values.append(parsed)
            if values:
                return max(values)

        text = f"{potion.get('name') or ''} {potion.get('description') or ''}"
        lowered = text.lower()
        if not any(token in lowered for token in ("heal", "healing", "restore", "recover", "regain")) and not any(
            token in text for token in ("恢复", "治疗", "回复", "生命")
        ):
            return 0
        digits = [int(item) for item in re.findall(r"\d+", text)]
        return max(digits) if digits else -1

    def _potion_damage_amount(self, potion: dict[str, Any]) -> int:
        potion_id = str(potion.get("id") or "").strip().upper()
        if potion_id in {"POWDERED_DEMISE", "ASHWATER", "EXPLOSIVE_AMPOULE"}:
            return 12
        vars_payload = potion.get("vars") or {}
        if isinstance(vars_payload, dict):
            values = []
            for key, value in vars_payload.items():
                if "damage" in str(key).lower():
                    parsed = self._safe_int(value, None)
                    if parsed is not None and parsed > 0:
                        values.append(parsed)
            if values:
                return max(values)

        text = f"{potion.get('name') or ''} {potion.get('description') or ''}"
        lowered = text.lower()
        if not any(token in lowered for token in ("damage", "deal")) and not any(token in text for token in ("伤害", "造成")):
            return 0
        digits = [int(item) for item in re.findall(r"\d+", text)]
        return max(digits) if digits else 0

    def _potion_self_damage_amount(self, potion: dict[str, Any]) -> int:
        text = f"{potion.get('name') or ''} {potion.get('description') or ''}"
        lowered = text.lower()
        if not any(token in lowered for token in ("everyone", "all characters")) and "\u6240\u6709\u4eba" not in text:
            return 0
        return self._potion_damage_amount(potion)

    def _potion_block_amount(self, potion: dict[str, Any]) -> int:
        vars_payload = potion.get("vars") or {}
        if isinstance(vars_payload, dict):
            values = []
            for key, value in vars_payload.items():
                if "block" in str(key).lower():
                    parsed = self._safe_int(value, None)
                    if parsed is not None and parsed > 0:
                        values.append(parsed)
            if values:
                return max(values)

        text = f"{potion.get('name') or ''} {potion.get('description') or ''}"
        lowered = text.lower()
        if not any(token in lowered for token in ("block", "armor", "shield")) and not any(token in text for token in ("格挡", "护甲", "护盾")):
            return 0
        digits = [int(item) for item in re.findall(r"\d+", text)]
        return max(digits) if digits else 0

    def _potion_strength_amount(self, potion: dict[str, Any]) -> int:
        vars_payload = potion.get("vars") or {}
        if isinstance(vars_payload, dict):
            values = []
            for key, value in vars_payload.items():
                if "strength" in str(key).lower():
                    parsed = self._safe_int(value, None)
                    if parsed is not None and parsed > 0:
                        values.append(parsed)
            if values:
                return max(values)

        text = f"{potion.get('name') or ''} {potion.get('description') or ''}"
        lowered = text.lower()
        if "strength" not in lowered and "力量" not in text:
            return 0
        digits = [int(item) for item in re.findall(r"\d+", text)]
        return max(digits) if digits else 1

    def _potion_dexterity_amount(self, potion: dict[str, Any]) -> int:
        vars_payload = potion.get("vars") or {}
        if isinstance(vars_payload, dict):
            values = []
            for key, value in vars_payload.items():
                if "dex" in str(key).lower() or "dexterity" in str(key).lower():
                    parsed = self._safe_int(value, None)
                    if parsed is not None and parsed > 0:
                        values.append(parsed)
            if values:
                return max(values)

        text = f"{potion.get('name') or ''} {potion.get('description') or ''}"
        lowered = text.lower()
        if "dex" not in lowered and "dexterity" not in lowered and "敏捷" not in text:
            return 0
        digits = [int(item) for item in re.findall(r"\d+", text)]
        return max(digits) if digits else 1

    def _hand_block_card_count(self, state: GameStateView) -> int:
        return sum(1 for card in state.playable_cards() if self._card_block(card) > 0)

    def _hand_attack_card_count(self, state: GameStateView) -> int:
        return sum(1 for card in state.playable_cards() if self._card_damage(card) > 0)

    def _potion_energy_amount(self, potion: dict[str, Any]) -> int:
        vars_payload = potion.get("vars") or {}
        if isinstance(vars_payload, dict):
            values = []
            for key, value in vars_payload.items():
                if "energy" in str(key).lower():
                    parsed = self._safe_int(value, None)
                    if parsed is not None and parsed > 0:
                        values.append(parsed)
            if values:
                return max(values)

        text = f"{potion.get('name') or ''} {potion.get('description') or ''}"
        lowered = text.lower()
        if "energy" not in lowered and "能量" not in text:
            return 0
        digits = [int(item) for item in re.findall(r"\d+", text)]
        return max(digits) if digits else 1

    def _potion_enemy_debuff_hint(self, potion: dict[str, Any]) -> float:
        text = f"{potion.get('id') or ''} {potion.get('name') or ''} {potion.get('description') or ''}".lower()
        score = 0.0
        if any(token in text for token in ("weak", "虚弱")):
            score += 26.0
        if any(token in text for token in ("bind", "binding", "缚", "束缚", "entangle", "缠绕")):
            score += 20.0
        if any(token in text for token in ("vulnerable", "易伤")):
            score += 8.0
        if any(token in text for token in ("frail", "脆弱")):
            score += 6.0
        return score

    def _is_known_boss_setup_potion(self, potion: dict[str, Any]) -> bool:
        potion_id = str(potion.get("id") or "").strip().upper()
        return potion_id in {
            "BLESSING_OF_THE_FORGE",
            "SWIFT_POTION",
            "SKILL_POTION",
            "COLORLESS_POTION",
            "DROPLET_OF_PRECOGNITION",
            "MAZALETHS_GIFT",
            "TOUCH_OF_INSANITY",
        }

    def _boss_setup_potion_role(self, potion: dict[str, Any]) -> str:
        potion_id = str(potion.get("id") or "").strip().upper()
        if potion_id in {"SWIFT_POTION", "SKILL_POTION", "COLORLESS_POTION", "DROPLET_OF_PRECOGNITION"}:
            return "draw"
        if potion_id in {"BLESSING_OF_THE_FORGE", "MAZALETHS_GIFT", "TOUCH_OF_INSANITY"}:
            return "setup"
        return "unknown"

    def _pick_combat_potion_action(self, state: GameStateView) -> FlowAction | None:
        incoming_damage = self._enemy_intended_damage(state)
        block_gap = max(0, incoming_damage - state.block)
        missing_hp = max(state.max_hp - state.hp, 0)
        hp_ratio = state.hp / max(1, state.max_hp)
        act, floor, room_type, is_boss_room = self._combat_context(state)
        is_boss_floor = floor >= 16 or is_boss_room
        is_act2_boss = act >= 2 and is_boss_floor
        is_elite_room = "elite" in room_type
        high_value_combat = is_boss_floor or is_elite_room
        late_act_pressure = floor >= 13
        early_boss_turn = is_boss_floor and (state.round in (None, 1, 2, 3))
        early_high_value_turn = high_value_combat and (state.round in (None, 1, 2, 3))
        early_setup_turn = is_boss_floor and (state.round in (None, 1, 2, 3))

        best_choice: tuple[float, dict[str, Any]] | None = None
        best_self_choice: tuple[float, dict[str, Any]] | None = None
        best_setup_choice: tuple[float, dict[str, Any], str] | None = None
        for potion in state.player.get("potions") or []:
            if not isinstance(potion, dict) or potion.get("index") is None:
                continue
            potion_id = str(potion.get("id") or "").strip().upper()
            if potion_id == "POTION.DISTILLED_CHAOS":
                continue
            target_type = self._text(potion.get("target_type"))
            if "enemy" in target_type:
                continue

            block_amount = self._potion_block_amount(potion)
            if block_amount > 0:
                if block_gap > 0 and (high_value_combat or incoming_damage >= state.hp or hp_ratio < 0.28):
                    score = min(block_amount, block_gap) * 1.4
                    if incoming_damage >= state.hp:
                        score += 30.0
                    elif is_act2_boss and block_gap >= 8:
                        score += 14.0
                    elif is_boss_floor and block_gap >= 8:
                        score += 10.0
                    elif not high_value_combat:
                        score -= 8.0
                    if best_self_choice is None or score > best_self_choice[0]:
                        best_self_choice = (score, potion)

            strength_amount = self._potion_strength_amount(potion)
            if strength_amount > 0 and early_high_value_turn:
                attack_card_count = self._hand_attack_card_count(state)
                if attack_card_count > 0:
                    score = 14.0 + strength_amount * 6.0 + min(attack_card_count, 3) * 2.5
                    if best_self_choice is None or score > best_self_choice[0]:
                        best_self_choice = (score, potion)

            energy_amount = self._potion_energy_amount(potion)
            if energy_amount > 0 and (
                early_high_value_turn
                or incoming_damage >= state.hp
                or (high_value_combat and hp_ratio < 0.45)
                or (high_value_combat and late_act_pressure and hp_ratio < 0.7)
            ):
                _, spendable_cost = self._playable_followup_pressure(state, {"index": None})
                if spendable_cost > state.energy:
                    score = 10.0 + min(energy_amount, spendable_cost - state.energy) * 5.0
                    if late_act_pressure and hp_ratio < 0.7:
                        score += 5.0
                    if best_self_choice is None or score > best_self_choice[0]:
                        best_self_choice = (score, potion)

            dexterity_amount = self._potion_dexterity_amount(potion)
            if dexterity_amount > 0 and early_high_value_turn:
                block_card_count = self._hand_block_card_count(state)
                if block_card_count > 0:
                    score = 12.0 + dexterity_amount * 5.0 + min(block_card_count, 3) * 2.5
                    if incoming_damage > state.block:
                        score += 4.0
                    if best_self_choice is None or score > best_self_choice[0]:
                        best_self_choice = (score, potion)

            if self._is_known_boss_setup_potion(potion) and early_setup_turn:
                role = self._boss_setup_potion_role(potion)
                score = 14.0 + (8.0 if is_boss_floor else 0.0)
                if role == "draw":
                    score += 6.0
                    if is_act2_boss:
                        score += 5.0
                    if block_gap >= 10:
                        score += min(block_gap, 20) * 1.15
                        if self._hand_block_card_count(state) <= 1:
                            score += 10.0
                    elif is_act2_boss and block_gap >= 8:
                        score += min(block_gap, 16) * 1.2
                        if self._hand_block_card_count(state) <= 1:
                            score += 8.0
                    elif incoming_damage <= 0:
                        score += 4.0
                elif incoming_damage <= 0:
                    score += 6.0
                elif block_gap <= 8 and hp_ratio >= 0.5:
                    score += 2.0
                else:
                    score -= 6.0
                if best_setup_choice is None or score > best_setup_choice[0]:
                    best_setup_choice = (score, potion, role)

            if missing_hp <= 0:
                continue
            if incoming_damage <= 0 and hp_ratio > (0.72 if high_value_combat else 0.28):
                continue
            heal_amount = self._potion_heal_amount(potion)
            if heal_amount == 0:
                continue
            effective_heal = missing_hp if heal_amount < 0 else min(heal_amount, missing_hp)
            score = float(effective_heal)
            if incoming_damage >= state.hp:
                score += 20.0
            elif high_value_combat and hp_ratio < 0.72:
                score += 12.0
            elif hp_ratio < 0.25:
                score += 10.0
            elif hp_ratio < 0.4:
                score += 5.0
            if best_choice is None or score > best_choice[0]:
                best_choice = (score, potion)

        best_attack_choice: tuple[float, dict[str, Any], dict[str, Any] | None] | None = None
        for potion in state.player.get("potions") or []:
            if not isinstance(potion, dict) or potion.get("index") is None:
                continue
            potion_id = str(potion.get("id") or "").strip().upper()
            if potion_id == "POTION.DISTILLED_CHAOS":
                continue
            target_type = self._text(potion.get("target_type"))
            if "enemy" not in target_type:
                continue
            potion_damage = self._potion_damage_amount(potion)
            if potion_damage <= 0:
                continue
            self_damage = self._potion_self_damage_amount(potion)
            if self_damage > 0:
                enemies = state.living_enemies()
                kills_all_enemies = bool(enemies) and all(potion_damage >= self._enemy_effective_hp(enemy) for enemy in enemies)
                if state.hp <= self_damage or not kills_all_enemies:
                    continue
            target = self._pick_damage_target(state.living_enemies(), potion_damage)
            if target is None:
                continue
            target_ehp = self._enemy_effective_hp(target)
            score = float(potion_damage)
            if self_damage > 0:
                score -= self_damage * 2.5
                score += 45.0
            if potion_damage >= target_ehp:
                score += 30.0 + self._enemy_threat(target) * 1.5
            if incoming_damage >= state.hp:
                score += 12.0
            elif hp_ratio < 0.35:
                score += 5.0
            if best_attack_choice is None or score > best_attack_choice[0]:
                best_attack_choice = (score, potion, target)

        best_debuff_choice: tuple[float, dict[str, Any], dict[str, Any] | None] | None = None
        for potion in state.player.get("potions") or []:
            if not isinstance(potion, dict) or potion.get("index") is None:
                continue
            potion_id = str(potion.get("id") or "").strip().upper()
            if potion_id == "POTION.DISTILLED_CHAOS":
                continue
            target_type = self._text(potion.get("target_type"))
            if "enemy" not in target_type:
                continue
            if self._potion_damage_amount(potion) > 0:
                continue
            debuff_hint = self._potion_enemy_debuff_hint(potion)
            if debuff_hint <= 0:
                continue
            enemies = state.living_enemies()
            target = max(enemies, key=self._enemy_threat) if enemies else None
            score = debuff_hint
            if high_value_combat:
                score += 8.0
            if early_boss_turn:
                score += 6.0
            if block_gap > 0:
                score += min(block_gap, 22) * 1.25
            if incoming_damage >= state.hp:
                score += 28.0
            elif hp_ratio < 0.55 and block_gap >= 8:
                score += 10.0
            if best_debuff_choice is None or score > best_debuff_choice[0]:
                best_debuff_choice = (score, potion, target)

        if best_attack_choice is not None and (
            incoming_damage >= state.hp
            or (not high_value_combat and hp_ratio < 0.35 and best_attack_choice[0] >= 35.0)
            or (high_value_combat and hp_ratio < 0.35)
            or (high_value_combat and best_attack_choice[0] >= 35.0)
            or (early_high_value_turn and self._potion_damage_amount(best_attack_choice[1]) >= 10)
        ):
            args = {"potion_index": best_attack_choice[1].get("index", 0)}
            if "all" not in self._text(best_attack_choice[1].get("target_type")) and best_attack_choice[2] is not None:
                args["target_index"] = best_attack_choice[2].get("index")
            return FlowAction("use_potion", args)

        if best_debuff_choice is not None and (
            incoming_damage >= state.hp
            or (high_value_combat and block_gap >= 8)
            or (is_act2_boss and block_gap >= 5 and hp_ratio < 0.85)
            or (is_boss_floor and block_gap >= 6 and hp_ratio < 0.8)
        ):
            args = {"potion_index": best_debuff_choice[1].get("index", 0)}
            if "all" not in self._text(best_debuff_choice[1].get("target_type")) and best_debuff_choice[2] is not None:
                args["target_index"] = best_debuff_choice[2].get("index")
            return FlowAction("use_potion", args)

        if best_setup_choice is not None and (
            (
                early_setup_turn
                and (incoming_damage <= 0 or (block_gap <= 8 and hp_ratio >= 0.5))
            )
            or (
                best_setup_choice[2] == "draw"
                and is_boss_floor
                and block_gap >= 10
                and hp_ratio < 0.8
            )
            or (
                best_setup_choice[2] == "draw"
                and is_act2_boss
                and block_gap >= 8
                and hp_ratio < 0.85
            )
        ):
            return FlowAction("use_potion", {"potion_index": best_setup_choice[1].get("index", 0)})

        if best_self_choice is not None and (
            incoming_damage >= state.hp
            or (high_value_combat and best_self_choice[0] >= 22.0)
            or (early_high_value_turn and best_self_choice[0] >= 16.0)
        ):
            return FlowAction("use_potion", {"potion_index": best_self_choice[1].get("index", 0)})

        if best_choice is not None and (
            incoming_damage >= state.hp
            or hp_ratio < 0.22
            or (high_value_combat and hp_ratio < 0.72)
        ):
            return FlowAction("use_potion", {"potion_index": best_choice[1].get("index", 0)})
        return None

    def _enemy_effective_hp(self, enemy: dict[str, Any]) -> int:
        return self._safe_int(enemy.get("hp"), 0) + self._safe_int(enemy.get("block"), 0)

    def _pick_damage_target(self, enemies: list[dict[str, Any]], damage: int) -> dict[str, Any] | None:
        if not enemies:
            return None

        def score(enemy: dict[str, Any]) -> tuple[float, int, int]:
            effective_hp = self._enemy_effective_hp(enemy)
            lethal = 1 if damage >= effective_hp and damage > 0 else 0
            threat = self._enemy_threat(enemy)
            return (
                lethal * 1000 + threat * 100 - effective_hp,
                -effective_hp,
                -self._safe_int(enemy.get("index"), 0),
            )

        return max(enemies, key=score)

    def _pick_combat_target(
        self,
        state: GameStateView,
        card: dict[str, Any],
        enemies: list[dict[str, Any]],
    ) -> dict[str, Any]:
        damage = self._card_damage(card)
        _, floor, room_type, is_boss_room = self._combat_context(state)
        if floor >= 16 and is_boss_room and self._is_kin_boss(state) and len(enemies) > 1 and damage > 0:
            def kin_target_score(enemy: dict[str, Any]) -> tuple[int, int, int, int, int]:
                enemy_id = self._text(enemy.get("id") or enemy.get("name"))
                is_priest = 1 if "priest" in enemy_id or "神官" in enemy_id else 0
                is_follower = 1 if "follower" in enemy_id or "同族教徒" in enemy_id else 0
                effective_hp = self._enemy_effective_hp(enemy)
                lethal = 1 if damage >= effective_hp else 0
                threat = self._enemy_threat(enemy)
                return (
                    lethal,
                    is_follower,
                    -effective_hp,
                    threat,
                    -is_priest,
                )

            return max(enemies, key=kin_target_score)
        if floor >= 16 and is_boss_room and len(enemies) > 1 and damage > 0:
            def boss_target_score(enemy: dict[str, Any]) -> tuple[float, int, int]:
                effective_hp = self._enemy_effective_hp(enemy)
                lethal = 1 if damage >= effective_hp else 0
                threat = self._enemy_threat(enemy)
                return (
                    lethal * 5000.0 + threat * 35.0 + min(damage, effective_hp) * 3.0 - effective_hp * 3.0,
                    -effective_hp,
                    -self._safe_int(enemy.get("index"), 0),
                )

            return max(enemies, key=boss_target_score)
        return self._pick_damage_target(enemies, damage) or enemies[0]

    def _combat_card_score(self, state: GameStateView, card: dict[str, Any]) -> float:
        card_id = str(card.get("id") or "").strip().upper()
        if card_id in {"CARD.CASCADE", "CARD.HAVOC"}:
            return -100.0
        card_name = self._text(card.get("name") or card.get("card_id") or card.get("id"))
        description = self._text(card.get("description"))
        card_type = self._text(card.get("type"))
        cost = max(0, self._safe_int(card.get("cost"), 0))
        damage = self._card_damage(card)
        block = self._card_block(card)
        hp_loss = self._card_hp_loss(card)
        draw_amount = self._card_draw(card)
        energy_gain = self._card_energy_gain(card)
        enemies = state.living_enemies()
        attack_pressure = self._enemy_attack_pressure(state)
        incoming_damage = self._enemy_intended_damage(state)
        hp_ratio = state.hp / max(1, state.max_hp)
        act, floor, room_type, is_boss_room = self._combat_context(state)
        is_boss_floor = floor >= 16 or is_boss_room
        is_act2_boss = act >= 2 and is_boss_floor
        is_vantom = self._is_vantom_boss(state)
        is_insatiable = self._is_insatiable_boss(state)
        is_ceremonial_beast = self._is_ceremonial_beast_boss(state)
        is_kin = self._is_kin_boss(state)
        is_targeted_boss = is_boss_floor or is_act2_boss or is_ceremonial_beast or is_kin
        round_no = self._safe_int(state.round, 0)
        sandpit_turns = self._sandpit_turns(state) if is_insatiable else 0
        lowest_enemy_ehp = min(self._enemy_effective_hp(enemy) for enemy in enemies)
        block_gap = max(0, incoming_damage - state.block)
        lethal_pressure = block_gap >= state.hp
        can_tank = state.hp > incoming_damage * 2
        boss_high_pressure = (is_boss_floor and block_gap >= 10) or (is_act2_boss and block_gap >= 8)
        boss_imminent_death = (is_boss_floor and block_gap >= max(1, state.hp - 2)) or (
            is_act2_boss and block_gap >= max(1, state.hp - 3)
        )
        can_lethal_now = damage > 0 and damage >= lowest_enemy_ehp

        score = 0.0
        if card_id == "CARD.FRANTIC_ESCAPE":
            if is_insatiable:
                if sandpit_turns <= 1:
                    score += 150.0
                elif sandpit_turns == 2:
                    score += 95.0
                elif sandpit_turns == 3:
                    score += 42.0
                else:
                    score += 18.0
                if state.energy <= cost:
                    score += 8.0
            else:
                score -= 10.0
        if damage > 0:
            score += damage * 1.55
            if is_boss_floor:
                score += damage * 0.75
            if is_ceremonial_beast and 1 <= round_no <= 4 and incoming_damage <= 24 and hp_ratio >= 0.35:
                score += damage * 0.95
            if is_kin and len(enemies) > 1 and 1 <= round_no <= 4:
                score += damage * 0.45
                follower_ehps = [
                    self._enemy_effective_hp(enemy)
                    for enemy in enemies
                    if any(token in self._text(enemy.get("id") or enemy.get("name")) for token in ("follower", "同族教徒"))
                ]
                if follower_ehps:
                    min_follower_ehp = min(follower_ehps)
                    if damage >= min_follower_ehp:
                        score += 42.0
                    elif damage >= max(8, int(min_follower_ehp * 0.5)):
                        score += 12.0
            if is_vantom:
                score += damage * 2.2
            if damage >= lowest_enemy_ehp:
                score += 18.0
            if "vulnerable" in description or "易伤" in description:
                score += 7.0 if state.round in (None, 1, 2) else 3.0
                if is_boss_floor:
                    score += 4.0
        if block > 0:
            if attack_pressure > 0:
                score += min(block, 12) * (1.35 if hp_ratio < 0.55 else 1.0)
            else:
                score += block * 0.25
            if is_act2_boss and block_gap > 0:
                score += min(block, block_gap) * 1.15 + 4.0
            if is_ceremonial_beast and 1 <= round_no <= 4 and incoming_damage <= 18 and damage <= 0:
                score -= min(block, 12) * 1.2
            current_projected_hp = state.hp + state.block - incoming_damage
            projected_hp_after_block = state.hp + state.block + block - incoming_damage
            prevented = max(0, min(block, incoming_damage - state.block))
            score += prevented * 1.8
            if incoming_damage > 0 and projected_hp_after_block > current_projected_hp:
                score += 2.0
            if current_projected_hp <= 0 < projected_hp_after_block:
                score += 24.0
            elif projected_hp_after_block <= 0:
                score -= 4.0
            if is_boss_floor and damage <= 0 and not lethal_pressure:
                comfortable_hp = max(18, int(state.max_hp * 0.3))
                if current_projected_hp >= comfortable_hp:
                    score -= min(block, 12) * 1.35
                    if any(token in card_name for token in ("defend", "防御")):
                        score -= 4.0
            if is_vantom and damage <= 0 and current_projected_hp > 0:
                score -= min(block, 16) * 2.3
                if any(token in card_name for token in ("defend", "防御")):
                    score -= 7.0
        if "draw" in description or "抽牌" in description:
            score += 3.0
        if draw_amount > 0:
            score += min(draw_amount, 4) * 3.0
            score += self._guaranteed_draw_followup_bonus(state, card)
            if is_boss_floor and cost <= 1 and state.energy >= cost:
                followup_count, spendable_cost = self._playable_followup_pressure(state, card)
                if followup_count >= 2 and spendable_cost > max(0, state.energy - cost):
                    score += 8.0 + min(followup_count, 5) * 1.2
            if is_boss_floor:
                score += min(draw_amount, 4) * 1.8
            if is_act2_boss and block_gap > 0:
                score += min(draw_amount, 3) * 3.0
        if "strength" in description or "力量" in description:
            score += 3.5
            if is_boss_floor and state.round in (None, 1, 2, 3):
                score += 5.0
        if "energy" in description or "能量" in description:
            score += 2.5
        if energy_gain > 0:
            followup_count, spendable_cost = self._playable_followup_pressure(state, card)
            usable_energy = max(0, min(energy_gain, spendable_cost - max(0, state.energy - cost)))
            score += usable_energy * 4.0 + min(followup_count, 4) * 0.8
            if is_boss_floor and cost == 0 and usable_energy > 0:
                score += 12.0 + usable_energy * 3.0
            if is_boss_floor and usable_energy > 0 and spendable_cost >= state.energy + 2:
                score += 8.0
            if is_boss_floor and followup_count >= 2 and usable_energy > 0:
                score += 8.0 + usable_energy * 3.0 + min(followup_count, 5) * 1.4
            if followup_count < 2 or usable_energy <= 0:
                score -= 7.0
        if "weak" in description or "虚弱" in description:
            score += 2.5
            if block_gap > 0:
                score += 10.0 + min(block_gap, 24) * 0.85
                if is_targeted_boss:
                    score += 6.0
                if is_act2_boss:
                    score += 6.0
                if is_ceremonial_beast and hp_ratio < 0.55:
                    score += 8.0
        if hp_loss > 0:
            absolute_hp_floor = hp_loss + (8 if is_boss_floor else 6)
            if state.hp <= absolute_hp_floor and not lethal_pressure:
                score -= 45.0
            if energy_gain > 0:
                followup_count, spendable_cost = self._playable_followup_pressure(state, card)
                can_convert_hp = followup_count >= 2 and spendable_cost > state.energy
                score -= hp_loss * (2.6 if hp_ratio < 0.45 else 0.8)
                if is_vantom and state.round and state.round >= 2 and hp_ratio < 0.75:
                    score -= hp_loss * 2.5
                if can_convert_hp and hp_ratio >= 0.45:
                    score += 35.0
                if not can_convert_hp:
                    score -= 8.0
                if is_boss_floor and hp_ratio < 0.42 and not lethal_pressure:
                    followup_quality = self._playable_followup_quality(state, card)
                    if followup_quality < 4.5:
                        score -= 22.0
                    if incoming_damage <= state.block and followup_quality < 7.0:
                        score -= 14.0
                if is_boss_floor and not lethal_pressure:
                    followup_quality = self._playable_followup_quality(state, card)
                    projected_hp = state.hp - hp_loss + state.block - incoming_damage
                    if projected_hp < max(18, int(state.max_hp * 0.22)) and followup_quality < 7.0:
                        score -= 18.0
                    if hp_ratio < 0.68 and followup_quality < 5.5:
                        score -= 10.0
                    if incoming_damage <= state.block and followup_quality < 8.0:
                        score -= 8.0
                if state.hp <= hp_loss + 14 and not can_convert_hp:
                    score -= 18.0
                if is_boss_floor and block_gap > 0:
                    score -= hp_loss * 5.0
                    if can_convert_hp:
                        score -= 10.0 + min(block_gap, 12) * 1.5
                if (is_ceremonial_beast or is_kin) and block_gap > 0:
                    score -= hp_loss * 6.0
            else:
                score -= hp_loss * (2.4 if hp_ratio < 0.6 else 1.1)
        if "lose hp" in description or "失去生命" in description or "失去" in description and "生命" in description:
            if energy_gain <= 0:
                score -= 6.0 if hp_ratio < 0.6 else 2.5
        if "exhaust" in description or "消耗" in description:
            score -= 1.25
        if self._is_random_exhaust_risk(card):
            if block_gap <= 0:
                score -= 28.0
            elif not lethal_pressure and can_tank:
                score -= 18.0
            else:
                score -= 6.0
            if self._has_valuable_exhaust_target_in_hand(state, card):
                score -= 14.0
        if card_type == "attack":
            score += 1.5
            if is_boss_floor:
                score += 2.0
                if any(token in card_name for token in ("strike", "打击")) and draw_amount <= 0 and energy_gain <= 0:
                    score -= 3.0
            if is_vantom:
                score += 5.0
        elif card_type == "skill":
            score += 0.5
        elif card_type == "power":
            if lethal_pressure:
                score -= 12.0
            elif incoming_damage > 15 and hp_ratio < 0.55:
                score -= 2.0
            else:
                score += 2.0 if state.round in (None, 1, 2, 3) else -1.0
            if is_ceremonial_beast and 1 <= round_no <= 3 and incoming_damage <= 22:
                score += 8.0
            if is_boss_floor and not lethal_pressure:
                if state.round in (None, 1, 2):
                    score += 11.0
                elif state.round == 3:
                    score += 7.0
                else:
                    score += 2.0
                scaling_text = f"{card_name} {description}"
                if any(
                    token in scaling_text
                    for token in (
                        "demon",
                        "inflame",
                        "rupture",
                        "aggression",
                        "barricade",
                        "feel no pain",
                        "dark embrace",
                        "corruption",
                        "力量",
                        "恶魔",
                    )
                ):
                    score += 5.0

        if cost <= 0:
            score += 2.0
        else:
            score += min((damage + block * 0.8) / max(cost, 1), 8.0)
            score -= max(0, cost - max(1, state.energy)) * 2.0

        if any(token in card_name for token in ("headbutt", "剑柄")):
            score += 7.0
            if is_boss_floor:
                score += 4.0
        if any(token in card_name for token in ("thunderclap", "雷霆", "anger", "愤怒", "bludgeon", "重锤")):
            score += 4.0
        if card_id == "CARD.ANGER" and is_boss_floor:
            score += 6.0
        if "grapple" in card_name or "擒拿" in card_name:
            score -= 5.0
        if "rampage" in card_name or "暴走" in card_name:
            score -= 3.0
        if "forgotten_ritual" in card_name or "forgotten ritual" in card_name:
            score -= 5.0

        if card_id in {"CARD.BLOODLETTING", "CARD.OFFERING"} and is_targeted_boss:
            followup_count, spendable_cost = self._playable_followup_pressure(state, card)
            round_no = self._safe_int(state.round, 0)
            if is_ceremonial_beast or is_kin:
                hard_block = (
                    block_gap > 0
                    or incoming_damage <= 0 and round_no >= 6
                    or state.hp <= max(34, int(state.max_hp * 0.62))
                    or followup_count < 3
                    or spendable_cost <= state.energy
                )
                if hard_block:
                    return -240.0
            if block_gap > 0:
                score -= 45.0 + min(block_gap, 20) * 1.8
                if followup_count < 3 or spendable_cost <= state.energy:
                    score -= 22.0
            if state.hp <= max(24, int(state.max_hp * 0.42)):
                score -= 35.0
            if is_ceremonial_beast or is_kin:
                score -= 12.0
                if incoming_damage <= 0 and round_no >= 6:
                    score -= 55.0
                if state.hp <= max(34, int(state.max_hp * 0.62)):
                    score -= 85.0
                if followup_count < 3 or spendable_cost <= state.energy:
                    score -= 35.0
            if is_ceremonial_beast and hp_ratio < 0.75:
                score -= 80.0

        if boss_high_pressure:
            if block > 0:
                score += min(block, block_gap) * 2.8 + 8.0
            if block <= 0 and not can_lethal_now:
                score -= 16.0 + min(block_gap, 20) * 0.9
                if is_ceremonial_beast and hp_ratio < 0.45:
                    score -= 22.0
            if damage <= 0 and block <= 0 and hp_loss <= 0 and card_type != "power":
                score -= 18.0
            if cost >= 2 and damage > 0 and block <= 0 and not can_lethal_now:
                if "weak" in description or "虚弱" in description:
                    score -= 2.0
                else:
                    score -= 14.0
                if is_act2_boss:
                    score -= 10.0
                if is_ceremonial_beast and hp_ratio < 0.45:
                    score -= 10.0
            if card_id in {"CARD.BLUDGEON", "CARD.HEAVY_BLADE", "CARD.PERFECTED_STRIKE"} and block <= 0 and not can_lethal_now:
                score -= 16.0

        if boss_imminent_death and block <= 0 and not can_lethal_now:
            score -= 30.0
        if is_ceremonial_beast and incoming_damage <= 0 and block > 0 and damage <= 0:
            score -= 20.0

        if attack_pressure <= 0 and block > damage + 2:
            score -= 4.0
        if attack_pressure > 0 and hp_ratio < 0.4 and block > 0:
            score += 5.0
        if block_gap > 0 and block > 0 and (lethal_pressure or not can_tank or block_gap > 15):
            score += 8.0
        if incoming_damage >= state.hp and damage > 0 and damage < lowest_enemy_ehp:
            score -= 8.0
        return score

    def _pick_card_reward_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        deck = list(state.player.get("deck") or [])
        choice = choose_card_reward(state.cards, deck, state.can_skip, state)
        if choice is not None:
            return FlowAction("choose_card_reward", {"card_index": choice.get("index", 0)})
        return FlowAction("skip_reward")

    def _pick_shop_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        purge_cost = shop_purge_cost(state)
        deck = list(state.player.get("deck") or [])
        best_relic = best_shop_relic(state.relics, state.gold)
        best_card = best_shop_card(state.cards, state.gold, deck)
        purge_target = choose_purge_target(deck)
        if should_prioritize_shop_purge(deck, state.gold, purge_cost, best_card=best_card, best_relic=best_relic):
            return FlowAction("purge_card", {"card_index": purge_target.get("index", 0)})

        if best_relic is not None:
            return FlowAction("buy_relic", {"relic_index": best_relic.get("index", 0)})

        if best_card is not None:
            return FlowAction("buy_card", {"card_index": best_card.get("index", 0)})

        if state.gold >= purge_cost and purge_target is not None:
            return FlowAction("purge_card", {"card_index": purge_target.get("index", 0)})

        best_potion = best_shop_potion(state.potions, state.gold)
        context = state.raw.get("context") or {}
        floor = self._safe_int(context.get("floor") or state.raw.get("floor"), 0)
        player_potions = [potion for potion in (state.player.get("potions") or []) if isinstance(potion, dict)]
        potion_slots_total = self._safe_int(state.player.get("potion_slots_total"), 3)
        has_potion_space = len(player_potions) < max(1, potion_slots_total)
        if best_potion is not None and has_potion_space and (
            state.gold >= self.config.shop_high_gold_threshold
            or floor >= 10
            or state.hp / max(1, state.max_hp) < 0.65
        ):
            return FlowAction("buy_potion", {"potion_index": best_potion.get("index", 0)})

        return FlowAction("leave_shop")

    def _pick_event_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        options = state.options or state.choices
        pick = choose_event_option(options, state)
        if pick is None:
            return FlowAction("proceed")
        return FlowAction("choose_option", {"option_index": pick.get("index", 0)})

    def _pick_rest_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        enabled = [option for option in state.options if option.get("is_enabled", True)]
        if not enabled:
            return FlowAction("proceed")

        def option_text(option: dict) -> str:
            return str(option.get("option_id") or option.get("name") or option.get("label") or "").lower()

        hp_ratio = state.hp / max(1, state.max_hp)
        smith = next(
            (
                option
                for option in enabled
                if "smith" in option_text(option) or "upgrade" in option_text(option)
            ),
            None,
        )
        heal_threshold = self.config.rest_heal_threshold
        floor = self._safe_int((state.raw.get("context") or {}).get("floor"), 0)
        act = self._safe_int((state.raw.get("context") or {}).get("act"), 0)
        if act >= 2:
            heal_threshold = max(heal_threshold, 0.82)
            if floor >= 12:
                heal_threshold = max(heal_threshold, 0.88)
        elif floor <= 8:
            heal_threshold = max(heal_threshold, 0.7)
        elif floor >= 14:
            heal_threshold = max(heal_threshold, 0.88)
        elif floor >= 12:
            heal_threshold = max(heal_threshold, 0.72)
        if hp_ratio < max(0.4, heal_threshold):
            heal = next(
                (
                    option
                    for option in enabled
                    if option.get("option_id") == "HEAL"
                    or "heal" in option_text(option)
                    or "rest" in option_text(option)
                ),
                None,
            )
            if heal:
                return FlowAction("choose_option", {"option_index": heal.get("index", 0)})
        elif smith:
            return FlowAction("choose_option", {"option_index": smith.get("index", 0)})

        return FlowAction("choose_option", {"option_index": enabled[0].get("index", 0)})

    def _pick_card_select_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        indices = [str(card.get("index", i)) for i, card in enumerate(state.cards)]
        if not indices:
            return FlowAction("skip_select")
        pick_count = max(1, min(max(1, state.min_select), state.max_select, len(indices)))
        room_type = self._text((state.raw.get("context") or {}).get("room_type"))
        is_combat_select = any(token in room_type for token in ("monster", "elite", "boss", "combat"))

        if is_shop_context(state):
            purge_targets = choose_purge_targets(state.cards, pick_count)
            if len(purge_targets) >= pick_count:
                return FlowAction("select_cards", {"indices": ",".join(str(card.get("index", 0)) for card in purge_targets)})

        if is_combat_select:
            exhaust_targets = choose_purge_targets(state.cards, pick_count)
            if len(exhaust_targets) >= pick_count:
                return FlowAction("select_cards", {"indices": ",".join(str(card.get("index", 0)) for card in exhaust_targets)})

        upgrade_targets = choose_upgrade_targets(state.cards, pick_count, state)
        if len(upgrade_targets) >= pick_count:
            return FlowAction("select_cards", {"indices": ",".join(str(card.get("index", 0)) for card in upgrade_targets)})

        if state.min_select == 0:
            return FlowAction("skip_select")
        return FlowAction("select_cards", {"indices": ",".join(indices[:pick_count])})

    def _pick_bundle_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        if not state.bundles:
            return FlowAction("proceed")
        pick = min(
            state.bundles,
            key=lambda bundle: (
                self._safe_int(bundle.get("cost"), 0),
                self._safe_int(bundle.get("index"), 9999),
            ),
        )
        return FlowAction("select_bundle", {"bundle_index": pick.get("index", 0)})
