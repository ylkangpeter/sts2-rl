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
    choose_upgrade_target,
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
            return FlowAction("end_turn")

        card_scores = [(self._combat_card_score(state, item), item) for item in playable]
        best_score, card = max(card_scores, key=lambda item: item[0])
        if best_score < -5.0:
            return FlowAction("end_turn")
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

    def _enemy_attack_pressure(self, state: GameStateView) -> int:
        pressure = 0
        for enemy in state.living_enemies():
            if enemy.get("intends_attack"):
                pressure += 1
                continue
            for intent in enemy.get("intents") or []:
                if self._intent_is_attack(intent):
                    pressure += 1
                    break
        return pressure

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
        for enemy in state.living_enemies():
            for intent in enemy.get("intents") or []:
                if not isinstance(intent, dict) or not self._intent_is_attack(intent):
                    continue
                total += self._intent_damage(intent)
        return total

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

    def _pick_combat_potion_action(self, state: GameStateView) -> FlowAction | None:
        incoming_damage = self._enemy_intended_damage(state)
        missing_hp = max(state.max_hp - state.hp, 0)
        hp_ratio = state.hp / max(1, state.max_hp)

        best_choice: tuple[float, dict[str, Any]] | None = None
        for potion in state.player.get("potions") or []:
            if not isinstance(potion, dict) or potion.get("index") is None:
                continue
            target_type = self._text(potion.get("target_type"))
            if "enemy" in target_type:
                continue
            if missing_hp <= 0:
                continue
            if incoming_damage <= 0 and hp_ratio > 0.28:
                continue
            heal_amount = self._potion_heal_amount(potion)
            if heal_amount == 0:
                continue
            effective_heal = missing_hp if heal_amount < 0 else min(heal_amount, missing_hp)
            score = float(effective_heal)
            if incoming_damage >= state.hp:
                score += 20.0
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
            target_type = self._text(potion.get("target_type"))
            if "enemy" not in target_type:
                continue
            potion_damage = self._potion_damage_amount(potion)
            if potion_damage <= 0:
                continue
            target = self._pick_damage_target(state.living_enemies(), potion_damage)
            if target is None:
                continue
            target_ehp = self._enemy_effective_hp(target)
            score = float(potion_damage)
            if potion_damage >= target_ehp:
                score += 30.0 + self._enemy_threat(target) * 1.5
            if incoming_damage >= state.hp:
                score += 12.0
            elif hp_ratio < 0.35:
                score += 5.0
            if best_attack_choice is None or score > best_attack_choice[0]:
                best_attack_choice = (score, potion, target)

        if best_attack_choice is not None and (
            incoming_damage >= state.hp
            or hp_ratio < 0.35
            or best_attack_choice[0] >= 35.0
        ):
            args = {"potion_index": best_attack_choice[1].get("index", 0)}
            if "all" not in self._text(best_attack_choice[1].get("target_type")) and best_attack_choice[2] is not None:
                args["target_index"] = best_attack_choice[2].get("index")
            return FlowAction("use_potion", args)

        if best_choice is not None and (incoming_damage >= state.hp or hp_ratio < 0.3):
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
        return self._pick_damage_target(enemies, damage) or enemies[0]

    def _combat_card_score(self, state: GameStateView, card: dict[str, Any]) -> float:
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
        lowest_enemy_ehp = min(self._enemy_effective_hp(enemy) for enemy in enemies)
        block_gap = max(0, incoming_damage - state.block)
        lethal_pressure = block_gap >= state.hp
        can_tank = state.hp > incoming_damage * 2

        score = 0.0
        if damage > 0:
            score += damage * 1.55
            if damage >= lowest_enemy_ehp:
                score += 18.0
            if "vulnerable" in description or "易伤" in description:
                score += 7.0 if state.round in (None, 1, 2) else 3.0
        if block > 0:
            if attack_pressure > 0:
                score += min(block, 12) * (1.35 if hp_ratio < 0.55 else 1.0)
            else:
                score += block * 0.25
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
        if "draw" in description or "抽牌" in description:
            score += 3.0
        if draw_amount > 0:
            score += min(draw_amount, 4) * 3.0
        if "strength" in description or "力量" in description:
            score += 3.5
        if "energy" in description or "能量" in description:
            score += 2.5
        if energy_gain > 0:
            followup_count, spendable_cost = self._playable_followup_pressure(state, card)
            usable_energy = max(0, min(energy_gain, spendable_cost - max(0, state.energy - cost)))
            score += usable_energy * 4.0 + min(followup_count, 4) * 0.8
            if followup_count < 2 or usable_energy <= 0:
                score -= 7.0
        if "weak" in description or "虚弱" in description:
            score += 2.5
        if hp_loss > 0:
            if energy_gain > 0:
                followup_count, spendable_cost = self._playable_followup_pressure(state, card)
                can_convert_hp = followup_count >= 2 and spendable_cost > state.energy
                score -= hp_loss * (2.6 if hp_ratio < 0.45 else 0.8)
                if can_convert_hp and hp_ratio >= 0.45:
                    score += 35.0
                if not can_convert_hp:
                    score -= 8.0
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
        elif card_type == "skill":
            score += 0.5
        elif card_type == "power":
            if lethal_pressure:
                score -= 12.0
            elif incoming_damage > 15 and hp_ratio < 0.55:
                score -= 2.0
            else:
                score += 2.0 if state.round in (None, 1, 2, 3) else -1.0

        if cost <= 0:
            score += 2.0
        else:
            score += min((damage + block * 0.8) / max(cost, 1), 8.0)
            score -= max(0, cost - max(1, state.energy)) * 2.0

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
        choice = choose_card_reward(state.cards, deck, state.can_skip)
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
        if best_potion is not None and state.gold >= self.config.shop_high_gold_threshold:
            return FlowAction("buy_potion", {"potion_index": best_potion.get("index", 0)})

        buy_probability = self.config.shop_buy_probability
        if state.gold >= self.config.shop_high_gold_threshold:
            buy_probability = max(buy_probability, 0.8)
        if rng.random() < buy_probability:
            best_card = best_shop_card(state.cards, state.gold, list(state.player.get("deck") or []))
            if best_card is not None:
                return FlowAction("buy_card", {"card_index": best_card.get("index", 0)})
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
        if floor <= 8:
            heal_threshold = max(heal_threshold, 0.7)
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

        pick = rng.choice(enabled)
        return FlowAction("choose_option", {"option_index": pick.get("index", 0)})

    def _pick_card_select_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        indices = [str(card.get("index", i)) for i, card in enumerate(state.cards)]
        if not indices:
            return FlowAction("skip_select")

        if is_shop_context(state):
            purge_target = choose_purge_target(state.cards)
            if purge_target is not None:
                return FlowAction("select_cards", {"indices": str(purge_target.get("index", 0))})

        upgrade_target = choose_upgrade_target(state.cards)
        if upgrade_target is not None:
            return FlowAction("select_cards", {"indices": str(upgrade_target.get("index", 0))})

        if state.min_select == 0 and rng.random() < self.config.card_select_skip_probability:
            return FlowAction("skip_select")

        upper_bound = max(1, min(state.max_select, len(indices)))
        lower_bound = max(1, state.min_select)
        pick_count = rng.randint(lower_bound, upper_bound)
        picks = rng.sample(indices, k=pick_count)
        return FlowAction("select_cards", {"indices": ",".join(picks)})

    def _pick_bundle_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        if not state.bundles:
            return FlowAction("proceed")
        pick = rng.choice(state.bundles)
        return FlowAction("select_bundle", {"bundle_index": pick.get("index", 0)})
