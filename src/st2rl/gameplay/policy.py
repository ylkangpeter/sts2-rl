# -*- coding: utf-8 -*-
"""Decision policy split by gameplay stage."""

import random

from st2rl.gameplay.config import FlowPolicyConfig
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
        hp_ratio = state.hp / max(1, state.max_hp)
        weighted_choices: list[dict] = []
        for choice in choices:
            room_type = str(choice.get("room_type") or choice.get("type") or "").lower()
            weight = 1.0
            if "elite" in room_type:
                weight = 2.6 if hp_ratio >= 0.6 else 0.55
            elif "shop" in room_type:
                if state.gold >= self.config.shop_high_gold_threshold:
                    weight = 2.1
                elif state.gold < self.config.shop_low_gold_threshold:
                    weight = 0.35
            elif "rest" in room_type:
                weight = 1.8 if hp_ratio < self.config.rest_heal_threshold else 0.85
            elif "treasure" in room_type:
                weight = 1.4
            weighted_choices.extend([choice] * max(1, int(round(weight * 4))))
        pick = rng.choice(weighted_choices or choices)
        return FlowAction("select_map_node", {"col": pick["col"], "row": pick["row"]})

    def _pick_combat_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        enemies = state.living_enemies()
        if not enemies:
            return FlowAction("proceed")

        playable = state.playable_cards()
        if not playable:
            return FlowAction("end_turn")

        card = rng.choice(playable)
        args = {"card_index": card["index"]}
        if card_needs_enemy_target(card):
            args["target_index"] = rng.choice(enemies)["index"]
        return FlowAction("play_card", args)

    def _pick_card_reward_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        if state.cards and (not state.can_skip or rng.random() < self.config.card_reward_take_probability):
            card = rng.choice(state.cards)
            return FlowAction("choose_card_reward", {"card_index": card.get("index", 0)})
        return FlowAction("skip_reward")

    def _pick_shop_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        if state.gold < self.config.shop_low_gold_threshold:
            return FlowAction("leave_shop")

        candidates: list[FlowAction] = []
        for card in state.cards:
            if card.get("cost", 9999) <= state.gold:
                candidates.append(FlowAction("buy_card", {"card_index": card.get("index", 0)}))
        for relic in state.relics:
            if relic.get("cost", 9999) <= state.gold:
                candidates.extend([FlowAction("buy_relic", {"relic_index": relic.get("index", 0)})] * 3)
        for potion in state.potions:
            if potion.get("cost", 9999) <= state.gold:
                candidates.append(FlowAction("buy_potion", {"potion_index": potion.get("index", 0)}))
        purge_cost = int(state.raw.get("purge_cost") or 999999)
        if state.gold >= purge_cost:
            for card in state.player.get("deck") or []:
                if isinstance(card, dict) and card.get("index") is not None:
                    candidates.append(FlowAction("purge_card", {"card_index": card.get("index", 0)}))

        buy_probability = self.config.shop_buy_probability
        if state.gold >= self.config.shop_high_gold_threshold:
            buy_probability = max(buy_probability, 0.8)
        if candidates and rng.random() < buy_probability:
            return rng.choice(candidates)
        return FlowAction("leave_shop")

    def _pick_event_action(self, state: GameStateView, rng: random.Random) -> FlowAction:
        options = state.options or state.choices
        unlocked = [option for option in options if not option.get("is_locked")]
        if not unlocked:
            return FlowAction("proceed")
        pick = rng.choice(unlocked)
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
        if hp_ratio < self.config.rest_heal_threshold:
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
