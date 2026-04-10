# -*- coding: utf-8 -*-
"""Decision policy split by gameplay stage."""

import random

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

        playable = state.playable_cards()
        if not playable:
            return FlowAction("end_turn")

        card = rng.choice(playable)
        args = {"card_index": card["index"]}
        if card_needs_enemy_target(card):
            args["target_index"] = rng.choice(enemies)["index"]
        return FlowAction("play_card", args)

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
        if hp_ratio < max(0.4, self.config.rest_heal_threshold - 0.12):
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
