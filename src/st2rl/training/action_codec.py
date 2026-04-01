# -*- coding: utf-8 -*-
"""RL action encoding/decoding for the canonical HTTP CLI environment."""

from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from st2rl.gameplay.types import FlowAction

ACTION_NAMES = [
    "proceed",
    "end_turn",
    "play_card",
    "use_potion",
    "select_map_choice",
    "choose_card_reward",
    "skip_reward",
    "choose_option",
    "buy_card",
    "buy_relic",
    "buy_potion",
    "purge_card",
    "leave_shop",
    "select_bundle",
    "select_cards",
    "skip_select",
]

ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}


def create_action_space() -> gym.Space:
    """Create the PPO-friendly discrete action space."""
    return spaces.MultiDiscrete(
        [
            len(ACTION_NAMES),  # action type
            16,  # slot 1 / primary index
            16,  # slot 2
            16,  # slot 3
            16,  # slot 4
            16,  # slot 5
            16,  # slot 6
            16,  # slot 7
            16,  # slot 8
        ]
    )


def decode_action(action: np.ndarray) -> FlowAction:
    """Decode MultiDiscrete action into canonical action."""
    action_type = ACTION_NAMES[int(action[0])]
    slots = [int(value) for value in action[1:9]]
    primary = slots[0]
    secondary = slots[1]

    if action_type == "play_card":
        args: Dict[str, int] = {"card_index": primary}
        if secondary < 15:
            args["target_index"] = secondary
        return FlowAction("play_card", args)

    if action_type == "use_potion":
        args = {"potion_index": primary}
        if secondary < 15:
            args["target_index"] = secondary
        return FlowAction("use_potion", args)

    if action_type == "select_map_choice":
        return FlowAction("select_map_node", {"choice_index": primary})

    if action_type == "choose_card_reward":
        return FlowAction("choose_card_reward", {"card_index": primary})

    if action_type == "choose_option":
        return FlowAction("choose_option", {"option_index": primary})

    if action_type == "buy_card":
        return FlowAction("buy_card", {"card_index": primary})

    if action_type == "buy_relic":
        return FlowAction("buy_relic", {"relic_index": primary})

    if action_type == "buy_potion":
        return FlowAction("buy_potion", {"potion_index": primary})

    if action_type == "purge_card":
        return FlowAction("purge_card", {"card_index": primary})

    if action_type == "select_bundle":
        return FlowAction("select_bundle", {"bundle_index": primary})

    if action_type == "select_cards":
        picks: list[str] = []
        seen_slots: set[int] = set()
        for slot in slots:
            if slot in seen_slots:
                continue
            seen_slots.add(slot)
            if 0 <= slot < 16:
                picks.append(str(slot))
        return FlowAction("select_cards", {"indices": ",".join(picks)})

    return FlowAction(action_type)
