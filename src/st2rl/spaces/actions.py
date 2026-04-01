"""Action space definitions for STS2 environment"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, List


class ActionSpace:
    """Action space for STS2 environment"""

    # Action types (matching STS2-Agent)
    ACTION_TYPES = [
        'play_card', 'use_potion', 'end_turn',
        'continue_run', 'abandon_run', 'open_character_select',
        'open_timeline', 'close_main_menu_submenu', 'choose_timeline_epoch',
        'confirm_timeline_overlay', 'choose_map_node', 'collect_rewards_and_proceed',
        'claim_reward', 'choose_reward_card', 'skip_reward_cards',
        'select_deck_card', 'confirm_selection', 'proceed',
        'open_chest', 'choose_treasure_relic', 'choose_event_option',
        'choose_rest_option', 'open_shop_inventory', 'close_shop_inventory',
        'buy_card', 'buy_relic', 'buy_potion',
        'remove_card_at_shop', 'select_character', 'embark',
        'unready', 'increase_ascension', 'decrease_ascension',
        'discard_potion', 'run_console_command', 'confirm_modal',
        'dismiss_modal', 'return_to_main_menu'
    ]

    @staticmethod
    def create_action_space() -> spaces.MultiDiscrete:
        """Create the action space for STS2 environment

        Returns:
            MultiDiscrete action space
        """
        # Action space dimensions:
        # 0: action_type (0-38)
        # 1: card_index (0-9, -1 mapped to 10)
        # 2: target_index (0-4, -1 mapped to 5)
        # 3: option_index (0-4, -1 mapped to 5)
        # 4: potion_index (0-4, -1 mapped to 5)
        return spaces.MultiDiscrete([
            len(ActionSpace.ACTION_TYPES),  # action_type
            11,  # card_index (0-9 + 10 for -1)
            6,   # target_index (0-4 + 5 for -1)
            6,   # option_index (0-4 + 5 for -1)
            6    # potion_index (0-4 + 5 for -1)
        ])

    @staticmethod
    def get_valid_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get list of valid actions for current state

        Args:
            state: Current game state

        Returns:
            List of valid action dictionaries
        """
        screen = state.get('screen', 'unknown')
        valid_actions = []

        if screen == 'COMBAT':
            valid_actions.extend(ActionSpace._get_combat_actions(state))
        elif screen == 'REWARD':
            valid_actions.extend(ActionSpace._get_reward_actions(state))
        elif screen == 'MAP':
            valid_actions.extend(ActionSpace._get_map_actions(state))
        elif screen == 'EVENT':
            valid_actions.extend(ActionSpace._get_event_actions(state))
        elif screen == 'REST':
            valid_actions.extend(ActionSpace._get_rest_actions(state))
        elif screen == 'SHOP':
            valid_actions.extend(ActionSpace._get_shop_actions(state))
        elif screen == 'CHEST':
            valid_actions.extend(ActionSpace._get_chest_actions(state))
        elif screen == 'MAIN_MENU':
            valid_actions.extend(ActionSpace._get_main_menu_actions(state))
        elif screen == 'CHARACTER_SELECT':
            valid_actions.extend(ActionSpace._get_character_select_actions(state))
        elif screen == 'TIMELINE':
            valid_actions.extend(ActionSpace._get_timeline_actions(state))

        return valid_actions

    @staticmethod
    def _get_combat_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get valid combat actions"""
        actions = []
        player = state.get('run', {}).get('player', {})
        battle = state.get('battle', {})

        # Play card actions
        hand = player.get('hand', [])
        for i, card in enumerate(hand):
            actions.append({
                'action_type': 'play_card',
                'card_index': i,
                'target_index': -1,
                'option_index': -1,
                'potion_index': -1
            })

        # Use potion actions
        potions = player.get('potions', [])
        for i, potion in enumerate(potions):
            actions.append({
                'action_type': 'use_potion',
                'card_index': -1,
                'target_index': -1,
                'option_index': i,
                'potion_index': -1
            })

        # End turn action
        actions.append({
            'action_type': 'end_turn',
            'card_index': -1,
            'target_index': -1,
            'option_index': -1,
            'potion_index': -1
        })

        return actions

    @staticmethod
    def _get_reward_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get valid reward actions"""
        actions = []

        # Collect rewards and proceed
        actions.append({
            'action_type': 'collect_rewards_and_proceed',
            'card_index': -1,
            'target_index': -1,
            'option_index': -1,
            'potion_index': -1
        })

        return actions

    @staticmethod
    def _get_map_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get valid map actions"""
        actions = []
        map_data = state.get('map', {})
        
        # Try to get available nodes from available_nodes field first
        available_nodes = map_data.get('available_nodes', [])
        if available_nodes:
            for i, node in enumerate(available_nodes):
                actions.append({
                    'action_type': 'choose_map_node',
                    'card_index': -1,
                    'target_index': -1,
                    'option_index': i,
                    'potion_index': -1
                })
        else:
            # Fallback to nodes field
            nodes = map_data.get('nodes', [])
            for i, node in enumerate(nodes):
                if node.get('is_available', False):
                    actions.append({
                        'action_type': 'choose_map_node',
                        'card_index': -1,
                        'target_index': -1,
                        'option_index': i,
                        'potion_index': -1
                    })

        return actions

    @staticmethod
    def _get_event_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get valid event actions"""
        actions = []
        event = state.get('event', {})
        options = event.get('options', [])

        for i, option in enumerate(options):
            actions.append({
                'action_type': 'choose_event_option',
                'card_index': -1,
                'target_index': -1,
                'option_index': i,
                'potion_index': -1
            })

        return actions

    @staticmethod
    def _get_rest_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get valid rest actions"""
        actions = []

        # Choose rest option
        actions.append({
            'action_type': 'choose_rest_option',
            'card_index': -1,
            'target_index': -1,
            'option_index': 0,  # Default to first option
            'potion_index': -1
        })

        return actions

    @staticmethod
    def _get_shop_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get valid shop actions"""
        actions = []

        # Proceed from shop
        actions.append({
            'action_type': 'proceed',
            'card_index': -1,
            'target_index': -1,
            'option_index': -1,
            'potion_index': -1
        })

        return actions

    @staticmethod
    def _get_chest_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get valid chest actions"""
        actions = []

        # Open chest
        actions.append({
            'action_type': 'open_chest',
            'card_index': -1,
            'target_index': -1,
            'option_index': -1,
            'potion_index': -1
        })

        return actions

    @staticmethod
    def _get_main_menu_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get valid main menu actions"""
        actions = []

        # Open character select
        actions.append({
            'action_type': 'open_character_select',
            'card_index': -1,
            'target_index': -1,
            'option_index': -1,
            'potion_index': -1
        })

        return actions

    @staticmethod
    def _get_character_select_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get valid character select actions"""
        actions = []

        # Select character
        for i in range(4):  # Assuming 4 characters
            actions.append({
                'action_type': 'select_character',
                'card_index': -1,
                'target_index': -1,
                'option_index': i,
                'potion_index': -1
            })

        return actions

    @staticmethod
    def _get_timeline_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get valid timeline actions"""
        actions = []

        # Choose timeline epoch
        actions.append({
            'action_type': 'choose_timeline_epoch',
            'card_index': -1,
            'target_index': -1,
            'option_index': 0,  # Default to first epoch
            'potion_index': -1
        })

        return actions

    @staticmethod
    def action_to_api_call(action: np.ndarray, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert action array to API call parameters

        Args:
            action: Action array from MultiDiscrete action space
            state: Current game state

        Returns:
            API call parameters dictionary
        """
        # Map back from MultiDiscrete indices to actual values
        def map_back(value, offset):
            # Convert numpy types to Python native types
            value = int(value)
            return value if value < offset else -1

        action_type_idx = int(action[0])
        action_type = ActionSpace.ACTION_TYPES[action_type_idx]
        api_call = {'action': action_type}

        card_index = map_back(action[1], 10)
        target_index = map_back(action[2], 5)
        option_index = map_back(action[3], 5)
        potion_index = map_back(action[4], 5)

        if action_type == 'play_card':
            if card_index >= 0:
                api_call['card_index'] = card_index
                if target_index >= 0:
                    api_call['target_index'] = target_index

        elif action_type == 'use_potion':
            if potion_index >= 0:
                api_call['option_index'] = potion_index
                if target_index >= 0:
                    api_call['target_index'] = target_index

        elif action_type == 'choose_map_node':
            if option_index >= 0:
                api_call['option_index'] = option_index

        elif action_type == 'choose_event_option':
            if option_index >= 0:
                api_call['option_index'] = option_index

        elif action_type == 'choose_rest_option':
            if option_index >= 0:
                api_call['option_index'] = option_index

        elif action_type == 'claim_reward':
            if option_index >= 0:
                api_call['option_index'] = option_index

        elif action_type == 'choose_reward_card':
            if option_index >= 0:
                api_call['option_index'] = option_index

        elif action_type == 'select_character':
            if option_index >= 0:
                api_call['option_index'] = option_index

        elif action_type == 'choose_timeline_epoch':
            if option_index >= 0:
                api_call['option_index'] = option_index

        elif action_type == 'buy_card':
            if option_index >= 0:
                api_call['option_index'] = option_index

        elif action_type == 'buy_relic':
            if option_index >= 0:
                api_call['option_index'] = option_index

        elif action_type == 'buy_potion':
            if option_index >= 0:
                api_call['option_index'] = option_index

        elif action_type == 'discard_potion':
            if option_index >= 0:
                api_call['option_index'] = option_index

        return api_call