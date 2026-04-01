"""Observation space definitions for STS2 environment"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any


class ObservationSpace:
    """Observation space for STS2 environment"""

    # State type mapping
    STATE_TYPES = [
        'menu', 'unknown', 'monster', 'elite', 'boss', 'hand_select',
        'rewards', 'card_reward', 'map', 'event', 'rest_site', 'shop',
        'treasure', 'card_select', 'bundle_select', 'relic_select',
        'crystal_sphere', 'overlay'
    ]

    # Card types
    CARD_TYPES = ['Attack', 'Skill', 'Power', 'Status', 'Curse']

    # Card rarities
    CARD_RARITIES = ['Common', 'Uncommon', 'Rare', 'Special']

    # Target types
    TARGET_TYPES = ['None', 'Self', 'AnyEnemy', 'AllEnemies', 'AnyAlly', 'AnyPlayer']

    @staticmethod
    def create_observation_space() -> spaces.Dict:
        """Create the observation space for STS2 environment

        Returns:
            Dictionary observation space
        """
        return spaces.Dict({
            # Game state type
            'state_type': spaces.Discrete(len(ObservationSpace.STATE_TYPES)),

            # Run information
            'act': spaces.Box(low=1, high=10, shape=(1,), dtype=np.int32),
            'floor': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            'ascension': spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),

            # Player basic stats
            'player_hp': spaces.Box(low=0, high=999, shape=(1,), dtype=np.int32),
            'player_max_hp': spaces.Box(low=1, high=999, shape=(1,), dtype=np.int32),
            'player_block': spaces.Box(low=0, high=999, shape=(1,), dtype=np.int32),
            'player_gold': spaces.Box(low=0, high=9999, shape=(1,), dtype=np.int32),

            # Combat stats (only meaningful during combat)
            'player_energy': spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
            'player_max_energy': spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
            'hand_size': spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
            'draw_pile_count': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            'discard_pile_count': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),

            # Hand cards (fixed size for simplicity)
            'hand_cards': spaces.Box(
                low=0, high=1,
                shape=(10, 20),  # 10 cards max, 20 features per card
                dtype=np.float32
            ),

            # Enemies (up to 5 enemies)
            'enemies': spaces.Box(
                low=0, high=1,
                shape=(5, 15),  # 5 enemies max, 15 features per enemy
                dtype=np.float32
            ),

            # Map information
            'current_position': spaces.Box(low=0, high=20, shape=(2,), dtype=np.int32),
            'next_options': spaces.Box(
                low=0, high=1,
                shape=(5, 4),  # 5 options max, 4 features per option
                dtype=np.float32
            ),

            # Rewards
            'reward_items': spaces.Box(
                low=0, high=1,
                shape=(5, 3),  # 5 items max, 3 features per item
                dtype=np.float32
            ),

            # Shop items
            'shop_items': spaces.Box(
                low=0, high=1,
                shape=(15, 4),  # 15 items max, 4 features per item
                dtype=np.float32
            ),

            # Event options
            'event_options': spaces.Box(
                low=0, high=1,
                shape=(5, 2),  # 5 options max, 2 features per option
                dtype=np.float32
            ),

            # Rest site options
            'rest_options': spaces.Box(
                low=0, high=1,
                shape=(3, 2),  # 3 options max, 2 features per option
                dtype=np.float32
            ),

            # Card selection
            'card_select_cards': spaces.Box(
                low=0, high=1,
                shape=(10, 15),  # 10 cards max, 15 features per card
                dtype=np.float32
            ),

            # Relic selection
            'relic_select_relics': spaces.Box(
                low=0, high=1,
                shape=(3, 5),  # 3 relics max, 5 features per relic
                dtype=np.float32
            ),
        })

    @staticmethod
    def process_state(state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Process raw game state into observation

        Args:
            state: Raw game state from API

        Returns:
            Processed observation dictionary
        """
        obs = {}

        # State type mapping from STS2-Agent to our internal types
        screen_to_state_type = {
            'MAIN_MENU': 'menu',
            'MAP': 'map',
            'COMBAT': 'monster',
            'EVENT': 'event',
            'SHOP': 'shop',
            'REST': 'rest_site',
            'CHEST': 'treasure',
            'REWARD': 'rewards',
            'CHARACTER_SELECT': 'menu',
            'TIMELINE': 'menu',
            'GAME_OVER': 'overlay',
            'unknown': 'unknown'
        }

        # State type
        screen = state.get('screen', 'unknown')
        state_type = screen_to_state_type.get(screen, 'unknown')
        obs['state_type'] = ObservationSpace.STATE_TYPES.index(state_type) if state_type in ObservationSpace.STATE_TYPES else 1

        # Run information
        run = state.get('run', {})
        obs['act'] = np.array([run.get('act', 1)], dtype=np.int32)
        obs['floor'] = np.array([run.get('floor', 0)], dtype=np.int32)
        obs['ascension'] = np.array([run.get('ascension', 0)], dtype=np.int32)

        # Player stats
        player = state.get('run', {}).get('player', {})
        obs['player_hp'] = np.array([player.get('hp', 0)], dtype=np.int32)
        obs['player_max_hp'] = np.array([player.get('max_hp', 1)], dtype=np.int32)
        obs['player_block'] = np.array([player.get('block', 0)], dtype=np.int32)
        obs['player_gold'] = np.array([player.get('gold', 0)], dtype=np.int32)
        obs['player_energy'] = np.array([player.get('energy', 0)], dtype=np.int32)
        obs['player_max_energy'] = np.array([player.get('max_energy', 3)], dtype=np.int32)

        # Hand cards
        hand = player.get('hand', [])
        obs['hand_cards'] = ObservationSpace._process_cards(hand, max_cards=10, features_per_card=20)

        # Enemies
        combat = state.get('battle', {})
        enemies = combat.get('enemies', [])
        obs['enemies'] = ObservationSpace._process_enemies(enemies, max_enemies=5)

        # Hand size and pile counts
        obs['hand_size'] = np.array([len(hand)], dtype=np.int32)
        obs['draw_pile_count'] = np.array([player.get('draw_pile_count', 0)], dtype=np.int32)
        obs['discard_pile_count'] = np.array([player.get('discard_pile_count', 0)], dtype=np.int32)

        # Map
        map_data = state.get('map', {})
        if map_data is not None and isinstance(map_data, dict):
            current_pos = map_data.get('current_position', {})
            obs['current_position'] = np.array([
                current_pos.get('col', 0),
                current_pos.get('row', 0)
            ], dtype=np.int32)
            next_options = map_data.get('nodes', [])
            obs['next_options'] = ObservationSpace._process_map_options(next_options, max_options=5)
        else:
            obs['current_position'] = np.array([0, 0], dtype=np.int32)
            obs['next_options'] = ObservationSpace._process_map_options([], max_options=5)

        # Rewards
        rewards = state.get('reward', {})
        if rewards is not None and isinstance(rewards, dict):
            reward_items = rewards.get('items', [])
            obs['reward_items'] = ObservationSpace._process_rewards(reward_items, max_items=5)
        else:
            obs['reward_items'] = ObservationSpace._process_rewards([], max_items=5)

        # Shop
        shop = state.get('shop', {})
        if shop is not None and isinstance(shop, dict):
            shop_items = shop.get('items', [])
            obs['shop_items'] = ObservationSpace._process_shop_items(shop_items, max_items=15)
        else:
            obs['shop_items'] = ObservationSpace._process_shop_items([], max_items=15)

        # Event
        event = state.get('event', {})
        if event is not None and isinstance(event, dict):
            event_options = event.get('options', [])
            obs['event_options'] = ObservationSpace._process_event_options(event_options, max_options=5)
        else:
            obs['event_options'] = ObservationSpace._process_event_options([], max_options=5)

        # Rest site
        rest = state.get('rest', {})
        if rest is not None and isinstance(rest, dict):
            rest_options = rest.get('options', [])
            obs['rest_options'] = ObservationSpace._process_rest_options(rest_options, max_options=3)
        else:
            obs['rest_options'] = ObservationSpace._process_rest_options([], max_options=3)

        # Card selection
        card_select = state.get('selection', {})
        if card_select is not None and isinstance(card_select, dict):
            card_select_cards = card_select.get('cards', [])
            obs['card_select_cards'] = ObservationSpace._process_cards(card_select_cards, max_cards=10, features_per_card=15)
        else:
            obs['card_select_cards'] = ObservationSpace._process_cards([], max_cards=10, features_per_card=15)

        # Relic selection
        relic_select = state.get('chest', {})
        if relic_select is not None and isinstance(relic_select, dict):
            relic_select_relics = relic_select.get('relics', [])
            obs['relic_select_relics'] = ObservationSpace._process_relics(relic_select_relics, max_relics=3)
        else:
            obs['relic_select_relics'] = ObservationSpace._process_relics([], max_relics=3)

        return obs

    @staticmethod
    def _process_cards(cards: list, max_cards: int, features_per_card: int) -> np.ndarray:
        """Process card list into fixed-size array"""
        card_array = np.zeros((max_cards, features_per_card), dtype=np.float32)
        for i, card in enumerate(cards[:max_cards]):
            card_array[i, 0] = ObservationSpace.CARD_TYPES.index(card.get('type', 'Attack')) if card.get('type') in ObservationSpace.CARD_TYPES else 0
            card_array[i, 1] = float(card.get('cost', 0)) if card.get('cost') != 'X' else 10.0
            card_array[i, 2] = 1.0 if card.get('can_play', False) else 0.0
            card_array[i, 3] = 1.0 if card.get('is_upgraded', False) else 0.0
            # Add more features as needed
        return card_array

    @staticmethod
    def _process_enemies(enemies: list, max_enemies: int) -> np.ndarray:
        """Process enemy list into fixed-size array"""
        enemy_array = np.zeros((max_enemies, 15), dtype=np.float32)
        for i, enemy in enumerate(enemies[:max_enemies]):
            enemy_array[i, 0] = enemy.get('hp', 0) / 100.0  # Normalize
            enemy_array[i, 1] = enemy.get('max_hp', 1) / 100.0  # Normalize
            enemy_array[i, 2] = enemy.get('block', 0) / 100.0  # Normalize
            # Add more features like intents, status effects, etc.
        return enemy_array

    @staticmethod
    def _process_map_options(options: list, max_options: int) -> np.ndarray:
        """Process map options into fixed-size array"""
        option_array = np.zeros((max_options, 4), dtype=np.float32)
        for i, option in enumerate(options[:max_options]):
            option_array[i, 0] = option.get('col', 0) / 20.0  # Normalize
            option_array[i, 1] = option.get('row', 0) / 20.0  # Normalize
            # Add more features like node type
        return option_array

    @staticmethod
    def _process_rewards(rewards: list, max_items: int) -> np.ndarray:
        """Process reward items into fixed-size array"""
        reward_array = np.zeros((max_items, 3), dtype=np.float32)
        for i, reward in enumerate(rewards[:max_items]):
            reward_type = reward.get('type', 'gold')
            reward_array[i, 0] = 1.0 if reward_type == 'gold' else 0.0
            reward_array[i, 1] = 1.0 if reward_type == 'card' else 0.0
            reward_array[i, 2] = 1.0 if reward_type == 'relic' else 0.0
        return reward_array

    @staticmethod
    def _process_shop_items(items: list, max_items: int) -> np.ndarray:
        """Process shop items into fixed-size array"""
        item_array = np.zeros((max_items, 4), dtype=np.float32)
        for i, item in enumerate(items[:max_items]):
            item_array[i, 0] = item.get('cost', 0) / 200.0  # Normalize
            item_array[i, 1] = 1.0 if item.get('can_afford', False) else 0.0
            item_array[i, 2] = 1.0 if item.get('is_stocked', False) else 0.0
            # Add category info
        return item_array

    @staticmethod
    def _process_event_options(options: list, max_options: int) -> np.ndarray:
        """Process event options into fixed-size array"""
        option_array = np.zeros((max_options, 2), dtype=np.float32)
        for i, option in enumerate(options[:max_options]):
            option_array[i, 0] = 1.0 if not option.get('is_locked', True) else 0.0
            option_array[i, 1] = 1.0 if option.get('is_proceed', False) else 0.0
        return option_array

    @staticmethod
    def _process_rest_options(options: list, max_options: int) -> np.ndarray:
        """Process rest site options into fixed-size array"""
        option_array = np.zeros((max_options, 2), dtype=np.float32)
        for i, option in enumerate(options[:max_options]):
            option_array[i, 0] = 1.0 if option.get('is_enabled', False) else 0.0
            # Add option type info
        return option_array

    @staticmethod
    def _process_relics(relics: list, max_relics: int) -> np.ndarray:
        """Process relic list into fixed-size array"""
        relic_array = np.zeros((max_relics, 5), dtype=np.float32)
        for i, relic in enumerate(relics[:max_relics]):
            # Add relic features
            relic_array[i, 0] = 1.0  # Present
        return relic_array