"""Helper functions for STS2 environment"""

import numpy as np
from typing import Dict, Any, List

from st2rl.spaces import ActionSpace


def get_valid_actions(state: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
    """Get list of valid actions for current state

    Args:
        state: Current game state

    Returns:
        List of valid action dictionaries
    """
    return ActionSpace.get_valid_actions(state)


def mask_invalid_actions(
    action_values: np.ndarray,
    valid_actions: List[Dict[str, np.ndarray]]
) -> np.ndarray:
    """Mask invalid actions in action value array

    Args:
        action_values: Array of action values
        valid_actions: List of valid action dictionaries

    Returns:
        Masked action values with invalid actions set to -inf
    """
    if len(valid_actions) == 0:
        return np.full_like(action_values, -np.inf)

    # This is a simplified version - in practice, you'd need to map
    # valid actions to their indices in the action space
    return action_values


def normalize_observation(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalize observation values

    Args:
        obs: Observation dictionary

    Returns:
        Normalized observation dictionary
    """
    normalized = obs.copy()

    # Normalize HP values to [0, 1]
    if 'player_hp' in normalized and 'player_max_hp' in normalized:
        max_hp = normalized['player_max_hp']
        if max_hp > 0:
            normalized['player_hp'] = normalized['player_hp'] / max_hp

    # Normalize gold to [0, 1] (assuming max gold is around 1000)
    if 'player_gold' in normalized:
        normalized['player_gold'] = np.clip(normalized['player_gold'] / 1000.0, 0, 1)

    # Normalize energy
    if 'player_energy' in normalized and 'player_max_energy' in normalized:
        max_energy = normalized['player_max_energy']
        if max_energy > 0:
            normalized['player_energy'] = normalized['player_energy'] / max_energy

    return normalized


def calculate_action_mask(state: Dict[str, Any]) -> np.ndarray:
    """Calculate binary mask for valid actions

    Args:
        state: Current game state

    Returns:
        Binary mask where 1 indicates valid action
    """
    valid_actions = get_valid_actions(state)

    # Create mask based on action types
    mask = np.zeros(len(ActionSpace.ACTION_TYPES), dtype=np.int32)

    for action in valid_actions:
        action_type = action['action_type']
        mask[action_type] = 1

    return mask