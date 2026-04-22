"""Main STS2 Gymnasium environment"""

import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional

from st2rl.client import Sts2Client
from st2rl.spaces import ObservationSpace, ActionSpace
from st2rl.utils.logger import get_run_logger


class STS2Env(gym.Env):
    """Slay the Spire 2 environment for reinforcement learning

    This environment connects to the STS2-Agent mod and provides a Gymnasium-compatible
    interface for training RL agents.
    """

    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8080,
        reward_scale: float = 1.0,
        max_steps: int = 10000,
        render_mode: Optional[str] = None,
        character_index: int = 0
    ):
        """Initialize STS2 environment

        Args:
            host: STS2-Agent server host
            port: STS2-Agent server port
            reward_scale: Scaling factor for rewards
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ('human' or None)
            character_index: Index of the character to select (0-based)
        """
        super().__init__()

        self.client = Sts2Client(host=host, port=port)
        self.reward_scale = reward_scale
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.character_index = character_index

        # Define observation and action spaces
        self.observation_space = ObservationSpace.create_observation_space()
        self.action_space = ActionSpace.create_action_space()

        # Initialize state
        self.current_state: Dict[str, Any] = {}
        self.previous_state: Optional[Dict[str, Any]] = None
        self.previous_hp = 0
        self.previous_gold = 0
        self.previous_floor = 0
        self.steps = 0
        self.step_count: int = 0
        self.episode_reward: float = 0.0

        # Logger
        self.logger = get_run_logger()

        # Reward tracking
        self.previous_hp: int = 0
        self.previous_gold: int = 0
        self.previous_floor: int = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Wait for connection if needed
        if not self.client.is_connected():
            self.logger.info("Waiting for STS2 game to start...")
            if not self.client.wait_for_connection():
                self.logger.error("Could not connect to STS2 game. Make sure the game is running with STS2-Agent mod enabled.")
                raise RuntimeError("Could not connect to STS2 game. Make sure the game is running with STS2-Agent mod enabled.")

        # Get initial state
        state = self.client.get_state()
        self.current_state = state if state is not None and isinstance(state, dict) else {}
        self.previous_state = None
        self.step_count = 0
        self.episode_reward = 0.0

        # Handle initial state and navigate to game start
        self._navigate_to_game_start()

        # Get updated state after navigation
        state = self.client.get_state()
        self.logger.info(f"Got state: {state}")
        self.current_state = state if state is not None and isinstance(state, dict) else {}

        # Ensure current_state is not None
        if self.current_state is None:
            self.logger.error("current_state is None, setting to empty dict")
            self.current_state = {}
        # Ensure current_state is a dict
        if not isinstance(self.current_state, dict):
            self.logger.error(f"current_state is not a dict: {type(self.current_state)}, setting to empty dict")
            self.current_state = {}

        # Initialize reward tracking
        try:
            player = self.current_state.get('run', {}).get('player', {})
            self.previous_hp = player.get('hp', 0)
            self.previous_gold = player.get('gold', 0)
            run = self.current_state.get('run', {})
            self.previous_floor = run.get('floor', 0)
        except Exception as e:
            self.logger.error(f"Error initializing reward tracking: {e}")
            # Set default values
            self.previous_hp = 0
            self.previous_gold = 0
            self.previous_floor = 0

        # Process observation
        obs = ObservationSpace.process_state(self.current_state)

        # Info dictionary
        info = {
            'state_type': self.current_state.get('screen', 'unknown'),
            'is_menu': self.current_state.get('screen') == 'MAIN_MENU',
            'step_count': self.step_count
        }

        return obs, info

    def _navigate_to_game_start(self):
        """Navigate from main menu to game start"""
        # Ensure current_state is not None
        if self.current_state is None:
            self.current_state = {}
        # Ensure current_state is a dict
        if not isinstance(self.current_state, dict):
            self.current_state = {}
            
        # Get current screen
        try:
            current_screen = self.current_state.get('screen', '')
        except Exception as e:
            self.logger.error(f"Error getting current screen: {e}")
            current_screen = ''
        
        # Handle different initial screens
        if current_screen == 'MAIN_MENU':
            # From main menu, open character select
            self.logger.info("Navigating from main menu to character select...")
            try:
                self.client.open_character_select()
                state = self.client.get_state()
                if state is not None and isinstance(state, dict):
                    self.current_state = state
                    current_screen = self.current_state.get('screen', '')
            except Exception as e:
                self.logger.error(f"Error opening character select: {e}")
                return
        
        if current_screen == 'CHARACTER_SELECT':
            # Select character
            self.logger.info(f"Selecting character at index {self.character_index}...")
            try:
                self.client.select_character(self.character_index)
                state = self.client.get_state()
                if state is not None and isinstance(state, dict):
                    self.current_state = state
                    current_screen = self.current_state.get('screen', '')
            except Exception as e:
                self.logger.error(f"Error selecting character: {e}")
                return
        
        if current_screen == 'TIMELINE':
            # Choose timeline epoch (default to first)
            self.logger.info("Choosing timeline epoch...")
            try:
                self.client.choose_timeline_epoch(0)
                state = self.client.get_state()
                if state is not None and isinstance(state, dict):
                    self.current_state = state
                    current_screen = self.current_state.get('screen', '')
            except Exception as e:
                self.logger.error(f"Error choosing timeline epoch: {e}")
                return
        
        if current_screen == 'TIMELINE':
            # Confirm timeline overlay
            self.logger.info("Confirming timeline overlay...")
            try:
                self.client.confirm_timeline_overlay()
                state = self.client.get_state()
                if state is not None and isinstance(state, dict):
                    self.current_state = state
                    current_screen = self.current_state.get('screen', '')
            except Exception as e:
                self.logger.error(f"Error confirming timeline overlay: {e}")
                return
        
        if current_screen == 'MAIN_MENU' or current_screen == 'CHARACTER_SELECT':
            # Embark on new run
            self.logger.info("Embarking on new run...")
            try:
                self.client.embark()
                state = self.client.get_state()
                if state is not None and isinstance(state, dict):
                    self.current_state = state
            except Exception as e:
                self.logger.error(f"Error embarking on new run: {e}")
                return

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment

        Args:
            action: Action array from MultiDiscrete space

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Store previous state for reward calculation
        self.previous_state = self.current_state.copy() if self.current_state else None

        # Convert action to API call
        api_call = ActionSpace.action_to_api_call(action, self.current_state)

        # Execute action
        try:
            # Check if we need to handle special state transitions
            needs_special = self._needs_special_transition()
            self.logger.info(f"Step {self.step_count}: Current screen={self.current_state.get('screen', 'unknown')}, needs_special={needs_special}")
            
            if needs_special:
                self.logger.info("Handling special transition...")
                self._handle_special_transition()
            else:
                # Execute regular action
                self.logger.info(f"Executing regular action: {api_call}")
                response = self.client.execute_action(**api_call)
        except Exception as e:
            # Action failed, give penalty
            self.logger.error(f"Error executing action: {e}")
            reward = -0.1 * self.reward_scale
            terminated = False
            truncated = False
            info = {
                'error': str(e),
                'state_type': self.current_state.get('screen', 'unknown'),
                'step_count': self.step_count
            }
            return ObservationSpace.process_state(self.current_state), reward, terminated, truncated, info

        # Get new state
        self.current_state = self.client.get_state()
        self.step_count += 1

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps

        # If terminated, return to main menu for next episode
        if terminated:
            self._return_to_main_menu()

        # Process observation
        obs = ObservationSpace.process_state(self.current_state)

        # Update reward tracking
        player = self.current_state.get('run', {}).get('player', {})
        self.previous_hp = player.get('hp', 0)
        self.previous_gold = player.get('gold', 0)
        run = self.current_state.get('run', {})
        self.previous_floor = run.get('floor', 0)

        self.episode_reward += reward

        # Info dictionary
        info = {
            'state_type': self.current_state.get('screen', 'unknown'),
            'is_menu': self.current_state.get('screen') == 'MAIN_MENU',
            'step_count': self.step_count,
            'episode_reward': self.episode_reward
        }

        # Render if needed
        if self.render_mode == 'human':
            self._render()

        return obs, reward, terminated, truncated, info

    def _needs_special_transition(self) -> bool:
        """Check if a special state transition is needed"""
        if self.current_state is None:
            return False
        current_screen = self.current_state.get('screen', '')
        self.logger.debug(f"Current screen: {current_screen}, needs special transition: {current_screen in ['GAME_OVER', 'REWARD', 'EVENT', 'SHOP', 'REST', 'CHEST', 'CARD_SELECTION']}")
        return current_screen in ['GAME_OVER', 'REWARD', 'EVENT', 'SHOP', 'REST', 'CHEST', 'CARD_SELECTION']

    def _handle_special_transition(self):
        """Handle special state transitions"""
        current_screen = self.current_state.get('screen', '')
        
        if current_screen == 'GAME_OVER':
            # Handle game over - return to main menu
            self.logger.info("Game over, returning to main menu...")
            self.client.return_to_main_menu()
            # Update state
            self.current_state = self.client.get_state()
        elif current_screen == 'REWARD':
            # Handle reward selection
            self.logger.info("Handling reward selection...")
            # Check if we need to choose a card
            reward = self.current_state.get('reward', {})
            if reward and isinstance(reward, dict):
                items = reward.get('items', [])
                if items:
                    option_index = 0
                    self.logger.info(f"Choosing reward card at index {option_index}...")
                    self.client.choose_reward_card(option_index)
                else:
                    # Collect rewards and proceed
                    self.logger.info("Collecting rewards...")
                    self.client.collect_rewards_and_proceed()
            else:
                # Collect rewards and proceed
                self.logger.info("Collecting rewards...")
                self.client.collect_rewards_and_proceed()
            # Update state
            self.current_state = self.client.get_state()
        elif current_screen == 'EVENT':
            # Choose random event option
            event = self.current_state.get('event', {})
            if event and isinstance(event, dict):
                options = event.get('options', [])
                if options:
                    option_index = 0
                    self.logger.info(f"Choosing event option at index {option_index}...")
                    self.client.choose_event_option(option_index)
                else:
                    self.logger.info("Proceeding from event...")
                    self.client.proceed()
            else:
                self.logger.info("Proceeding from event...")
                self.client.proceed()
            # Update state
            self.current_state = self.client.get_state()
        elif current_screen == 'SHOP':
            # Proceed from shop
            self.logger.info("Leaving shop...")
            self.client.proceed()
            # Update state
            self.current_state = self.client.get_state()
        elif current_screen == 'REST':
            # Choose random rest option
            rest = self.current_state.get('rest', {})
            if rest and isinstance(rest, dict):
                options = rest.get('options', [])
                if options:
                    option_index = 0
                    self.logger.info(f"Choosing rest option at index {option_index}...")
                    self.client.choose_rest_option(option_index)
                else:
                    self.logger.info("Proceeding from rest site...")
                    self.client.proceed()
            else:
                self.logger.info("Proceeding from rest site...")
                self.client.proceed()
            # Update state
            self.current_state = self.client.get_state()
        elif current_screen == 'CHEST':
            # Open chest
            self.logger.info("Opening chest...")
            self.client.open_chest()
            # Update state
            self.current_state = self.client.get_state()
        elif current_screen == 'TREASURE':
            # Choose treasure relic
            chest = self.current_state.get('chest', {})
            if chest and isinstance(chest, dict):
                relics = chest.get('relics', [])
                if relics:
                    option_index = 0
                    self.logger.info(f"Choosing treasure relic at index {option_index}...")
                    self.client.choose_treasure_relic(option_index)
                else:
                    self.logger.info("Proceeding from treasure...")
                    self.client.proceed()
            else:
                self.logger.info("Proceeding from treasure...")
                self.client.proceed()
            # Update state
            self.current_state = self.client.get_state()
        elif current_screen == 'CARD_SELECTION':
            # Handle card selection
            self.logger.info("Handling card selection...")
            selection = self.current_state.get('selection', {})
            if selection and isinstance(selection, dict):
                cards = selection.get('cards', [])
                if cards:
                    option_index = 0
                    self.logger.info(f"Choosing card at index {option_index}...")
                    self.client.select_deck_card(option_index)
                else:
                    self.logger.info("Proceeding from card selection...")
                    self.client.proceed()
            else:
                self.logger.info("Proceeding from card selection...")
                self.client.proceed()
            # Update state
            self.current_state = self.client.get_state()

    def _return_to_main_menu(self):
        """Return to main menu after episode ends"""
        current_screen = self.current_state.get('screen', '')
        
        if current_screen == 'GAME_OVER':
            print("Returning to main menu from game over...")
            self.client.return_to_main_menu()
        elif current_screen != 'MAIN_MENU':
            print("Returning to main menu...")
            self.client.return_to_main_menu()

    def _calculate_reward(self) -> float:
        """Calculate reward based on state changes

        Returns:
            Reward value
        """
        reward = 0.0

        if self.previous_state is None or self.current_state is None:
            return reward

        # Get current and previous player states
        player = self.current_state.get('run', {}).get('player', {})
        prev_player = self.previous_state.get('run', {}).get('player', {})

        # Get current and previous run states
        run = self.current_state.get('run', {})
        prev_run = self.previous_state.get('run', {})

        # HP changes
        hp_change = player.get('hp', 0) - self.previous_hp
        if hp_change > 0:
            reward += hp_change * 0.1  # Healing is good
        elif hp_change < 0:
            reward += hp_change * 0.5  # Taking damage is bad

        # Gold changes
        gold_change = player.get('gold', 0) - self.previous_gold
        reward += gold_change * 0.01  # Gaining gold is slightly good

        # Floor progress
        floor_change = run.get('floor', 0) - self.previous_floor
        if floor_change > 0:
            reward += floor_change * 1.0  # Progressing is good

        # Check for state transitions that indicate success
        current_screen = self.current_state.get('screen', '')
        prev_screen = self.previous_state.get('screen', '')

        # Combat victory
        if prev_screen == 'COMBAT' and current_screen == 'REWARD':
            reward += 1.0  # Combat victory

        # Penalties for bad states
        if current_screen == 'GAME_OVER':
            reward -= 10.0  # Death penalty

        return reward * self.reward_scale

    def _is_terminated(self) -> bool:
        """Check if episode should terminate

        Returns:
            True if terminated, False otherwise
        """
        if self.current_state is None:
            return True

        current_screen = self.current_state.get('screen', '')

        # Game over screen
        if current_screen == 'GAME_OVER':
            return True

        # Only terminate when we reach GAME_OVER screen, not just when HP is 0
        # This allows the game to transition to GAME_OVER state naturally

        # Maximum steps reached
        if self.step_count >= self.max_steps:
            return True

        return False

    def _render(self):
        """Render the environment (human mode)"""
        if self.current_state is None:
            return

        screen = self.current_state.get('screen', 'unknown')
        player = self.current_state.get('run', {}).get('player', {})
        run = self.current_state.get('run', {})

        print(f"\n{'='*60}")
        print(f"Step: {self.step_count} | Screen: {screen}")
        print(f"{'='*60}")
        print(f"HP: {player.get('hp', 0)}/{player.get('max_hp', 0)} | "
              f"Gold: {player.get('gold', 0)} | "
              f"Floor: {run.get('floor', 0)} | "
              f"Act: {run.get('act', 1)}")

        if screen == 'COMBAT':
            battle = self.current_state.get('battle', {})
            print(f"Energy: {player.get('energy', 0)}/{player.get('max_energy', 3)} | "
                  f"Block: {player.get('block', 0)}")
            print(f"Hand size: {len(player.get('hand', []))}")
            enemies = battle.get('enemies', [])
            for i, enemy in enumerate(enemies):
                print(f"Enemy {i}: {enemy.get('name', 'Unknown')} - "
                      f"HP: {enemy.get('hp', 0)}/{enemy.get('max_hp', 0)}")

        print(f"Episode Reward: {self.episode_reward:.2f}")
        print(f"{'='*60}\n")

    def close(self):
        """Clean up resources"""
        if self.client:
            self.client.close()
