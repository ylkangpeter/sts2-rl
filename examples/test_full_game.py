# -*- coding: utf-8 -*-
"""Test script for full game flow from menu to death"""

import time
import random
import requests
from st2rl.envs import STS2Env
from st2rl.spaces.actions import ActionSpace
from st2rl.utils.logger import get_run_logger


def get_singleplayer_state():
    """Get singleplayer state from new API endpoint"""
    try:
        response = requests.get('http://localhost:15526/api/v1/singleplayer')
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None


def test_full_game_flow():
    """Test full game flow from menu to death"""
    logger = get_run_logger()
    logger.info("Starting full game flow test...")
    
    # Generate a batch timestamp for all episodes
    import datetime
    batch_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Batch timestamp: {batch_timestamp}")
    
    # Create environment
    env = STS2Env(
        host='localhost',
        port=8080,
        render_mode='human',
        character_index=0  # Select first character
    )
    
    try:
        # Reset environment (handles any initial state)
        logger.info("=== Resetting environment ===")
        try:
            obs, info = env.reset()
            if info and 'state_type' in info:
                logger.info(f"Initial state: {info['state_type']}")
            else:
                logger.info(f"Initial state: unknown, info: {info}")
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Set default values
            obs = {}
            info = {'state_type': 'unknown'}
        
        # Get state from new API endpoint
        singleplayer_state = get_singleplayer_state()
        if singleplayer_state:
            logger.info(f"Singleplayer state: {singleplayer_state}")
        
        # If initial state is not in game, navigate to game start
        try:
            current_state = env.current_state
            logger.info(f"env.current_state type: {type(current_state)}")
            logger.info(f"env.current_state value: {current_state}")
            current_screen = current_state.get('screen', '') if current_state else ''
            logger.info(f"Initial screen: {current_screen}")
            
            # Try to navigate to game start regardless of current screen
            logger.info("Navigating to game start...")
            # Force navigation to game start
            try:
                # Check if we need to continue a run instead of starting a new one
                current_state = env.current_state
                current_screen = current_state.get('screen', '') if current_state else ''
                
                if current_screen == 'MAIN_MENU':
                    # Check available actions
                    available_actions = current_state.get('available_actions', [])
                    logger.info(f"Available actions: {available_actions}")
                    
                    if 'continue_run' in available_actions:
                        # Continue existing run
                        logger.info("Continuing existing run...")
                        try:
                            env.client.continue_run()
                            time.sleep(2)
                            state = env.client.get_state()
                            env.current_state = state
                            new_screen = env.current_state.get('screen', 'unknown') if env.current_state else 'unknown'
                            logger.info(f"State after continue_run: {new_screen}")
                        except Exception as e:
                            logger.error(f"Error continuing run: {e}")
                    else:
                        # Start new run
                        env._navigate_to_game_start()
                else:
                    # Already in game or game start screen, continue
                    logger.info(f"Already in {current_screen} screen, continuing...")
                
                # Update state
                state = env.client.get_state()
                logger.info(f"env.client.get_state() type: {type(state)}")
                logger.info(f"env.client.get_state() value: {state}")
                env.current_state = state
                new_screen = env.current_state.get('screen', 'unknown') if env.current_state else 'unknown'
                logger.info(f"New state after navigation: {new_screen}")
            except Exception as e:
                logger.error(f"Error navigating to game start: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            
            # If still in UNKNOWN state, try to return to main menu first
            try:
                current_screen = env.current_state.get('screen', '') if env.current_state else ''
                logger.info(f"Current screen after navigation: {current_screen}")
                if current_screen == 'UNKNOWN':
                    logger.info("Still in UNKNOWN state, trying to return to main menu...")
            except Exception as e:
                logger.error(f"Error getting current screen: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        except Exception as e:
            logger.error(f"Error in test script: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Get state from new API endpoint
            singleplayer_state = get_singleplayer_state()
            if singleplayer_state:
                logger.info(f"Singleplayer state: {singleplayer_state}")
            try:
                # Try different methods to reset game state
                methods = [
                    ('return_to_main_menu', []),
                    ('proceed', []),
                    ('dismiss_modal', []),
                    ('open_character_select', [])
                ]
                
                for method_name, args in methods:
                    logger.info(f"Trying {method_name}...")
                    method = getattr(env.client, method_name, None)
                    if method:
                        try:
                            method(*args)
                            env.current_state = env.client.get_state()
                            new_screen = env.current_state.get('screen', 'unknown')
                            logger.info(f"State after {method_name}: {new_screen}")
                            if new_screen != 'UNKNOWN':
                                break
                        except Exception as e:
                            logger.error(f"Error calling {method_name}: {e}")
            except Exception as e:
                logger.error(f"Error resetting game state: {e}")
        
        # Track episode count
        episode_count = 0
        max_episodes = 100  # Run 100 episodes for testing
        
        while episode_count < max_episodes:
            logger.info(f"\n=== {batch_timestamp} - Episode {episode_count + 1} ===")
            step_count = 0
            max_steps = 5000  # Increase max steps to allow the game to progress
            
            # Initialize termination flag
            terminated = False
            
            # Reset environment at the start of each episode
            try:
                obs, info = env.reset()
                if info and 'state_type' in info:
                    logger.info(f"Initial state: {info['state_type']}")
                else:
                    logger.info(f"Initial state: unknown, info: {info}")
            except Exception as e:
                logger.error(f"Error in reset: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Set default values
                obs = {}
                info = {'state_type': 'unknown'}
            
            # Ensure current_state is initialized
            if env.current_state is None:
                env.current_state = {}
            
            while step_count < max_steps and not terminated:
                try:
                    # Get current state
                    current_state = env.current_state
                    # Ensure current_state is a dictionary
                    if current_state is None:
                        current_state = {}
                    current_screen = current_state.get('screen', '')
                except Exception as e:
                    logger.error(f"Error getting current state: {e}")
                    # Set default values
                    current_state = {}
                    current_screen = 'unknown'
                
                # Get state from new API endpoint
                singleplayer_state = get_singleplayer_state()
                if singleplayer_state and isinstance(singleplayer_state, dict):
                    # Get state type from new API
                    state_type = singleplayer_state.get('state_type', 'unknown')
                else:
                    state_type = 'unknown'
                    singleplayer_state = {}
                
                # Get detailed state information
                if current_state:
                    player = current_state.get('run', {}).get('player', {})
                    hp = player.get('hp', 0)
                    max_hp = player.get('max_hp', 0)
                    gold = player.get('gold', 0)
                    run = current_state.get('run', {})
                    floor = run.get('floor', 0)
                else:
                    player = {}
                    hp = 0
                    max_hp = 0
                    gold = 0
                    run = {}
                    floor = 0
                
                # Get player HP from new API if available
                if singleplayer_state and 'player' in singleplayer_state:
                    api_hp = singleplayer_state['player'].get('hp', 0)
                    api_max_hp = singleplayer_state['player'].get('max_hp', 0)
                    if api_hp > 0:
                        hp = api_hp
                        max_hp = api_max_hp
                
                logger.info(f"Step {step_count}: Current screen={current_screen}, State type={state_type}, HP={hp}/{max_hp}, Gold={gold}, Floor={floor}")
                
                # Track screen state to detect if we're stuck
                if 'screen_history' not in locals():
                    screen_history = []
                screen_history.append(current_screen)
                
                # Keep only the last 10 screen states to check for repetition
                if len(screen_history) > 10:
                    screen_history = screen_history[-10:]
                
                # Check if we're stuck on the same screen
                screen_stuck = len(screen_history) == 10 and all(s == screen_history[0] for s in screen_history)
                player_dead_in_combat = hp <= 0 and current_screen == 'COMBAT'
                
                # Don't consider combat screens as stuck, as combat can take multiple steps
                is_combat_screen = current_screen == 'COMBAT' or state_type in ['monster', 'elite', 'boss']
                
                if screen_stuck and not is_combat_screen:
                    logger.info(f"Stuck on screen: {current_screen}, forcing termination...")
                    terminated = True
                elif player_dead_in_combat:
                    # Track how long we've been waiting for GAME_OVER
                    if 'dead_wait_count' not in locals():
                        dead_wait_count = 0
                    dead_wait_count += 1
                    
                    if dead_wait_count > 30:  # Wait at most 30 steps for GAME_OVER
                        logger.info("Waiting for GAME_OVER screen timed out, forcing termination...")
                        terminated = True
                    else:
                        logger.info(f"Player dead, waiting for GAME_OVER screen... (wait count: {dead_wait_count})")
                else:
                    # Execute appropriate action based on current screen and new API state
                    try:
                        # Store previous state for reward calculation
                        env.previous_state = env.current_state.copy() if env.current_state else None
                        
                        # Check state type from new API
                        if state_type == 'bundle_select':
                            logger.info("Found bundle_select state, using STS2MCP API to choose a bundle...")
                            # Choose a random bundle
                            bundle_count = len(singleplayer_state.get('bundle_select', {}).get('bundles', []))
                            if bundle_count > 0:
                                bundle_index = random.randint(0, bundle_count - 1)
                                logger.info(f"Choosing bundle at index {bundle_index}")
                                # Use STS2MCP API to select bundle
                                try:
                                    # First select the bundle
                                    response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                        'action': 'select_bundle',
                                        'index': bundle_index
                                    })
                                    logger.info(f"Select bundle response: {response.json()}")
                                    time.sleep(1)
                                    # Then confirm the selection
                                    response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                        'action': 'confirm_bundle_selection'
                                    })
                                    logger.info(f"Confirm bundle selection response: {response.json()}")
                                    time.sleep(1)
                                except Exception as e:
                                    logger.error(f"Error using STS2MCP API: {e}")
                        # Check if game state has reward information from new API
                        elif state_type in ['combat_rewards', 'rewards', 'card_reward']:
                            logger.info(f"Found reward information (state_type={state_type}), using STS2MCP API to choose a reward...")
                            # Use STS2MCP API to handle rewards
                            try:
                                if state_type == 'combat_rewards' or state_type == 'rewards':
                                    # Claim a random reward
                                    rewards = singleplayer_state.get('rewards', {}).get('items', [])
                                    if rewards:
                                        reward_index = random.randint(0, len(rewards) - 1)
                                        logger.info(f"Claiming reward at index {reward_index}: {rewards[reward_index]['description']}")
                                        response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                            'action': 'claim_reward',
                                            'index': reward_index
                                        })
                                        logger.info(f"Claim reward response: {response.json()}")
                                    else:
                                        # Proceed if no rewards
                                        logger.info("No rewards available, proceeding...")
                                        response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                            'action': 'proceed'
                                        })
                                        logger.info(f"Proceed response: {response.json()}")
                                elif state_type == 'card_reward':
                                    # Select a random card reward
                                    cards = singleplayer_state.get('card_reward', {}).get('cards', [])
                                    if cards:
                                        card_index = random.randint(0, len(cards) - 1)
                                        logger.info(f"Selecting card reward at index {card_index}: {cards[card_index]['name']}")
                                        response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                            'action': 'select_card_reward',
                                            'card_index': card_index
                                        })
                                        logger.info(f"Select card reward response: {response.json()}")
                                    else:
                                        # Skip card reward if no cards
                                        logger.info("No card rewards available, skipping...")
                                        response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                            'action': 'skip_card_reward'
                                        })
                                        logger.info(f"Skip card reward response: {response.json()}")
                                time.sleep(1)
                            except Exception as e:
                                logger.error(f"Error using STS2MCP API for rewards: {e}")
                        # Check if game state has hand select information from new API
                        elif state_type == 'hand_select':
                            logger.info("Found hand select information, using STS2MCP API to select a card...")
                            # Use STS2MCP API to handle hand select
                            try:
                                # Get cards from hand select
                                cards = singleplayer_state.get('hand_select', {}).get('cards', [])
                                if cards:
                                    # Choose a random card
                                    card_index = random.randint(0, len(cards) - 1)
                                    logger.info(f"Selecting card at index {card_index}: {cards[card_index]['name']}")
                                    # Select the card
                                    response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                        'action': 'combat_select_card',
                                        'card_index': card_index
                                    })
                                    logger.info(f"Select card response: {response.json()}")
                                    time.sleep(1)
                                    # Confirm the selection
                                    response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                        'action': 'combat_confirm_selection'
                                    })
                                    logger.info(f"Confirm selection response: {response.json()}")
                                time.sleep(1)
                            except Exception as e:
                                logger.error(f"Error using STS2MCP API for hand select: {e}")
                        # Check if game state has card select information from new API
                        elif state_type == 'card_select':
                            logger.info("Found card select information, using STS2MCP API to select a card...")
                            # Use STS2MCP API to handle card select
                            try:
                                # Get cards from card select
                                cards = singleplayer_state.get('card_select', {}).get('cards', [])
                                if cards:
                                    # Choose a random card
                                    card_index = random.randint(0, len(cards) - 1)
                                    logger.info(f"Selecting card at index {card_index}: {cards[card_index]['name']}")
                                    # Select the card
                                    response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                        'action': 'select_card',
                                        'index': card_index
                                    })
                                    logger.info(f"Select card response: {response.json()}")
                                    time.sleep(1)
                                    # Confirm the selection
                                    response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                        'action': 'confirm_selection'
                                    })
                                    logger.info(f"Confirm selection response: {response.json()}")
                                time.sleep(1)
                            except Exception as e:
                                logger.error(f"Error using STS2MCP API for card select: {e}")
                        # Check if game state has treasure information from new API
                        elif state_type == 'treasure':
                            logger.info("Found treasure information, using STS2MCP API to claim a relic...")
                            # Use STS2MCP API to handle treasure
                            try:
                                # Get relics from treasure
                                relics = singleplayer_state.get('treasure', {}).get('relics', [])
                                if relics:
                                    # Choose a random relic
                                    relic_index = random.randint(0, len(relics) - 1)
                                    logger.info(f"Claiming relic at index {relic_index}: {relics[relic_index]['name']}")
                                    # Claim the relic
                                    response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                        'action': 'claim_treasure_relic',
                                        'index': relic_index
                                    })
                                    logger.info(f"Claim relic response: {response.json()}")
                                else:
                                    # Proceed if no relics
                                    logger.info("No relics available, proceeding...")
                                    response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                        'action': 'proceed'
                                    })
                                    logger.info(f"Proceed response: {response.json()}")
                                time.sleep(1)
                            except Exception as e:
                                logger.error(f"Error using STS2MCP API for treasure: {e}")
                        # Check if game state has map information
                        elif current_state and current_state.get('map'):
                            logger.info("Found map information, trying to choose a map node...")
                            # Get available map nodes
                            map_data = current_state.get('map', {})
                            available_nodes = map_data.get('available_nodes', [])
                            nodes = map_data.get('nodes', [])
                            
                            # Try to find a valid node index
                            if available_nodes:
                                # Choose a random available node
                                node_index = random.randint(0, len(available_nodes) - 1)
                                logger.info(f"Choosing map node at index {node_index}")
                                env.client.choose_map_node(node_index)
                            elif nodes:
                                # Choose a random node
                                node_index = random.randint(0, len(nodes) - 1)
                                logger.info(f"Choosing map node at index {node_index}")
                                env.client.choose_map_node(node_index)
                        # Check if game state has combat information
                        elif (current_state and current_state.get('combat')) or state_type in ['monster', 'elite', 'boss']:
                            logger.info("Found combat information, using STS2MCP API to execute combat action...")
                            # Use STS2MCP API to execute combat action
                            try:
                                # Get player hand, energy, and potions from STS2MCP state
                                hand = singleplayer_state.get('player', {}).get('hand', [])
                                energy = singleplayer_state.get('player', {}).get('energy', 0)
                                potions = singleplayer_state.get('potions', [])
                                
                                # Filter playable cards
                                playable_cards = [card for card in hand if card.get('can_play', False)]
                                
                                # Filter usable potions
                                usable_potions = [potion for potion in potions if potion.get('can_use', False)]
                                
                                # Randomly decide to use a potion or play a card
                                action_choice = random.random()
                                
                                if action_choice < 0.2 and usable_potions:  # 20% chance to use a potion
                                    # Choose a random usable potion
                                    potion = random.choice(usable_potions)
                                    potion_slot = potion.get('index', 0)
                                    logger.info(f"Using potion at slot {potion_slot}: {potion.get('name', 'Unknown')}")
                                    # Use the potion
                                    response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                        'action': 'use_potion',
                                        'slot': potion_slot
                                    })
                                    logger.info(f"Use potion response: {response.json()}")
                                elif playable_cards:
                                    # Choose a random playable card
                                    card = random.choice(playable_cards)
                                    card_index = card.get('index', 0)
                                    logger.info(f"Playing card at index {card_index}: {card['name']}")
                                    # Check if card requires target
                                    target = None
                                    enemies = singleplayer_state.get('battle', {}).get('enemies', [])
                                    if enemies and card.get('target_type') in ['AnyEnemy', 'Enemy']:
                                        target = enemies[0].get('entity_id')
                                    # Play the card
                                    response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                        'action': 'play_card',
                                        'card_index': card_index,
                                        'target': target
                                    })
                                    logger.info(f"Play card response: {response.json()}")
                                else:
                                    # End turn if no playable cards or energy
                                    logger.info("No playable cards or energy, ending turn...")
                                    response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                        'action': 'end_turn'
                                    })
                                    logger.info(f"End turn response: {response.json()}")
                                time.sleep(1)
                            except Exception as e:
                                logger.error(f"Error using STS2MCP API for combat: {e}")
                        # Check if game state has reward information
                        elif current_state and current_state.get('reward'):
                            logger.info("Found reward information, trying to choose a reward...")
                            env._handle_special_transition()
                        # Check if game state has event information
                        elif current_state and current_state.get('event'):
                            logger.info("Found event information, trying to choose an event option...")
                            env._handle_special_transition()
                        # Check if game state has shop information
                        elif current_state and current_state.get('shop'):
                            logger.info("Found shop information, trying to purchase items...")
                            # Use STS2MCP API to handle shop
                            try:
                                # Get shop information from STS2MCP
                                shop = singleplayer_state.get('shop', {})
                                cards = shop.get('cards', [])
                                potions = shop.get('potions', [])
                                
                                # Combine all items and filter affordable ones
                                all_items = []
                                for card in cards:
                                    if card.get('enough_gold', False):
                                        all_items.append(('card', card.get('index', 0), card.get('name', 'Unknown card')))
                                for potion in potions:
                                    if potion.get('enough_gold', False):
                                        all_items.append(('potion', potion.get('index', 0), potion.get('name', 'Unknown potion')))
                                
                                if all_items:
                                    # Choose a random affordable item
                                    item_type, item_index, item_name = random.choice(all_items)
                                    logger.info(f"Purchasing {item_type} at index {item_index}: {item_name}")
                                    # Use STS2MCP API to purchase item
                                    response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                        'action': 'shop_purchase',
                                        'index': item_index
                                    })
                                    logger.info(f"Shop purchase response: {response.json()}")
                                    time.sleep(1)
                                else:
                                    # If no affordable items, try to proceed
                                    logger.info("No affordable items, proceeding...")
                                    response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                                        'action': 'proceed'
                                    })
                                    logger.info(f"Proceed response: {response.json()}")
                                    time.sleep(2)
                            except Exception as e:
                                logger.error(f"Error handling shop: {e}")
                                # Fallback to _handle_special_transition
                                env._handle_special_transition()
                        # Check if game state has rest information
                        elif current_state and current_state.get('rest'):
                            logger.info("Found rest information, trying to choose a rest option...")
                            env._handle_special_transition()
                        # Check if game state has chest information
                        elif current_state and current_state.get('chest'):
                            logger.info("Found chest information, trying to open chest...")
                            env._handle_special_transition()
                        # Check if game state has card selection information
                        elif current_state and current_state.get('selection'):
                            logger.info("Found card selection information, trying to choose a card...")
                            env._handle_special_transition()
                        # Check if we're in main menu
                        elif current_screen == 'MAIN_MENU':
                            logger.info("In MAIN_MENU, starting a new game...")
                            # Start a new game
                            try:
                                # Open character select
                                env.client.open_character_select()
                                time.sleep(1)
                                # Select a random character
                                character_index = 0  # Default to first character
                                env.client.select_character(character_index)
                                time.sleep(1)
                                # Start the run
                                env.client.embark()
                                time.sleep(2)
                            except Exception as e:
                                logger.error(f"Error starting new game: {e}")
                        # For UNKNOWN screen, try to proceed
                        elif current_screen == 'UNKNOWN':
                            logger.info("In UNKNOWN screen, trying to proceed...")
                            # Try to proceed
                            try:
                                env.client.proceed()
                            except Exception as e:
                                logger.error(f"Error proceeding: {e}")
                                # Try to return to main menu
                                try:
                                    env.client.return_to_main_menu()
                                except Exception as e:
                                    logger.error(f"Error returning to main menu: {e}")
                        else:
                            # For other screens, use random action
                            action = env.action_space.sample()
                            obs, reward, terminated, truncated, info = env.step(action)
                    except Exception as e:
                        logger.error(f"Error executing action: {e}")
                        reward = -0.1
                        terminated = False
                        truncated = False
                        info = {
                            'error': str(e),
                            'state_type': current_screen,
                            'step_count': env.step_count
                        }
                
                # Update state
                env.current_state = env.client.get_state()
                
                # Get updated player state
                if env.current_state:
                    player = env.current_state.get('run', {}).get('player', {})
                    hp = player.get('hp', 0)
                    max_hp = player.get('max_hp', 0)
                else:
                    player = {}
                    hp = 0
                    max_hp = 0
                
                # Calculate reward and check termination
                reward = env._calculate_reward()
                
                # Check termination conditions
                current_screen = env.current_state.get('screen', '') if env.current_state else ''
                game_over = current_screen == 'GAME_OVER'
                player_dead = hp <= 0
                max_steps_reached = env.step_count >= env.max_steps
                
                # Only terminate if game is actually over or max steps reached
                # Allow game to transition to GAME_OVER screen after player death
                # Preserve any previous termination flag (e.g., from screen stuck detection)
                if 'terminated' not in locals() or not terminated:
                    terminated = game_over or max_steps_reached
                truncated = max_steps_reached
                
                # If player is dead but not at GAME_OVER screen yet, continue
                if player_dead and not game_over:
                    logger.info("Player dead, waiting for GAME_OVER screen...")
                
                # Update step count
                env.step_count += 1
                
                # Update reward tracking
                env.previous_hp = hp
                env.previous_gold = player.get('gold', 0)
                if env.current_state:
                    run = env.current_state.get('run', {})
                    env.previous_floor = run.get('floor', 0)
                else:
                    env.previous_floor = 0
                env.episode_reward += reward
                
                # Info dictionary
                info = {
                    'state_type': current_screen,
                    'is_menu': current_screen == 'MAIN_MENU',
                    'step_count': env.step_count,
                    'episode_reward': env.episode_reward
                }
                
                # Log detailed state after action
                logger.info(f"After action: HP={hp}/{max_hp}, Screen={current_screen}, Game Over={game_over}, Player Dead={player_dead}")
                
                # Log step information
                if step_count % 20 == 0:
                    logger.info(f"Step {step_count}: State={current_screen}, Reward={reward:.2f}")
                
                # Increment step count
                step_count += 1
            
            # Episode completed
            logger.info(f"Episode {episode_count + 1} terminated after {step_count} steps")
            final_screen = env.current_state.get('screen', 'unknown') if env.current_state else 'unknown'
            logger.info(f"Final state: {final_screen}")
            logger.info(f"Final HP: {hp}/{max_hp}")
            
            # Return to main menu after death
            logger.info("Returning to main menu...")
            try:
                # First try to handle any current screen (like SHOP) before returning to main menu
                current_state = env.current_state
                current_screen = current_state.get('screen', '') if current_state else ''
                
                if current_screen == 'SHOP':
                    # Try to leave shop first
                    logger.info("Trying to leave shop...")
                    try:
                        # Use STS2MCP API to leave shop (using proceed action)
                        response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                            'action': 'proceed'
                        })
                        logger.info(f"Leave shop response: {response.json()}")
                        time.sleep(2)
                    except Exception as e:
                        logger.error(f"Error leaving shop: {e}")
                
                # Try multiple methods to return to main menu
                methods = [
                    ('return_to_main_menu', []),
                    ('abandon_run', []),
                    ('proceed', [])
                ]
                
                for method_name, args in methods:
                    logger.info(f"Trying {method_name}...")
                    method = getattr(env.client, method_name, None)
                    if method:
                        try:
                            method(*args)
                            time.sleep(2)  # Give time for transition
                            # Verify we're back at main menu
                            state = env.client.get_state()
                            env.current_state = state if state is not None and isinstance(state, dict) else {}
                            current_screen = env.current_state.get('screen', '')
                            logger.info(f"State after {method_name}: {current_screen}")
                            if current_screen == 'MAIN_MENU':
                                logger.info("Return to main menu successful!")
                                break
                        except Exception as e:
                            logger.error(f"Error calling {method_name}: {e}")
                
                # If still not in main menu, try STS2MCP API
                if current_screen != 'MAIN_MENU':
                    logger.info("Trying STS2MCP API to return to main menu...")
                    try:
                        response = requests.post('http://localhost:15526/api/v1/singleplayer', json={
                            'action': 'return_to_main_menu'
                        })
                        logger.info(f"STS2MCP return to main menu response: {response.json()}")
                        time.sleep(2)
                        state = env.client.get_state()
                        env.current_state = state if state is not None and isinstance(state, dict) else {}
                        current_screen = env.current_state.get('screen', '')
                        logger.info(f"Final state after STS2MCP: {current_screen}")
                    except Exception as e:
                        logger.error(f"Error using STS2MCP API: {e}")
                        # Ensure current_state is still a dictionary
                        if env.current_state is None:
                            env.current_state = {}
            except Exception as e:
                logger.error(f"Error returning to main menu: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Ensure current_state is still a dictionary even if there's an error
                if env.current_state is None:
                    env.current_state = {}
            
            # Increment episode count
            episode_count += 1
            
            # Ensure current_state is initialized for next episode
            if env.current_state is None:
                env.current_state = {}
            
    except Exception as e:
        logger.error(f"Error in test script: {e}")
    finally:
        # Cleanup
        logger.info("Test completed")


if __name__ == "__main__":
    test_full_game_flow()
