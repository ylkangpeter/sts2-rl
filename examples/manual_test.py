#!/usr/bin/env python3
"""Manual test script for STS2 environment"""

from st2rl.envs import STS2Env
import numpy as np

def manual_test():
    """Manual test of the STS2 environment"""
    print("Starting manual test of STS2 environment...")
    print("Make sure Slay the Spire 2 is running with STS2MCP mod enabled!")
    
    # Create environment
    env = STS2Env(render_mode='human')
    
    try:
        # Reset environment
        obs, info = env.reset()
        print(f"Initial state: {info['state_type']}")
        
        step = 0
        total_reward = 0
        
        while True:
            print(f"\n{'='*60}")
            print(f"Step: {step + 1}")
            print(f"State: {info['state_type']}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"{'='*60}")
            
            # Get current state
            current_state = env.client.get_state()
            
            # Print state information based on state type
            if info['state_type'] == 'map':
                map_data = current_state.get('map', {})
                print(f"Current position: {map_data.get('current_position', {})}")
                print(f"Available nodes: {len(map_data.get('next_options', []))}")
                
                # Print available map nodes
                next_options = map_data.get('next_options', [])
                for i, option in enumerate(next_options):
                    print(f"  {i}: {option['type']} at ({option['col']}, {option['row']})")
                
                # Ask user to select a node
                node_index = input("Enter node index to select (or 'q' to quit): ")
                if node_index == 'q':
                    break
                
                try:
                    node_index = int(node_index)
                    if 0 <= node_index < len(next_options):
                        # Create action for map node selection
                        action = {
                            'action_type': np.array([0], dtype=np.int32),  # 0 for map action
                            'map_node_index': np.array([node_index], dtype=np.int32),
                            'card_index': np.array([-1], dtype=np.int32),
                            'target_index': np.array([-1], dtype=np.int32),
                            'potion_index': np.array([-1], dtype=np.int32),
                            'event_option_index': np.array([-1], dtype=np.int32),
                            'rest_option_index': np.array([-1], dtype=np.int32),
                            'reward_index': np.array([-1], dtype=np.int32),
                            'shop_item_index': np.array([-1], dtype=np.int32)
                        }
                    else:
                        print("Invalid node index!")
                        continue
                except ValueError:
                    print("Invalid input!")
                    continue
                    
            elif info['state_type'] == 'monster':
                battle_data = current_state.get('battle', {})
                player = battle_data.get('player', {})
                enemies = battle_data.get('enemies', [])
                hand = battle_data.get('hand', [])
                
                print(f"Player: HP {player.get('current_hp', 0)}/{player.get('max_hp', 0)}, Energy: {player.get('energy', 0)}/{player.get('max_energy', 0)}")
                print(f"Enemies: {len(enemies)}")
                for i, enemy in enumerate(enemies):
                    print(f"  {i}: {enemy.get('name', 'Unknown')} - HP: {enemy.get('current_hp', 0)}/{enemy.get('max_hp', 0)}")
                print(f"Hand: {len(hand)} cards")
                for i, card in enumerate(hand):
                    print(f"  {i}: {card.get('name', 'Unknown')} (Cost: {card.get('cost', 0)})")
                
                # Ask user to select an action
                print("Actions:")
                print("  0: Play card")
                print("  1: Use potion")
                print("  2: End turn")
                action_choice = input("Enter action choice (0-2, or 'q' to quit): ")
                if action_choice == 'q':
                    break
                
                try:
                    action_choice = int(action_choice)
                    if action_choice == 0:  # Play card
                        card_index = input("Enter card index: ")
                        try:
                            card_index = int(card_index)
                            if 0 <= card_index < len(hand):
                                # Ask for target if needed
                                target_index = -1
                                if len(enemies) > 0:
                                    target_index = input("Enter target index (-1 for no target): ")
                                    try:
                                        target_index = int(target_index)
                                    except ValueError:
                                        target_index = -1
                                
                                action = {
                                    'action_type': np.array([1], dtype=np.int32),  # 1 for play_card
                                    'card_index': np.array([card_index], dtype=np.int32),
                                    'target_index': np.array([target_index], dtype=np.int32),
                                    'map_node_index': np.array([-1], dtype=np.int32),
                                    'potion_index': np.array([-1], dtype=np.int32),
                                    'event_option_index': np.array([-1], dtype=np.int32),
                                    'rest_option_index': np.array([-1], dtype=np.int32),
                                    'reward_index': np.array([-1], dtype=np.int32),
                                    'shop_item_index': np.array([-1], dtype=np.int32)
                                }
                            else:
                                print("Invalid card index!")
                                continue
                        except ValueError:
                            print("Invalid input!")
                            continue
                    elif action_choice == 1:  # Use potion
                        # Get available potions
                        potions = player.get('potions', [])
                        print(f"Available potions: {len(potions)}")
                        for i, potion in enumerate(potions):
                            print(f"  {i}: {potion.get('name', 'Unknown')}")
                        
                        potion_index = input("Enter potion index: ")
                        try:
                            potion_index = int(potion_index)
                            if 0 <= potion_index < len(potions):
                                # Ask for target if needed
                                target_index = -1
                                if len(enemies) > 0:
                                    target_index = input("Enter target index (-1 for no target): ")
                                    try:
                                        target_index = int(target_index)
                                    except ValueError:
                                        target_index = -1
                                
                                action = {
                                    'action_type': np.array([2], dtype=np.int32),  # 2 for use_potion
                                    'potion_index': np.array([potion_index], dtype=np.int32),
                                    'target_index': np.array([target_index], dtype=np.int32),
                                    'map_node_index': np.array([-1], dtype=np.int32),
                                    'card_index': np.array([-1], dtype=np.int32),
                                    'event_option_index': np.array([-1], dtype=np.int32),
                                    'rest_option_index': np.array([-1], dtype=np.int32),
                                    'reward_index': np.array([-1], dtype=np.int32),
                                    'shop_item_index': np.array([-1], dtype=np.int32)
                                }
                            else:
                                print("Invalid potion index!")
                                continue
                        except ValueError:
                            print("Invalid input!")
                            continue
                    elif action_choice == 2:  # End turn
                        action = {
                            'action_type': np.array([3], dtype=np.int32),  # 3 for end_turn
                            'map_node_index': np.array([-1], dtype=np.int32),
                            'card_index': np.array([-1], dtype=np.int32),
                            'target_index': np.array([-1], dtype=np.int32),
                            'potion_index': np.array([-1], dtype=np.int32),
                            'event_option_index': np.array([-1], dtype=np.int32),
                            'rest_option_index': np.array([-1], dtype=np.int32),
                            'reward_index': np.array([-1], dtype=np.int32),
                            'shop_item_index': np.array([-1], dtype=np.int32)
                        }
                    else:
                        print("Invalid action choice!")
                        continue
                except ValueError:
                    print("Invalid input!")
                    continue
                    
            elif info['state_type'] == 'event':
                event_data = current_state.get('event', {})
                options = event_data.get('options', [])
                
                print(f"Event: {event_data.get('name', 'Unknown')}")
                print(f"Description: {event_data.get('description', '')}")
                print(f"Options: {len(options)}")
                for i, option in enumerate(options):
                    print(f"  {i}: {option.get('text', 'Unknown')}")
                
                # Ask user to select an option
                option_index = input("Enter option index (or 'q' to quit): ")
                if option_index == 'q':
                    break
                
                try:
                    option_index = int(option_index)
                    if 0 <= option_index < len(options):
                        action = {
                            'action_type': np.array([4], dtype=np.int32),  # 4 for choose_event_option
                            'event_option_index': np.array([option_index], dtype=np.int32),
                            'map_node_index': np.array([-1], dtype=np.int32),
                            'card_index': np.array([-1], dtype=np.int32),
                            'target_index': np.array([-1], dtype=np.int32),
                            'potion_index': np.array([-1], dtype=np.int32),
                            'rest_option_index': np.array([-1], dtype=np.int32),
                            'reward_index': np.array([-1], dtype=np.int32),
                            'shop_item_index': np.array([-1], dtype=np.int32)
                        }
                    else:
                        print("Invalid option index!")
                        continue
                except ValueError:
                    print("Invalid input!")
                    continue
                    
            elif info['state_type'] == 'rest':
                rest_data = current_state.get('rest', {})
                options = rest_data.get('options', [])
                
                print("Rest Site Options:")
                for i, option in enumerate(options):
                    print(f"  {i}: {option.get('text', 'Unknown')}")
                
                # Ask user to select an option
                option_index = input("Enter option index (or 'q' to quit): ")
                if option_index == 'q':
                    break
                
                try:
                    option_index = int(option_index)
                    if 0 <= option_index < len(options):
                        action = {
                            'action_type': np.array([5], dtype=np.int32),  # 5 for choose_rest_option
                            'rest_option_index': np.array([option_index], dtype=np.int32),
                            'map_node_index': np.array([-1], dtype=np.int32),
                            'card_index': np.array([-1], dtype=np.int32),
                            'target_index': np.array([-1], dtype=np.int32),
                            'potion_index': np.array([-1], dtype=np.int32),
                            'event_option_index': np.array([-1], dtype=np.int32),
                            'reward_index': np.array([-1], dtype=np.int32),
                            'shop_item_index': np.array([-1], dtype=np.int32)
                        }
                    else:
                        print("Invalid option index!")
                        continue
                except ValueError:
                    print("Invalid input!")
                    continue
                    
            elif info['state_type'] == 'reward':
                reward_data = current_state.get('reward', {})
                options = reward_data.get('options', [])
                
                print("Reward Options:")
                for i, option in enumerate(options):
                    print(f"  {i}: {option.get('name', 'Unknown')}")
                
                # Ask user to select an option
                option_index = input("Enter option index (or 'q' to quit): ")
                if option_index == 'q':
                    break
                
                try:
                    option_index = int(option_index)
                    if 0 <= option_index < len(options):
                        action = {
                            'action_type': np.array([6], dtype=np.int32),  # 6 for choose_reward_option
                            'reward_index': np.array([option_index], dtype=np.int32),
                            'map_node_index': np.array([-1], dtype=np.int32),
                            'card_index': np.array([-1], dtype=np.int32),
                            'target_index': np.array([-1], dtype=np.int32),
                            'potion_index': np.array([-1], dtype=np.int32),
                            'event_option_index': np.array([-1], dtype=np.int32),
                            'rest_option_index': np.array([-1], dtype=np.int32),
                            'shop_item_index': np.array([-1], dtype=np.int32)
                        }
                    else:
                        print("Invalid option index!")
                        continue
                except ValueError:
                    print("Invalid input!")
                    continue
                    
            elif info['state_type'] == 'shop':
                shop_data = current_state.get('shop', {})
                items = shop_data.get('items', [])
                
                print(f"Shop - Gold: {shop_data.get('gold', 0)}")
                print(f"Items: {len(items)}")
                for i, item in enumerate(items):
                    print(f"  {i}: {item.get('name', 'Unknown')} - Cost: {item.get('cost', 0)}")
                
                # Ask user to select an item
                item_index = input("Enter item index to purchase (or 'q' to quit, 'l' to leave): ")
                if item_index == 'q':
                    break
                elif item_index == 'l':
                    # Leave shop action
                    action = {
                        'action_type': np.array([8], dtype=np.int32),  # 8 for leave_shop
                        'map_node_index': np.array([-1], dtype=np.int32),
                        'card_index': np.array([-1], dtype=np.int32),
                        'target_index': np.array([-1], dtype=np.int32),
                        'potion_index': np.array([-1], dtype=np.int32),
                        'event_option_index': np.array([-1], dtype=np.int32),
                        'rest_option_index': np.array([-1], dtype=np.int32),
                        'reward_index': np.array([-1], dtype=np.int32),
                        'shop_item_index': np.array([-1], dtype=np.int32)
                    }
                else:
                    try:
                        item_index = int(item_index)
                        if 0 <= item_index < len(items):
                            action = {
                                'action_type': np.array([7], dtype=np.int32),  # 7 for shop_purchase
                                'shop_item_index': np.array([item_index], dtype=np.int32),
                                'map_node_index': np.array([-1], dtype=np.int32),
                                'card_index': np.array([-1], dtype=np.int32),
                                'target_index': np.array([-1], dtype=np.int32),
                                'potion_index': np.array([-1], dtype=np.int32),
                                'event_option_index': np.array([-1], dtype=np.int32),
                                'rest_option_index': np.array([-1], dtype=np.int32),
                                'reward_index': np.array([-1], dtype=np.int32)
                            }
                        else:
                            print("Invalid item index!")
                            continue
                    except ValueError:
                        print("Invalid input!")
                        continue
                        
            else:
                print(f"Unsupported state type: {info['state_type']}")
                break
            
            # Execute action
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                
                if terminated or truncated:
                    print(f"\nEpisode ended after {step} steps")
                    print(f"Total reward: {total_reward:.2f}")
                    print(f"Terminated: {terminated}, Truncated: {truncated}")
                    break
            except Exception as e:
                print(f"Error executing action: {e}")
                continue
                
    finally:
        env.close()
        print("Test completed!")

if __name__ == "__main__":
    manual_test()
