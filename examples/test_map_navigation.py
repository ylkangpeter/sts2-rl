#!/usr/bin/env python3
"""Test script for map navigation"""

from st2rl.envs import STS2Env
import numpy as np

def test_map_navigation():
    """Test map navigation functionality"""
    print("Testing map navigation...")
    
    # Create environment
    env = STS2Env(render_mode='human')
    
    try:
        # Reset environment
        obs, info = env.reset()
        print(f"Initial state: {info['state_type']}")
        
        if info['state_type'] == 'map':
            print("Starting map navigation test...")
            
            # Get current map state
            current_state = env.client.get_state()
            map_data = current_state.get('map', {})
            
            print("Current map state:")
            print(f"Current position: {map_data.get('current_position', {})}")
            print(f"Available nodes: {len(map_data.get('next_options', []))}")
            
            # Test selecting each available map node
            next_options = map_data.get('next_options', [])
            for i, option in enumerate(next_options):
                print(f"\nTesting node {i}: {option['type']} at ({option['col']}, {option['row']})")
                
                # Create action for map node selection
                action = {
                    'action_type': np.array([0], dtype=np.int32),  # 0 for map action
                    'map_node_index': np.array([i], dtype=np.int32),
                    'card_index': np.array([-1], dtype=np.int32),
                    'target_index': np.array([-1], dtype=np.int32),
                    'potion_index': np.array([-1], dtype=np.int32),
                    'event_option_index': np.array([-1], dtype=np.int32),
                    'rest_option_index': np.array([-1], dtype=np.int32),
                    'reward_index': np.array([-1], dtype=np.int32),
                    'shop_item_index': np.array([-1], dtype=np.int32)
                }
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"Action result:")
                print(f"  New state: {info['state_type']}")
                print(f"  Reward: {reward:.2f}")
                print(f"  Terminated: {terminated}")
                print(f"  Truncated: {truncated}")
                
                # If we entered a new state, return to map
                if info['state_type'] != 'map' and not terminated:
                    print("Returning to map...")
                    # For simplicity, we'll just get the state to see if we're back to map
                    current_state = env.client.get_state()
                    print(f"Current state after action: {current_state.get('state_type')}")
                    
        else:
            print(f"Current state is not map: {info['state_type']}")
            
    finally:
        env.close()

if __name__ == "__main__":
    test_map_navigation()
