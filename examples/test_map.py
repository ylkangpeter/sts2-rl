#!/usr/bin/env python3
"""Test script for map state and actions"""

from st2rl.envs import STS2Env

def test_map_environment():
    """Test the environment with map state"""
    print("Testing STS2 environment with map state...")
    
    # Create environment
    env = STS2Env(render_mode='human')
    
    try:
        # Reset environment
        obs, info = env.reset()
        print(f"Initial state: {info['state_type']}")
        
        if info['state_type'] == 'map':
            print("Successfully connected to game in map state!")
            
            # Test getting map state
            current_state = env.client.get_state()
            map_data = current_state.get('map', {})
            
            print("Map information:")
            print(f"Current position: {map_data.get('current_position', {})}")
            print(f"Next options: {len(map_data.get('next_options', []))} available")
            
            # Test map actions
            if map_data.get('next_options'):
                print("\nAvailable map nodes:")
                for i, option in enumerate(map_data['next_options']):
                    print(f"Option {i}: {option['type']} at ({option['col']}, {option['row']})")
            
        else:
            print(f"Current state is not map: {info['state_type']}")
            
    finally:
        env.close()

if __name__ == "__main__":
    test_map_environment()
