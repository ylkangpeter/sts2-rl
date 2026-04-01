"""Test script for STS2-Agent integration"""

import time
from st2rl.envs import STS2Env
from st2rl.utils.logger import get_run_logger


def test_sts2_agent_integration():
    """Test integration with STS2-Agent"""
    logger = get_run_logger()
    logger.info("Starting STS2-Agent integration test...")
    
    # Create environment
    env = STS2Env(
        host='localhost',
        port=8080,
        render_mode='human',
        character_index=0  # Select first character
    )
    
    try:
        # Reset environment (should navigate from main menu to game)
        logger.info("=== Resetting environment ===")
        obs, info = env.reset()
        logger.info(f"Initial state: {info['state_type']}")
        
        # Test a few steps
        logger.info("=== Testing a few steps ===")
        for step in range(5):
            # Get random action
            action = env.action_space.sample()
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            
            logger.info(f"Step {step+1}: State={info['state_type']}, Reward={reward:.2f}")
            
            # Check if episode terminated
            if terminated or truncated:
                logger.info("Episode terminated")
                break
            
            # Wait a bit to see what's happening
            time.sleep(1)
            
    finally:
        # Close environment
        env.close()
        logger.info("Test completed")


if __name__ == "__main__":
    test_sts2_agent_integration()
