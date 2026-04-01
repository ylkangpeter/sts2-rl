"""Simple environment test"""

import gymnasium as gym
import st2rl


def test_environment():
    """Test basic environment functionality"""
    print("Creating STS2 environment...")
    env = gym.make('STS2-v0')

    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    print("\nResetting environment...")
    print("Make sure Slay the Spire 2 is running with STS2MCP mod enabled!")

    try:
        obs, info = env.reset()
        print("Environment reset successfully!")
        print(f"Observation keys: {obs.keys()}")
        print(f"Info: {info}")

        print("\nSampling random action...")
        action = env.action_space.sample()
        print(f"Action: {action}")

        print("\nStepping environment...")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")

        print("\nEnvironment test completed successfully!")

    except Exception as e:
        print(f"\nError during test: {e}")
        print("Make sure:")
        print("1. Slay the Spire 2 is running")
        print("2. STS2MCP mod is enabled")
        print("3. A game run is in progress")

    finally:
        env.close()


if __name__ == "__main__":
    test_environment()