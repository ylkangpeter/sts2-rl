"""Test environment with random agent"""

from st2rl.envs import STS2Env


def test_random_agent(episodes: int = 1, steps: int = 1000):
    """Test environment with random agent

    Args:
        episodes: Number of episodes to run
        steps: Maximum steps per episode
    """
    # Create environment
    env = STS2Env(render_mode='human', max_steps=steps)

    for episode in range(episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{episodes}")
        print(f"{'='*60}")

        # Reset environment
        obs, info = env.reset()

        episode_reward = 0
        step = 0

        while step < steps:
            # Sample random action
            action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step += 1

            # Check if episode ended
            if terminated or truncated:
                print(f"\nEpisode finished after {step} steps")
                print(f"Total reward: {episode_reward:.2f}")
                print(f"Terminated: {terminated}, Truncated: {truncated}")
                break

    env.close()
    print("\nTest completed!")


if __name__ == "__main__":
    print("Starting random agent test...")
    print("Make sure Slay the Spire 2 is running with STS2MCP mod enabled!")
    print()

    test_random_agent(episodes=1, steps=1000)