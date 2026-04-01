"""Train PPO agent on STS2 environment"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from st2rl.envs import STS2Env


class TrainingCallback(BaseCallback):
    """Custom callback for training progress"""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.verbose > 0 and self.n_calls % 100 == 0:
            print(f"Step {self.num_timesteps}: "
                  f"Mean reward (last 100): {np.mean(self.locals['rollout_buffer'].rewards):.2f}")
        return True


def train_ppo(
    total_timesteps: int = 100000,
    save_path: str = "./models",
    log_interval: int = 100
):
    """Train PPO agent

    Args:
        total_timesteps: Total training timesteps
        save_path: Path to save models
        log_interval: Logging interval
    """
    print("Creating STS2 environment...")

    # Create environment
    env = STS2Env(max_steps=10000)

    print("Initializing PPO agent...")

    # Create PPO agent
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs"
    )

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix="ppo_sts2"
    )
    training_callback = TrainingCallback(verbose=1)

    print("Starting training...")

    # Train agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, training_callback]
    )

    print("Training completed!")

    # Save final model
    model.save(f"{save_path}/ppo_sts2_final")
    print(f"Model saved to {save_path}/ppo_sts2_final")

    return model


def test_trained_model(model_path: str, episodes: int = 5):
    """Test trained model

    Args:
        model_path: Path to trained model
        episodes: Number of test episodes
    """
    print(f"Loading model from {model_path}...")

    # Create environment
    env = STS2Env(render_mode='human', max_steps=10000)

    # Load model
    model = PPO.load(model_path)

    print("Starting testing...")

    for episode in range(episodes):
        print(f"\n{'='*60}")
        print(f"Test Episode {episode + 1}/{episodes}")
        print(f"{'='*60}")

        obs, info = env.reset()
        episode_reward = 0
        step = 0

        while True:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step += 1

            if terminated or truncated:
                print(f"\nEpisode finished after {step} steps")
                print(f"Total reward: {episode_reward:.2f}")
                break

    env.close()
    print("\nTesting completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train or test PPO agent on STS2')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--model-path', type=str, default='./models/ppo_sts2_final.zip',
                        help='Path to model (for testing)')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of test episodes')

    args = parser.parse_args()

    if args.mode == 'train':
        print("Make sure Slay the Spire 2 is running with STS2MCP mod enabled!")
        train_ppo(total_timesteps=args.timesteps)
    else:
        test_trained_model(args.model_path, episodes=args.episodes)