# -*- coding: utf-8 -*-
"""Quick start example for unified STS2 RL

This example demonstrates:
1. Training in headless mode
2. Fine-tuning in UI mode
3. Running the trained model
"""

import sys

from st2rl.training import UnifiedTrainer
from st2rl.inference import UnifiedRunner


def example_1_train_headless():
    """示例1: 在 Headless 模式下训练"""
    print("\n" + "=" * 60)
    print("Example 1: Train in Headless Mode")
    print("=" * 60)

    config = {
        'mode': 'headless',
        'environment': {
            'character': 'Ironclad',
            'max_steps': 500,
        },
        'model': {
            'learning_rate': 3e-4,
            'verbose': 1,
        },
        'training': {
            'total_timesteps': 10000,  # 少量步数用于演示
            'save_freq': 5000,
        },
        'paths': {
            'save_dir': './models/unified',
            'model_name': 'ppo_sts2_demo',
        }
    }

    trainer = UnifiedTrainer(config)
    trainer.create_environment()
    trainer.create_model()
    trainer.train()

    return './models/unified/ppo_sts2_demo'


def example_2_finetune_ui(model_path: str):
    """示例2: 在 UI 模式下微调"""
    print("\n" + "=" * 60)
    print("Example 2: Fine-tune in UI Mode")
    print("=" * 60)

    config = {
        'mode': 'ui',
        'environment': {
            'character': 'Ironclad',
            'max_steps': 5000,
            'host': 'localhost',
            'agent_port': 8080,
            'mcp_port': 15526,
            'use_mcp': True,
        },
        'model': {
            'learning_rate': 1e-4,  # 更低的学习率
            'verbose': 1,
        },
        'training': {
            'total_timesteps': 5000,
            'save_freq': 1000,
        },
        'paths': {
            'save_dir': './models/unified',
            'model_name': 'ppo_sts2_finetuned',
        }
    }

    trainer = UnifiedTrainer(config)
    trainer.create_environment()
    trainer.create_model(model_path)  # 加载预训练模型
    trainer.train()

    return './models/unified/ppo_sts2_finetuned'


def example_3_run_model(model_path: str):
    """示例3: 运行训练好的模型"""
    print("\n" + "=" * 60)
    print("Example 3: Run Trained Model")
    print("=" * 60)

    runner = UnifiedRunner(
        model_path=model_path,
        mode='ui',
        env_config={
            'character': 'Ironclad',
            'max_steps': 5000,
        },
        deterministic=True
    )

    # 运行5局
    for result in runner.run(num_episodes=5):
        print(f"Episode {result['episode']}: "
              f"Reward={result['total_reward']:.2f}, "
              f"Steps={result['steps']}")

    runner.close()


def example_4_direct_usage():
    """示例4: 直接使用 API"""
    print("\n" + "=" * 60)
    print("Example 4: Direct API Usage")
    print("=" * 60)

    from st2rl.environments import EnvironmentFactory
    from st2rl.models import UnifiedPPOModel

    # 创建环境
    env = EnvironmentFactory.create('headless', character='Ironclad', max_steps=500)

    # 创建模型
    model = UnifiedPPOModel(env, learning_rate=3e-4)

    # 训练
    model.train(total_timesteps=10000)

    # 保存
    model.save('./models/unified/my_model')

    # 加载并运行
    loaded_model = UnifiedPPOModel.load('./models/unified/my_model')

    obs, _ = env.reset()
    for _ in range(100):
        action, _ = loaded_model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()


if __name__ == '__main__':
    print("STS2 RL Unified Framework - Quick Start Examples")
    print("=" * 60)

    # 选择要运行的示例
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=int, choices=[1, 2, 3, 4], default=1,
                        help='Which example to run')
    parser.add_argument('--model', type=str, help='Model path for example 2 or 3')
    args = parser.parse_args()

    if args.example == 1:
        model_path = example_1_train_headless()
        print(f"\nModel saved to: {model_path}")

    elif args.example == 2:
        if not args.model:
            print("Please provide --model for fine-tuning")
            sys.exit(1)
        model_path = example_2_finetune_ui(args.model)
        print(f"\nFine-tuned model saved to: {model_path}")

    elif args.example == 3:
        if not args.model:
            print("Please provide --model for running")
            sys.exit(1)
        example_3_run_model(args.model)

    elif args.example == 4:
        example_4_direct_usage()

    print("\nDone!")
