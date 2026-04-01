# STS2 RL Unified Framework

此文档保留为早期统一架构草稿。当前可运行方案请优先参考 `README.md`，它描述的是基于本地 `sts2-cli` fork 的现行训练栈。

统一架构的 Slay the Spire 2 强化学习框架，支持 Headless 和 UI 两种模式，模型互通。

## 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                    训练流程 (Training)                        │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐  │
│  │   Trainer    │──────│  Environment │──────│  Model   │  │
│  │              │      │ (headless/ui)│      │  (PPO)   │  │
│  └──────────────┘      └──────────────┘      └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    运行流程 (Inference)                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐  │
│  │    Runner    │──────│  Environment │──────│  Model   │  │
│  │              │      │ (headless/ui)│      │(trained) │  │
│  └──────────────┘      └──────────────┘      └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 核心特性

1. **模型互通**: Headless 训练的模型可以直接在 UI 模式下使用
2. **统一接口**: 两种模式使用相同的 API
3. **继续训练**: 支持加载已有模型继续训练
4. **灵活配置**: 通过 YAML 配置文件管理所有参数

## 快速开始

### 1. Headless 训练（快速预训练）

```bash
python scripts/train.py --config configs/train_headless.yaml
```

或使用命令行参数：

```bash
python scripts/train.py --mode headless --timesteps 1000000 --character Ironclad
```

### 2. UI 微调（在真实游戏上微调）

```bash
python scripts/finetune.py --base-model models/unified/ppo_sts2_headless --timesteps 100000
```

### 3. 运行训练好的模型

```bash
# UI 模式
python scripts/run.py --model models/unified/ppo_sts2_finetuned --mode ui --episodes 10

# Headless 模式
python scripts/run.py --model models/unified/ppo_sts2_headless --mode headless --episodes 100
```

## Python API 使用

### 基础用法

```python
from st2rl import UnifiedTrainer, UnifiedRunner, EnvironmentFactory

# 1. Headless 训练
config = {
    'mode': 'headless',
    'environment': {'character': 'Ironclad', 'max_steps': 500},
    'training': {'total_timesteps': 100000},
}

trainer = UnifiedTrainer(config)
trainer.create_environment()
trainer.create_model()
trainer.train(save_path='models/my_model')

# 2. UI 微调
trainer.switch_mode('ui')
trainer.continue_training(
    model_path='models/my_model',
    additional_timesteps=10000,
    save_path='models/my_model_finetuned'
)

# 3. 运行模型
runner = UnifiedRunner(
    model_path='models/my_model_finetuned',
    mode='ui'
)
runner.run(num_episodes=10)
```

### 直接使用环境

```python
from st2rl.environments import EnvironmentFactory
from st2rl.models import UnifiedPPOModel

# 创建环境
env = EnvironmentFactory.create('headless', character='Ironclad')

# 创建并训练模型
model = UnifiedPPOModel(env)
model.train(total_timesteps=10000)

# 保存
model.save('models/my_model')

# 加载并运行
loaded_model = UnifiedPPOModel.load('models/my_model', env=env)

obs, _ = env.reset()
for _ in range(1000):
    action, _ = loaded_model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## 配置文件说明

### Headless 训练配置

```yaml
mode: headless

environment:
  character: Ironclad
  max_steps: 500
  verbose: false

model:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99

training:
  total_timesteps: 1000000
  save_freq: 10000

paths:
  save_dir: "./models/unified"
  model_name: "ppo_sts2_headless"
```

### UI 训练配置

```yaml
mode: ui

environment:
  character: Ironclad
  max_steps: 5000
  host: localhost
  agent_port: 8080
  mcp_port: 15526
  use_mcp: true

model:
  learning_rate: 3.0e-4
  # ... 其他参数

training:
  total_timesteps: 100000
  save_freq: 10000
```

## 工作流程建议

### 推荐的工作流程

1. **Headless 预训练**
   - 使用 Headless 模式进行大规模预训练（100万+ 步）
   - 快速迭代和实验不同的超参数

2. **UI 微调**
   - 加载 Headless 预训练的模型
   - 在真实游戏上进行微调（10万步）
   - 使用更低的学习率

3. **评估和部署**
   - 在 UI 模式下评估模型性能
   - 部署到实际游戏环境

### 示例脚本

```bash
# 1. Headless 预训练
python scripts/train.py \
    --mode headless \
    --timesteps 1000000 \
    --character Ironclad

# 2. UI 微调
python scripts/finetune.py \
    --base-model models/unified/ppo_sts2_headless \
    --timesteps 100000

# 3. 运行评估
python scripts/run.py \
    --model models/unified/ppo_sts2_finetuned \
    --mode ui \
    --episodes 10 \
    --render
```

## 模块说明

### Core 模块

- `state_adapter.py`: 统一状态表示，将 headless 和 UI 的状态转换为相同格式
- `spaces.py`: 统一的观察和动作空间定义

### Environments 模块

- `base.py`: 环境基类，定义统一接口
- `headless.py`: Headless 环境（使用 sts2-cli）
- `ui.py`: UI 环境（使用 STS2-Agent 和 STS2MCP）
- `factory.py`: 环境工厂，根据配置创建对应环境

### Models 模块

- `ppo_model.py`: 统一的 PPO 模型，支持保存/加载/继续训练

### Training 模块

- `trainer.py`: 统一训练器，支持模式切换
- `callbacks.py`: 训练回调（检查点、进度等）

### Inference 模块

- `runner.py`: 统一运行器，用于部署训练好的模型

## 注意事项

1. **Headless 模式需要**: sts2-cli 项目和 .NET SDK
2. **UI 模式需要**: 游戏运行并启用 STS2-Agent Mod
3. **模型互通**: 两种模式训练的模型格式完全相同
4. **观察空间**: 统一为 30 维向量（可在 `state_adapter.py` 中调整）

## 扩展开发

### 添加新的训练算法

在 `models/` 目录下创建新的模型类，继承相同的接口：

```python
class UnifiedA2CModel:
    def train(self, total_timesteps, callback=None):
        # 实现训练逻辑
        pass
    
    def save(self, path):
        # 实现保存逻辑
        pass
    
    @classmethod
    def load(cls, path, env=None):
        # 实现加载逻辑
        pass
```

### 自定义奖励函数

继承环境基类并覆盖 `_calculate_reward` 方法：

```python
class CustomRewardEnv(UnifiedSTS2Env):
    def _calculate_reward(self) -> float:
        # 自定义奖励计算
        reward = 0.0
        # ... 你的逻辑
        return reward
```

## 问题排查

### Headless 模式连接失败

- 检查 sts2-cli 路径是否正确
- 确认 .NET SDK 已安装
- 检查游戏文件路径

### UI 模式连接失败

- 确认游戏已启动
- 检查 STS2-Agent Mod 是否启用
- 确认端口配置正确（默认 8080）

### 模型加载失败

- 确认模型文件存在（.zip 和 .config.json）
- 检查模型版本兼容性
