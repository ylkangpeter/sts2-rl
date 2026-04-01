# sts2-rl

基于本地 `sts2-cli` fork 的 Slay the Spire 2 强化学习项目。当前主线是通过 `sts2-cli` 提供的 HTTP 服务做回归测试、并发训练和训练监控。

## 项目关系

- 本仓库负责训练、策略流程、奖励函数、dashboard 和运维脚本。
- 运行时依赖另一个本地仓库 `sts2-cli`。
- 推荐目录结构：

```text
workspace/
  sts2-cli/
  st2rl/
```

- 如果你的 `sts2-cli` 不在相邻目录，设置 `STS2_CLI_ROOT` 即可。

## 依赖

- Python 3.10+
- [Slay the Spire 2](https://store.steampowered.com/app/2868840/Slay_the_Spire_2/)
- [.NET 9+ SDK](https://dotnet.microsoft.com/download)
- 一个可运行 HTTP 服务的本地 `sts2-cli` fork

安装：

```powershell
cd .\st2rl
pip install -r requirements.txt
pip install -e .
```

如果只想装训练扩展，也可以：

```powershell
pip install -e ".[rl]"
```

## 配置

运行栈配置在 [configs/runtime_stack.json](D:/github/st2rl/configs/runtime_stack.json)。

- `projects.sts2_cli_root` 默认指向相邻目录 `..\sts2-cli`
- `projects.game_dir` 默认留空，优先读取环境变量 `STS2_GAME_DIR`
- 所有路径都改成了相对路径，方便迁移到别的机器

推荐先设置游戏目录：

```powershell
$env:STS2_GAME_DIR="C:\path\to\SlayTheSpire2"
```

如果 `sts2-cli` 不在相邻目录，再补一个：

```powershell
$env:STS2_CLI_ROOT="C:\path\to\sts2-cli"
```

## 使用方式

启动 service + dashboard：

```powershell
cd .\st2rl
.\scripts\start_stack.ps1
```

如果也要一起拉起训练 client：

```powershell
.\scripts\start_stack.ps1 -IncludeClient
```

手工启动 `sts2-cli` HTTP 服务：

```powershell
cd ..\sts2-cli
python .\python\http_game_service.py
```

健康检查：

```powershell
Invoke-RestMethod http://localhost:5000/health
```

## 回归测试

回归链路入口是 [examples/test_cli_game.py](D:/github/st2rl/examples/test_cli_game.py)：

```powershell
cd .\st2rl
python .\examples\test_cli_game.py --workers 4 --rounds 50
```

当前流程拆分为：

- `src/st2rl/gameplay/config.py`
- `src/st2rl/gameplay/types.py`
- `src/st2rl/gameplay/policy.py`
- `src/st2rl/gameplay/runner.py`
- `src/st2rl/protocols/http_cli.py`

## 正式训练

训练入口是 [scripts/train_http_cli_rl.py](D:/github/st2rl/scripts/train_http_cli_rl.py)，默认配置在 [configs/train_http_cli_rl.yaml](D:/github/st2rl/configs/train_http_cli_rl.yaml)。

启动训练：

```powershell
cd .\st2rl
python .\scripts\train_http_cli_rl.py
```

命令行覆盖示例：

```powershell
python .\scripts\train_http_cli_rl.py --num-envs 4 --vec-env subproc --experiment-name baseline_reward_v1
```

训练输出目录：

```text
models/http_cli_rl/<experiment_name>/<run_id>/
```

## Dashboard

启动 dashboard：

```powershell
cd .\st2rl
python .\scripts\training_dashboard.py
```

打开 [http://127.0.0.1:8787](http://127.0.0.1:8787) 查看训练进度、槽位状态、最近 episode 和 checkpoint。

## 备注

- [README_UNIFIED.md](D:/github/st2rl/README_UNIFIED.md) 保留为旧架构草稿，当前以本 README 为准。
- 当前正式训练链路已经不再依赖旧的 `STS2MCP` 工作流。
