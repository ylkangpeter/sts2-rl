# sts2-rl

Reinforcement learning and regression tooling for Slay the Spire 2, designed to run against a local sibling checkout of `sts2-cli`.

Recommended layout:

```text
workspace/
  sts2-cli/
  sts2-rl/
```

## Requirements

- Python 3.10+
- .NET 9 SDK
- Slay the Spire 2 installed locally
- A local checkout of `sts2-cli`

Install:

```powershell
cd .\sts2-rl
pip install -r requirements.txt
pip install -e .
```

If you only want the RL extras:

```powershell
pip install -e ".[rl]"
```

## Privacy-Safe Runtime Config

Tracked config lives in:

- `configs/runtime_stack.json`

This file is a safe template and should stay machine-agnostic.

Local overrides belong in:

- `configs/runtime_stack.local.json`

This file is Git-ignored, so it is safe to put your local paths there.

Starter example:

- `configs/runtime_stack.local.example.json`

Typical local file:

```json
{
  "python_executable": "D:\\venvs\\sts2\\Scripts\\python.exe",
  "projects": {
    "game_dir": "D:\\game\\SlayTheSpire2"
  },
  "client": {
    "enabled": true,
    "args": ["--experiment-name", "baseline_reward_v1"]
  }
}
```

Runtime loading order is:

1. `runtime_stack.json`
2. `runtime_stack.local.json` overrides it when present

The following scripts support this automatically:

- `scripts/start_stack.ps1`
- `scripts/training_dashboard.py`
- `scripts/training_watchdog.py`
- `scripts/training_session_supervisor.py`

You can also avoid storing the game path in a file and set it via env var:

```powershell
$env:STS2_GAME_DIR="C:\path\to\SlayTheSpire2"
```

If `sts2-cli` is not in a sibling directory, set:

```powershell
$env:STS2_CLI_ROOT="C:\path\to\sts2-cli"
```

## Quick Start

1. Prepare `sts2-cli`.

```powershell
cd ..\sts2-cli
pip install -r requirements.txt
```

2. Create your local runtime override.

```powershell
copy .\configs\runtime_stack.local.example.json .\configs\runtime_stack.local.json
```

Then edit `configs/runtime_stack.local.json` with your local Python path and game directory.

3. Start the service and dashboard.

```powershell
cd ..\sts2-rl
.\scripts\start_stack.ps1
```

If you also want to launch the training client:

```powershell
.\scripts\start_stack.ps1 -IncludeClient
```

4. Open the dashboard:

```text
http://127.0.0.1:8787
```

## Manual Commands

Start the HTTP service manually:

```powershell
cd ..\sts2-cli
python .\python\http_game_service.py
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

Run a smoke test:

```powershell
cd ..\sts2-rl
python .\examples\test_cli_game.py --workers 4 --rounds 50
```

Train PPO:

```powershell
python .\scripts\train_http_cli_rl.py --experiment-name baseline_reward_v1
```

Outputs are written under:

```text
models/http_cli_rl/<experiment_name>/<run_id>/
```

## Repo Hygiene

These local/runtime files are intentionally not committed:

- `configs/runtime_stack.local.json`
- `logs/`
- `models/`

That keeps machine-specific paths and large runtime artifacts out of Git history.
