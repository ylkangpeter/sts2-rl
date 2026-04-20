# Training Stack Contract

## Core Goal

This repository exists to train an RL agent for Slay the Spire 2 reliably and continuously. Training correctness is more important than auxiliary tooling.

## Required Runtime Shape

Every training launch should have exactly three top-level processes:

1. `dashboard`
2. `training client`
3. `game server`

The `training client` and `game server` may each manage their own internal worker pools, but starting a new game must not create a brand new top-level process. A game end should clean per-game state completely so the next game starts cleanly on the reused worker.

## Control Model

- `start` means: bring up the game server if needed, then start the training client.
- `stop` means: stop the training client and the game server stack completely.
- Do not expose or rely on `pause` / `resume`.
- Avoid auto-restart loops that keep training alive while known bugs remain unresolved.
- If an abnormal session is discovered, the preferred workflow is:
  1. capture the seed / session details,
  2. locate and fix the bug,
  3. restart the full runtime cleanly.

## Data Flow Expectations

- Live dashboard data should primarily come from in-memory state and client/server HTTP communication.
- Disk should be used mainly for checkpoints, compact history, and incident evidence.
- After process bootstrap and one-time startup loading complete, runtime code should treat disk as write-mostly.
- Outside startup/bootstrap, processes should not repeatedly read disk to recover live state, poll peer status, rebuild operator views, or exchange control information.
- Cross-process runtime data flow must use HTTP or another in-memory IPC channel. Do not implement read-after-write file loops between `dashboard`, `training client`, and `game server`.
- Avoid repeated heavy filesystem scans and repeated full historical recomputation during refresh.
- After runtime bootstrap completes, live status, trend, slot/session summaries, and operator control flow must not depend on reading files from disk.
- Runtime disk usage should be write-oriented only: checkpoints, compact append-only history, crash evidence, and other forensics are allowed, but live state/statistics/control should be served from memory or HTTP APIs.
- If one process needs data from another process during runtime, expose it over HTTP or another in-memory IPC path instead of implementing read-after-write file loops.

## Resume Semantics

- Restarting training should continue from the latest existing checkpoint/model by default.
- Starting from scratch should happen only when explicitly requested.
- Operator-facing status should clearly show whether the current run resumed from a previous checkpoint and which checkpoint was used.

## Correctness Definition

Training is considered healthy only when all of the following hold:

1. New games keep starting continuously.
2. Each game reaches a real terminal outcome:
   - death, with HP reduced to 0 or below, or
   - final victory / clear.
3. If a game ends without victory and final HP is not 0, treat it as suspicious.
4. If a single game runs too long without finishing, treat it as suspicious.
   - Practical threshold: 10 minutes per game.
   - Shorter heuristics such as per-floor timing can be used as hints, but final judgment should be based on outcome and end-state, not only service liveness.
5. Deadlocks, repeated no-op loops, stale state, or sessions that stop making progress are abnormalities and should be investigated.

## Monitoring Expectations For Future AI Operators

- Monitor training as an ongoing operational responsibility, not a one-time check.
- Judge health from outcomes, not only from HTTP 200s.
- Use concrete seeds / game IDs when investigating.
- Prefer fixing root causes over masking problems with endless restarts.
- Surface suspicious sessions explicitly in telemetry/dashboard output so operators do not need to infer them from raw logs.

## Script And Runtime Output Discipline

- Never treat a backtest, training run, or server probe as healthy just because the process exits with code 0 or prints a summary.
- Inspect script output, result JSON, worker debug endpoints, and relevant stderr tails for every run used to make a decision.
- Any non-empty per-seed `err`, `error`, Python/C# exception, `MissingMethodException`, `NullReferenceException`, `Game not found`, timeout, deadlock hint, or warning from the game server/headless runtime must be investigated.
- Do not filter these problems out of metrics and continue. Either fix the root cause, prove with a narrow explanation that the warning is intentionally benign and suppress/record it appropriately, or stop and report the blocker.
- Backtest metrics are valid only after runtime quality is clean enough that errors, exceptions, and warnings are not contaminating the sample.
- If a script emits noisy warnings that are expected in headless mode, update the mock/headless layer so the warning no longer appears, or make the script fail fast with a clear diagnostic until the warning is handled.
- When a server-side fix changes `sts2-cli`, terminate existing headless workers and rebuild before validating; stale worker processes must not be mixed with new code.

## Policy-Change Backtest Gate (Required)

- Any change that affects gameplay decisions (card reward, shop, rest/campfire, event choice, map routing, combat fallback) must run deterministic seed backtests before restarting long training.
- Backtest dataset must include three parts every time:
  1. all historical seeds whose terminal floor is `>=17` at the moment the script runs,
  2. a fixed curated seed set with broad floor distribution (at least floor bands around `5`, `10`, and `15`),
  3. an additional random seed sample regenerated every run to reduce overfitting risk.
- The fixed curated seeds should be versioned in-repo (default: `configs/backtest/seeds_regression_v1.txt`) and reused for before/after comparison unless explicitly replaced.
- Acceptance rule:
  1. none of the three datasets may regress on grouped core metrics (`avg_floor`, `ge10`, `ge15`, `ge17`),
  2. overall merged metrics should improve over baseline.
- Evaluation is distribution-level, not per-seed mandatory dominance. Do not require every single seed to outperform baseline.
- If any dataset regresses, rollback or retune before resuming long training.

## Repository-Specific Notes

- Python source files should start with `# -*- coding: utf-8 -*-`.
- Keep the runtime simple and deterministic on Windows.
- Avoid spawning visible console windows for worker subprocesses.
- This repository depends on a local `sts2-cli` checkout or fork.
- Prefer resolving `sts2-cli` from `STS2_CLI_ROOT`, then sibling directories such as `../sts2-cli`.
- Do not hardcode personal absolute paths in committed docs, configs, or code defaults.
