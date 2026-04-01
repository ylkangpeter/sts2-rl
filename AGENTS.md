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
- Avoid repeated heavy filesystem scans and repeated full historical recomputation during refresh.

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

## Repository-Specific Notes

- Python source files should start with `# -*- coding: utf-8 -*-`.
- Keep the runtime simple and deterministic on Windows.
- Avoid spawning visible console windows for worker subprocesses.
- This repository depends on a local `sts2-cli` checkout or fork.
- Prefer resolving `sts2-cli` from `STS2_CLI_ROOT`, then sibling directories such as `../sts2-cli`.
- Do not hardcode personal absolute paths in committed docs, configs, or code defaults.
