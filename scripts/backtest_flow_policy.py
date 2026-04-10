# -*- coding: utf-8 -*-
"""Deterministic flow-policy backtest on explicit seed lists."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from st2rl.gameplay.config import FlowPolicyConfig
from st2rl.gameplay.policy import SimpleFlowPolicy
from st2rl.gameplay.types import FlowAction, GameStateView
from st2rl.protocols.http_cli import HttpCliProtocol, HttpCliProtocolConfig


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest flow policy with fixed seeds.")
    parser.add_argument(
        "--seeds-file",
        type=Path,
        default=Path("configs/backtest/seeds_regression_v1.txt"),
        help="UTF-8 text file, one seed per line.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--character", type=str, default="Ironclad")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    return parser.parse_args()


def _load_seeds(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    seeds = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    seen: set[str] = set()
    unique: list[str] = []
    for seed in seeds:
        if seed in seen:
            continue
        seen.add(seed)
        unique.append(seed)
    return unique


def _state_floor(state: GameStateView) -> int:
    context = state.raw.get("context") or {}
    return _safe_int(context.get("floor"), 0)


def _final_hp(state: GameStateView) -> int:
    return _safe_int(state.player.get("hp"), 0)


def _run_single_seed(
    protocol: HttpCliProtocol,
    policy: SimpleFlowPolicy,
    seed: str,
    *,
    character: str,
    max_steps: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    game_id = ""
    max_floor = 0
    steps = 0
    outcome: dict[str, Any] = {
        "seed": seed,
        "success": False,
        "victory": False,
        "steps": 0,
        "max_floor": 0,
        "final_hp": 0,
        "error": "",
    }
    try:
        start = protocol.start_game(character, seed)
        game_id = start.game_id
        raw_state = start.raw_state or protocol.get_state(game_id)
        state = protocol.adapt_state(raw_state)
        max_floor = max(max_floor, _state_floor(state))
        while steps < max_steps:
            if state.game_over:
                break
            action = protocol.sanitize_action(state, policy.choose_action(state, rng), rng)
            result = protocol.step(game_id, action)
            if result.status != "success":
                recover = protocol.recover_action_from_error(result.raw, state) or FlowAction("proceed")
                retry = protocol.step(game_id, protocol.sanitize_action(state, recover, rng))
                if retry.status != "success":
                    outcome["error"] = str(retry.message or result.message or "step_failed")
                    raw_state = result.last_state or state.raw
                    state = protocol.adapt_state(raw_state)
                    break
                result = retry
            raw_state = result.state or state.raw
            state = protocol.adapt_state(raw_state)
            max_floor = max(max_floor, _state_floor(state))
            steps += 1
        outcome.update(
            {
                "success": True,
                "victory": bool(state.victory),
                "steps": steps,
                "max_floor": max_floor,
                "final_hp": _final_hp(state),
            }
        )
        if steps >= max_steps and not state.game_over:
            outcome["error"] = f"max_steps_exceeded:{max_steps}"
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        outcome["error"] = str(exc)
    finally:
        if game_id:
            try:
                protocol.close_game(game_id)
            except Exception:
                pass
    return outcome


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    done = [row for row in rows if row.get("success")]
    total = len(rows)
    success = len(done)
    floors = [int(row.get("max_floor") or 0) for row in done]
    victories = sum(1 for row in done if row.get("victory"))
    return {
        "total": total,
        "success": success,
        "victory": victories,
        "avg_floor": round(sum(floors) / len(floors), 2) if floors else 0.0,
        "ge5": sum(1 for value in floors if value >= 5),
        "ge10": sum(1 for value in floors if value >= 10),
        "ge15": sum(1 for value in floors if value >= 15),
        "ge17": sum(1 for value in floors if value >= 17),
    }


def main() -> None:
    args = _parse_args()
    seeds = _load_seeds(args.seeds_file)
    if not seeds:
        raise SystemExit("No seeds found in seeds file.")

    protocol = HttpCliProtocol(HttpCliProtocolConfig(timeout_seconds=args.timeout_seconds))
    policy = SimpleFlowPolicy(FlowPolicyConfig())
    protocol.health_check(retries=2)

    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        row = _run_single_seed(
            protocol,
            policy,
            seed,
            character=args.character,
            max_steps=max(100, args.max_steps),
        )
        rows.append(row)
        print(
            f"[{index:03d}/{len(seeds):03d}] seed={seed} success={row['success']} "
            f"victory={row['victory']} floor={row['max_floor']} steps={row['steps']} err={row['error']}"
        )

    payload = {"summary": _summary(rows), "rows": rows}
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
