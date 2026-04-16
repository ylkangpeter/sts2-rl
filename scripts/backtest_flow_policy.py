# -*- coding: utf-8 -*-
"""Deterministic flow-policy backtest with gate-style dataset collection."""

from __future__ import annotations

import argparse
import json
import random
import time
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
    parser = argparse.ArgumentParser(description="Backtest flow policy with required gate datasets.")
    parser.add_argument(
        "--seeds-file",
        type=Path,
        default=Path("configs/backtest/seeds_regression_v1.txt"),
        help="UTF-8 curated seed file, one seed per line.",
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        default=Path("models/http_cli_rl"),
        help="Root directory containing historical training outputs.",
    )
    parser.add_argument(
        "--dataset-mode",
        type=str,
        choices=("auto", "single"),
        default="auto",
        help="auto: historical>=threshold + curated + random sample; single: only the seeds file.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="manual_file",
        help="Dataset name used in single mode.",
    )
    parser.add_argument(
        "--history-floor-threshold",
        type=int,
        default=17,
        help="Minimum terminal floor required for historical seed inclusion.",
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=16,
        help="How many fresh random seeds to generate per run in auto mode.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--character", type=str, default="Ironclad")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument(
        "--worker-slot",
        type=int,
        default=None,
        help="Optional fixed game-server worker slot for this backtest process.",
    )
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


def _iter_history_rows(root: Path):
    if not root.exists():
        return
    for history_path in sorted(root.glob("**/dashboard/slots/slot_*.history.jsonl")):
        try:
            for raw_line in history_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
        except (OSError, json.JSONDecodeError):
            continue


def _row_floor(row: dict[str, Any]) -> int:
    return _safe_int(row.get("max_floor", row.get("floor")), 0)


def _row_has_terminal_outcome(row: dict[str, Any]) -> bool:
    if bool(row.get("active", False)):
        return False
    if bool(row.get("victory")):
        return True
    hp = _safe_int(row.get("final_hp", row.get("hp")), 0)
    if hp <= 0:
        return True
    return bool(row.get("game_over") or row.get("terminated") or row.get("truncated"))


def _collect_historical_seeds(root: Path, *, min_floor: int) -> list[str]:
    best_by_seed: dict[str, tuple[int, str]] = {}
    for row in _iter_history_rows(root):
        seed = str(row.get("seed") or "").strip()
        if not seed or not _row_has_terminal_outcome(row):
            continue
        floor = _row_floor(row)
        if floor < min_floor:
            continue
        recorded_at = str(row.get("recorded_at") or row.get("updated_at") or "")
        current = best_by_seed.get(seed)
        if current is None or (floor, recorded_at) > current:
            best_by_seed[seed] = (floor, recorded_at)
    ranked = sorted(best_by_seed.items(), key=lambda item: (-item[1][0], item[1][1], item[0]))
    return [seed for seed, _meta in ranked]


def _generate_random_seeds(count: int) -> list[str]:
    count = max(0, int(count))
    if count <= 0:
        return []
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    rng = random.Random()
    return [f"bt_rand_{run_tag}_{index:03d}_{rng.randrange(10_000, 1_000_000)}" for index in range(1, count + 1)]


def _dedupe_preserve_order(seeds: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for seed in seeds:
        if not seed or seed in seen:
            continue
        seen.add(seed)
        unique.append(seed)
    return unique


def _build_datasets(args: argparse.Namespace) -> list[dict[str, Any]]:
    curated_seeds = _load_seeds(args.seeds_file)
    if args.dataset_mode == "single":
        if not curated_seeds:
            raise SystemExit("No seeds found in seeds file.")
        return [
            {
                "name": str(args.dataset_name or "manual_file"),
                "source": str(args.seeds_file),
                "seeds": curated_seeds,
            }
        ]

    historical_seeds = _collect_historical_seeds(
        args.history_root,
        min_floor=max(1, int(args.history_floor_threshold)),
    )
    random_seeds = _generate_random_seeds(args.random_count)

    datasets = [
        {
            "name": f"historical_ge{int(args.history_floor_threshold)}",
            "source": str(args.history_root),
            "seeds": historical_seeds,
        },
        {
            "name": "curated_regression",
            "source": str(args.seeds_file),
            "seeds": curated_seeds,
        },
        {
            "name": "random_fresh",
            "source": "generated_per_run",
            "seeds": random_seeds,
        },
    ]
    for dataset in datasets:
        dataset["seeds"] = _dedupe_preserve_order(list(dataset.get("seeds") or []))
    empty = [dataset["name"] for dataset in datasets if not dataset["seeds"]]
    if empty:
        raise SystemExit(f"Dataset collection produced no seeds for: {', '.join(empty)}")
    return datasets


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
    worker_slot: int | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    game_id = ""
    max_floor = 0
    steps = 0
    state: GameStateView | None = None
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
        start = protocol.start_game(character, seed, worker_slot=worker_slot)
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
        completed = bool(state and state.game_over and not outcome["error"])
        outcome.update(
            {
                "success": completed,
                "victory": bool(state.victory) if state else False,
                "steps": steps,
                "max_floor": max_floor,
                "final_hp": _final_hp(state) if state else 0,
            }
        )
        if state is not None and steps >= max_steps and not state.game_over:
            outcome["error"] = f"max_steps_exceeded:{max_steps}"
            outcome["success"] = False
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


def _run_dataset(
    protocol: HttpCliProtocol,
    policy: SimpleFlowPolicy,
    dataset: dict[str, Any],
    *,
    character: str,
    max_steps: int,
    worker_slot: int | None = None,
) -> dict[str, Any]:
    name = str(dataset.get("name") or "dataset")
    seeds = list(dataset.get("seeds") or [])
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        row = _run_single_seed(
            protocol,
            policy,
            seed,
            character=character,
            max_steps=max(100, max_steps),
            worker_slot=worker_slot,
        )
        row["dataset"] = name
        rows.append(row)
        print(
            f"[{name} {index:03d}/{len(seeds):03d}] seed={seed} success={row['success']} "
            f"victory={row['victory']} floor={row['max_floor']} steps={row['steps']} err={row['error']}"
        )
    return {
        "name": name,
        "source": str(dataset.get("source") or ""),
        "seed_count": len(seeds),
        "seeds": seeds,
        "summary": _summary(rows),
        "rows": rows,
    }


def main() -> None:
    args = _parse_args()
    datasets = _build_datasets(args)

    protocol = HttpCliProtocol(HttpCliProtocolConfig(timeout_seconds=args.timeout_seconds))
    policy = SimpleFlowPolicy(FlowPolicyConfig())
    protocol.health_check(retries=2)

    dataset_payloads: list[dict[str, Any]] = []
    merged_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        payload = _run_dataset(
            protocol,
            policy,
            dataset,
            character=args.character,
            max_steps=max(100, args.max_steps),
            worker_slot=args.worker_slot,
        )
        dataset_payloads.append(payload)
        merged_rows.extend(payload["rows"])
        print(json.dumps({"dataset": payload["name"], **payload["summary"]}, ensure_ascii=False, indent=2))

    payload = {
        "config": {
            "dataset_mode": args.dataset_mode,
            "character": args.character,
            "max_steps": args.max_steps,
            "timeout_seconds": args.timeout_seconds,
            "history_root": str(args.history_root),
            "history_floor_threshold": int(args.history_floor_threshold),
            "curated_seeds_file": str(args.seeds_file),
            "random_count": int(args.random_count),
        },
        "datasets": dataset_payloads,
        "merged_summary": _summary(merged_rows),
    }
    print(json.dumps(payload["merged_summary"], ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
