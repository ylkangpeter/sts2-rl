# -*- coding: utf-8 -*-
"""Deterministic flow-policy backtest with gate-style dataset collection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import threading
import time
from pathlib import Path
from typing import Any

from st2rl.gameplay.config import FlowPolicyConfig
from st2rl.gameplay.policy import SimpleFlowPolicy
from st2rl.gameplay.types import FlowAction, GameStateView
from st2rl.protocols.base import ProtocolStepResult
from st2rl.protocols.http_cli import HttpCliProtocol, HttpCliProtocolConfig
from st2rl.training.backtest_cache import (
    BacktestRolloutCache,
    action_hash,
    action_payload,
    source_digest,
    state_hash,
)


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
        choices=("auto", "single", "focused"),
        default="auto",
        help=(
            "focused: high-floor + suspicious historical seeds for first-pass tuning; "
            "auto: full gate with historical>=threshold + curated + random sample; "
            "single: only the seeds file."
        ),
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
        "--history-limit",
        type=int,
        default=0,
        help="Maximum historical seeds to include in auto mode; 0 means all matching seeds.",
    )
    parser.add_argument(
        "--focused-high-floor-threshold",
        type=int,
        default=17,
        help="Minimum global floor for focused high-floor seed inclusion.",
    )
    parser.add_argument(
        "--focused-anomaly-floor-threshold",
        type=int,
        default=10,
        help="Minimum global floor for suspicious focused seed inclusion unless explicit errors exist.",
    )
    parser.add_argument(
        "--focused-high-floor-limit",
        type=int,
        default=240,
        help="Maximum high-floor historical seeds in focused mode.",
    )
    parser.add_argument(
        "--focused-anomaly-limit",
        type=int,
        default=120,
        help="Maximum suspicious historical seeds in focused mode.",
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=16,
        help="How many fresh random seeds to generate per run in auto mode.",
    )
    parser.add_argument(
        "--random-run-tag",
        type=str,
        default="",
        help="Optional tag for reproducible random seed generation within one before/after gate run.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--character", type=str, default="Ironclad")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--timeout-seconds", type=int, default=10)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("logs/backtest_cache"),
        help="Directory for exact deterministic rollout cache.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable exact rollout cache reads and writes.",
    )
    parser.add_argument(
        "--cache-version",
        type=str,
        default=None,
        help="Optional manual cache namespace suffix; defaults to source and engine digests.",
    )
    parser.add_argument(
        "--worker-slot",
        type=int,
        default=None,
        help="Optional fixed game-server worker slot for this backtest process.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent backtest games to run against reusable game-server workers.",
    )
    parser.add_argument(
        "--allow-runtime-errors",
        action="store_true",
        help="Write results but do not fail the process when any seed reports an error.",
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


def _global_floor(act: int, act_floor: int) -> int:
    act = max(1, int(act))
    act_floor = max(0, int(act_floor))
    if act <= 1:
        return act_floor
    if act == 2:
        return 18 + act_floor
    if act == 3:
        return 35 + act_floor
    return 51 + act_floor + max(0, act - 4) * 17


def _global_floor_from_progress(progress: Any) -> int:
    value = _safe_int(progress, 0)
    if value <= 0:
        return 0
    if value < 100:
        return value
    return _global_floor(value // 100 + 1, value % 100)


def _row_global_floor(row: dict[str, Any]) -> int:
    for key in ("max_global_floor", "final_global_floor", "global_floor"):
        value = _safe_int(row.get(key), 0)
        if value > 0:
            return value
    progress_floor = _global_floor_from_progress(row.get("max_progress"))
    if progress_floor > 0:
        return progress_floor
    act = _safe_int(row.get("max_global_act") or row.get("max_act") or row.get("act"), 0)
    floor = _safe_int(row.get("max_global_act_floor") or row.get("final_floor") or row.get("floor"), 0)
    if act > 0 and floor > 0:
        return _global_floor(act, floor)
    return _row_floor(row)


def _row_is_suspicious(row: dict[str, Any]) -> bool:
    if row.get("error"):
        return True
    flags = row.get("anomaly_flags") or []
    if isinstance(flags, list) and flags:
        return True
    if bool(row.get("truncated")) and not bool(row.get("victory")):
        return True
    if not bool(row.get("active", False)) and not bool(row.get("victory")):
        final_hp = _safe_int(row.get("final_hp", row.get("hp")), 0)
        if final_hp > 0:
            return True
    return False


def _row_has_terminal_outcome(row: dict[str, Any]) -> bool:
    if bool(row.get("active", False)):
        return False
    if bool(row.get("victory")):
        return True
    hp = _safe_int(row.get("final_hp", row.get("hp")), 0)
    if hp <= 0:
        return True
    return bool(row.get("game_over") or row.get("terminated") or row.get("truncated"))


def _collect_historical_seeds(root: Path, *, min_floor: int, limit: int = 0) -> list[str]:
    best_by_seed: dict[str, tuple[int, str]] = {}
    for row in _iter_history_rows(root):
        seed = str(row.get("seed") or "").strip()
        if not seed or not _row_has_terminal_outcome(row):
            continue
        floor = _row_global_floor(row)
        if floor < min_floor:
            continue
        recorded_at = str(row.get("recorded_at") or row.get("updated_at") or "")
        current = best_by_seed.get(seed)
        if current is None or (floor, recorded_at) > current:
            best_by_seed[seed] = (floor, recorded_at)
    ranked = sorted(best_by_seed.items(), key=lambda item: (-item[1][0], item[1][1], item[0]))
    if limit > 0:
        ranked = ranked[:limit]
    return [seed for seed, _meta in ranked]


def _collect_focused_historical_seeds(
    root: Path,
    *,
    high_floor_threshold: int,
    anomaly_floor_threshold: int,
    high_floor_limit: int,
    anomaly_limit: int,
) -> tuple[list[str], list[str]]:
    high_by_seed: dict[str, tuple[int, str]] = {}
    anomaly_by_seed: dict[str, tuple[int, str]] = {}
    for row in _iter_history_rows(root):
        seed = str(row.get("seed") or "").strip()
        if not seed or not _row_has_terminal_outcome(row):
            continue
        global_floor = _row_global_floor(row)
        recorded_at = str(row.get("recorded_at") or row.get("updated_at") or "")
        if global_floor >= high_floor_threshold:
            current = high_by_seed.get(seed)
            if current is None or (global_floor, recorded_at) > current:
                high_by_seed[seed] = (global_floor, recorded_at)
        if _row_is_suspicious(row) and (global_floor >= anomaly_floor_threshold or row.get("error")):
            current = anomaly_by_seed.get(seed)
            if current is None or (global_floor, recorded_at) > current:
                anomaly_by_seed[seed] = (global_floor, recorded_at)

    def ranked(values: dict[str, tuple[int, str]], limit: int) -> list[str]:
        rows = sorted(values.items(), key=lambda item: (-item[1][0], item[1][1], item[0]))
        return [seed for seed, _meta in rows[: max(0, limit)]]

    return ranked(high_by_seed, high_floor_limit), ranked(anomaly_by_seed, anomaly_limit)


def _generate_random_seeds(count: int, run_tag: str = "") -> list[str]:
    count = max(0, int(count))
    if count <= 0:
        return []
    run_tag = str(run_tag or "").strip() or time.strftime("%Y%m%d_%H%M%S")
    rng = random.Random(run_tag)
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
    if args.dataset_mode == "focused":
        high_floor_seeds, anomaly_seeds = _collect_focused_historical_seeds(
            args.history_root,
            high_floor_threshold=max(1, int(args.focused_high_floor_threshold)),
            anomaly_floor_threshold=max(1, int(args.focused_anomaly_floor_threshold)),
            high_floor_limit=max(0, int(args.focused_high_floor_limit)),
            anomaly_limit=max(0, int(args.focused_anomaly_limit)),
        )
        datasets = [
            {
                "name": f"focused_high_ge_global_{int(args.focused_high_floor_threshold)}",
                "source": str(args.history_root),
                "seeds": high_floor_seeds,
            },
            {
                "name": f"focused_anomaly_ge_global_{int(args.focused_anomaly_floor_threshold)}",
                "source": str(args.history_root),
                "seeds": anomaly_seeds,
            },
        ]
        datasets = [dataset for dataset in datasets if dataset["seeds"]]
        if not datasets:
            raise SystemExit("Focused dataset collection produced no high-floor or suspicious seeds.")
        for dataset in datasets:
            dataset["seeds"] = _dedupe_preserve_order(list(dataset.get("seeds") or []))
        return datasets

    historical_seeds = _collect_historical_seeds(
        args.history_root,
        min_floor=max(1, int(args.history_floor_threshold)),
        limit=max(0, int(args.history_limit)),
    )
    random_seeds = _generate_random_seeds(args.random_count, str(args.random_run_tag or ""))

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


def _state_act(state: GameStateView) -> int:
    context = state.raw.get("context") or {}
    return max(1, _safe_int(context.get("act"), 1))


def _state_global_floor(state: GameStateView) -> int:
    return _global_floor(_state_act(state), _state_floor(state))


def _invalid_progression_reason(state: GameStateView) -> str:
    act = _state_act(state)
    floor = _state_floor(state)
    global_floor = _state_global_floor(state)
    if act < 1:
        return f"invalid_floor_progression:A{act}F{floor}/G{global_floor}"
    if act > 4:
        return f"invalid_floor_progression:A{act}F{floor}/G{global_floor}"
    max_floor_by_act = {1: 17, 2: 18, 3: 18, 4: 1}
    max_floor = max_floor_by_act.get(act, 17)
    if floor > max_floor:
        return f"invalid_floor_progression:A{act}F{floor}/G{global_floor}"
    if global_floor > 54:
        return f"invalid_floor_progression:A{act}F{floor}/G{global_floor}"
    return ""


def _update_floor_peak(state: GameStateView, peak: dict[str, int]) -> dict[str, int]:
    act = _state_act(state)
    floor = _state_floor(state)
    global_floor = _global_floor(act, floor)
    if global_floor > peak["global_floor"]:
        peak.update({"act": act, "floor": floor, "global_floor": global_floor})
    return peak


def _is_boss_state(state: GameStateView) -> bool:
    context = state.raw.get("context") or {}
    room_type = str(context.get("room_type") or "").strip().lower()
    return room_type == "boss"


def _boss_cleared(before: GameStateView, after: GameStateView) -> bool:
    if not _is_boss_state(before) or before.decision != "combat_play":
        return False
    if after.game_over and not after.victory:
        return False
    if after.victory:
        return True
    if after.decision in {"card_reward", "map_node", "map_select"} and not after.living_enemies():
        return True
    return False


def _final_hp(state: GameStateView) -> int:
    return _safe_int(state.player.get("hp"), 0)


def _replay_cached_seed(
    protocol: HttpCliProtocol,
    policy: SimpleFlowPolicy,
    cached,
    seed: str,
    *,
    max_steps: int,
) -> dict[str, Any] | None:
    rng = random.Random(seed)
    state = protocol.adapt_state(cached.initial_state)
    if _invalid_progression_reason(state):
        return None
    steps = 0
    max_floor = _state_floor(state)
    max_global_floor = _state_global_floor(state)
    peak = {"act": _state_act(state), "floor": max_floor, "global_floor": max_global_floor}
    boss_attempt = False
    boss_clear = False
    act_boss_attempt = {1: False, 2: False, 3: False}
    act_boss_clear = {1: False, 2: False, 3: False}
    for entry in cached.transitions:
        if steps >= max_steps:
            return None
        if state_hash(state.raw) != str(entry.get("state_hash") or ""):
            return None
        action = protocol.sanitize_action(state, policy.choose_action(state, rng), rng)
        if action_hash(action) != str(entry.get("action_hash") or ""):
            return None
        result = ProtocolStepResult(
            status=str(entry.get("status") or "error"),
            state=entry.get("state"),
            reward=float(entry.get("reward") or 0.0),
            message=str(entry.get("message") or ""),
            last_state=entry.get("last_state"),
            raw=dict(entry.get("raw") or {}),
        )
        if result.status != "success" or not result.state:
            return None
        previous_state = state
        previous_boss_act = _state_act(previous_state)
        is_boss = _is_boss_state(previous_state)
        boss_attempt = boss_attempt or is_boss
        if is_boss and previous_boss_act in act_boss_attempt:
            act_boss_attempt[previous_boss_act] = True
        state = protocol.adapt_state(result.state)
        if _invalid_progression_reason(state):
            return None
        cleared_boss = _boss_cleared(previous_state, state)
        boss_clear = boss_clear or cleared_boss
        if cleared_boss and previous_boss_act in act_boss_clear:
            act_boss_clear[previous_boss_act] = True
        max_floor = max(max_floor, _state_floor(state))
        peak = _update_floor_peak(state, peak)
        max_global_floor = peak["global_floor"]
        steps += 1
        if state.game_over:
            break
    if not state.game_over:
        return None
    outcome = dict(cached.outcome)
    outcome.update(
        {
            "success": True,
            "victory": bool(state.victory),
            "steps": steps,
            "max_floor": max_floor,
            "final_act": _state_act(state),
            "final_floor": _state_floor(state),
            "final_global_floor": _state_global_floor(state),
            "max_act": peak["act"],
            "max_act_floor": peak["floor"],
            "max_global_floor": max_global_floor,
            "final_hp": _final_hp(state),
            "error": "",
            "cache_hit": True,
            "boss_attempt": boss_attempt,
            "boss_clear": boss_clear,
            "act1_boss_attempt": act_boss_attempt[1],
            "act1_boss_clear": act_boss_clear[1],
            "act2_boss_attempt": act_boss_attempt[2],
            "act2_boss_clear": act_boss_clear[2],
            "act3_boss_attempt": act_boss_attempt[3],
            "act3_boss_clear": act_boss_clear[3],
        }
    )
    return outcome


def _run_single_seed(
    protocol: HttpCliProtocol,
    policy: SimpleFlowPolicy,
    seed: str,
    *,
    character: str,
    max_steps: int,
    worker_slot: int | None = None,
    cache: BacktestRolloutCache | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    game_id = ""
    max_floor = 0
    max_global_floor = 0
    peak = {"act": 1, "floor": 0, "global_floor": 0}
    steps = 0
    state: GameStateView | None = None
    initial_state: dict[str, Any] | None = None
    transitions: list[dict[str, Any]] = []
    recovery_count = 0
    resync_count = 0
    outcome: dict[str, Any] = {
        "seed": seed,
        "success": False,
        "victory": False,
        "steps": 0,
        "max_floor": 0,
        "final_act": 1,
        "final_floor": 0,
        "final_global_floor": 0,
        "max_act": 1,
        "max_act_floor": 0,
        "max_global_floor": 0,
        "final_hp": 0,
        "error": "",
        "cache_hit": False,
        "boss_attempt": False,
        "boss_clear": False,
        "act1_boss_attempt": False,
        "act1_boss_clear": False,
        "act2_boss_attempt": False,
        "act2_boss_clear": False,
        "act3_boss_attempt": False,
        "act3_boss_clear": False,
        "recovery_count": 0,
        "resync_count": 0,
    }
    if cache is not None:
        cached = cache.load(seed=seed, character=character)
        if cached is not None:
            replayed = _replay_cached_seed(protocol, policy, cached, seed, max_steps=max_steps)
            if replayed is not None:
                return replayed
            cache.mismatches += 1
    try:
        start = protocol.start_game(character, seed, worker_slot=worker_slot)
        game_id = start.game_id
        raw_state = start.raw_state or protocol.get_state(game_id)
        initial_state = raw_state
        state = protocol.adapt_state(raw_state)
        invalid_progression = _invalid_progression_reason(state)
        if invalid_progression:
            outcome["error"] = invalid_progression
            return outcome
        max_floor = max(max_floor, _state_floor(state))
        peak = _update_floor_peak(state, peak)
        max_global_floor = peak["global_floor"]
        while steps < max_steps:
            if state.game_over:
                break
            action = protocol.sanitize_action(state, policy.choose_action(state, rng), rng)
            before_hash = state_hash(state.raw)
            previous_state = state
            previous_boss_act = _state_act(previous_state)
            is_boss = _is_boss_state(previous_state)
            outcome["boss_attempt"] = bool(outcome["boss_attempt"] or is_boss)
            if is_boss and 1 <= previous_boss_act <= 3:
                outcome[f"act{previous_boss_act}_boss_attempt"] = True
            result = protocol.step(game_id, action)
            if result.status != "success":
                recovery_count += 1
                if protocol.should_resync_after_error(result.raw, state):
                    try:
                        raw_state = protocol.get_state(game_id)
                        state = protocol.adapt_state(raw_state)
                        invalid_progression = _invalid_progression_reason(state)
                        if invalid_progression:
                            outcome["error"] = invalid_progression
                            break
                        resync_count += 1
                        steps += 1
                        continue
                    except Exception:
                        pass
                recover = protocol.recover_action_from_error(result.raw, state) or FlowAction("proceed")
                action = protocol.sanitize_action(state, recover, rng)
                before_hash = state_hash(state.raw)
                previous_state = state
                retry = protocol.step(game_id, action)
                if retry.status != "success":
                    recovery_count += 1
                    if protocol.should_resync_after_error(retry.raw, state):
                        try:
                            raw_state = protocol.get_state(game_id)
                            state = protocol.adapt_state(raw_state)
                            invalid_progression = _invalid_progression_reason(state)
                            if invalid_progression:
                                outcome["error"] = invalid_progression
                                break
                            resync_count += 1
                            steps += 1
                            continue
                        except Exception:
                            pass
                    outcome["error"] = str(retry.message or result.message or "step_failed")
                    raw_state = result.last_state or state.raw
                    state = protocol.adapt_state(raw_state)
                    break
                result = retry
            transitions.append(
                {
                    "state_hash": before_hash,
                    "action": action_payload(action),
                    "action_hash": action_hash(action),
                    "status": result.status,
                    "state": result.state,
                    "reward": result.reward,
                    "message": result.message,
                    "last_state": result.last_state,
                    "raw": result.raw,
                }
            )
            raw_state = result.state or state.raw
            state = protocol.adapt_state(raw_state)
            invalid_progression = _invalid_progression_reason(state)
            if invalid_progression:
                outcome["error"] = invalid_progression
                break
            cleared_boss = _boss_cleared(previous_state, state)
            outcome["boss_clear"] = bool(outcome["boss_clear"] or cleared_boss)
            if cleared_boss and 1 <= previous_boss_act <= 3:
                outcome[f"act{previous_boss_act}_boss_clear"] = True
            max_floor = max(max_floor, _state_floor(state))
            peak = _update_floor_peak(state, peak)
            max_global_floor = peak["global_floor"]
            steps += 1
        completed = bool(state and state.game_over and not outcome["error"])
        outcome.update(
            {
                "success": completed,
                "victory": bool(state.victory) if state else False,
                "steps": steps,
                "max_floor": max_floor,
                "final_act": _state_act(state) if state else 1,
                "final_floor": _state_floor(state) if state else 0,
                "final_global_floor": _state_global_floor(state) if state else 0,
                "max_act": peak["act"],
                "max_act_floor": peak["floor"],
                "max_global_floor": max_global_floor,
                "final_hp": _final_hp(state) if state else 0,
                "recovery_count": recovery_count,
                "resync_count": resync_count,
            }
        )
        if state is not None and steps >= max_steps and not state.game_over:
            outcome["error"] = f"max_steps_exceeded:{max_steps}"
            outcome["success"] = False
        if cache is not None and initial_state is not None and outcome.get("success") and recovery_count == 0 and resync_count == 0:
            cache.store(
                seed=seed,
                character=character,
                initial_state=initial_state,
                transitions=transitions,
                outcome=outcome,
            )
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
    errored = [row for row in rows if row.get("error") or not row.get("success")]
    floors = [int(row.get("max_floor") or 0) for row in done]
    global_floors = [int(row.get("max_global_floor") or row.get("final_global_floor") or row.get("max_floor") or 0) for row in done]
    victories = sum(1 for row in done if row.get("victory"))
    cache_hits = sum(1 for row in rows if row.get("cache_hit"))
    return {
        "total": total,
        "success": success,
        "errors": len(errored),
        "victory": victories,
        "avg_floor": round(sum(floors) / len(floors), 2) if floors else 0.0,
        "avg_global_floor": round(sum(global_floors) / len(global_floors), 2) if global_floors else 0.0,
        "ge5": sum(1 for value in floors if value >= 5),
        "ge10": sum(1 for value in floors if value >= 10),
        "ge15": sum(1 for value in floors if value >= 15),
        "ge17": sum(1 for value in floors if value >= 17),
        "ge18": sum(1 for value in floors if value >= 18),
        "gt17": sum(1 for value in floors if value > 17),
        "ge_global_17": sum(1 for value in global_floors if value >= 17),
        "ge_global_34": sum(1 for value in global_floors if value >= 34),
        "ge_global_51": sum(1 for value in global_floors if value >= 51),
        "ge_global_52": sum(1 for value in global_floors if value >= 52),
        "boss_attempt": sum(1 for row in done if row.get("boss_attempt")),
        "boss_clear": sum(1 for row in done if row.get("boss_clear")),
        "act1_boss_attempt": sum(1 for row in done if row.get("act1_boss_attempt")),
        "act1_boss_clear": sum(1 for row in done if row.get("act1_boss_clear")),
        "act2_boss_attempt": sum(1 for row in done if row.get("act2_boss_attempt")),
        "act2_boss_clear": sum(1 for row in done if row.get("act2_boss_clear")),
        "act3_boss_attempt": sum(1 for row in done if row.get("act3_boss_attempt")),
        "act3_boss_clear": sum(1 for row in done if row.get("act3_boss_clear")),
        "cache_hits": cache_hits,
    }


def _runtime_error_examples(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        error = str(row.get("error") or "").strip()
        if not error and row.get("success"):
            continue
        examples.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "success": bool(row.get("success")),
                "error": error[:500],
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _run_dataset(
    protocol: HttpCliProtocol,
    policy: SimpleFlowPolicy,
    dataset: dict[str, Any],
    *,
    character: str,
    max_steps: int,
    worker_slot: int | None = None,
    workers: int = 1,
    cache: BacktestRolloutCache | None = None,
) -> dict[str, Any]:
    name = str(dataset.get("name") or "dataset")
    seeds = list(dataset.get("seeds") or [])
    rows_by_index: dict[int, dict[str, Any]] = {}
    worker_count = max(1, int(workers))

    def run_indexed_seed(index: int, seed: str, slot: int | None) -> tuple[int, dict[str, Any]]:
        row = _run_single_seed(
            protocol,
            policy,
            seed,
            character=character,
            max_steps=max(100, max_steps),
            worker_slot=slot,
            cache=cache,
        )
        row["dataset"] = name
        return index, row

    if worker_slot is not None:
        worker_count = 1

    if worker_count <= 1 or len(seeds) <= 1:
        for completed_count, (index, seed) in enumerate(enumerate(seeds, start=1), start=1):
            _, row = run_indexed_seed(index, seed, worker_slot)
            rows_by_index[index] = row
            print(
                f"[{name} {completed_count:03d}/{len(seeds):03d}] index={index:03d} seed={row['seed']} "
                f"success={row['success']} victory={row['victory']} "
                f"final=A{row.get('final_act', 1)}F{row.get('final_floor', row['max_floor'])}/G{row.get('final_global_floor', row['max_floor'])} "
                f"max=A{row.get('max_act', row.get('final_act', 1))}F{row.get('max_act_floor', row['max_floor'])}/G{row.get('max_global_floor', row['max_floor'])} "
                f"steps={row['steps']} cache={bool(row.get('cache_hit'))} err={row['error']}",
                flush=True,
            )
    else:
        completed_lock = threading.Lock()
        completed_count = 0
        slot_batches: list[list[tuple[int, str]]] = [[] for _ in range(worker_count)]
        for index, seed in enumerate(seeds, start=1):
            slot_batches[(index - 1) % worker_count].append((index, seed))

        def run_slot_batch(slot: int, batch: list[tuple[int, str]]) -> list[tuple[int, dict[str, Any]]]:
            nonlocal completed_count
            slot_rows: list[tuple[int, dict[str, Any]]] = []
            for index, seed in batch:
                _, row = run_indexed_seed(index, seed, slot)
                slot_rows.append((index, row))
                with completed_lock:
                    completed_count += 1
                    current_count = completed_count
                print(
                    f"[{name} {current_count:03d}/{len(seeds):03d}] index={index:03d} seed={row['seed']} "
                    f"success={row['success']} victory={row['victory']} "
                    f"final=A{row.get('final_act', 1)}F{row.get('final_floor', row['max_floor'])}/G{row.get('final_global_floor', row['max_floor'])} "
                    f"max=A{row.get('max_act', row.get('final_act', 1))}F{row.get('max_act_floor', row['max_floor'])}/G{row.get('max_global_floor', row['max_floor'])} "
                    f"steps={row['steps']} cache={bool(row.get('cache_hit'))} err={row['error']}",
                    flush=True,
                )
            return slot_rows

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(run_slot_batch, slot, batch)
                for slot, batch in enumerate(slot_batches)
                if batch
            ]
            for future in as_completed(futures):
                for index, row in future.result():
                    rows_by_index[index] = row
    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
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
    cache: BacktestRolloutCache | None = None
    cache_namespace = ""
    if not args.no_cache:
        repo_root = Path(__file__).resolve().parents[1]
        cache_namespace = args.cache_version or f"flow_policy_{source_digest(repo_root)}"
        cache_path = args.cache_dir / "rollouts.sqlite3"
        cache = BacktestRolloutCache(cache_path, namespace=cache_namespace)
        print(json.dumps({"cache": cache.stats()}, ensure_ascii=False, indent=2))

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
            workers=args.workers,
            cache=cache,
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
            "history_limit": int(args.history_limit),
            "focused_high_floor_threshold": int(args.focused_high_floor_threshold),
            "focused_anomaly_floor_threshold": int(args.focused_anomaly_floor_threshold),
            "focused_high_floor_limit": int(args.focused_high_floor_limit),
            "focused_anomaly_limit": int(args.focused_anomaly_limit),
            "curated_seeds_file": str(args.seeds_file),
            "random_count": int(args.random_count),
            "random_run_tag": str(args.random_run_tag or ""),
            "workers": max(1, int(args.workers)),
            "cache_enabled": cache is not None,
            "cache_namespace": cache_namespace,
        },
        "datasets": dataset_payloads,
        "merged_summary": _summary(merged_rows),
        "runtime_error_examples": _runtime_error_examples(merged_rows),
        "cache": cache.stats() if cache is not None else {"enabled": False},
    }
    print(json.dumps(payload["merged_summary"], ensure_ascii=False, indent=2))
    if payload["runtime_error_examples"]:
        print(json.dumps({"runtime_error_examples": payload["runtime_error_examples"]}, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if payload["runtime_error_examples"] and not args.allow_runtime_errors:
        raise SystemExit("Backtest produced runtime errors; refusing to treat metrics as valid.")


if __name__ == "__main__":
    main()
