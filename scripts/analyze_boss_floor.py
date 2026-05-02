# -*- coding: utf-8 -*-
"""Analyze deterministic floor-17 boss attempts for flow-policy tuning."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import threading
from pathlib import Path
from statistics import mean
import sys
from typing import Any

from st2rl.gameplay.config import FlowPolicyConfig
from st2rl.gameplay.enemy_intent_script import describe_enemy_intent
from st2rl.gameplay.policy import SimpleFlowPolicy
from st2rl.gameplay.types import FlowAction, GameStateView
from st2rl.protocols.http_cli import HttpCliProtocol, HttpCliProtocolConfig

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backtest_flow_policy import _boss_cleared, _is_boss_state, _state_floor


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
    parser = argparse.ArgumentParser(description="Trace and aggregate floor-17 boss attempts.")
    parser.add_argument("--backtest-json", type=Path, default=None, help="Backtest JSON whose rows contain seeds.")
    parser.add_argument("--seeds-file", type=Path, default=None, help="UTF-8 seed file, one seed per line.")
    parser.add_argument(
        "--focus",
        choices=("all", "boss_attempt", "boss_fail", "boss_clear", "act2_boss_attempt", "act2_boss_fail", "act2_boss_clear"),
        default="boss_attempt",
    )
    parser.add_argument("--boss-act", type=int, default=1, help="Boss act to trace; use 2 for Act2/global floor 34.")
    parser.add_argument("--max-seeds", type=int, default=120)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--character", type=str, default="Ironclad")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--out", type=Path, default=Path("logs/backtests/boss_floor_analysis.json"))
    parser.add_argument(
        "--allow-runtime-errors",
        action="store_true",
        help="Write analysis but do not fail the process when any traced seed reports an error.",
    )
    return parser.parse_args()


def _load_seed_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.backtest_json is not None:
        data = json.loads(args.backtest_json.read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []
        for dataset in data.get("datasets") or []:
            for row in dataset.get("rows") or []:
                if isinstance(row, dict) and row.get("seed"):
                    rows.append(dict(row))
        if args.focus == "boss_attempt":
            rows = [row for row in rows if row.get("boss_attempt")]
        elif args.focus == "boss_fail":
            rows = [row for row in rows if row.get("boss_attempt") and not row.get("boss_clear")]
        elif args.focus == "boss_clear":
            rows = [row for row in rows if row.get("boss_clear")]
        elif args.focus == "act2_boss_attempt":
            rows = [row for row in rows if row.get("act2_boss_attempt")]
        elif args.focus == "act2_boss_fail":
            rows = [row for row in rows if row.get("act2_boss_attempt") and not row.get("act2_boss_clear")]
        elif args.focus == "act2_boss_clear":
            rows = [row for row in rows if row.get("act2_boss_clear")]
        return _dedupe_seed_rows(rows)[: max(0, args.max_seeds)]

    if args.seeds_file is None:
        raise SystemExit("Provide --backtest-json or --seeds-file.")
    seeds = [
        line.strip()
        for line in args.seeds_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return _dedupe_seed_rows([{"seed": seed} for seed in seeds])[: max(0, args.max_seeds)]


def _dedupe_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        seed = str(row.get("seed") or "").strip()
        if not seed or seed in seen:
            continue
        seen.add(seed)
        unique.append(row)
    return unique


def _enemy_intended_damage(state: GameStateView) -> int:
    total = 0
    for enemy in state.living_enemies():
        intents = enemy.get("intents") or []
        if isinstance(intents, list):
            for intent in intents:
                if not isinstance(intent, dict):
                    continue
                damage = _safe_int(intent.get("damage"), 0)
                hits = max(1, _safe_int(intent.get("hits"), 1))
                total += damage * hits
        total += _safe_int(enemy.get("intent_damage"), 0)
    return total


def _card_id(card: dict[str, Any]) -> str:
    return str(card.get("id") or card.get("card_id") or card.get("name") or "").strip()


def _card_damage(policy: SimpleFlowPolicy, card: dict[str, Any]) -> int:
    return policy._card_damage(card)


def _card_block(policy: SimpleFlowPolicy, card: dict[str, Any]) -> int:
    return policy._card_block(card)


def _card_hp_loss(policy: SimpleFlowPolicy, card: dict[str, Any]) -> int:
    return policy._card_hp_loss(card)


def _compact_powers(powers: Any) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for power in powers or []:
        if not isinstance(power, dict):
            continue
        compact.append(
            {
                "id": power.get("id") or power.get("power_id") or power.get("name"),
                "name": power.get("name") or power.get("display_name") or power.get("id"),
                "amount": power.get("amount"),
            }
        )
    return compact


def _combat_boss_label(state: GameStateView) -> str:
    context = state.raw.get("context") or {}
    boss = context.get("boss")
    if isinstance(boss, dict):
        for key in ("name", "display_name", "boss_name", "id", "boss_id"):
            value = str(boss.get(key) or "").strip()
            if value:
                return value
    for key in ("boss_name", "boss_id"):
        value = str(context.get(key) or "").strip()
        if value:
            return value
    living = state.living_enemies()
    if living:
        return " + ".join(str(enemy.get("name") or enemy.get("id") or "?") for enemy in living)
    return "UNKNOWN_BOSS"


def _combat_target_name(action: FlowAction, living: list[dict[str, Any]]) -> str:
    target_index = action.args.get("target_index")
    for enemy in living:
        if enemy.get("index") == target_index:
            return str(enemy.get("name") or enemy.get("id") or "")
    return ""


def _combat_snapshot(state: GameStateView, policy: SimpleFlowPolicy, action: FlowAction, step: int) -> dict[str, Any]:
    playable = state.playable_cards()
    played_card = next((card for card in playable if card.get("index") == action.args.get("card_index")), None)
    living = state.living_enemies()
    intent_snapshot = describe_enemy_intent(state, horizon=3)
    intent_rows = list(intent_snapshot.get("rows") or [])
    mismatch_rows = [row for row in intent_rows if row.get("mismatch")]
    return {
        "step": step,
        "boss_name": _combat_boss_label(state),
        "floor": _state_floor(state),
        "round": state.round,
        "decision": state.decision,
        "hp": state.hp,
        "max_hp": state.max_hp,
        "block": state.block,
        "energy": state.energy,
        "incoming": _enemy_intended_damage(state),
        "action": action.name,
        "action_args": dict(action.args),
        "target_name": _combat_target_name(action, living),
        "played_card": _card_id(played_card or {}) if played_card else "",
        "played_damage": _card_damage(policy, played_card or {}) if played_card else 0,
        "played_block": _card_block(policy, played_card or {}) if played_card else 0,
        "played_hp_loss": _card_hp_loss(policy, played_card or {}) if played_card else 0,
        "playable_count": len(playable),
        "playable_cards": [_card_id(card) for card in playable],
        "player_powers": _compact_powers(state.player.get("powers") or state.raw.get("player_powers") or []),
        "policy_debug": dict(policy._last_policy_debug),
        "intent_mismatch_count": len(mismatch_rows),
        "intent_mismatches": [
            {
                "enemy": row.get("enemy"),
                "script": row.get("script"),
                "observed_category": row.get("observed_category"),
                "observed_intent": row.get("observed_intent"),
                "expected_categories": row.get("expected_categories"),
                "expected_phase": (row.get("expected_step") or {}).get("special_phase_tag"),
            }
            for row in mismatch_rows
        ],
        "enemy_intent": [
            {
                "enemy": row.get("enemy"),
                "script": row.get("script"),
                "observed_category": row.get("observed_category"),
                "observed_intent": row.get("observed_intent"),
                "observed_intended_damage": row.get("observed_intended_damage"),
                "expected_categories": row.get("expected_categories"),
                "expected_phase": (row.get("expected_step") or {}).get("special_phase_tag"),
                "next_turn_forecast": row.get("next_turn_forecast"),
                "short_horizon_forecast": row.get("short_horizon_forecast"),
                "predicted_two_turn_damage": row.get("predicted_two_turn_damage"),
                "scaling_risk": row.get("scaling_risk"),
                "mismatch": bool(row.get("mismatch")),
            }
            for row in intent_rows
        ],
        "enemies": [
            {
                "id": enemy.get("id"),
                "name": enemy.get("name"),
                "hp": enemy.get("hp"),
                "block": enemy.get("block"),
                "powers": _compact_powers(enemy.get("powers") or []),
            }
            for enemy in living
        ],
    }


def _boss_signature(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        enemies = row.get("enemies") or []
        if enemies:
            ids = [str(enemy.get("id") or enemy.get("name") or "?") for enemy in enemies]
            return "+".join(ids)
    return "NO_BOSS_TRACE"


def _enemy_role(enemy: dict[str, Any]) -> str:
    enemy_id = str(enemy.get("id") or enemy.get("name") or "").strip().lower()
    if "priest" in enemy_id or "神官" in enemy_id:
        return "priest"
    if "follower" in enemy_id or "教徒" in enemy_id:
        return "follower"
    return "other"


def _kin_trace_metrics(trace: list[dict[str, Any]]) -> dict[str, Any]:
    early_attack_targets: Counter[str] = Counter()
    first_priest_target_round: int | None = None
    follower_alive_by_round: dict[int, int] = {}
    for row in trace:
        round_no = _safe_int(row.get("round"), 0)
        enemies = list(row.get("enemies") or [])
        if round_no in (3, 4) and round_no not in follower_alive_by_round:
            follower_alive_by_round[round_no] = sum(1 for enemy in enemies if _enemy_role(enemy) == "follower")
        target_name = str(row.get("target_name") or "")
        target_role = _enemy_role({"name": target_name})
        is_damage_action = (
            (row.get("action") == "play_card" and _safe_int(row.get("played_damage"), 0) > 0)
            or (row.get("action") == "use_potion" and bool(target_name))
        )
        if 1 <= round_no <= 3 and is_damage_action:
            early_attack_targets.update([target_role if target_role != "other" else (target_name or "unknown")])
        if first_priest_target_round is None and target_role == "priest" and is_damage_action:
            first_priest_target_round = round_no
    return {
        "early_attack_targets_round_1_3": dict(early_attack_targets),
        "follower_alive_round_3": follower_alive_by_round.get(3),
        "follower_alive_round_4": follower_alive_by_round.get(4),
        "first_priest_target_round": first_priest_target_round,
    }


def _deck_counter(state: GameStateView | None) -> Counter[str]:
    if state is None:
        return Counter()
    return Counter(_card_id(card) for card in state.player.get("deck") or [] if isinstance(card, dict))


def _trace_seed(seed_row: dict[str, Any], args: argparse.Namespace, worker_slot: int | None) -> dict[str, Any]:
    seed = str(seed_row.get("seed") or "")
    protocol = HttpCliProtocol(HttpCliProtocolConfig(timeout_seconds=args.timeout_seconds))
    policy = SimpleFlowPolicy(FlowPolicyConfig())
    rng = random.Random(seed)
    game_id = ""
    state: GameStateView | None = None
    boss_entry_state: GameStateView | None = None
    boss_rows: list[dict[str, Any]] = []
    boss_attempt = False
    boss_clear = False
    error = ""
    target_act = max(1, _safe_int(args.boss_act, 1))
    try:
        start = protocol.start_game(args.character, seed, worker_slot=worker_slot)
        game_id = start.game_id
        state = protocol.adapt_state(start.raw_state or protocol.get_state(game_id))
        for step in range(args.max_steps):
            if state.game_over:
                break
            action = protocol.sanitize_action(state, policy.choose_action(state, rng), rng)
            previous = state
            previous_act = _safe_int((previous.raw.get("context") or {}).get("act"), 1)
            is_target_boss = _is_boss_state(previous) and previous_act == target_act
            if is_target_boss:
                boss_attempt = True
                if boss_entry_state is None:
                    boss_entry_state = previous
                boss_rows.append(_combat_snapshot(previous, policy, action, step))
            result = protocol.step(game_id, action)
            if result.status != "success":
                if protocol.should_resync_after_error(result.raw, state):
                    try:
                        raw_state = protocol.get_state(game_id)
                        state = protocol.adapt_state(raw_state)
                        continue
                    except Exception:
                        pass
                recover = protocol.recover_action_from_error(result.raw, previous)
                if recover is None:
                    error = str(result.message or "step_failed")
                    break
                action = protocol.sanitize_action(previous, recover, rng)
                result = protocol.step(game_id, action)
                if result.status != "success":
                    if protocol.should_resync_after_error(result.raw, previous):
                        try:
                            raw_state = protocol.get_state(game_id)
                            state = protocol.adapt_state(raw_state)
                            continue
                        except Exception:
                            pass
                    error = str(result.message or "retry_failed")
                    break
            state = protocol.adapt_state(result.state or previous.raw)
            cleared_target_boss = is_target_boss and _boss_cleared(previous, state)
            boss_clear = boss_clear or cleared_target_boss
            if cleared_target_boss:
                break
        if state is not None and not state.game_over and not error and not boss_clear:
            error = "max_steps_exceeded"
    except Exception as exc:  # pragma: no cover - runtime integration
        error = str(exc)
    finally:
        if game_id:
            try:
                protocol.close_game(game_id)
            except Exception:
                pass

    entry_hp = boss_entry_state.hp if boss_entry_state is not None else None
    entry_max_hp = boss_entry_state.max_hp if boss_entry_state is not None else None
    deck = _deck_counter(boss_entry_state)
    combat_actions = [row for row in boss_rows if row.get("decision") == "combat_play"]
    played_cards = [row.get("played_card") for row in combat_actions if row.get("played_card")]
    kin_metrics = _kin_trace_metrics(boss_rows) if _boss_signature(boss_rows).startswith("KIN_") else {}
    return {
        "seed": seed,
        "source": seed_row,
        "error": error,
        "boss_attempt": boss_attempt,
        "boss_clear": boss_clear,
        "victory": bool(state.victory) if state is not None else False,
        "final_hp": state.hp if state is not None else None,
        "max_floor": _state_floor(state) if state is not None else 0,
        "boss": _boss_signature(boss_rows),
        "entry_hp": entry_hp,
        "entry_max_hp": entry_max_hp,
        "boss_steps": len(boss_rows),
        "boss_rounds": max((_safe_int(row.get("round"), 0) for row in boss_rows), default=0),
        "cards_played": sum(1 for row in combat_actions if row.get("action") == "play_card"),
        "avg_playable": round(mean([row.get("playable_count", 0) for row in combat_actions]), 2) if combat_actions else 0,
        "hp_loss_cards": sum(1 for row in combat_actions if row.get("played_hp_loss", 0) > 0),
        "block_cards": sum(1 for row in combat_actions if row.get("played_block", 0) > 0),
        "damage_cards": sum(1 for row in combat_actions if row.get("played_damage", 0) > 0),
        "potions_used": sum(1 for row in boss_rows if row.get("action") == "use_potion"),
        "intent_mismatch_count": sum(_safe_int(row.get("intent_mismatch_count"), 0) for row in boss_rows),
        "deck": dict(deck),
        "played_cards": Counter(played_cards),
        "kin_metrics": kin_metrics,
        "trace": boss_rows,
        "trace_tail": boss_rows[-12:],
    }


def _bucket_hp(value: Any) -> str:
    hp = _safe_int(value, -1)
    if hp < 0:
        return "unknown"
    if hp < 35:
        return "<35"
    if hp < 45:
        return "35-44"
    if hp < 55:
        return "45-54"
    if hp < 65:
        return "55-64"
    return "65+"


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    errored = [row for row in results if row.get("error")]
    valid = [row for row in results if not row.get("error")]
    attempted = [row for row in valid if row.get("boss_attempt")]
    clears = [row for row in attempted if row.get("boss_clear")]
    fails = [row for row in attempted if not row.get("boss_clear")]

    by_boss: dict[str, dict[str, Any]] = {}
    for boss, rows_iter in _group_by(attempted, "boss").items():
        rows = list(rows_iter)
        boss_clears = [row for row in rows if row.get("boss_clear")]
        boss_fails = [row for row in rows if not row.get("boss_clear")]
        by_boss[boss] = _summarize_group(rows, boss_clears, boss_fails)

    by_entry_hp = {
        bucket: _summarize_group(rows, [row for row in rows if row.get("boss_clear")], [row for row in rows if not row.get("boss_clear")])
        for bucket, rows in _group_by(attempted, lambda row: _bucket_hp(row.get("entry_hp"))).items()
    }

    mismatch_by_boss = {
        boss: _summarize_mismatch_group(rows)
        for boss, rows in _group_by(attempted, "boss").items()
    }

    return {
        "run_quality": {
            "total": len(results),
            "valid": len(valid),
            "errored": len(errored),
            "error_rate": round(len(errored) / len(results), 4) if results else 0.0,
            "error_examples": [
                {"seed": row.get("seed"), "error": str(row.get("error") or "")[:240]}
                for row in errored[:12]
            ],
        },
        "summary": _summarize_group(attempted, clears, fails),
        "by_boss": by_boss,
        "by_boss_mismatch": mismatch_by_boss,
        "by_entry_hp": by_entry_hp,
        "mismatch_hotspots": _mismatch_hotspots(attempted),
        "common_fail_deck": dict(_common_counter(fails, "deck").most_common(30)),
        "common_clear_deck": dict(_common_counter(clears, "deck").most_common(30)),
        "common_fail_played": dict(_common_counter(fails, "played_cards").most_common(30)),
        "common_clear_played": dict(_common_counter(clears, "played_cards").most_common(30)),
        "deck_delta_clear_minus_fail": _counter_rate_delta(clears, fails, "deck"),
        "played_delta_clear_minus_fail": _counter_rate_delta(clears, fails, "played_cards"),
        "representative_fails": sorted(fails, key=lambda row: (_safe_int(row.get("entry_hp"), 0), row.get("boss")))[:12],
        "representative_clears": sorted(clears, key=lambda row: (_safe_int(row.get("entry_hp"), 0), row.get("boss")))[:12],
    }


def _group_by(rows: list[dict[str, Any]], key: str | Any) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_key = key(row) if callable(key) else row.get(key)
        grouped[str(group_key)].append(row)
    return dict(grouped)


def _common_counter(rows: list[dict[str, Any]], field: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        values = row.get(field) or {}
        if isinstance(values, dict):
            counter.update({str(key): _safe_int(value, 0) for key, value in values.items()})
    return counter


def _counter_rate_delta(clears: list[dict[str, Any]], fails: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    clear_counter = _common_counter(clears, field)
    fail_counter = _common_counter(fails, field)
    clear_count = max(1, len(clears))
    fail_count = max(1, len(fails))
    rows = []
    for key in sorted(set(clear_counter) | set(fail_counter)):
        clear_rate = clear_counter.get(key, 0) / clear_count
        fail_rate = fail_counter.get(key, 0) / fail_count
        rows.append(
            {
                "card": key,
                "delta": round(clear_rate - fail_rate, 4),
                "clear_per_attempt": round(clear_rate, 4),
                "fail_per_attempt": round(fail_rate, 4),
            }
        )
    rows.sort(key=lambda row: row["delta"], reverse=True)
    return rows[:20] + list(reversed(rows[-20:]))


def _summarize_group(rows: list[dict[str, Any]], clears: list[dict[str, Any]], fails: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(field: str, source: list[dict[str, Any]]) -> float:
        values = [_safe_int(row.get(field), -1) for row in source if row.get(field) is not None]
        values = [value for value in values if value >= 0]
        return round(mean(values), 2) if values else 0.0

    summary = {
        "attempts": len(rows),
        "clears": len(clears),
        "fails": len(fails),
        "clear_rate": round(len(clears) / len(rows), 4) if rows else 0.0,
        "avg_entry_hp": avg("entry_hp", rows),
        "clear_avg_entry_hp": avg("entry_hp", clears),
        "fail_avg_entry_hp": avg("entry_hp", fails),
        "avg_boss_rounds": avg("boss_rounds", rows),
        "avg_cards_played": avg("cards_played", rows),
        "avg_playable": round(mean([float(row.get("avg_playable") or 0.0) for row in rows]), 2) if rows else 0.0,
        "avg_hp_loss_cards": avg("hp_loss_cards", rows),
        "avg_potions_used": avg("potions_used", rows),
        "avg_intent_mismatches": avg("intent_mismatch_count", rows),
    }
    kin_rows = [row for row in rows if isinstance(row.get("kin_metrics"), dict) and row.get("kin_metrics")]
    if kin_rows:
        target_counter: Counter[str] = Counter()
        follower_round3: list[int] = []
        follower_round4: list[int] = []
        first_priest_round: list[int] = []
        for row in kin_rows:
            metrics = dict(row.get("kin_metrics") or {})
            target_counter.update({str(key): _safe_int(value, 0) for key, value in (metrics.get("early_attack_targets_round_1_3") or {}).items()})
            if metrics.get("follower_alive_round_3") is not None:
                follower_round3.append(_safe_int(metrics.get("follower_alive_round_3"), 0))
            if metrics.get("follower_alive_round_4") is not None:
                follower_round4.append(_safe_int(metrics.get("follower_alive_round_4"), 0))
            if metrics.get("first_priest_target_round") is not None:
                first_priest_round.append(_safe_int(metrics.get("first_priest_target_round"), 0))
        summary["kin_trace"] = {
            "early_attack_targets_round_1_3": dict(target_counter),
            "avg_follower_alive_round_3": round(mean(follower_round3), 2) if follower_round3 else 0.0,
            "avg_follower_alive_round_4": round(mean(follower_round4), 2) if follower_round4 else 0.0,
            "avg_first_priest_target_round": round(mean(first_priest_round), 2) if first_priest_round else 0.0,
        }
    return summary


def _summarize_mismatch_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = len(rows)
    mismatch_attempts = sum(1 for row in rows if _safe_int(row.get("intent_mismatch_count"), 0) > 0)
    total_mismatches = sum(_safe_int(row.get("intent_mismatch_count"), 0) for row in rows)
    round_counter: Counter[str] = Counter()
    phase_counter: Counter[str] = Counter()
    enemy_counter: Counter[str] = Counter()
    for row in rows:
        for trace in list(row.get("trace") or []):
            for mismatch in list(trace.get("intent_mismatches") or []):
                round_counter.update([str(trace.get("round") or "?")])
                phase = str(mismatch.get("expected_phase") or "unknown")
                phase_counter.update([phase])
                enemy_counter.update([str(mismatch.get("enemy") or "unknown")])
    return {
        "attempts": attempts,
        "mismatch_attempts": mismatch_attempts,
        "mismatch_attempt_rate": round(mismatch_attempts / attempts, 4) if attempts else 0.0,
        "total_mismatches": total_mismatches,
        "top_rounds": [{"round": key, "count": value} for key, value in round_counter.most_common(8)],
        "top_phases": [{"phase": key, "count": value} for key, value in phase_counter.most_common(8)],
        "top_enemies": [{"enemy": key, "count": value} for key, value in enemy_counter.most_common(8)],
    }


def _mismatch_hotspots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hotspot_counter: Counter[tuple[str, str, str, str]] = Counter()
    for row in rows:
        boss = str(row.get("boss") or "UNKNOWN_BOSS")
        for trace in list(row.get("trace") or []):
            trace_round = str(trace.get("round") or "?")
            for mismatch in list(trace.get("intent_mismatches") or []):
                hotspot_counter.update(
                    [
                        (
                            boss,
                            str(mismatch.get("enemy") or "unknown"),
                            trace_round,
                            str(mismatch.get("expected_phase") or "unknown"),
                        )
                    ]
                )
    return [
        {
            "boss": boss,
            "enemy": enemy,
            "round": round_no,
            "phase": phase,
            "count": count,
        }
        for (boss, enemy, round_no, phase), count in hotspot_counter.most_common(16)
    ]


def main() -> None:
    args = _parse_args()
    seed_rows = _load_seed_rows(args)
    if not seed_rows:
        raise SystemExit("No seeds selected.")

    results: list[dict[str, Any]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, args.workers)
    completed_lock = threading.Lock()
    completed_count = 0
    slot_batches: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    for index, seed_row in enumerate(seed_rows):
        slot_batches[index % worker_count].append(seed_row)

    def run_slot_batch(slot: int, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal completed_count
        slot_results: list[dict[str, Any]] = []
        for seed_row in batch:
            row = _trace_seed(seed_row, args, slot)
            slot_results.append(row)
            with completed_lock:
                completed_count += 1
                current_count = completed_count
            print(
                f"[{current_count:03d}/{len(seed_rows):03d}] seed={row.get('seed')} "
                f"boss={row.get('boss')} clear={row.get('boss_clear')} "
                f"entry_hp={row.get('entry_hp')} err={row.get('error') or ''}"
            )
        return slot_results

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(run_slot_batch, slot, batch)
            for slot, batch in enumerate(slot_batches)
            if batch
        ]
        for future in as_completed(futures):
            results.extend(future.result())

    payload = {
        "config": {
            "backtest_json": str(args.backtest_json) if args.backtest_json else None,
            "seeds_file": str(args.seeds_file) if args.seeds_file else None,
            "focus": args.focus,
            "boss_act": args.boss_act,
            "max_seeds": args.max_seeds,
            "workers": args.workers,
            "character": args.character,
        },
        "aggregate": _aggregate(results),
        "results": results,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"]["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    if payload["aggregate"]["run_quality"]["errored"] and not args.allow_runtime_errors:
        raise SystemExit("Boss analysis produced runtime errors; refusing to treat metrics as valid.")


if __name__ == "__main__":
    main()
