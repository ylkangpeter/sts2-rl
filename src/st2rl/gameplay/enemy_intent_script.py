# -*- coding: utf-8 -*-
"""Enemy intent script database, forecast, and runtime validation."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from st2rl.gameplay.types import GameStateView

_SCRIPT_PATH = Path(__file__).resolve().parent / "data" / "enemy_intent_script_v1.json"
_ALERT_PATH = Path("logs") / "intent_script_alerts.jsonl"
_STATS_PATH = Path("logs") / "intent_script_stats.json"
_DASHBOARD_INCIDENT_URL = "http://127.0.0.1:8787/api/telemetry/runtime/incident"

_INTENT_CATEGORIES = ("attack", "buff", "debuff", "block", "summon", "unknown")


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


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


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _text(value)
    return text in {"1", "true", "yes", "y"}


def _enemy_key(enemy: dict[str, Any]) -> str:
    return " ".join(_text(enemy.get(key)) for key in ("id", "name", "display_name"))


def _intent_text(enemy: dict[str, Any]) -> str:
    intent_value = enemy.get("intent")
    if isinstance(intent_value, dict):
        parts = [intent_value.get("id"), intent_value.get("name"), intent_value.get("description")]
        return " ".join(_text(part) for part in parts)
    intents = enemy.get("intents") or []
    if isinstance(intents, list) and intents:
        parts: list[str] = []
        for intent in intents:
            if not isinstance(intent, dict):
                parts.append(_text(intent))
                continue
            parts.extend(
                [
                    _text(intent.get("type")),
                    _text(intent.get("id")),
                    _text(intent.get("name")),
                    _text(intent.get("description")),
                ]
            )
        joined = " ".join(part for part in parts if part)
        if joined:
            return joined
    return " ".join(_text(enemy.get(key)) for key in ("intent", "intent_id", "intent_name", "intent_description"))


def _intent_category(enemy: dict[str, Any]) -> str:
    text = _intent_text(enemy)
    if not text:
        return "unknown"
    if any(token in text for token in ("attack", "strike", "slash", "beam", "bite", "hit", "damage", "攻击")):
        return "attack"
    if any(token in text for token in ("debuff", "weak", "vulnerable", "frail", "减益", "虚弱", "易伤")):
        return "debuff"
    if any(token in text for token in ("buff", "strength", "armor", "ritual", "power", "强化", "力量", "增益")):
        return "buff"
    if any(token in text for token in ("defend", "block", "护甲", "格挡", "防御")):
        return "block"
    if any(token in text for token in ("summon", "spawn", "call", "召唤")):
        return "summon"
    return "unknown"


def _enemy_intended_damage(enemy: dict[str, Any]) -> int:
    intents = enemy.get("intents") or []
    if isinstance(intents, list):
        total = 0
        for intent in intents:
            if not isinstance(intent, dict):
                continue
            damage = max(0, _safe_int(intent.get("damage"), 0))
            hits = max(1, _safe_int(intent.get("hits"), 1))
            total += damage * hits
        if total > 0:
            return total
    for key in ("intent_damage", "damage", "intentDamage"):
        value = enemy.get(key)
        if value is None:
            continue
        return max(0, _safe_int(value, 0))
    return 0


@lru_cache(maxsize=1)
def _load_db() -> dict[str, Any]:
    if not _SCRIPT_PATH.exists():
        return {"version": "missing", "entries": []}
    try:
        return json.loads(_SCRIPT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": "invalid", "entries": []}


def _normalized_step(round_no: int, step: Any) -> dict[str, Any]:
    payload = step if isinstance(step, dict) else {}
    categories = [
        str(item).strip().lower()
        for item in (payload.get("intent_categories") or [])
        if str(item).strip()
    ]
    categories = [item for item in categories if item in _INTENT_CATEGORIES]
    return {
        "round": _safe_int(payload.get("round"), round_no) or round_no,
        "intent_categories": categories,
        "damage_max": max(0, _safe_int(payload.get("damage_max"), 0)),
        "hit_count": max(0, _safe_int(payload.get("hit_count"), 0)),
        "targets_all": _safe_bool(payload.get("targets_all")),
        "adds_block": _safe_bool(payload.get("adds_block")),
        "adds_buff": _safe_bool(payload.get("adds_buff")),
        "adds_debuff": _safe_bool(payload.get("adds_debuff")),
        "spawn_or_scale": _safe_bool(payload.get("spawn_or_scale")),
        "special_phase_tag": str(payload.get("special_phase_tag") or "").strip().lower(),
    }


def _match_entry(enemy: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    enemy_key = _enemy_key(enemy)
    if not enemy_key:
        return None
    for entry in entries:
        tokens = [token for token in (entry.get("match_any") or []) if isinstance(token, str)]
        if tokens and any(_text(token) in enemy_key for token in tokens):
            return entry
    return None


def _current_step_with_branch_overrides(enemy: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    enemy_key = _enemy_key(enemy)
    observed_text = _intent_text(enemy)
    if "ceremonial_beast" in enemy_key and "stun" in observed_text:
        stunned = dict(step)
        stunned["intent_categories"] = []
        stunned["damage_max"] = 0
        stunned["hit_count"] = 0
        stunned["adds_block"] = False
        stunned["adds_buff"] = False
        stunned["adds_debuff"] = False
        stunned["spawn_or_scale"] = False
        stunned["special_phase_tag"] = "stun"
        return stunned
    return step


def _record_validation_stats(state: GameStateView, alerts: list[dict[str, Any]], matched_scripts: list[str]) -> None:
    _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        stats = json.loads(_STATS_PATH.read_text(encoding="utf-8")) if _STATS_PATH.exists() else {}
    except (OSError, json.JSONDecodeError):
        stats = {}

    stats.setdefault("updated_at", "")
    stats.setdefault("total_validations", 0)
    stats.setdefault("total_alerts", 0)
    stats.setdefault("per_script", {})
    stats["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["total_validations"] = _safe_int(stats.get("total_validations"), 0) + len(matched_scripts)
    stats["total_alerts"] = _safe_int(stats.get("total_alerts"), 0) + len(alerts)

    per_script = stats.get("per_script") or {}
    for script_name in matched_scripts:
        entry = dict(per_script.get(script_name) or {})
        entry["validations"] = _safe_int(entry.get("validations"), 0) + 1
        entry.setdefault("alerts", 0)
        per_script[script_name] = entry
    for row in alerts:
        script_name = str(row.get("script") or "unknown")
        entry = dict(per_script.get(script_name) or {})
        entry.setdefault("validations", 0)
        entry["alerts"] = _safe_int(entry.get("alerts"), 0) + 1
        if entry.get("validations", 0) > 0:
            entry["mismatch_rate"] = round(entry["alerts"] / max(1, entry["validations"]), 4)
        entry["last_alert"] = row.get("ts")
        per_script[script_name] = entry
    for script_name, entry in per_script.items():
        validations = _safe_int(entry.get("validations"), 0)
        alerts_count = _safe_int(entry.get("alerts"), 0)
        entry["mismatch_rate"] = round(alerts_count / max(1, validations), 4) if validations > 0 else 0.0
        per_script[script_name] = entry
    stats["per_script"] = per_script
    stats["overall_mismatch_rate"] = round(
        _safe_int(stats.get("total_alerts"), 0) / max(1, _safe_int(stats.get("total_validations"), 0)),
        4,
    )
    _STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def _post_dashboard_incident(payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        _DASHBOARD_INCIDENT_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=2.0) as response:
            response.read()
    except (OSError, URLError, TimeoutError):
        return


def _report_intent_mismatch_incident(alerts: list[dict[str, Any]]) -> None:
    for row in alerts:
        expected_categories = list(row.get("expected_categories") or [])
        expected_step = dict(row.get("expected_step") or {})
        observed_category = str(row.get("observed_category") or "")
        observed_intent = str(row.get("observed_intent") or "")
        enemy_name = str(row.get("enemy") or "unknown")
        payload = {
            "source": "training_client",
            "type": "enemy_intent_mismatch",
            "severity": "error",
            "recorded_at": row.get("ts"),
            "seed": row.get("seed"),
            "game_id": row.get("game_id"),
            "run_id": row.get("run_id"),
            "act": row.get("act"),
            "floor": row.get("floor"),
            "round": row.get("round"),
            "enemy": enemy_name,
            "script": row.get("script"),
            "expected_categories": expected_categories,
            "expected_step": expected_step,
            "observed_category": observed_category,
            "observed_intent": observed_intent,
            "observed_intended_damage": row.get("observed_intended_damage"),
            "reason": (
                f"enemy intent mismatch: enemy={enemy_name} round={row.get('round')} "
                f"expected={','.join(expected_categories) or '-'} observed={observed_category or '-'} "
                f"intent={observed_intent or '-'}"
            ),
        }
        _post_dashboard_incident(payload)


def describe_enemy_intent(state: GameStateView, *, horizon: int = 3) -> dict[str, Any]:
    db = _load_db()
    entries = [entry for entry in (db.get("entries") or []) if isinstance(entry, dict)]
    round_no = int(state.round or 0)
    rows: list[dict[str, Any]] = []
    for enemy in state.living_enemies():
        entry = _match_entry(enemy, entries)
        if entry is None:
            continue
        timeline = entry.get("timeline") or {}
        current_step = _current_step_with_branch_overrides(enemy, _normalized_step(round_no, timeline.get(str(round_no))))
        forecast_rows: list[dict[str, Any]] = []
        for offset in range(max(1, horizon)):
            forecast_rows.append(_normalized_step(round_no + offset + 1, timeline.get(str(round_no + offset + 1))))
        observed_category = _intent_category(enemy)
        expected_categories = list(current_step.get("intent_categories") or [])
        mismatch = bool(expected_categories) and observed_category not in expected_categories
        next_turn_forecast = forecast_rows[0] if forecast_rows else _normalized_step(round_no + 1, {})
        rows.append(
            {
                "enemy_index": enemy.get("index"),
                "enemy": _enemy_key(enemy),
                "script": _text(entry.get("name")),
                "observed_category": observed_category,
                "observed_intent": _intent_text(enemy),
                "observed_intended_damage": _enemy_intended_damage(enemy),
                "expected_step": current_step,
                "expected_categories": expected_categories,
                "next_turn_forecast": next_turn_forecast,
                "short_horizon_forecast": forecast_rows,
                "predicted_two_turn_damage": sum(step["damage_max"] for step in forecast_rows[:2]),
                "scaling_risk": 1 if any(step["spawn_or_scale"] or step["adds_buff"] for step in forecast_rows[:2]) else 0,
                "mismatch": mismatch,
            }
        )
    return {
        "db_version": db.get("version"),
        "matched_enemies": len(rows),
        "mismatch_count": sum(1 for row in rows if row.get("mismatch")),
        "rows": rows,
    }


def forecast_enemy_intent(state: GameStateView, *, horizon: int = 3) -> dict[str, Any]:
    snapshot = describe_enemy_intent(state, horizon=horizon)
    expected_damage_t1 = 0
    expected_damage_t2 = 0
    scaling_risk = 0
    rows: list[dict[str, Any]] = []
    for row in list(snapshot.get("rows") or []):
        next_turn = dict(row.get("next_turn_forecast") or {})
        forecast_rows = list(row.get("short_horizon_forecast") or [])
        expected_damage_t1 += max(0, _safe_int(next_turn.get("damage_max"), 0))
        if len(forecast_rows) > 1:
            expected_damage_t2 += max(0, _safe_int(forecast_rows[1].get("damage_max"), 0))
        if any(bool(step.get("spawn_or_scale")) for step in forecast_rows):
            scaling_risk += 1
        rows.append(
            {
                "enemy_index": row.get("enemy_index"),
                "enemy": row.get("enemy"),
                "script": row.get("script"),
                "next_turn_forecast": next_turn,
                "short_horizon_forecast": forecast_rows,
                "predicted_two_turn_damage": row.get("predicted_two_turn_damage", 0),
                "scaling_risk": row.get("scaling_risk", 0),
            }
        )
    return {
        "db_version": snapshot.get("db_version"),
        "matched_enemies": snapshot.get("matched_enemies", 0),
        "expected_damage_next_turn": expected_damage_t1,
        "expected_damage_turn_plus_one": expected_damage_t1,
        "expected_damage_turn_plus_two": expected_damage_t2,
        "predicted_two_turn_pressure": expected_damage_t1 + expected_damage_t2,
        "enemy_scaling_risk": scaling_risk,
        "rows": rows,
    }


def validate_enemy_intent(state: GameStateView, *, seen: set[str] | None = None) -> list[dict[str, Any]]:
    round_no = int(state.round or 0)
    context = state.raw.get("context") or {}
    act = int(context.get("act") or 0)
    floor = int(context.get("floor") or 0)
    seed = str(state.raw.get("seed") or "")
    game_id = str(state.raw.get("game_id") or "")
    snapshot = describe_enemy_intent(state, horizon=3)
    alerts: list[dict[str, Any]] = []
    matched_scripts: list[str] = []
    for row in list(snapshot.get("rows") or []):
        script_name = str(row.get("script") or "")
        matched_scripts.append(script_name)
        expected_categories = list(row.get("expected_categories") or [])
        if not expected_categories:
            continue
        observed = str(row.get("observed_category") or "")
        if not row.get("mismatch"):
            continue
        enemy_name = str(row.get("enemy") or "unknown")
        dedupe = f"{game_id}|{seed}|{act}|{floor}|{round_no}|{enemy_name}|{observed}"
        if seen is not None and dedupe in seen:
            continue
        if seen is not None:
            seen.add(dedupe)
        alerts.append(
            {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seed": seed,
                "game_id": game_id,
                "act": act,
                "floor": floor,
                "round": round_no,
                "enemy": enemy_name,
                "script": script_name,
                "expected_categories": expected_categories,
                "expected_step": dict(row.get("expected_step") or {}),
                "observed_category": observed,
                "observed_intent": row.get("observed_intent"),
                "observed_intended_damage": row.get("observed_intended_damage"),
            }
        )
    _record_validation_stats(state, alerts, matched_scripts)
    if alerts:
        _ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _ALERT_PATH.open("a", encoding="utf-8") as handle:
            for row in alerts:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        _report_intent_mismatch_incident(alerts)
    return alerts
