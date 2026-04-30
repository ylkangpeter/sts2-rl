# -*- coding: utf-8 -*-
"""Enemy intent script database, forecast, and runtime validation."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from st2rl.gameplay.types import GameStateView

_SCRIPT_PATH = Path(__file__).resolve().parent / "data" / "enemy_intent_script_v1.json"
_ALERT_PATH = Path("logs") / "intent_script_alerts.jsonl"


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _enemy_key(enemy: dict[str, Any]) -> str:
    return " ".join(_text(enemy.get(key)) for key in ("id", "name", "display_name"))


def _intent_text(enemy: dict[str, Any]) -> str:
    intent_value = enemy.get("intent")
    if isinstance(intent_value, dict):
        parts = [intent_value.get("id"), intent_value.get("name"), intent_value.get("description")]
        return " ".join(_text(part) for part in parts)
    return " ".join(_text(enemy.get(key)) for key in ("intent", "intent_id", "intent_name", "intent_description"))


def _intent_category(enemy: dict[str, Any]) -> str:
    text = _intent_text(enemy)
    if not text:
        return "unknown"
    if any(token in text for token in ("attack", "strike", "slash", "beam", "bite", "hit", "伤害", "攻击")):
        return "attack"
    if any(token in text for token in ("buff", "strength", "armor", "ritual", "power", "强化", "力量", "增益")):
        return "buff"
    if any(token in text for token in ("debuff", "weak", "vulnerable", "frail", "减益", "虚弱", "易伤")):
        return "debuff"
    if any(token in text for token in ("defend", "block", "护甲", "格挡", "防御")):
        return "block"
    if any(token in text for token in ("summon", "spawn", "call", "召唤")):
        return "summon"
    return "unknown"


def _enemy_intended_damage(enemy: dict[str, Any]) -> int:
    for key in ("intent_damage", "damage", "intentDamage"):
        value = enemy.get(key)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


@lru_cache(maxsize=1)
def _load_db() -> dict[str, Any]:
    if not _SCRIPT_PATH.exists():
        return {"version": "missing", "entries": []}
    try:
        return json.loads(_SCRIPT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": "invalid", "entries": []}


def _match_entry(enemy: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    enemy_key = _enemy_key(enemy)
    if not enemy_key:
        return None
    for entry in entries:
        tokens = [token for token in (entry.get("match_any") or []) if isinstance(token, str)]
        if not tokens:
            continue
        if any(_text(token) in enemy_key for token in tokens):
            return entry
    return None


def forecast_enemy_intent(state: GameStateView, *, horizon: int = 2) -> dict[str, Any]:
    db = _load_db()
    entries = [entry for entry in (db.get("entries") or []) if isinstance(entry, dict)]
    round_no = int(state.round or 0)
    expected_damage_next_turn = 0
    matched = 0
    rows: list[dict[str, Any]] = []
    for enemy in state.living_enemies():
        entry = _match_entry(enemy, entries)
        if entry is None:
            continue
        matched += 1
        timeline = entry.get("timeline") or {}
        per_enemy = {"enemy": _enemy_key(enemy), "script": _text(entry.get("name")), "forecast": []}
        for offset in range(horizon):
            step = timeline.get(str(round_no + offset + 1)) or {}
            if not isinstance(step, dict):
                step = {}
            categories = [str(item).strip().lower() for item in (step.get("intent_categories") or []) if str(item).strip()]
            damage_max = int(step.get("damage_max") or 0)
            if offset == 0:
                expected_damage_next_turn += max(0, damage_max)
            per_enemy["forecast"].append(
                {
                    "round": round_no + offset + 1,
                    "intent_categories": categories,
                    "damage_max": damage_max,
                }
            )
        rows.append(per_enemy)
    return {
        "db_version": db.get("version"),
        "matched_enemies": matched,
        "expected_damage_next_turn": expected_damage_next_turn,
        "rows": rows,
    }


def validate_enemy_intent(state: GameStateView, *, seen: set[str] | None = None) -> list[dict[str, Any]]:
    db = _load_db()
    entries = [entry for entry in (db.get("entries") or []) if isinstance(entry, dict)]
    round_no = int(state.round or 0)
    context = state.raw.get("context") or {}
    act = int(context.get("act") or 0)
    floor = int(context.get("floor") or 0)
    seed = str(state.raw.get("seed") or "")
    game_id = str(state.raw.get("game_id") or "")
    alerts: list[dict[str, Any]] = []
    for enemy in state.living_enemies():
        entry = _match_entry(enemy, entries)
        if entry is None:
            continue
        timeline = entry.get("timeline") or {}
        step = timeline.get(str(round_no)) or {}
        if not isinstance(step, dict):
            continue
        expected_categories = [str(item).strip().lower() for item in (step.get("intent_categories") or []) if str(item).strip()]
        if not expected_categories:
            continue
        observed = _intent_category(enemy)
        if observed in expected_categories:
            continue
        enemy_name = str(enemy.get("id") or enemy.get("name") or "unknown")
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
                "script": str(entry.get("name") or ""),
                "expected_categories": expected_categories,
                "observed_category": observed,
                "observed_intent": _intent_text(enemy),
                "observed_intended_damage": _enemy_intended_damage(enemy),
            }
        )
    if alerts:
        _ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _ALERT_PATH.open("a", encoding="utf-8") as handle:
            for row in alerts:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return alerts

