# -*- coding: utf-8 -*-
"""Live dashboard backed by compact telemetry files and training control APIs."""

import json
import os
import subprocess
import threading
import time
from collections import OrderedDict, deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from flask import Flask, Response, jsonify, request, send_file

from st2rl.core.runtime_config import (
    deep_merge_dicts,
    load_runtime_stack_config,
    load_training_launch_defaults,
)

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT / "models" / "http_cli_rl"
ALL_TIME_SUMMARY_PATH = MODELS_ROOT / "all_time_summary.json"
ALL_TIME_SUMMARY_TTL_SECONDS = 300
LIVE_SESSION_CACHE_LIMIT = 100
LIVE_SESSION_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
FAVICON_CANDIDATES = [ROOT / "avatar.ico", ROOT / "favicon.ico"]
SNAPSHOT_CACHE_TTL_SECONDS = 2.0
HISTORICAL_BEST_CACHE_TTL_SECONDS = 30.0
ACTIVE_RUN_STATUS_TTL_SECONDS = 180.0
SNAPSHOT_WORKER_POLL_SECONDS = 2.0
SNAPSHOT_CACHE_LOCK = threading.Lock()
SNAPSHOT_CACHE: dict[str, Any] = {
    "generated_at": 0.0,
    "payload": None,
    "last_error": None,
    "last_error_at": None,
}
HISTORICAL_BEST_CACHE: dict[str, Any] = {"generated_at": 0.0, "payload": None}
WINDOWS_PROCESS_CACHE_TTL_SECONDS = 2.0
WINDOWS_PROCESS_CACHE_LOCK = threading.Lock()
WINDOWS_PROCESS_CACHE: dict[str, Any] = {"generated_at": 0.0, "rows": []}
SNAPSHOT_WORKER_THREAD: threading.Thread | None = None
SNAPSHOT_WORKER_STARTED = False
LAUNCHER_DIR = ROOT / "logs" / "launcher"
WATCHDOG_INCIDENTS_DIR = ROOT / "logs" / "watchdog" / "incidents"
SESSION_SUPERVISOR_INCIDENTS_DIR = ROOT / "logs" / "session_supervisor" / "incidents"
WATCHDOG_STATUS_PATH = ROOT / "logs" / "watchdog" / "current_status.json"
SESSION_SUPERVISOR_STATUS_PATH = ROOT / "logs" / "session_supervisor" / "current_status.json"
TELEMETRY_LOCK = threading.Lock()
TELEMETRY_STATE: dict[str, Any] = {
    "bootstrapped": False,
    "current_run_id": None,
    "runs": {},
    "control_requests": {},
    "historical_best": [],
    "all_time_summary": {
        "updated_at": None,
        "run_count": 0,
        "runs_seen": 0,
        "runs_finished": 0,
        "wins": 0,
        "best_floor": 0,
        "avg_reward_finished": None,
    },
    "recent_incidents": [],
    "watchdog_status": {},
    "session_supervisor_status": {},
    "launcher_logs": {},
}
SESSION_DETAILS_RECENT_LIMIT = 80

app = Flask(__name__)


@app.after_request
def add_no_cache_headers(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _invalidate_snapshot_cache() -> None:
    with SNAPSHOT_CACHE_LOCK:
        SNAPSHOT_CACHE["generated_at"] = 0.0


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    max_lines = 400
    max_bytes = 512 * 1024
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            read_size = min(file_size, max_bytes)
            handle.seek(max(0, file_size - read_size))
            chunk = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = chunk.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _read_log_tail(path: Path, max_lines: int = 40, max_chars: int = 6000) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    tail = "\n".join(lines[-max_lines:]).strip()
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_runtime_config() -> dict[str, Any]:
    return load_runtime_stack_config(ROOT)


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    return deep_merge_dicts(base, override)


def _load_training_defaults() -> dict[str, Any]:
    return load_training_launch_defaults(ROOT)


def _service_base_url() -> str:
    config = _load_runtime_config()
    service = config.get("service") or {}
    if isinstance(service, dict) and service.get("base_url"):
        return str(service.get("base_url")).rstrip("/")
    return "http://127.0.0.1:5000"


def _service_request(
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(f"{_service_base_url()}{path}", data=data, method=method, headers=headers)
    with urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _service_game_rows() -> list[dict[str, Any]]:
    try:
        payload = _service_request("/games")
    except Exception:
        return []
    games = payload.get("games") or []
    return [item for item in games if isinstance(item, dict)]


def _close_service_games(game_ids: list[str]) -> dict[str, Any]:
    closed: list[str] = []
    failed: list[dict[str, Any]] = []
    for game_id in game_ids:
        try:
            result = _service_request(f"/close/{game_id}", method="POST", payload={})
        except Exception as exc:
            failed.append({"game_id": game_id, "message": str(exc)})
            continue
        if result.get("status") == "success":
            closed.append(game_id)
        else:
            failed.append({"game_id": game_id, "message": str(result.get("message") or result)})
    return {"closed": closed, "failed": failed}


def _cleanup_service_games(exclude_game_ids: set[str] | None = None) -> dict[str, Any]:
    exclude = {str(item) for item in (exclude_game_ids or set()) if item}
    try:
        result = _service_request(
            "/admin/cleanup",
            method="POST",
            payload={"exclude_game_ids": sorted(exclude)},
            timeout=12.0,
        )
        if result.get("status") == "success":
            return {
                "requested": list(result.get("requested") or []),
                "closed": list(result.get("closed") or []),
                "failed": list(result.get("failed") or []),
                "remaining": list(result.get("remaining") or []),
                "mode": "service_admin_cleanup",
            }
    except Exception as exc:
        fallback_error = str(exc)
    else:
        fallback_error = str(result)

    candidates = []
    for row in _service_game_rows():
        game_id = str(row.get("game_id") or "")
        if not game_id or game_id in exclude:
            continue
        if row.get("alive", True):
            candidates.append(game_id)
    result = _close_service_games(candidates)
    result["requested"] = candidates
    result["mode"] = "dashboard_close_loop"
    result["fallback_error"] = fallback_error
    return result


def _cleanup_and_shutdown_service_workers() -> dict[str, Any]:
    try:
        result = _service_request(
            "/admin/cleanup",
            method="POST",
            payload={"exclude_game_ids": [], "terminate_workers": True},
            timeout=20.0,
        )
        if result.get("status") == "success":
            result["mode"] = "service_admin_cleanup_terminate_workers"
            return result
    except Exception as exc:
        return {"status": "error", "message": str(exc), "mode": "service_admin_cleanup_terminate_workers"}
    return {"status": "error", "message": "unexpected cleanup response", "mode": "service_admin_cleanup_terminate_workers"}


def _service_game_map() -> dict[str, dict[str, Any]]:
    rows = _service_game_rows()
    return {
        str(row.get("game_id") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("game_id") or "")
    }


def _all_run_dirs() -> list[Path]:
    if not MODELS_ROOT.exists():
        return []
    candidates = [item for item in MODELS_ROOT.glob("*/*") if item.is_dir()]
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates


def _training_payload(run_dir: Path) -> dict[str, Any]:
    data = _read_json(run_dir / "dashboard" / "training_status.json")
    return data if isinstance(data, dict) else {}


def _iso_to_timestamp(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


def _seconds_since(value: Any) -> float | None:
    ts = _iso_to_timestamp(value)
    if ts is None:
        return None
    return max(time.time() - ts, 0.0)


def _run_activity_snapshot(run_dir: Path) -> dict[str, Any]:
    dashboard_dir = run_dir / "dashboard"
    slots_dir = dashboard_dir / "slots"
    training_path = dashboard_dir / "training_status.json"
    training = _training_payload(run_dir)
    status = str(training.get("status") or "").lower()
    training_updated_ts = _iso_to_timestamp(training.get("updated_at"))
    active_slot_count = 0
    latest_slot_ts: float | None = None
    latest_slot_updated_ts: float | None = None

    if slots_dir.exists():
        for slot_file in slots_dir.glob("slot_*.json"):
            try:
                slot_stat_ts = slot_file.stat().st_mtime
            except OSError:
                slot_stat_ts = None
            if slot_stat_ts is not None:
                latest_slot_ts = max(latest_slot_ts or slot_stat_ts, slot_stat_ts)
            row = _read_json(slot_file)
            if not isinstance(row, dict) or not row:
                continue
            if bool(row.get("active", False)):
                active_slot_count += 1
            row_updated_ts = _iso_to_timestamp(row.get("updated_at"))
            if row_updated_ts is not None:
                latest_slot_updated_ts = max(latest_slot_updated_ts or row_updated_ts, row_updated_ts)

    latest_file_ts: float | None = None
    if training_path.exists():
        try:
            latest_file_ts = max(latest_file_ts or 0.0, training_path.stat().st_mtime)
        except OSError:
            pass
    if latest_slot_ts is not None:
        latest_file_ts = max(latest_file_ts or 0.0, latest_slot_ts)

    freshness_candidates = [ts for ts in [latest_file_ts, training_updated_ts, latest_slot_updated_ts] if ts is not None]
    freshest_ts = max(freshness_candidates) if freshness_candidates else run_dir.stat().st_mtime
    freshness_age_seconds = max(0.0, datetime.now().timestamp() - freshest_ts)
    training_recent = training_updated_ts is not None and max(0.0, datetime.now().timestamp() - training_updated_ts) <= ACTIVE_RUN_STATUS_TTL_SECONDS
    slots_recent = latest_slot_updated_ts is not None and max(0.0, datetime.now().timestamp() - latest_slot_updated_ts) <= ACTIVE_RUN_STATUS_TTL_SECONDS
    latest_file_recent = latest_file_ts is not None and max(0.0, datetime.now().timestamp() - latest_file_ts) <= ACTIVE_RUN_STATUS_TTL_SECONDS
    active = (
        (status in {"running", "paused", "stopping"} and training_recent)
        or (active_slot_count > 0 and (slots_recent or latest_file_recent))
    )
    return {
        "run_dir": run_dir,
        "training": training,
        "status": status,
        "active_slot_count": active_slot_count,
        "freshest_ts": freshest_ts,
        "freshness_age_seconds": freshness_age_seconds,
        "active": active,
    }


def _select_run_dir(*, require_active: bool = False) -> Path | None:
    with TELEMETRY_LOCK:
        memory_run = _telemetry_current_run(require_active=require_active)
        if memory_run is not None and memory_run.get("run_dir"):
            return Path(str(memory_run["run_dir"]))
    candidates: list[dict[str, Any]] = []
    for run_dir in _all_run_dirs():
        activity = _run_activity_snapshot(run_dir)
        if require_active and not activity.get("active"):
            continue
        candidates.append(activity)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            1 if item.get("active") else 0,
            float(item.get("freshest_ts") or 0.0),
            int(item.get("active_slot_count") or 0),
        ),
        reverse=True,
    )
    return candidates[0]["run_dir"]


def _active_run_dir() -> Path | None:
    return _select_run_dir(require_active=True)


def _latest_run_dir() -> Path | None:
    return _select_run_dir(require_active=False)


def _read_session_details(run_dir: Path, game_id: str) -> dict[str, Any]:
    data = _read_json(run_dir / "dashboard" / "sessions" / f"{game_id}.json")
    return data if isinstance(data, dict) else {}


def _find_slot_row(run_dir: Path, game_id: str) -> dict[str, Any]:
    slots_dir = run_dir / "dashboard" / "slots"
    if not slots_dir.exists():
        return {}
    for current_file in sorted(slots_dir.glob("slot_*.json")):
        row = _read_json(current_file)
        if isinstance(row, dict) and str(row.get("game_id") or "") == game_id:
            return row
    return {}


def _cache_live_session(game_id: str, session: dict[str, Any]) -> dict[str, Any]:
    LIVE_SESSION_CACHE[game_id] = deepcopy(session)
    LIVE_SESSION_CACHE.move_to_end(game_id)
    while len(LIVE_SESSION_CACHE) > LIVE_SESSION_CACHE_LIMIT:
        LIVE_SESSION_CACHE.popitem(last=False)
    return deepcopy(session)


def _snapshot_state(raw_state: dict[str, Any]) -> dict[str, Any]:
    player = raw_state.get("player") or {}
    context = raw_state.get("context") or {}
    enemies = []
    for enemy in raw_state.get("enemies") or []:
        if not isinstance(enemy, dict):
            continue
        enemies.append(
            {
                "index": enemy.get("index"),
                "name": enemy.get("name"),
                "hp": enemy.get("hp"),
                "max_hp": enemy.get("max_hp"),
                "block": enemy.get("block"),
                "intent": enemy.get("intent"),
            }
        )
    hand = []
    for card in raw_state.get("hand") or []:
        if not isinstance(card, dict):
            continue
        hand.append(
            {
                "index": card.get("index"),
                "id": card.get("id"),
                "name": card.get("name"),
                "cost": card.get("cost"),
                "target_type": card.get("target_type"),
                "can_play": card.get("can_play"),
            }
        )
    return {
        "decision": raw_state.get("decision"),
        "context": dict(context) if isinstance(context, dict) else {},
        "hp": player.get("hp"),
        "max_hp": player.get("max_hp"),
        "block": player.get("block"),
        "gold": player.get("gold"),
        "energy": raw_state.get("energy"),
        "max_energy": raw_state.get("max_energy"),
        "deck_size": player.get("deck_size"),
        "hand_size": len(raw_state.get("hand") or []),
        "round": raw_state.get("round"),
        "turn": raw_state.get("turn"),
        "game_over": raw_state.get("game_over"),
        "victory": raw_state.get("victory"),
        "enemies": enemies,
        "hand": hand,
        "cards": list(raw_state.get("cards") or []),
        "options": list(raw_state.get("options") or []),
        "relics": list(player.get("relics") or []),
        "potions": list(player.get("potions") or []),
        "shop_relics": list(raw_state.get("relics") or []),
        "shop_potions": list(raw_state.get("potions") or []),
        "purge_cost": raw_state.get("purge_cost"),
        "deck": list(player.get("deck") or []),
    }


def _int_or_default(value: Any, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _map_snapshot_from_full_map(full_map: dict[str, Any]) -> dict[str, Any]:
    context = full_map.get("context") or {}
    act = _int_or_default(context.get("act"), 0)
    nodes: list[dict[str, Any]] = []
    visited_ids: list[str] = []
    for row in full_map.get("rows") or []:
        for node in row or []:
            if not isinstance(node, dict):
                continue
            col = _int_or_default(node.get("col"), -1)
            row_index = _int_or_default(node.get("row"), -1)
            node_id = f"a{act}_c{col}_r{row_index}"
            if node.get("visited"):
                visited_ids.append(node_id)
            nodes.append(
                {
                    "id": node_id,
                    "act": act,
                    "col": col,
                    "row": row_index,
                    "room_type": node.get("type"),
                    "symbol": None,
                    "current": bool(node.get("current")),
                    "visited": bool(node.get("visited")),
                    "children": list(node.get("children") or []),
                }
            )
    boss = full_map.get("boss") or {}
    snapshot = {
        "act": act,
        "nodes": nodes,
        "visited_node_ids": list(dict.fromkeys(visited_ids)),
        "context": dict(context) if isinstance(context, dict) else {},
        "current_coord": dict(full_map.get("current_coord") or {}),
    }
    if isinstance(boss, dict) and boss:
        snapshot["boss_info"] = {
            "boss_id": boss.get("id"),
            "boss_name": boss.get("name"),
            "name": boss.get("name"),
            "id": boss.get("id"),
            "col": boss.get("col"),
            "row": boss.get("row"),
        }
    return snapshot


def _build_live_session(run_dir: Path, game_id: str) -> dict[str, Any]:
    try:
        with urlopen(f"{_service_base_url()}/state/{game_id}", timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return deepcopy(LIVE_SESSION_CACHE.get(game_id, {}))

    if not isinstance(payload, dict) or payload.get("status") != "success":
        return deepcopy(LIVE_SESSION_CACHE.get(game_id, {}))

    raw_state = payload.get("state") or {}
    if not isinstance(raw_state, dict):
        return deepcopy(LIVE_SESSION_CACHE.get(game_id, {}))

    slot_row = _find_slot_row(run_dir, game_id)
    cached = deepcopy(LIVE_SESSION_CACHE.get(game_id, {}))
    if not raw_state.get("full_map"):
        try:
            with urlopen(f"{_service_base_url()}/map/{game_id}", timeout=2.0) as response:
                map_payload = json.loads(response.read().decode("utf-8"))
            if isinstance(map_payload, dict) and map_payload.get("status") == "success":
                full_map = map_payload.get("map")
                if isinstance(full_map, dict):
                    raw_state["full_map"] = full_map
        except (OSError, URLError, TimeoutError, json.JSONDecodeError):
            pass
    state_snapshot = _snapshot_state(raw_state)
    context = state_snapshot.get("context") or {}
    boss = context.get("boss") or {}
    summary = dict(cached.get("summary") or {})
    slot_row = _normalize_floor_fields(slot_row) if slot_row else {}
    act_value = slot_row.get("act") or summary.get("act") or context.get("act") or 0
    floor_value = slot_row.get("floor") or summary.get("final_floor") or context.get("floor") or 0
    current_global_floor = _global_floor(act_value, floor_value) if _safe_int(act_value, 0) > 0 else 0
    summary.update(
        {
            "game_id": game_id,
            "seed": slot_row.get("seed") or summary.get("seed"),
            "character": slot_row.get("character") or summary.get("character") or "Ironclad",
            "slot": slot_row.get("slot") or summary.get("slot"),
            "episode_index": slot_row.get("episode_index") or summary.get("episode_index"),
            "episode_reward": slot_row.get("episode_reward") if slot_row else summary.get("episode_reward", 0.0),
            "episode_steps": slot_row.get("episode_steps") if slot_row else summary.get("episode_steps", 0),
            "victory": bool(raw_state.get("victory")),
            "terminated": bool(raw_state.get("game_over")),
            "truncated": False,
            "max_floor": max(_safe_int(slot_row.get("max_floor"), 0), _safe_int(summary.get("max_floor"), 0), _safe_int(floor_value, 0)),
            "max_floor_local": max(_safe_int(slot_row.get("max_floor_local"), 0), _safe_int(summary.get("max_floor_local"), 0), _safe_int(floor_value, 0)),
            "global_floor": current_global_floor,
            "max_global_floor": max(_row_global_floor(slot_row), _row_global_floor(summary), current_global_floor),
            "act": act_value,
            "final_floor": floor_value,
            "final_global_floor": current_global_floor,
            "final_hp": slot_row.get("hp") if slot_row else state_snapshot.get("hp"),
            "max_hp": slot_row.get("max_hp") if slot_row else state_snapshot.get("max_hp"),
            "final_gold": slot_row.get("gold") if slot_row else state_snapshot.get("gold"),
            "started_at": slot_row.get("started_at") or summary.get("started_at"),
            "finished_at": None if not raw_state.get("game_over") else _now_iso(),
            "boss_id": slot_row.get("boss_id") or boss.get("id") or summary.get("boss_id"),
            "boss_name": slot_row.get("boss_name") or boss.get("name") or summary.get("boss_name"),
        }
    )

    session = {
        "summary": summary,
        "initial_state": cached.get("initial_state") or state_snapshot,
        "final_state": state_snapshot,
        "maps": list(cached.get("maps") or []),
        "nodes": list(cached.get("nodes") or []),
        "trace": list(cached.get("trace") or []),
        "saved_at": _now_iso(),
    }

    full_map = raw_state.get("full_map") or {}
    if isinstance(full_map, dict) and full_map.get("rows"):
        act_map = _map_snapshot_from_full_map(full_map)
        remaining_maps = [item for item in session["maps"] if int(item.get("act") or 0) != int(act_map.get("act") or 0)]
        remaining_maps.append(act_map)
        remaining_maps.sort(key=lambda item: int(item.get("act") or 0))
        session["maps"] = remaining_maps

    current_coord = ((raw_state.get("full_map") or {}).get("current_coord") or {}) if isinstance(raw_state.get("full_map"), dict) else {}
    col = current_coord.get("col")
    row = current_coord.get("row")
    act = int(context.get("act") or 0)
    floor = int(context.get("floor") or 0)
    if col is not None and row is not None and act > 0:
        current_col = _int_or_default(col, -1)
        current_row = _int_or_default(row, -1)
        node_id = f"a{act}_c{current_col}_r{current_row}"
        current_nodes = [item for item in session["nodes"] if item.get("node_id") != node_id]
        current_nodes.append(
            {
                "node_id": node_id,
                "act": act,
                "floor": floor,
                "col": current_col,
                "row": current_row,
                "room_type": context.get("room_type"),
                "entry_state": state_snapshot,
                "exit_state": state_snapshot,
                "monsters": [enemy.get("name") for enemy in state_snapshot.get("enemies") or [] if enemy.get("name")],
                "actions": [],
                "rewards": [],
            }
        )
        current_nodes.sort(key=lambda item: (int(item.get("act") or 0), int(item.get("floor") or 0), str(item.get("node_id") or "")))
        session["nodes"] = current_nodes

    return _cache_live_session(game_id, session)


def _build_service_only_session(game_id: str) -> dict[str, Any]:
    cached = deepcopy(LIVE_SESSION_CACHE.get(game_id, {}))
    try:
        with urlopen(f"{_service_base_url()}/state/{game_id}", timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return cached
    if not isinstance(payload, dict) or payload.get("status") != "success":
        return cached
    raw_state = payload.get("state") or {}
    if not isinstance(raw_state, dict):
        return cached
    if not raw_state.get("full_map"):
        try:
            with urlopen(f"{_service_base_url()}/map/{game_id}", timeout=2.0) as response:
                map_payload = json.loads(response.read().decode("utf-8"))
            if isinstance(map_payload, dict) and map_payload.get("status") == "success" and isinstance(map_payload.get("map"), dict):
                raw_state["full_map"] = map_payload["map"]
        except (OSError, URLError, TimeoutError, json.JSONDecodeError):
            pass
    state_snapshot = _snapshot_state(raw_state)
    context = state_snapshot.get("context") or {}
    cached_summary = _normalize_floor_fields(dict(cached.get("summary") or {}))
    act_value = context.get("act") or cached_summary.get("act") or 0
    floor_value = context.get("floor") or cached_summary.get("final_floor") or cached_summary.get("floor") or 0
    current_global_floor = _global_floor(act_value, floor_value) if _safe_int(act_value, 0) > 0 else 0
    summary = {
        "game_id": game_id,
        "seed": cached_summary.get("seed"),
        "character": cached_summary.get("character") or "Ironclad",
        "slot": cached_summary.get("slot"),
        "episode_index": cached_summary.get("episode_index"),
        "episode_reward": cached_summary.get("episode_reward", 0.0),
        "episode_steps": cached_summary.get("episode_steps", 0),
        "victory": bool(raw_state.get("victory")),
        "terminated": bool(raw_state.get("game_over")),
        "truncated": False,
        "max_floor": max(_safe_int(cached_summary.get("max_floor"), 0), _safe_int(floor_value, 0)),
        "max_floor_local": max(_safe_int(cached_summary.get("max_floor_local"), 0), _safe_int(floor_value, 0)),
        "global_floor": current_global_floor,
        "max_global_floor": max(_row_global_floor(cached_summary), current_global_floor),
        "act": act_value,
        "final_floor": floor_value,
        "final_global_floor": current_global_floor,
        "final_hp": state_snapshot.get("hp"),
        "max_hp": state_snapshot.get("max_hp"),
        "final_gold": state_snapshot.get("gold"),
        "started_at": cached_summary.get("started_at"),
        "finished_at": None if not raw_state.get("game_over") else _now_iso(),
        "boss_id": ((context.get("boss") or {}).get("id") if isinstance(context.get("boss"), dict) else None),
        "boss_name": ((context.get("boss") or {}).get("name") if isinstance(context.get("boss"), dict) else None),
    }
    session = {
        "summary": summary,
        "initial_state": cached.get("initial_state") or state_snapshot,
        "final_state": state_snapshot,
        "maps": [],
        "nodes": list(cached.get("nodes") or []),
        "trace": list(cached.get("trace") or []),
        "saved_at": _now_iso(),
    }
    full_map = raw_state.get("full_map") or {}
    if isinstance(full_map, dict) and full_map.get("rows"):
        session["maps"] = [_map_snapshot_from_full_map(full_map)]
    return _cache_live_session(game_id, session)


def _extract_boss_name_from_session(session: dict[str, Any]) -> str | None:
    if not isinstance(session, dict):
        return None
    summary = session.get("summary") or {}
    if isinstance(summary, dict) and summary.get("boss_name"):
        return str(summary.get("boss_name"))
    for container_key in ("initial_state", "final_state"):
        container = session.get(container_key) or {}
        context = (container.get("context") or {}) if isinstance(container, dict) else {}
        boss = context.get("boss") or {}
        if isinstance(boss, dict) and boss.get("name"):
            return str(boss.get("name"))
    for map_item in session.get("maps") or []:
        boss_info = (map_item.get("boss_info") or {}) if isinstance(map_item, dict) else {}
        if isinstance(boss_info, dict) and boss_info.get("boss_name"):
            return str(boss_info.get("boss_name"))
        boss = (map_item.get("boss") or {}) if isinstance(map_item, dict) else {}
        if isinstance(boss, dict) and boss.get("name"):
            return str(boss.get("name"))
    return None


def _enrich_rows_with_boss(run_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    session_cache: dict[str, str | None] = {}
    for row in rows:
        item = dict(row)
        if item.get("boss_name"):
            enriched.append(item)
            continue
        game_id = str(item.get("game_id") or "")
        if not game_id:
            enriched.append(item)
            continue
        if game_id not in session_cache:
            session_cache[game_id] = _extract_boss_name_from_session(_read_session_details(run_dir, game_id))
        if session_cache[game_id]:
            item["boss_name"] = session_cache[game_id]
        enriched.append(item)
    return enriched


def _fetch_live_boss_name(game_id: str) -> str | None:
    if not game_id:
        return None
    try:
        with urlopen(f"{_service_base_url()}/state/{game_id}", timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("status") != "success":
        return None
    state = payload.get("state") or {}
    context = state.get("context") or {}
    boss = context.get("boss") or {}
    if isinstance(boss, dict):
        return boss.get("name") or boss.get("display_name") or boss.get("id")
    return None


def _enrich_active_slots_with_live_boss(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if not item.get("boss_name"):
            item["boss_name"] = _fetch_live_boss_name(str(item.get("game_id") or ""))
        enriched.append(item)
    return enriched


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


def _global_floor(act: Any, floor: Any) -> int:
    act_value = max(1, _safe_int(act, 1))
    floor_value = max(0, _safe_int(floor, 0))
    if act_value <= 1:
        return floor_value
    if act_value == 2:
        return 18 + floor_value
    if act_value == 3:
        return 35 + floor_value
    return 51 + floor_value + max(0, act_value - 4) * 17


def _global_floor_from_progress(progress: Any) -> int:
    value = _safe_int(progress, 0)
    if value <= 0:
        return 0
    if value < 100:
        return value
    act = value // 100 + 1
    floor = value % 100
    return _global_floor(act, floor)


def _row_global_floor(row: dict[str, Any]) -> int:
    for key in ("max_global_floor", "final_global_floor", "global_floor"):
        value = _safe_int(row.get(key), 0)
        if value > 0:
            return value
    progress_floor = _global_floor_from_progress(row.get("max_progress"))
    if progress_floor > 0:
        return progress_floor
    max_global_act = _safe_int(row.get("max_global_act"), 0)
    max_global_act_floor = _safe_int(row.get("max_global_act_floor"), 0)
    if max_global_act > 0 and max_global_act_floor > 0:
        return _global_floor(max_global_act, max_global_act_floor)
    act = _safe_int(row.get("act"), 0)
    floor = _safe_int(row.get("final_floor") or row.get("floor"), 0)
    if act > 0 and floor > 0:
        return _global_floor(act, floor)
    return _safe_int(row.get("max_floor"), 0)


def _normalize_floor_fields(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    act = _safe_int(item.get("act"), 0)
    floor = _safe_int(item.get("floor") or item.get("final_floor"), 0)
    if act > 0 and floor > 0:
        item.setdefault("global_floor", _global_floor(act, floor))
    if "max_floor_local" not in item and item.get("max_floor") is not None:
        item["max_floor_local"] = item.get("max_floor")
    max_global_floor = _row_global_floor(item)
    if max_global_floor > 0:
        item["max_global_floor"] = max_global_floor
    return item


def _session_row_key(row: dict[str, Any]) -> str:
    game_id = str(row.get("game_id") or "").strip()
    if game_id:
        return game_id
    slot = str(row.get("slot") or "").strip()
    episode = _safe_int(row.get("episode_index"), 0)
    seed = str(row.get("seed") or "").strip()
    return f"{slot}-{episode}-{seed}"


def _merge_session_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        row = _normalize_floor_fields(raw)
        key = _session_row_key(row)
        existing = _normalize_floor_fields(merged.get(key, {}))
        max_floor = max(
            _safe_int(existing.get("max_floor"), 0),
            _safe_int(row.get("max_floor"), 0),
            _safe_int(row.get("final_floor") or row.get("floor"), 0),
        )
        max_floor_local = max(
            _safe_int(existing.get("max_floor_local"), 0),
            _safe_int(row.get("max_floor_local"), 0),
            _safe_int(row.get("final_floor") or row.get("floor"), 0),
        )
        max_global_floor = max(_row_global_floor(existing), _row_global_floor(row))
        max_progress = max(_safe_int(existing.get("max_progress"), 0), _safe_int(row.get("max_progress"), 0))
        item = dict(existing)
        item.update(row)
        item["max_floor"] = max_floor
        item["max_floor_local"] = max_floor_local
        item["max_global_floor"] = max_global_floor
        item["max_progress"] = max_progress
        merged[key] = item
    return list(merged.values())


def _build_floor_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    merged_rows = _merge_session_rows(rows)
    completed = [row for row in merged_rows if not bool(row.get("active", False))]
    floors = [max(0, _row_global_floor(_normalize_floor_fields(row))) for row in completed]
    floors = [value for value in floors if value > 0]
    if not floors:
        return {
            "total": 0,
            "max_floor": 0,
            "max_count": 0,
            "points": [],
            "act_markers": [17, 34, 51],
        }
    max_floor = max(floors)
    counts = [0] * max_floor
    for value in floors:
        counts[value - 1] += 1
    points = [{"floor": index + 1, "count": count} for index, count in enumerate(counts)]
    return {
        "total": len(floors),
        "max_floor": max_floor,
        "max_count": max(counts) if counts else 0,
        "points": points,
        "act_markers": [marker for marker in [17, 34, 51] if marker <= max_floor],
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_health_flags(row: dict[str, Any]) -> list[str]:
    flags = [str(item) for item in (row.get("anomaly_flags") or []) if item]
    active = bool(row.get("active", False))
    victory = bool(row.get("victory"))
    terminated = bool(row.get("terminated"))
    truncated = bool(row.get("truncated"))
    final_hp = row.get("final_hp", row.get("hp"))
    if not active and not victory and _safe_int(final_hp, 0) > 0:
        flags.append("nonzero_hp_end")
    if not active and truncated and not terminated and not victory:
        flags.append("truncated_without_game_over")
    elapsed_seconds = _safe_float(row.get("elapsed_seconds"))
    uptime_seconds = _safe_float(row.get("uptime_seconds"))
    duration_seconds = elapsed_seconds if elapsed_seconds is not None else uptime_seconds
    updated_age_seconds = _seconds_since(row.get("updated_at"))
    stagnant_steps = _safe_int(row.get("stagnant_steps"), 0)
    active_stall_hint = stagnant_steps >= 2 or (updated_age_seconds is not None and updated_age_seconds > 75)
    if active and duration_seconds is not None and duration_seconds > 600 and active_stall_hint:
        flags.append("overlong_active")
    elif not active and duration_seconds is not None and duration_seconds > 600:
        flags.append("overlong_episode")
    termination_reason = str(row.get("termination_reason") or "").strip()
    if termination_reason in {"protocol_deadlock", "stuck_abort", "step_exception", "fatal_protocol_error"}:
        flags.append(termination_reason)
    return list(dict.fromkeys(flags))


def _annotate_health(row: dict[str, Any]) -> dict[str, Any]:
    item = _normalize_floor_fields(row)
    flags = _row_health_flags(item)
    item["health_flags"] = flags
    item["flags_display"] = ", ".join(flags)
    item["suspicious"] = bool(flags)
    return item


def _score(item: dict[str, Any]) -> tuple[Any, ...]:
    suspicious = 1 if _row_health_flags(item) else 0
    return (
        -suspicious,
        1 if item.get("victory") else 0,
        _row_global_floor(item),
        _safe_int(item.get("max_act") or item.get("act"), 0),
        1 if item.get("act1_boss_clear") else 0,
        round(float(item.get("episode_reward", 0.0)), 3),
        -int(item.get("episode_steps", 0)),
    )


def _telemetry_empty_run(run_id: str, run_dir: str, experiment_name: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "experiment_name": experiment_name,
        "training": {},
        "active_slots": {},
        "slot_histories": {},
        "top_sessions": [],
        "session_details": {},
        "finished_game_ids": set(),
        "seen_game_ids": set(),
    }


def _telemetry_get_run(payload: dict[str, Any], *, create: bool = True) -> dict[str, Any] | None:
    run_id = str(payload.get("run_id") or "").strip()
    run_dir = str(payload.get("run_dir") or "").strip()
    experiment_name = str(payload.get("experiment_name") or "").strip()
    if not run_id and run_dir:
        run_id = Path(run_dir).name
    if not run_dir and run_id and experiment_name:
        run_dir = str(MODELS_ROOT / experiment_name / run_id)
    if not run_id:
        return None
    runs = TELEMETRY_STATE.setdefault("runs", {})
    run = runs.get(run_id)
    if run is None and create:
        run = _telemetry_empty_run(run_id, run_dir, experiment_name)
        runs[run_id] = run
    if run is None:
        return None
    if run_dir:
        run["run_dir"] = run_dir
    if experiment_name:
        run["experiment_name"] = experiment_name
    return run


def _telemetry_mark_run_idle(*, run_id: str = "", run_dir: str = "", clear_slots: bool = True) -> None:
    with TELEMETRY_LOCK:
        runs = TELEMETRY_STATE.get("runs") or {}
        target_run = runs.get(run_id) if run_id else None
        if target_run is None and run_dir:
            for candidate in runs.values():
                if str(candidate.get("run_dir") or "") == run_dir:
                    target_run = candidate
                    break
        if target_run is None:
            return
        training = dict(target_run.get("training") or {})
        training["status"] = "idle"
        training["updated_at"] = _now_iso()
        training["process_pid"] = 0
        training["process_count"] = 0
        training["thread_count"] = 0
        training["fps"] = 0.0
        target_run["training"] = training
        if clear_slots:
            target_run["active_slots"] = {}
    _invalidate_snapshot_cache()


def _telemetry_sort_top(rows: list[dict[str, Any]], *, limit: int = 50) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("game_id") or f"{row.get('slot')}:{row.get('episode_index')}")
        if not key:
            continue
        row = _normalize_floor_fields(row)
        existing = deduped.get(key)
        if existing is None or _score(row) > _score(existing):
            deduped[key] = dict(row)
    return sorted(deduped.values(), key=_score, reverse=True)[:limit]


def _telemetry_update_historical_best(summary: dict[str, Any], run: dict[str, Any]) -> None:
    if not summary.get("game_id"):
        return
    entry = _normalize_floor_fields(summary)
    entry["run_id"] = run.get("run_id")
    entry["run_dir"] = run.get("run_dir")
    entry["experiment_name"] = run.get("experiment_name")
    historical = [row for row in (TELEMETRY_STATE.get("historical_best") or []) if row.get("game_id") != entry.get("game_id")]
    historical.append(entry)
    TELEMETRY_STATE["historical_best"] = _telemetry_sort_top(historical, limit=50)


def _prune_session_details(run: dict[str, Any], *, recent_limit: int = SESSION_DETAILS_RECENT_LIMIT) -> None:
    details = run.get("session_details")
    if not isinstance(details, dict) or not details:
        return
    keep_ids: set[str] = set()
    keep_ids.update(
        str(row.get("game_id") or "")
        for row in (run.get("top_sessions") or [])
        if isinstance(row, dict) and str(row.get("game_id") or "")
    )
    keep_ids.update(
        str(row.get("game_id") or "")
        for row in (run.get("active_slots") or {}).values()
        if isinstance(row, dict) and str(row.get("game_id") or "")
    )
    keys = list(details.keys())
    if recent_limit > 0 and keys:
        keep_ids.update(keys[-recent_limit:])
    removable = [key for key in keys if key not in keep_ids]
    for key in removable:
        details.pop(key, None)


def _compact_session_details_for_memory(details: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(details, dict):
        return {}
    compact = dict(details)
    # Large per-step trace stays on disk evidence; dashboard runtime keeps map/node drill-down only.
    compact.pop("trace", None)
    return compact


def _telemetry_update_all_time(summary: dict[str, Any], run: dict[str, Any]) -> None:
    all_time = dict(TELEMETRY_STATE.get("all_time_summary") or {})
    seen_key = str(summary.get("game_id") or "")
    finished_ids = run.setdefault("finished_game_ids", set())
    seen_ids = run.setdefault("seen_game_ids", set())
    if seen_key:
        seen_ids.add(seen_key)
    total_seen = sum(len(item.get("seen_game_ids") or set()) for item in (TELEMETRY_STATE.get("runs") or {}).values())
    total_finished = sum(len(item.get("finished_game_ids") or set()) for item in (TELEMETRY_STATE.get("runs") or {}).values())
    all_time["updated_at"] = _now_iso()
    all_time["run_count"] = len(TELEMETRY_STATE.get("runs") or {})
    all_time["runs_seen"] = total_seen
    all_time["runs_finished"] = total_finished
    all_time["best_floor"] = max(_safe_int(all_time.get("best_floor"), 0), _row_global_floor(_normalize_floor_fields(summary)))
    wins = _safe_int(all_time.get("wins"), 0)
    if bool(summary.get("victory")) and seen_key and seen_key not in finished_ids:
        wins += 1
    all_time["wins"] = wins
    TELEMETRY_STATE["all_time_summary"] = all_time


def _telemetry_snapshot_from_memory(run: dict[str, Any]) -> dict[str, Any]:
    active_slots = [_annotate_health(dict(row)) for row in run.get("active_slots", {}).values() if bool(row.get("active", False))]
    active_slots.sort(key=lambda item: str(item.get("slot") or ""))
    top_sessions = [_annotate_health(dict(row)) for row in run.get("top_sessions") or []]
    suspicious_completed = [row for row in top_sessions if row.get("suspicious")]
    overlong_active = [row for row in active_slots if "overlong_active" in (row.get("health_flags") or [])]
    recent_incidents = _filter_incidents_for_run(TELEMETRY_STATE.get("recent_incidents") or [], run)
    active_game_ids = {str(item.get("game_id") or "") for item in active_slots if str(item.get("game_id") or "")}
    problem_list = _build_problem_list(recent_incidents, suspicious_completed, active_game_ids=active_game_ids, runtime_healthy=True)
    detailed_ids = set((run.get("session_details") or {}).keys())
    for item in active_slots:
        item["details_available"] = bool(item.get("details_available") or item.get("game_id") in detailed_ids)
    for item in top_sessions:
        item["details_available"] = bool(item.get("details_available") or item.get("game_id") in detailed_ids)
    training = deepcopy(run.get("training") or {})
    resources = _collect_resource_summary(training)
    trend = _build_trend_summary(training, top_sessions)
    floor_distribution_rows: list[dict[str, Any]] = [dict(row) for row in top_sessions]
    for rows in (run.get("slot_histories") or {}).values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                floor_distribution_rows.append(dict(row))
    floor_distribution = _build_floor_distribution(floor_distribution_rows)
    return {
        "updated_at": training.get("updated_at"),
        "run_dir": run.get("run_dir"),
        "training": training,
        "active_slots": active_slots,
        "top_sessions": top_sessions,
        "slot_histories": deepcopy(run.get("slot_histories") or {}),
        "checkpoints": [],
        "seen_sessions": max(len(run.get("seen_game_ids") or set()), _safe_int(run.get("snapshot_seen_sessions"), 0)),
        "completed_sessions": max(len(run.get("finished_game_ids") or set()), _safe_int(run.get("snapshot_completed_sessions"), 0)),
        "defaults": _load_training_defaults(),
        "monitoring": {
            "resume_model_path": training.get("resume_model_path"),
            "resume_load_status": training.get("resume_load_status"),
            "resume_failure_reason": training.get("resume_failure_reason"),
            "overlong_active_count": len(overlong_active),
            "overlong_active": overlong_active[:10],
            "suspicious_completed_count": len(suspicious_completed),
            "suspicious_completed": suspicious_completed[:10],
            "recent_incidents_count": len(recent_incidents),
            "recent_incidents": recent_incidents,
            "problem_list": problem_list,
            "resources": resources,
            "trend": trend,
            "floor_distribution": floor_distribution,
            "watchdog_status": deepcopy(TELEMETRY_STATE.get("watchdog_status") or {}),
            "session_supervisor_status": deepcopy(TELEMETRY_STATE.get("session_supervisor_status") or {}),
        },
    }


def _telemetry_current_run(require_active: bool = False) -> dict[str, Any] | None:
    runs = TELEMETRY_STATE.get("runs") or {}
    if not runs:
        return None
    current_run_id = str(TELEMETRY_STATE.get("current_run_id") or "")
    current = runs.get(current_run_id) if current_run_id else None
    if current is not None:
        training_status = str((current.get("training") or {}).get("status") or "").lower()
        active_slots = [row for row in (current.get("active_slots") or {}).values() if bool(row.get("active", False))]
        if not require_active or training_status in {"running", "paused", "stopping"} or active_slots:
            return current
    scored: list[tuple[Any, dict[str, Any]]] = []
    for run in runs.values():
        training = run.get("training") or {}
        training_status = str(training.get("status") or "").lower()
        updated_ts = _iso_to_timestamp(training.get("updated_at")) or 0.0
        active_slots = [row for row in (run.get("active_slots") or {}).values() if bool(row.get("active", False))]
        if require_active and training_status not in {"running", "paused", "stopping"} and not active_slots:
            continue
        score = (
            1 if training_status in {"running", "paused", "stopping"} or active_slots else 0,
            len(active_slots),
            updated_ts,
        )
        scored.append((score, run))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _sanitize_incident_payload(payload: dict[str, Any]) -> dict[str, Any]:
    row = dict(payload)
    for key in (
        "session",
        "details",
        "trace",
        "nodes",
        "maps",
        "initial_state",
        "final_state",
        "session_snapshot",
        "close_result",
        "cleanup_result",
        "stack_stop",
    ):
        row.pop(key, None)
    recent_errors = row.get("recent_errors")
    if isinstance(recent_errors, list):
        row["recent_errors"] = [str(item)[:240] for item in recent_errors[:3]]
    reason = str(row.get("reason") or "")
    if len(reason) > 400:
        row["reason"] = reason[:400] + "..."
    return row


def _collect_run_snapshot(run_dir: Path) -> dict[str, Any]:
    dashboard_dir = run_dir / "dashboard"
    training = _training_payload(run_dir)
    persisted_leaderboard = _read_json(dashboard_dir / "session_leaderboard.json")
    persisted_leaderboard = persisted_leaderboard if isinstance(persisted_leaderboard, list) else []
    slots_dir = dashboard_dir / "slots"
    live_games = _service_game_map()

    active_slots = []
    histories = []
    slot_history_map: dict[str, list[dict[str, Any]]] = {}
    if slots_dir.exists():
        for current_file in sorted(slots_dir.glob("slot_*.json")):
            current = _read_json(current_file)
            if isinstance(current, dict) and current:
                active_slots.append(current)
        for history_file in sorted(slots_dir.glob("slot_*.history.jsonl")):
            rows = _read_jsonl(history_file)
            slot_key = history_file.stem.replace(".history", "")
            slot_history_map[slot_key] = [_normalize_floor_fields(row) for row in rows[-50:]]
            histories.extend(rows)

    history_by_job: dict[str, dict[str, Any]] = {}
    for row in histories:
        row = _normalize_floor_fields(row)
        key = row.get("game_id") or f"{row.get('slot')}-{row.get('episode_index')}"
        existing = _normalize_floor_fields(history_by_job.get(key, {}))
        max_floor = max(
            _safe_int(existing.get("max_floor"), 0),
            _safe_int(row.get("max_floor"), 0),
            _safe_int(row.get("floor"), 0),
        )
        max_floor_local = max(
            _safe_int(existing.get("max_floor_local"), 0),
            _safe_int(row.get("max_floor_local"), 0),
            _safe_int(row.get("floor"), 0),
        )
        max_global_floor = max(_row_global_floor(existing), _row_global_floor(row))
        max_progress = max(_safe_int(existing.get("max_progress"), 0), _safe_int(row.get("max_progress") or row.get("floor"), 0))
        merged = dict(existing)
        merged.update(row)
        merged["max_floor"] = max_floor
        merged["max_floor_local"] = max_floor_local
        merged["max_global_floor"] = max_global_floor
        merged["max_progress"] = max_progress
        history_by_job[key] = merged

    normalized_active_slots = []
    for slot in active_slots:
        slot = _normalize_floor_fields(slot)
        key = slot.get("game_id") or f"{slot.get('slot')}-{slot.get('episode_index')}"
        live_row = live_games.get(str(slot.get("game_id") or ""))
        if isinstance(live_row, dict):
            slot["uptime_seconds"] = live_row.get("uptime_seconds")
            slot["service_created_at"] = live_row.get("created_at")
            slot["service_start_time"] = live_row.get("start_time")
            slot["service_alive"] = live_row.get("alive")
        existing = _normalize_floor_fields(history_by_job.get(key, {}))
        max_floor = max(
            _safe_int(existing.get("max_floor"), 0),
            _safe_int(slot.get("max_floor"), 0),
            _safe_int(slot.get("floor"), 0),
        )
        max_floor_local = max(
            _safe_int(existing.get("max_floor_local"), 0),
            _safe_int(slot.get("max_floor_local"), 0),
            _safe_int(slot.get("floor"), 0),
        )
        current_progress = _safe_int(slot.get("max_progress"), 0)
        if current_progress <= 0:
            current_progress = max(0, (_safe_int(slot.get("act"), 0) - 1) * 100 + _safe_int(slot.get("floor"), 0))
        slot["max_progress"] = current_progress
        slot = _normalize_floor_fields(slot)
        max_global_floor = max(_row_global_floor(existing), _row_global_floor(slot))
        max_progress = max(_safe_int(existing.get("max_progress"), 0), current_progress)
        merged = dict(existing)
        merged.update(slot)
        merged["max_floor"] = max_floor
        merged["max_floor_local"] = max_floor_local
        merged["max_global_floor"] = max_global_floor
        merged["max_progress"] = max_progress
        history_by_job[key] = merged
        normalized_active_slots.append(slot)
    active_slots = normalized_active_slots

    computed_top_sessions = sorted(history_by_job.values(), key=_score, reverse=True)[:50]
    merged_leaderboard_rows = [row for row in persisted_leaderboard if isinstance(row, dict)] + computed_top_sessions
    top_sessions = _telemetry_sort_top(merged_leaderboard_rows, limit=50)
    top_sessions = _enrich_rows_with_boss(run_dir, top_sessions)
    # Avoid high-frequency per-slot live boss HTTP calls in snapshot refresh.
    # These requests contend with training traffic and can stall step throughput.
    active_slots = _enrich_rows_with_boss(run_dir, active_slots)
    active_slots = [_annotate_health(item) for item in active_slots]
    top_sessions = [_annotate_health(item) for item in top_sessions]

    sessions_dir = dashboard_dir / "sessions"
    detailed_ids = {path.stem for path in sessions_dir.glob("*.json")} if sessions_dir.exists() else set()
    for item in active_slots:
        item["details_available"] = bool(
            item.get("active")
            or item.get("details_available", False)
            or item.get("game_id") in detailed_ids
        )
    for item in top_sessions:
        item["details_available"] = bool(
            item.get("active")
            or item.get("details_available", False)
            or item.get("game_id") in detailed_ids
        )

    checkpoints = []
    for model_file in sorted(run_dir.glob("*.zip"), key=lambda item: item.stat().st_mtime, reverse=True):
        checkpoints.append(
            {
                "name": model_file.name,
                "path": str(model_file),
                "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                "updated_at": model_file.stat().st_mtime,
            }
        )

    suspicious_completed = sorted(
        (_annotate_health(item) for item in history_by_job.values() if not item.get("active", False)),
        key=lambda item: (
            _safe_int(item.get("episode_index"), 0),
            _row_global_floor(item),
            _safe_float(item.get("elapsed_seconds")) or 0.0,
        ),
        reverse=True,
    )
    suspicious_completed = [item for item in suspicious_completed if item.get("suspicious")]
    overlong_active = [item for item in active_slots if "overlong_active" in (item.get("health_flags") or [])]
    recent_incidents = _filter_incidents_for_run(
        _collect_recent_runtime_incidents(),
        {"run_id": run_dir.name, "run_dir": str(run_dir), "training": training},
    )
    active_game_ids = {str(item.get("game_id") or "") for item in active_slots if str(item.get("game_id") or "")}
    problem_list = _build_problem_list(recent_incidents, suspicious_completed, active_game_ids=active_game_ids, runtime_healthy=True)
    resources = _collect_resource_summary(training)
    trend = _build_trend_summary(training, top_sessions)
    floor_distribution = _build_floor_distribution(list(history_by_job.values()))
    completed_session_count = max(
        sum(1 for item in history_by_job.values() if not item.get("active", False)),
        _safe_int(training.get("episodes_finished"), 0),
    )
    seen_session_count = max(len(history_by_job), completed_session_count)

    return {
        "updated_at": training.get("updated_at"),
        "run_dir": str(run_dir),
        "training": training,
        "active_slots": sorted(active_slots, key=lambda item: str(item.get("slot"))),
        "top_sessions": top_sessions,
        "slot_histories": slot_history_map,
        "checkpoints": checkpoints,
        "seen_sessions": seen_session_count,
        "completed_sessions": completed_session_count,
        "defaults": _load_training_defaults(),
        "monitoring": {
            "resume_model_path": training.get("resume_model_path"),
            "resume_load_status": training.get("resume_load_status"),
            "resume_failure_reason": training.get("resume_failure_reason"),
            "overlong_active_count": len(overlong_active),
            "overlong_active": overlong_active[:10],
            "suspicious_completed_count": len(suspicious_completed),
            "suspicious_completed": suspicious_completed[:10],
            "recent_incidents_count": len(recent_incidents),
            "recent_incidents": recent_incidents,
            "problem_list": problem_list,
            "resources": resources,
            "trend": trend,
            "floor_distribution": floor_distribution,
            "watchdog_status": _read_json(WATCHDOG_STATUS_PATH),
            "session_supervisor_status": _read_json(SESSION_SUPERVISOR_STATUS_PATH),
        },
    }


def _collect_historical_best() -> list[dict[str, Any]]:
    if TELEMETRY_STATE.get("bootstrapped"):
        return deepcopy(TELEMETRY_STATE.get("historical_best") or [])
    cached_payload = HISTORICAL_BEST_CACHE.get("payload")
    cached_at = float(HISTORICAL_BEST_CACHE.get("generated_at") or 0.0)
    now = time.time()
    if isinstance(cached_payload, list) and (now - cached_at) < HISTORICAL_BEST_CACHE_TTL_SECONDS:
        return deepcopy(cached_payload)

    merged: dict[str, dict[str, Any]] = {}
    for run_dir in _all_run_dirs():
        leaderboard = _read_json(run_dir / "dashboard" / "session_leaderboard.json")
        if not isinstance(leaderboard, list):
            continue
        for row in leaderboard:
            if not isinstance(row, dict):
                continue
            key = str(row.get("game_id") or f"{run_dir.name}:{row.get('slot')}:{row.get('episode_index')}")
            existing = merged.get(key)
            if existing is None or _score(row) > _score(existing):
                entry = dict(row)
                entry["run_id"] = run_dir.name
                entry["experiment_name"] = run_dir.parent.name
                entry["run_dir"] = str(run_dir)
                if not entry.get("boss_name"):
                    entry["boss_name"] = _extract_boss_name_from_session(_read_session_details(run_dir, str(entry.get("game_id") or "")))
                merged[key] = entry
    payload = sorted(merged.values(), key=_score, reverse=True)[:50]
    payload = [_annotate_health(item) for item in payload]
    HISTORICAL_BEST_CACHE["generated_at"] = now
    HISTORICAL_BEST_CACHE["payload"] = deepcopy(payload)
    return payload


def _compute_all_time_summary() -> dict[str, Any]:
    seen_ids: set[str] = set()
    finished_ids: set[str] = set()
    victories = 0
    best_floor = 0
    reward_total = 0.0
    reward_count = 0
    run_count = 0

    for run_dir in _all_run_dirs():
        run_count += 1
        dashboard_dir = run_dir / "dashboard"
        slots_dir = dashboard_dir / "slots"

        if slots_dir.exists():
            for history_file in slots_dir.glob("slot_*.history.jsonl"):
                for row in _read_jsonl(history_file):
                    if not isinstance(row, dict):
                        continue
                    game_id = str(row.get("game_id") or "")
                    if not game_id or game_id in finished_ids:
                        continue
                    finished_ids.add(game_id)
                    seen_ids.add(game_id)
                    best_floor = max(best_floor, _row_global_floor(_normalize_floor_fields(row)))
                    if row.get("victory"):
                        victories += 1
                    try:
                        reward_total += float(row.get("episode_reward", 0.0) or 0.0)
                        reward_count += 1
                    except (TypeError, ValueError):
                        pass

        leaderboard = _read_json(dashboard_dir / "session_leaderboard.json")
        if isinstance(leaderboard, list):
            for row in leaderboard:
                if not isinstance(row, dict):
                    continue
                game_id = str(row.get("game_id") or "")
                if not game_id:
                    continue
                seen_ids.add(game_id)
                best_floor = max(best_floor, _row_global_floor(_normalize_floor_fields(row)))

        for current_file in slots_dir.glob("slot_*.json") if slots_dir.exists() else []:
            row = _read_json(current_file)
            if not isinstance(row, dict) or not row:
                continue
            game_id = str(row.get("game_id") or "")
            if game_id:
                seen_ids.add(game_id)

    return {
        "updated_at": _now_iso(),
        "run_count": run_count,
        "runs_seen": len(seen_ids),
        "runs_finished": len(finished_ids),
        "wins": victories,
        "best_floor": best_floor,
        "avg_reward_finished": round(reward_total / reward_count, 2) if reward_count else None,
    }


def _all_time_summary_is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    age_seconds = max(0.0, datetime.now().timestamp() - path.stat().st_mtime)
    return age_seconds >= ALL_TIME_SUMMARY_TTL_SECONDS


def _get_all_time_summary() -> dict[str, Any]:
    if TELEMETRY_STATE.get("bootstrapped"):
        return deepcopy(TELEMETRY_STATE.get("all_time_summary") or {})
    if _all_time_summary_is_stale(ALL_TIME_SUMMARY_PATH):
        MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        payload = _compute_all_time_summary()
        ALL_TIME_SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    data = _read_json(ALL_TIME_SUMMARY_PATH)
    if isinstance(data, dict):
        return data
    payload = _compute_all_time_summary()
    ALL_TIME_SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _set_runtime_control(run_id: str | None, command: str | None) -> None:
    normalized_run_id = str(run_id or "").strip()
    normalized_command = str(command or "").strip().lower()
    with TELEMETRY_LOCK:
        requests = dict(TELEMETRY_STATE.get("control_requests") or {})
        if normalized_run_id:
            if normalized_command:
                requests[normalized_run_id] = {
                    "run_id": normalized_run_id,
                    "command": normalized_command,
                    "updated_at": _now_iso(),
                }
            else:
                requests.pop(normalized_run_id, None)
        elif not normalized_command:
            requests.clear()
        TELEMETRY_STATE["control_requests"] = requests


def _get_runtime_control(run_id: str | None = None) -> dict[str, Any]:
    normalized_run_id = str(run_id or "").strip()
    with TELEMETRY_LOCK:
        requests = TELEMETRY_STATE.get("control_requests") or {}
        if normalized_run_id:
            payload = requests.get(normalized_run_id) or {}
            return deepcopy(payload) if isinstance(payload, dict) else {}
        current_run_id = str(TELEMETRY_STATE.get("current_run_id") or "").strip()
        if current_run_id:
            payload = requests.get(current_run_id) or {}
            return deepcopy(payload) if isinstance(payload, dict) else {}
    return {}


def _force_training_status(run_dir: Path, status: str) -> None:
    training_path = run_dir / "dashboard" / "training_status.json"
    payload = _training_payload(run_dir)
    payload["status"] = status
    payload["updated_at"] = _now_iso()
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _python_executable() -> str:
    return str(_load_runtime_config().get("python_executable") or "python")


def _node_python_executable(node: dict[str, Any] | None = None) -> str:
    if isinstance(node, dict):
        value = str(node.get("python_executable") or "").strip()
        if value:
            return value
    return _python_executable()


def _runtime_node(name: str) -> dict[str, Any]:
    node = _load_runtime_config().get(name) or {}
    return node if isinstance(node, dict) else {}


def _resolve_managed_path(base_dir: Path, value: Any) -> Path:
    raw = str(value or "").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _shared_runtime_env() -> dict[str, str]:
    runtime = _load_runtime_config()
    projects = runtime.get("projects") or {}
    game_dir = str((projects or {}).get("game_dir") or "").strip()
    env: dict[str, str] = {"PYTHONUNBUFFERED": "1"}
    src_dir = str((ROOT / "src").resolve())
    existing_pythonpath = os.environ.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([src_dir, existing_pythonpath])
    else:
        env["PYTHONPATH"] = src_dir
    if game_dir:
        env["STS2_GAME_DIR"] = game_dir
    return env


def _windows_process_kwargs(*, detached: bool = False) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW
        if detached:
            creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        kwargs["creationflags"] = creationflags
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
    return kwargs


def _client_config() -> dict[str, Any]:
    return _runtime_node("client")


def _service_config() -> dict[str, Any]:
    return _runtime_node("service")


def _watchdog_config() -> dict[str, Any]:
    return _runtime_node("watchdog")


def _session_supervisor_config() -> dict[str, Any]:
    return _runtime_node("session_supervisor")


def _support_node_names(*, include_disabled: bool = False) -> list[str]:
    names = ["service"]
    if include_disabled or bool(_watchdog_config().get("enabled", False)):
        names.append("watchdog")
    if include_disabled or bool(_session_supervisor_config().get("enabled", False)):
        names.append("session_supervisor")
    return names


def _pid_file(name: str) -> Path:
    return LAUNCHER_DIR / f"{name}.pid"


def _record_pid(name: str, pid: int) -> None:
    LAUNCHER_DIR.mkdir(parents=True, exist_ok=True)
    _pid_file(name).write_text(str(int(pid)), encoding="utf-8")


def _clear_pid(name: str) -> None:
    path = _pid_file(name)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"$ErrorActionPreference='Stop'; Get-Process -Id {int(pid)} | Out-Null",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            **_windows_process_kwargs(),
        )
    except Exception:
        return False
    return result.returncode == 0


def _pid_from_file(name: str) -> int | None:
    path = _pid_file(name)
    if not path.exists():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except Exception:
        _clear_pid(name)
        return None
    if _process_exists(pid):
        return pid
    _clear_pid(name)
    return None


def _managed_process_ids(name: str) -> list[int]:
    pid = _pid_from_file(name)
    node = _runtime_node(name)
    candidates: list[int] = []
    seen: set[int] = set()
    if pid is not None:
        seen.add(pid)
        candidates.append(pid)
    patterns = [str(node.get("process_match") or "").strip()]
    script_name = Path(str(node.get("script") or "")).name.strip()
    if script_name:
        patterns.append(script_name)
    for pattern in patterns:
        if not pattern:
            continue
        for current_pid in _find_python_process_ids(pattern):
            if current_pid not in seen:
                seen.add(current_pid)
                candidates.append(current_pid)
    return candidates


def _find_python_process_ids(process_match: str) -> list[int]:
    needle = str(process_match or "").strip()
    if not needle:
        return []
    normalized = needle.replace("\\", "/").lower()
    if psutil is not None:
        matches: list[int] = []
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = str(process.info.get("name") or "").lower()
                if name not in {"python.exe", "pythonw.exe"}:
                    continue
                cmdline = " ".join(str(part) for part in (process.info.get("cmdline") or []))
                if normalized in cmdline.replace("\\", "/").lower():
                    matches.append(int(process.info["pid"]))
            except Exception:
                continue
        if matches:
            return matches
    matches: list[int] = []
    for row in _windows_process_rows():
        name = str(row.get("Name") or row.get("name") or "").lower()
        if name not in {"python.exe", "pythonw.exe"}:
            continue
        command_line = str(row.get("CommandLine") or row.get("command_line") or "").replace("\\", "/").lower()
        if normalized in command_line:
            pid = _safe_int(row.get("ProcessId") or row.get("pid"), 0)
            if pid > 0:
                matches.append(pid)
    return matches


def _windows_process_rows(*, force_refresh: bool = False) -> list[dict[str, Any]]:
    if not force_refresh:
        with WINDOWS_PROCESS_CACHE_LOCK:
            cached_rows = WINDOWS_PROCESS_CACHE.get("rows")
            cached_at = float(WINDOWS_PROCESS_CACHE.get("generated_at") or 0.0)
        if isinstance(cached_rows, list) and cached_rows and (time.time() - cached_at) < WINDOWS_PROCESS_CACHE_TTL_SECONDS:
            return cached_rows
    try:
        raw = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -in @('python.exe','pythonw.exe','dotnet.exe') } | "
                "Select-Object ProcessId,ParentProcessId,Name,CommandLine,WorkingSetSize,PageFileUsage,ThreadCount | "
                "ConvertTo-Json -Depth 3 -Compress",
            ],
            text=True,
            cwd=str(ROOT),
            timeout=15,
            **_windows_process_kwargs(),
        )
    except Exception:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    if isinstance(parsed, dict):
        rows = [parsed]
    elif isinstance(parsed, list):
        rows = [row for row in parsed if isinstance(row, dict)]
    with WINDOWS_PROCESS_CACHE_LOCK:
        WINDOWS_PROCESS_CACHE["generated_at"] = time.time()
        WINDOWS_PROCESS_CACHE["rows"] = rows
    return rows


def _process_resource_stats_by_pid(pid: int) -> dict[str, Any]:
    if pid <= 0:
        return {}
    try:
        raw = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Get-Process -Id {int(pid)} | "
                "Select-Object Id,ProcessName,WorkingSet64,PrivateMemorySize64,@{Name='ThreadCount';Expression={$_.Threads.Count}} | "
                "ConvertTo-Json -Depth 4 -Compress",
            ],
            text=True,
            cwd=str(ROOT),
            timeout=10,
            **_windows_process_kwargs(),
        )
        row = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(row, dict):
        return {}
    thread_count = _safe_int(row.get("ThreadCount"), 0)
    return {
        "pid": pid,
        "name": str(row.get("ProcessName") or ""),
        "working_set_mb": round(float(_safe_int(row.get("WorkingSet64"), 0)) / (1024 * 1024), 1),
        "private_mb": round(float(_safe_int(row.get("PrivateMemorySize64"), 0)) / (1024 * 1024), 1),
        "threads": thread_count,
    }


def _find_training_process_ids() -> list[int]:
    pid = _pid_from_file("client")
    candidates: list[int] = []
    seen: set[int] = set()
    if pid is not None:
        seen.add(pid)
        candidates.append(pid)
    for current_pid in _find_python_process_ids("train_http_cli_rl.py"):
        if current_pid not in seen:
            seen.add(current_pid)
            candidates.append(current_pid)
    return candidates


def _find_service_process_ids() -> list[int]:
    pid = _pid_from_file("service")
    if pid is not None:
        return [pid]
    service = _service_config()
    candidates: list[int] = []
    seen: set[int] = set()
    patterns = [
        str(service.get("process_match") or "").strip(),
        Path(str(service.get("script") or "")).name.strip(),
        "http_game_service.py",
        "run_sts2_cli_service.py",
    ]
    for pattern in patterns:
        if not pattern:
            continue
        for current_pid in _find_python_process_ids(pattern):
            if current_pid not in seen:
                seen.add(current_pid)
                candidates.append(current_pid)
    return candidates


def _terminate_processes(pids: list[int]) -> list[int]:
    terminated: list[int] = []
    for pid in pids:
        try:
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Stop-Process -Id {int(pid)} -Force",
                ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            **_windows_process_kwargs(),
            )
            terminated.append(int(pid))
        except Exception:
            continue
    return terminated


def _terminate_training_processes(pids: list[int]) -> list[int]:
    return _terminate_processes(pids)


def _wait_for_process_exit(
    probe,
    timeout_seconds: float = 8.0,
    poll_interval_seconds: float = 0.5,
) -> list[int]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        pids = probe()
        if not pids:
            return []
        time.sleep(poll_interval_seconds)
    return probe()


def _wait_for_training_exit(timeout_seconds: float = 8.0, poll_interval_seconds: float = 0.5) -> list[int]:
    return _wait_for_process_exit(
        _find_training_process_ids,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


def _service_health_ok(timeout: float = 3.0) -> bool:
    try:
        health = _service_request("/health", timeout=timeout)
    except Exception:
        return False
    return str(health.get("status") or "").lower() == "healthy"


def _start_managed_process(name: str, node: dict[str, Any], *, default_root: Path) -> dict[str, Any]:
    if not bool(node.get("enabled", False)):
        return {"ok": False, "message": f"{name} disabled in runtime config"}

    script_path = _resolve_managed_path(default_root, node.get("script"))
    workdir = _resolve_managed_path(default_root, node.get("workdir") or default_root)
    if not script_path.exists():
        return {"ok": False, "message": f"{name} script missing", "script": str(script_path)}

    log_dir = LAUNCHER_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.out.log"
    stderr_path = log_dir / f"{name}.err.log"

    env = os.environ.copy()
    env.update(_shared_runtime_env())
    for key, value in (node.get("env") or {}).items():
        env[str(key)] = str(value)

    args = [str(script_path), *[str(item) for item in (node.get("args") or [])]]
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open("a", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            [_node_python_executable(node), *args],
            cwd=str(workdir),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=env,
            **_windows_process_kwargs(detached=True),
        )
    _record_pid(name, process.pid)
    return {
        "ok": True,
        "pid": process.pid,
        "script": str(script_path),
        "workdir": str(workdir),
    }


def _ensure_service_ready() -> dict[str, Any]:
    if _service_health_ok(timeout=2.0):
        return {"ok": True, "healthy": True, "started": False}

    runtime = _load_runtime_config()
    service = _service_config()
    projects = runtime.get("projects") or {}
    default_root = Path(str(projects.get("sts2_cli_root") or ROOT))
    service_pids = _find_service_process_ids()
    terminated_pids: list[int] = []
    if service_pids:
        terminated_pids = _terminate_processes(service_pids)
        _wait_for_process_exit(_find_service_process_ids, timeout_seconds=5.0, poll_interval_seconds=0.5)

    start_result = _start_managed_process("service", service, default_root=default_root)
    if not start_result.get("ok"):
        return {
            "ok": False,
            "message": str(start_result.get("message") or "failed to start service"),
            "terminated_pids": terminated_pids,
            "start_result": start_result,
        }

    deadline = time.time() + 30.0
    while time.time() < deadline:
        if _service_health_ok(timeout=2.0):
            return {
                "ok": True,
                "healthy": True,
                "started": True,
                "pid": start_result.get("pid"),
                "terminated_pids": terminated_pids,
            }
        time.sleep(1.0)

    return {
        "ok": False,
        "message": "service did not become healthy after start",
        "pid": start_result.get("pid"),
        "terminated_pids": terminated_pids,
    }


def _ensure_generic_process_ready(name: str, node: dict[str, Any], *, default_root: Path) -> dict[str, Any]:
    if not bool(node.get("enabled", False)):
        return {"ok": True, "enabled": False, "started": False}
    existing_pids = _managed_process_ids(name)
    if existing_pids:
        if len(existing_pids) == 1:
            return {"ok": True, "enabled": True, "started": False, "pids": existing_pids}
        terminated = _terminate_processes(existing_pids)
        remaining = _wait_for_process_exit(
            lambda name=name: _managed_process_ids(name),
            timeout_seconds=6.0,
            poll_interval_seconds=0.4,
        )
        if remaining:
            return {
                "ok": False,
                "enabled": True,
                "started": False,
                "message": f"duplicate {name} processes could not be cleaned",
                "pids": remaining,
                "terminated_pids": terminated,
            }
        _clear_pid(name)
    start_result = _start_managed_process(name, node, default_root=default_root)
    if not start_result.get("ok"):
        return {
            "ok": False,
            "enabled": True,
            "started": False,
            "message": str(start_result.get("message") or f"failed to start {name}"),
            "start_result": start_result,
        }
    time.sleep(1.0)
    current_pids = _managed_process_ids(name)
    if current_pids:
        return {"ok": True, "enabled": True, "started": True, "pids": current_pids}
    return {
        "ok": False,
        "enabled": True,
        "started": False,
        "message": f"{name} did not stay running after start",
        "start_result": start_result,
    }


def _ensure_support_processes_ready() -> dict[str, Any]:
    service_result = _ensure_service_ready()
    results: dict[str, Any] = {"service": service_result}
    if not service_result.get("ok"):
        return {"ok": False, "nodes": results}

    watchdog_result = _ensure_generic_process_ready("watchdog", _watchdog_config(), default_root=ROOT)
    results["watchdog"] = watchdog_result

    supervisor_result = _ensure_generic_process_ready(
        "session_supervisor",
        _session_supervisor_config(),
        default_root=ROOT,
    )
    results["session_supervisor"] = supervisor_result
    warnings: list[str] = []
    if not watchdog_result.get("ok"):
        warnings.append(str(watchdog_result.get("message") or "watchdog unavailable"))
    if not supervisor_result.get("ok"):
        warnings.append(str(supervisor_result.get("message") or "session supervisor unavailable"))
    return {"ok": True, "nodes": results, "warnings": warnings}


def _stop_support_stack() -> dict[str, Any]:
    results: dict[str, Any] = {}
    cleanup_result: dict[str, Any] | None = None
    if _service_health_ok(timeout=2.0):
        cleanup_result = _cleanup_and_shutdown_service_workers()
    results["service_cleanup"] = cleanup_result

    for name in _support_node_names(include_disabled=True):
        pids = _find_service_process_ids() if name == "service" else _managed_process_ids(name)
        terminated = _terminate_processes(pids) if pids else []
        remaining = _wait_for_process_exit(
            (lambda: _find_service_process_ids()) if name == "service" else (lambda name=name: _managed_process_ids(name)),
            timeout_seconds=6.0,
            poll_interval_seconds=0.4,
        ) if terminated else []
        if not remaining:
            _clear_pid(name)
        results[name] = {
            "requested_pids": pids,
            "terminated_pids": terminated,
            "remaining_pids": remaining,
        }
    return results


def _latest_training_status() -> str:
    run_dir = _latest_run_dir()
    if run_dir is None:
        return "idle"
    training = _training_payload(run_dir)
    return str(training.get("status") or "idle").lower()


def _launcher_log_paths() -> dict[str, Path]:
    log_dir = LAUNCHER_DIR
    return {
        "client_stdout": log_dir / "client.out.log",
        "client_stderr": log_dir / "client.err.log",
        "dashboard_stdout": log_dir / "dashboard.out.log",
        "dashboard_stderr": log_dir / "dashboard.err.log",
        "watchdog_stdout": log_dir / "watchdog.out.log",
        "watchdog_stderr": log_dir / "watchdog.err.log",
        "session_supervisor_stdout": log_dir / "session_supervisor.out.log",
        "session_supervisor_stderr": log_dir / "session_supervisor.err.log",
    }


def _collect_launcher_logs(max_lines: int = 30) -> dict[str, Any]:
    _ = max_lines
    return deepcopy(TELEMETRY_STATE.get("launcher_logs") or {})


def _process_resource_stats(pid: int, windows_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if pid <= 0:
        return {}
    if psutil is not None:
        try:
            process = psutil.Process(pid)
            mem = process.memory_info()
            private_value = getattr(mem, "private", None)
            if private_value is None:
                private_value = getattr(mem, "vms", 0)
            return {
                "pid": pid,
                "name": process.name(),
                "working_set_mb": round(float(getattr(mem, "rss", 0)) / (1024 * 1024), 1),
                "private_mb": round(float(private_value) / (1024 * 1024), 1),
                "threads": int(process.num_threads()),
            }
        except Exception:
            pass
    rows = windows_rows if isinstance(windows_rows, list) else _windows_process_rows()
    for row in rows:
        row_pid = _safe_int(row.get("ProcessId") or row.get("pid"), 0)
        if row_pid != pid:
            continue
        working_set = float(_safe_int(row.get("WorkingSetSize"), 0)) / (1024 * 1024)
        private_mb = float(_safe_int(row.get("PageFileUsage"), 0)) / 1024.0
        return {
            "pid": pid,
            "name": str(row.get("Name") or row.get("name") or ""),
            "working_set_mb": round(working_set, 1),
            "private_mb": round(private_mb, 1),
            "threads": _safe_int(row.get("ThreadCount"), 0),
        }
    return _process_resource_stats_by_pid(pid)


def _collect_resource_summary(training: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "training": {},
        "dashboard": {},
        "service": {},
        "service_workers": {},
        "top_level_working_set_mb": 0.0,
        "top_level_private_mb": 0.0,
        "top_level_process_count": 0,
        "total_working_set_mb": 0.0,
        "total_private_mb": 0.0,
    }

    windows_rows = _windows_process_rows() if psutil is None else None

    training_pid = _safe_int(training.get("process_pid"), 0)
    if training_pid <= 0:
        training_pids = _find_training_process_ids()
        training_pid = training_pids[0] if training_pids else 0
    training_stats = _process_resource_stats(training_pid, windows_rows=windows_rows)
    if not training_stats:
        fallback_training = {
            "pid": _safe_int(training.get("process_pid"), 0),
            "working_set_mb": float(training.get("memory_working_set_mb") or 0.0),
            "private_mb": float(training.get("memory_private_mb") or 0.0),
            "threads": _safe_int(training.get("thread_count"), 0),
        }
        if fallback_training["pid"] > 0 or fallback_training["working_set_mb"] > 0:
            training_stats = fallback_training
    if training_stats:
        summary["training"] = training_stats

    dashboard_pids = _managed_process_ids("dashboard")
    dashboard_stats = _process_resource_stats(dashboard_pids[0], windows_rows=windows_rows) if dashboard_pids else {}
    if dashboard_stats:
        summary["dashboard"] = dashboard_stats

    service_pids = _find_service_process_ids()
    service_stats = _process_resource_stats(service_pids[0], windows_rows=windows_rows) if service_pids else {}
    if service_stats:
        summary["service"] = service_stats

    worker_count = 0
    worker_working_set = 0.0
    worker_private = 0.0
    if psutil and service_pids:
        try:
            service_process = psutil.Process(service_pids[0])
            for child in service_process.children(recursive=True):
                if child.name().lower() != "dotnet.exe":
                    continue
                worker_count += 1
                mem = child.memory_info()
                worker_working_set += float(getattr(mem, "rss", 0)) / (1024 * 1024)
                private_value = getattr(mem, "private", None)
                if private_value is None:
                    private_value = getattr(mem, "vms", 0)
                worker_private += float(private_value) / (1024 * 1024)
        except Exception:
            pass
    elif service_pids:
        rows = windows_rows if isinstance(windows_rows, list) else _windows_process_rows()
        by_parent: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            parent_pid = _safe_int(row.get("ParentProcessId"), 0)
            by_parent.setdefault(parent_pid, []).append(row)
        pending = [service_pids[0]]
        seen_pids: set[int] = set()
        while pending:
            parent_pid = pending.pop()
            for row in by_parent.get(parent_pid, []):
                child_pid = _safe_int(row.get("ProcessId"), 0)
                if child_pid <= 0 or child_pid in seen_pids:
                    continue
                seen_pids.add(child_pid)
                pending.append(child_pid)
                name = str(row.get("Name") or "").lower()
                if name != "dotnet.exe":
                    continue
                worker_count += 1
                worker_working_set += float(_safe_int(row.get("WorkingSetSize"), 0)) / (1024 * 1024)
                worker_private += float(_safe_int(row.get("PageFileUsage"), 0)) / 1024.0
    summary["service_workers"] = {
        "count": worker_count,
        "working_set_mb": round(worker_working_set, 1),
        "private_mb": round(worker_private, 1),
    }

    top_level_process_count = 0
    for section in ("training", "dashboard", "service"):
        stats = summary.get(section) or {}
        if stats:
            top_level_process_count += 1
        summary["total_working_set_mb"] += float(stats.get("working_set_mb") or 0.0)
        summary["total_private_mb"] += float(stats.get("private_mb") or 0.0)
    summary["top_level_working_set_mb"] = round(summary["total_working_set_mb"], 1)
    summary["top_level_private_mb"] = round(summary["total_private_mb"], 1)
    summary["top_level_process_count"] = top_level_process_count
    summary["total_working_set_mb"] = round(summary["total_working_set_mb"] + worker_working_set, 1)
    summary["total_private_mb"] = round(summary["total_private_mb"] + worker_private, 1)
    return summary


def _trend_window(rows: list[dict[str, Any]], start: int, size: int) -> list[dict[str, Any]]:
    return rows[start:start + size]


def _trend_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [_safe_float(item.get(key)) for item in rows]
    clean = [value for value in values if value is not None]
    return round(sum(clean) / len(clean), 2) if clean else 0.0


def _build_trend_summary(training: dict[str, Any], top_sessions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(item) for item in top_sessions if isinstance(item, dict)]
    rows.sort(key=lambda item: str(item.get("finished_at") or item.get("recorded_at") or ""), reverse=True)
    recent = _trend_window(rows, 0, 12)
    previous = _trend_window(rows, 12, 12)
    recent_floor = round(sum(_row_global_floor(_normalize_floor_fields(item)) for item in recent) / len(recent), 2) if recent else 0.0
    previous_floor = round(sum(_row_global_floor(_normalize_floor_fields(item)) for item in previous) / len(previous), 2) if previous else 0.0
    recent_reward = _trend_mean(recent, "episode_reward")
    previous_reward = _trend_mean(previous, "episode_reward")
    floor_delta = round(recent_floor - previous_floor, 2)
    reward_delta = round(recent_reward - previous_reward, 2)
    degrading = bool(previous and ((floor_delta <= -2.0) or (reward_delta <= -25.0)))
    recommend_discard = degrading and str(training.get("resume_load_status") or "") == "resumed"
    return {
        "recent_count": len(recent),
        "previous_count": len(previous),
        "recent_floor_mean": recent_floor,
        "previous_floor_mean": previous_floor,
        "recent_reward_mean": recent_reward,
        "previous_reward_mean": previous_reward,
        "floor_delta": floor_delta,
        "reward_delta": reward_delta,
        "degrading": degrading,
        "recommend_discard_history": recommend_discard,
        "resume_model_path": training.get("resume_model_path"),
    }


def _collect_recent_runtime_incidents(max_items: int = 12) -> list[dict[str, Any]]:
    if TELEMETRY_STATE.get("bootstrapped"):
        return deepcopy((TELEMETRY_STATE.get("recent_incidents") or [])[:max_items])
    rows: list[dict[str, Any]] = []
    sources = (
        ("watchdog", WATCHDOG_INCIDENTS_DIR),
        ("session_supervisor", SESSION_SUPERVISOR_INCIDENTS_DIR),
    )
    for source_name, directory in sources:
        if not directory.exists():
            continue
        files = sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:max_items]
        for path in files:
            payload = _read_json(path)
            if not isinstance(payload, dict):
                continue
            row = _sanitize_incident_payload(payload)
            row["source"] = source_name
            row["path"] = str(path)
            row["file_updated_at"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
            rows.append(row)
    rows.sort(key=lambda item: str(item.get("recorded_at") or item.get("file_updated_at") or ""), reverse=True)
    return rows[:max_items]


def _run_start_iso(run: dict[str, Any]) -> str:
    training = run.get("training") or {}
    run_id = str(training.get("run_id") or run.get("run_id") or "")
    if run_id:
        try:
            return datetime.strptime(run_id, "%Y%m%d_%H%M%S").isoformat(timespec="seconds")
        except ValueError:
            pass
    return str(training.get("updated_at") or run.get("updated_at") or "")


def _filter_incidents_for_run(rows: list[dict[str, Any]], run: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(run, dict):
        return [dict(item) for item in rows if isinstance(item, dict)]
    training = run.get("training") or {}
    run_id = str(training.get("run_id") or run.get("run_id") or "")
    run_start_iso = _run_start_iso(run)
    filtered: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        incident_run_id = str(item.get("run_id") or "")
        recorded_at = str(item.get("recorded_at") or item.get("file_updated_at") or "")
        if run_id and incident_run_id and incident_run_id != run_id:
            continue
        if run_start_iso and recorded_at and recorded_at < run_start_iso:
            continue
        filtered.append(dict(item))
    return filtered


def _incident_time(value: Any) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(str(value)).isoformat(timespec="seconds")
    except ValueError:
        return str(value)


def _build_problem_list(
    recent_incidents: list[dict[str, Any]],
    suspicious_completed: list[dict[str, Any]],
    active_game_ids: set[str] | None = None,
    runtime_healthy: bool = False,
) -> list[dict[str, Any]]:
    problems: dict[str, dict[str, Any]] = {}
    active_game_ids = active_game_ids or set()

    for row in recent_incidents:
        seed = str(row.get("seed") or "")
        game_id = str(row.get("game_id") or "")
        key = game_id or seed or str(row.get("path") or "")
        if not key:
            continue
        discovered_at = _incident_time(row.get("recorded_at") or row.get("file_updated_at"))
        resolved_explicit = bool(row.get("resolution_status") or row.get("resolution_action"))
        resolved_inferred = False
        incident_type = str(row.get("type") or "")
        if not resolved_explicit:
            if game_id and game_id not in active_game_ids:
                resolved_inferred = True
            elif not game_id and runtime_healthy and incident_type in {
                "dashboard_unavailable",
                "service_unavailable",
                "service_unhealthy",
                "training_client_missing",
            }:
                resolved_inferred = True
        resolved = resolved_explicit or resolved_inferred
        resolved_at = discovered_at if resolved_explicit else ""
        status = "resolved" if resolved_explicit else ("inferred_resolved" if resolved_inferred else "open")
        reason = str(row.get("reason") or row.get("flags_display") or row.get("type") or "")
        current = problems.get(key) or {
            "key": key,
            "seed": seed,
            "game_id": game_id,
            "run_id": str(row.get("run_id") or ""),
            "status": status,
            "discovered_at": discovered_at,
            "resolved_at": resolved_at,
            "reason": reason,
            "sources": [],
        }
        current["seed"] = current.get("seed") or seed
        current["game_id"] = current.get("game_id") or game_id
        current["run_id"] = current.get("run_id") or str(row.get("run_id") or "")
        current["reason"] = current.get("reason") or reason
        if discovered_at and (not current.get("discovered_at") or discovered_at < str(current.get("discovered_at"))):
            current["discovered_at"] = discovered_at
        if resolved_at and (not current.get("resolved_at") or resolved_at > str(current.get("resolved_at"))):
            current["resolved_at"] = resolved_at
        if resolved_explicit:
            current["status"] = "resolved"
        elif resolved_inferred and current.get("status") != "resolved":
            current["status"] = "inferred_resolved"
        current.setdefault("sources", []).append(
            {
                "source": str(row.get("source") or ""),
                "type": incident_type,
                "reason": reason,
                "recorded_at": discovered_at,
                "resolution_status": str(row.get("resolution_status") or row.get("resolution_action") or ("inferred" if resolved_inferred else "")),
            }
        )
        problems[key] = current

    for row in suspicious_completed:
        seed = str(row.get("seed") or "")
        game_id = str(row.get("game_id") or "")
        key = game_id or seed
        if not key or key in problems:
            continue
        discovered_at = _incident_time(row.get("finished_at") or row.get("updated_at") or row.get("started_at"))
        problems[key] = {
            "key": key,
            "seed": seed,
            "game_id": game_id,
            "run_id": str(row.get("run_id") or ""),
            "status": "needs_review",
            "discovered_at": discovered_at,
            "resolved_at": "",
            "reason": str(row.get("flags_display") or row.get("termination_reason") or "suspicious_completed"),
            "sources": [
                {
                    "source": "dashboard",
                    "type": "suspicious_completed",
                    "reason": str(row.get("flags_display") or ""),
                    "recorded_at": discovered_at,
                    "resolution_status": "",
                }
            ],
        }

    payload = list(problems.values())
    payload.sort(key=lambda item: str(item.get("discovered_at") or ""), reverse=True)
    return payload[:20]


def _parse_total_timesteps(raw: Any) -> int | None:
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _parse_num_envs(raw: Any) -> int | None:
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= 64 else None


def _parse_vec_env(raw: Any) -> str | None:
    if raw in (None, ""):
        return None
    value = str(raw).strip().lower()
    return value if value in {"dummy", "subproc", "threaded"} else None


def _start_training_process(
    total_timesteps: int | None = None,
    num_envs: int | None = None,
    vec_env: str | None = None,
) -> dict[str, Any]:
    defaults = _load_training_defaults()
    if total_timesteps is None:
        total_timesteps = _safe_int(defaults.get("total_timesteps"), 1_000_000)
    if num_envs is None:
        num_envs = _safe_int(defaults.get("num_envs"), 20)
    if not vec_env:
        vec_env = str(defaults.get("vec_env") or "threaded")
    active_run_dir = _active_run_dir()
    existing_pids = _find_training_process_ids()
    live_service_games = _service_game_rows()
    if active_run_dir is not None and existing_pids:
        return {"ok": False, "message": "training already running", "run_dir": str(active_run_dir), "pids": existing_pids}
    if active_run_dir is not None and not existing_pids:
        _force_training_status(active_run_dir, "stopped")
        _telemetry_mark_run_idle(run_dir=str(active_run_dir), clear_slots=True)
        _set_runtime_control(active_run_dir.name, None)
    if existing_pids:
        status = _latest_training_status()
        if status in {"finished", "stopped", "idle", "error", ""}:
            terminated = _terminate_training_processes(existing_pids)
            remaining = _wait_for_training_exit(timeout_seconds=3.0, poll_interval_seconds=0.3)
            if remaining:
                return {
                    "ok": False,
                    "message": "stale training processes detected and could not be cleaned",
                    "pids": remaining,
                }
        else:
            return {"ok": False, "message": "training already running", "pids": existing_pids, "status": status}

    support_result = _ensure_support_processes_ready()
    if not support_result.get("ok"):
        return {
            "ok": False,
            "error_code": "runtime_unavailable",
            "message": "runtime support processes unavailable",
            "runtime": support_result,
            "launcher_logs": _collect_launcher_logs(),
        }

    cleanup_result = _cleanup_service_games()
    hard_cleanup_result: dict[str, Any] | None = None
    health_after_cleanup = _service_request("/health", timeout=5.0) if _service_health_ok(timeout=2.0) else {}
    if (
        isinstance(health_after_cleanup, dict)
        and _int_or_default(health_after_cleanup.get("active_games"), 0) == 0
        and _int_or_default(health_after_cleanup.get("busy_workers"), 0) > 0
    ):
        hard_cleanup_result = _cleanup_and_shutdown_service_workers()
        support_result = _ensure_support_processes_ready()
        if not support_result.get("ok"):
            return {
                "ok": False,
                "error_code": "runtime_unavailable",
                "message": "runtime support processes unavailable after worker cleanup",
                "runtime": support_result,
                "service_cleanup": cleanup_result,
                "hard_service_cleanup": hard_cleanup_result,
                "launcher_logs": _collect_launcher_logs(),
            }

    client = _client_config()
    workdir = Path(str(client.get("workdir") or ROOT))
    script = Path(str(client.get("script") or ROOT / "scripts" / "train_http_cli_rl.py"))
    if not script.is_absolute():
        script = (ROOT / script).resolve()
    args = [str(script), *[str(item) for item in (client.get("args") or [])]]
    if total_timesteps is not None:
        args.extend(["--timesteps", str(int(total_timesteps))])
    if num_envs is not None:
        args.extend(["--num-envs", str(int(num_envs))])
    if vec_env:
        args.extend(["--vec-env", vec_env])
    env = os.environ.copy()
    for key, value in (client.get("env") or {}).items():
        env[str(key)] = str(value)

    log_dir = LAUNCHER_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "client.out.log"
    stderr_path = log_dir / "client.err.log"
    _set_runtime_control(None, None)
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open("a", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            [_node_python_executable(client), *args],
            cwd=str(workdir),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=env,
            **_windows_process_kwargs(detached=True),
        )
    _record_pid("client", process.pid)
    deadline = time.time() + 4.0
    while time.time() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            _clear_pid("client")
            return {
                "ok": False,
                "error_code": "process_exited",
                "message": "training process exited immediately",
                "pid": process.pid,
                "exit_code": exit_code,
                "runtime": support_result,
                "service_cleanup": cleanup_result,
                "hard_service_cleanup": hard_cleanup_result,
                "launcher_logs": _collect_launcher_logs(),
            }
        if _active_run_dir() is not None or process.pid in _find_training_process_ids():
            return {
                "ok": True,
                "pid": process.pid,
                "runtime": support_result,
                "service_cleanup": cleanup_result,
                "hard_service_cleanup": hard_cleanup_result,
            }
        time.sleep(0.2)
    return {
        "ok": True,
        "pid": process.pid,
        "message": "training process started; waiting for first status update",
        "runtime": support_result,
        "service_cleanup": cleanup_result,
        "hard_service_cleanup": hard_cleanup_result,
    }


def _stop_training_process() -> dict[str, Any]:
    run_dir = _active_run_dir()
    existing_pids = _find_training_process_ids()
    live_service_games = _service_game_rows()
    support_pids = {name: _managed_process_ids(name) for name in _support_node_names(include_disabled=True)}
    if run_dir is None and not existing_pids and not live_service_games and not any(support_pids.values()):
        return {"ok": False, "message": "no active training"}

    if run_dir is not None:
        _set_runtime_control(run_dir.name, "stop")
    else:
        latest_run = _latest_run_dir()
        if latest_run is not None:
            _force_training_status(latest_run, "stopped")
            _set_runtime_control(latest_run.name, None)

    remaining = _wait_for_training_exit(timeout_seconds=8.0, poll_interval_seconds=0.5)
    forced = False
    terminated: list[int] = []
    terminated = _terminate_training_processes(remaining)
    remaining = _wait_for_training_exit(timeout_seconds=3.0, poll_interval_seconds=0.3)
    if terminated:
        forced = True
    if not remaining:
        _clear_pid("client")
    stack_stop = _stop_support_stack()
    idle_target = run_dir or _latest_run_dir()
    if idle_target is not None:
        _force_training_status(idle_target, "stopped")
        _telemetry_mark_run_idle(run_dir=str(idle_target), clear_slots=True)
        _set_runtime_control(idle_target.name, None)
    stop_ok = not remaining and all(not item.get("remaining_pids") for item in stack_stop.values() if isinstance(item, dict) and "remaining_pids" in item)
    return {
        "ok": stop_ok,
        "stopped": True,
        "forced": forced,
        "terminated_pids": terminated,
        "remaining_pids": remaining,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "stack_stop": stack_stop,
    }


def _seed_run_state_from_snapshot(snapshot: dict[str, Any]) -> None:
    run_dir = str(snapshot.get("run_dir") or "").strip()
    training = dict(snapshot.get("training") or {})
    run = _telemetry_get_run(
        {
            "run_id": str(training.get("run_id") or Path(run_dir).name if run_dir else ""),
            "run_dir": run_dir,
            "experiment_name": str(training.get("experiment_name") or (Path(run_dir).parent.name if run_dir else "")),
        }
    )
    if run is None:
        return
    run["training"] = training
    run["snapshot_seen_sessions"] = _safe_int(snapshot.get("seen_sessions"), 0)
    run["snapshot_completed_sessions"] = _safe_int(snapshot.get("completed_sessions"), 0)
    for row in snapshot.get("active_slots") or []:
        if not isinstance(row, dict):
            continue
        run["active_slots"][str(row.get("slot") or "")] = dict(row)
        if row.get("game_id"):
            run["seen_game_ids"].add(str(row["game_id"]))
    for slot_key, rows in (snapshot.get("slot_histories") or {}).items():
        if isinstance(rows, list):
            run["slot_histories"][str(slot_key)] = list(rows)[-50:]
            for row in rows:
                if not isinstance(row, dict):
                    continue
                game_id = str(row.get("game_id") or "")
                if game_id:
                    run["seen_game_ids"].add(game_id)
                    if not bool(row.get("active", False)):
                        run["finished_game_ids"].add(game_id)
    top_sessions = [dict(row) for row in (snapshot.get("top_sessions") or []) if isinstance(row, dict)]
    run["top_sessions"] = _telemetry_sort_top(top_sessions)
    for row in top_sessions:
        game_id = str(row.get("game_id") or "")
        if game_id:
            run["seen_game_ids"].add(game_id)
            if not bool(row.get("active", False)):
                run["finished_game_ids"].add(game_id)


def _ensure_telemetry_bootstrapped() -> None:
    with TELEMETRY_LOCK:
        if TELEMETRY_STATE.get("bootstrapped"):
            return
        cached_all_time = _read_json(ALL_TIME_SUMMARY_PATH)
        historical_best: list[dict[str, Any]] = []
        try:
            # One-time bootstrap scan: load global historical best into memory.
            loaded = _collect_historical_best()
            if isinstance(loaded, list):
                historical_best = [dict(row) for row in loaded if isinstance(row, dict)]
        except Exception:
            historical_best = []
        TELEMETRY_STATE["historical_best"] = historical_best
        TELEMETRY_STATE["all_time_summary"] = cached_all_time if isinstance(cached_all_time, dict) and cached_all_time else {
            "updated_at": _now_iso(),
            "run_count": 0,
            "runs_seen": 0,
            "runs_finished": 0,
            "wins": 0,
            "best_floor": 0,
            "avg_reward_finished": None,
        }
        TELEMETRY_STATE["watchdog_status"] = {}
        TELEMETRY_STATE["session_supervisor_status"] = {}
        TELEMETRY_STATE["recent_incidents"] = []
        TELEMETRY_STATE["launcher_logs"] = {}
        current_run = _telemetry_current_run(require_active=False)
        TELEMETRY_STATE["current_run_id"] = current_run.get("run_id") if current_run is not None else None
        TELEMETRY_STATE["bootstrapped"] = True


def _memory_launcher_logs() -> dict[str, Any]:
    return deepcopy(TELEMETRY_STATE.get("launcher_logs") or {})


def _telemetry_ingest_training(payload: dict[str, Any]) -> None:
    _ensure_telemetry_bootstrapped()
    with TELEMETRY_LOCK:
        run = _telemetry_get_run(payload)
        if run is None:
            return
        run["training"] = dict(payload)
        TELEMETRY_STATE["current_run_id"] = run.get("run_id")
    _invalidate_snapshot_cache()


def _telemetry_ingest_slot_current(payload: dict[str, Any]) -> None:
    _ensure_telemetry_bootstrapped()
    payload = _normalize_floor_fields(payload)
    with TELEMETRY_LOCK:
        run = _telemetry_get_run(payload)
        if run is None:
            return
        slot_key = str(payload.get("slot") or "")
        run["active_slots"][slot_key] = dict(payload)
        game_id = str(payload.get("game_id") or "")
        if game_id:
            run["seen_game_ids"].add(game_id)
        TELEMETRY_STATE["current_run_id"] = run.get("run_id")
    _invalidate_snapshot_cache()


def _telemetry_ingest_slot_history(payload: dict[str, Any]) -> None:
    _ensure_telemetry_bootstrapped()
    payload = _normalize_floor_fields(payload)
    with TELEMETRY_LOCK:
        run = _telemetry_get_run(payload)
        if run is None:
            return
        slot_key = str(payload.get("slot") or "")
        history = deque(run["slot_histories"].get(slot_key) or [], maxlen=50)
        history.append(dict(payload))
        run["slot_histories"][slot_key] = list(history)
        game_id = str(payload.get("game_id") or "")
        if game_id:
            run["seen_game_ids"].add(game_id)
            run["finished_game_ids"].add(game_id)
        run["active_slots"][slot_key] = dict(payload)
        _telemetry_update_all_time(payload, run)
    _invalidate_snapshot_cache()


def _telemetry_ingest_session(payload: dict[str, Any]) -> None:
    _ensure_telemetry_bootstrapped()
    with TELEMETRY_LOCK:
        run = _telemetry_get_run(payload)
        if run is None:
            return
        summary = _normalize_floor_fields(dict(payload.get("summary") or {}))
        details = _compact_session_details_for_memory(dict(payload.get("details") or {}))
        leaderboard = dict(payload.get("leaderboard") or {})
        game_id = str(summary.get("game_id") or "")
        if game_id:
            run["seen_game_ids"].add(game_id)
            run["finished_game_ids"].add(game_id)
            run["session_details"][game_id] = details
        if leaderboard.get("ranked"):
            summary["details_available"] = True
        top_rows = [row for row in (run.get("top_sessions") or []) if row.get("game_id") != game_id]
        top_rows.append(summary)
        run["top_sessions"] = _telemetry_sort_top(top_rows)
        _prune_session_details(run)
        _telemetry_update_historical_best(summary, run)
        _telemetry_update_all_time(summary, run)
        TELEMETRY_STATE["current_run_id"] = run.get("run_id")
    _invalidate_snapshot_cache()


def _telemetry_ingest_runtime_incident(payload: dict[str, Any], source: str) -> None:
    _ensure_telemetry_bootstrapped()
    with TELEMETRY_LOCK:
        row = _sanitize_incident_payload(payload)
        row["source"] = source
        row["file_updated_at"] = row.get("recorded_at") or _now_iso()
        recent = [item for item in (TELEMETRY_STATE.get("recent_incidents") or []) if item.get("path") != row.get("path")]
        recent.insert(0, row)
        TELEMETRY_STATE["recent_incidents"] = recent[:20]
    _invalidate_snapshot_cache()


def _telemetry_ingest_runtime_status(payload: dict[str, Any], source: str) -> None:
    _ensure_telemetry_bootstrapped()
    with TELEMETRY_LOCK:
        if source == "watchdog":
            TELEMETRY_STATE["watchdog_status"] = dict(payload)
        elif source == "session_supervisor":
            TELEMETRY_STATE["session_supervisor_status"] = dict(payload)
    _invalidate_snapshot_cache()


def _telemetry_ingest_runtime_log(payload: dict[str, Any], source: str) -> None:
    _ensure_telemetry_bootstrapped()
    with TELEMETRY_LOCK:
        logs = dict(TELEMETRY_STATE.get("launcher_logs") or {})
        channel = str(payload.get("channel") or source or "runtime")
        logs[channel] = {
            "source": source,
            "updated_at": str(payload.get("updated_at") or _now_iso()),
            "tail": str(payload.get("tail") or ""),
            "path": str(payload.get("path") or ""),
        }
        TELEMETRY_STATE["launcher_logs"] = logs
    _invalidate_snapshot_cache()


def _build_snapshot_payload() -> dict[str, Any]:
    _ensure_telemetry_bootstrapped()
    with TELEMETRY_LOCK:
        memory_run = _telemetry_current_run(require_active=False)
        all_time_summary = deepcopy(TELEMETRY_STATE.get("all_time_summary") or {})
        historical_best = deepcopy(TELEMETRY_STATE.get("historical_best") or [])
        launcher_logs = _memory_launcher_logs()
        recent_incidents = deepcopy(TELEMETRY_STATE.get("recent_incidents") or [])
        watchdog_status = deepcopy(TELEMETRY_STATE.get("watchdog_status") or {})
        session_supervisor_status = deepcopy(TELEMETRY_STATE.get("session_supervisor_status") or {})
    if memory_run is None:
        latest_run = _latest_run_dir()
        if latest_run is not None:
            snapshot = _collect_run_snapshot(latest_run)
            _seed_run_state_from_snapshot(snapshot)
            with TELEMETRY_LOCK:
                memory_run = _telemetry_current_run(require_active=False)
            if memory_run is not None:
                snapshot = _telemetry_snapshot_from_memory(memory_run)
                training_status = str((snapshot.get("training") or {}).get("status") or "").lower()
                if training_status not in {"running", "paused", "stopping"}:
                    snapshot["training"] = dict(snapshot.get("training") or {})
                    snapshot["training"]["status"] = "idle"
                    snapshot["active_slots"] = []
                snapshot["all_time_summary"] = all_time_summary
                snapshot["historical_best"] = historical_best
                snapshot["launcher_logs"] = launcher_logs
                return snapshot
        problem_list = _build_problem_list(recent_incidents, [], active_game_ids=set(), runtime_healthy=False)
        defaults = _load_training_defaults()
        return {
            "updated_at": None,
            "run_dir": None,
            "training": {"status": "idle"},
            "active_slots": [],
            "top_sessions": [],
            "seen_sessions": 0,
            "completed_sessions": 0,
            "defaults": defaults,
            "monitoring": {
                "resume_model_path": None,
                "resume_load_status": None,
                "resume_failure_reason": None,
                "overlong_active_count": 0,
                "overlong_active": [],
                "suspicious_completed_count": 0,
                "suspicious_completed": [],
                "recent_incidents_count": len(recent_incidents),
                "recent_incidents": recent_incidents,
                "problem_list": problem_list,
                "resources": {},
                "trend": {},
                "watchdog_status": watchdog_status,
                "session_supervisor_status": session_supervisor_status,
            },
            "all_time_summary": all_time_summary,
            "historical_best": historical_best,
            "launcher_logs": launcher_logs,
        }

    live_training_pids = _find_training_process_ids()
    live_service_games = _service_game_rows()
    memory_training = memory_run.get("training") or {}
    memory_status = str(memory_training.get("status") or "").lower()
    memory_updated_age = _seconds_since(memory_training.get("updated_at"))
    if (
        memory_status in {"running", "paused", "stopping"}
        and not live_training_pids
        and not live_service_games
        and memory_updated_age is not None
        and memory_updated_age >= 10
    ):
        _telemetry_mark_run_idle(
            run_id=str(memory_run.get("run_id") or ""),
            run_dir=str(memory_run.get("run_dir") or ""),
            clear_slots=True,
        )
        with TELEMETRY_LOCK:
            memory_run = _telemetry_current_run(require_active=False)

    snapshot = _telemetry_snapshot_from_memory(memory_run)
    snapshot_training = dict(snapshot.get("training") or {})
    snapshot_status = str(snapshot_training.get("status") or "").lower()
    snapshot_updated_age = _seconds_since(snapshot_training.get("updated_at"))
    if (
        snapshot_status in {"running", "paused", "stopping"}
        and not live_training_pids
        and not live_service_games
        and snapshot_updated_age is not None
        and snapshot_updated_age >= 10
    ):
        snapshot_training["status"] = "idle"
        snapshot_training["process_pid"] = 0
        snapshot_training["process_count"] = 0
        snapshot_training["thread_count"] = 0
        snapshot_training["fps"] = 0.0
        snapshot["training"] = snapshot_training
        snapshot["active_slots"] = []

    training_status = str((snapshot.get("training") or {}).get("status") or "").lower()
    if training_status not in {"running", "paused", "stopping"}:
        snapshot["training"] = dict(snapshot.get("training") or {})
        snapshot["training"]["status"] = "idle"
        snapshot["training"]["process_pid"] = 0
        snapshot["training"]["process_count"] = 0
        snapshot["training"]["thread_count"] = 0
        snapshot["training"]["fps"] = 0.0
        snapshot["active_slots"] = []
    snapshot["all_time_summary"] = all_time_summary
    snapshot["historical_best"] = historical_best
    snapshot["launcher_logs"] = launcher_logs
    return snapshot


def _refresh_snapshot_cache_once() -> dict[str, Any]:
    payload = _build_snapshot_payload()
    with SNAPSHOT_CACHE_LOCK:
        SNAPSHOT_CACHE["generated_at"] = time.time()
        SNAPSHOT_CACHE["payload"] = deepcopy(payload)
        SNAPSHOT_CACHE["last_error"] = None
        SNAPSHOT_CACHE["last_error_at"] = None
    return payload


def _snapshot_cache_payload() -> dict[str, Any] | None:
    with SNAPSHOT_CACHE_LOCK:
        payload = SNAPSHOT_CACHE.get("payload")
        return deepcopy(payload) if isinstance(payload, dict) else None


def _snapshot_cache_is_stale(max_age_seconds: float | None = None) -> bool:
    age_limit = SNAPSHOT_CACHE_TTL_SECONDS if max_age_seconds is None else float(max_age_seconds)
    with SNAPSHOT_CACHE_LOCK:
        cached_at = float(SNAPSHOT_CACHE.get("generated_at") or 0.0)
    if cached_at <= 0.0:
        return True
    return (time.time() - cached_at) >= max(0.0, age_limit)


def _snapshot_worker_loop() -> None:
    while True:
        try:
            _refresh_snapshot_cache_once()
        except Exception as exc:
            with SNAPSHOT_CACHE_LOCK:
                SNAPSHOT_CACHE["last_error"] = repr(exc)
                SNAPSHOT_CACHE["last_error_at"] = _now_iso()
        time.sleep(SNAPSHOT_WORKER_POLL_SECONDS)


def _ensure_snapshot_worker_started() -> None:
    global SNAPSHOT_WORKER_THREAD, SNAPSHOT_WORKER_STARTED
    with SNAPSHOT_CACHE_LOCK:
        if SNAPSHOT_WORKER_STARTED and SNAPSHOT_WORKER_THREAD is not None and SNAPSHOT_WORKER_THREAD.is_alive():
            return
        SNAPSHOT_WORKER_THREAD = threading.Thread(
            target=_snapshot_worker_loop,
            name="dashboard-snapshot-worker",
            daemon=True,
        )
        SNAPSHOT_WORKER_THREAD.start()
        SNAPSHOT_WORKER_STARTED = True


def collect_snapshot() -> dict[str, Any]:
    now = time.time()
    with SNAPSHOT_CACHE_LOCK:
        cached_payload = SNAPSHOT_CACHE.get("payload")
        cached_at = float(SNAPSHOT_CACHE.get("generated_at") or 0.0)
        if isinstance(cached_payload, dict) and (now - cached_at) < SNAPSHOT_CACHE_TTL_SECONDS:
            return deepcopy(cached_payload)

    return _refresh_snapshot_cache_once()


def _load_dashboard_html() -> str:
    return (ROOT / "scripts" / "training_dashboard_frontend.html").read_text(encoding="utf-8")


HTML = _load_dashboard_html()


@app.route("/")
def index() -> Response:
    return Response(HTML, mimetype="text/html")


@app.route("/favicon.ico")
def favicon() -> Response:
    for path in FAVICON_CANDIDATES:
        if path.exists():
            return send_file(path, mimetype="image/x-icon")
    return Response(status=404)


@app.route("/api/state")
def api_state() -> Response:
    _ensure_snapshot_worker_started()
    payload = _snapshot_cache_payload()
    if payload is None or _snapshot_cache_is_stale(max_age_seconds=SNAPSHOT_WORKER_POLL_SECONDS * 2.5):
        payload = collect_snapshot()
    return jsonify(payload)


@app.route("/api/health")
def api_health() -> Response:
    _ensure_snapshot_worker_started()
    payload = _snapshot_cache_payload()
    if payload is None:
        payload = {}
    training = payload.get("training") if isinstance(payload, dict) else {}
    return jsonify(
        {
            "status": "ok",
            "updated_at": payload.get("updated_at") if isinstance(payload, dict) else None,
            "run_id": training.get("run_id") if isinstance(training, dict) else None,
            "training_status": training.get("status") if isinstance(training, dict) else None,
            "active_slots": len(payload.get("active_slots") or []) if isinstance(payload, dict) else 0,
        }
    )


@app.route("/api/runtime/control")
def api_runtime_control() -> Response:
    run_id = request.args.get("run_id")
    payload = _get_runtime_control(run_id)
    if not payload:
        payload = {"run_id": str(run_id or TELEMETRY_STATE.get("current_run_id") or ""), "command": None}
    return jsonify(payload)


@app.route("/api/session/<game_id>")
def api_session(game_id: str) -> Response:
    _ensure_telemetry_bootstrapped()
    with TELEMETRY_LOCK:
        for run in (TELEMETRY_STATE.get("runs") or {}).values():
            details = (run.get("session_details") or {}).get(game_id)
            if isinstance(details, dict) and details:
                return jsonify(details)
    session = _build_service_only_session(game_id)
    if not session:
        return jsonify({"error": "session not found"}), 404
    return jsonify(session)


@app.route("/api/telemetry/training", methods=["POST"])
def api_telemetry_training() -> Response:
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "message": "invalid payload"}), 400
    _telemetry_ingest_training(payload)
    return jsonify({"ok": True})


@app.route("/api/telemetry/slot/current", methods=["POST"])
def api_telemetry_slot_current() -> Response:
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "message": "invalid payload"}), 400
    _telemetry_ingest_slot_current(payload)
    return jsonify({"ok": True})


@app.route("/api/telemetry/slot/history", methods=["POST"])
def api_telemetry_slot_history() -> Response:
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "message": "invalid payload"}), 400
    _telemetry_ingest_slot_history(payload)
    return jsonify({"ok": True})


@app.route("/api/telemetry/session", methods=["POST"])
def api_telemetry_session() -> Response:
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "message": "invalid payload"}), 400
    _telemetry_ingest_session(payload)
    return jsonify({"ok": True})


@app.route("/api/telemetry/runtime/incident", methods=["POST"])
def api_telemetry_runtime_incident() -> Response:
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "message": "invalid payload"}), 400
    source = str(payload.get("source") or "runtime")
    _telemetry_ingest_runtime_incident(payload, source)
    return jsonify({"ok": True})


@app.route("/api/telemetry/runtime/status", methods=["POST"])
def api_telemetry_runtime_status() -> Response:
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "message": "invalid payload"}), 400
    source = str(payload.get("source") or "runtime")
    _telemetry_ingest_runtime_status(payload, source)
    return jsonify({"ok": True})


@app.route("/api/telemetry/runtime/log", methods=["POST"])
def api_telemetry_runtime_log() -> Response:
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "message": "invalid payload"}), 400
    source = str(payload.get("source") or "runtime")
    _telemetry_ingest_runtime_log(payload, source)
    return jsonify({"ok": True})


@app.route("/api/control/start", methods=["POST"])
def api_control_start() -> Response:
    payload = request.get_json(silent=True) or {}
    total_timesteps = _parse_total_timesteps(payload.get("total_timesteps"))
    num_envs = _parse_num_envs(payload.get("num_envs"))
    vec_env = _parse_vec_env(payload.get("vec_env"))
    if payload.get("total_timesteps") not in (None, "") and total_timesteps is None:
        return jsonify({"ok": False, "message": "invalid total_timesteps"}), 400
    if payload.get("num_envs") not in (None, "") and num_envs is None:
        return jsonify({"ok": False, "message": "invalid num_envs"}), 400
    if payload.get("vec_env") not in (None, "") and vec_env is None:
        return jsonify({"ok": False, "message": "invalid vec_env"}), 400
    result = _start_training_process(total_timesteps=total_timesteps, num_envs=num_envs, vec_env=vec_env)
    if result.get("ok"):
        status = 200
    elif result.get("error_code") == "process_exited":
        status = 500
    else:
        status = 409
    return jsonify(result), status


@app.route("/api/control/pause", methods=["POST"])
def api_control_pause() -> Response:
    return jsonify({"ok": False, "message": "pause removed; use stop/start"}), 410


@app.route("/api/control/resume", methods=["POST"])
def api_control_resume() -> Response:
    return jsonify({"ok": False, "message": "resume removed; use start"}), 410


@app.route("/api/control/stop", methods=["POST"])
def api_control_stop() -> Response:
    result = _stop_training_process()
    status = 200 if result.get("ok") else 409
    return jsonify(result), status


def main() -> None:
    _ensure_snapshot_worker_started()
    app.run(host="127.0.0.1", port=8787, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
