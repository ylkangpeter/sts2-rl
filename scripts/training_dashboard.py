# -*- coding: utf-8 -*-
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

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT / "models" / "http_cli_rl"
RUNTIME_CONFIG_PATHS = [
    ROOT / "configs" / "runtime_stack.json",
    ROOT / "configs" / "runtime_stack.local.json",
]
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
    merged: dict[str, Any] = {}
    for path in RUNTIME_CONFIG_PATHS:
        data = _read_json(path)
        if isinstance(data, dict):
            merged = _deep_merge_dicts(merged, data)
    return merged


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


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
            int(item.get("active_slot_count") or 0),
            float(item.get("freshest_ts") or 0.0),
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
            "max_floor": slot_row.get("floor") or summary.get("max_floor") or context.get("floor") or 0,
            "act": slot_row.get("act") or summary.get("act") or context.get("act") or 0,
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
    summary = {
        "game_id": game_id,
        "seed": (cached.get("summary") or {}).get("seed"),
        "character": (cached.get("summary") or {}).get("character") or "Ironclad",
        "slot": (cached.get("summary") or {}).get("slot"),
        "episode_index": (cached.get("summary") or {}).get("episode_index"),
        "episode_reward": (cached.get("summary") or {}).get("episode_reward", 0.0),
        "episode_steps": (cached.get("summary") or {}).get("episode_steps", 0),
        "victory": bool(raw_state.get("victory")),
        "terminated": bool(raw_state.get("game_over")),
        "truncated": False,
        "max_floor": context.get("floor") or 0,
        "act": context.get("act") or 0,
        "final_hp": state_snapshot.get("hp"),
        "max_hp": state_snapshot.get("max_hp"),
        "final_gold": state_snapshot.get("gold"),
        "started_at": (cached.get("summary") or {}).get("started_at"),
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
    if active and duration_seconds is not None and duration_seconds > 600:
        flags.append("overlong_active")
    elif not active and duration_seconds is not None and duration_seconds > 600:
        flags.append("overlong_episode")
    termination_reason = str(row.get("termination_reason") or "").strip()
    if termination_reason in {"protocol_deadlock", "stuck_abort", "step_exception", "fatal_protocol_error"}:
        flags.append(termination_reason)
    return list(dict.fromkeys(flags))


def _annotate_health(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
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
        int(item.get("max_floor", 0)),
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
        existing = deduped.get(key)
        if existing is None or _score(row) > _score(existing):
            deduped[key] = dict(row)
    return sorted(deduped.values(), key=_score, reverse=True)[:limit]


def _telemetry_update_historical_best(summary: dict[str, Any], run: dict[str, Any]) -> None:
    if not summary.get("game_id"):
        return
    entry = dict(summary)
    entry["run_id"] = run.get("run_id")
    entry["run_dir"] = run.get("run_dir")
    entry["experiment_name"] = run.get("experiment_name")
    historical = [row for row in (TELEMETRY_STATE.get("historical_best") or []) if row.get("game_id") != entry.get("game_id")]
    historical.append(entry)
    TELEMETRY_STATE["historical_best"] = _telemetry_sort_top(historical, limit=50)


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
    all_time["best_floor"] = max(_safe_int(all_time.get("best_floor"), 0), _safe_int(summary.get("max_floor"), 0))
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
    recent_incidents = deepcopy(TELEMETRY_STATE.get("recent_incidents") or [])
    active_game_ids = {str(item.get("game_id") or "") for item in active_slots if str(item.get("game_id") or "")}
    problem_list = _build_problem_list(recent_incidents, suspicious_completed, active_game_ids=active_game_ids, runtime_healthy=True)
    detailed_ids = set((run.get("session_details") or {}).keys())
    for item in active_slots:
        item["details_available"] = bool(item.get("details_available") or item.get("game_id") in detailed_ids)
    for item in top_sessions:
        item["details_available"] = bool(item.get("details_available") or item.get("game_id") in detailed_ids)
    return {
        "updated_at": (run.get("training") or {}).get("updated_at"),
        "run_dir": run.get("run_dir"),
        "training": deepcopy(run.get("training") or {}),
        "active_slots": active_slots,
        "top_sessions": top_sessions,
        "slot_histories": deepcopy(run.get("slot_histories") or {}),
        "checkpoints": [],
        "seen_sessions": len(run.get("seen_game_ids") or set()),
        "completed_sessions": len(run.get("finished_game_ids") or set()),
        "monitoring": {
            "resume_model_path": (run.get("training") or {}).get("resume_model_path"),
            "resume_load_status": (run.get("training") or {}).get("resume_load_status"),
            "resume_failure_reason": (run.get("training") or {}).get("resume_failure_reason"),
            "overlong_active_count": len(overlong_active),
            "overlong_active": overlong_active[:10],
            "suspicious_completed_count": len(suspicious_completed),
            "suspicious_completed": suspicious_completed[:10],
            "recent_incidents_count": len(recent_incidents),
            "recent_incidents": recent_incidents,
            "problem_list": problem_list,
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
            slot_history_map[slot_key] = rows[-50:]
            histories.extend(rows)

    history_by_job: dict[str, dict[str, Any]] = {}
    for row in histories:
        key = row.get("game_id") or f"{row.get('slot')}-{row.get('episode_index')}"
        existing = history_by_job.get(key, {})
        max_floor = max(int(existing.get("max_floor", 0)), int(row.get("floor", 0)))
        merged = dict(existing)
        merged.update(row)
        merged["max_floor"] = max_floor
        history_by_job[key] = merged

    for slot in active_slots:
        key = slot.get("game_id") or f"{slot.get('slot')}-{slot.get('episode_index')}"
        live_row = live_games.get(str(slot.get("game_id") or ""))
        if isinstance(live_row, dict):
            slot["uptime_seconds"] = live_row.get("uptime_seconds")
            slot["service_created_at"] = live_row.get("created_at")
            slot["service_start_time"] = live_row.get("start_time")
            slot["service_alive"] = live_row.get("alive")
        existing = history_by_job.get(key, {})
        max_floor = max(int(existing.get("max_floor", 0)), int(slot.get("floor", 0)))
        merged = dict(existing)
        merged.update(slot)
        merged["max_floor"] = max_floor
        history_by_job[key] = merged

    computed_top_sessions = sorted(history_by_job.values(), key=_score, reverse=True)[:50]
    merged_leaderboard_rows = [row for row in persisted_leaderboard if isinstance(row, dict)] + computed_top_sessions
    top_sessions = _telemetry_sort_top(merged_leaderboard_rows, limit=50)
    top_sessions = _enrich_rows_with_boss(run_dir, top_sessions)
    active_slots = _enrich_active_slots_with_live_boss(_enrich_rows_with_boss(run_dir, active_slots))
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
            _safe_int(item.get("max_floor"), 0),
            _safe_float(item.get("elapsed_seconds")) or 0.0,
        ),
        reverse=True,
    )
    suspicious_completed = [item for item in suspicious_completed if item.get("suspicious")]
    overlong_active = [item for item in active_slots if "overlong_active" in (item.get("health_flags") or [])]
    recent_incidents = _collect_recent_runtime_incidents()
    active_game_ids = {str(item.get("game_id") or "") for item in active_slots if str(item.get("game_id") or "")}
    problem_list = _build_problem_list(recent_incidents, suspicious_completed, active_game_ids=active_game_ids, runtime_healthy=True)

    return {
        "updated_at": training.get("updated_at"),
        "run_dir": str(run_dir),
        "training": training,
        "active_slots": sorted(active_slots, key=lambda item: str(item.get("slot"))),
        "top_sessions": top_sessions,
        "slot_histories": slot_history_map,
        "checkpoints": checkpoints,
        "seen_sessions": len(history_by_job),
        "completed_sessions": sum(1 for item in history_by_job.values() if not item.get("active", False)),
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
                    best_floor = max(best_floor, int(row.get("max_floor", row.get("floor", 0)) or 0))
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
                best_floor = max(best_floor, int(row.get("max_floor", 0) or 0))

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


def _control_path(run_dir: Path) -> Path:
    return run_dir / "dashboard" / "control.json"


def _write_control(run_dir: Path, command: str) -> None:
    payload = {"command": command}
    _control_path(run_dir).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _support_node_names() -> list[str]:
    names = ["service"]
    if bool(_watchdog_config().get("enabled", False)):
        names.append("watchdog")
    if bool(_session_supervisor_config().get("enabled", False)):
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
    if pid is not None:
        return [pid]
    node = _runtime_node(name)
    candidates: list[int] = []
    seen: set[int] = set()
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
    escaped = needle.replace("\\", "\\\\").replace("'", "''")
    try:
        output = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
                f"Where-Object {{ $_.CommandLine -and $_.CommandLine -match '{escaped}' }} | "
                "ForEach-Object { $_.ProcessId }",
            ],
            text=True,
            cwd=str(ROOT),
            timeout=12,
            **_windows_process_kwargs(),
        )
    except Exception:
        return []
    return [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]


def _find_training_process_ids() -> list[int]:
    pid = _pid_from_file("client")
    if pid is not None:
        return [pid]
    return _find_python_process_ids("train_http_cli_rl.py")


def _find_service_process_ids() -> list[int]:
    pid = _pid_from_file("service")
    if pid is not None:
        return [pid]
    service = _service_config()
    return _find_python_process_ids(str(service.get("process_match") or "http_game_service.py"))


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
        return {"ok": True, "enabled": True, "started": False, "pids": existing_pids}
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

    for name in _support_node_names():
        pids = _managed_process_ids(name)
        terminated = _terminate_processes(pids) if pids else []
        remaining = _wait_for_process_exit(
            lambda name=name: _managed_process_ids(name),
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
    payload: dict[str, Any] = {}
    for name, path in _launcher_log_paths().items():
        payload[name] = {
            "path": str(path),
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else None,
            "tail": _read_log_tail(path, max_lines=max_lines),
        }
    return payload


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
        resolved = bool(row.get("resolution_status") or row.get("resolution_action"))
        incident_type = str(row.get("type") or "")
        if not resolved:
            if game_id and game_id not in active_game_ids:
                resolved = True
            elif not game_id and runtime_healthy and incident_type in {
                "dashboard_unavailable",
                "service_unavailable",
                "service_unhealthy",
                "training_client_missing",
            }:
                resolved = True
        resolved_at = discovered_at if resolved else ""
        reason = str(row.get("reason") or row.get("flags_display") or row.get("type") or "")
        current = problems.get(key) or {
            "key": key,
            "seed": seed,
            "game_id": game_id,
            "run_id": str(row.get("run_id") or ""),
            "status": "resolved" if resolved else "open",
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
        if resolved:
            current["status"] = "resolved"
        current.setdefault("sources", []).append(
            {
                "source": str(row.get("source") or ""),
                    "type": incident_type,
                "reason": reason,
                "recorded_at": discovered_at,
                "resolution_status": str(row.get("resolution_status") or row.get("resolution_action") or ""),
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
            "status": "resolved",
            "discovered_at": discovered_at,
            "resolved_at": discovered_at,
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


def _start_training_process(total_timesteps: int | None = None, num_envs: int | None = None) -> dict[str, Any]:
    active_run_dir = _active_run_dir()
    existing_pids = _find_training_process_ids()
    live_service_games = _service_game_rows()
    if active_run_dir is not None and (existing_pids or live_service_games):
        return {"ok": False, "message": "training already running", "run_dir": str(active_run_dir), "pids": existing_pids}
    if active_run_dir is not None and not existing_pids and not live_service_games:
        _force_training_status(active_run_dir, "stopped")
        _telemetry_mark_run_idle(run_dir=str(active_run_dir), clear_slots=True)
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
    env = os.environ.copy()
    for key, value in (client.get("env") or {}).items():
        env[str(key)] = str(value)

    log_dir = LAUNCHER_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "client.out.log"
    stderr_path = log_dir / "client.err.log"
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
    support_pids = {name: _managed_process_ids(name) for name in _support_node_names()}
    if run_dir is None and not existing_pids and not live_service_games and not any(support_pids.values()):
        return {"ok": False, "message": "no active training"}

    if run_dir is not None:
        _write_control(run_dir, "stop")
    else:
        latest_run = _latest_run_dir()
        if latest_run is not None:
            _force_training_status(latest_run, "stopped")

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
    for row in snapshot.get("active_slots") or []:
        if not isinstance(row, dict):
            continue
        run["active_slots"][str(row.get("slot") or "")] = dict(row)
        if row.get("game_id"):
            run["seen_game_ids"].add(str(row["game_id"]))
    for slot_key, rows in (snapshot.get("slot_histories") or {}).items():
        if isinstance(rows, list):
            run["slot_histories"][str(slot_key)] = list(rows)[-50:]
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
        TELEMETRY_STATE["historical_best"] = _collect_historical_best()
        TELEMETRY_STATE["all_time_summary"] = cached_all_time if isinstance(cached_all_time, dict) and cached_all_time else {
            "updated_at": _now_iso(),
            "run_count": 0,
            "runs_seen": 0,
            "runs_finished": 0,
            "wins": 0,
            "best_floor": 0,
            "avg_reward_finished": None,
        }
        TELEMETRY_STATE["watchdog_status"] = _read_json(WATCHDOG_STATUS_PATH)
        TELEMETRY_STATE["session_supervisor_status"] = _read_json(SESSION_SUPERVISOR_STATUS_PATH)
        TELEMETRY_STATE["recent_incidents"] = _collect_recent_runtime_incidents()
        TELEMETRY_STATE["launcher_logs"] = _collect_launcher_logs(max_lines=20)
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
        summary = dict(payload.get("summary") or {})
        details = dict(payload.get("details") or {})
        leaderboard = dict(payload.get("leaderboard") or {})
        game_id = str(summary.get("game_id") or "")
        if game_id:
            run["seen_game_ids"].add(game_id)
            run["finished_game_ids"].add(game_id)
            run["session_details"][game_id] = details
            while len(run["session_details"]) > 200:
                oldest_key = next(iter(run["session_details"].keys()))
                run["session_details"].pop(oldest_key, None)
        if leaderboard.get("ranked"):
            summary["details_available"] = True
        top_rows = [row for row in (run.get("top_sessions") or []) if row.get("game_id") != game_id]
        top_rows.append(summary)
        run["top_sessions"] = _telemetry_sort_top(top_rows)
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
        problem_list = _build_problem_list(recent_incidents, [], active_game_ids=set(), runtime_healthy=False)
        return {
            "updated_at": None,
            "run_dir": None,
            "training": {"status": "idle"},
            "active_slots": [],
            "top_sessions": [],
            "seen_sessions": 0,
            "completed_sessions": 0,
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
    training_status = str((snapshot.get("training") or {}).get("status") or "").lower()
    if training_status not in {"running", "paused", "stopping"}:
        snapshot["training"] = dict(snapshot.get("training") or {})
        snapshot["training"]["status"] = "idle"
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


HTML = """<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STS2 RL Dashboard</title>
  <link rel="icon" href="/favicon.ico" sizes="any">
  <style>
    :root {
      color-scheme: dark;
      --bg:#171a1f;
      --bg-2:#1d2127;
      --panel:#23272f;
      --panel-strong:#2a2f38;
      --line:#3a414d;
      --line-soft:#303641;
      --text:#edf2f7;
      --muted:#9aa5b4;
      --accent:#7fb9ff;
      --accent-soft:rgba(127,185,255,0.16);
      --good:#6cc08b;
      --warn:#e1b15a;
      --danger:#de7373;
      --shadow:0 14px 34px rgba(0,0,0,0.32);
    }
    body { margin:0; font:14px/1.4 "Segoe UI", "PingFang SC", sans-serif; background:linear-gradient(180deg,var(--bg),var(--bg-2)); color:var(--text); }
    .wrap { max-width:1500px; margin:0 auto; padding:24px; }
    h1 { margin:0 0 16px; font-size:30px; letter-spacing:.01em; }
    h2 { margin:20px 0 10px; font-size:18px; color:var(--text); }
    .meta { color:var(--muted); margin-bottom:16px; }
    .grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:16px; }
    .card { background:linear-gradient(180deg,var(--panel-strong),var(--panel)); border:1px solid var(--line); border-radius:16px; padding:14px 16px; box-shadow:var(--shadow); }
    .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
    .value { font-size:28px; font-weight:800; margin-top:6px; color:var(--accent); }
    .sub { color:var(--muted); margin-top:4px; }
    .muted { color:var(--muted); }
    table { width:100%; border-collapse:collapse; background:linear-gradient(180deg,var(--panel-strong),var(--panel)); border:1px solid var(--line); border-radius:16px; overflow:hidden; box-shadow:var(--shadow); }
    th, td { padding:10px 12px; border-bottom:1px solid var(--line-soft); text-align:left; white-space:nowrap; }
    th { background:#262b33; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
    tr:last-child td { border-bottom:none; }
    tr:hover td { background:rgba(127,185,255,0.08); }
    .mono { font-family:Consolas, monospace; font-size:12px; }
    .metric-strong { color:var(--accent); font-weight:800; }
    .metric-good { color:var(--good); font-weight:700; }
    .metric-warn { color:var(--warn); font-weight:700; }
    .pill { display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px; background:var(--accent-soft); color:var(--accent); font-weight:700; }
    .room { font-weight:700; color:#d6e6f8; }
    .clickable-row { cursor:pointer; }
    .detail-shell { margin-top:14px; }
    .detail-grid { display:grid; grid-template-columns:1.1fr 1.5fr 1fr; gap:14px; align-items:start; }
    .summary-list { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:8px 14px; margin-top:10px; }
    .summary-item { padding:10px 12px; border:1px solid var(--line-soft); border-radius:12px; background:rgba(255,255,255,0.03); }
    .summary-item strong { display:block; font-size:18px; color:var(--accent); margin-top:3px; }
    .summary-item strong.compact-text { font-size:14px; line-height:1.35; word-break:break-all; }
    .map-card { min-height:500px; }
    .map-toolbar, .control-row { display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 12px; }
    .map-tab, .control-btn { border:1px solid var(--line); background:var(--panel-strong); color:var(--text); border-radius:999px; padding:6px 12px; cursor:pointer; }
    .map-tab.active, .control-btn.primary { background:var(--accent-soft); color:var(--accent); border-color:rgba(47,128,237,0.35); }
    .control-btn.good { background:rgba(62,143,99,0.12); color:var(--good); border-color:rgba(62,143,99,0.3); }
    .control-btn.warn { background:rgba(208,138,47,0.12); color:var(--warn); border-color:rgba(208,138,47,0.3); }
    .control-btn.danger { background:rgba(199,81,81,0.12); color:var(--danger); border-color:rgba(199,81,81,0.3); }
    .tree-wrap { overflow:auto; border:1px solid var(--line-soft); border-radius:16px; background:linear-gradient(180deg, #435979, #242930); padding:10px; }
    .node-detail-empty { padding:24px; color:var(--muted); }
    .node-button { cursor:pointer; }
    .node-button text { pointer-events:none; font-weight:700; font-size:13px; }
    .node-button.active { filter: drop-shadow(0 0 6px rgba(62,143,99,0.24)); }
    .node-stats { margin-top:10px; display:grid; gap:10px; }
    .node-block { border:1px solid var(--line-soft); border-radius:12px; padding:10px 12px; background:rgba(255,255,255,0.03); }
    .action-log { max-height:260px; overflow:auto; margin-top:8px; border:1px solid var(--line-soft); border-radius:12px; background:rgba(0,0,0,0.12); }
    .action-row { padding:8px 10px; border-bottom:1px solid var(--line-soft); }
    .action-row:last-child { border-bottom:none; }
    .reward-tag { display:inline-flex; margin:4px 6px 0 0; padding:4px 8px; border-radius:999px; background:rgba(127,185,255,0.12); color:var(--accent); }
    .slot-grid { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:12px; }
    .alerts { display:grid; gap:10px; margin-bottom:16px; }
    .alert { border:1px solid var(--line); border-radius:14px; padding:12px 14px; box-shadow:var(--shadow); }
    .alert.info { background:rgba(47,128,237,0.12); border-color:rgba(47,128,237,0.3); }
    .alert.warn { background:rgba(208,138,47,0.12); border-color:rgba(208,138,47,0.3); }
    .alert.danger { background:rgba(199,81,81,0.14); border-color:rgba(199,81,81,0.34); }
    .alert-title { font-weight:800; margin-bottom:6px; }
    .alert-body { white-space:pre-wrap; word-break:break-word; }
    .alert pre { margin:10px 0 0; padding:10px 12px; background:rgba(0,0,0,0.18); border:1px solid var(--line-soft); border-radius:12px; overflow:auto; white-space:pre-wrap; }
    .alert-tools { display:flex; justify-content:flex-end; margin-top:8px; }
    .alert-toggle { border:1px solid var(--line); background:var(--panel-strong); color:var(--text); border-radius:999px; padding:3px 10px; cursor:pointer; font-size:12px; }
    .alert-toggle:hover { border-color:var(--accent); color:var(--accent); }
    .problem-shell { margin:12px 0 18px; display:flex; flex-direction:column; gap:10px; }
    .problem-item { border:1px solid var(--line); border-radius:14px; background:var(--panel); }
    .problem-item summary { cursor:pointer; list-style:none; padding:12px 14px; display:flex; justify-content:space-between; gap:12px; align-items:center; font-weight:700; }
    .problem-item summary::-webkit-details-marker { display:none; }
    .problem-item[open] summary { border-bottom:1px solid var(--line-soft); }
    .problem-title { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
    .problem-meta { color:var(--muted); font-size:12px; white-space:nowrap; }
    .problem-body { padding:12px 14px 14px; display:grid; gap:10px; }
    .problem-grid { display:grid; grid-template-columns:160px 1fr; gap:8px 12px; }
    .problem-label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:0.08em; }
    .problem-value { color:var(--text); word-break:break-word; }
    .problem-source-list { margin:0; padding-left:18px; color:var(--muted); }
    @media (max-width: 1100px) {
      .grid { grid-template-columns:repeat(2,1fr); }
      .detail-grid { grid-template-columns:1fr; }
      .slot-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) {
      .wrap { padding:16px; }
      .grid { grid-template-columns:1fr; }
      h1 { font-size:24px; }
      .value { font-size:24px; }
      table { display:block; overflow:auto; }
      .slot-grid { grid-template-columns:1fr; }
    }
    svg polyline { stroke-linecap:round; stroke-linejoin:round; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>STS2 RL Dashboard</h1>
    <div class="meta" id="meta">loading...</div>
    <div class="alerts" id="alerts"></div>
    <div id="problemList"></div>
    <div class="control-row" id="controls"></div>
    <div class="grid" id="cards"></div>
    <h2>Active Slots</h2>
    <table id="activeTable"></table>
    <h2>Slot History</h2>
    <div id="slotCharts"></div>
    <h2>Checkpoints</h2>
    <table id="checkpointTable"></table>
    <h2>Current Run Top 50</h2>
    <table id="topTable"></table>
    <div class="detail-shell" id="sessionDetail"></div>
    <h2>Historical Best</h2>
    <table id="historyTable"></table>
  </div>
  <script>
    window.onerror = function(message, source, lineno, colno) {
      const meta = document.getElementById('meta');
      if (meta) meta.textContent = `JS error: ${message} @${lineno}:${colno}`;
    };
    window.onunhandledrejection = function(event) {
      const meta = document.getElementById('meta');
      const reason = event && event.reason ? (event.reason.message || String(event.reason)) : 'unknown rejection';
      if (meta) meta.textContent = `Promise error: ${reason}`;
    };
    function fmt(v) { return v === null || v === undefined || v === '' ? '-' : v; }
    function pct(v) { return (v ?? 0).toFixed(2) + '%'; }
    function formatDateTime(value) {
      if (!value) return '-';
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return String(value);
      const pad = (n) => String(n).padStart(2, '0');
      return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
    }
    function formatDuration(startedAt, finishedAt) {
      if (!startedAt) return '-';
      const start = new Date(startedAt);
      const end = finishedAt ? new Date(finishedAt) : new Date();
      if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return '-';
      const totalSeconds = Math.max(0, Math.round((end.getTime() - start.getTime()) / 1000));
      const minutes = Math.floor(totalSeconds / 60);
      const seconds = totalSeconds % 60;
      return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
    }
    function inferEpisodeStart(row) {
      if (row.started_at) return row.started_at;
      const candidates = [row.game_id, row.seed];
      for (const value of candidates) {
        const text = String(value || '');
        const match = text.match(/_(\d{13})(?:_|$)/) || text.match(/_(\d{10})(?:_|$)/);
        if (!match) continue;
        const raw = match[1];
        const epochMs = raw.length === 13 ? Number(raw) : Number(raw) * 1000;
        if (!Number.isFinite(epochMs)) continue;
        return new Date(epochMs).toISOString();
      }
      return null;
    }
    function withTimeFields(rows) {
      return (rows || []).map((row) => ({
        ...row,
        started_at_display: formatDateTime(row.started_at),
        duration_display: formatDuration(row.started_at, row.finished_at),
        final_hp_display: row.final_hp === null || row.final_hp === undefined
          ? '-'
          : `${row.final_hp} / ${fmt(row.max_hp)}`,
        final_gold_display: fmt(row.final_gold),
      }));
    }
    function escapeHtml(v) {
      return String(v ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
    }
    let currentSession = null;
    let currentSessionGameId = null;
    let currentAct = null;
    let currentNodeId = null;
    let currentNodeStageIndex = 0;
    let currentRunDir = null;
    let activeAlerts = [];
    let alertCounter = 0;
    let alertExpansionState = {};
    let controlsInitialized = false;
    function positiveIntegerText(value, fallbackValue) {
      const numeric = Number(value);
      if (Number.isFinite(numeric) && numeric > 0) {
        return String(Math.trunc(numeric));
      }
      return String(fallbackValue);
    }
    function kpi(value, cls='metric-strong') {
      return `<span class="${cls}">${fmt(value)}</span>`;
    }
    function pushAlert(title, body, level='danger', details='', source='request') {
      const key = JSON.stringify([title || '', body || '', level || '', details || '', source || '']);
      if (activeAlerts.some((item) => JSON.stringify([item.title || '', item.body || '', item.level || '', item.details || '', item.source || '']) === key)) {
        return;
      }
      const expanded = Boolean(alertExpansionState[key]);
      activeAlerts.push({ id: `alert-${alertCounter++}`, key, title, body, level, details, source, expanded });
      renderAlerts();
    }
    function clearAlerts() {
      activeAlerts = [];
      renderAlerts();
    }
    function clearStateAlerts() {
      activeAlerts = activeAlerts.filter((item) => item.source !== 'state');
      renderAlerts();
    }
    function renderAlerts() {
      const shell = document.getElementById('alerts');
      if (!activeAlerts.length) {
        shell.innerHTML = '';
        return;
      }
      shell.innerHTML = activeAlerts.map((item) => `
        <div class="alert ${escapeHtml(item.level || 'danger')}">
          <div class="alert-title">${escapeHtml(item.title || 'Error')}</div>
          <div class="alert-body">${escapeHtml(item.body || '')}</div>
          ${item.details ? `
            <div class="alert-tools">
              <button class="alert-toggle" type="button" data-alert-id="${escapeHtml(item.id)}">
                ${item.expanded ? 'Hide logs' : 'Show logs'}
              </button>
            </div>
            ${item.expanded ? `<pre>${escapeHtml(item.details)}</pre>` : ''}
          ` : ''}
        </div>
      `).join('');
      Array.from(document.querySelectorAll('.alert-toggle')).forEach((button) => {
        button.addEventListener('click', () => {
          const alertId = button.dataset.alertId;
          activeAlerts = activeAlerts.map((item) => {
            if (item.id !== alertId) return item;
            const expanded = !item.expanded;
            if (item.key) {
              alertExpansionState[item.key] = expanded;
            }
            return { ...item, expanded };
          });
          renderAlerts();
        });
      });
    }
    function buildErrorFromResponse(status, payload) {
      const error = new Error(payload?.message || `Request failed: HTTP ${status}`);
      error.status = status;
      error.payload = payload || {};
      return error;
    }
    async function postJson(url, payload) {
      const resp = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload || {}),
      });
      let data = {};
      try {
        data = await resp.json();
      } catch (_err) {
        data = {};
      }
      if (!resp.ok || data?.ok === false) {
        throw buildErrorFromResponse(resp.status, data);
      }
      return data;
    }
    function extractLogTail(payload) {
      return payload?.launcher_logs?.client_stderr?.tail
        || payload?.launcher_logs?.dashboard_stderr?.tail
        || payload?.stderr_tail
        || '';
    }
    function showRequestError(actionLabel, err) {
      const payload = err?.payload || {};
      const details = extractLogTail(payload);
      pushAlert(
        `${actionLabel} failed`,
        payload?.message || err?.message || 'unknown error',
        'danger',
        details,
      );
    }
    function renderControls(training) {
      const status = training?.status || 'idle';
      const running = status === 'running';
      const stopping = status === 'stopping';
      const timestepsValue = positiveIntegerText(training?.total_timesteps, 1000000);
      const numEnvsValue = positiveIntegerText(training?.num_envs, 4);
      const mainButton = (running || stopping)
        ? `<button id="trainingMainAction" class="control-btn danger" ${stopping ? 'disabled' : ''}>Stop</button>`
        : `<button id="trainingMainAction" class="control-btn primary">Start</button>`;
      return [
        `<label class="pill">steps <input id="totalTimestepsInput" type="number" min="1" step="1" value="${escapeHtml(timestepsValue)}" ${running || stopping ? 'disabled' : ''} style="width:120px;margin-left:8px;background:#18202a;color:#f6f8fb;border:1px solid #2b3540;border-radius:8px;padding:6px 8px;"></label>`,
        `<label class="pill">threads <input id="numEnvsInput" type="number" min="1" max="64" step="1" value="${escapeHtml(numEnvsValue)}" ${running || stopping ? 'disabled' : ''} style="width:72px;margin-left:8px;background:#18202a;color:#f6f8fb;border:1px solid #2b3540;border-radius:8px;padding:6px 8px;"></label>`,
        mainButton,
        `<span id="trainingStatusPill" class="pill">status: ${escapeHtml(status)}</span>`,
      ].join('');
    }
    function syncControls(training) {
      const shell = document.getElementById('controls');
      if (!controlsInitialized || !document.getElementById('trainingMainAction')) {
        shell.innerHTML = renderControls(training);
        controlsInitialized = true;
      }
      const status = training?.status || 'idle';
      const running = status === 'running';
      const stopping = status === 'stopping';
      const busy = running || stopping;
      const totalTimestepsInput = document.getElementById('totalTimestepsInput');
      const numEnvsInput = document.getElementById('numEnvsInput');
      const mainAction = document.getElementById('trainingMainAction');
      const statusPill = document.getElementById('trainingStatusPill');
      const serverTimesteps = positiveIntegerText(training?.total_timesteps, 1000000);
      const serverNumEnvs = positiveIntegerText(training?.num_envs, 4);

      if (busy) {
        totalTimestepsInput.value = serverTimesteps;
        numEnvsInput.value = serverNumEnvs;
      } else {
        if (!String(totalTimestepsInput.value || '').trim()) {
          totalTimestepsInput.value = serverTimesteps;
        }
        if (!String(numEnvsInput.value || '').trim()) {
          numEnvsInput.value = serverNumEnvs;
        }
      }

      totalTimestepsInput.disabled = busy;
      numEnvsInput.disabled = busy;
      statusPill.textContent = `status: ${status}`;
      mainAction.textContent = busy ? 'Stop' : 'Start';
      mainAction.disabled = stopping;
      mainAction.className = busy ? 'control-btn danger' : 'control-btn primary';
      mainAction.onclick = busy ? stopTraining : startTraining;
    }
    async function startTraining() {
      clearAlerts();
      const stepInput = document.getElementById('totalTimestepsInput');
      const envInput = document.getElementById('numEnvsInput');
      const totalTimesteps = stepInput ? Number(stepInput.value || 0) : 0;
      const numEnvs = envInput ? Number(envInput.value || 0) : 0;
      try {
        const payload = {};
        if (Number.isFinite(totalTimesteps) && totalTimesteps > 0) {
          payload.total_timesteps = Math.trunc(totalTimesteps);
        }
        if (Number.isFinite(numEnvs) && numEnvs > 0) {
          payload.num_envs = Math.trunc(numEnvs);
        }
        const result = await postJson('/api/control/start', payload);
        if (result?.message) {
          pushAlert('Start requested', result.message, 'info');
        }
      } catch (err) {
        showRequestError('Start training', err);
      }
      await refresh();
    }
    async function stopTraining() {
      clearAlerts();
      try {
        await postJson('/api/control/stop', {});
      } catch (err) {
        showRequestError('Stop training', err);
      }
      await refresh();
    }
    function renderStateAlerts(data) {
      const clientErr = data?.launcher_logs?.client_stderr || {};
      const training = data?.training || {};
      const monitoring = data?.monitoring || {};
      if ((training.status === 'idle' || training.status === 'error') && clientErr.tail) {
        pushAlert(
          'Training client error',
          `status=${training.status} | log_updated_at=${fmt(clientErr.updated_at)}`,
          training.status === 'error' ? 'danger' : 'warn',
          clientErr.tail,
          'state',
        );
      }
      const overlongActive = monitoring?.overlong_active || [];
      if (overlongActive.length) {
        pushAlert(
          'Overlong active slots',
          overlongActive.map((item) => `slot=${fmt(item.slot)} seed=${fmt(item.seed)} uptime=${fmt(item.uptime_seconds)}s floor=${fmt(item.floor)} hp=${fmt(item.hp)}`).join('\\n'),
          'warn',
          '',
          'state',
        );
      }
      const suspiciousCompleted = monitoring?.suspicious_completed || [];
      if (suspiciousCompleted.length) {
        pushAlert(
          'Suspicious completed runs',
          suspiciousCompleted.map((item) => `seed=${fmt(item.seed)} flags=${fmt(item.flags_display)} floor=${fmt(item.max_floor)} hp=${fmt(item.final_hp ?? item.hp)} truncated=${fmt(item.truncated)} victory=${fmt(item.victory)}`).join('\\n'),
          'danger',
          '',
          'state',
        );
      }
      const recentIncidents = monitoring?.recent_incidents || [];
      if (recentIncidents.length) {
        pushAlert(
          'Automation incidents',
          recentIncidents.slice(0, 6).map((item) => `${fmt(item.source)} | ${fmt(item.type)} | seed=${fmt(item.seed)} | run=${fmt(item.run_id)} | resolution=${fmt(item.resolution_status || item.resolution_action || '-')}`).join('\\n'),
          'warn',
          '',
          'state',
        );
      }
    }
    function renderProblemList(data) {
      const shell = document.getElementById('problemList');
      const problems = data?.monitoring?.problem_list || [];
      if (!problems.length) {
        shell.innerHTML = '';
        return;
      }
      shell.innerHTML = `
        <h2>Detected Issues</h2>
        <div class="problem-shell">
          ${problems.map((item, index) => `
            <details class="problem-item" ${index === 0 ? 'open' : ''}>
              <summary>
                <span class="problem-title">
                  <span>${escapeHtml(item.seed || item.game_id || 'unknown')}</span>
                  <span class="pill">${escapeHtml(item.status || 'open')}</span>
                </span>
                <span class="problem-meta">${escapeHtml(item.run_id || '-')}</span>
              </summary>
              <div class="problem-body">
                <div class="problem-grid">
                  <div class="problem-label">Found At</div>
                  <div class="problem-value">${escapeHtml(formatDateTime(item.discovered_at))}</div>
                  <div class="problem-label">Resolved At</div>
                  <div class="problem-value">${escapeHtml(formatDateTime(item.resolved_at))}</div>
                  <div class="problem-label">Reason</div>
                  <div class="problem-value">${escapeHtml(item.reason || '-')}</div>
                </div>
                <div>
                  <div class="problem-label">Timeline</div>
                  <ul class="problem-source-list">
                    ${(item.sources || []).map((source) => `<li>${escapeHtml(source.source || '-')} | ${escapeHtml(source.type || '-')} | ${escapeHtml(formatDateTime(source.recorded_at))} | ${escapeHtml(source.reason || '-')} | ${escapeHtml(source.resolution_status || '-')}</li>`).join('')}
                  </ul>
                </div>
              </div>
            </details>
          `).join('')}
        </div>
      `;
    }
    function renderCards(data) {
      const t = data.training || {};
      const allTime = data.all_time_summary || {};
      const monitoring = data.monitoring || {};
      const resumeStateMap = {
        resumed: 'Resume',
        fresh: 'Fresh',
        fallback_fresh: 'Fallback Fresh',
      };
      const resumeStatus = resumeStateMap[t.resume_load_status] || fmt(t.resume_load_status);
      const resumePath = t.resume_model_path
        ? `<span class="mono">${escapeHtml(t.resume_model_path)}</span>`
        : 'No previous checkpoint';
      const resumeFailure = t.resume_failure_reason
        ? ` | ${escapeHtml(t.resume_failure_reason)}`
        : '';
      const cards = [
        ['Total Timesteps', kpi(t.total_timesteps), `Current ${kpi(t.current_timesteps)} | Progress ${kpi(pct(t.progress_pct || 0), 'metric-good')}`],
        ['Parallelism', kpi(t.num_envs), `VecEnv <span class="pill">${fmt(t.vec_env)}</span> | FPS ${kpi(t.fps, 'metric-good')}`],
        ['Jobs Seen', kpi(data.seen_sessions), `Completed ${kpi(data.completed_sessions, 'metric-good')} | Finished Episodes ${kpi(t.episodes_finished)}`],
        ['All-Time Runs', kpi(allTime.runs_seen), `Finished ${kpi(allTime.runs_finished, 'metric-good')} | Best Floor ${kpi(allTime.best_floor, 'metric-good')}`],
        ['Reward', kpi(t.mean_reward_100, 'metric-warn'), `Mean length 100 ${kpi(t.mean_length_100)}`],
        ['Resume', kpi(resumeStatus, t.resume_load_status === 'fallback_fresh' ? 'metric-warn' : 'metric-good'), `${resumePath}${resumeFailure}`],
        ['Health', kpi(monitoring.suspicious_completed_count || 0, (monitoring.suspicious_completed_count || 0) ? 'metric-warn' : 'metric-good'), `Overlong ${kpi(monitoring.overlong_active_count || 0, (monitoring.overlong_active_count || 0) ? 'metric-warn' : 'metric-good')} | Incidents ${kpi(monitoring.recent_incidents_count || 0, (monitoring.recent_incidents_count || 0) ? 'metric-warn' : 'metric-good')}`],
      ];
      return cards.map(([label,value,sub]) => `<div class="card"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`).join('');
    }
    function renderTable(el, columns, rows) {
      const head = `<tr>${columns.map(c => `<th>${c.label}</th>`).join('')}</tr>`;
      const body = rows.map(row => `<tr>${columns.map(c => `<td class="${c.className || ''}">${fmt(row[c.key])}</td>`).join('')}</tr>`).join('');
      el.innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
    }
    function sparkline(values, color) {
      if (!values.length) return '';
      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = Math.max(1, max - min);
      const pts = values.map((v, i) => {
        const x = (i / Math.max(1, values.length - 1)) * 220;
        const y = 50 - ((v - min) / range) * 44;
        return `${x},${y}`;
      }).join(' ');
      return `<svg width="220" height="54" viewBox="0 0 220 54"><polyline fill="none" stroke="${color}" stroke-width="2.5" points="${pts}" /></svg>`;
    }
    function renderSlotCharts(slotHistories) {
      const keys = Object.keys(slotHistories || {}).sort();
      if (!keys.length) return '<div class="muted">No completed episode history yet.</div>';
      return `<div class="slot-grid">${keys.map(key => {
        const rows = slotHistories[key] || [];
        const rewards = rows.map(r => Number(r.episode_reward || 0));
        const floors = rows.map(r => Number(r.floor || 0));
        const steps = rows.map(r => Number(r.episode_steps || 0));
        return `<div class="card" style="margin-bottom:12px">
          <div class="label">${key}</div>
          <div class="sub">recent episodes: ${rows.length}</div>
          <div style="display:flex; gap:16px; flex-wrap:wrap; margin-top:8px">
            <div><div class="label">Reward</div>${sparkline(rewards, '#d08a2f')}</div>
            <div><div class="label">Floor</div>${sparkline(floors, '#3e8f63')}</div>
            <div><div class="label">Steps</div>${sparkline(steps, '#2f80ed')}</div>
          </div>
        </div>`;
      }).join('')}</div>`;
    }
    function normalizeRoomType(roomType) {
      const value = String(roomType || '').toLowerCase();
      if (!value) return 'Unknown';
      if (value === 'unknown') return 'Event';
      if (value.includes('elite')) return 'Elite';
      if (value.includes('monster')) return 'Monster';
      if (value.includes('event')) return 'Event';
      if (value.includes('rest') || value.includes('campfire') || value.includes('smith')) return 'Rest';
      if (value.includes('shop') || value.includes('merchant') || value.includes('store')) return 'Shop';
      if (value.includes('treasure') || value.includes('chest')) return 'Treasure';
      if (value.includes('boss')) return 'Boss';
      if (value.includes('startreward') || value.includes('neow') || value.includes('map')) return 'StartReward';
      return roomType || 'Unknown';
    }
    function roomGlyph(roomType) {
      const normalized = normalizeRoomType(roomType);
      if (normalized === 'Monster') return 'M';
      if (normalized === 'Elite') return 'E';
      if (normalized === 'Event') return '?';
      if (normalized === 'Rest') return 'R';
      if (normalized === 'Shop') return '$';
      if (normalized === 'Treasure') return 'T';
      if (normalized === 'Boss') return 'B';
      if (normalized === 'StartReward') return 'N';
      return '路';
    }
    function roomClass(roomType) {
      const normalized = normalizeRoomType(roomType);
      if (normalized === 'Monster') return { fill: '#f8fbff', stroke: '#6f7f8f', text: '#2c3c4c' };
      if (normalized === 'Elite') return { fill: '#fff3e6', stroke: '#d08a2f', text: '#9a5e12' };
      if (normalized === 'Event') return { fill: '#f7f2ff', stroke: '#8b78d8', text: '#6147b8' };
      if (normalized === 'Rest') return { fill: '#edf8ef', stroke: '#4c9c68', text: '#2b6d45' };
      if (normalized === 'Shop') return { fill: '#eef8fb', stroke: '#4b99b7', text: '#2d6f88' };
      if (normalized === 'Treasure') return { fill: '#fff8e8', stroke: '#c69a2d', text: '#8a6612' };
      if (normalized === 'Boss') return { fill: '#ffeceb', stroke: '#c75151', text: '#973737' };
      if (normalized === 'StartReward') return { fill: '#edf3ff', stroke: '#2f80ed', text: '#245ca7' };
      return { fill: '#f2f5f8', stroke: '#8e99a6', text: '#62707e' };
    }
    function escapeId(value) {
      return String(value || '').replace(/[^a-zA-Z0-9_-]/g, '_');
    }
    function buildVisitedEdgeSet(activeMap) {
      const set = new Set();
      const nodeLookup = new Map(
        (activeMap.nodes || [])
          .filter(node => node && node.id && node.col !== null && node.row !== null)
          .map(node => [node.id, node])
      );
      const orderedVisited = [];
      for (const nodeId of (activeMap.visited_node_ids || [])) {
        const node = nodeLookup.get(nodeId);
        if (!node) continue;
        const last = orderedVisited[orderedVisited.length - 1];
        if (last && last.id === node.id) continue;
        orderedVisited.push(node);
      }
      for (let i = 1; i < orderedVisited.length; i += 1) {
        const prev = orderedVisited[i - 1];
        const next = orderedVisited[i];
        set.add(`${prev.id}->${next.id}`);
      }
      return set;
    }
    function trimLine(x1, y1, x2, y2, startRadius, endRadius) {
      const dx = x2 - x1;
      const dy = y2 - y1;
      const distance = Math.sqrt(dx * dx + dy * dy) || 1;
      const ux = dx / distance;
      const uy = dy / distance;
      return {
        x1: x1 + ux * startRadius,
        y1: y1 + uy * startRadius,
        x2: x2 - ux * endRadius,
        y2: y2 - uy * endRadius,
      };
    }
    function renderSummary(session) {
      const s = session.summary || {};
      const items = [
        ['Character', s.character],
        ['Boss', s.boss_name],
        ['Seed', s.seed],
        ['Act', s.act],
        ['Max Floor', s.max_floor],
        ['Final HP', `${fmt(s.final_hp)} / ${fmt(s.max_hp)}`],
        ['Final Gold', s.final_gold],
        ['Reward', s.episode_reward],
        ['Steps', s.episode_steps],
        ['Created At', formatDateTime(s.started_at)],
        ['Duration', formatDuration(s.started_at, s.finished_at)],
        ['Victory', s.victory],
        ['Terminated', s.terminated],
      ];
      return `<div class="card">
        <div class="label">Run Summary</div>
        <div class="summary-list">
          ${items.map(([label, value]) => {
            const compactClass = label === 'Seed' ? 'compact-text' : '';
            return `<div class="summary-item"><div class="label">${escapeHtml(label)}</div><strong class="${compactClass}">${escapeHtml(fmt(value))}</strong></div>`;
          }).join('')}
        </div>
      </div>`;
    }
    function renderMap(session) {
      const maps = session.maps || [];
      if (!maps.length) return '<div class="card map-card"><div class="node-detail-empty">No map snapshot persisted for this run.</div></div>';
      if (!currentAct || !maps.find(m => m.act === currentAct)) currentAct = maps[0].act;
      const activeMap = maps.find(m => m.act === currentAct) || maps[0];
      const activeBoss = activeMap.boss_info || activeMap.boss || {};
      const activeBossName = activeBoss.name || activeBoss.boss_name || activeBoss.display_name || activeBoss.id || session.summary?.boss_name || '-';
      const visitedIds = new Set(activeMap.visited_node_ids || []);
      const visitedEdges = buildVisitedEdgeSet(activeMap);
      const baseNodes = (activeMap.nodes || []).map(node => ({ ...node, room_type: normalizeRoomType(node.room_type) }));
      const nodes = [...baseNodes];
      const nodeLookup = Object.fromEntries(nodes.map(node => [node.id, node]));

      const visitedDetailNodes = (session.nodes || []).filter(node => Number(node.act) === Number(activeMap.act));
      visitedDetailNodes.forEach((detailNode) => {
        if ((detailNode.col === null || detailNode.row === null) && detailNode.floor === 1) {
          const startId = `a${activeMap.act}_start`;
          if (!nodeLookup[startId]) {
            const startNode = { id: startId, act: activeMap.act, col: 0, row: 0, room_type: 'StartReward', children: [] };
            nodes.push(startNode);
            nodeLookup[startId] = startNode;
          }
        }
      });

      let graphNodes = nodes.filter(node => node.col !== null && node.row !== null);
      if (!graphNodes.length) {
        graphNodes = visitedDetailNodes
          .filter(node => node.col !== null && node.row !== null)
          .map(node => ({
            id: node.node_id,
            act: node.act,
            col: Number(node.col),
            row: Number(node.row),
            room_type: normalizeRoomType(node.room_type),
            children: [],
          }));
      }
      const allRows = graphNodes.map(n => Number(n.row ?? 0));
      const allCols = graphNodes.map(n => Number(n.col ?? 0));
      const maxRow = Math.max(1, ...allRows, 1);
      const minRow = Math.min(0, ...allRows, 0);
      const minCol = Math.min(0, ...allCols, 0);
      const maxCol = Math.max(0, ...allCols, 0);
      const colCount = Math.max(1, maxCol - minCol + 1);
      const rowCount = Math.max(2, maxRow - minRow + 1);
      const xStep = 68;
      const yStep = 74;
      const width = Math.max(520, colCount * xStep + 96);
      const height = Math.max(420, rowCount * yStep + 88);
      const xOf = (col) => 48 + (Number(col ?? 0) - minCol) * xStep;
      const yOf = (row) => 44 + (maxRow - Number(row ?? 0)) * yStep;

      const allEdges = [];
      graphNodes.forEach((node) => {
        (node.children || []).forEach((child) => {
          const childId = `a${activeMap.act}_c${child.col}_r${child.row}`;
          const childNode = nodeLookup[childId] || graphNodes.find(item => item.id === childId);
          if (!childNode) return;
          const fromX = xOf(node.col);
          const fromY = yOf(node.row);
          const toX = xOf(childNode.col);
          const toY = yOf(childNode.row);
          const trimmed = trimLine(fromX, fromY, toX, toY, visitedIds.has(node.id) ? 19 : 15, visitedIds.has(childId) ? 19 : 15);
          allEdges.push({ from: node.id, to: childId, x1: trimmed.x1, y1: trimmed.y1, x2: trimmed.x2, y2: trimmed.y2 });
        });
      });

      const lineMarkup = allEdges.map((edge) => {
        const visited = visitedEdges.has(`${edge.from}->${edge.to}`);
        const stroke = visited ? '#1f5fbf' : '#98a6b5';
        const dash = visited ? '' : '4 4';
        const width = visited ? '4.8' : '2.2';
        const opacity = visited ? '1' : '0.92';
        return `<line x1="${edge.x1}" y1="${edge.y1}" x2="${edge.x2}" y2="${edge.y2}" stroke="${stroke}" stroke-dasharray="${dash}" stroke-width="${width}" opacity="${opacity}" style="stroke:${stroke} !important;stroke-width:${width} !important;${dash ? `stroke-dasharray:${dash} !important;` : ''}opacity:${opacity} !important;"/>`;
      }).join('');

      const rowGuides = Array.from({ length: rowCount }, (_, index) => {
        const rowValue = maxRow - index;
        const y = yOf(rowValue);
        return `<g><line x1="20" y1="${y}" x2="${width - 18}" y2="${y}" stroke="#e2e8ef" stroke-width="1"/><text x="6" y="${y + 4}" fill="#8a97a4" font-size="11">F${rowValue}</text></g>`;
      }).join('');

      const nodeMarkup = graphNodes.map(node => {
        const x = xOf(node.col);
        const y = yOf(node.row);
        const active = currentNodeId === node.id ? 'active' : '';
        const visited = visitedIds.has(node.id);
        const colors = roomClass(node.room_type);
        const isActive = currentNodeId === node.id;
        const stroke = isActive ? '#3e8f63' : (visited ? '#2f80ed' : 'transparent');
        const fill = '#34373d';
        const textColor = '#f2f6fb';
        const radius = visited ? 19 : 15;
        const strokeWidth = isActive ? 4 : (visited ? 3.5 : 0);
        return `<g class="node-button ${active}" data-node-id="${node.id}" transform="translate(${x},${y})">
          <circle cx="0" cy="0" r="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}" style="fill:${fill} !important;stroke:${stroke} !important;stroke-width:${strokeWidth} !important;"></circle>
          <text x="0" y="4" text-anchor="middle" fill="${textColor}" style="fill:${textColor} !important;">${roomGlyph(node.room_type)}</text>
        </g>`;
      }).join('');
      const legendItems = [
        'N Start',
        'M Monster',
        'E Elite',
        '? Event',
        'T Treasure',
        '$ Shop',
        'R Rest',
        'B Boss',
      ].map(text => `<span class="pill">${escapeHtml(text)}</span>`).join('');
      return `<div class="card map-card">
        <div class="label">Act Map</div>
        <div class="map-toolbar">${maps.map(map => `<button class="map-tab ${map.act === activeMap.act ? 'active' : ''}" data-act="${map.act}">Act ${map.act}</button>`).join('')}</div>
        <div class="sub">Boss: <span class="metric-warn">${escapeHtml(activeBossName)}</span></div>
        <div class="map-toolbar">${legendItems}</div>
        <div class="tree-wrap">
          <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" data-darkreader-ignore="true">
            ${rowGuides}
            ${lineMarkup}
            ${nodeMarkup}
          </svg>
        </div>
      </div>`;
    }
    function renderNodeDetail(session) {
      const nodes = session.nodes || [];
      if (!nodes.length) return '<div class="card"><div class="node-detail-empty">No node detail persisted.</div></div>';
      if (!currentNodeId || !nodes.find(node => node.node_id === currentNodeId)) currentNodeId = nodes[0].node_id;
      const selectedNode = nodes.find(item => item.node_id === currentNodeId) || nodes[0];
      const stageNodes = nodes.filter((item) =>
        Number(item.act) === Number(selectedNode.act) &&
        Number(item.floor) === Number(selectedNode.floor) &&
        normalizeRoomType(item.room_type) === normalizeRoomType(selectedNode.room_type)
      );
      const orderedStageNodes = stageNodes.length > 1
        ? [...stageNodes].sort((left, right) => {
            const leftActions = (left.actions || []).length;
            const rightActions = (right.actions || []).length;
            if (leftActions !== rightActions) return leftActions - rightActions;
            return String(left.node_id || '').localeCompare(String(right.node_id || ''));
          })
        : [selectedNode];
      if (currentNodeStageIndex >= orderedStageNodes.length) currentNodeStageIndex = 0;
      const node = orderedStageNodes[currentNodeStageIndex] || selectedNode;
      const rewards = (node.rewards || []).map(item => `<span class="reward-tag">${escapeHtml(item.type)}: ${escapeHtml(item.name ?? item.amount ?? '')}</span>`).join('');
      const actions = (node.actions || []).map(item => `<div class="action-row">
        <div><strong>#${escapeHtml(item.step)}</strong> ${escapeHtml(item.action)}</div>
        <div>${escapeHtml(item.action_detail || item.action)}</div>
        <div class="muted">before=${escapeHtml(item.decision_before)} after=${escapeHtml(item.decision_after)} reward=${escapeHtml(item.reward)}</div>
      </div>`).join('');
      const entryShopCards = node.entry_state?.cards || [];
      const entryShopRelics = node.entry_state?.shop_relics || [];
      const entryShopPotions = node.entry_state?.shop_potions || [];
      const entryPurgeCost = node.entry_state?.purge_cost;
      const purgeCandidates = node.entry_state?.deck || [];
      const shopItemName = (item) => typeof item === 'string' ? item : (item?.name || item?.id || '?');
      const shopDetails = normalizeRoomType(node.room_type) === 'Shop'
        ? `<div class="node-block"><div class="label">Shop Stock</div><div>
            <div>Cards: ${entryShopCards.length ? entryShopCards.map((item) => escapeHtml(shopItemName(item))).join(', ') : 'None'}</div>
            <div>Relics: ${entryShopRelics.length ? entryShopRelics.map((item) => escapeHtml(shopItemName(item))).join(', ') : 'None'}</div>
            <div>Potions: ${entryShopPotions.length ? entryShopPotions.map((item) => escapeHtml(shopItemName(item))).join(', ') : 'None'}</div>
            <div>Purge: ${entryPurgeCost !== null && entryPurgeCost !== undefined ? `${escapeHtml(entryPurgeCost)} gold (${escapeHtml(purgeCandidates.length)} candidates)` : 'Unavailable'}</div>
          </div></div>`
        : '';
      const stageTabs = orderedStageNodes.length > 1
        ? `<div class="map-toolbar">${orderedStageNodes.map((item, index) => `
            <button class="map-tab ${index === currentNodeStageIndex ? 'active' : ''}" data-stage-index="${index}">
              Stage ${index + 1}
            </button>
          `).join('')}</div>`
        : '';
      return `<div class="card">
        <div class="label">Node Detail</div>
        <div class="sub">Act ${escapeHtml(node.act)} | Floor ${escapeHtml(node.floor)} | ${escapeHtml(normalizeRoomType(node.room_type))}</div>
        ${stageTabs}
        <div class="node-stats">
          <div class="node-block"><div class="label">Monsters</div><div>${(node.monsters || []).length ? (node.monsters || []).map(escapeHtml).join(', ') : 'None recorded'}</div></div>
          <div class="node-block"><div class="label">Entry</div><div>HP ${escapeHtml(node.entry_state?.hp)} / ${escapeHtml(node.entry_state?.max_hp)} | Gold ${escapeHtml(node.entry_state?.gold)} | Decision ${escapeHtml(node.entry_state?.decision)}</div></div>
          <div class="node-block"><div class="label">Exit</div><div>HP ${escapeHtml(node.exit_state?.hp)} / ${escapeHtml(node.exit_state?.max_hp)} | Gold ${escapeHtml(node.exit_state?.gold)} | Decision ${escapeHtml(node.exit_state?.decision)}</div></div>
          ${shopDetails}
          <div class="node-block"><div class="label">Rewards</div><div>${rewards || '<span class="muted">No explicit rewards captured.</span>'}</div></div>
          <div class="node-block"><div class="label">Actions</div><div class="action-log">${actions || '<div class="action-row muted">No actions recorded.</div>'}</div></div>
        </div>
      </div>`;
    }
    function bindSessionInteractions() {
      Array.from(document.querySelectorAll('.map-tab')).forEach((button) => {
        if (button.dataset.act !== undefined) {
          button.addEventListener('click', () => {
            currentAct = Number(button.dataset.act);
            currentNodeId = null;
            currentNodeStageIndex = 0;
            renderSessionDetail();
          });
          return;
        }
        if (button.dataset.stageIndex !== undefined) {
          button.addEventListener('click', () => {
            currentNodeStageIndex = Number(button.dataset.stageIndex) || 0;
            renderSessionDetail();
          });
        }
      });
      Array.from(document.querySelectorAll('.node-button')).forEach((node) => {
        node.addEventListener('click', () => {
          currentNodeId = node.dataset.nodeId;
          currentNodeStageIndex = 0;
          renderSessionDetail();
        });
      });
    }
    function renderSessionDetail() {
      const shell = document.getElementById('sessionDetail');
      if (!currentSession) {
        shell.innerHTML = '';
        return;
      }
      shell.innerHTML = `<h2>Selected Job</h2><div class="detail-grid">
        ${renderSummary(currentSession)}
        ${renderMap(currentSession)}
        ${renderNodeDetail(currentSession)}
      </div>`;
      bindSessionInteractions();
    }
    async function loadSession(gameId, runDir = null) {
      currentSessionGameId = gameId;
      const suffix = runDir ? `?run_dir=${encodeURIComponent(runDir)}` : '';
      const resp = await fetch(`/api/session/${encodeURIComponent(gameId)}${suffix}`);
      if (!resp.ok) {
        currentSession = null;
        renderSessionDetail();
        return;
      }
        currentSession = await resp.json();
        currentAct = null;
        currentNodeId = null;
        currentNodeStageIndex = 0;
        renderSessionDetail();
      }
    async function refresh() {
      let data = null;
      try {
        const resp = await fetch('/api/state');
        if (!resp.ok) {
          throw buildErrorFromResponse(resp.status, {});
        }
        data = await resp.json();
      } catch (err) {
        pushAlert('Failed to refresh state', err?.message || 'unable to fetch state', 'danger', '', 'state');
        return;
      }
        clearStateAlerts();
        renderStateAlerts(data);
        renderProblemList(data);
        const t = data.training || {};
      currentRunDir = data.run_dir;
      document.getElementById('meta').textContent = `updated: ${fmt(data.updated_at)} | experiment: ${fmt(t.experiment_name)} | run: ${fmt(t.run_id)} | dir: ${fmt(data.run_dir)}`;
      syncControls(t);
      document.getElementById('cards').innerHTML = renderCards(data);
      const activeSlots = withTimeFields(data.active_slots || []).map((row) => ({
        ...row,
        uptime_display: formatDuration(inferEpisodeStart(row), null),
      }));
      renderTable(document.getElementById('activeTable'), [
        {key:'slot',label:'Slot'},
        {key:'episode_index',label:'Episode'},
        {key:'floor',label:'Floor'},
        {key:'act',label:'Act'},
        {key:'boss_name',label:'Boss'},
        {key:'room_type',label:'Room'},
        {key:'decision',label:'Decision'},
        {key:'hp',label:'HP'},
        {key:'gold',label:'Gold'},
        {key:'energy',label:'Energy'},
        {key:'uptime_display',label:'Uptime'},
        {key:'episode_steps',label:'Steps'},
        {key:'episode_reward',label:'Reward'},
        {key:'flags_display',label:'Flags'},
        {key:'seed',label:'Seed', className:'mono'},
      ], activeSlots);
      Array.from(document.querySelectorAll('#activeTable tbody tr')).forEach((row) => {
        const roomCell = row.children[5];
        const floorCell = row.children[2];
        const rewardCell = row.children[12];
        if (roomCell) roomCell.classList.add('room');
        if (floorCell) floorCell.classList.add('metric-strong');
        if (rewardCell) rewardCell.classList.add('metric-warn');
        const slot = activeSlots[row.rowIndex - 1];
        if (slot && slot.details_available) {
          row.classList.add('clickable-row');
          row.addEventListener('click', () => loadSession(slot.game_id, data.run_dir));
        }
      });
      document.getElementById('slotCharts').innerHTML = renderSlotCharts(data.slot_histories || {});
      renderTable(document.getElementById('checkpointTable'), [
        {key:'name',label:'File'},
        {key:'size_mb',label:'Size(MB)'},
        {key:'path',label:'Path', className:'mono'},
      ], data.checkpoints || []);
      const topSessions = withTimeFields(data.top_sessions || []);
      renderTable(document.getElementById('topTable'), [
        {key:'slot',label:'Slot'},
        {key:'episode_index',label:'Episode'},
        {key:'boss_name',label:'Boss'},
        {key:'max_floor',label:'Max Floor'},
        {key:'started_at_display',label:'Created At'},
        {key:'duration_display',label:'Duration'},
        {key:'final_hp_display',label:'Final HP'},
        {key:'final_gold_display',label:'Final Gold'},
        {key:'episode_steps',label:'Steps'},
        {key:'episode_reward',label:'Reward'},
        {key:'flags_display',label:'Flags'},
        {key:'victory',label:'Victory'},
        {key:'seed',label:'Seed', className:'mono'},
      ], topSessions);
      Array.from(document.querySelectorAll('#topTable tbody tr')).forEach((row) => {
        row.classList.add('clickable-row');
        const floorCell = row.children[3];
        const hpCell = row.children[6];
        const rewardCell = row.children[9];
        const victoryCell = row.children[11];
        if (floorCell) floorCell.classList.add('metric-good');
        if (hpCell) hpCell.classList.add('metric-strong');
        if (rewardCell) rewardCell.classList.add('metric-warn');
        if (victoryCell && victoryCell.textContent === 'True') victoryCell.classList.add('metric-good');
        const session = topSessions[row.rowIndex - 1];
        if (session && session.details_available) {
          row.addEventListener('click', () => loadSession(session.game_id, data.run_dir));
        }
      });
      const historicalBest = withTimeFields(data.historical_best || []);
      renderTable(document.getElementById('historyTable'), [
        {key:'experiment_name',label:'Experiment'},
        {key:'run_id',label:'Run'},
        {key:'boss_name',label:'Boss'},
        {key:'max_floor',label:'Max Floor'},
        {key:'started_at_display',label:'Created At'},
        {key:'duration_display',label:'Duration'},
        {key:'final_hp_display',label:'Final HP'},
        {key:'final_gold_display',label:'Final Gold'},
        {key:'episode_reward',label:'Reward'},
        {key:'episode_steps',label:'Steps'},
        {key:'flags_display',label:'Flags'},
        {key:'victory',label:'Victory'},
        {key:'seed',label:'Seed', className:'mono'},
      ], historicalBest);
      Array.from(document.querySelectorAll('#historyTable tbody tr')).forEach((row) => {
        row.classList.add('clickable-row');
        const floorCell = row.children[3];
        const hpCell = row.children[6];
        const rewardCell = row.children[8];
        const victoryCell = row.children[11];
        if (floorCell) floorCell.classList.add('metric-good');
        if (hpCell) hpCell.classList.add('metric-strong');
        if (rewardCell) rewardCell.classList.add('metric-warn');
        if (victoryCell && victoryCell.textContent === 'True') victoryCell.classList.add('metric-good');
          const session = historicalBest[row.rowIndex - 1];
          if (session) {
            const runDir = session.run_dir || '';
            row.addEventListener('click', () => loadSession(session.game_id, runDir));
          }
      });
    }
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>"""


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


@app.route("/api/session/<game_id>")
def api_session(game_id: str) -> Response:
    _ensure_telemetry_bootstrapped()
    run_dir_arg = request.args.get("run_dir")
    run_dir = Path(run_dir_arg) if run_dir_arg else _latest_run_dir()
    with TELEMETRY_LOCK:
        for run in (TELEMETRY_STATE.get("runs") or {}).values():
            details = (run.get("session_details") or {}).get(game_id)
            if isinstance(details, dict) and details:
                return jsonify(details)
    if run_dir is None:
        session = _build_service_only_session(game_id)
        if not session:
            return jsonify({"error": "run not found"}), 404
        return jsonify(session)
    session = _read_session_details(run_dir, game_id)
    if not session:
        session = _build_live_session(run_dir, game_id)
    if not session:
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


@app.route("/api/control/start", methods=["POST"])
def api_control_start() -> Response:
    payload = request.get_json(silent=True) or {}
    total_timesteps = _parse_total_timesteps(payload.get("total_timesteps"))
    num_envs = _parse_num_envs(payload.get("num_envs"))
    if payload.get("total_timesteps") not in (None, "") and total_timesteps is None:
        return jsonify({"ok": False, "message": "invalid total_timesteps"}), 400
    if payload.get("num_envs") not in (None, "") and num_envs is None:
        return jsonify({"ok": False, "message": "invalid num_envs"}), 400
    result = _start_training_process(total_timesteps=total_timesteps, num_envs=num_envs)
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
