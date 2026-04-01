# -*- coding: utf-8 -*-
"""Session-level supervisor for detecting stuck training games for investigation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG_PATH = ROOT / "configs" / "runtime_stack.json"
SUPERVISOR_ROOT = ROOT / "logs" / "session_supervisor"
INCIDENTS_DIR = SUPERVISOR_ROOT / "incidents"
STATE_PATH = SUPERVISOR_ROOT / "state.json"
STATUS_PATH = SUPERVISOR_ROOT / "current_status.json"
RUN_DIR_SAFE_CHARS = ":/\\"


@dataclass(slots=True)
class SupervisorConfig:
    poll_seconds: int = 15
    floor_timeout_seconds: int = 60
    stale_slot_seconds: int = 75
    stagnant_steps_threshold: int = 2
    intervention_cooldown_seconds: int = 90
    uptime_grace_seconds: int = 60


def _now() -> datetime:
    return datetime.now()


def _now_iso() -> str:
    return _now().isoformat(timespec="seconds")


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _seconds_since(value: Any) -> float | None:
    dt = _parse_iso(value)
    if dt is None:
        return None
    return max((_now() - dt).total_seconds(), 0.0)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return data


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_runtime_config() -> dict[str, Any]:
    data = _read_json(RUNTIME_CONFIG_PATH, {})
    return data if isinstance(data, dict) else {}


def _session_supervisor_config() -> SupervisorConfig:
    runtime = _load_runtime_config()
    payload = runtime.get("session_supervisor") or {}
    if not isinstance(payload, dict):
        return SupervisorConfig()
    return SupervisorConfig(
        poll_seconds=max(5, _safe_int(payload.get("poll_seconds"), 15)),
        floor_timeout_seconds=max(30, _safe_int(payload.get("floor_timeout_seconds"), 60)),
        stale_slot_seconds=max(30, _safe_int(payload.get("stale_slot_seconds"), 75)),
        stagnant_steps_threshold=max(1, _safe_int(payload.get("stagnant_steps_threshold"), 2)),
        intervention_cooldown_seconds=max(15, _safe_int(payload.get("intervention_cooldown_seconds"), 90)),
        uptime_grace_seconds=max(30, _safe_int(payload.get("uptime_grace_seconds"), 60)),
    )


def _service_base_url(runtime: dict[str, Any]) -> str:
    service = runtime.get("service") or {}
    host = str(service.get("host") or "127.0.0.1")
    port = _safe_int(service.get("port"), 5000)
    return f"http://{host}:{port}"


def _dashboard_base_url() -> str:
    return "http://127.0.0.1:8787"


def _json_request(
    url: str,
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
    request = Request(url, data=data, method=method, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _dashboard_state() -> dict[str, Any]:
    return _json_request(f"{_dashboard_base_url()}/api/state")


def _fetch_session_snapshot(game_id: str, run_dir: str | None) -> dict[str, Any]:
    encoded_game_id = quote(game_id, safe="")
    url = f"{_dashboard_base_url()}/api/session/{encoded_game_id}"
    if run_dir:
        encoded_run_dir = quote(run_dir, safe=RUN_DIR_SAFE_CHARS)
        url = f"{url}?run_dir={encoded_run_dir}"
    try:
        payload = _json_request(url, timeout=8.0)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_state() -> dict[str, Any]:
    state = _read_json(
        STATE_PATH,
        {
            "watch_started_at": _now_iso(),
            "current_run_id": None,
            "current_run_dir": None,
            "current_run_boundary_at": None,
            "active_floor_progress": {},
            "handled_games": {},
            "processed_incidents": [],
        },
    )
    if not isinstance(state, dict):
        state = {}
    state.setdefault("watch_started_at", _now_iso())
    state.setdefault("current_run_id", None)
    state.setdefault("current_run_dir", None)
    state.setdefault("current_run_boundary_at", None)
    state.setdefault("active_floor_progress", {})
    state.setdefault("handled_games", {})
    state.setdefault("processed_incidents", [])
    return state


def _trim_state(state: dict[str, Any]) -> dict[str, Any]:
    handled_games = state.get("handled_games") or {}
    if isinstance(handled_games, dict):
        items = list(handled_games.items())[-512:]
        state["handled_games"] = {str(key): str(value) for key, value in items}
    else:
        state["handled_games"] = {}
    processed = state.get("processed_incidents") or []
    state["processed_incidents"] = list(dict.fromkeys(str(item) for item in processed))[-2000:]
    progress = state.get("active_floor_progress") or {}
    if isinstance(progress, dict):
        items = list(progress.items())[-128:]
        state["active_floor_progress"] = {str(key): value for key, value in items}
    else:
        state["active_floor_progress"] = {}
    return state


def _refresh_run_scope(snapshot: dict[str, Any], state: dict[str, Any]) -> None:
    training = snapshot.get("training") or {}
    run_dir = str(snapshot.get("run_dir") or "")
    run_id = str(training.get("run_id") or Path(run_dir).name or "")
    if not run_id:
        return
    if state.get("current_run_id") == run_id and state.get("current_run_dir") == run_dir:
        return
    state["current_run_id"] = run_id
    state["current_run_dir"] = run_dir
    state["current_run_boundary_at"] = _now_iso()
    state["active_floor_progress"] = {}
    print(f"[session-supervisor] monitoring current run only: run_id={run_id}", flush=True)


def _slot_key(slot: dict[str, Any]) -> str:
    return str(slot.get("slot") or slot.get("game_id") or slot.get("seed") or "")


def _floor_age_seconds(slot: dict[str, Any], state: dict[str, Any]) -> float | None:
    progress_by_slot = state.setdefault("active_floor_progress", {})
    if not isinstance(progress_by_slot, dict):
        progress_by_slot = {}
        state["active_floor_progress"] = progress_by_slot

    key = _slot_key(slot)
    game_id = str(slot.get("game_id") or "")
    floor = max(1, _safe_int(slot.get("floor"), 1))
    observed_at = str(slot.get("updated_at") or slot.get("started_at") or _now_iso())

    current = progress_by_slot.get(key)
    if not isinstance(current, dict) or str(current.get("game_id") or "") != game_id or _safe_int(current.get("floor"), -1) != floor:
        progress_by_slot[key] = {
            "game_id": game_id,
            "floor": floor,
            "floor_since": observed_at,
            "last_seen_at": observed_at,
        }
        return 0.0

    current["last_seen_at"] = observed_at
    return _seconds_since(current.get("floor_since") or observed_at)


def _incident_key(issue_type: str, run_id: str, game_id: str) -> str:
    return f"{issue_type}::{run_id}::{game_id}"


def _already_handled(game_id: str, state: dict[str, Any], config: SupervisorConfig) -> bool:
    handled_games = state.setdefault("handled_games", {})
    if not isinstance(handled_games, dict):
        handled_games = {}
        state["handled_games"] = handled_games
    last_handled_at = _parse_iso(handled_games.get(game_id))
    if last_handled_at is None:
        return False
    return (_now() - last_handled_at).total_seconds() < config.intervention_cooldown_seconds


def _mark_handled(game_id: str, state: dict[str, Any]) -> None:
    handled_games = state.setdefault("handled_games", {})
    if not isinstance(handled_games, dict):
        handled_games = {}
        state["handled_games"] = handled_games
    handled_games[game_id] = _now_iso()


def _record_incident(payload: dict[str, Any], state: dict[str, Any]) -> None:
    key = _incident_key(str(payload.get("type") or ""), str(payload.get("run_id") or ""), str(payload.get("game_id") or ""))
    if key in set(state.get("processed_incidents") or []):
        return
    timestamp = _now().strftime("%Y%m%d_%H%M%S")
    safe_seed = str(payload.get("seed") or payload.get("game_id") or "incident").replace(":", "_")
    path = INCIDENTS_DIR / f"{timestamp}_{payload.get('type')}_{safe_seed}.json"
    enriched = dict(payload)
    enriched["recorded_at"] = _now_iso()
    _write_json(path, enriched)
    state.setdefault("processed_incidents", []).append(key)
    print(
        f"[session-supervisor] incident recorded: type={payload.get('type')} seed={payload.get('seed')} action={payload.get('resolution_action')}",
        flush=True,
    )


def _active_issue(slot: dict[str, Any], config: SupervisorConfig, state: dict[str, Any], run_id: str, run_dir: str) -> dict[str, Any] | None:
    if not bool(slot.get("active", True)):
        return None
    game_id = str(slot.get("game_id") or "")
    seed = str(slot.get("seed") or "")
    if not game_id or not seed:
        return None

    reasons: list[str] = []
    floor_age_seconds = _floor_age_seconds(slot, state)
    updated_age_seconds = _seconds_since(slot.get("updated_at"))
    stagnant_steps = _safe_int(slot.get("stagnant_steps"), 0)
    elapsed_seconds = slot.get("uptime_seconds")
    if elapsed_seconds in (None, ""):
        elapsed_seconds = _seconds_since(slot.get("started_at"))
    else:
        try:
            elapsed_seconds = float(elapsed_seconds)
        except (TypeError, ValueError):
            elapsed_seconds = _seconds_since(slot.get("started_at"))

    if elapsed_seconds is not None and elapsed_seconds < config.uptime_grace_seconds:
        return None

    if floor_age_seconds is not None and floor_age_seconds > config.floor_timeout_seconds:
        reasons.append(f"floor_stalled>{config.floor_timeout_seconds}s")
    if updated_age_seconds is not None and updated_age_seconds > config.stale_slot_seconds:
        reasons.append(f"slot_update_stale>{config.stale_slot_seconds}s")
    if stagnant_steps >= config.stagnant_steps_threshold:
        reasons.append(f"stagnant_steps>={config.stagnant_steps_threshold}")

    if not reasons:
        return None

    return {
        "type": "stuck_active_session",
        "run_id": run_id,
        "run_dir": run_dir,
        "slot": slot.get("slot"),
        "game_id": game_id,
        "seed": seed,
        "floor": slot.get("floor"),
        "hp": slot.get("hp"),
        "max_hp": slot.get("max_hp"),
        "started_at": slot.get("started_at"),
        "updated_at": slot.get("updated_at"),
        "elapsed_seconds": elapsed_seconds,
        "uptime_seconds": elapsed_seconds,
        "floor_age_seconds": floor_age_seconds,
        "updated_age_seconds": updated_age_seconds,
        "stagnant_steps": stagnant_steps,
        "reasons": reasons,
        "reason": ", ".join(reasons),
    }


def _resolve_issue(issue: dict[str, Any], runtime: dict[str, Any], state: dict[str, Any], config: SupervisorConfig) -> dict[str, Any]:
    game_id = str(issue.get("game_id") or "")
    if not game_id:
        issue["resolution_action"] = "skip"
        issue["resolution_status"] = "no_game_id"
        return issue

    if _already_handled(game_id, state, config):
        issue["resolution_action"] = "skip"
        issue["resolution_status"] = "cooldown"
        return issue

    session = _fetch_session_snapshot(game_id, str(issue.get("run_dir") or ""))
    if session:
        issue["session_snapshot"] = session

    issue["resolution_action"] = "investigate_required"
    issue["resolution_status"] = "recorded"
    _mark_handled(game_id, state)
    return issue


def _write_status(snapshot: dict[str, Any] | None, incident_count: int, state: dict[str, Any]) -> None:
    payload = {
        "updated_at": _now_iso(),
        "run_id": ((snapshot or {}).get("training") or {}).get("run_id") if snapshot else None,
        "active_slots": len((snapshot or {}).get("active_slots") or []),
        "incidents_detected": incident_count,
        "handled_games": len((state.get("handled_games") or {})),
    }
    _write_json(STATUS_PATH, payload)


def main() -> None:
    SUPERVISOR_ROOT.mkdir(parents=True, exist_ok=True)
    INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    print("[session-supervisor] started", flush=True)

    while True:
        runtime = _load_runtime_config()
        config = _session_supervisor_config()
        state = _load_state()
        snapshot: dict[str, Any] | None = None
        incident_count = 0

        _write_status(snapshot, incident_count, state)

        try:
            snapshot = _dashboard_state()
            if not isinstance(snapshot, dict):
                raise RuntimeError("dashboard returned non-dict payload")

            _refresh_run_scope(snapshot, state)
            run_id = str((snapshot.get("training") or {}).get("run_id") or "")
            run_dir = str(snapshot.get("run_dir") or "")

            active_slots = [slot for slot in (snapshot.get("active_slots") or []) if isinstance(slot, dict)]
            active_progress = state.get("active_floor_progress") or {}
            active_keys = {_slot_key(slot) for slot in active_slots}
            if isinstance(active_progress, dict):
                for existing_key in list(active_progress.keys()):
                    if existing_key not in active_keys:
                        active_progress.pop(existing_key, None)

            for slot in active_slots:
                issue = _active_issue(slot, config, state, run_id, run_dir)
                if issue is None:
                    continue
                issue = _resolve_issue(issue, runtime, state, config)
                _record_incident(issue, state)
                incident_count += 1
                _write_status(snapshot, incident_count, state)
        except (URLError, TimeoutError, OSError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"[session-supervisor] dashboard unavailable: {exc}", flush=True)
        except Exception as exc:
            print(f"[session-supervisor] unexpected error: {exc}", flush=True)
        finally:
            _trim_state(state)
            _write_json(STATE_PATH, state)
            _write_status(snapshot, incident_count, state)

        time.sleep(config.poll_seconds)


if __name__ == "__main__":
    main()
