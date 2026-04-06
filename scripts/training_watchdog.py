# -*- coding: utf-8 -*-
"""Continuous watchdog for HTTP CLI RL training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG_PATHS = [
    ROOT / "configs" / "runtime_stack.json",
    ROOT / "configs" / "runtime_stack.local.json",
]
WATCHDOG_ROOT = ROOT / "logs" / "watchdog"
INCIDENTS_DIR = WATCHDOG_ROOT / "incidents"
STATE_PATH = WATCHDOG_ROOT / "watchdog_state.json"
STATUS_PATH = WATCHDOG_ROOT / "current_status.json"
SESSION_SUPERVISOR_STATUS_PATH = ROOT / "logs" / "session_supervisor" / "current_status.json"


@dataclass(slots=True)
class WatchdogConfig:
    poll_seconds: int = 15
    per_floor_timeout_seconds: int = 180
    stale_slot_seconds: int = 75
    stagnant_steps_threshold: int = 2
    max_uptime_seconds: int = 900
    restart_cooldown_seconds: int = 180
    no_training_grace_seconds: int = 45
    total_timesteps: int = 1_000_000


def _now() -> datetime:
    return datetime.now()


def _now_iso() -> str:
    return _now().isoformat(timespec="seconds")


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


def _dashboard_base_url() -> str:
    runtime = _load_runtime_config()
    dashboard = runtime.get("dashboard") or {}
    host = str(dashboard.get("host") or "127.0.0.1")
    port = _safe_int(dashboard.get("port"), 8787)
    return f"http://{host}:{port}"


def _post_dashboard(path: str, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        f"{_dashboard_base_url()}{path}",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=1.0) as response:
            response.read()
    except (OSError, TimeoutError, URLError):
        return


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


def _load_runtime_config() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in RUNTIME_CONFIG_PATHS:
        data = _read_json(path, {})
        if isinstance(data, dict):
            merged = _deep_merge_dicts(merged, data)
    return merged


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _watchdog_config() -> WatchdogConfig:
    runtime = _load_runtime_config()
    payload = runtime.get("watchdog") or {}
    if not isinstance(payload, dict):
        return WatchdogConfig()
    return WatchdogConfig(
        poll_seconds=max(5, _safe_int(payload.get("poll_seconds"), 15)),
        per_floor_timeout_seconds=max(30, _safe_int(payload.get("per_floor_timeout_seconds"), 60)),
        stale_slot_seconds=max(30, _safe_int(payload.get("stale_slot_seconds"), 75)),
        stagnant_steps_threshold=max(1, _safe_int(payload.get("stagnant_steps_threshold"), 2)),
        max_uptime_seconds=max(120, _safe_int(payload.get("max_uptime_seconds"), 900)),
        restart_cooldown_seconds=max(30, _safe_int(payload.get("restart_cooldown_seconds"), 180)),
        no_training_grace_seconds=max(10, _safe_int(payload.get("no_training_grace_seconds"), 45)),
        total_timesteps=max(1, _safe_int(payload.get("total_timesteps"), 1_000_000)),
    )


def _config_dir() -> Path:
    return ROOT / "configs"


def _project_root(runtime: dict[str, Any], key: str, fallback: Path) -> Path:
    projects = runtime.get("projects") or {}
    raw = str((projects or {}).get(key) or "").strip()
    if not raw:
        return fallback
    path = Path(raw)
    if path.is_absolute():
        return path
    return (_config_dir() / path).resolve()


def _configured_num_envs(runtime: dict[str, Any]) -> int:
    client = runtime.get("client") or {}
    args = [str(item) for item in (client.get("args") or [])]
    for index, value in enumerate(args):
        if value == "--num-envs" and index + 1 < len(args):
            return max(1, _safe_int(args[index + 1], 1))
    return 1


def _recommended_num_envs(runtime: dict[str, Any], state: dict[str, Any], snapshot: dict[str, Any] | None = None) -> int:
    configured = max(1, _configured_num_envs(runtime))
    current = _safe_int(state.get("current_num_envs"), 0)
    if current <= 0 and snapshot:
        training = (snapshot.get("training") or {})
        current = _safe_int(training.get("num_envs"), 0)
        if current <= 0:
            current = len(snapshot.get("active_slots") or [])
    if current <= 0:
        current = configured
    history = state.get("restart_history") or []
    recent_cutoff = _now().timestamp() - 3600
    recent_count = 0
    performance_markers = (
        "worker_leak_detected",
        "slow_active_game",
        "performance",
        "throughput",
        "overload",
        "capacity",
    )
    if isinstance(history, list):
        for value in history:
            dt = None
            reason = ""
            if isinstance(value, dict):
                dt = _parse_iso(value.get("at"))
                reason = str(value.get("reason") or "").lower()
            else:
                dt = _parse_iso(value)
            if dt is None or dt.timestamp() < recent_cutoff:
                continue
            if not reason:
                continue
            if any(marker in reason for marker in performance_markers):
                recent_count += 1
    if configured >= 10:
        if recent_count >= 6:
            return min(configured, 6)
        if recent_count >= 3:
            return min(configured, 8)
    return min(configured, current if current > 0 else configured)


def _service_base_url(runtime: dict[str, Any]) -> str:
    service = runtime.get("service") or {}
    host = str(service.get("host") or "127.0.0.1")
    port = _safe_int(service.get("port"), 5000)
    return f"http://{host}:{port}"


def _dashboard_base_url() -> str:
    return "http://127.0.0.1:8787"


def _json_request(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: float = 5.0) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, method=method, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _dashboard_state() -> dict[str, Any]:
    return _json_request(f"{_dashboard_base_url()}/api/state")


def _post_dashboard(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return _json_request(f"{_dashboard_base_url()}{path}", method="POST", payload=payload or {})


def _post_dashboard_best_effort(path: str, payload: dict[str, Any] | None = None) -> bool:
    try:
        _post_dashboard(path, payload)
        return True
    except (URLError, TimeoutError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[watchdog] dashboard telemetry post failed: path={path} error={exc}", flush=True)
        return False


def _service_health(runtime: dict[str, Any]) -> dict[str, Any]:
    return _json_request(f"{_service_base_url(runtime)}/health")


def _subprocess_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
    return kwargs


def _run_powershell(command: str, timeout: int = 20) -> str:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        **_subprocess_kwargs(),
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return output.strip()


def _find_python_pids(pattern: str) -> list[int]:
    escaped = pattern.replace("\\", "\\\\").replace("'", "''")
    try:
        output = _run_powershell(
            "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
            f"Where-Object {{ $_.CommandLine -and $_.CommandLine -match '{escaped}' }} | "
            "ForEach-Object { $_.ProcessId }"
        )
    except Exception:
        return []
    return [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]


def _pid_file(name: str) -> Path:
    return ROOT / "logs" / "launcher" / f"{name}.pid"


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
            **_subprocess_kwargs(),
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
        try:
            path.unlink()
        except OSError:
            pass
        return None
    if _process_exists(pid):
        return pid
    try:
        path.unlink()
    except OSError:
        pass
    return None


def _node_pid_name(node: dict[str, Any]) -> str | None:
    script_name = Path(str(node.get("script") or "")).name.lower()
    return {
        "train_http_cli_rl.py": "client",
        "training_dashboard.py": "dashboard",
        "training_watchdog.py": "watchdog",
        "training_session_supervisor.py": "session_supervisor",
        "http_game_service.py": "service",
    }.get(script_name)


def _resolve_managed_path(base_dir: Path, value: Any) -> Path:
    raw = str(value or "").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _shared_runtime_env(runtime: dict[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {"PYTHONUNBUFFERED": "1"}
    projects = runtime.get("projects") or {}
    game_dir = str((projects or {}).get("game_dir") or "").strip()
    src_dir = str((ROOT / "src").resolve())
    existing_pythonpath = os.environ.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([src_dir, existing_pythonpath])
    else:
        env["PYTHONPATH"] = src_dir
    if game_dir:
        env["STS2_GAME_DIR"] = game_dir
    return env


def _node_python_executable(runtime: dict[str, Any], node: dict[str, Any] | None) -> str:
    if isinstance(node, dict):
        value = str(node.get("python_executable") or "").strip()
        if value:
            return value
    return str(runtime.get("python_executable") or sys.executable or "python")


def _start_managed_process(name: str, node: dict[str, Any] | None, runtime: dict[str, Any]) -> bool:
    if not isinstance(node, dict) or not bool(node.get("enabled", False)):
        return False
    default_root = _project_root(runtime, "sts2_cli_root" if name == "service" else "st2rl_root", ROOT)
    workdir = _resolve_managed_path(default_root, node.get("workdir") or default_root)
    script_path = _resolve_managed_path(default_root, node.get("script"))
    if not script_path.exists():
        print(f"[watchdog] cannot start {name}: script missing at {script_path}", flush=True)
        return False

    log_dir = ROOT / "logs" / "launcher"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.out.log"
    stderr_path = log_dir / f"{name}.err.log"

    env = os.environ.copy()
    env.update(_shared_runtime_env(runtime))
    for key, value in (node.get("env") or {}).items():
        env[str(key)] = str(value)
    python_executable = _node_python_executable(runtime, node)

    try:
        with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open("a", encoding="utf-8") as stderr_handle:
            subprocess.Popen(
                [python_executable, str(script_path), *[str(item) for item in (node.get("args") or [])]],
                cwd=str(workdir),
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=env,
                **_subprocess_kwargs(),
            )
        print(f"[watchdog] started managed process: {name}", flush=True)
        return True
    except Exception as exc:
        print(f"[watchdog] failed to start managed process {name}: {exc}", flush=True)
        return False


def _is_managed_process_running(node: dict[str, Any] | None) -> bool:
    if not isinstance(node, dict):
        return True
    if not bool(node.get("enabled", False)):
        return True
    pid_name = _node_pid_name(node)
    if pid_name:
        pid = _pid_from_file(pid_name)
        if pid is not None:
            return True
    process_match = str(node.get("process_match") or "").strip()
    if not process_match:
        return True
    if "training_session_supervisor.py" in process_match:
        status = _read_json(SESSION_SUPERVISOR_STATUS_PATH, {})
        if isinstance(status, dict):
            updated_at = status.get("updated_at")
            age_seconds = _seconds_since(updated_at)
            if age_seconds is not None and age_seconds <= 120:
                return True
    return bool(_find_python_pids(process_match))


def _stop_processes(pids: list[int]) -> None:
    for pid in pids:
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {int(pid)} -Force"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,
                **_subprocess_kwargs(),
            )
        except Exception:
            continue


def _wait_for_url(url: str, *, timeout_seconds: int = 30) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            _json_request(url, timeout=3.0)
            return True
        except Exception:
            time.sleep(1.0)
    return False


def _start_stack(*, include_client: bool = True) -> None:
    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(ROOT / "scripts" / "start_stack.ps1"),
    ]
    if include_client:
        command.append("-IncludeClient")
    subprocess.run(
        command,
        cwd=str(ROOT),
        timeout=90,
        **_subprocess_kwargs(),
    )


def _pid_candidates(name: str, node: dict[str, Any] | None) -> list[int]:
    candidates: list[int] = []
    seen: set[int] = set()
    pid = _pid_from_file(name)
    if pid is not None:
        seen.add(pid)
        candidates.append(pid)
    process_match = str((node or {}).get("process_match") or "").strip()
    if process_match:
        for current_pid in _find_python_pids(process_match):
            if current_pid not in seen:
                seen.add(current_pid)
                candidates.append(current_pid)
    return candidates


def _listening_port_pids(port: int) -> list[int]:
    try:
        output = _run_powershell(
            "Get-NetTCPConnection -State Listen -LocalPort "
            f"{int(port)} -ErrorAction SilentlyContinue | ForEach-Object {{ $_.OwningProcess }}"
        )
    except Exception:
        return []
    return [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]


def _terminate_runtime_stack(runtime: dict[str, Any]) -> None:
    targets: set[int] = set()
    for name in ("client", "service", "dashboard", "session_supervisor"):
        node = runtime.get(name) or {}
        targets.update(_pid_candidates(name, node))
    targets.update(_listening_port_pids(5000))
    targets.update(_listening_port_pids(8787))
    if targets:
        _stop_processes(sorted(targets))
        time.sleep(2.0)
    for name in ("client", "service", "dashboard", "session_supervisor"):
        path = _pid_file(name)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def _powershell_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _spawn_hard_restart() -> None:
    patterns = [
        "http_game_service.py",
        "training_dashboard.py",
        "training_watchdog.py",
        "training_session_supervisor.py",
        "train_http_cli_rl.py",
    ]
    pattern_list = ",".join(_powershell_literal(item) for item in patterns)
    start_stack_path = _powershell_literal(str(ROOT / "scripts" / "start_stack.ps1"))
    current_pid = int(os.getpid())
    script = f"""
$patterns = @({pattern_list})
$targets = @()
$targets += Get-CimInstance Win32_Process |
  Where-Object {{ $_.CommandLine -and ($_.Name -in @('python.exe','pythonw.exe')) }} |
  Where-Object {{ $cmd = $_.CommandLine; ($patterns | Where-Object {{ $cmd -like "*$_*" }}).Count -gt 0 }} |
  Select-Object -ExpandProperty ProcessId
foreach ($port in 5000,8787) {{
  $owner = netstat -ano -p tcp | Select-String 'LISTENING' | Where-Object {{ $_.Line -match "[:\\.]$port\\s+" }} | Select-Object -First 1
  if ($owner) {{
    $parts = ($owner.Line -replace '\\s+', ' ').Trim().Split(' ')
    $targets += [int]$parts[-1]
  }}
}}
$targets += {current_pid}
$targets = $targets | Where-Object {{ $_ }} | Sort-Object -Unique
if ($targets) {{
  Stop-Process -Id $targets -Force -ErrorAction SilentlyContinue
}}
Start-Sleep -Seconds 3
Start-Process -WindowStyle Hidden -FilePath 'powershell.exe' -ArgumentList @('-ExecutionPolicy','Bypass','-File',{start_stack_path}) | Out-Null
"""
    subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", script],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        **_subprocess_kwargs(),
    )


def _load_state() -> dict[str, Any]:
    state = _read_json(
        STATE_PATH,
        {
            "watch_started_at": _now_iso(),
            "processed_history_games": [],
            "processed_incidents": [],
            "last_restart_at": None,
            "training_missing_since": None,
            "restart_count": 0,
            "restart_history": [],
            "current_num_envs": 0,
            "current_run_id": None,
            "current_run_dir": None,
            "current_run_boundary_at": None,
            "active_floor_progress": {},
        },
    )
    if not isinstance(state, dict):
        state = {}
    state.setdefault("watch_started_at", _now_iso())
    state.setdefault("processed_history_games", [])
    state.setdefault("processed_incidents", [])
    state.setdefault("last_restart_at", None)
    state.setdefault("training_missing_since", None)
    state.setdefault("restart_count", 0)
    state.setdefault("restart_history", [])
    state.setdefault("current_num_envs", 0)
    state.setdefault("current_run_id", None)
    state.setdefault("current_run_dir", None)
    state.setdefault("current_run_boundary_at", None)
    state.setdefault("active_floor_progress", {})
    return state


def _trim_state(state: dict[str, Any]) -> dict[str, Any]:
    state["processed_history_games"] = list(dict.fromkeys(state.get("processed_history_games") or []))[-2000:]
    state["processed_incidents"] = list(dict.fromkeys(state.get("processed_incidents") or []))[-2000:]
    raw_history = state.get("restart_history") or []
    history: list[Any] = []
    for item in raw_history:
        if isinstance(item, dict):
            history.append(
                {
                    "at": str(item.get("at") or ""),
                    "reason": str(item.get("reason") or ""),
                }
            )
        else:
            history.append(str(item))
    state["restart_history"] = history[-50:]
    state["current_num_envs"] = max(1, _safe_int(state.get("current_num_envs"), _configured_num_envs(_load_runtime_config())))
    active_floor_progress = state.get("active_floor_progress") or {}
    if isinstance(active_floor_progress, dict):
        items = list(active_floor_progress.items())[-128:]
        state["active_floor_progress"] = {str(key): value for key, value in items}
    else:
        state["active_floor_progress"] = {}
    return state


def _incident_key(incident: dict[str, Any]) -> str:
    return f"{incident.get('type')}::{incident.get('run_id')}::{incident.get('game_id') or incident.get('slot') or 'global'}"


def _history_boundary(state: dict[str, Any]) -> datetime | None:
    return _parse_iso(state.get("current_run_boundary_at")) or _parse_iso(state.get("watch_started_at"))


def _run_scope_age_seconds(state: dict[str, Any]) -> float | None:
    return _seconds_since(state.get("current_run_boundary_at") or state.get("watch_started_at"))


def _event_at_or_after_boundary(values: list[Any], boundary: datetime | None) -> bool:
    if boundary is None:
        return True
    for value in values:
        dt = _parse_iso(value)
        if dt is not None and dt >= boundary:
            return True
    return False


def _refresh_run_scope(snapshot: dict[str, Any], state: dict[str, Any]) -> None:
    training = snapshot.get("training") or {}
    run_dir = str(snapshot.get("run_dir") or "")
    run_id = str(training.get("run_id") or Path(run_dir).name or "")
    num_envs = _safe_int(training.get("num_envs"), 0)
    if num_envs <= 0:
        num_envs = len(snapshot.get("active_slots") or [])
    if num_envs > 0:
        state["current_num_envs"] = num_envs
    if not run_id:
        return
    if state.get("current_run_id") == run_id and state.get("current_run_dir") == run_dir:
        return
    state["current_run_id"] = run_id
    state["current_run_dir"] = run_dir
    state["current_run_boundary_at"] = _now_iso()
    state["processed_history_games"] = []
    state["active_floor_progress"] = {}
    state["training_missing_since"] = None
    print(f"[watchdog] monitoring current run only: run_id={run_id}", flush=True)


def _slot_progress_key(slot: dict[str, Any]) -> str:
    slot_name = str(slot.get("slot") or "")
    seed = str(slot.get("seed") or "")
    game_id = str(slot.get("game_id") or "")
    return slot_name or game_id or seed


def _floor_age_seconds(slot: dict[str, Any], state: dict[str, Any]) -> float | None:
    progress_by_slot = state.setdefault("active_floor_progress", {})
    if not isinstance(progress_by_slot, dict):
        progress_by_slot = {}
        state["active_floor_progress"] = progress_by_slot

    key = _slot_progress_key(slot)
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
    floor_since = current.get("floor_since") or observed_at
    return _seconds_since(floor_since)


def _session_details_path(run_dir: Path, game_id: str) -> Path:
    return run_dir / "dashboard" / "sessions" / f"{game_id}.json"


def _slot_history_paths(run_dir: Path) -> list[Path]:
    slots_dir = run_dir / "dashboard" / "slots"
    if not slots_dir.exists():
        return []
    return sorted(slots_dir.glob("slot_*.history.jsonl"))


def _read_history_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _collect_recent_errors(run_dir: Path, game_id: str) -> list[dict[str, Any]]:
    details = _read_json(_session_details_path(run_dir, game_id), {})
    if not isinstance(details, dict):
        return []
    trace = details.get("trace") or []
    if not isinstance(trace, list):
        return []
    errors = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "").lower() != "error":
            continue
        errors.append(
            {
                "step": item.get("step"),
                "action": item.get("action"),
                "action_detail": item.get("action_detail"),
                "reward": item.get("reward"),
            }
        )
    return errors[-5:]


def _record_incident(state: dict[str, Any], incident: dict[str, Any]) -> None:
    key = _incident_key(incident)
    if key in set(state.get("processed_incidents") or []):
        return
    INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _now().strftime("%Y%m%d_%H%M%S")
    safe_seed = str(incident.get("seed") or incident.get("game_id") or "incident").replace(":", "_")
    path = INCIDENTS_DIR / f"{timestamp}_{incident.get('type')}_{safe_seed}.json"
    payload = dict(incident)
    payload["recorded_at"] = _now_iso()
    _write_json(path, payload)
    _post_dashboard_best_effort("/api/telemetry/runtime/incident", {**payload, "source": "watchdog", "path": str(path)})
    state.setdefault("processed_incidents", []).append(key)
    print(f"[watchdog] incident recorded: {incident.get('type')} seed={incident.get('seed')} path={path}", flush=True)


def _detect_active_slot_incidents(snapshot: dict[str, Any], config: WatchdogConfig, state: dict[str, Any]) -> list[dict[str, Any]]:
    incidents: list[dict[str, Any]] = []
    run_id = str((snapshot.get("training") or {}).get("run_id") or "")
    boundary = _history_boundary(state)
    active_slots = [slot for slot in (snapshot.get("active_slots") or []) if isinstance(slot, dict)]
    active_progress = state.get("active_floor_progress") or {}
    active_keys = {_slot_progress_key(slot) for slot in active_slots}
    if isinstance(active_progress, dict):
        for existing_key in list(active_progress.keys()):
            if existing_key not in active_keys:
                active_progress.pop(existing_key, None)

    for slot in active_slots:
        if not isinstance(slot, dict):
            continue
        if not _event_at_or_after_boundary([slot.get("updated_at"), slot.get("started_at")], boundary):
            continue
        game_id = str(slot.get("game_id") or "")
        seed = str(slot.get("seed") or "")
        floor = max(1, _safe_int(slot.get("floor"), 1))
        elapsed_seconds = _seconds_since(slot.get("started_at"))
        floor_age_seconds = _floor_age_seconds(slot, state)
        updated_age_seconds = _seconds_since(slot.get("updated_at"))
        stagnant_steps = _safe_int(slot.get("stagnant_steps"), 0)
        timeout_seconds = config.per_floor_timeout_seconds
        common = {
            "run_id": run_id,
            "slot": slot.get("slot"),
            "game_id": game_id,
            "seed": seed,
            "floor": floor,
            "hp": slot.get("hp"),
            "max_hp": slot.get("max_hp"),
            "stagnant_steps": stagnant_steps,
            "started_at": slot.get("started_at"),
            "updated_at": slot.get("updated_at"),
            "elapsed_seconds": elapsed_seconds,
            "floor_age_seconds": floor_age_seconds,
        }
        if floor_age_seconds is not None and floor_age_seconds > timeout_seconds:
            incidents.append(
                {
                    **common,
                    "type": "slow_active_game",
                    "reason": f"floor {floor} stalled for {floor_age_seconds:.0f}s, exceeded threshold {timeout_seconds}s",
                    "restart_required": False,
                }
            )
        if updated_age_seconds is not None and updated_age_seconds > config.stale_slot_seconds:
            incidents.append(
                {
                    **common,
                    "type": "stale_active_slot",
                    "reason": f"slot update age {updated_age_seconds:.0f}s exceeded threshold {config.stale_slot_seconds}s",
                    "restart_required": False,
                }
            )
        if stagnant_steps >= config.stagnant_steps_threshold:
            incidents.append(
                {
                    **common,
                    "type": "stagnant_active_slot",
                    "reason": f"slot stagnant_steps={stagnant_steps} exceeded threshold {config.stagnant_steps_threshold}",
                    "restart_required": False,
                }
            )
        if elapsed_seconds is not None and elapsed_seconds > config.max_uptime_seconds:
            incidents.append(
                {
                    **common,
                    "type": "overlong_active_slot",
                    "reason": f"slot uptime {elapsed_seconds:.0f}s exceeded threshold {config.max_uptime_seconds}s",
                    "restart_required": False,
                }
            )
    return incidents


def _detect_history_incidents(snapshot: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    incidents: list[dict[str, Any]] = []
    run_dir_raw = snapshot.get("run_dir")
    if not run_dir_raw:
        return incidents
    run_dir = Path(str(run_dir_raw))
    if not run_dir.exists():
        return incidents

    processed_games = set(str(item) for item in state.get("processed_history_games") or [])
    run_id = str((snapshot.get("training") or {}).get("run_id") or run_dir.name)
    boundary = _history_boundary(state)

    for history_path in _slot_history_paths(run_dir):
        for row in _read_history_rows(history_path)[-200:]:
            if not isinstance(row, dict):
                continue
            game_id = str(row.get("game_id") or "")
            if not game_id or game_id in processed_games:
                continue
            if not _event_at_or_after_boundary(
                [
                    row.get("recorded_at"),
                    row.get("finished_at"),
                    row.get("updated_at"),
                    row.get("started_at"),
                ],
                boundary,
            ):
                continue
            processed_games.add(game_id)
            state.setdefault("processed_history_games", []).append(game_id)

            victory = bool(row.get("victory"))
            final_hp = _safe_int(row.get("hp"), -1)
            truncated = bool(row.get("truncated"))
            terminated = bool(row.get("terminated"))
            termination_reason = str(row.get("termination_reason") or "")
            anomaly_flags = [str(item).strip() for item in (row.get("anomaly_flags") or row.get("health_flags") or []) if str(item).strip()]
            flags_display = str(row.get("flags_display") or "")
            common = {
                "type": "completed_session_issue",
                "run_id": run_id,
                "slot": row.get("slot"),
                "game_id": game_id,
                "seed": row.get("seed"),
                "floor": row.get("floor"),
                "hp": row.get("hp"),
                "max_hp": row.get("max_hp"),
                "terminated": terminated,
                "truncated": truncated,
                "victory": victory,
                "termination_reason": termination_reason,
                "anomaly_flags": anomaly_flags,
                "flags_display": flags_display,
                "episode_steps": row.get("episode_steps"),
                "recorded_at": row.get("recorded_at"),
                "recent_errors": _collect_recent_errors(run_dir, game_id),
                "restart_required": False,
            }
            if not victory and final_hp > 0:
                incidents.append(
                    {
                        **common,
                        "type": "ended_with_hp",
                        "reason": f"session ended with hp={final_hp} without victory",
                        "restart_required": False,
                    }
                )
            if truncated and not victory:
                incidents.append(
                    {
                        **common,
                        "type": "truncated_session",
                        "reason": f"session truncated without victory (termination_reason={termination_reason or 'unknown'})",
                        "restart_required": False,
                    }
                )
            deadlock_markers = [termination_reason.lower(), flags_display.lower(), *[flag.lower() for flag in anomaly_flags]]
            if any("deadlock" in marker or "protocol_deadlock" in marker or "stuck" in marker for marker in deadlock_markers if marker):
                incidents.append(
                    {
                    **common,
                    "type": "deadlock_session",
                    "reason": f"session ended with deadlock markers: termination_reason={termination_reason or '-'} flags={flags_display or anomaly_flags}",
                    "restart_required": True,
                }
            )
    return incidents


def _restart_allowed(state: dict[str, Any], config: WatchdogConfig) -> bool:
    last_restart_at = _parse_iso(state.get("last_restart_at"))
    if last_restart_at is None:
        return True
    return (_now() - last_restart_at).total_seconds() >= config.restart_cooldown_seconds


def _stop_service(runtime: dict[str, Any]) -> None:
    service = runtime.get("service") or {}
    process_match = str(service.get("process_match") or "http_game_service.py")
    pids = _find_python_pids(process_match)
    if pids:
        _stop_processes(pids)


def _restart_runtime(runtime: dict[str, Any], config: WatchdogConfig, reason: str, state: dict[str, Any]) -> None:
    if not _restart_allowed(state, config):
        print(f"[watchdog] restart skipped due to cooldown: {reason}", flush=True)
        return

    state["last_restart_at"] = _now_iso()
    state["restart_count"] = _safe_int(state.get("restart_count"), 0) + 1
    history = list(state.get("restart_history") or [])
    history.append({"at": state["last_restart_at"], "reason": reason})
    state["restart_history"] = history
    desired_num_envs = _recommended_num_envs(runtime, state)
    state["current_num_envs"] = desired_num_envs
    _write_json(STATE_PATH, _trim_state(state))
    print(f"[watchdog] restarting full stack in-process: {reason}", flush=True)
    try:
        _terminate_runtime_stack(runtime)
        _clear_pid = _pid_file("watchdog")
        if _clear_pid.exists():
            try:
                _clear_pid.unlink()
            except OSError:
                pass
        _start_stack(include_client=True)
        dashboard_ok = _wait_for_url(f"{_dashboard_base_url()}/api/state", timeout_seconds=45)
        service_ok = _wait_for_url(f"{_service_base_url(runtime)}/health", timeout_seconds=45)
        print(
            f"[watchdog] restart completed: dashboard_ok={dashboard_ok} service_ok={service_ok}",
            flush=True,
        )
    except Exception as exc:
        print(f"[watchdog] in-process restart failed, falling back to hard restart: {exc}", flush=True)
        _spawn_hard_restart()
    raise SystemExit(0)


def _handle_restartable_incidents(
    runtime: dict[str, Any],
    config: WatchdogConfig,
    state: dict[str, Any],
    incidents: list[dict[str, Any]],
) -> None:
    restartable = [incident for incident in incidents if bool(incident.get("restart_required"))]
    if not restartable:
        return
    first = restartable[0]
    reason = str(first.get("reason") or first.get("type") or "incident")
    seed = str(first.get("seed") or "")
    if seed:
        reason = f"{reason} | seed={seed}"
    _restart_runtime(runtime, config, reason, state)


def _ensure_training_running(runtime: dict[str, Any], snapshot: dict[str, Any], config: WatchdogConfig, state: dict[str, Any]) -> None:
    training = snapshot.get("training") or {}
    status = str(training.get("status") or "").lower()
    active_slots = snapshot.get("active_slots") or []
    updated_age_seconds = _seconds_since(training.get("updated_at"))
    client = runtime.get("client") or {}
    client_running = _is_managed_process_running(client)

    fresh_dashboard_activity = updated_age_seconds is not None and updated_age_seconds < max(config.no_training_grace_seconds, 30)
    if (status == "running" or active_slots) and fresh_dashboard_activity:
        num_envs = _safe_int(training.get("num_envs"), 0)
        if num_envs <= 0:
            num_envs = len(active_slots)
        if num_envs > 0:
            state["current_num_envs"] = num_envs
        state["training_missing_since"] = None
        return

    if not client_running:
        since = _parse_iso(state.get("training_missing_since"))
        if since is None:
            state["training_missing_since"] = _now_iso()
            since = _parse_iso(state.get("training_missing_since"))
        missing_age_seconds = (_now() - since).total_seconds() if since is not None else 0.0
        should_expect_training = status in {"running", "paused", "stopping"} or bool(active_slots)
        if should_expect_training and fresh_dashboard_activity and missing_age_seconds < max(config.no_training_grace_seconds, 30):
            return
        if should_expect_training or (
            updated_age_seconds is not None and updated_age_seconds >= config.no_training_grace_seconds
        ):
            print(
                f"[watchdog] training client missing while status={status or 'unknown'} active_slots={len(active_slots)} updated_age={updated_age_seconds} missing_age={missing_age_seconds}",
                flush=True,
            )
            _restart_runtime(runtime, config, "training_client_missing", state)
        return

    if status == "running" or active_slots:
        num_envs = _safe_int(training.get("num_envs"), 0)
        if num_envs <= 0:
            num_envs = len(active_slots)
        if num_envs > 0:
            state["current_num_envs"] = num_envs
        state["training_missing_since"] = None
        return

    since = _parse_iso(state.get("training_missing_since"))
    if since is None:
        state["training_missing_since"] = _now_iso()
        return

    if (_now() - since).total_seconds() < config.no_training_grace_seconds:
        return

    print("[watchdog] no active training detected; starting client", flush=True)
    try:
        desired_num_envs = _recommended_num_envs(runtime, state, snapshot)
        state["current_num_envs"] = desired_num_envs
        result = _post_dashboard(
            "/api/control/start",
            {"total_timesteps": config.total_timesteps, "num_envs": desired_num_envs},
        )
        print(f"[watchdog] start result: {result}", flush=True)
    except Exception as exc:
        print(f"[watchdog] dashboard start failed, falling back to stack restart: {exc}", flush=True)
        _restart_runtime(_load_runtime_config(), config, "training_missing", state)
    state["training_missing_since"] = None


def _ensure_runtime_healthy(runtime: dict[str, Any], snapshot: dict[str, Any] | None, config: WatchdogConfig, state: dict[str, Any]) -> None:
    dashboard = runtime.get("dashboard") or {}
    if not _is_managed_process_running(dashboard):
        _restart_runtime(runtime, config, "dashboard_process_missing", state)
        return

    try:
        health = _service_health(runtime)
    except Exception as exc:
        _restart_runtime(runtime, config, f"service_unavailable: {exc}", state)
        return

    if str(health.get("status") or "").lower() != "healthy":
        _restart_runtime(runtime, config, f"service_unhealthy: {health}", state)
        return

    training = (snapshot or {}).get("training") or {}
    expected_envs = _safe_int(training.get("num_envs"), 0)
    active_slots = [slot for slot in ((snapshot or {}).get("active_slots") or []) if isinstance(slot, dict) and bool(slot.get("active", False))]
    active_games = _safe_int(health.get("active_games"), 0)
    busy_workers = _safe_int(health.get("busy_workers"), 0)
    run_scope_age_seconds = _run_scope_age_seconds(state)
    startup_grace_active = run_scope_age_seconds is not None and run_scope_age_seconds < 120
    if busy_workers > active_games + 1:
        if startup_grace_active and active_games == 0:
            return
        _restart_runtime(
            runtime,
            config,
            f"worker_leak_detected: active_games={active_games} busy_workers={busy_workers}",
            state,
        )
        return
    if expected_envs > 1 and active_games < max(1, expected_envs - 1) and len(active_slots) < max(1, expected_envs - 1):
        if startup_grace_active:
            return
        _restart_runtime(
            runtime,
            config,
            f"parallelism_collapsed: expected_envs={expected_envs} active_games={active_games} active_slots={len(active_slots)}",
            state,
        )
        return

    session_supervisor = runtime.get("session_supervisor") or {}
    if not _is_managed_process_running(session_supervisor):
        print("[watchdog] session_supervisor missing; starting session_supervisor only", flush=True)
        _start_managed_process("session_supervisor", session_supervisor, runtime)


def _write_status(snapshot: dict[str, Any] | None, incidents: list[dict[str, Any]], state: dict[str, Any]) -> None:
    payload = {
        "updated_at": _now_iso(),
        "run_id": ((snapshot or {}).get("training") or {}).get("run_id") if snapshot else None,
        "active_slots": len((snapshot or {}).get("active_slots") or []),
        "incidents_detected": len(incidents),
        "last_restart_at": state.get("last_restart_at"),
        "restart_count": state.get("restart_count"),
        "target_num_envs": state.get("current_num_envs"),
    }
    _write_json(STATUS_PATH, payload)
    _post_dashboard_best_effort("/api/telemetry/runtime/status", {**payload, "source": "watchdog"})


def main() -> None:
    WATCHDOG_ROOT.mkdir(parents=True, exist_ok=True)
    INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    print("[watchdog] started", flush=True)

    while True:
        state = _load_state()
        config = _watchdog_config()
        runtime = _load_runtime_config()
        incidents: list[dict[str, Any]] = []
        snapshot: dict[str, Any] | None = None

        try:
            snapshot = _dashboard_state()
            if not isinstance(snapshot, dict):
                raise RuntimeError("dashboard returned non-dict payload")

            _refresh_run_scope(snapshot, state)
            incidents.extend(_detect_active_slot_incidents(snapshot, config, state))
            incidents.extend(_detect_history_incidents(snapshot, state))

            for incident in incidents:
                _record_incident(state, incident)

            _handle_restartable_incidents(runtime, config, state, incidents)
            _ensure_runtime_healthy(runtime, snapshot, config, state)
            _ensure_training_running(runtime, snapshot, config, state)
        except (URLError, TimeoutError, OSError, RuntimeError, json.JSONDecodeError) as exc:
            incident = {
                "type": "dashboard_unavailable",
                "run_id": None,
                "game_id": None,
                "seed": None,
                "reason": str(exc),
                "restart_required": True,
            }
            _record_incident(state, incident)
            _restart_runtime(runtime, config, f"dashboard_unavailable: {exc}", state)
            incidents = [incident]
        except Exception as exc:
            print(f"[watchdog] unexpected error: {exc}", flush=True)
            incidents = []
        finally:
            _trim_state(state)
            _write_json(STATE_PATH, state)
            _write_status(snapshot, incidents, state)

        time.sleep(config.poll_seconds)


if __name__ == "__main__":
    main()
