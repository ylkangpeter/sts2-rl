# -*- coding: utf-8 -*-
"""Lightweight telemetry writers for training dashboards."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _dashboard_base_url() -> str:
    return str(os.environ.get("ST2RL_DASHBOARD_URL") or "http://127.0.0.1:8787").rstrip("/")


def _run_metadata(root: Path | None, model_run_dir: Path | None = None) -> dict[str, str]:
    if root is None:
        return {}
    dashboard_dir = root
    run_dir = model_run_dir if model_run_dir is not None else dashboard_dir.parent
    experiment_dir = run_dir.parent
    return {
        "dashboard_dir": str(dashboard_dir),
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "experiment_name": experiment_dir.name if experiment_dir.name else "",
    }


def _post_dashboard(path: str, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        f"{_dashboard_base_url()}{path}",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=0.75) as response:
            response.read()
    except (OSError, TimeoutError, URLError):
        return


class SlotTelemetry:
    """Per-slot telemetry persisted as small JSON/JSONL files."""

    def __init__(self, root_dir: str | None, slot_id: int, *, model_run_dir: str | None = None):
        self.root = Path(root_dir) if root_dir else None
        self.slot_id = slot_id
        self.current_path = None
        self.history_path = None
        self.run_meta = _run_metadata(self.root, Path(model_run_dir) if model_run_dir else None)
        self._last_current_write_ts = 0.0
        self._min_current_update_interval_seconds = 1.0
        if self.root:
            slots_dir = self.root / "slots"
            slots_dir.mkdir(parents=True, exist_ok=True)
            self.current_path = slots_dir / f"slot_{slot_id:02d}.json"
            self.history_path = slots_dir / f"slot_{slot_id:02d}.history.jsonl"

    def write_current(self, payload: dict[str, Any], *, force: bool = False) -> None:
        if not self.current_path:
            return
        now_ts = datetime.now().timestamp()
        if not force and (now_ts - self._last_current_write_ts) < self._min_current_update_interval_seconds:
            return
        data = dict(payload)
        data.update(self.run_meta)
        data["updated_at"] = _now_iso()
        self.current_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        _post_dashboard("/api/telemetry/slot/current", data)
        self._last_current_write_ts = now_ts

    def append_history(self, payload: dict[str, Any]) -> None:
        if not self.history_path:
            return
        data = dict(payload)
        data.update(self.run_meta)
        data["recorded_at"] = _now_iso()
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")
        _post_dashboard("/api/telemetry/slot/history", data)


class SessionLeaderboardStore:
    """Persist detailed leaderboard sessions for dashboard drill-down."""

    def __init__(self, root_dir: str | None, *, limit: int = 50, model_run_dir: str | None = None):
        self.root = Path(root_dir) if root_dir else None
        self.limit = limit
        self.sessions_dir = None
        self.index_path = None
        self.best_path = None
        if self.root:
            self.sessions_dir = self.root / "sessions"
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            self.index_path = self.root / "session_leaderboard.json"
            self.best_path = self.root / "best_session.json"
        self.run_meta = _run_metadata(self.root, Path(model_run_dir) if model_run_dir else None)

    def _read_index(self) -> list[dict[str, Any]]:
        if not self.index_path or not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _write_index(self, rows: list[dict[str, Any]]) -> None:
        if not self.index_path:
            return
        self.index_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
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

    @classmethod
    def _global_floor(cls, act: Any, floor: Any) -> int:
        act_value = max(1, cls._safe_int(act, 1))
        floor_value = max(0, cls._safe_int(floor, 0))
        if act_value <= 1:
            return floor_value
        if act_value == 2:
            return 18 + floor_value
        if act_value == 3:
            return 35 + floor_value
        return 51 + floor_value + max(0, act_value - 4) * 17

    @classmethod
    def _global_floor_from_progress(cls, progress: Any) -> int:
        value = cls._safe_int(progress, 0)
        if value <= 0:
            return 0
        if value < 100:
            return value
        act = value // 100 + 1
        floor = value % 100
        return cls._global_floor(act, floor)

    @classmethod
    def _entry_global_floor(cls, entry: dict[str, Any]) -> int:
        explicit = cls._safe_int(entry.get("max_global_floor"), 0)
        if explicit > 0:
            return explicit
        progress_floor = cls._global_floor_from_progress(entry.get("max_progress"))
        if progress_floor > 0:
            return progress_floor
        max_global_act = cls._safe_int(entry.get("max_global_act"), 0)
        max_global_act_floor = cls._safe_int(entry.get("max_global_act_floor"), 0)
        if max_global_act > 0 and max_global_act_floor > 0:
            return cls._global_floor(max_global_act, max_global_act_floor)
        act = cls._safe_int(entry.get("act"), 0)
        floor = cls._safe_int(entry.get("final_floor") or entry.get("floor"), 0)
        if act > 0 and floor > 0:
            return cls._global_floor(act, floor)
        return cls._safe_int(entry.get("max_floor"), 0)

    @classmethod
    def _health_flags(cls, entry: dict[str, Any]) -> list[str]:
        flags: list[str] = []
        victory = bool(entry.get("victory"))
        final_hp = cls._safe_int(entry.get("final_hp"), 0)
        if not victory and final_hp > 0:
            flags.append("nonzero_hp_end")
        if bool(entry.get("truncated")) and not bool(entry.get("terminated")):
            flags.append("truncated_without_game_over")
        return flags

    @staticmethod
    def _score(entry: dict[str, Any]) -> tuple[Any, ...]:
        suspicious = 1 if SessionLeaderboardStore._health_flags(entry) else 0
        return (
            -suspicious,
            1 if entry.get("victory") else 0,
            SessionLeaderboardStore._entry_global_floor(entry),
            int(entry.get("max_act") or entry.get("act", 0)),
            1 if entry.get("act1_boss_clear") else 0,
            round(float(entry.get("episode_reward", 0.0)), 3),
            -int(entry.get("episode_steps", 0)),
        )

    def _session_path(self, game_id: str) -> Path | None:
        if not self.sessions_dir or not game_id:
            return None
        return self.sessions_dir / f"{game_id}.json"

    def write_session_if_ranked(self, summary: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
        if not self.root or not summary.get("game_id"):
            return {"ranked": False, "rank": None, "best": False}

        leaderboard = [row for row in self._read_index() if row.get("game_id") != summary.get("game_id")]
        leaderboard.append(dict(summary))
        leaderboard.sort(key=self._score, reverse=True)
        leaderboard = leaderboard[: self.limit]
        self._write_index(leaderboard)

        ranked_ids = {row.get("game_id") for row in leaderboard}
        session_path = self._session_path(str(summary["game_id"]))
        ranked = str(summary["game_id"]) in ranked_ids
        if ranked and session_path:
            payload = dict(details)
            payload["saved_at"] = _now_iso()
            session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if self.sessions_dir:
            for path in self.sessions_dir.glob("*.json"):
                if path.stem not in ranked_ids:
                    try:
                        path.unlink()
                    except OSError:
                        pass

        best = bool(leaderboard and leaderboard[0].get("game_id") == summary.get("game_id"))
        if best and self.best_path:
            best_payload = dict(details)
            if ranked:
                best_payload["saved_at"] = _now_iso()
            self.best_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        rank = None
        for index, row in enumerate(leaderboard, start=1):
            if row.get("game_id") == summary.get("game_id"):
                rank = index
                break
        _post_dashboard(
            "/api/telemetry/session",
            {
                **self.run_meta,
                "summary": dict(summary),
                "details": dict(details),
                "leaderboard": {
                    "ranked": ranked,
                    "rank": rank,
                    "best": best,
                },
            },
        )
        return {"ranked": ranked, "rank": rank, "best": best}

    def read_session(self, game_id: str) -> dict[str, Any]:
        path = self._session_path(game_id)
        if not path or not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


class TrainingControl:
    """File-based training control used by the dashboard and trainer."""

    def __init__(self, root_dir: str | None):
        self.root = Path(root_dir) if root_dir else None
        self.path = self.root / "control.json" if self.root else None

    def read(self) -> dict[str, Any]:
        if not self.path or not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def write(self, command: str, extra: dict[str, Any] | None = None) -> None:
        if not self.path:
            return
        data = {"command": command, "updated_at": _now_iso()}
        if extra:
            data.update(extra)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self) -> None:
        if self.path and self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass


class DashboardControlClient:
    """Fetch runtime control commands from the dashboard over HTTP."""

    def __init__(self, run_id: str | None, *, poll_interval_seconds: float = 0.5):
        self.run_id = str(run_id or "").strip()
        self.poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self._last_poll_ts = 0.0
        self._cached: dict[str, Any] = {}

    def read(self, *, force: bool = False) -> dict[str, Any]:
        now_ts = datetime.now().timestamp()
        if not force and self._cached and (now_ts - self._last_poll_ts) < self.poll_interval_seconds:
            return dict(self._cached)
        params = f"?run_id={self.run_id}" if self.run_id else ""
        request = Request(f"{_dashboard_base_url()}/api/runtime/control{params}", method="GET")
        try:
            with urlopen(request, timeout=0.5) as response:
                raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            self._cached = payload if isinstance(payload, dict) else {}
            self._last_poll_ts = now_ts
        except (OSError, TimeoutError, URLError, json.JSONDecodeError):
            return dict(self._cached)
        return dict(self._cached)


class TrainingStatusWriter:
    """Writes compact training progress snapshots for dashboard use."""

    def __init__(self, root_dir: str, payload: dict[str, Any], *, model_run_dir: str | None = None):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "training_status.json"
        self.payload = dict(payload)
        self.payload.update(_run_metadata(self.root, Path(model_run_dir) if model_run_dir else None))
        self._last_write_ts = 0.0
        self._last_status = None
        self._min_update_interval_seconds = 1.0

    def write(self, updates: dict[str, Any], *, force: bool = False) -> None:
        now_ts = datetime.now().timestamp()
        data = dict(self.payload)
        data.update(updates)
        status = data.get("status")
        if not force and status == self._last_status and (now_ts - self._last_write_ts) < self._min_update_interval_seconds:
            return
        data["updated_at"] = _now_iso()
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        _post_dashboard("/api/telemetry/training", data)
        self._last_write_ts = now_ts
        self._last_status = status
