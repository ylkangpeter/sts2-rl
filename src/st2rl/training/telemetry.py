# -*- coding: utf-8 -*-
"""Lightweight telemetry writers for training dashboards."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class SlotTelemetry:
    """Per-slot telemetry persisted as small JSON/JSONL files."""

    def __init__(self, root_dir: str | None, slot_id: int):
        self.root = Path(root_dir) if root_dir else None
        self.slot_id = slot_id
        self.current_path = None
        self.history_path = None
        if self.root:
            slots_dir = self.root / "slots"
            slots_dir.mkdir(parents=True, exist_ok=True)
            self.current_path = slots_dir / f"slot_{slot_id:02d}.json"
            self.history_path = slots_dir / f"slot_{slot_id:02d}.history.jsonl"

    def write_current(self, payload: dict[str, Any]) -> None:
        if not self.current_path:
            return
        data = dict(payload)
        data["updated_at"] = _now_iso()
        self.current_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_history(self, payload: dict[str, Any]) -> None:
        if not self.history_path:
            return
        data = dict(payload)
        data["recorded_at"] = _now_iso()
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")


class SessionLeaderboardStore:
    """Persist detailed leaderboard sessions for dashboard drill-down."""

    def __init__(self, root_dir: str | None, *, limit: int = 50):
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
            int(entry.get("max_floor", 0)),
            int(entry.get("act", 0)),
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
        if best and self.best_path and session_path and session_path.exists():
            self.best_path.write_text(session_path.read_text(encoding="utf-8"), encoding="utf-8")

        rank = None
        for index, row in enumerate(leaderboard, start=1):
            if row.get("game_id") == summary.get("game_id"):
                rank = index
                break
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


class TrainingStatusWriter:
    """Writes compact training progress snapshots for dashboard use."""

    def __init__(self, root_dir: str, payload: dict[str, Any]):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "training_status.json"
        self.payload = dict(payload)

    def write(self, updates: dict[str, Any]) -> None:
        data = dict(self.payload)
        data.update(updates)
        data["updated_at"] = _now_iso()
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
