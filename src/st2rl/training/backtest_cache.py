# -*- coding: utf-8 -*-
"""SQLite-backed exact rollout cache for deterministic backtests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from st2rl.gameplay.types import FlowAction


SCHEMA_VERSION = 1


@dataclass(slots=True)
class CachedRollout:
    """A cached full-game rollout."""

    initial_state: dict[str, Any]
    transitions: list[dict[str, Any]]
    outcome: dict[str, Any]


def stable_json(value: Any) -> str:
    """Serialize JSON-like values deterministically."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_digest(value: Any) -> str:
    """Return a compact deterministic digest for JSON-like values."""
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def action_payload(action: FlowAction) -> dict[str, Any]:
    """Return a deterministic action payload."""
    return {"name": action.name, "args": dict(action.args or {})}


def action_hash(action: FlowAction) -> str:
    """Return a deterministic action hash."""
    return stable_digest(action_payload(action))


def state_hash(raw_state: dict[str, Any]) -> str:
    """Return a deterministic raw-state hash for exact replay checks."""
    return stable_digest(_strip_volatile(raw_state))


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_volatile(item)
            for key, item in value.items()
            if str(key) not in {"state_version", "generated_at", "updated_at"}
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


def source_digest(repo_root: Path) -> str:
    """Digest source files that affect deterministic flow-policy backtests."""
    paths = [
        repo_root / "scripts" / "backtest_flow_policy.py",
        repo_root / "src" / "st2rl" / "gameplay" / "heuristics.py",
        repo_root / "src" / "st2rl" / "gameplay" / "policy.py",
        repo_root / "src" / "st2rl" / "protocols" / "http_cli.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
        digest.update(b"\0")
    digest.update(_git_rev(repo_root).encode("utf-8"))
    cli_root = repo_root.parent / "sts2-cli"
    if cli_root.exists():
        digest.update(_git_rev(cli_root).encode("utf-8"))
    return digest.hexdigest()[:16]


def _git_rev(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "nogit"


class BacktestRolloutCache:
    """Exact full-rollout cache keyed by seed, character, and policy version."""

    def __init__(self, path: Path, *, namespace: str) -> None:
        self.path = path
        self.namespace = namespace
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.mismatches = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rollouts (
                    cache_key TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    seed TEXT NOT NULL,
                    character TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    step_count INTEGER NOT NULL,
                    max_floor INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rollouts_namespace_seed ON rollouts(namespace, seed)"
            )
            conn.commit()

    def key(self, *, seed: str, character: str) -> str:
        return stable_digest(
            {
                "schema": SCHEMA_VERSION,
                "namespace": self.namespace,
                "seed": seed,
                "character": character,
            }
        )

    def load(self, *, seed: str, character: str) -> CachedRollout | None:
        key = self.key(seed=seed, character=character)
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT payload_json FROM rollouts WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        try:
            payload = json.loads(str(row[0]))
            self.hits += 1
            return CachedRollout(
                initial_state=dict(payload["initial_state"]),
                transitions=list(payload["transitions"]),
                outcome=dict(payload["outcome"]),
            )
        except Exception:
            self.mismatches += 1
            return None

    def store(
        self,
        *,
        seed: str,
        character: str,
        initial_state: dict[str, Any],
        transitions: list[dict[str, Any]],
        outcome: dict[str, Any],
    ) -> None:
        if not transitions or not outcome.get("success"):
            return
        now = time.time()
        key = self.key(seed=seed, character=character)
        payload = {
            "schema": SCHEMA_VERSION,
            "namespace": self.namespace,
            "seed": seed,
            "character": character,
            "initial_state": initial_state,
            "transitions": transitions,
            "outcome": outcome,
        }
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO rollouts (
                    cache_key, namespace, seed, character, payload_json,
                    step_count, max_floor, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    step_count = excluded.step_count,
                    max_floor = excluded.max_floor,
                    updated_at = excluded.updated_at
                """,
                (
                    key,
                    self.namespace,
                    seed,
                    character,
                    stable_json(payload),
                    int(outcome.get("steps") or 0),
                    int(outcome.get("max_floor") or 0),
                    now,
                    now,
                ),
            )
            conn.commit()
        self.stores += 1

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "path": str(self.path),
            "namespace": self.namespace,
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "mismatches": self.mismatches,
        }
