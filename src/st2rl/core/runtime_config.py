# -*- coding: utf-8 -*-
"""Shared runtime configuration loaders for the training stack."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_runtime_stack_config(root: Path) -> dict[str, Any]:
    config = {}
    for path in (
        root / "configs" / "runtime_stack.json",
        root / "configs" / "runtime_stack.local.json",
    ):
        config = deep_merge_dicts(config, load_json_config(path))
    return config


def default_train_config_path(root: Path) -> Path:
    return root / "configs" / "train_http_cli_rl.yaml"


def load_train_config(path: Path) -> dict[str, Any]:
    return load_yaml_config(path)


def load_training_launch_defaults(root: Path) -> dict[str, Any]:
    return train_launch_defaults(load_train_config(default_train_config_path(root)))


def train_launch_defaults(config: dict[str, Any]) -> dict[str, Any]:
    training = dict(config.get("training") or {})
    paths = dict(config.get("paths") or {})
    return {
        "total_timesteps": int(training.get("total_timesteps") or 1000000),
        "num_envs": int(training.get("num_envs") or 1),
        "vec_env": str(training.get("vec_env") or "threaded").lower(),
        "save_freq": int(training.get("save_freq") or 0),
        "save_dir": str(paths.get("save_dir") or "./models/http_cli_rl"),
        "experiment_name": str(paths.get("experiment_name") or "baseline_reward_v1"),
        "model_name": str(paths.get("model_name") or "ppo_http_cli_rl"),
    }


def runtime_base_url(root: Path, node_name: str, *, default_host: str, default_port: int) -> str:
    runtime = load_runtime_stack_config(root)
    node = runtime.get(node_name) or {}
    if not isinstance(node, dict):
        node = {}
    explicit = str(node.get("base_url") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    host = str(node.get("host") or default_host).strip() or default_host
    port = int(node.get("port") or default_port)
    return f"http://{host}:{port}"


def runtime_dashboard_base_url(root: Path) -> str:
    return runtime_base_url(root, "dashboard", default_host="127.0.0.1", default_port=8787)


def runtime_service_base_url(root: Path) -> str:
    return runtime_base_url(root, "service", default_host="127.0.0.1", default_port=5000)
