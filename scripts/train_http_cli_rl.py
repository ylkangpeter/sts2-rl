# -*- coding: utf-8 -*-
"""Train PPO on the formal HTTP CLI reinforcement learning environment."""

import argparse
import json
import zipfile
from pathlib import Path

from st2rl.core.runtime_config import default_train_config_path, load_train_config
from st2rl.training.trainer import UnifiedTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the HTTP CLI RL environment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_train_config_path(Path(__file__).resolve().parents[1]),
        help="Path to YAML training config.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional existing model path without .zip for continued training.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Optional override for total timesteps.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name override for the output directory.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Optional override for parallel environment count.",
    )
    parser.add_argument(
        "--vec-env",
        type=str,
        default=None,
        choices=["dummy", "subproc", "threaded"],
        help="Optional override for vectorized environment type.",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Force a brand new training run instead of auto-resuming the latest model in the same experiment.",
    )
    return parser.parse_args()


def _iter_history_rows(run_dir: Path):
    dashboard_slots = run_dir / "dashboard" / "slots"
    if not dashboard_slots.exists():
        return
    for history_path in dashboard_slots.glob("slot_*.history.jsonl"):
        try:
            for line in history_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
        except OSError:
            continue
        except json.JSONDecodeError:
            continue


def _run_quality_is_bad(run_dir: Path) -> bool:
    finished = 0
    anomaly_flags = 0
    floor_sum = 0.0
    ge17 = 0
    for row in _iter_history_rows(run_dir):
        if bool(row.get("active", False)):
            continue
        finished += 1
        floor = float(row.get("max_floor", row.get("floor", 0)) or 0.0)
        floor_sum += floor
        if floor >= 17:
            ge17 += 1
        flags = row.get("anomaly_flags") or []
        if isinstance(flags, list) and flags:
            anomaly_flags += 1
    if finished <= 0:
        return False
    avg_floor = floor_sum / max(1, finished)
    anomaly_ratio = anomaly_flags / max(1, finished)
    ge17_ratio = ge17 / max(1, finished)
    return (
        (finished >= 80 and avg_floor < 4.0)
        or (finished >= 200 and avg_floor < 9.0 and ge17_ratio < 0.03)
        or anomaly_ratio > 0.45
    )


def resolve_resume_model_path(config: dict, explicit_model_path: str | None, fresh_start: bool) -> str | None:
    if fresh_start:
        return None
    if explicit_model_path:
        return explicit_model_path

    paths = config.get("paths", {}) or {}
    save_dir = Path(str(paths.get("save_dir", "./models/unified")))
    experiment_name = str(paths.get("experiment_name") or paths.get("model_name") or "default")
    model_name = str(paths.get("model_name") or "ppo_sts2")
    experiment_dir = save_dir / experiment_name
    if not experiment_dir.exists():
        return None

    candidates = sorted(
        experiment_dir.glob(f"*/{model_name}*.zip"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        run_dir = candidate.parent
        if _run_quality_is_bad(run_dir):
            continue
        try:
            if candidate.stat().st_size <= 0:
                continue
            with zipfile.ZipFile(candidate) as archive:
                archive.testzip()
            return str(candidate.with_suffix(""))
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
            continue
    return None


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config)
    if args.experiment_name:
        config.setdefault("paths", {})
        config["paths"]["experiment_name"] = args.experiment_name
    if args.num_envs is not None:
        config.setdefault("training", {})
        config["training"]["num_envs"] = args.num_envs
    if args.vec_env is not None:
        config.setdefault("training", {})
        config["training"]["vec_env"] = args.vec_env
    model_path = resolve_resume_model_path(config, args.model_path, args.fresh_start)
    trainer = UnifiedTrainer(config)
    try:
        trainer.create_environment()
        trainer.create_model(model_path=model_path)
        trainer.train(total_timesteps=args.timesteps)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
