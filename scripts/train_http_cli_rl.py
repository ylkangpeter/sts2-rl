# -*- coding: utf-8 -*-
"""Train PPO on the formal HTTP CLI reinforcement learning environment."""

import argparse
from pathlib import Path

import yaml

from st2rl.training.trainer import UnifiedTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the HTTP CLI RL environment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("D:/github/st2rl/configs/train_http_cli_rl.yaml"),
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
        choices=["dummy", "subproc"],
        help="Optional override for vectorized environment type.",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Force a brand new training run instead of auto-resuming the latest model in the same experiment.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
    if not candidates:
        return None
    return str(candidates[0].with_suffix(""))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
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
    trainer.create_environment()
    trainer.create_model(model_path=model_path)
    trainer.train(total_timesteps=args.timesteps)


if __name__ == "__main__":
    main()
