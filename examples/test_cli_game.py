# -*- coding: utf-8 -*-
"""Run one full game through the reusable HTTP CLI flow runner."""

import argparse
import os

from st2rl.gameplay.config import FlowPolicyConfig, FlowRunnerConfig
from st2rl.gameplay.policy import SimpleFlowPolicy
from st2rl.gameplay.runner import FlowRunner
from st2rl.protocols.http_cli import HttpCliProtocol, HttpCliProtocolConfig
from st2rl.utils.logger import get_run_logger

TOTAL_ROUNDS = 50
DEFAULT_MAX_WORKERS = 4
CHARACTER = "Ironclad"
GAME_DIR = os.environ.get("STS2_GAME_DIR")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel HTTP CLI game flow tests.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Parallel worker count. Default: auto-detect, capped at {DEFAULT_MAX_WORKERS}.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help=f"Total test rounds. Default: {TOTAL_ROUNDS}.",
    )
    return parser.parse_args()


def build_runner() -> FlowRunner:
    logger = get_run_logger()
    runner_config = FlowRunnerConfig(
        character=CHARACTER,
        total_rounds=TOTAL_ROUNDS,
        default_max_workers=DEFAULT_MAX_WORKERS,
    )
    policy = SimpleFlowPolicy(FlowPolicyConfig())
    protocol = HttpCliProtocol(
        HttpCliProtocolConfig(
            game_dir=GAME_DIR,
        )
    )
    return FlowRunner(protocol=protocol, policy=policy, config=runner_config, logger=logger)


def test_cli_game_flow(worker_id: int = 0):
    """Compatibility wrapper for existing ad-hoc callers."""
    return build_runner().run_single_game(worker_id).to_dict()


def run_parallel_games(*, workers: int | None = None, total_rounds: int | None = None) -> None:
    build_runner().run_parallel_games(workers=workers, total_rounds=total_rounds)


if __name__ == "__main__":
    args = _parse_args()
    run_parallel_games(workers=args.workers, total_rounds=args.rounds)
