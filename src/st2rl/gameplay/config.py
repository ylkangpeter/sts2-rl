# -*- coding: utf-8 -*-
"""Configuration models for reusable game flow logic."""

from dataclasses import dataclass


@dataclass(slots=True)
class FlowPolicyConfig:
    """Policy tuning knobs grouped by gameplay stage."""

    card_reward_take_probability: float = 0.85
    shop_buy_probability: float = 0.35
    shop_low_gold_threshold: int = 100
    shop_high_gold_threshold: int = 250
    card_select_skip_probability: float = 0.2
    rest_heal_threshold: float = 0.5


@dataclass(slots=True)
class FlowRunnerConfig:
    """Execution configuration for running full game flows."""

    character: str = "Ironclad"
    max_steps: int | None = None
    total_rounds: int = 50
    default_max_workers: int = 4
    health_check_retries: int = 2
    stuck_warn_threshold: int = 60
    stuck_abort_threshold: int = 140
    no_action_combat_abort_threshold: int = 12
    no_action_combat_proceed_threshold: int = 3
    step_delay_seconds: float = 0.03
    recovery_delay_seconds: float = 0.05
    initial_state_poll_attempts: int = 20
    initial_state_poll_interval_seconds: float = 0.2
