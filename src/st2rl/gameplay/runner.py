# -*- coding: utf-8 -*-
"""Generic full-game flow runner built on top of a protocol adapter."""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from st2rl.gameplay.config import FlowRunnerConfig
from st2rl.gameplay.policy import SimpleFlowPolicy
from st2rl.gameplay.types import FlowAction, FlowRunResult, FlowRunSummary, GameStateView
from st2rl.protocols.base import FlowProtocol

SEED_PREFIX_ENV = "STS2_HTTP_SEED_PREFIX"
FORCE_SEED_ENV = "STS2_HTTP_SEED"
WORKER_COUNT_ENV = "STS2_TEST_WORKERS"


class FlowRunner:
    """Run one or many full-game flow tests against a backend protocol."""

    def __init__(
        self,
        protocol: FlowProtocol,
        policy: SimpleFlowPolicy,
        config: FlowRunnerConfig | None = None,
        *,
        logger,
    ):
        self.protocol = protocol
        self.policy = policy
        self.config = config or FlowRunnerConfig()
        self.logger = logger

    def _build_seed(self, worker_id: int) -> str:
        forced_seed = os.environ.get(FORCE_SEED_ENV)
        if forced_seed:
            return f"{forced_seed}_w{worker_id:02d}"
        seed_prefix = os.environ.get(SEED_PREFIX_ENV, "auto")
        return f"{seed_prefix}_{int(time.time() * 1000)}_{worker_id:02d}_{random.randint(1000, 9999)}"

    def _detect_worker_count(self, total_rounds: int) -> int:
        configured = os.environ.get(WORKER_COUNT_ENV)
        if configured:
            try:
                return max(1, min(int(configured), total_rounds))
            except ValueError:
                pass

        cpu_count = os.cpu_count()
        if not isinstance(cpu_count, int) or cpu_count <= 0:
            return min(self.config.default_max_workers, total_rounds)
        return max(1, min(cpu_count, self.config.default_max_workers, total_rounds))

    def _log_step(self, tag: str, steps: int, state: GameStateView, action: FlowAction, seed: str) -> None:
        self.logger.info("%s step=%s decision=%s action=%s seed=%s", tag, steps, state.decision, action.name, seed)

    def _log_step_end(self, tag: str, steps: int, state: GameStateView, reward: float, total_reward: float, seed: str) -> None:
        self.logger.info(
            "%s step_end=%s decision=%s hp=%s/%s gold=%s hand=%s deck=%s reward=%.2f total_reward=%.2f seed=%s",
            tag,
            steps,
            state.decision,
            state.hp,
            state.max_hp,
            state.gold,
            len(state.hand),
            state.deck_size,
            reward,
            total_reward,
            seed,
        )
        if steps % 25 == 0:
            self.logger.info(
                "%s step=%s decision=%s hp=%s/%s gold=%s total_reward=%.2f seed=%s",
                tag,
                steps,
                state.decision,
                state.hp,
                state.max_hp,
                state.gold,
                total_reward,
                seed,
            )

    def run_single_game(self, worker_id: int = 0) -> FlowRunResult:
        tag = f"[W{worker_id:02d}]"
        rng = random.Random()
        self.logger.info("%s Starting HTTP CLI game full-flow test", tag)

        self.protocol.health_check(retries=self.config.health_check_retries)
        seed = self._build_seed(worker_id)
        start = self.protocol.start_game(self.config.character, seed)
        game_id = start.game_id
        self.logger.info("%s Game created: %s seed=%s", tag, game_id, seed)

        total_reward = 0.0
        steps = 0
        raw_state = start.raw_state or self.protocol.get_state(game_id)
        state = self.protocol.adapt_state(raw_state)
        last_fingerprint: Optional[str] = None
        stagnant_steps = 0
        no_action_combat_stagnant_steps = 0

        if not raw_state or not state.decision:
            for _ in range(self.config.initial_state_poll_attempts):
                time.sleep(self.config.initial_state_poll_interval_seconds)
                raw_state = self.protocol.get_state(game_id)
                state = self.protocol.adapt_state(raw_state)
                if state.decision:
                    break

        try:
            while self.config.max_steps is None or steps < self.config.max_steps:
                if state.game_over:
                    break

                action = self.protocol.sanitize_action(state, self.policy.choose_action(state, rng), rng)
                self._log_step(tag, steps, state, action, seed)
                result = self.protocol.step(game_id, action)
                if result.status != "success":
                    self.logger.warning(
                        "%s step failed: decision=%s action=%s seed=%s game_id=%s message=%s",
                        tag,
                        state.decision,
                        action.name,
                        seed,
                        game_id,
                        result.message[:220],
                    )
                    raw_state = result.last_state or state.raw
                    state = self.protocol.adapt_state(raw_state)
                    recover_action = self.protocol.recover_action_from_error(result.raw, state) or FlowAction("proceed")
                    retry = self.protocol.step(game_id, self.protocol.sanitize_action(state, recover_action, rng))
                    if retry.status != "success":
                        try:
                            raw_state = self.protocol.get_state(game_id)
                            state = self.protocol.adapt_state(raw_state)
                            steps += 1
                            time.sleep(self.config.recovery_delay_seconds)
                            continue
                        except Exception as exc:
                            raise RuntimeError(f"Step failed and state recovery failed: {retry.raw}") from exc
                    result = retry

                reward = float(result.reward)
                total_reward += reward
                raw_state = result.state or state.raw
                state = self.protocol.adapt_state(raw_state)
                steps += 1
                time.sleep(self.config.step_delay_seconds)

                fingerprint = state.fingerprint()
                state_repeated = fingerprint == last_fingerprint
                if state_repeated:
                    stagnant_steps += 1
                else:
                    stagnant_steps = 0
                    last_fingerprint = fingerprint

                if state.decision == "combat_play" and not state.playable_cards() and state_repeated:
                    no_action_combat_stagnant_steps += 1
                else:
                    no_action_combat_stagnant_steps = 0

                if no_action_combat_stagnant_steps == self.config.no_action_combat_proceed_threshold:
                    self.logger.warning(
                        "%s no-action combat repeated %s times; trying forced proceed recovery",
                        tag,
                        no_action_combat_stagnant_steps,
                    )
                    forced_proceed = FlowAction("proceed", {"_force_if_stuck": True})
                    forced_result = self.protocol.step(game_id, self.protocol.sanitize_action(state, forced_proceed, rng))
                    if forced_result.status == "success":
                        raw_state = forced_result.state or state.raw
                        state = self.protocol.adapt_state(raw_state)
                        last_fingerprint = state.fingerprint()
                        stagnant_steps = 0
                        no_action_combat_stagnant_steps = 0
                        continue

                if stagnant_steps == self.config.stuck_warn_threshold:
                    self.logger.warning(
                        "%s state appears stuck for %s steps at decision=%s; trying forced end_turn/proceed recovery",
                        tag,
                        stagnant_steps,
                        state.decision,
                    )
                    if state.decision == "combat_play" and not state.playable_cards():
                        forced_action = FlowAction("proceed", {"_force_if_stuck": True})
                    elif state.decision == "combat_play":
                        forced_action = FlowAction("end_turn", {"_force_if_stuck": True})
                    else:
                        forced_action = FlowAction("proceed")
                    self.protocol.step(game_id, self.protocol.sanitize_action(state, forced_action, rng))

                if no_action_combat_stagnant_steps == self.config.no_action_combat_abort_threshold:
                    raise RuntimeError(
                        f"{tag} no-action combat deadlock detected after "
                        f"{no_action_combat_stagnant_steps} repeated states: {state.summary()}"
                    )

                if stagnant_steps >= self.config.stuck_abort_threshold:
                    raise RuntimeError(
                        f"{tag} stuck loop detected at decision={state.decision} "
                        f"for {stagnant_steps} repeated states: {state.summary()}"
                    )

                self._log_step_end(tag, steps, state, reward, total_reward, seed)

            if self.config.max_steps is not None and steps >= self.config.max_steps:
                raise RuntimeError(f"Exceeded MAX_STEPS={self.config.max_steps} without game_over")

            self.logger.info(
                "%s Finished game in %s steps. victory=%s total_reward=%.2f seed=%s",
                tag,
                steps,
                state.victory,
                total_reward,
                seed,
            )
            return FlowRunResult(
                success=True,
                worker_id=worker_id,
                game_id=game_id,
                seed=seed,
                steps=steps,
                decision=state.decision,
                victory=state.victory,
                total_reward=total_reward,
            )
        except Exception as exc:
            self.logger.exception(
                "%s game failed: seed=%s game_id=%s steps=%s decision=%s error=%s",
                tag,
                seed,
                game_id,
                steps,
                state.decision,
                exc,
            )
            return FlowRunResult(
                success=False,
                worker_id=worker_id,
                game_id=game_id,
                seed=seed,
                steps=steps,
                decision=state.decision,
                error=str(exc),
            )
        finally:
            self.protocol.close_game(game_id)

    def run_parallel_games(self, *, workers: Optional[int] = None, total_rounds: Optional[int] = None) -> FlowRunSummary:
        effective_rounds = self.config.total_rounds if total_rounds is None else max(1, total_rounds)
        effective_workers = self._detect_worker_count(effective_rounds) if workers is None else max(1, min(workers, effective_rounds))
        self.logger.info("Running parallel game tests with workers=%s total_rounds=%s", effective_workers, effective_rounds)

        results: list[FlowRunResult] = []
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [executor.submit(self.run_single_game, index + 1) for index in range(effective_rounds)]
            for future in as_completed(futures):
                results.append(future.result())

        ok = [result for result in results if result.success]
        failed = [result for result in results if not result.success]
        victory = sum(1 for result in ok if result.victory)
        defeat = sum(1 for result in ok if not result.victory)
        avg_steps = round(sum(result.steps for result in ok) / len(ok), 2) if ok else 0.0
        avg_reward = round(sum(result.total_reward for result in ok) / len(ok), 2) if ok else 0.0
        min_steps = min((result.steps for result in ok), default=0)
        max_steps = max((result.steps for result in ok), default=0)

        self.logger.info(
            "Parallel summary: workers=%s total_rounds=%s total=%s success=%s fail=%s victory=%s defeat=%s avg_steps=%s min_steps=%s max_steps=%s avg_reward=%s",
            effective_workers,
            effective_rounds,
            len(results),
            len(ok),
            len(failed),
            victory,
            defeat,
            avg_steps,
            min_steps,
            max_steps,
            avg_reward,
        )
        if failed:
            self.logger.warning("Failed runs for replay/debug (%s):", len(failed))
            for result in failed:
                self.logger.warning(
                    "  worker=%s seed=%s game_id=%s steps=%s decision=%s error=%s",
                    result.worker_id,
                    result.seed,
                    result.game_id,
                    result.steps,
                    result.decision,
                    result.error[:220],
                )

        return FlowRunSummary(
            workers=effective_workers,
            total_rounds=effective_rounds,
            total=len(results),
            success=len(ok),
            fail=len(failed),
            victory=victory,
            defeat=defeat,
            avg_steps=avg_steps,
            min_steps=min_steps,
            max_steps=max_steps,
            avg_reward=avg_reward,
            failed_runs=failed,
        )
