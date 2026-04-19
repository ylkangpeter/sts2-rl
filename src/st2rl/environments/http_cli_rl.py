# -*- coding: utf-8 -*-
"""Gymnasium environment for training through the sts2-cli HTTP protocol."""

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from st2rl.gameplay.knowledge_matcher import DEFAULT_MISMATCH_LOG_PATH, KnowledgeMatcher
from st2rl.gameplay.types import FlowAction, GameStateView
from st2rl.protocols.http_cli import (
    HttpCliProtocol,
    HttpCliProtocolConfig,
    mark_headless_unsupported_card,
)
from st2rl.training.action_codec import create_action_space, decode_action
from st2rl.training.reward import RewardConfig, RewardTracker
from st2rl.training.telemetry import SessionLeaderboardStore, SlotTelemetry
from st2rl.utils.logger import get_run_logger


@dataclass(slots=True)
class HttpCliEnvConfig:
    """Configuration for the HTTP CLI RL environment."""

    character: str = "Ironclad"
    max_steps: int | None = 5000
    game_dir: str | None = None
    base_url: str = "http://127.0.0.1:5000"
    timeout_seconds: int = 30
    health_check_retries: int = 2
    step_delay_seconds: float = 0.0
    recovery_delay_seconds: float = 0.01
    initial_state_poll_attempts: int = 20
    initial_state_poll_interval_seconds: float = 0.05
    stuck_warn_threshold: int = 60
    stuck_abort_threshold: int = 140
    no_action_combat_abort_threshold: int = 12
    no_action_combat_proceed_threshold: int = 3
    async_combat_poll_attempts: int = 4
    async_combat_poll_interval_seconds: float = 0.05
    observation_max_hand: int = 10
    observation_max_enemies: int = 5
    seed_offset: int = 0
    telemetry_dir: str | None = None
    knowledge_mismatch_log_path: str = str(DEFAULT_MISMATCH_LOG_PATH)
    overlong_seed_log_path: str | None = None


class HttpCliRlEnv(gym.Env):
    """Formal RL environment built on the reusable HTTP CLI protocol."""

    metadata = {"render_modes": ["human"]}

    DECISION_TYPES = [
        "unknown",
        "combat_play",
        "map_node",
        "card_reward",
        "shop",
        "event",
        "rest_site",
        "card_select",
        "bundle_select",
        "game_over",
    ]

    def __init__(
        self,
        character: str = "Ironclad",
        max_steps: int | None = None,
        game_dir: str | None = None,
        base_url: str = "http://localhost:5000",
        reward_config: RewardConfig | Dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__()
        config_values = {
            "character": character,
            "max_steps": max_steps,
            "game_dir": game_dir,
            "base_url": base_url,
        }
        valid_fields = set(HttpCliEnvConfig.__dataclass_fields__.keys())
        for key, value in kwargs.items():
            if key in valid_fields:
                config_values[key] = value
        self.config = HttpCliEnvConfig(**config_values)
        self.protocol = HttpCliProtocol(
            HttpCliProtocolConfig(
                base_url=self.config.base_url,
                timeout_seconds=self.config.timeout_seconds,
                game_dir=self.config.game_dir,
            )
        )
        if isinstance(reward_config, dict):
            reward_config = RewardConfig(**reward_config)
        self.reward_tracker = RewardTracker(reward_config or RewardConfig())
        self.logger = get_run_logger()
        self.telemetry = SlotTelemetry(self.config.telemetry_dir, self.config.seed_offset)
        self.session_store = SessionLeaderboardStore(self.config.telemetry_dir)
        self.knowledge_matcher = KnowledgeMatcher(Path(self.config.knowledge_mismatch_log_path))
        default_overlong_log = Path(self.config.telemetry_dir or "logs") / "overlong_episode_seeds.jsonl"
        self._overlong_seed_log_path = Path(self.config.overlong_seed_log_path or default_overlong_log)
        self._anomaly_log_path = Path(self.config.telemetry_dir or "logs") / "session_anomalies.jsonl"
        self.action_space = create_action_space()
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(129,), dtype=np.float32)

        self._episode_index = 0
        self._step_count = 0
        self._episode_reward = 0.0
        self._game_id: Optional[str] = None
        self._seed: Optional[str] = None
        self._state: Optional[GameStateView] = None
        self._last_fingerprint: Optional[str] = None
        self._last_progress_marker: Optional[str] = None
        self._stagnant_steps = 0
        self._no_action_combat_stagnant_steps = 0
        self._episode_started_at: Optional[str] = None
        self._episode_trace: list[dict[str, Any]] = []
        self._episode_initial_snapshot: dict[str, Any] = {}
        self._pending_map_target: Optional[dict[str, Any]] = None
        self._protocol_error_streak = 0
        self._deadlock_error_streak = 0
        self._termination_reason: Optional[str] = None
        self._combat_deadlock_card_ids: set[str] = set()

    def _normalize_room_type(self, value: Any, *, symbol: Any = None) -> str:
        text = str(value or symbol or "").strip().lower()
        if not text:
            return "Unknown"
        if any(token in text for token in ("elite", "elites")) or text == "e":
            return "Elite"
        if any(token in text for token in ("monster", "combat")) or text == "m":
            return "Monster"
        if any(token in text for token in ("event", "question")) or text == "?":
            return "Event"
        if any(token in text for token in ("rest", "campfire", "fire", "smith")) or text == "r":
            return "Rest"
        if any(token in text for token in ("shop", "merchant", "store")) or text in ("$", "s"):
            return "Shop"
        if any(token in text for token in ("treasure", "chest")) or text == "t":
            return "Treasure"
        if "boss" in text or text == "b":
            return "Boss"
        if "map" in text or "neow" in text or "start" in text:
            return "StartReward"
        return str(value or symbol or "Unknown")

    def _build_seed(self) -> str:
        forced_seed = os.environ.get("STS2_HTTP_SEED")
        if forced_seed:
            return f"{forced_seed}_rl_{self._episode_index:04d}"
        return f"rl_{int(time.time() * 1000)}_{self.config.seed_offset:02d}_{self._episode_index:04d}_{random.randint(1000, 9999)}"

    def _slot_payload(self, *, active: bool) -> dict[str, Any]:
        state = self._state
        summary = state.summary() if state is not None else {}
        context = (state.raw.get("context") if state is not None else {}) or {}
        boss_info = self._extract_boss_info(state)
        return {
            "slot": f"{self.config.seed_offset:02d}",
            "seed_offset": self.config.seed_offset,
            "episode_index": self._episode_index,
            "game_id": self._game_id,
            "seed": self._seed,
            "started_at": self._episode_started_at,
            "active": active,
            "decision": summary.get("decision"),
            "floor": int(context.get("floor") or state.raw.get("floor") or 0) if state is not None else 0,
            "act": int(context.get("act") or state.raw.get("act") or 0) if state is not None else 0,
            "room_type": context.get("room_type"),
            "hp": summary.get("player_hp", 0),
            "max_hp": summary.get("player_max_hp", 0),
            "gold": summary.get("gold", 0),
            "energy": summary.get("energy", 0),
            "hand_size": summary.get("hand_size", 0),
            "enemy_count": summary.get("enemy_count", 0),
            "episode_reward": round(self._episode_reward, 2),
            "episode_steps": self._step_count,
            "stagnant_steps": self._stagnant_steps,
            "boss_id": boss_info.get("boss_id"),
            "boss_name": boss_info.get("boss_name"),
            "uptime_seconds": self._episode_elapsed_seconds(),
        }

    def _safe_number(self, value: Any, default: int = 0) -> int:
        try:
            if value is None or value == "":
                return default
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default

    def _state_context(self, state: Optional[GameStateView]) -> dict[str, Any]:
        if state is None:
            return {"act": 0, "floor": 0, "room_type": None}
        context = dict(state.raw.get("context") or {})
        return {
            "act": self._safe_number(context.get("act") or state.raw.get("act"), 0),
            "floor": self._safe_number(context.get("floor") or state.raw.get("floor"), 0),
            "room_type": context.get("room_type") or state.raw.get("room_type") or state.decision,
        }

    def _extract_boss_info(self, state: Optional[GameStateView]) -> dict[str, Any]:
        if state is None:
            return {"boss_id": None, "boss_name": None}
        context = dict(state.raw.get("context") or {})
        boss = dict(context.get("boss") or {})
        if not boss and state.full_map:
            boss = dict(state.full_map.get("boss") or {})
        return {
            "boss_id": boss.get("id"),
            "boss_name": boss.get("name") or boss.get("display_name") or boss.get("id"),
        }

    def _compact_card(self, card: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": card.get("index"),
            "id": card.get("id"),
            "name": card.get("name"),
            "cost": card.get("cost"),
            "type": card.get("type"),
            "rarity": card.get("rarity"),
            "description": card.get("description"),
            "stats": dict(card.get("stats") or {}) if isinstance(card.get("stats"), dict) else {},
            "keywords": list(card.get("keywords") or []) if isinstance(card.get("keywords"), (list, tuple)) else [],
            "after_upgrade": dict(card.get("after_upgrade") or {}) if isinstance(card.get("after_upgrade"), dict) else {},
            "upgraded": bool(card.get("upgraded")),
            "target_type": card.get("target_type"),
            "can_play": bool(card.get("can_play")),
        }

    def _compact_enemy(self, enemy: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": enemy.get("index"),
            "name": enemy.get("name") or enemy.get("id"),
            "hp": enemy.get("hp"),
            "max_hp": enemy.get("max_hp"),
            "block": enemy.get("block"),
            "intent": enemy.get("intent"),
        }

    def _compact_option(self, option: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": option.get("index"),
            "option_id": option.get("option_id"),
            "name": option.get("name") or option.get("label") or option.get("option_id"),
            "label": option.get("label"),
            "description": option.get("description"),
            "is_locked": bool(option.get("is_locked")),
            "is_enabled": bool(option.get("is_enabled", True)),
            "col": option.get("col"),
            "row": option.get("row"),
            "room_type": self._normalize_room_type(option.get("room_type"), symbol=option.get("symbol")),
            "symbol": option.get("symbol") or option.get("icon"),
        }

    def _compact_shop_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": item.get("index"),
            "id": item.get("id"),
            "name": item.get("name"),
            "cost": item.get("cost"),
            "description": item.get("description"),
        }

    def _normalize_map_node(self, node: dict[str, Any]) -> dict[str, Any]:
        col = self._safe_number(node.get("col"), -1)
        row = self._safe_number(node.get("row"), -1)
        children = []
        for child in node.get("children") or []:
            if isinstance(child, dict):
                children.append(
                    {
                        "col": self._safe_number(child.get("col"), -1),
                        "row": self._safe_number(child.get("row"), -1),
                    }
                )
        return {
            "id": f"a{self._safe_number(node.get('act'), 0)}_c{col}_r{row}" if node.get("act") is not None else f"c{col}_r{row}",
            "col": col,
            "row": row,
            "room_type": self._normalize_room_type(node.get("room_type") or node.get("type"), symbol=node.get("symbol") or node.get("icon")),
            "symbol": node.get("symbol") or node.get("icon"),
            "current": bool(node.get("current")),
            "children": children,
        }

    def _extract_map_snapshot(self, state: Optional[GameStateView]) -> dict[str, Any]:
        context = self._state_context(state)
        act = context["act"]
        boss_info = self._extract_boss_info(state)
        nodes = []
        source_nodes = []
        if state is not None:
            full_map = state.full_map or {}
            rows = full_map.get("rows") or []
            if rows:
                for row in rows:
                    source_nodes.extend(list(row or []))
                boss = full_map.get("boss") or {}
                if boss:
                    source_nodes.append(
                        {
                            "col": boss.get("col"),
                            "row": boss.get("row"),
                            "type": "Boss",
                            "symbol": "Boss",
                            "children": [],
                        }
                    )
            else:
                source_nodes = list(state.map_nodes)
        for node in source_nodes:
            normalized = self._normalize_map_node(node)
            normalized["act"] = act
            if normalized["row"] >= 0 and normalized["col"] >= 0:
                normalized["id"] = f"a{act}_c{normalized['col']}_r{normalized['row']}"
            nodes.append(normalized)
        snapshot = {"act": act, "nodes": nodes}
        if state is not None and state.full_map:
            snapshot["context"] = dict(state.full_map.get("context") or {})
            snapshot["current_coord"] = dict(state.full_map.get("current_coord") or {})
            snapshot["boss"] = dict(state.full_map.get("boss") or {})
        if boss_info.get("boss_id") or boss_info.get("boss_name"):
            snapshot["boss_info"] = boss_info
        return snapshot

    def _state_snapshot(self, state: Optional[GameStateView]) -> dict[str, Any]:
        if state is None:
            return {}
        context = self._state_context(state)
        summary = state.summary()
        player_relics = list(state.player.get("relics") or [])
        player_potions = list(state.player.get("potions") or [])
        return {
            "decision": state.decision,
            "context": context,
            "hp": state.hp,
            "max_hp": state.max_hp,
            "block": state.block,
            "gold": state.gold,
            "energy": state.energy,
            "max_energy": state.max_energy,
            "deck_size": state.deck_size,
            "hand_size": len(state.hand),
            "draw_pile_count": state.raw.get("draw_pile_count") or len(state.draw_pile),
            "discard_pile_count": state.raw.get("discard_pile_count") or len(state.discard_pile),
            "exhaust_pile_count": state.raw.get("exhaust_pile_count") or len(state.exhaust_pile),
            "round": state.round,
            "turn": state.turn,
            "can_skip": state.can_skip,
            "min_select": state.min_select,
            "max_select": state.max_select,
            "game_over": state.game_over,
            "victory": state.victory,
            "player": {
                "hp": state.hp,
                "max_hp": state.max_hp,
                "block": state.block,
                "gold": state.gold,
                "deck_size": state.deck_size,
                "deck": [self._compact_card(card) for card in (state.player.get("deck") or []) if isinstance(card, dict)],
                "relics": [dict(relic) for relic in player_relics if isinstance(relic, dict)],
                "potions": [dict(potion) for potion in player_potions if isinstance(potion, dict)],
            },
            "enemies": [self._compact_enemy(enemy) for enemy in state.enemies],
            "hand": [self._compact_card(card) for card in state.hand],
            "cards": [self._compact_card(card) for card in state.cards],
            "options": [self._compact_option(option) for option in state.options],
            "choices": [self._compact_option(option) for option in state.choices],
            "map": [self._normalize_map_node(node) for node in state.map_nodes if isinstance(node, dict)],
            "full_map": dict(state.full_map or {}),
            "relics": [relic.get("name") or relic.get("id") for relic in player_relics if isinstance(relic, dict)],
            "potions": [potion.get("name") or potion.get("id") for potion in player_potions if isinstance(potion, dict)],
            "shop_relics": [self._compact_shop_item(relic) for relic in state.relics if isinstance(relic, dict)],
            "shop_potions": [self._compact_shop_item(potion) for potion in state.potions if isinstance(potion, dict)],
            "purge_cost": state.raw.get("purge_cost"),
            "summary": summary,
        }

    def _extract_rewards(self, before: dict[str, Any], after: dict[str, Any], action: FlowAction) -> list[dict[str, Any]]:
        rewards: list[dict[str, Any]] = []
        if not before or not after:
            return rewards
        gold_delta = self._safe_number(after.get("gold")) - self._safe_number(before.get("gold"))
        if gold_delta > 0:
            rewards.append({"type": "gold", "amount": gold_delta})
        hp_delta = self._safe_number(after.get("hp")) - self._safe_number(before.get("hp"))
        if hp_delta > 0:
            rewards.append({"type": "hp", "amount": hp_delta})
        new_relics = sorted(set(after.get("relics", [])) - set(before.get("relics", [])))
        for relic in new_relics:
            rewards.append({"type": "relic", "name": relic})
        new_potions = sorted(set(after.get("potions", [])) - set(before.get("potions", [])))
        for potion in new_potions:
            rewards.append({"type": "potion", "name": potion})
        before_card_names = {card.get("name") for card in before.get("cards", []) if card.get("name")}
        after_card_names = {card.get("name") for card in after.get("cards", []) if card.get("name")}
        for card_name in sorted(after_card_names - before_card_names):
            rewards.append({"type": "card_seen", "name": card_name})
        if action.name == "choose_card_reward":
            chosen = next((card.get("name") for card in before.get("cards", []) if card.get("index") == action.args.get("card_index")), None)
            if chosen:
                rewards.append({"type": "card_chosen", "name": chosen})
        return rewards

    def _lookup_by_index(self, items: list[dict[str, Any]], index: Any) -> Optional[dict[str, Any]]:
        for item in items or []:
            if item.get("index") == index:
                return item
        return None

    def _display_name(self, item: Optional[dict[str, Any]], *, fallback: Any = None) -> str:
        if not item:
            return str(fallback or "")
        for key in ("name", "label", "description", "id", "option_id"):
            value = item.get(key)
            if value:
                return str(value)
        return str(fallback or "")

    def _describe_action(
        self,
        *,
        before_state: Optional[GameStateView],
        action: FlowAction,
        after_state: Optional[GameStateView],
    ) -> str:
        before = before_state
        args = dict(action.args or {})
        if before is None:
            return action.name

        if action.name == "play_card":
            card = self._lookup_by_index(before.hand, args.get("card_index"))
            card_name = self._display_name(card, fallback=f"card[{args.get('card_index')}]")
            target = self._lookup_by_index(before.enemies, args.get("target_index"))
            target_name = self._display_name(target)
            return f"play {card_name} -> {target_name}" if target_name else f"play {card_name}"

        if action.name == "select_map_node":
            col = args.get("col")
            row = args.get("row")
            target = next(
                (
                    node
                    for node in (before.map_nodes or [])
                    if self._safe_number(node.get("col"), -1) == self._safe_number(col, -1)
                    and self._safe_number(node.get("row"), -1) == self._safe_number(row, -1)
                ),
                None,
            )
            room_type = self._normalize_room_type((target or {}).get("room_type"), symbol=(target or {}).get("symbol"))
            return f"select map ({col},{row}) -> {room_type}"

        if action.name == "choose_option":
            option = self._lookup_by_index(before.options or before.choices, args.get("option_index"))
            option_name = self._display_name(option, fallback=f"option[{args.get('option_index')}]")
            return f"choose {option_name}"

        if action.name == "choose_card_reward":
            card = self._lookup_by_index(before.cards, args.get("card_index"))
            fallback = f"card[{args.get('card_index')}]"
            return f"take {self._display_name(card, fallback=fallback)}"

        if action.name == "select_cards":
            raw_indices = str(args.get("indices", ""))
            picks = [item.strip() for item in raw_indices.split(",") if item.strip()]
            names = []
            for raw_index in picks:
                card = self._lookup_by_index(before.cards, self._safe_number(raw_index, raw_index))
                names.append(self._display_name(card, fallback=raw_index))
            return f"select cards: {', '.join(names)}" if names else "select cards"

        if action.name == "buy_card":
            card = self._lookup_by_index(before.cards, args.get("card_index"))
            card_name = self._display_name(card, fallback=f"card[{args.get('card_index')}]")
            cost = (card or {}).get("cost")
            return f"buy card {card_name}" + (f" ({cost}g)" if cost is not None else "")

        if action.name == "buy_relic":
            relic = self._lookup_by_index(before.relics, args.get("relic_index"))
            relic_name = self._display_name(relic, fallback=f"relic[{args.get('relic_index')}]")
            cost = (relic or {}).get("cost")
            return f"buy relic {relic_name}" + (f" ({cost}g)" if cost is not None else "")

        if action.name == "buy_potion":
            potion = self._lookup_by_index(before.potions, args.get("potion_index"))
            potion_name = self._display_name(potion, fallback=f"potion[{args.get('potion_index')}]")
            cost = (potion or {}).get("cost")
            return f"buy potion {potion_name}" + (f" ({cost}g)" if cost is not None else "")

        if action.name == "purge_card":
            deck = [card for card in (before.player.get("deck") or []) if isinstance(card, dict)]
            card = self._lookup_by_index(deck, args.get("card_index"))
            fallback = f"card[{args.get('card_index')}]"
            return f"purge {self._display_name(card, fallback=fallback)}"

        if action.name == "use_potion":
            potion = self._lookup_by_index(before.player.get("potions", []), args.get("potion_index"))
            potion_name = self._display_name(potion, fallback=f"potion[{args.get('potion_index')}]")
            target = self._lookup_by_index(before.enemies, args.get("target_index"))
            target_name = self._display_name(target)
            return f"use potion {potion_name} -> {target_name}" if target_name else f"use potion {potion_name}"

        if action.name == "leave_shop":
            return "leave shop"
        if action.name == "skip_reward":
            return "skip reward"
        if action.name == "skip_select":
            return "skip select"
        if action.name == "end_turn":
            return "end turn"
        if action.name == "proceed":
            next_decision = after_state.decision if after_state is not None else None
            return f"proceed -> {next_decision}" if next_decision else "proceed"
        if action.name == "select_bundle":
            bundle = self._lookup_by_index(before.bundles, args.get("bundle_index"))
            return f"select bundle {self._display_name(bundle, fallback=args.get('bundle_index'))}"
        return action.name

    def _record_trace_step(
        self,
        *,
        before_state: Optional[GameStateView],
        action: FlowAction,
        after_state: Optional[GameStateView],
        reward: float,
        status: str,
    ) -> None:
        before_snapshot = self._state_snapshot(before_state)
        after_snapshot = self._state_snapshot(after_state)
        trace_item = {
            "step": self._step_count,
            "status": status,
            "action": action.name,
            "action_args": dict(action.args),
            "action_detail": self._describe_action(before_state=before_state, action=action, after_state=after_state),
            "before": before_snapshot,
            "after": after_snapshot,
            "reward": round(float(reward), 4),
            "rewards_found": self._extract_rewards(before_snapshot, after_snapshot, action),
        }
        if self._pending_map_target and after_snapshot.get("context", {}).get("floor", 0) > 0:
            trace_item["map_target"] = dict(self._pending_map_target)
        if before_state is not None and before_state.decision in ("map_node", "map_select"):
            map_snapshot = self._extract_map_snapshot(before_state)
            if map_snapshot.get("nodes"):
                trace_item["map_snapshot"] = map_snapshot
        self._episode_trace.append(trace_item)
        if action.name == "select_map_node":
            self._pending_map_target = {
                "act": self._state_context(before_state).get("act", 0),
                "col": action.args.get("col"),
                "row": action.args.get("row"),
            }
        elif after_snapshot.get("context", {}).get("floor", 0) > 0:
            self._pending_map_target = None

    def _build_node_id(self, act: int, floor: int, map_target: Optional[dict[str, Any]]) -> str:
        if map_target and map_target.get("col") is not None and map_target.get("row") is not None:
            return f"a{act}_c{map_target.get('col')}_r{map_target.get('row')}"
        return f"a{act}_f{floor}"

    def _build_session_details(self, *, terminated: bool, truncated: bool) -> dict[str, Any]:
        initial = dict(self._episode_initial_snapshot)
        final = self._state_snapshot(self._state)
        maps_by_act: dict[int, dict[str, Any]] = {}
        nodes_by_id: dict[str, dict[str, Any]] = {}
        ordered_nodes: list[str] = []
        active_room_target: Optional[dict[str, Any]] = None

        for item in self._episode_trace:
            map_snapshot = item.get("map_snapshot") or {}
            act = self._safe_number(map_snapshot.get("act"), 0)
            if act > 0 and map_snapshot.get("nodes"):
                act_map = maps_by_act.setdefault(act, {"act": act, "nodes": {}, "visited_node_ids": []})
                for node in map_snapshot["nodes"]:
                    existing = act_map["nodes"].get(node["id"], {})
                    merged = dict(existing)
                    merged.update(node)
                    act_map["nodes"][node["id"]] = merged

        for item in self._episode_trace:
            after = item.get("after") or {}
            before = item.get("before") or {}
            context = after.get("context") or {}
            act = self._safe_number(context.get("act"), 0)
            floor = self._safe_number(context.get("floor"), 0)
            if act <= 0 or floor <= 0:
                continue

            if item.get("action") == "select_map_node":
                map_target = {
                    "act": act,
                    "floor": floor,
                    "col": item.get("action_args", {}).get("col"),
                    "row": item.get("action_args", {}).get("row"),
                    "room_type": None,
                }
                map_snapshot = item.get("map_snapshot") or {}
                for map_node in map_snapshot.get("nodes") or []:
                    if (
                        self._safe_number(map_node.get("col"), -1) == self._safe_number(map_target.get("col"), -1)
                        and self._safe_number(map_node.get("row"), -1) == self._safe_number(map_target.get("row"), -1)
                    ):
                        map_target["room_type"] = map_node.get("room_type")
                        break
                if not map_target.get("room_type"):
                    map_target["room_type"] = context.get("room_type")
                active_room_target = map_target

            room_target = None
            if active_room_target and self._safe_number(active_room_target.get("act"), 0) == act:
                room_target = active_room_target

            node_id = self._build_node_id(act, floor, room_target)
            room_type = (room_target or {}).get("room_type") or context.get("room_type")
            node = nodes_by_id.get(node_id)
            if node is None:
                node = {
                    "node_id": node_id,
                    "act": act,
                    "floor": floor,
                    "col": (room_target or {}).get("col"),
                    "row": (room_target or {}).get("row"),
                    "room_type": self._normalize_room_type(room_type),
                    "entry_state": item.get("before") or after,
                    "exit_state": after,
                    "monsters": [],
                    "rewards": [],
                    "actions": [],
                }
                nodes_by_id[node_id] = node
                ordered_nodes.append(node_id)
            node["exit_state"] = after
            if room_type and not node.get("room_type"):
                node["room_type"] = self._normalize_room_type(room_type)
            for enemy in after.get("enemies", []):
                enemy_name = enemy.get("name")
                if enemy_name and enemy_name not in node["monsters"]:
                    node["monsters"].append(enemy_name)
            node["actions"].append(
                {
                    "step": item.get("step"),
                    "decision_before": (item.get("before") or {}).get("decision"),
                    "decision_after": after.get("decision"),
                    "action": item.get("action"),
                    "action_detail": item.get("action_detail"),
                    "action_args": item.get("action_args"),
                    "reward": item.get("reward"),
                }
            )
            for reward_item in item.get("rewards_found", []):
                if reward_item not in node["rewards"]:
                    node["rewards"].append(reward_item)

            act_map = maps_by_act.setdefault(act, {"act": act, "nodes": {}, "visited_node_ids": []})
            act_map["visited_node_ids"].append(node_id)
            if node_id in act_map["nodes"]:
                if node.get("room_type"):
                    act_map["nodes"][node_id]["room_type"] = node.get("room_type")
            else:
                act_map["nodes"][node_id] = {
                    "id": node_id,
                    "act": act,
                    "col": node.get("col", -1),
                    "row": node.get("row", floor),
                    "room_type": self._normalize_room_type(node.get("room_type")),
                    "children": [],
                }

            if after.get("decision") == "map_node" and before.get("decision") != "map_node":
                active_room_target = None

        maps = []
        for act in sorted(maps_by_act):
            act_map = maps_by_act[act]
            act_map["visited_node_ids"] = list(dict.fromkeys(act_map["visited_node_ids"]))
            act_map["nodes"] = sorted(
                act_map["nodes"].values(),
                key=lambda node: (self._safe_number(node.get("row"), 999), self._safe_number(node.get("col"), 999)),
            )
            maps.append(act_map)

        max_floor = max((self._safe_number(node.get("floor"), 0) for node in nodes_by_id.values()), default=self._safe_number(final.get("context", {}).get("floor"), 0))
        boss_info = self._extract_boss_info(self._state)
        summary = {
            "game_id": self._game_id,
            "seed": self._seed,
            "character": self.config.character,
            "slot": f"{self.config.seed_offset:02d}",
            "episode_index": self._episode_index,
            "episode_reward": round(self._episode_reward, 2),
            "episode_steps": self._step_count,
            "victory": bool(final.get("victory")),
            "terminated": terminated,
            "truncated": truncated,
            "max_floor": max_floor,
            "act": self._safe_number(final.get("context", {}).get("act"), 0),
            "final_hp": self._safe_number(final.get("hp"), 0),
            "max_hp": self._safe_number(final.get("max_hp"), 0),
            "final_gold": self._safe_number(final.get("gold"), 0),
            "started_at": self._episode_started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_seconds": self._episode_elapsed_seconds(),
            "boss_id": boss_info.get("boss_id"),
            "boss_name": boss_info.get("boss_name"),
        }
        return {
            "summary": summary,
            "initial_state": initial,
            "final_state": final,
            "maps": maps,
            "nodes": [nodes_by_id[node_id] for node_id in ordered_nodes],
            "trace": self._episode_trace,
        }

    def _write_current_slot(self, *, active: bool, force: bool = False) -> None:
        self.telemetry.write_current(self._slot_payload(active=active), force=force)

    def _record_episode_summary(self, *, terminated: bool, truncated: bool) -> None:
        payload = self._slot_payload(active=False)
        payload["terminated"] = terminated
        payload["truncated"] = truncated
        payload["game_over"] = bool(self._state.game_over) if self._state is not None else False
        payload["victory"] = bool(self._state.victory) if self._state is not None else False
        if self._termination_reason:
            payload["termination_reason"] = self._termination_reason
        if truncated and self.config.max_steps is not None and self._step_count >= self.config.max_steps:
            payload["termination_reason"] = self._termination_reason or "max_steps_exceeded"
            self._record_overlong_seed()
        details = self._build_session_details(terminated=terminated, truncated=truncated)
        anomaly_flags = self._episode_anomaly_flags(details["summary"])
        if anomaly_flags:
            payload["anomaly_flags"] = anomaly_flags
            details["summary"]["anomaly_flags"] = anomaly_flags
            self._record_episode_anomaly(details["summary"])
        leaderboard_result = self.session_store.write_session_if_ranked(details["summary"], details)
        payload["details_available"] = bool(leaderboard_result.get("ranked"))
        payload["leaderboard_rank"] = leaderboard_result.get("rank")
        payload["is_best"] = bool(leaderboard_result.get("best"))
        self.telemetry.append_history(payload)

    def _record_overlong_seed(self) -> None:
        self._overlong_seed_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "seed": self._seed,
            "game_id": self._game_id,
            "slot": f"{self.config.seed_offset:02d}",
            "episode_index": self._episode_index,
            "episode_steps": self._step_count,
            "max_steps": self.config.max_steps,
            "started_at": self._episode_started_at,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self._overlong_seed_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _episode_elapsed_seconds(self) -> float | None:
        if not self._episode_started_at:
            return None
        try:
            started = time.mktime(time.strptime(self._episode_started_at, "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            return None
        return round(max(time.time() - started, 0.0), 3)

    def _episode_anomaly_flags(self, summary: dict[str, Any]) -> list[str]:
        flags: list[str] = []
        victory = bool(summary.get("victory"))
        final_hp = self._safe_number(summary.get("final_hp"), 0)
        elapsed_seconds = summary.get("elapsed_seconds")
        game_over = bool(self._state.game_over) if self._state is not None else False
        if not victory and final_hp > 0:
            flags.append("nonzero_hp_end")
        if not game_over and bool(summary.get("truncated")):
            flags.append("truncated_without_game_over")
        if isinstance(elapsed_seconds, (int, float)) and float(elapsed_seconds) > 600:
            flags.append("overlong_episode")
        if self._termination_reason in {"protocol_deadlock", "stuck_abort", "step_exception", "fatal_protocol_error"}:
            flags.append(str(self._termination_reason))
        return list(dict.fromkeys(flags))

    def _record_episode_anomaly(self, summary: dict[str, Any]) -> None:
        self._anomaly_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(summary)
        payload["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._anomaly_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _decision_index(self, decision: str) -> int:
        try:
            return self.DECISION_TYPES.index(decision)
        except ValueError:
            return 0

    def _progress_marker(self, state: Optional[GameStateView]) -> str:
        if state is None:
            return "none"

        context = self._state_context(state)
        enemies = tuple(
            (
                enemy.get("name"),
                self._safe_number(enemy.get("hp")),
                self._safe_number(enemy.get("block")),
                bool(enemy.get("intends_attack")),
                tuple(intent.get("type") for intent in (enemy.get("intents") or [])),
            )
            for enemy in state.enemies
        )
        hand = tuple(
            (
                card.get("id") or card.get("name"),
                self._safe_number(card.get("cost"), 99),
                bool(card.get("can_play")),
            )
            for card in state.hand
        )
        choices = tuple(
            (
                choice.get("index"),
                choice.get("col"),
                choice.get("row"),
                choice.get("room_type"),
                bool(choice.get("is_locked")),
                bool(choice.get("is_enabled", True)),
            )
            for choice in (state.options or state.choices)
        )
        return str(
            (
                state.decision,
                self._safe_number(context.get("act")),
                self._safe_number(context.get("floor")),
                context.get("room_type"),
                state.round,
                state.turn,
                state.hp,
                state.max_hp,
                state.block,
                state.gold,
                state.energy,
                state.max_energy,
                state.deck_size,
                len(state.player.get("relics") or []),
                len(state.player.get("potions") or []),
                len(state.raw.get("player_powers") or []),
                enemies,
                hand,
                choices,
                bool(state.game_over),
                bool(state.victory),
            )
        )

    def _best_effort_combat_action(self, state: Optional[GameStateView]) -> Optional[FlowAction]:
        if state is None or state.decision != "combat_play":
            return None
        playable = [
            card
            for card in state.playable_cards()
            if str(card.get("id") or "").strip().upper() not in self._combat_deadlock_card_ids
        ]
        enemies = state.living_enemies()
        if playable:
            attack_cards = [card for card in playable if str(card.get("type") or "").lower() == "attack"]
            preferred = attack_cards or playable
            card = min(
                preferred,
                key=lambda item: (
                    int(item.get("cost") or 0),
                    0 if str(item.get("type") or "").lower() == "attack" else 1,
                    int(item.get("index") or 999),
                ),
            )
            args = {"card_index": card.get("index")}
            if card.get("target_type") == "AnyEnemy" and enemies:
                target = min(enemies, key=lambda enemy: int(enemy.get("hp") or 9999))
                args["target_index"] = target.get("index")
            return FlowAction("play_card", args)

        potions = [potion for potion in (state.player.get("potions") or []) if isinstance(potion, dict)]
        safe_potions = []
        for potion in potions:
            target_type = str(potion.get("target_type") or "").strip().lower()
            if "allenemies" in target_type or "all_enemies" in target_type or "anyenemy" in target_type or target_type.endswith("enemy"):
                safe_potions.append(potion)
        if safe_potions:
            potion = safe_potions[0]
            args = {"potion_index": potion.get("index", 0)}
            if "anyenemy" in str(potion.get("target_type") or "").strip().lower() and enemies:
                target = min(enemies, key=lambda enemy: int(enemy.get("hp") or 9999))
                args["target_index"] = target.get("index")
            return FlowAction("use_potion", args)
        if enemies:
            return FlowAction("end_turn")
        return FlowAction("proceed")

    def _remember_deadlock_card(self, state: Optional[GameStateView], action: Optional[FlowAction]) -> None:
        if state is None or action is None or action.name != "play_card":
            return
        selected_index = action.args.get("card_index")
        if selected_index is None:
            return
        for card in state.hand:
            if not isinstance(card, dict) or card.get("index") != selected_index:
                continue
            card_id = str(card.get("id") or "").strip().upper()
            if not card_id:
                return
            self._combat_deadlock_card_ids.add(card_id)
            mark_headless_unsupported_card(card)
            self.logger.warning(
                "Marking card as headless-unsupported after combat deadlock: slot=%s seed=%s game_id=%s card_id=%s card_name=%s",
                self.config.seed_offset,
                self._seed,
                self._game_id,
                card_id,
                card.get("name"),
            )
            return

    def _encode_cards(self, cards: list[Dict[str, Any]], max_count: int) -> list[float]:
        features: list[float] = []
        for card in cards[:max_count]:
            features.extend(
                [
                    float(card.get("cost", 0)) / 5.0,
                    1.0 if card.get("can_play") else 0.0,
                    1.0 if card.get("target_type") == "AnyEnemy" else 0.0,
                    1.0 if card.get("is_upgraded") else 0.0,
                ]
            )
        while len(features) < max_count * 4:
            features.extend([0.0, 0.0, 0.0, 0.0])
        return features

    def _encode_enemies(self, enemies: list[Dict[str, Any]], max_count: int) -> list[float]:
        features: list[float] = []
        for enemy in enemies[:max_count]:
            features.extend(
                [
                    float(enemy.get("hp", 0)) / 200.0,
                    float(enemy.get("block", 0)) / 100.0,
                    1.0 if enemy.get("hp", 0) > 0 else 0.0,
                ]
            )
        while len(features) < max_count * 3:
            features.extend([0.0, 0.0, 0.0])
        return features

    def _room_type_features(self, room_type: Any) -> list[float]:
        normalized = self._normalize_room_type(room_type).lower()
        return [
            1.0 if normalized == "monster" else 0.0,
            1.0 if normalized == "elite" else 0.0,
            1.0 if normalized == "rest" else 0.0,
            1.0 if normalized == "shop" else 0.0,
            1.0 if normalized == "treasure" else 0.0,
            1.0 if normalized == "event" else 0.0,
            1.0 if normalized == "boss" else 0.0,
        ]

    def _get_observation(self, state: GameStateView) -> np.ndarray:
        knowledge_summary = self.knowledge_matcher.summarize_state(state)
        context = self._state_context(state)
        playable_cards = state.playable_cards()
        playable_types = [str(card.get("type") or "").lower() for card in playable_cards]
        enemy_intents = [
            str(intent.get("type") or "")
            for enemy in state.enemies
            for intent in (enemy.get("intents") or [])
            if isinstance(intent, dict)
        ]
        attack_intents = sum(1 for intent in enemy_intents if intent == "Attack")
        non_attack_intents = sum(1 for intent in enemy_intents if intent and intent != "Attack")
        enemy_blocks = [float(enemy.get("block") or 0) for enemy in state.enemies]
        shop_card_count = len(state.cards) if state.decision == "shop" else 0
        shop_relic_count = len(state.relics) if state.decision == "shop" else 0
        shop_potion_count = len(state.potions) if state.decision == "shop" else 0
        player_relic_count = len(state.player.get("relics") or [])
        player_potion_count = len(state.player.get("potions") or [])
        player_power_count = len(state.raw.get("player_powers") or [])
        extra_vector = [
            float(context.get("act") or 0) / 3.0,
            float(context.get("floor") or 0) / 20.0,
            *self._room_type_features(context.get("room_type")),
            float(state.raw.get("draw_pile_count") or 0) / 40.0,
            float(state.raw.get("discard_pile_count") or 0) / 40.0,
            float(state.raw.get("exhaust_pile_count") or 0) / 20.0,
            float(player_relic_count) / 20.0,
            float(player_potion_count) / 5.0,
            float(player_power_count) / 10.0,
            float(len(playable_cards)) / 10.0,
            float(sum(1 for card_type in playable_types if card_type == "attack")) / 10.0,
            float(sum(1 for card_type in playable_types if card_type == "skill")) / 10.0,
            float(sum(1 for card_type in playable_types if card_type == "power")) / 10.0,
            float(attack_intents) / 5.0,
            float(non_attack_intents) / 5.0,
            (sum(enemy_blocks) / len(enemy_blocks) / 50.0) if enemy_blocks else 0.0,
            float(shop_card_count) / 10.0,
            float(shop_relic_count) / 10.0,
            float(shop_potion_count) / 10.0,
            float(len(state.choices)) / 10.0,
            float(state.min_select or 0) / 5.0,
            float(state.max_select or 0) / 5.0,
            1.0 if state.gold >= int(state.raw.get("purge_cost") or 999999) else 0.0,
            float(len(state.bundles)) / 5.0,
            1.0 if any(card.get("target_type") == "AnyEnemy" for card in playable_cards) else 0.0,
            1.0 if any(str(potion.get("target_type") or "") == "AnyEnemy" for potion in (state.player.get("potions") or [])) else 0.0,
            float(len(state.player.get("deck") or [])) / 50.0,
            1.0 if bool((context.get("boss") or {}).get("id")) else 0.0,
        ]
        vector = [
            self._decision_index(state.decision) / max(1, len(self.DECISION_TYPES) - 1),
            state.hp / 100.0,
            state.max_hp / 100.0,
            state.block / 100.0,
            state.gold / 500.0,
            state.energy / 5.0,
            state.max_energy / 5.0,
            state.deck_size / 50.0,
            len(state.hand) / 10.0,
            len(state.living_enemies()) / 5.0,
            float(state.round or 0) / 10.0,
            float(state.turn or 0) / 10.0,
            len(state.cards) / 10.0,
            len(state.options or state.choices) / 10.0,
            len(state.bundles) / 5.0,
            1.0 if state.can_skip else 0.0,
        ]
        vector.extend(self._encode_cards(state.hand, self.config.observation_max_hand))
        vector.extend(self._encode_enemies(state.enemies, self.config.observation_max_enemies))
        vector.extend(float(value) for value in knowledge_summary["features"])
        vector.extend(extra_vector)
        return np.array(vector, dtype=np.float32)

    def _apply_transition_tracking(self, state: GameStateView) -> None:
        fingerprint = state.fingerprint()
        progress_marker = self._progress_marker(state)
        state_repeated = progress_marker == self._last_progress_marker
        if state_repeated:
            self._stagnant_steps += 1
        else:
            self._stagnant_steps = 0
            self._last_progress_marker = progress_marker
            self._last_fingerprint = fingerprint

        if state.decision == "combat_play" and not state.playable_cards() and state_repeated:
            self._no_action_combat_stagnant_steps += 1
        else:
            self._no_action_combat_stagnant_steps = 0

    def _force_progress_if_needed(self) -> Tuple[bool, float]:
        if self._state is None or self._game_id is None:
            return False, 0.0

        if self._no_action_combat_stagnant_steps == self.config.no_action_combat_proceed_threshold:
            forced = FlowAction("proceed", {"_force_if_stuck": True})
            result = self.protocol.step(self._game_id, self.protocol.sanitize_action(self._state, forced, random))
            if result.status == "success" and result.state:
                self._state = self.protocol.adapt_state(result.state)
                self.reward_tracker.reset(self._state)
                self._last_fingerprint = self._state.fingerprint()
                self._last_progress_marker = self._progress_marker(self._state)
                self._stagnant_steps = 0
                self._no_action_combat_stagnant_steps = 0
                return True, 0.0

        if self._stagnant_steps >= self.config.stuck_abort_threshold:
            return False, self.reward_tracker.on_stuck()

        return False, 0.0

    def _is_deadlock_error(self, payload: dict[str, Any] | None) -> bool:
        if not payload:
            return False
        message = str(payload.get("message", "") or "").lower()
        last_state = payload.get("last_state")
        tokens = (
            "stale",
            "did not produce a new state",
            "no new state",
            "state unchanged",
        )
        return any(token in message for token in tokens) or bool(last_state and "state" in str(last_state).lower())

    def _is_shop_like_error(self, payload: dict[str, Any] | None) -> bool:
        if not payload:
            return False
        message = str(payload.get("message", "") or "").lower()
        return any(
            token in message
            for token in (
                "buy relic failed",
                "buy potion failed",
                "buy card failed",
                "relic already purchased",
                "not enough gold",
                "not in a shop",
                "purge_card",
                "remove_card",
                "buy_relic",
                "buy_potion",
                "buy_card",
            )
        )

    def _should_poll_for_async_combat_progress(
        self,
        before_state: Optional[GameStateView],
        action: FlowAction,
    ) -> bool:
        if before_state is None or before_state.decision != "combat_play":
            return False
        if before_state.playable_cards():
            return False
        return action.name in {"end_turn", "proceed"}

    def _poll_for_async_combat_progress(
        self,
        before_marker: str,
    ) -> bool:
        if self._game_id is None:
            return False

        for _ in range(max(1, self.config.async_combat_poll_attempts)):
            time.sleep(max(0.01, self.config.async_combat_poll_interval_seconds))
            try:
                raw_state = self.protocol.get_state(self._game_id)
            except Exception:
                continue
            candidate = self.protocol.adapt_state(raw_state)
            if self._progress_marker(candidate) == before_marker:
                continue
            self._state = candidate
            self.reward_tracker.reset(candidate)
            self._protocol_error_streak = 0
            self._deadlock_error_streak = 0
            return True

        return False

    def _zero_observation(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _truncate_episode_on_exception(
        self,
        *,
        before_state: Optional[GameStateView],
        action: Optional[FlowAction],
        exc: Exception,
        termination_reason: str,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.logger.exception(
            "HTTP CLI environment failure: reason=%s slot=%s episode=%s seed=%s game_id=%s step=%s",
            termination_reason,
            self.config.seed_offset,
            self._episode_index,
            self._seed,
            self._game_id,
            self._step_count,
        )
        reward = self.reward_tracker.on_stuck()
        obs = self._get_observation(self._state) if self._state is not None else self._zero_observation()
        self._episode_reward += reward
        self._termination_reason = termination_reason
        trace_action = action or FlowAction("proceed", {"_error": termination_reason})
        if before_state is not None or self._state is not None:
            try:
                self._record_trace_step(
                    before_state=before_state or self._state,
                    action=trace_action,
                    after_state=self._state,
                    reward=reward,
                    status=termination_reason,
                )
            except Exception:
                self.logger.exception(
                    "Failed to record trace for environment failure: reason=%s seed=%s game_id=%s",
                    termination_reason,
                    self._seed,
                    self._game_id,
                )
        info = {
            "decision": self._state.decision if self._state is not None else None,
            "episode_reward": self._episode_reward,
            "episode_steps": self._step_count,
            "game_id": self._game_id,
            "error": f"{type(exc).__name__}: {exc}",
            "termination_reason": termination_reason,
        }
        if self._state is not None:
            info["state_summary"] = self._state.summary()
        try:
            self._record_episode_summary(terminated=False, truncated=True)
        except Exception:
            self.logger.exception(
                "Failed to record episode summary after environment failure: reason=%s seed=%s game_id=%s",
                termination_reason,
                self._seed,
                self._game_id,
            )
        self._write_current_slot(active=False, force=True)
        self.close()
        return obs, reward, False, True, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._episode_index += 1
        episode_seed = self._build_seed()
        last_exc: Exception | None = None

        for attempt in range(1, 4):
            self.close()
            self._seed = episode_seed
            self._step_count = 0
            self._episode_reward = 0.0
            self._last_fingerprint = None
            self._last_progress_marker = None
            self._stagnant_steps = 0
            self._no_action_combat_stagnant_steps = 0
            self._episode_started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._episode_trace = []
            self._episode_initial_snapshot = {}
            self._pending_map_target = None
            self._protocol_error_streak = 0
            self._deadlock_error_streak = 0
            self._termination_reason = None
            self._combat_deadlock_card_ids = set()
            try:
                self.protocol.health_check(retries=self.config.health_check_retries)
                start = self.protocol.start_game(self.config.character, self._seed, self.config.seed_offset)
                self._game_id = start.game_id
                raw_state = start.raw_state or self.protocol.get_state(self._game_id)
                self._state = self.protocol.adapt_state(raw_state)

                if not raw_state or not self._state.decision:
                    for _ in range(self.config.initial_state_poll_attempts):
                        time.sleep(self.config.initial_state_poll_interval_seconds)
                        raw_state = self.protocol.get_state(self._game_id)
                        self._state = self.protocol.adapt_state(raw_state)
                        if self._state.decision:
                            break

                self.reward_tracker.reset(self._state)
                self._last_fingerprint = self._state.fingerprint()
                self._last_progress_marker = self._progress_marker(self._state)
                self._episode_initial_snapshot = self._state_snapshot(self._state)
                obs = self._get_observation(self._state)
                info = {
                    "game_id": self._game_id,
                    "decision": self._state.decision,
                    "episode_reward": self._episode_reward,
                    "episode_steps": self._step_count,
                }
                self._write_current_slot(active=True, force=True)
                return obs, info
            except Exception as exc:
                last_exc = exc
                self.logger.exception(
                    "Environment reset attempt failed: attempt=%s slot=%s episode=%s seed=%s game_id=%s",
                    attempt,
                    self.config.seed_offset,
                    self._episode_index,
                    self._seed,
                    self._game_id,
                )
                self.close()
                time.sleep(min(2.0, 0.5 * attempt))

        raise RuntimeError(
            f"Failed to reset HTTP CLI environment after retries for seed {self._seed}"
        ) from last_exc

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._state is None or self._game_id is None:
            raise RuntimeError("Environment not reset")
        before_state = self._state
        decoded = decode_action(action)
        sanitized = self.protocol.sanitize_action(self._state, decoded, random)
        try:
            before_marker = self._progress_marker(before_state)
            result = self.protocol.step(self._game_id, sanitized)
            reward = 0.0
            abort_episode = False
            truncated = False
            termination_reason: Optional[str] = None
            if result.status != "success":
                self._protocol_error_streak += 1
                if self._is_deadlock_error(result.raw):
                    self._deadlock_error_streak += 1
                if self._should_poll_for_async_combat_progress(before_state, sanitized) and self._poll_for_async_combat_progress(before_marker):
                    result = type(result)(status="success", state=self._state.raw if self._state is not None else None)
                else:
                    reward += self.reward_tracker.on_invalid_action()
                    recovery = self.protocol.recover_action_from_error(result.raw, self._state) or FlowAction("proceed")
                    retry = self.protocol.step(self._game_id, self.protocol.sanitize_action(self._state, recovery, random))
                    if retry.status != "success":
                        self._protocol_error_streak += 1
                        if self._is_deadlock_error(retry.raw):
                            self._deadlock_error_streak += 1
                            self._remember_deadlock_card(before_state, sanitized)
                        if self._should_poll_for_async_combat_progress(self._state, recovery) and self._poll_for_async_combat_progress(before_marker):
                            result = type(result)(status="success", state=self._state.raw if self._state is not None else None)
                        else:
                            should_probe_state = not (
                                self._state is None
                                or self._state.decision != "combat_play"
                                or self._is_shop_like_error(result.raw)
                                or self._is_shop_like_error(retry.raw)
                            )
                            if should_probe_state:
                                try:
                                    raw_state = self.protocol.get_state(self._game_id)
                                    probed_state = self.protocol.adapt_state(raw_state)
                                    progressed_from_probe = self._progress_marker(probed_state) != before_marker
                                    if progressed_from_probe:
                                        self._state = probed_state
                                        result = type(result)(status="success", state=raw_state)
                                        self._protocol_error_streak = 0
                                        self._deadlock_error_streak = 0
                                        time.sleep(self.config.recovery_delay_seconds)
                                    else:
                                        self._state = probed_state
                                        forced_combat_action = self._best_effort_combat_action(self._state) or FlowAction("proceed")
                                        forced_result = self.protocol.step(
                                            self._game_id,
                                            self.protocol.sanitize_action(self._state, forced_combat_action, random),
                                        )
                                        if forced_result.status == "success" and forced_result.state:
                                            result = forced_result
                                            self._state = self.protocol.adapt_state(forced_result.state)
                                            self._protocol_error_streak = 0
                                            self._deadlock_error_streak = 0
                                            reward += self.reward_tracker.on_invalid_action()
                                            time.sleep(self.config.recovery_delay_seconds)
                                        else:
                                            try:
                                                raw_state = self.protocol.get_state(self._game_id)
                                                refreshed_state = self.protocol.adapt_state(raw_state)
                                                if self._progress_marker(refreshed_state) != before_marker:
                                                    result = type(result)(status="success", state=raw_state)
                                                    self._state = refreshed_state
                                                    self._protocol_error_streak = 0
                                                    self._deadlock_error_streak = 0
                                                    reward += self.reward_tracker.on_invalid_action()
                                                    time.sleep(self.config.recovery_delay_seconds)
                                                else:
                                                    reward += self.reward_tracker.on_invalid_action()
                                                    abort_episode = True
                                                    termination_reason = "protocol_deadlock"
                                                    time.sleep(self.config.recovery_delay_seconds)
                                            except Exception:
                                                reward += self.reward_tracker.on_invalid_action()
                                                abort_episode = True
                                                termination_reason = "protocol_deadlock"
                                                time.sleep(self.config.recovery_delay_seconds)
                                except Exception:
                                    reward += self.reward_tracker.on_stuck()
                                    abort_episode = True
                                    termination_reason = "protocol_deadlock"
                            else:
                                reward += self.reward_tracker.on_stuck()
                                abort_episode = True
                                termination_reason = "protocol_deadlock"
                    else:
                        result = retry
            else:
                self._protocol_error_streak = 0
                self._deadlock_error_streak = 0

            if result.status != "success" and (
                self._deadlock_error_streak >= 2 or self._protocol_error_streak >= 4
            ):
                reward += self.reward_tracker.on_stuck()
                abort_episode = True
                termination_reason = "protocol_deadlock"

            if result.state:
                self._state = self.protocol.adapt_state(result.state)

            reward += self.reward_tracker.compute(self._state)
            time.sleep(self.config.step_delay_seconds)
            self._apply_transition_tracking(self._state)
            progressed = self._progress_marker(self._state) != before_marker
            if progressed:
                self._protocol_error_streak = 0
                self._deadlock_error_streak = 0

            forced_progress, forced_reward = self._force_progress_if_needed()
            reward += forced_reward
            if forced_progress and self._state is not None:
                reward += self.reward_tracker.compute(self._state)
                progressed = True
                self._protocol_error_streak = 0
                self._deadlock_error_streak = 0

            if progressed:
                self._step_count += 1

            if self._stagnant_steps >= self.config.stuck_abort_threshold:
                abort_episode = True
                termination_reason = termination_reason or "stuck_abort"

            terminated = bool(self._state.game_over)
            truncated = truncated or abort_episode or bool(self.config.max_steps is not None and self._step_count >= self.config.max_steps)
            if truncated and not terminated:
                reward += self.reward_tracker.on_stuck()
                termination_reason = termination_reason or "max_steps_exceeded"
            if terminated and self._state.game_over and termination_reason is None:
                termination_reason = "game_over"

            self._episode_reward += reward
            self._termination_reason = termination_reason
            self._record_trace_step(
                before_state=before_state,
                action=sanitized,
                after_state=self._state,
                reward=reward,
                status="success" if result.status == "success" else result.status,
            )
            obs = self._get_observation(self._state)
            info = {
                "decision": self._state.decision,
                "episode_reward": self._episode_reward,
                "episode_steps": self._step_count,
                "game_id": self._game_id,
                "state_summary": self._state.summary(),
            }
            if termination_reason:
                info["termination_reason"] = termination_reason
            self._write_current_slot(active=not (terminated or truncated), force=bool(terminated or truncated))
            if terminated or truncated:
                self._record_episode_summary(terminated=terminated, truncated=truncated)
                if self._game_id:
                    try:
                        self.protocol.close_game(self._game_id)
                    except Exception:
                        self.logger.warning(
                            "Failed to close finished game: slot=%s episode=%s seed=%s game_id=%s",
                            self.config.seed_offset,
                            self._episode_index,
                            self._seed,
                            self._game_id,
                            exc_info=True,
                        )
                    finally:
                        self._game_id = None
            return obs, reward, terminated, truncated, info
        except Exception as exc:
            return self._truncate_episode_on_exception(
                before_state=before_state,
                action=sanitized,
                exc=exc,
                termination_reason="step_exception",
            )

    def render(self) -> None:
        if self._state is None:
            return
        print(
            f"decision={self._state.decision} hp={self._state.hp}/{self._state.max_hp} "
            f"gold={self._state.gold} energy={self._state.energy} hand={len(self._state.hand)}"
        )

    def close(self) -> None:
        self._write_current_slot(active=False, force=True)
        if self._game_id:
            try:
                self.protocol.close_game(self._game_id)
            except Exception:
                pass
        self._game_id = None
        self._seed = None
        self._state = None
