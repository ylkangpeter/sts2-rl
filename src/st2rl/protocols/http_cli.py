# -*- coding: utf-8 -*-
"""HTTP CLI protocol adapter for sts2-cli."""

import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException

from st2rl.gameplay.heuristics import (
    best_shop_card,
    best_shop_potion,
    best_shop_relic,
    choose_card_reward,
    choose_event_option,
    choose_map_node_choice,
    choose_purge_target,
    choose_purge_targets,
    choose_upgrade_target,
    choose_upgrade_targets,
    is_shop_context,
    shop_purge_cost,
    should_prioritize_shop_purge,
    should_replace_potion,
    worst_owned_potion,
)
from st2rl.gameplay.types import FlowAction, GameStateView, card_needs_enemy_target
from st2rl.protocols.base import FlowProtocol, ProtocolStartResult, ProtocolStepResult

DECISION_ALIAS = {
    "map_select": "map_node",
    "event_choice": "event",
}

ACTION_ALIAS = {
    "choose_card_reward": "choose_card",
    "skip_reward": "skip_reward",
    "leave_shop": "leave_shop",
    "choose_option": "choose_option",
    "select_map_node": "select_map_node",
    "play_card": "play_card",
    "end_turn": "end_turn",
    "proceed": "proceed",
    "buy_card": "buy_card",
    "buy_relic": "buy_relic",
    "buy_potion": "buy_potion",
    "discard_potion": "discard_potion",
    "use_potion": "use_potion",
    "purge_card": "purge_card",
    "select_bundle": "select_bundle",
    "select_cards": "select_cards",
    "skip_select": "skip_select",
}

UNSUPPORTED_HEADLESS_CARD_IDS = {
    "CARD.WHIRLWIND",
}


def _combat_potion_mode(potion: Dict[str, Any]) -> str:
    target_type = str(potion.get("target_type") or "").strip().lower()
    description = str(potion.get("description") or "").strip().lower()
    if any(token in description for token in ("everyone", "all characters", "\u6240\u6709\u4eba")):
        return "unsafe"
    if not target_type:
        return "unsafe"
    if "allenemies" in target_type or "all_enemies" in target_type:
        return "safe_aoe"
    if "anyenemy" in target_type or "enemy" in target_type:
        return "safe_enemy"
    if "self" in target_type or "player" in target_type or target_type in {"none", "no_target"}:
        return "safe_self"
    return "unsafe"


def _is_buyable_shop_card(card: Dict[str, Any]) -> bool:
    if not isinstance(card, dict):
        return False
    index = card.get("index")
    if index is None:
        return False
    name = str(card.get("name") or "").strip()
    card_id = str(card.get("id") or "").strip()
    if not name or name.startswith("?."):
        return False
    if not card_id or card_id.lower() == "none":
        return False
    return True


def _is_placeholder_shop_relic(relic: Dict[str, Any]) -> bool:
    if not isinstance(relic, dict):
        return True
    name = str(relic.get("name") or "").strip()
    description = str(relic.get("description") or "").strip()
    if not name or name.startswith("?.") or name == "?":
        return True
    if description.startswith("?.") or description == "?":
        return True
    return False


def _is_headless_unsupported_card(card: Dict[str, Any]) -> bool:
    card_id = str(card.get("id") or "").strip().upper()
    return card_id in UNSUPPORTED_HEADLESS_CARD_IDS


def mark_headless_unsupported_card(card: Dict[str, Any] | None) -> None:
    if not isinstance(card, dict):
        return
    card_id = str(card.get("id") or "").strip().upper()
    if card_id:
        UNSUPPORTED_HEADLESS_CARD_IDS.add(card_id)


def _safe_int(value: Any, default: Optional[int] = 0) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _potion_slot_counts(state: GameStateView) -> tuple[int, int, int]:
    player = state.player or {}
    player_potions = [potion for potion in (player.get("potions") or []) if isinstance(potion, dict)]
    used = _safe_int(player.get("potion_slots_used"), len(player_potions)) or 0
    total = _safe_int(player.get("potion_slots_total"), 0) or 0
    if total <= 0:
        total = _safe_int(player.get("potion_slots"), 0) or 0
    if total <= 0:
        total = max(3, used)
    used = max(used, len(player_potions))
    return total, used, max(total - used, 0)


def _potion_heal_amount(potion: Dict[str, Any]) -> int:
    vars_payload = potion.get("vars") or {}
    if isinstance(vars_payload, dict):
        values = []
        for key, value in vars_payload.items():
            if any(token in str(key).lower() for token in ("heal", "health", "hp")):
                parsed = _safe_int(value, None)
                if parsed is not None and parsed > 0:
                    values.append(parsed)
        if values:
            return max(values)

    text = f"{potion.get('name') or ''} {potion.get('description') or ''}"
    lowered = text.lower()
    chinese_heal_tokens = ("恢复", "治疗", "回复", "生命")
    if not any(token in lowered for token in ("heal", "healing", "restore", "recover", "regain")) and not any(
        token in text for token in chinese_heal_tokens
    ):
        return 0
    digits = [int(match) for match in re.findall(r"\d+", text)]
    return max(digits) if digits else -1


def _is_beneficial_healing_potion(state: GameStateView, potion: Dict[str, Any]) -> bool:
    if state.max_hp <= state.hp:
        return False
    if potion.get("index") is None:
        return False
    target_type = str(potion.get("target_type") or "").strip().lower()
    if "enemy" in target_type:
        return False
    return _potion_heal_amount(potion) != 0


def _best_healing_potion_action(state: GameStateView) -> Optional[FlowAction]:
    missing_hp = max(state.max_hp - state.hp, 0)
    if missing_hp <= 0:
        return None

    candidates: list[tuple[tuple[int, int, int, int], Dict[str, Any]]] = []
    for potion in (state.player.get("potions") or []):
        if not isinstance(potion, dict) or not _is_beneficial_healing_potion(state, potion):
            continue
        amount = _potion_heal_amount(potion)
        unknown_amount = amount < 0
        effective_amount = missing_hp if unknown_amount else min(amount, missing_hp)
        waste = 0 if unknown_amount else max(amount - missing_hp, 0)
        candidates.append(
            (
                (
                    1 if unknown_amount else 0,
                    waste,
                    -effective_amount,
                    _safe_int(potion.get("index"), 9999) or 9999,
                ),
                potion,
            )
        )
    if not candidates:
        return None
    _, potion = min(candidates, key=lambda item: item[0])
    return FlowAction("use_potion", {"potion_index": potion.get("index", 0)})


def _card_stat(card: Dict[str, Any], *names: str) -> int:
    stats = card.get("stats") or {}
    if not isinstance(stats, dict):
        return 0
    lowered = {str(key).lower(): value for key, value in stats.items()}
    best = 0
    for name in names:
        best = max(best, _safe_int(lowered.get(name.lower()), 0) or 0)
    return best


def _card_draw(card: Dict[str, Any]) -> int:
    description = str(card.get("description") or "").lower()
    if "draw" in description or "抽" in description:
        return _card_stat(card, "cards", "draw")
    return _card_stat(card, "draw")


def _card_energy_gain(card: Dict[str, Any]) -> int:
    return _card_stat(card, "energy")


def _card_damage(card: Dict[str, Any]) -> int:
    return _card_stat(card, "damage")


def _card_block(card: Dict[str, Any]) -> int:
    return _card_stat(card, "block")


def _enemy_effective_hp(enemy: Dict[str, Any]) -> int:
    return max(0, (_safe_int(enemy.get("hp"), 0) or 0) + (_safe_int(enemy.get("block"), 0) or 0))


def _best_enemy_target(enemies: list[Dict[str, Any]], damage_hint: int = 0) -> Dict[str, Any] | None:
    if not enemies:
        return None

    def target_key(enemy: Dict[str, Any]) -> tuple[int, int, int, int]:
        ehp = _enemy_effective_hp(enemy)
        threat = _safe_int(enemy.get("intent_damage"), 0) or 0
        lethal = 1 if damage_hint > 0 and damage_hint >= ehp else 0
        return (
            -lethal,
            -threat,
            ehp,
            _safe_int(enemy.get("index"), 9999) or 9999,
        )

    return min(enemies, key=target_key)


def _best_playable_card(playable: list[Dict[str, Any]], state: GameStateView) -> Dict[str, Any] | None:
    if not playable:
        return None
    incoming_damage = sum((_safe_int(enemy.get("intent_damage"), 0) or 0) for enemy in state.living_enemies())

    def card_key(card: Dict[str, Any]) -> tuple[float, int]:
        damage = _card_damage(card)
        block = _card_block(card)
        draw = _card_draw(card)
        energy = _card_energy_gain(card)
        cost = _safe_int(card.get("cost"), 0) or 0
        block_gap = max(0, incoming_damage - state.block)
        prevented = min(block, block_gap)
        score = damage * 2.0 + prevented * 1.6 + draw * 8.0 + energy * 9.0
        if cost <= 0:
            score += 3.0
        else:
            score -= cost * 0.5
        return (-score, _safe_int(card.get("index"), 9999) or 9999)

    return min(playable, key=card_key)


def _first_valid_bundle(bundles: list[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not bundles:
        return None
    return min(
        bundles,
        key=lambda bundle: (
            _safe_int(bundle.get("cost"), 0) or 0,
            _safe_int(bundle.get("index"), 9999) or 9999,
        ),
    )


def _card_choice_text(card: Dict[str, Any]) -> str:
    return " ".join(
        str(card.get(key) or "").strip().lower()
        for key in ("id", "card_id", "name", "description", "type")
    )


def _is_curse_choice_state(state: GameStateView) -> bool:
    cards = [card for card in state.cards if isinstance(card, dict)]
    if not cards or state.min_select > 1:
        return False
    if len(cards) > 4:
        return False
    room_type = str((state.raw.get("context") or {}).get("room_type") or "").strip().lower()
    if not any(token in room_type for token in ("monster", "elite", "boss", "combat")):
        return False
    curse_like = 0
    for card in cards:
        text = _card_choice_text(card)
        card_type = str(card.get("type") or "").strip().lower()
        if card_type == "curse" or any(
            token in text
            for token in (
                "curse",
                "诅咒",
                "decay",
                "瓦解",
                "衰朽",
                "sloth",
                "lazy",
                "懒惰",
                "corruption",
                "腐化",
            )
        ):
            curse_like += 1
    return curse_like >= max(1, len(cards) - 1)


def _curse_choice_score(card: Dict[str, Any]) -> float:
    text = _card_choice_text(card)
    score = 0.0
    if any(token in text for token in ("decay", "瓦解", "衰朽", "每回合结束时失去生命", "end of turn lose")):
        score += 90.0
    if any(token in text for token in ("sloth", "lazy", "懒惰", "每回合最多打3张", "最多打出3张", "play up to 3")):
        score -= 180.0
    if any(token in text for token in ("writhe", "心灵腐化", "corruption", "腐化")):
        score -= 30.0
    return score


def _pick_preferred_curse_card(state: GameStateView) -> Dict[str, Any] | None:
    if not _is_curse_choice_state(state):
        return None
    cards = [card for card in state.cards if isinstance(card, dict)]
    if not cards:
        return None
    return max(
        cards,
        key=lambda card: (
            _curse_choice_score(card),
            -(_safe_int(card.get("index"), 9999) or 9999),
        ),
    )


@dataclass(slots=True)
class HttpCliProtocolConfig:
    """Transport and session settings for the HTTP CLI protocol."""

    base_url: str = "http://127.0.0.1:5000"
    timeout_seconds: int = 10
    game_dir: str | None = None
    close_retries: int = 5
    transport_retry_delay_seconds: float = 0.15
    close_retry_delay_seconds: float = 0.2


class HttpCliProtocol(FlowProtocol):
    """Protocol adapter for the `sts2-cli` HTTP service."""

    def __init__(self, config: HttpCliProtocolConfig | None = None):
        self.config = config or HttpCliProtocolConfig()
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session_lock = threading.Lock()
        self._pending_event_select: dict[str, Any] | None = None
        self._start_lock = threading.Lock()

    def _get_session(self) -> requests.Session:
        return self._session

    def _clear_pending_event_select(self) -> None:
        self._pending_event_select = None

    def _remember_pending_event_select(self, state: GameStateView, action: FlowAction) -> None:
        if action.name != "choose_option":
            self._clear_pending_event_select()
            return
        option_index = action.args.get("option_index")
        selected = next((option for option in state.options if option.get("index") == option_index), None)
        if not isinstance(selected, dict):
            selected = next((option for option in state.choices if option.get("index") == option_index), None)
        if not isinstance(selected, dict):
            self._clear_pending_event_select()
            return
        text = " ".join(
            str(selected.get(key) or "")
            for key in ("option_id", "name", "label", "title", "description", "text_key")
        ).lower()
        self._pending_event_select = {
            "floor": _safe_int((state.raw.get("context") or {}).get("floor"), 0),
            "text": text,
        }

    def _pending_event_select_text(self, state: GameStateView) -> str:
        pending = self._pending_event_select
        if not isinstance(pending, dict):
            return ""
        current_floor = _safe_int((state.raw.get("context") or {}).get("floor"), 0)
        if pending.get("floor") != current_floor:
            return ""
        return str(pending.get("text") or "")

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        retries: int = 0,
    ) -> Dict[str, Any]:
        session = self._get_session()
        url = f"{self.config.base_url}{path}"
        last_error: Optional[str] = None

        for attempt in range(retries + 1):
            response: Optional[requests.Response] = None
            try:
                timeout = max(1, min(int(self.config.timeout_seconds), 10))
                with self._session_lock:
                    if method == "GET":
                        response = session.get(url, timeout=timeout)
                    elif method == "POST":
                        response = session.post(url, json=payload, timeout=timeout)
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                try:
                    data = response.json()
                except Exception:
                    data = {"status": "error", "message": response.text}

                if response.status_code >= 400:
                    data.setdefault("status", "error")
                    data.setdefault("message", f"HTTP {response.status_code}")
                return data
            except RequestException as exc:
                last_error = str(exc)
                if attempt >= retries:
                    break
                time.sleep(self.config.transport_retry_delay_seconds * (attempt + 1))
            finally:
                if response is not None:
                    response.close()

        return {"status": "error", "message": last_error or "request failed", "request_path": path}

    def _detect_game_dir(self) -> Optional[str]:
        candidates = [
            os.environ.get("STS2_GAME_DIR"),
            self.config.game_dir,
            r"C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2",
        ]
        for path in candidates:
            if path and os.path.isdir(path):
                return path
        return None

    def _serialize_action(self, action: FlowAction) -> Dict[str, Any]:
        return {
            "cmd": "action",
            "action": ACTION_ALIAS.get(action.name, action.name),
            "args": dict(action.args),
        }

    def health_check(self, retries: int = 0) -> None:
        data = self._request("GET", "/health", retries=retries)
        if data.get("status") != "healthy":
            raise RuntimeError(f"Service health check failed: {data}")

    def _list_games(self) -> list[Dict[str, Any]]:
        data = self._request("GET", "/games", retries=1)
        if data.get("status") != "success":
            return []
        games = data.get("games")
        return list(games) if isinstance(games, list) else []

    def _list_workers(self) -> list[Dict[str, Any]]:
        data = self._request("GET", "/admin/worker_debug", retries=1)
        if data.get("status") != "success":
            return []
        workers = data.get("workers")
        return list(workers) if isinstance(workers, list) else []

    @staticmethod
    def _slot_matches_game(slot: int, game: Dict[str, Any]) -> bool:
        for key in ("worker_slot", "slot", "worker"):
            value = game.get(key)
            if value is None:
                continue
            try:
                if int(value) == int(slot):
                    return True
            except (TypeError, ValueError):
                pass

        slot_token = f"_{int(slot):02d}_"
        for key in ("seed", "game_id"):
            value = str(game.get(key) or "")
            if slot_token in value:
                return True
        return False

    def _close_worker_slot_game(self, worker_slot: int) -> bool:
        for game in self._list_games():
            if not isinstance(game, dict) or not self._slot_matches_game(worker_slot, game):
                continue
            game_id = str(game.get("game_id") or "").strip()
            if not game_id:
                continue
            result = self._request("POST", f"/close/{game_id}", payload={}, retries=1)
            if result.get("status") == "success":
                return True
        worker_key = f"slot_{int(worker_slot):02d}"
        for worker in self._list_workers():
            if not isinstance(worker, dict) or str(worker.get("worker_key") or "") != worker_key:
                continue
            game_id = str(worker.get("game_id") or "").strip()
            if not game_id:
                return False
            result = self._request("POST", f"/close/{game_id}", payload={}, retries=1)
            if result.get("status") == "success":
                return True
        return False

    def _reset_worker_slot(self, worker_slot: int) -> bool:
        worker_key = f"slot_{int(worker_slot):02d}"
        result = self._request("POST", f"/admin/workers/{worker_key}/reset", payload={}, retries=1)
        return result.get("status") == "success"

    def start_game(self, character: str, seed: str, worker_slot: int | None = None) -> ProtocolStartResult:
        with self._start_lock:
            payload = {"character": character, "seed": seed}
            if worker_slot is not None:
                payload["worker_slot"] = int(worker_slot)
            game_dir = self._detect_game_dir()
            if game_dir:
                payload["game_dir"] = game_dir

            attempts = 12 if worker_slot is not None else 1
            last_error: Dict[str, Any] | None = None
            for attempt in range(attempts):
                data = self._request("POST", "/start", payload=payload, retries=5)
                if data.get("status") == "success":
                    return ProtocolStartResult(game_id=data["game_id"], raw_state=data.get("state") or {})

                last_error = data
                message = str(data.get("message") or "")
                if worker_slot is None or attempt >= attempts - 1 or "busy" not in message.lower():
                    break
                if not self._close_worker_slot_game(int(worker_slot)):
                    self._reset_worker_slot(int(worker_slot))
                time.sleep(self.config.close_retry_delay_seconds)

        raise RuntimeError(f"Start game failed: {last_error}")

    def get_state(self, game_id: str) -> Dict[str, Any]:
        data = self._request("GET", f"/state/{game_id}", retries=2)
        if data.get("status") != "success":
            raise RuntimeError(f"Get state failed: {data}")
        return data["state"]

    def step(self, game_id: str, action: FlowAction) -> ProtocolStepResult:
        data = self._request("POST", f"/step/{game_id}", payload=self._serialize_action(action), retries=2)
        return ProtocolStepResult(
            status=str(data.get("status", "error")),
            state=data.get("state"),
            reward=float(data.get("reward", 0.0) or 0.0),
            message=str(data.get("message", "")),
            last_state=data.get("last_state"),
            raw=data,
        )

    def close_game(self, game_id: str) -> None:
        for _ in range(self.config.close_retries):
            result = self._request("POST", f"/close/{game_id}", payload={}, retries=1)
            if result.get("status") == "success":
                return
            time.sleep(self.config.close_retry_delay_seconds)

    def adapt_state(self, raw_state: Dict[str, Any]) -> GameStateView:
        return GameStateView.from_raw(raw_state, decision_alias=DECISION_ALIAS)

    def recover_action_from_error(
        self,
        error_payload: Dict[str, Any],
        state: GameStateView,
    ) -> Optional[FlowAction]:
        message = str(error_payload.get("message", ""))
        lower_message = message.lower()
        if "no free potion slot" in lower_message or "potion inventory full" in lower_message:
            if state.decision == "combat_play":
                return _best_healing_potion_action(state) or FlowAction("end_turn", {"_policy_allow_end_turn": True})
            return FlowAction("leave_shop", {"_force_if_stuck": True})
        if any(
            token in message
            for token in (
                "EnergyCostTooHigh",
                "Invalid card index",
                "Cannot play card",
                "Card could not be played",
                "Cannot use potion",
                "Use potion failed",
                "Potion action did not resolve",
            )
        ):
            return FlowAction("end_turn", {"_policy_allow_end_turn": True})
        if any(
            token in lower_message
            for token in (
                "too many cards",
                "play up to 3 cards",
                "cannot play more than 3",
                "max cards per turn",
                "lazy",
                "sloth",
                "懒惰",
                "最多打",
            )
        ):
            return FlowAction("end_turn", {"_policy_allow_end_turn": True})
        if any(
            token in message
            for token in (
                "Buy relic failed",
                "Buy potion failed",
                "Buy card failed",
                "Card already purchased",
                "Potion already purchased",
                "Relic already purchased",
                "buy_card",
                "buy_relic",
                "buy_potion",
                "purge_card",
                "remove_card",
            )
        ):
            return FlowAction("leave_shop", {"_force_if_stuck": True})
        if "No pending card selection" in message:
            return FlowAction("proceed", {"_force_if_stuck": True})
        if state.decision in ("map_node", "map_select") and "No pending card selection" in message:
            return FlowAction("proceed", {"_force_if_stuck": True})
        return None

    def should_resync_after_error(self, error_payload: Dict[str, Any], state: Optional[GameStateView] = None) -> bool:
        message = str(error_payload.get("message") or "").lower()
        if any(
            token in message
            for token in (
                "no pending card selection",
                "not in combat",
                "invalid card index",
                "cannot play card",
                "card could not be played",
            )
        ):
            return True
        if any(
            token in message
            for token in (
                "timed out",
                "timeout",
                "request failed",
                "connection refused",
                "connection reset",
                "connection aborted",
                "read timed out",
                "connect timeout",
            )
        ):
            return True
        if state is not None and state.decision == "card_select" and "selection" in message:
            return True
        return False

    def sanitize_action(self, state: GameStateView, action: FlowAction, rng: Any) -> FlowAction:
        safe = FlowAction(action.name, dict(action.args))
        if state.decision not in {"event", "card_select"}:
            self._clear_pending_event_select()

        if safe.name == "use_potion":
            selected_index = safe.args.get("potion_index")
            selected_potion = next(
                (
                    potion
                    for potion in (state.player.get("potions") or [])
                    if isinstance(potion, dict) and potion.get("index") == selected_index
                ),
                None,
            )
            if selected_potion is not None and _is_beneficial_healing_potion(state, selected_potion):
                return FlowAction("use_potion", {"potion_index": selected_potion.get("index", 0)})

        if state.decision == "combat_play":
            enemies = state.living_enemies()
            playable = [card for card in state.playable_cards() if not _is_headless_unsupported_card(card)]
            playable_by_index = {card.get("index"): card for card in playable}
            player_potions = [potion for potion in (state.player.get("potions") or []) if isinstance(potion, dict)]
            healing_potion_action = _best_healing_potion_action(state)
            force_proceed = bool(safe.args.pop("_force_if_stuck", False))
            allow_end_turn = bool(safe.args.pop("_policy_allow_end_turn", False))

            def _fallback_combat_action() -> FlowAction:
                if playable:
                    card = _best_playable_card(playable, state) or playable[0]
                    fallback_args = {"card_index": card.get("index")}
                    if card_needs_enemy_target(card) and enemies:
                        target = _best_enemy_target(enemies, _card_damage(card)) or enemies[0]
                        fallback_args["target_index"] = target.get("index")
                    return FlowAction("play_card", fallback_args)
                if healing_potion_action is not None:
                    return healing_potion_action
                if not enemies:
                    return FlowAction("proceed")
                return FlowAction("end_turn")

            if safe.name == "play_card":
                card_index = safe.args.get("card_index")
                if card_index not in playable_by_index:
                    if not playable:
                        return FlowAction("end_turn")
                    card = _best_playable_card(playable, state) or playable[0]
                    safe.args = {"card_index": card.get("index")}
                card = playable_by_index.get(safe.args.get("card_index"))
                if card and card_needs_enemy_target(card):
                    valid_targets = {enemy.get("index") for enemy in enemies}
                    if safe.args.get("target_index") not in valid_targets and enemies:
                        target = _best_enemy_target(enemies, _card_damage(card)) or enemies[0]
                        safe.args["target_index"] = target.get("index")
                else:
                    safe.args.pop("target_index", None)
                return safe

            if safe.name == "use_potion":
                valid_potions = {potion.get("index"): potion for potion in player_potions if potion.get("index") is not None}
                potion = valid_potions.get(safe.args.get("potion_index"))
                if potion is None:
                    if healing_potion_action is not None:
                        return healing_potion_action
                    return _fallback_combat_action()
                if _is_beneficial_healing_potion(state, potion):
                    return FlowAction("use_potion", {"potion_index": potion.get("index", 0)})
                potion_mode = _combat_potion_mode(potion)
                if potion_mode == "unsafe":
                    return _fallback_combat_action() if playable else FlowAction("end_turn")
                if potion_mode == "safe_enemy":
                    valid_targets = {enemy.get("index") for enemy in enemies}
                    if safe.args.get("target_index") not in valid_targets and enemies:
                        target = _best_enemy_target(enemies) or enemies[0]
                        safe.args["target_index"] = target.get("index")
                else:
                    safe.args.pop("target_index", None)
                return safe

            if safe.name == "proceed":
                if force_proceed or not enemies:
                    return FlowAction("proceed")
                if playable or healing_potion_action is not None:
                    return _fallback_combat_action()
                return FlowAction("end_turn")

            if safe.name == "end_turn":
                if allow_end_turn:
                    return FlowAction("end_turn")
                if playable or healing_potion_action is not None:
                    return _fallback_combat_action()
                if not enemies:
                    return FlowAction("proceed")
                return FlowAction("end_turn")

            return _fallback_combat_action()

        if state.decision in ("map_node", "map_select"):
            if state.choices:
                choice = choose_map_node_choice(state.choices, state) or state.choices[0]
                return FlowAction("select_map_node", {"col": choice.get("col"), "row": choice.get("row")})
            return FlowAction("proceed")

        if state.decision == "card_reward":
            if state.cards:
                picked = choose_card_reward(state.cards, list(state.player.get("deck") or []), state.can_skip, state)
                if picked is not None:
                    return FlowAction("choose_card_reward", {"card_index": picked.get("index", 0)})
            return FlowAction("skip_reward")

        if state.decision == "shop":
            _, _, free_potion_slots = _potion_slot_counts(state)
            affordable_cards = [
                card for card in state.cards if isinstance(card, dict) and _is_buyable_shop_card(card) and card.get("cost", 9999) <= state.gold
            ]
            affordable_shop_potions = [
                potion for potion in state.potions if isinstance(potion, dict) and potion.get("cost", 9999) <= state.gold
            ]
            affordable_potions = affordable_shop_potions if free_potion_slots > 0 else []
            affordable_relics = [
                relic
                for relic in state.relics
                if isinstance(relic, dict) and relic.get("cost", 9999) <= state.gold and not _is_placeholder_shop_relic(relic)
            ]
            deck = [card for card in (state.player.get("deck") or []) if isinstance(card, dict)]
            purge_cost = shop_purge_cost(state)
            best_relic = best_shop_relic(affordable_relics, state.gold)
            best_card = best_shop_card(affordable_cards, state.gold, deck)
            best_potion = best_shop_potion(affordable_shop_potions, state.gold)
            replacement_target = should_replace_potion(state.player.get("potions") or [], best_potion) if free_potion_slots <= 0 else None
            purge_target = choose_purge_target(deck)
            force_leave_shop = bool(safe.args.pop("_force_if_stuck", False))
            prioritize_purge = should_prioritize_shop_purge(
                deck,
                state.gold,
                purge_cost,
                best_card=best_card,
                best_relic=best_relic,
            )
            if safe.name == "leave_shop":
                if force_leave_shop:
                    return FlowAction("leave_shop")
                if replacement_target is not None:
                    return FlowAction("discard_potion", {"potion_index": replacement_target.get("index", 0)})
                if prioritize_purge and purge_target is not None:
                    return FlowAction("purge_card", {"card_index": purge_target.get("index", 0)})
                if state.gold >= 150 and best_relic is not None:
                    return FlowAction("buy_relic", {"relic_index": best_relic.get("index", 0)})
                if best_card is not None:
                    return FlowAction("buy_card", {"card_index": best_card.get("index", 0)})
                return FlowAction("leave_shop")
            if safe.name == "discard_potion":
                discardable = worst_owned_potion(state.player.get("potions") or [])
                if replacement_target is not None and discardable is not None and safe.args.get("potion_index") == discardable.get("index"):
                    return safe
                if replacement_target is not None:
                    return FlowAction("discard_potion", {"potion_index": replacement_target.get("index", 0)})
            if replacement_target is not None:
                return FlowAction("discard_potion", {"potion_index": replacement_target.get("index", 0)})
            if prioritize_purge and purge_target is not None:
                return FlowAction("purge_card", {"card_index": purge_target.get("index", 0)})
            if state.gold < 150 and safe.name not in {"buy_relic", "buy_card", "buy_potion", "purge_card"}:
                return FlowAction("leave_shop")
            if best_relic is not None:
                return FlowAction("buy_relic", {"relic_index": best_relic.get("index", 0)})
            if best_card is not None:
                return FlowAction("buy_card", {"card_index": best_card.get("index", 0)})
            if state.gold >= purge_cost and purge_target is not None:
                return FlowAction("purge_card", {"card_index": purge_target.get("index", 0)})
            if state.gold >= 250 and best_potion is not None:
                return FlowAction("buy_potion", {"potion_index": best_potion.get("index", 0)})
            return FlowAction("leave_shop")

        if state.decision in ("event", "event_choice"):
            options = state.options or state.choices
            unlocked = [option for option in options if not option.get("is_locked")]
            if unlocked:
                pick = choose_event_option(unlocked, state) or unlocked[0]
                event_action = FlowAction("choose_option", {"option_index": pick.get("index", 0)})
                self._remember_pending_event_select(state, event_action)
                return event_action
            self._clear_pending_event_select()
            return FlowAction("proceed")

        if state.decision == "rest_site":
            enabled = [option for option in state.options if option.get("is_enabled", True)]
            if enabled:
                heal = next(
                    (
                        option
                        for option in enabled
                        if str(option.get("option_id") or option.get("name") or option.get("label") or "").lower() in {"heal", "rest"}
                        or "heal" in str(option.get("name") or option.get("label") or "").lower()
                        or "rest" in str(option.get("name") or option.get("label") or "").lower()
                    ),
                    None,
                )
                smith = next(
                    (
                        option
                        for option in enabled
                        if "smith" in str(option.get("option_id") or option.get("name") or option.get("label") or "").lower()
                        or "upgrade" in str(option.get("option_id") or option.get("name") or option.get("label") or "").lower()
                    ),
                    None,
                )
                context = state.raw.get("context") or {}
                act = _safe_int(context.get("act"), 0) or 0
                floor = _safe_int(context.get("floor"), 0) or 0
                heal_threshold = 0.62
                if act >= 2:
                    heal_threshold = max(heal_threshold, 0.82)
                    if floor >= 12:
                        heal_threshold = max(heal_threshold, 0.88)
                elif floor <= 8:
                    heal_threshold = max(heal_threshold, 0.7)
                elif floor >= 14:
                    heal_threshold = max(heal_threshold, 0.88)
                elif floor >= 12:
                    heal_threshold = max(heal_threshold, 0.72)
                if state.hp / max(1, state.max_hp) < heal_threshold and heal is not None:
                    return FlowAction("choose_option", {"option_index": heal.get("index", 0)})
                if smith is not None:
                    return FlowAction("choose_option", {"option_index": smith.get("index", 0)})
                return FlowAction("choose_option", {"option_index": enabled[0].get("index", 0)})
            return FlowAction("proceed")

        if state.decision == "bundle_select":
            valid = {bundle.get("index") for bundle in state.bundles}
            if safe.name == "select_bundle" and safe.args.get("bundle_index") in valid:
                return safe
            if state.bundles:
                bundle = _first_valid_bundle(state.bundles) or state.bundles[0]
                return FlowAction("select_bundle", {"bundle_index": bundle.get("index", 0)})
            return FlowAction("proceed")

        if state.decision == "card_select":
            valid = {str(card.get("index", index)) for index, card in enumerate(state.cards)}
            pick_count = max(1, min(max(1, state.min_select), state.max_select, len(state.cards)))
            room_type = str((state.raw.get("context") or {}).get("room_type") or "").strip().lower()
            is_combat_select = any(token in room_type for token in ("monster", "elite", "boss", "combat"))
            event_select_text = self._pending_event_select_text(state) if "event" in room_type else ""
            is_event_purge_like = any(
                token in event_select_text
                for token in ("transform", "remove", "purge", "变化", "变形", "移除", "删除", "删牌")
            )
            is_event_upgrade_like = any(token in event_select_text for token in ("upgrade", "smith", "升级", "锻造"))

            def _recommended_select_action() -> FlowAction | None:
                if not state.cards:
                    return None
                preferred_curse = _pick_preferred_curse_card(state)
                if preferred_curse is not None:
                    return FlowAction("select_cards", {"indices": str(preferred_curse.get("index", 0))})
                if is_shop_context(state):
                    purge_targets = choose_purge_targets(state.cards, pick_count)
                    if len(purge_targets) >= pick_count:
                        return FlowAction("select_cards", {"indices": ",".join(str(card.get("index", 0)) for card in purge_targets)})
                if is_combat_select:
                    exhaust_targets = choose_purge_targets(state.cards, pick_count)
                    if len(exhaust_targets) >= pick_count:
                        return FlowAction("select_cards", {"indices": ",".join(str(card.get("index", 0)) for card in exhaust_targets)})
                if is_event_purge_like and not is_event_upgrade_like:
                    event_targets = choose_purge_targets(state.cards, pick_count)
                    if len(event_targets) >= pick_count:
                        return FlowAction("select_cards", {"indices": ",".join(str(card.get("index", 0)) for card in event_targets)})
                upgrade_targets = choose_upgrade_targets(state.cards, pick_count, state)
                if len(upgrade_targets) >= pick_count:
                    return FlowAction("select_cards", {"indices": ",".join(str(card.get("index", 0)) for card in upgrade_targets)})
                return None

            if safe.name == "skip_select" and state.min_select == 0:
                self._clear_pending_event_select()
                return safe
            if safe.name == "select_cards":
                raw_indices = str(safe.args.get("indices", ""))
                picks = [item.strip() for item in raw_indices.split(",") if item.strip()]
                if picks and all(item in valid for item in picks) and state.min_select <= len(picks) <= state.max_select:
                    recommended = _recommended_select_action()
                    self._clear_pending_event_select()
                    return recommended or safe
            if not state.cards:
                self._clear_pending_event_select()
                return FlowAction("skip_select")
            recommended = _recommended_select_action()
            self._clear_pending_event_select()
            if recommended is not None:
                return recommended
            indices = [str(card.get("index", index)) for index, card in enumerate(state.cards)]
            self._clear_pending_event_select()
            return FlowAction("select_cards", {"indices": ",".join(indices[: max(1, state.min_select)])})

        return safe

