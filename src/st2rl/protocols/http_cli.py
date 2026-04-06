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
    choose_purge_target,
    choose_upgrade_target,
    is_shop_context,
    shop_purge_cost,
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
    if not target_type:
        return "unsafe"
    if "allenemies" in target_type or "all_enemies" in target_type:
        return "safe_aoe"
    if "anyenemy" in target_type or "enemy" in target_type:
        return "safe_enemy"
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


@dataclass(slots=True)
class HttpCliProtocolConfig:
    """Transport and session settings for the HTTP CLI protocol."""

    base_url: str = "http://localhost:5000"
    timeout_seconds: int = 30
    game_dir: str | None = None
    close_retries: int = 5
    transport_retry_delay_seconds: float = 0.15
    close_retry_delay_seconds: float = 0.2


class HttpCliProtocol(FlowProtocol):
    """Protocol adapter for the `sts2-cli` HTTP service."""

    def __init__(self, config: HttpCliProtocolConfig | None = None):
        self.config = config or HttpCliProtocolConfig()
        self._session_local = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=0)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._session_local.session = session
        return session

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
                if method == "GET":
                    response = session.get(url, timeout=self.config.timeout_seconds)
                elif method == "POST":
                    response = session.post(url, json=payload, timeout=self.config.timeout_seconds)
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

    @staticmethod
    def _slot_matches_game(slot: int, game: Dict[str, Any]) -> bool:
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
        return False

    def start_game(self, character: str, seed: str, worker_slot: int | None = None) -> ProtocolStartResult:
        payload = {"character": character, "seed": seed}
        if worker_slot is not None:
            payload["worker_slot"] = int(worker_slot)
        game_dir = self._detect_game_dir()
        if game_dir:
            payload["game_dir"] = game_dir

        attempts = 2 if worker_slot is not None else 1
        last_error: Dict[str, Any] | None = None
        for attempt in range(attempts):
            data = self._request("POST", "/start", payload=payload)
            if data.get("status") == "success":
                return ProtocolStartResult(game_id=data["game_id"], raw_state=data.get("state") or {})

            last_error = data
            message = str(data.get("message") or "")
            if (
                worker_slot is None
                or attempt >= attempts - 1
                or "busy" not in message.lower()
                or not self._close_worker_slot_game(int(worker_slot))
            ):
                break
            time.sleep(self.config.close_retry_delay_seconds)

        raise RuntimeError(f"Start game failed: {last_error}")

    def get_state(self, game_id: str) -> Dict[str, Any]:
        data = self._request("GET", f"/state/{game_id}", retries=2)
        if data.get("status") != "success":
            raise RuntimeError(f"Get state failed: {data}")
        return data["state"]

    def step(self, game_id: str, action: FlowAction) -> ProtocolStepResult:
        data = self._request("POST", f"/step/{game_id}", payload=self._serialize_action(action))
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
            return _best_healing_potion_action(state) or FlowAction("leave_shop")
        if any(token in message for token in ("EnergyCostTooHigh", "Invalid card index", "Cannot play card", "Cannot use potion", "Use potion failed", "Potion action did not resolve")):
            return FlowAction("end_turn")
        if any(token in message for token in ("Buy relic failed", "Buy potion failed", "Buy card failed", "Card already purchased", "buy_card", "buy_relic", "buy_potion", "purge_card", "remove_card")):
            return FlowAction("leave_shop")
        if "No pending card selection" in message:
            return FlowAction("proceed")
        if state.decision in ("map_node", "map_select") and "No pending card selection" in message:
            return FlowAction("proceed")
        return None

    def sanitize_action(self, state: GameStateView, action: FlowAction, rng: Any) -> FlowAction:
        safe = FlowAction(action.name, dict(action.args))

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

            def _fallback_combat_action() -> FlowAction:
                if playable:
                    card = rng.choice(playable)
                    fallback_args = {"card_index": card.get("index")}
                    if card_needs_enemy_target(card) and enemies:
                        fallback_args["target_index"] = rng.choice(enemies).get("index")
                    return FlowAction("play_card", fallback_args)
                if healing_potion_action is not None:
                    return healing_potion_action
                safe_potions = [potion for potion in player_potions if _combat_potion_mode(potion) != "unsafe"]
                if safe_potions:
                    potion = rng.choice(safe_potions)
                    fallback_args = {"potion_index": potion.get("index", 0)}
                    if _combat_potion_mode(potion) == "safe_enemy" and enemies:
                        fallback_args["target_index"] = rng.choice(enemies).get("index")
                    return FlowAction("use_potion", fallback_args)
                if not enemies:
                    return FlowAction("proceed")
                return FlowAction("end_turn")

            if safe.name == "play_card":
                card_index = safe.args.get("card_index")
                if card_index not in playable_by_index:
                    if not playable:
                        return FlowAction("end_turn")
                    card = rng.choice(playable)
                    safe.args = {"card_index": card.get("index")}
                card = playable_by_index.get(safe.args.get("card_index"))
                if card and card_needs_enemy_target(card):
                    valid_targets = {enemy.get("index") for enemy in enemies}
                    if safe.args.get("target_index") not in valid_targets and enemies:
                        safe.args["target_index"] = rng.choice(enemies).get("index")
                else:
                    safe.args.pop("target_index", None)
                return safe

            if safe.name == "use_potion":
                valid_potions = {potion.get("index"): potion for potion in player_potions if potion.get("index") is not None}
                potion = valid_potions.get(safe.args.get("potion_index"))
                if potion is None:
                    if healing_potion_action is not None:
                        return healing_potion_action
                    if not valid_potions:
                        return FlowAction("end_turn")
                    potion = rng.choice(list(valid_potions.values()))
                    safe.args = {"potion_index": potion.get("index")}
                if _is_beneficial_healing_potion(state, potion):
                    return FlowAction("use_potion", {"potion_index": potion.get("index", 0)})
                potion_mode = _combat_potion_mode(potion)
                if potion_mode == "unsafe":
                    return _fallback_combat_action() if playable else FlowAction("end_turn")
                if potion_mode == "safe_enemy":
                    valid_targets = {enemy.get("index") for enemy in enemies}
                    if safe.args.get("target_index") not in valid_targets and enemies:
                        safe.args["target_index"] = rng.choice(enemies).get("index")
                else:
                    safe.args.pop("target_index", None)
                return safe

            if safe.name == "proceed":
                if force_proceed or not enemies:
                    return FlowAction("proceed")
                if playable or healing_potion_action is not None or player_potions:
                    return _fallback_combat_action()
                return FlowAction("end_turn")

            if safe.name == "end_turn":
                if playable or player_potions:
                    return _fallback_combat_action()
                if not enemies:
                    return FlowAction("proceed")
                return FlowAction("end_turn")

            return _fallback_combat_action()

        if state.decision in ("map_node", "map_select"):
            valid = {(choice.get("col"), choice.get("row")) for choice in state.choices}
            if safe.name == "select_map_node" and safe.args.get("choice_index") is not None and state.choices:
                choice_index = int(safe.args.get("choice_index") or 0)
                mapped_choice = state.choices[min(max(choice_index, 0), len(state.choices) - 1)]
                return FlowAction("select_map_node", {"col": mapped_choice.get("col"), "row": mapped_choice.get("row")})
            if safe.name != "select_map_node":
                if state.choices:
                    choice = rng.choice(state.choices)
                    return FlowAction("select_map_node", {"col": choice.get("col"), "row": choice.get("row")})
                return FlowAction("proceed")
            if (safe.args.get("col"), safe.args.get("row")) not in valid and state.choices:
                choice = rng.choice(state.choices)
                return FlowAction("select_map_node", {"col": choice.get("col"), "row": choice.get("row")})
            return safe

        if state.decision == "card_reward":
            valid_cards = {card.get("index") for card in state.cards}
            if safe.name == "choose_card_reward" and safe.args.get("card_index") in valid_cards:
                return safe
            if safe.name == "skip_reward" and state.can_skip:
                return safe
            if state.cards:
                picked = choose_card_reward(state.cards, list(state.player.get("deck") or []), state.can_skip)
                if picked is not None:
                    return FlowAction("choose_card_reward", {"card_index": picked.get("index", 0)})
            return FlowAction("skip_reward")

        if state.decision == "shop":
            _, _, free_potion_slots = _potion_slot_counts(state)
            healing_potion_action = _best_healing_potion_action(state)
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
            if safe.name == "leave_shop":
                if replacement_target is not None:
                    return FlowAction("discard_potion", {"potion_index": replacement_target.get("index", 0)})
                if free_potion_slots <= 0 and affordable_shop_potions and healing_potion_action is not None:
                    return healing_potion_action
                if state.gold >= 150 and best_relic is not None:
                    return FlowAction("buy_relic", {"relic_index": best_relic.get("index", 0)})
                if state.gold >= purge_cost and purge_target is not None:
                    return FlowAction("purge_card", {"card_index": purge_target.get("index", 0)})
                if best_card is not None:
                    return FlowAction("buy_card", {"card_index": best_card.get("index", 0)})
                return FlowAction("leave_shop")
            if safe.name == "buy_card":
                valid = {card.get("index") for card in affordable_cards}
                if safe.args.get("card_index") in valid:
                    if best_relic is not None:
                        return FlowAction("buy_relic", {"relic_index": best_relic.get("index", 0)})
                    if best_card is not None:
                        return FlowAction("buy_card", {"card_index": best_card.get("index", 0)})
                    return safe
            if safe.name == "buy_potion":
                if replacement_target is not None:
                    return FlowAction("discard_potion", {"potion_index": replacement_target.get("index", 0)})
                if free_potion_slots <= 0 and affordable_shop_potions and healing_potion_action is not None:
                    return healing_potion_action
                valid = {potion.get("index") for potion in affordable_potions}
                if safe.args.get("potion_index") in valid:
                    if best_relic is not None:
                        return FlowAction("buy_relic", {"relic_index": best_relic.get("index", 0)})
                    if best_card is not None and bool(best_card.get("on_sale")):
                        return FlowAction("buy_card", {"card_index": best_card.get("index", 0)})
                    if best_potion is not None:
                        return FlowAction("buy_potion", {"potion_index": best_potion.get("index", 0)})
                    return safe
            if safe.name == "buy_relic":
                valid = {relic.get("index") for relic in affordable_relics}
                if safe.args.get("relic_index") in valid:
                    if best_relic is not None:
                        return FlowAction("buy_relic", {"relic_index": best_relic.get("index", 0)})
                    return safe
            if safe.name == "purge_card":
                valid = {card.get("index") for card in deck if state.gold >= purge_cost}
                if safe.args.get("card_index") in valid:
                    if purge_target is not None:
                        return FlowAction("purge_card", {"card_index": purge_target.get("index", 0)})
                    return safe
            if safe.name == "discard_potion":
                discardable = worst_owned_potion(state.player.get("potions") or [])
                if replacement_target is not None and discardable is not None and safe.args.get("potion_index") == discardable.get("index"):
                    return safe
                if replacement_target is not None:
                    return FlowAction("discard_potion", {"potion_index": replacement_target.get("index", 0)})
            if free_potion_slots <= 0 and affordable_shop_potions and healing_potion_action is not None:
                return healing_potion_action
            if replacement_target is not None:
                return FlowAction("discard_potion", {"potion_index": replacement_target.get("index", 0)})
            if state.gold < 150 and safe.name not in {"buy_relic", "buy_card", "buy_potion", "purge_card"}:
                return FlowAction("leave_shop")
            if best_relic is not None:
                return FlowAction("buy_relic", {"relic_index": best_relic.get("index", 0)})
            if state.gold >= purge_cost and purge_target is not None:
                return FlowAction("purge_card", {"card_index": purge_target.get("index", 0)})
            if best_card is not None:
                return FlowAction("buy_card", {"card_index": best_card.get("index", 0)})
            if state.gold >= 250 and best_potion is not None:
                return FlowAction("buy_potion", {"potion_index": best_potion.get("index", 0)})
            return FlowAction("leave_shop")

        if state.decision in ("event", "event_choice"):
            options = state.options or state.choices
            valid = {option.get("index") for option in options if not option.get("is_locked")}
            if safe.name == "choose_option" and safe.args.get("option_index") in valid:
                return safe
            unlocked = [option for option in options if not option.get("is_locked")]
            if unlocked:
                return FlowAction("choose_option", {"option_index": rng.choice(unlocked).get("index", 0)})
            return FlowAction("proceed")

        if state.decision == "rest_site":
            valid = {option.get("index") for option in state.options if option.get("is_enabled", True)}
            if safe.name == "choose_option" and safe.args.get("option_index") in valid:
                return safe
            enabled = [option for option in state.options if option.get("is_enabled", True)]
            if enabled:
                return FlowAction("choose_option", {"option_index": rng.choice(enabled).get("index", 0)})
            return FlowAction("proceed")

        if state.decision == "bundle_select":
            valid = {bundle.get("index") for bundle in state.bundles}
            if safe.name == "select_bundle" and safe.args.get("bundle_index") in valid:
                return safe
            if state.bundles:
                return FlowAction("select_bundle", {"bundle_index": rng.choice(state.bundles).get("index", 0)})
            return FlowAction("proceed")

        if state.decision == "card_select":
            valid = {str(card.get("index", index)) for index, card in enumerate(state.cards)}
            if safe.name == "skip_select" and state.min_select == 0:
                return safe
            if safe.name == "select_cards":
                raw_indices = str(safe.args.get("indices", ""))
                picks = [item.strip() for item in raw_indices.split(",") if item.strip()]
                if picks and all(item in valid for item in picks) and state.min_select <= len(picks) <= state.max_select:
                    if is_shop_context(state):
                        purge_target = choose_purge_target(state.cards)
                        if purge_target is not None:
                            return FlowAction("select_cards", {"indices": str(purge_target.get("index", 0))})
                    upgrade_target = choose_upgrade_target(state.cards)
                    if upgrade_target is not None:
                        return FlowAction("select_cards", {"indices": str(upgrade_target.get("index", 0))})
                    return safe
            if not state.cards:
                return FlowAction("skip_select")
            if is_shop_context(state):
                purge_target = choose_purge_target(state.cards)
                if purge_target is not None:
                    return FlowAction("select_cards", {"indices": str(purge_target.get("index", 0))})
            upgrade_target = choose_upgrade_target(state.cards)
            if upgrade_target is not None:
                return FlowAction("select_cards", {"indices": str(upgrade_target.get("index", 0))})
            indices = [str(card.get("index", index)) for index, card in enumerate(state.cards)]
            return FlowAction("select_cards", {"indices": ",".join(indices[: max(1, state.min_select)])})

        return safe

