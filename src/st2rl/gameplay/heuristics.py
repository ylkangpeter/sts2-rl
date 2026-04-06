# -*- coding: utf-8 -*-
"""Shared heuristic helpers for flow policy and protocol sanitization."""

from __future__ import annotations

from typing import Any

from st2rl.gameplay.types import GameStateView

_NEGATIVE_TYPES = {"curse", "status"}
_BASIC_STRIKE_TOKENS = ("strike",)
_BASIC_DEFEND_TOKENS = ("defend",)
_EXHAUST_TOKENS = ("exhaust", "consume", "\u6d88\u8017")
_UTILITY_TOKENS = (
    "draw",
    "gain",
    "apply",
    "retain",
    "discard",
    "energy",
    "block",
    "damage",
    "heal",
    "strength",
    "vulnerable",
    "weak",
    "draw",
    "\u62bd\u724c",
    "\u83b7\u5f97",
    "\u65bd\u52a0",
    "\u4fdd\u7559",
    "\u5f03\u724c",
    "\u80fd\u91cf",
    "\u683c\u6321",
    "\u4f24\u5bb3",
    "\u6062\u590d",
    "\u529b\u91cf",
    "\u6613\u4f24",
    "\u865a\u5f31",
)
_EVENT_POSITIVE_TOKENS = (
    "relic",
    "potion",
    "remove",
    "purge",
    "upgrade",
    "smith",
    "transform",
    "gold",
    "max hp",
    "maxhp",
    "\u9057\u7269",
    "\u836f\u6c34",
    "\u5220\u724c",
    "\u79fb\u9664",
    "\u5347\u7ea7",
    "\u953b\u9020",
    "\u53d8\u5316",
    "\u91d1\u5e01",
    "\u6700\u5927\u751f\u547d",
)
_EVENT_NEGATIVE_TOKENS = (
    "curse",
    "lose hp",
    "lose max hp",
    "pay hp",
    "take damage",
    "damage",
    "injure",
    "\u8bc5\u5492",
    "\u5931\u53bb\u751f\u547d",
    "\u6263\u9664\u751f\u547d",
    "\u53d7\u4f24",
    "\u4f24\u5bb3",
)


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _stats_total(card: dict[str, Any]) -> int:
    stats = card.get("stats") or {}
    if not isinstance(stats, dict):
        return 0
    total = 0
    for value in stats.values():
        total += max(0, _safe_int(value, 0))
    return total


def _description(card: dict[str, Any]) -> str:
    return _text(card.get("description"))


def _keywords(card: dict[str, Any]) -> tuple[str, ...]:
    raw = card.get("keywords") or []
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(_text(item) for item in raw if item)


def is_basic_strike(card: dict[str, Any]) -> bool:
    text = f"{_text(card.get('name'))} {_text(card.get('id'))}"
    return any(token in text for token in _BASIC_STRIKE_TOKENS)


def is_basic_defend(card: dict[str, Any]) -> bool:
    text = f"{_text(card.get('name'))} {_text(card.get('id'))}"
    return any(token in text for token in _BASIC_DEFEND_TOKENS)


def is_negative_card(card: dict[str, Any]) -> bool:
    card_type = _text(card.get("type"))
    if card_type in _NEGATIVE_TYPES:
        return True
    text = f"{_text(card.get('name'))} {_text(card.get('id'))}"
    return any(token in text for token in ("curse", "status", "wound", "burn", "dazed", "void", "regret", "shame"))


def is_exhaust_card(card: dict[str, Any]) -> bool:
    description = _description(card)
    return any(token in description for token in _EXHAUST_TOKENS) or any(
        token in keyword for keyword in _keywords(card) for token in _EXHAUST_TOKENS
    )


def is_low_impact_card(card: dict[str, Any]) -> bool:
    if is_negative_card(card):
        return True
    if is_basic_strike(card) or is_basic_defend(card):
        return False
    description = _description(card)
    stats_total = _stats_total(card)
    if stats_total <= 0 and not description:
        return True
    return stats_total <= 0 and not any(token in description for token in _UTILITY_TOKENS)


def _rarity_value(value: Any) -> float:
    rarity = _text(value)
    mapping = {
        "token": 6.0,
        "ancient": 5.5,
        "rare": 4.0,
        "uncommon": 2.5,
        "common": 1.0,
        "basic": -1.5,
        "starter": -1.0,
        "shop": 2.0,
        "event": 1.5,
        "boss": 4.5,
        "curse": -8.0,
        "status": -8.0,
    }
    return mapping.get(rarity, -2.0)


def _deck_counts(deck: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "attack": 0,
        "skill": 0,
        "power": 0,
        "high_cost": 0,
        "exhaust": 0,
    }
    for card in deck:
        if not isinstance(card, dict):
            continue
        card_type = _text(card.get("type"))
        if card_type in counts:
            counts[card_type] += 1
        if _safe_int(card.get("cost"), 0) >= 2:
            counts["high_cost"] += 1
        if is_exhaust_card(card):
            counts["exhaust"] += 1
    return counts


def estimate_card_reward_score(card: dict[str, Any], deck: list[dict[str, Any]] | None = None) -> float:
    deck = [item for item in (deck or []) if isinstance(item, dict)]
    if is_negative_card(card):
        return -100.0

    score = _rarity_value(card.get("rarity"))
    score += min(_stats_total(card), 24) * 0.12
    score += min(sum(1 for token in _UTILITY_TOKENS if token in _description(card)), 4) * 0.35
    if card.get("after_upgrade"):
        score += 0.35

    if is_basic_strike(card) or is_basic_defend(card):
        score -= 3.5

    if is_exhaust_card(card):
        score -= 1.2

    cost = _safe_int(card.get("cost"), 0)
    counts = _deck_counts(deck)
    if cost >= 2 and counts["high_cost"] >= 3:
        score -= 0.8
    if is_exhaust_card(card) and counts["exhaust"] >= 3:
        score -= 0.6

    card_type = _text(card.get("type"))
    if card_type == "skill" and counts["attack"] > counts["skill"] + 1:
        score += 0.45
    if card_type == "attack" and counts["skill"] > counts["attack"] + 2:
        score += 0.25
    if card_type == "power":
        score += 0.4 if counts["power"] < 2 else -0.2

    card_id = _text(card.get("id"))
    if card_id:
        duplicates = sum(1 for item in deck if _text(item.get("id")) == card_id)
        score -= min(duplicates, 3) * 0.35

    return score


def choose_card_reward(cards: list[dict[str, Any]], deck: list[dict[str, Any]] | None, can_skip: bool) -> dict[str, Any] | None:
    indexed = [card for card in cards if isinstance(card, dict) and card.get("index") is not None]
    if not indexed:
        return None
    scored = sorted(
        ((estimate_card_reward_score(card, deck), card) for card in indexed),
        key=lambda item: (-item[0], _safe_int(item[1].get("index"), 9999)),
    )
    best_score, best_card = scored[0]
    deck_size = len(deck or [])
    if can_skip:
        if best_score < 1.25:
            return None
        if deck_size >= 12 and (is_basic_strike(best_card) or is_basic_defend(best_card)):
            return None
        if is_low_impact_card(best_card) and best_score < 2.0:
            return None
    return best_card


def choose_purge_target(deck: list[dict[str, Any]]) -> dict[str, Any] | None:
    indexed = [card for card in deck if isinstance(card, dict) and card.get("index") is not None]
    if not indexed:
        return None

    def score(card: dict[str, Any]) -> tuple[int, int, int, int]:
        return (
            0 if is_negative_card(card) else 1,
            0 if is_low_impact_card(card) else 1,
            0 if is_basic_strike(card) else 1,
            0 if is_basic_defend(card) else 1,
        )

    return min(indexed, key=score)


def choose_upgrade_target(cards: list[dict[str, Any]]) -> dict[str, Any] | None:
    indexed = [card for card in cards if isinstance(card, dict) and card.get("index") is not None]
    if not indexed:
        return None

    def score(card: dict[str, Any]) -> tuple[int, int, int, float, int]:
        return (
            1 if is_basic_strike(card) or is_basic_defend(card) else 0,
            1 if bool(card.get("upgraded")) else 0,
            0 if card.get("after_upgrade") else 1,
            -estimate_card_reward_score(card, cards),
            _safe_int(card.get("cost"), 99),
        )

    return min(indexed, key=score)


def shop_card_priority(card: dict[str, Any], deck: list[dict[str, Any]] | None = None) -> tuple[int, float, int, int]:
    return (
        0 if bool(card.get("on_sale")) else 1,
        -estimate_card_reward_score(card, deck),
        _safe_int(card.get("cost"), 9999),
        _safe_int(card.get("index"), 9999),
    )


def best_shop_card(cards: list[dict[str, Any]], gold: int, deck: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    affordable = [
        card
        for card in cards
        if isinstance(card, dict) and card.get("index") is not None and _safe_int(card.get("cost"), 9999) <= gold
    ]
    if not affordable:
        return None
    best = min(affordable, key=lambda card: shop_card_priority(card, deck))
    if estimate_card_reward_score(best, deck) < 1.5:
        return None
    return best


def estimate_relic_score(relic: dict[str, Any]) -> float:
    description = _text(relic.get("description"))
    score = 2.0
    score += min(sum(1 for token in _UTILITY_TOKENS if token in description), 4) * 0.3
    if any(token in description for token in ("shop", "\u5546\u5e97")):
        score += 0.2
    return score


def best_shop_relic(relics: list[dict[str, Any]], gold: int) -> dict[str, Any] | None:
    affordable = [
        relic
        for relic in relics
        if isinstance(relic, dict) and relic.get("index") is not None and _safe_int(relic.get("cost"), 9999) <= gold
    ]
    if not affordable:
        return None
    return min(
        affordable,
        key=lambda relic: (-estimate_relic_score(relic), _safe_int(relic.get("cost"), 9999), _safe_int(relic.get("index"), 9999)),
    )


def estimate_potion_score(potion: dict[str, Any]) -> float:
    description = _text(potion.get("description"))
    target_type = _text(potion.get("target_type"))
    score = _rarity_value(potion.get("rarity")) * 0.6 + 1.0
    score += min(sum(1 for token in _UTILITY_TOKENS if token in description), 4) * 0.35
    if "enemy" in target_type:
        score += 0.5
    if any(token in description for token in ("heal", "block", "strength", "\u6062\u590d", "\u683c\u6321", "\u529b\u91cf")):
        score += 0.5
    return score


def best_shop_potion(potions: list[dict[str, Any]], gold: int) -> dict[str, Any] | None:
    affordable = [
        potion
        for potion in potions
        if isinstance(potion, dict) and potion.get("index") is not None and _safe_int(potion.get("cost"), 9999) <= gold
    ]
    if not affordable:
        return None
    return min(
        affordable,
        key=lambda potion: (-estimate_potion_score(potion), _safe_int(potion.get("cost"), 9999), _safe_int(potion.get("index"), 9999)),
    )


def worst_owned_potion(potions: list[dict[str, Any]]) -> dict[str, Any] | None:
    owned = [potion for potion in potions if isinstance(potion, dict) and potion.get("index") is not None]
    if not owned:
        return None
    return min(owned, key=lambda potion: (estimate_potion_score(potion), _safe_int(potion.get("index"), 9999)))


def should_replace_potion(current_potions: list[dict[str, Any]], candidate_potion: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate_potion is None:
        return None
    worst = worst_owned_potion(current_potions)
    if worst is None:
        return None
    if estimate_potion_score(candidate_potion) > estimate_potion_score(worst) + 0.75:
        return worst
    return None


def choose_event_option(options: list[dict[str, Any]], state: GameStateView) -> dict[str, Any] | None:
    unlocked = [option for option in options if not option.get("is_locked")]
    if not unlocked:
        return None

    hp_ratio = state.hp / max(1, state.max_hp)

    def option_score(option: dict[str, Any]) -> tuple[float, int]:
        text = " ".join(
            _text(option.get(key))
            for key in ("option_id", "name", "label", "description")
        )
        score = 0.0
        score += sum(1.6 for token in _EVENT_POSITIVE_TOKENS if token in text)
        score -= sum(1.8 for token in _EVENT_NEGATIVE_TOKENS if token in text)
        if "skip" in text or "\u8df3\u8fc7" in text:
            score -= 0.25
        if hp_ratio < 0.45 and any(token in text for token in ("hp", "\u751f\u547d", "\u53d7\u4f24", "\u4f24\u5bb3")):
            score -= 2.0
        if any(token in text for token in ("remove", "purge", "\u5220\u724c", "\u79fb\u9664")):
            deck = list(state.player.get("deck") or [])
            if choose_purge_target(deck) is not None:
                score += 1.0
        return score, -_safe_int(option.get("index"), 0)

    return max(unlocked, key=option_score)


def shop_purge_cost(state: GameStateView) -> int:
    raw = state.raw or {}
    return _safe_int(raw.get("card_removal_cost", raw.get("purge_cost", 999999)), 999999)


def is_shop_context(state: GameStateView) -> bool:
    return _text((state.raw.get("context") or {}).get("room_type")) == "merchant"


def is_rest_context(state: GameStateView) -> bool:
    room_type = _text((state.raw.get("context") or {}).get("room_type"))
    return "rest" in room_type or room_type == "campfire"
