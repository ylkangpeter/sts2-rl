# -*- coding: utf-8 -*-
"""Shared heuristic helpers for flow policy and protocol sanitization."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from st2rl.gameplay.types import GameStateView
from st2rl.gameplay.knowledge_base import load_site_knowledge

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
    "\u9057\u7269",
    "\u836f\u6c34",
    "\u5220\u724c",
    "\u79fb\u9664",
    "\u5347\u7ea7",
    "\u953b\u9020",
    "\u53d8\u5316",
    "\u91d1\u5e01",
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
_CARD_REWARD_PRIORS = {
    "CARD.FLAME_BARRIER": 2.4,
    "CARD.SHRUG_IT_OFF": 2.7,
    "CARD.TRUE_GRIT": -0.2,
    "CARD.ARMAMENTS": 1.9,
    "CARD.TAUNT": 1.6,
    "CARD.WHIRLWIND": 1.5,
    "CARD.UPPERCUT": 1.5,
    "CARD.INFLAME": 1.4,
    "CARD.BATTLE_TRANCE": 2.5,
    "CARD.BURNING_PACT": 3.0,
    "CARD.GRAPPLE": 1.2,
    "CARD.BLOODLETTING": 3.0,
    "CARD.RAMPAGE": 1.0,
    "CARD.PILLAGE": 1.0,
    "CARD.HEADBUTT": 0.9,
    "CARD.HEMOKINESIS": 0.7,
    "CARD.FIGHT_ME": 0.5,
    "CARD.DISMANTLE": 0.4,
    "CARD.UNRELENTING": 0.2,
    "CARD.BODY_SLAM": -1.8,
    "CARD.TWIN_STRIKE": -1.8,
    "CARD.POMMEL_STRIKE": 2.4,
    "CARD.SETUP_STRIKE": -1.7,
    "CARD.HAVOC": -1.6,
    "CARD.PERFECTED_STRIKE": -1.6,
    "CARD.ASHEN_STRIKE": -1.5,
    "CARD.SPITE": -1.4,
    "CARD.FEEL_NO_PAIN": -1.4,
    "CARD.INFERNAL_BLADE": -1.3,
    "CARD.JUGGLING": -1.3,
    "CARD.FORGOTTEN_RITUAL": -1.3,
    "CARD.STAMPEDE": -1.2,
    "CARD.RUPTURE": -1.2,
    "CARD.PRIMAL_FORCE": -1.1,
    "CARD.BREAKTHROUGH": -0.9,
    "CARD.SWORD_BOOMERANG": -0.8,
    "CARD.MOLTEN_FIST": -0.7,
    "CARD.TREMBLE": -0.5,
}

_UNSUPPORTED_HEADLESS_CARD_IDS = {
    "CARD.WHIRLWIND",
}

_BLOCK_CARD_IDS = {
    "CARD.BLOOD_WALL",
    "CARD.FLAME_BARRIER",
    "CARD.SHRUG_IT_OFF",
    "CARD.TAUNT",
    "CARD.TRUE_GRIT",
    "CARD.IMPERVIOUS",
    "CARD.STONE_ARMOR",
    "CARD.CRIMSON_MANTLE",
    "CARD.ARMAMENTS",
}

_DRAW_CARD_IDS = {
    "CARD.BATTLE_TRANCE",
    "CARD.BURNING_PACT",
    "CARD.DARK_EMBRACE",
    "CARD.OFFERING",
    "CARD.POMMEL_STRIKE",
    "CARD.SHRUG_IT_OFF",
}

_ENERGY_CARD_IDS = {
    "CARD.BLOODLETTING",
    "CARD.EXPECT_A_FIGHT",
    "CARD.FORGOTTEN_RITUAL",
    "CARD.OFFERING",
}

_SELF_DAMAGE_CARD_IDS = {
    "CARD.BLOODLETTING",
    "CARD.BREAKTHROUGH",
    "CARD.CRIMSON_MANTLE",
    "CARD.HEMOKINESIS",
    "CARD.INFERNO",
    "CARD.OFFERING",
}

_SELF_DAMAGE_PAYOFF_IDS = {
    "CARD.RUPTURE",
    "CARD.INFERNO",
    "CARD.FEED",
    "CARD.REAPER",
    "CARD.TEAR_ASUNDER",
}

_EXHAUST_CONTROL_CARD_IDS = {
    "CARD.BRAND",
    "CARD.BURNING_PACT",
    "CARD.FIEND_FIRE",
    "CARD.SECOND_WIND",
    "CARD.TRUE_GRIT",
}

_ELITE_FIGHTER_IDS = {
    "CARD.FLAME_BARRIER",
    "CARD.SHRUG_IT_OFF",
    "CARD.ARMAMENTS",
    "CARD.TAUNT",
    "CARD.UPPERCUT",
    "CARD.INFLAME",
    "CARD.BATTLE_TRANCE",
    "CARD.BURNING_PACT",
    "CARD.BLOODLETTING",
    "CARD.HEADBUTT",
    "CARD.HEMOKINESIS",
    "CARD.PILLAGE",
}

_ARCHETYPE_CARD_WEIGHTS: dict[str, dict[str, float]] = {
    "strength": {
        "CARD.INFLAME": 3.0,
        "CARD.FIGHT_ME": 2.8,
        "CARD.RUPTURE": 2.6,
        "CARD.DEMON_FORM": 3.5,
        "CARD.TWIN_STRIKE": 1.4,
        "CARD.SWORD_BOOMERANG": 1.5,
        "CARD.THRASH": 2.0,
        "CARD.UPPERCUT": 1.2,
        "CARD.WHIRLWIND": 1.0,
    },
    "block": {
        "CARD.BODY_SLAM": 3.2,
        "CARD.SHRUG_IT_OFF": 2.2,
        "CARD.TRUE_GRIT": 0.4,
        "CARD.FLAME_BARRIER": 2.4,
        "CARD.TAUNT": 2.0,
        "CARD.STONE_ARMOR": 2.2,
        "CARD.JUGGERNAUT": 2.8,
        "CARD.BARRICADE": 3.2,
        "CARD.CRIMSON_MANTLE": 1.5,
        "CARD.IMPERVIOUS": 2.3,
    },
    "exhaust": {
        "CARD.CORRUPTION": 4.0,
        "CARD.DARK_EMBRACE": 3.8,
        "CARD.FEEL_NO_PAIN": 3.5,
        "CARD.TRUE_GRIT": 0.6,
        "CARD.BURNING_PACT": 2.5,
        "CARD.FORGOTTEN_RITUAL": 2.0,
        "CARD.BRAND": 2.2,
        "CARD.OFFERING": 2.5,
        "CARD.PACTS_END": 2.8,
        "CARD.ASHEN_STRIKE": 1.7,
        "CARD.THRASH": 1.6,
        "CARD.JUGGERNAUT": 1.7,
        "CARD.BODY_SLAM": 1.6,
        "CARD.EVIL_EYE": 1.6,
    },
    "bloodletting": {
        "CARD.BLOODLETTING": 3.4,
        "CARD.RUPTURE": 3.2,
        "CARD.INFERNO": 3.0,
        "CARD.BREAKTHROUGH": 2.3,
        "CARD.HEMOKINESIS": 2.2,
        "CARD.CRIMSON_MANTLE": 2.1,
        "CARD.OFFERING": 2.3,
        "CARD.BRAND": 1.8,
        "CARD.FEED": 1.8,
        "CARD.TEAR_ASUNDER": 2.2,
        "CARD.POMMEL_STRIKE": 1.7,
        "CARD.SHRUG_IT_OFF": 1.5,
        "CARD.BURNING_PACT": 1.6,
        "CARD.BATTLE_TRANCE": 1.7,
    },
    "strike": {
        "CARD.PERFECTED_STRIKE": 3.0,
        "CARD.POMMEL_STRIKE": 2.5,
        "CARD.TWIN_STRIKE": 2.0,
        "CARD.BREAKTHROUGH": 1.8,
        "CARD.TREMBLE": 1.2,
        "CARD.TAUNT": 1.8,
        "CARD.EXPECT_A_FIGHT": 2.0,
        "CARD.PYRE": 1.8,
        "CARD.HELLRAISER": 2.4,
        "CARD.COLOSSUS": 1.8,
        "CARD.CRUELTY": 1.6,
        "CARD.UPPERCUT": 1.4,
    },
}

_ARCHETYPE_ANCHORS: dict[str, set[str]] = {
    "strength": {"CARD.INFLAME", "CARD.FIGHT_ME", "CARD.RUPTURE", "CARD.DEMON_FORM"},
    "block": {"CARD.BODY_SLAM", "CARD.JUGGERNAUT", "CARD.BARRICADE", "CARD.STONE_ARMOR"},
    "exhaust": {"CARD.CORRUPTION", "CARD.DARK_EMBRACE", "CARD.FEEL_NO_PAIN"},
    "bloodletting": {"CARD.BLOODLETTING", "CARD.RUPTURE", "CARD.INFERNO", "CARD.CRIMSON_MANTLE"},
    "strike": {"CARD.PERFECTED_STRIKE", "CARD.HELLRAISER", "CARD.EXPECT_A_FIGHT"},
}

_ARCHETYPE_COMPONENT_HUNGRY = {"exhaust"}


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


def _stat_value(card: dict[str, Any], *names: str) -> int:
    stats = card.get("stats") or {}
    if not isinstance(stats, dict):
        return 0
    lowered = {str(key).lower(): value for key, value in stats.items()}
    return max((_safe_int(lowered.get(name.lower()), 0) for name in names), default=0)


def _card_draw_amount(card: dict[str, Any]) -> int:
    card_id = _card_id(card)
    if card_id in {"CARD.POMMEL_STRIKE", "CARD.SHRUG_IT_OFF", "CARD.BURNING_PACT", "CARD.BATTLE_TRANCE"}:
        return _stat_value(card, "cards", "draw")
    description = _description(card)
    if "draw" in description or "\u62bd" in description:
        return _stat_value(card, "cards", "draw")
    return _stat_value(card, "draw")


def _card_energy_amount(card: dict[str, Any]) -> int:
    return _stat_value(card, "energy")


def _description(card: dict[str, Any]) -> str:
    return _text(card.get("description"))


def _card_id(card: dict[str, Any]) -> str:
    return str(card.get("id") or "").strip().upper()


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
    card_id = _card_id(card)
    if card_id in {"CARD.BURN", "CARD.WOUND", "CARD.DAZED", "CARD.VOID", "CARD.REGRET", "CARD.SHAME"}:
        return True
    text = f"{_text(card.get('name'))} {_text(card.get('id'))}"
    return any(token in text for token in ("curse", "status", "wound", "dazed", "void", "regret", "shame"))


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
    return mapping.get(rarity, 0.0)


def _has_sparse_card_metadata(card: dict[str, Any]) -> bool:
    return not any(
        (
            card.get("rarity"),
            card.get("type"),
            card.get("description"),
            card.get("stats"),
            card.get("keywords"),
            card.get("after_upgrade"),
        )
    )


def _deck_counts(deck: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "attack": 0,
        "skill": 0,
        "power": 0,
        "block": 0,
        "high_cost": 0,
        "exhaust": 0,
        "exhaust_control": 0,
        "draw": 0,
        "energy": 0,
        "self_damage": 0,
        "self_damage_payoff": 0,
        "zero_cost": 0,
        "bloodletting": 0,
        "burning_pact": 0,
    }
    for card in deck:
        if not isinstance(card, dict):
            continue
        card_type = _text(card.get("type"))
        if card_type in counts:
            counts[card_type] += 1
        if _safe_int(card.get("cost"), 0) >= 2:
            counts["high_cost"] += 1
        if _safe_int(card.get("cost"), 1) <= 0:
            counts["zero_cost"] += 1
        description = _description(card)
        card_id = _card_id(card)
        if card_id in _BLOCK_CARD_IDS or (card_type == "skill" and any(token in description for token in ("block", "\u683c\u6321"))):
            counts["block"] += 1
        if is_exhaust_card(card):
            counts["exhaust"] += 1
        if card_id in _EXHAUST_CONTROL_CARD_IDS:
            counts["exhaust_control"] += 1
        if _card_draw_amount(card) > 0 or card_id in _DRAW_CARD_IDS or any(token in description for token in ("draw", "\u62bd\u724c")):
            counts["draw"] += 1
        if _card_energy_amount(card) > 0 or card_id in _ENERGY_CARD_IDS:
            counts["energy"] += 1
        if card_id in _SELF_DAMAGE_CARD_IDS:
            counts["self_damage"] += 1
        if card_id in _SELF_DAMAGE_PAYOFF_IDS:
            counts["self_damage_payoff"] += 1
        if card_id == "CARD.BLOODLETTING":
            counts["bloodletting"] += 1
        if card_id == "CARD.BURNING_PACT":
            counts["burning_pact"] += 1
    return counts


def _deck_basics(deck: list[dict[str, Any]]) -> dict[str, int]:
    strikes = 0
    defends = 0
    for card in deck:
        if not isinstance(card, dict):
            continue
        if is_basic_strike(card):
            strikes += 1
        elif is_basic_defend(card):
            defends += 1
    return {"strike": strikes, "defend": defends, "total": strikes + defends}


def _deck_archetype_scores(deck: list[dict[str, Any]]) -> dict[str, float]:
    scores = {name: 0.0 for name in _ARCHETYPE_CARD_WEIGHTS}
    for card in deck:
        if not isinstance(card, dict):
            continue
        card_id = _card_id(card)
        for name, weights in _ARCHETYPE_CARD_WEIGHTS.items():
            scores[name] += weights.get(card_id, 0.0)
        if _card_draw_amount(card) > 0:
            scores["bloodletting"] += 0.35
            scores["exhaust"] += 0.25
            scores["strike"] += 0.2
        if _card_energy_amount(card) > 0:
            scores["bloodletting"] += 0.35
            scores["strike"] += 0.25
            scores["strength"] += 0.2
        if is_exhaust_card(card):
            scores["exhaust"] += 0.35
        if _text(card.get("type")) == "skill" and any(token in _description(card) for token in ("block", "\u683c\u6321")):
            scores["block"] += 0.25
    return scores


def _deck_archetype_anchors(deck: list[dict[str, Any]]) -> dict[str, int]:
    anchors = {name: 0 for name in _ARCHETYPE_CARD_WEIGHTS}
    for card in deck:
        if not isinstance(card, dict):
            continue
        card_id = _card_id(card)
        for name, anchor_ids in _ARCHETYPE_ANCHORS.items():
            if card_id in anchor_ids:
                anchors[name] += 1
    return anchors


def _archetype_reward_bonus(card: dict[str, Any], deck: list[dict[str, Any]]) -> float:
    if not deck:
        return 0.0
    card_id = _card_id(card)
    deck_size = len(deck)
    scores = _deck_archetype_scores(deck)
    anchors = _deck_archetype_anchors(deck)
    bonus = 0.0
    for name, weights in _ARCHETYPE_CARD_WEIGHTS.items():
        weight = weights.get(card_id, 0.0)
        if weight <= 0:
            continue
        anchor_count = anchors.get(name, 0)
        affinity = scores.get(name, 0.0)
        if anchor_count <= 0:
            continue
        if deck_size < 15 and anchor_count < 2:
            continue
        bonus += min(weight * (0.07 + min(affinity, 12.0) * 0.012), 0.8)

    return bonus


@lru_cache(maxsize=1)
def _knowledge_card_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for card in load_site_knowledge().get("cards", []):
        if not isinstance(card, dict):
            continue
        keys = {
            str(card.get("localization_key") or "").strip().upper(),
            str(card.get("id") or "").strip().upper(),
            str(card.get("name_en") or "").strip().upper().replace(" ", "_"),
            str(card.get("name") or "").strip().upper().replace(" ", "_"),
        }
        for key in keys:
            if key:
                index[key] = card
    return index


def _lookup_card_knowledge(card: dict[str, Any]) -> dict[str, Any] | None:
    card_id = _card_id(card)
    name_key = _text(card.get("name")).upper().replace(" ", "_")
    index = _knowledge_card_index()
    for key in (card_id.replace("CARD.", ""), card_id, name_key):
        if key in index:
            return index[key]
    return None


def estimate_card_reward_score(card: dict[str, Any], deck: list[dict[str, Any]] | None = None) -> float:
    deck = [item for item in (deck or []) if isinstance(item, dict)]
    if _card_id(card) in _UNSUPPORTED_HEADLESS_CARD_IDS:
        return -100.0
    if is_negative_card(card):
        return -100.0

    knowledge = _lookup_card_knowledge(card)
    rarity = card.get("rarity")
    if not rarity and isinstance(knowledge, dict):
        rarity = knowledge.get("rarity")
    score = _rarity_value(rarity)
    non_engine_stats = max(0, _stats_total(card) - _card_draw_amount(card) - _card_energy_amount(card))
    score += min(non_engine_stats, 24) * 0.12
    score += min(sum(1 for token in _UTILITY_TOKENS if token in _description(card)), 4) * 0.35
    draw_amount = _card_draw_amount(card)
    energy_amount = _card_energy_amount(card)
    if draw_amount > 0:
        score += min(draw_amount, 4) * 0.75
    if energy_amount > 0:
        score += min(energy_amount, 3) * 0.95
    if card.get("after_upgrade"):
        score += 0.35

    if is_basic_strike(card) or is_basic_defend(card):
        score -= 3.5

    score += _CARD_REWARD_PRIORS.get(_card_id(card), 0.0)

    cost = _safe_int(card.get("cost"), _safe_int((knowledge or {}).get("energy"), 0))
    counts = _deck_counts(deck)
    basics = _deck_basics(deck)
    deck_size = len(deck)
    if cost >= 2 and counts["high_cost"] >= 3:
        score -= 0.8
    if is_exhaust_card(card) and _card_id(card) not in {"CARD.BURNING_PACT", "CARD.TRUE_GRIT"}:
        score -= 1.2
    if is_exhaust_card(card) and counts["exhaust"] >= 3 and _card_id(card) != "CARD.BURNING_PACT":
        score -= 0.6

    card_type = _text(card.get("type"))
    if card_type == "skill" and counts["attack"] > counts["skill"] + 1:
        score += 0.45
    if card_type == "attack" and counts["skill"] > counts["attack"] + 2:
        score += 0.25
    if card_type == "power":
        score += 0.4 if counts["power"] < 2 else -0.2

    card_id_upper = _card_id(card)
    block_deficit = deck_size >= 12 and counts["block"] <= max(2, counts["attack"] // 3)
    thick_deck = deck_size >= 22

    if card_id_upper == "CARD.BLOODLETTING":
        score += 0.6
        if counts["draw"] >= 2:
            score += 1.2
        if counts["zero_cost"] >= 3:
            score += 0.35
        if counts["burning_pact"] > 0:
            score += 1.0
    if card_id_upper == "CARD.BURNING_PACT":
        score += 0.5
        if basics["total"] >= 4:
            score += 0.7
        if counts["bloodletting"] > 0:
            score += 1.2
    if card_id_upper in {"CARD.SHRUG_IT_OFF", "CARD.FLAME_BARRIER", "CARD.TAUNT", "CARD.BLOOD_WALL"} and block_deficit:
        score += 0.5
    if card_id_upper in {"CARD.POMMEL_STRIKE", "CARD.BATTLE_TRANCE", "CARD.SHRUG_IT_OFF"} and thick_deck:
        score += 0.25
    if card_id_upper == "CARD.TRUE_GRIT":
        if not bool(card.get("upgraded")):
            score -= 1.8
        if counts["attack"] <= 5:
            score -= 0.8
        if _deck_archetype_anchors(deck).get("exhaust", 0) <= 0:
            score -= 0.8
    if card_id_upper in {"CARD.POMMEL_STRIKE", "CARD.SHRUG_IT_OFF"} and counts["bloodletting"] > 0:
        score += 0.8
    if counts["bloodletting"] > 0 and any(token in _description(card) for token in ("draw", "\u62bd\u724c")):
        score += 0.5
    score += _archetype_reward_bonus(card, deck)
    if deck_size >= 22 and cost >= 2 and counts["high_cost"] >= 4:
        score -= 0.5
    if deck_size >= 25 and score < 3.6:
        score -= 1.4

    card_id = _text(card.get("id"))
    if card_id:
        duplicates = sum(1 for item in deck if _text(item.get("id")) == card_id)
        score -= min(duplicates, 3) * 0.35

    if _has_sparse_card_metadata(card):
        if cost <= 0:
            score += 0.2
        elif cost >= 2:
            score -= 0.15 * min(cost - 1, 2)

    return score


def estimate_elite_readiness(deck: list[dict[str, Any]] | None, hp_ratio: float, floor: int) -> float:
    cards = [item for item in (deck or []) if isinstance(item, dict)]
    counts = _deck_counts(cards)
    basics = _deck_basics(cards)
    strong_cards = sum(1 for card in cards if _card_id(card) in _ELITE_FIGHTER_IDS or estimate_card_reward_score(card, cards) >= 2.4)
    defense_cards = sum(
        1
        for card in cards
        if _card_id(card) in {"CARD.FLAME_BARRIER", "CARD.SHRUG_IT_OFF", "CARD.TRUE_GRIT", "CARD.TAUNT"}
        or (_text(card.get("type")) == "skill" and any(token in _description(card) for token in ("block", "\u683c\u6321")))
    )
    readiness = hp_ratio * 2.2
    readiness += min(strong_cards, 5) * 0.35
    readiness += min(defense_cards, 4) * 0.25
    readiness += min(counts["power"], 2) * 0.15
    readiness += min(counts["exhaust"], 3) * 0.08
    readiness -= min(basics["total"], 9) * 0.12
    readiness -= max(0, counts["high_cost"] - 3) * 0.18
    if floor <= 6:
        readiness -= 1.2
    elif floor <= 8:
        readiness -= 0.7
    return readiness


def choose_map_node_choice(choices: list[dict[str, Any]], state: GameStateView) -> dict[str, Any] | None:
    indexed = [choice for choice in choices if isinstance(choice, dict)]
    if not indexed:
        return None

    context = state.raw.get("context") or {}
    floor = _safe_int(context.get("floor") or state.raw.get("floor"), 0)
    hp_ratio = state.hp / max(1, state.max_hp)
    deck = [card for card in (state.player.get("deck") or []) if isinstance(card, dict)]
    elite_readiness = estimate_elite_readiness(deck, hp_ratio, floor)
    purge_target = choose_purge_target(deck)
    node_by_coord: dict[tuple[int, int], dict[str, Any]] = {}
    for node in state.map_nodes:
        if isinstance(node, dict):
            node_by_coord[(_safe_int(node.get("col"), -1), _safe_int(node.get("row"), -1))] = node
    for row in (state.full_map.get("rows") or []):
        if not isinstance(row, list):
            continue
        for node in row:
            if isinstance(node, dict):
                node_by_coord[(_safe_int(node.get("col"), -1), _safe_int(node.get("row"), -1))] = node

    def room_type(choice: dict[str, Any]) -> str:
        node = node_by_coord.get((_safe_int(choice.get("col"), -1), _safe_int(choice.get("row"), -1)), {})
        return _text(
            choice.get("room_type")
            or choice.get("type")
            or choice.get("symbol")
            or choice.get("icon")
            or node.get("room_type")
            or node.get("type")
            or node.get("symbol")
        )

    def child_room_types(choice: dict[str, Any]) -> list[str]:
        node = node_by_coord.get((_safe_int(choice.get("col"), -1), _safe_int(choice.get("row"), -1)), {})
        children = node.get("children") or choice.get("children") or []
        rooms: list[str] = []
        for child in children:
            if not isinstance(child, dict):
                continue
            child_node = node_by_coord.get((_safe_int(child.get("col"), -1), _safe_int(child.get("row"), -1)), child)
            rooms.append(
                _text(
                    child_node.get("room_type")
                    or child_node.get("type")
                    or child_node.get("symbol")
                    or child.get("room_type")
                    or child.get("type")
                    or child.get("symbol")
                )
            )
        return rooms

    def score(choice: dict[str, Any]) -> tuple[float, int, int]:
        room = room_type(choice)
        next_rooms = child_room_types(choice)
        value = 0.0
        if "elite" in room:
            if floor <= 6:
                value = -4.5
            elif floor <= 8:
                value = -2.8 + elite_readiness
            elif floor <= 12:
                value = -1.0 + elite_readiness * 1.1
            else:
                value = 0.3 + elite_readiness * 1.15
            if hp_ratio < 0.55:
                value -= 1.5
        elif "rest" in room or "camp" in room or room == "r":
            value = 3.2 if hp_ratio < 0.5 else (2.1 if hp_ratio < 0.7 else 0.65)
        elif "shop" in room or room in {"merchant", "$", "s"}:
            value = 2.0 if state.gold >= 150 else (-0.25 if state.gold < 110 else 1.35)
            if purge_target is not None and state.gold >= 75:
                value += 0.7
        elif "treasure" in room or "chest" in room or room == "t":
            value = 1.9
        elif "event" in room or room == "?" or "question" in room:
            if floor <= 10 and hp_ratio >= 0.55:
                value = 1.3
            elif floor <= 10:
                value = 1.45
            else:
                value = 1.45
        elif "monster" in room or "combat" in room or room == "m":
            if floor <= 10 and hp_ratio >= 0.55:
                value = 1.45
            elif floor <= 10:
                value = 0.95
            else:
                value = 1.25
        elif "boss" in room or room == "b":
            value = 4.0
        else:
            value = 1.0
        if any("elite" in next_room for next_room in next_rooms) and hp_ratio < 0.72 and elite_readiness < 1.15:
            value -= 1.4
        if any("elite" in next_room for next_room in next_rooms) and ("rest" in room or "shop" in room):
            value += 0.35
        if floor <= 8 and "unknown" in room and any("elite" in next_room for next_room in next_rooms):
            value -= 0.65
        return (
            value,
            -_safe_int(choice.get("row"), 0),
            -_safe_int(choice.get("col"), 0),
        )

    return max(indexed, key=score)


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
    counts = _deck_counts(deck or [])
    sparse_offer = all(_has_sparse_card_metadata(card) for card in indexed)
    if can_skip:
        threshold = 0.0 if sparse_offer else 1.25
        if deck_size <= 14:
            threshold = min(threshold, -0.25)

        replacement: tuple[float, dict[str, Any]] | None = None
        for candidate_score, candidate_card in scored:
            if candidate_score < threshold:
                continue
            if deck_size >= 12 and (is_basic_strike(candidate_card) or is_basic_defend(candidate_card)):
                continue
            if is_low_impact_card(candidate_card) and candidate_score < 2.0:
                continue
            replacement = (candidate_score, candidate_card)
            break
        if replacement is None:
            return None
        best_score, best_card = replacement

        if best_score < threshold:
            return None
        if deck_size >= 25 and best_score < 3.4:
            return None
        if deck_size >= 22 and best_score < 2.5:
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

    def score(card: dict[str, Any]) -> tuple[int, int, int, int, int]:
        return (
            0 if is_negative_card(card) else 1,
            0 if is_basic_strike(card) else 1,
            0 if is_basic_defend(card) else 1,
            0 if is_low_impact_card(card) else 1,
            _safe_int(card.get("cost"), 9),
        )

    return min(indexed, key=score)


def should_prioritize_shop_purge(
    deck: list[dict[str, Any]] | None,
    gold: int,
    purge_cost: int,
    *,
    best_card: dict[str, Any] | None = None,
    best_relic: dict[str, Any] | None = None,
) -> bool:
    cards = [item for item in (deck or []) if isinstance(item, dict)]
    purge_target = choose_purge_target(cards)
    if purge_target is None or gold < purge_cost:
        return False

    basics = _deck_basics(cards)
    deck_size = len(cards)
    best_card_score = estimate_card_reward_score(best_card, cards) if isinstance(best_card, dict) else -999.0
    best_relic_score = estimate_relic_score(best_relic) if isinstance(best_relic, dict) else -999.0

    if deck_size >= 25:
        return True
    if is_negative_card(purge_target):
        return True
    if basics["total"] >= 6:
        return True
    if deck_size >= 20 and basics["total"] >= 3 and best_card_score < 3.2 and best_relic_score < 3.8:
        return True
    if best_card_score < 2.8 and best_relic_score < 3.4 and basics["total"] >= 3:
        return True
    if deck_size >= 22 and best_card_score < 3.2:
        return True
    return False


def choose_upgrade_target(cards: list[dict[str, Any]]) -> dict[str, Any] | None:
    indexed = [card for card in cards if isinstance(card, dict) and card.get("index") is not None]
    if not indexed:
        return None
    deck = [card for card in cards if isinstance(card, dict)]
    archetype_scores = _deck_archetype_scores(deck)
    archetype_anchors = _deck_archetype_anchors(deck)

    def core_priority(card: dict[str, Any]) -> int:
        card_id = _card_id(card)
        if card_id in {
            "CARD.BURNING_PACT",
            "CARD.BLOODLETTING",
            "CARD.SHRUG_IT_OFF",
            "CARD.POMMEL_STRIKE",
            "CARD.FLAME_BARRIER",
            "CARD.TRUE_GRIT",
            "CARD.BATTLE_TRANCE",
            "CARD.ARMAMENTS",
            "CARD.UPPERCUT",
            "CARD.HEMOKINESIS",
        }:
            return 0
        if any(token in _description(card) for token in ("draw", "\u62bd\u724c")):
            return 1
        return 2

    def archetype_priority(card: dict[str, Any]) -> float:
        card_id = _card_id(card)
        priority = 0.0
        for name, weights in _ARCHETYPE_CARD_WEIGHTS.items():
            weight = weights.get(card_id, 0.0)
            if weight <= 0:
                continue
            anchor_count = archetype_anchors.get(name, 0)
            if anchor_count <= 0:
                continue
            if len(deck) < 15 and anchor_count < 2:
                continue
            priority += weight * (1.0 + min(archetype_scores.get(name, 0.0), 12.0) / 12.0)
        return priority

    def score(card: dict[str, Any]) -> tuple[int, int, int, float, float, int]:
        return (
            core_priority(card),
            1 if is_basic_strike(card) or is_basic_defend(card) else 0,
            1 if bool(card.get("upgraded")) else 0,
            0 if card.get("after_upgrade") else 1,
            -archetype_priority(card),
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
    deck_size = len(deck or [])
    threshold = 1.5
    if deck_size >= 20:
        threshold = 2.1
    if deck_size >= 25:
        threshold = 3.4
    if estimate_card_reward_score(best, deck) < threshold:
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
            for key in ("option_id", "name", "label", "title", "description", "text_key")
        )
        vars_payload = option.get("vars") or {}
        score = 0.0
        score += sum(1.6 for token in _EVENT_POSITIVE_TOKENS if token in text)
        score -= sum(1.8 for token in _EVENT_NEGATIVE_TOKENS if token in text)
        if "relic" in text or "遗物" in text or "{relic}" in text:
            score += 2.5
        if "gold" in text or "金币" in text or "{gold}" in text:
            score -= 0.8
        if isinstance(vars_payload, dict):
            if any(str(key).lower() == "relic" for key in vars_payload):
                score += 4.0
            if any(str(key).lower() == "potion" for key in vars_payload):
                score += 2.2
            gold_amount = max(_safe_int(vars_payload.get("Gold"), 0), _safe_int(vars_payload.get("gold"), 0))
            if gold_amount > 0:
                score += min(gold_amount / 80.0, 2.5)
            hp_loss = max(_safe_int(vars_payload.get("HpLoss"), 0), _safe_int(vars_payload.get("hp_loss"), 0))
            if hp_loss > 0:
                score -= min(hp_loss / 3.0, 4.0)
                if hp_ratio < 0.6:
                    score -= 1.5
                if hp_ratio < 0.45:
                    score -= 1.5
            max_hp_delta = max(_safe_int(vars_payload.get("MaxHp"), 0), _safe_int(vars_payload.get("max_hp"), 0))
            if max_hp_delta > 0:
                loses_max_hp = (
                    "lose max hp" in text
                    or ("失去" in text and "最大生命" in text)
                    or ("lose hp" in text and "max hp" in text)
                    or ("扣除" in text and "最大生命" in text)
                    or ("减少" in text and "最大生命" in text)
                )
                gains_max_hp = (
                    "gain max hp" in text
                    or ("获得" in text and "最大生命" in text)
                    or ("提升" in text and "最大生命" in text)
                    or ("增加" in text and "最大生命" in text)
                    or "increase max hp" in text
                )
                if loses_max_hp:
                    score -= min(max_hp_delta / 2.5, 5.0)
                elif gains_max_hp:
                    score += min(max_hp_delta / 4.0, 3.0)
            if any(token in text for token in ("transform", "变化", "变形")):
                score += 1.8
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
