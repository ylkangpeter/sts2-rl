# -*- coding: utf-8 -*-
"""Shared heuristic helpers for flow policy and protocol sanitization."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from st2rl.gameplay.types import GameStateView
from st2rl.gameplay.knowledge_base import load_site_knowledge

_NEGATIVE_TYPES = {"curse", "status"}
_BASIC_STRIKE_IDS = {"CARD.STRIKE", "CARD.STRIKE_IRONCLAD"}
_BASIC_DEFEND_IDS = {"CARD.DEFEND", "CARD.DEFEND_IRONCLAD"}
_BASIC_STRIKE_NAMES = {"strike", "\u6253\u51fb"}
_BASIC_DEFEND_NAMES = {"defend", "\u9632\u5fa1"}
_BASIC_STRIKE_TOKENS = ("strike",)
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
    "CARD.FLAME_BARRIER": 3.1,
    "CARD.SHRUG_IT_OFF": 3.4,
    "CARD.TRUE_GRIT": -0.2,
    "CARD.ARMAMENTS": 2.2,
    "CARD.TAUNT": 2.4,
    "CARD.WHIRLWIND": 1.5,
    "CARD.UPPERCUT": 2.0,
    "CARD.INFLAME": 2.1,
    "CARD.BATTLE_TRANCE": 3.4,
    "CARD.BURNING_PACT": 3.7,
    "CARD.GRAPPLE": -0.4,
    "CARD.BLOODLETTING": 3.6,
    "CARD.RAMPAGE": -0.2,
    "CARD.PILLAGE": 1.0,
    "CARD.HEADBUTT": 3.2,
    "CARD.HEMOKINESIS": 0.7,
    "CARD.FIGHT_ME": 0.5,
    "CARD.DISMANTLE": 0.4,
    "CARD.UNRELENTING": 0.2,
    "CARD.BODY_SLAM": -1.8,
    "CARD.TWIN_STRIKE": -1.8,
    "CARD.POMMEL_STRIKE": 3.5,
    "CARD.SETUP_STRIKE": -1.7,
    "CARD.HAVOC": -1.6,
    "CARD.PERFECTED_STRIKE": -1.6,
    "CARD.ASHEN_STRIKE": -1.5,
    "CARD.SPITE": -1.4,
    "CARD.FEEL_NO_PAIN": -1.4,
    "CARD.INFERNAL_BLADE": -1.3,
    "CARD.JUGGLING": -1.3,
    "CARD.FORGOTTEN_RITUAL": -2.2,
    "CARD.STAMPEDE": -1.2,
    "CARD.RUPTURE": -1.2,
    "CARD.PRIMAL_FORCE": -1.1,
    "CARD.BREAKTHROUGH": -0.9,
    "CARD.SWORD_BOOMERANG": -0.8,
    "CARD.MOLTEN_FIST": -0.7,
    "CARD.TREMBLE": -0.5,
    "CARD.SPOILS_MAP": -1.6,
    "CARD.THUNDERCLAP": 2.2,
    "CARD.ANGER": 1.8,
    "CARD.BLUDGEON": 2.4,
    "CARD.HOWL_FROM_BEYOND": 1.6,
}

_UNSUPPORTED_HEADLESS_CARD_IDS = {
    "CARD.CASCADE",
    "CARD.HAVOC",
    "CARD.WHIRLWIND",
}

_UNSUPPORTED_HEADLESS_POTION_IDS = {
    "POTION.DISTILLED_CHAOS",
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
    "CARD.THUNDERCLAP",
    "CARD.ANGER",
    "CARD.BLUDGEON",
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


def _state_act_floor(state: GameStateView | None) -> tuple[int, int]:
    if state is None:
        return 1, 0
    context = state.raw.get("context") or {}
    act = _safe_int(context.get("act") or state.raw.get("act"), 1) or 1
    floor = _safe_int(context.get("floor") or state.raw.get("floor"), 0)
    return act, floor


def _boss_key_from_context(context: dict[str, Any]) -> str:
    boss_payload = context.get("boss")
    boss_text_parts: list[str] = []
    if isinstance(boss_payload, dict):
        boss_text_parts.extend(str(boss_payload.get(key) or "") for key in ("id", "name", "boss_id", "boss_name"))
    boss_text_parts.extend(str(context.get(key) or "") for key in ("boss_id", "boss_name"))
    return _text(" ".join(boss_text_parts))


def _state_boss_key(state: GameStateView | None) -> str:
    if state is None:
        return ""
    context = state.raw.get("context") or {}
    if not isinstance(context, dict):
        return ""
    return _boss_key_from_context(context)


def _act1_boss_profile_key(boss_key: str) -> str:
    key = _text(boss_key)
    if any(token in key for token in ("vantom", "墨影")):
        return "vantom"
    if any(token in key for token in ("kin", "同族")):
        return "kin"
    if any(token in key for token in ("ceremonial", "仪式兽")):
        return "ceremonial"
    return ""


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
    card_id = _card_id(card)
    if card_id == "CARD.POMMEL_STRIKE":
        return False
    name = _text(card.get("name"))
    text = f"{name} {_text(card.get('id'))}"
    return card_id in _BASIC_STRIKE_IDS or name in _BASIC_STRIKE_NAMES or any(token in text for token in _BASIC_STRIKE_TOKENS)


def is_basic_defend(card: dict[str, Any]) -> bool:
    card_id = _card_id(card)
    name = _text(card.get("name"))
    return card_id in _BASIC_DEFEND_IDS or name in _BASIC_DEFEND_NAMES


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


def estimate_card_reward_score(
    card: dict[str, Any],
    deck: list[dict[str, Any]] | None = None,
    *,
    act: int = 1,
    floor: int = 0,
    boss_key: str = "",
) -> float:
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
    cost = _safe_int(card.get("cost"), _safe_int((knowledge or {}).get("energy"), 0))
    draw_amount = _card_draw_amount(card)
    energy_amount = _card_energy_amount(card)
    if draw_amount > 0:
        score += min(draw_amount, 4) * 1.05
        if cost <= 1:
            score += 0.55
    if energy_amount > 0:
        score += min(energy_amount, 3) * 1.25
        if cost <= 1:
            score += 0.65
    if card.get("after_upgrade"):
        score += 0.35

    if is_basic_strike(card) or is_basic_defend(card):
        score -= 3.5

    score += _CARD_REWARD_PRIORS.get(_card_id(card), 0.0)

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
    act1_boss_profile = _act1_boss_profile_key(boss_key) if act == 1 else ""

    if card_id_upper == "CARD.BLOODLETTING":
        score += 1.0
        if counts["draw"] >= 2:
            score += 1.5
        elif counts["draw"] >= 1:
            score += 0.8
        if counts["zero_cost"] >= 3:
            score += 0.35
        if counts["burning_pact"] > 0:
            score += 1.0
        if act == 1 and floor <= 10 and counts["draw"] <= 1 and counts["zero_cost"] <= 1:
            score -= 0.8
    if card_id_upper == "CARD.BURNING_PACT":
        score += 0.9
        if basics["total"] >= 4:
            score += 0.7
        if counts["bloodletting"] > 0:
            score += 1.2
    if card_id_upper in {"CARD.SHRUG_IT_OFF", "CARD.FLAME_BARRIER", "CARD.TAUNT", "CARD.BLOOD_WALL"} and block_deficit:
        score += 0.8
    if card_id_upper == "CARD.HEADBUTT":
        score += 1.4
        if counts["draw"] >= 1:
            score += 0.8
        if basics["total"] >= 5:
            score += 0.4
    if card_id_upper in {"CARD.POMMEL_STRIKE", "CARD.BATTLE_TRANCE", "CARD.SHRUG_IT_OFF"}:
        score += 0.85
        if thick_deck:
            score += 0.45
        if deck_size <= 18:
            score += 0.35
    if card_id_upper in {"CARD.ARMAMENTS", "CARD.UPPERCUT", "CARD.INFLAME", "CARD.HEMOKINESIS"}:
        score += 0.35
    if card_id_upper in {"CARD.THUNDERCLAP", "CARD.ANGER", "CARD.BLUDGEON", "CARD.HOWL_FROM_BEYOND"}:
        score += 0.75
    if card_id_upper in {"CARD.GRAPPLE", "CARD.RAMPAGE", "CARD.FORGOTTEN_RITUAL", "CARD.SPOILS_MAP"}:
        score -= 1.0
    if card_id_upper == "CARD.TRUE_GRIT":
        if not bool(card.get("upgraded")):
            score -= 1.8
        if counts["attack"] <= 5:
            score -= 0.8
        if _deck_archetype_anchors(deck).get("exhaust", 0) <= 0:
            score -= 0.8
    if card_id_upper in {"CARD.POMMEL_STRIKE", "CARD.SHRUG_IT_OFF"} and counts["bloodletting"] > 0:
        score += 1.2
    if counts["bloodletting"] > 0 and any(token in _description(card) for token in ("draw", "\u62bd\u724c")):
        score += 0.5

    if act == 1 and floor <= 6:
        if card_id_upper == "CARD.UPPERCUT":
            score += 1.4
        elif card_id_upper == "CARD.ANGER":
            score += 3.0
        elif card_id_upper == "CARD.POMMEL_STRIKE" and counts["draw"] >= 1 and counts["energy"] <= 0:
            score -= 1.0
    if act == 1 and floor <= 14:
        if card_id_upper == "CARD.ANGER":
            score += 1.2
        elif card_id_upper == "CARD.SHRUG_IT_OFF":
            score += 0.7
        elif card_id_upper in {"CARD.IRON_WAVE", "CARD.UNRELENTING"}:
            score += 0.55
        elif card_id_upper == "CARD.ARMAMENTS" and counts["attack"] <= counts["skill"]:
            score -= 0.75
    if act == 1 and floor >= 9 and act1_boss_profile:
        if act1_boss_profile == "vantom":
            if card_id_upper in {"CARD.ANGER", "CARD.POMMEL_STRIKE", "CARD.THUNDERCLAP", "CARD.BATTLE_TRANCE"}:
                score += 1.35
            if draw_amount > 0 or energy_amount > 0:
                score += 0.55
            if cost >= 2 and card_id_upper in {"CARD.BLUDGEON", "CARD.PERFECTED_STRIKE", "CARD.HEAVY_BLADE"}:
                score -= 1.8
        elif act1_boss_profile == "kin":
            if card_id_upper in {"CARD.ANGER", "CARD.THUNDERCLAP", "CARD.UPPERCUT", "CARD.POMMEL_STRIKE", "CARD.BATTLE_TRANCE"}:
                score += 1.9
            if card_id_upper in {"CARD.SHRUG_IT_OFF", "CARD.FLAME_BARRIER", "CARD.TAUNT", "CARD.ARMAMENTS"}:
                score += 1.35
            if draw_amount > 0 or energy_amount > 0:
                score += 0.55
            if card_id_upper in {"CARD.BLOODLETTING", "CARD.OFFERING"} and floor <= 13:
                score -= 1.0
            if card_type == "attack" and cost >= 2 and draw_amount <= 0:
                score -= 1.35
            if card_id_upper in {"CARD.BLUDGEON", "CARD.PERFECTED_STRIKE", "CARD.HEAVY_BLADE", "CARD.DEMON_FORM"}:
                score -= 1.7
        elif act1_boss_profile == "ceremonial":
            if card_id_upper in {"CARD.SHRUG_IT_OFF", "CARD.FLAME_BARRIER", "CARD.TAUNT", "CARD.ARMAMENTS"}:
                score += 1.2
            if card_id_upper in {"CARD.UPPERCUT", "CARD.BATTLE_TRANCE", "CARD.POMMEL_STRIKE"}:
                score += 0.65
            if card_id_upper in {"CARD.BLOODLETTING", "CARD.OFFERING", "CARD.HEMOKINESIS"}:
                score -= 0.9

    if act >= 2:
        if card_id_upper in {"CARD.SHRUG_IT_OFF", "CARD.FLAME_BARRIER", "CARD.TAUNT", "CARD.BLOOD_WALL", "CARD.ARMAMENTS"}:
            score += 0.75
        if card_id_upper in {"CARD.POMMEL_STRIKE", "CARD.BATTLE_TRANCE", "CARD.BURNING_PACT", "CARD.BLOODLETTING", "CARD.HEADBUTT"}:
            score += 0.55
        if draw_amount > 0 or energy_amount > 0:
            score += 0.45
        if any(token in _description(card) for token in ("weak", "\u865a\u5f31", "vulnerable", "\u6613\u4f24")):
            score += 0.35
        if cost >= 2 and card_id_upper not in {"CARD.FLAME_BARRIER", "CARD.BLUDGEON"}:
            score -= 0.45
        if card_type == "attack" and draw_amount <= 0 and _card_id(card) not in _ELITE_FIGHTER_IDS:
            score -= 0.25
        if floor >= 8 and counts["block"] <= 4 and card_id_upper in _BLOCK_CARD_IDS:
            score += 0.45

    score += _archetype_reward_bonus(card, deck)
    if deck_size >= 22 and cost >= 2 and counts["high_cost"] >= 4:
        score -= 0.5
    if deck_size >= 20 and card_id_upper not in _DRAW_CARD_IDS and card_id_upper not in _ENERGY_CARD_IDS and score < 3.0:
        score -= 0.6
    if deck_size >= 25 and score < 3.6:
        score -= 1.4

    card_id = _text(card.get("id"))
    if card_id:
        duplicates = sum(1 for item in deck if _text(item.get("id")) == card_id)
        score -= min(duplicates, 3) * 0.35
        if act == 1 and floor <= 6 and card_id_upper == "CARD.POMMEL_STRIKE" and duplicates >= 1:
            score -= 2.0
        if card_id_upper == "CARD.BLOODLETTING" and duplicates >= 1 and counts["burning_pact"] <= 0:
            score -= 1.0
        if act == 1 and card_id_upper == "CARD.ARMAMENTS" and duplicates >= 1:
            score -= 1.2

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
    strong_cards = sum(1 for card in cards if _card_id(card) in _ELITE_FIGHTER_IDS or estimate_card_reward_score(card, cards, floor=floor) >= 2.4)
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
    act = _safe_int(context.get("act") or state.raw.get("act"), 1) or 1
    boss_key = _boss_key_from_context(context if isinstance(context, dict) else {})
    act1_boss_profile = _act1_boss_profile_key(boss_key) if act == 1 else ""
    hard_act1_boss = bool(act1_boss_profile)
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
            if act == 2:
                value = -6.2 + elite_readiness * 0.55
                if floor <= 10:
                    value -= 1.4
                if hp_ratio < 0.9:
                    value -= 2.2
                if hp_ratio < 0.72:
                    value -= 2.0
            elif act >= 3:
                value = -3.1 + elite_readiness * 0.85
                if hp_ratio < 0.62:
                    value -= 1.8
            elif floor <= 6:
                value = -5.2
            elif floor <= 8:
                value = -3.4 + elite_readiness
            elif floor <= 12:
                value = -3.2 + elite_readiness * 0.75
            else:
                value = -4.3 + elite_readiness * 0.65
            if hp_ratio < 0.65:
                value -= 2.0
            if floor >= 11 and hp_ratio < 0.8:
                value -= 2.6
            if floor >= 13 and hp_ratio < 0.9:
                value -= 3.0
            if hard_act1_boss and floor >= 9:
                value -= 2.2
            if hard_act1_boss and floor >= 12 and hp_ratio < 0.92:
                value -= 2.0
            if floor >= 10 and act1_boss_profile == "ceremonial":
                value -= 1.1
            if floor >= 10 and act1_boss_profile == "kin":
                value -= 1.5
            if floor >= 11 and act1_boss_profile == "vantom" and hp_ratio < 0.9:
                value -= 0.85
        elif "rest" in room or "camp" in room or room == "r":
            value = 4.4 if hp_ratio < 0.6 else (3.4 if hp_ratio < 0.8 else 1.15)
            if act == 2:
                value += 1.8 if hp_ratio < 0.85 else 0.45
            if floor >= 11:
                value += 2.35
            if floor >= 14:
                value += 1.35
            if hard_act1_boss and floor >= 10:
                value += 1.45 if hp_ratio < 0.92 else 0.7
            if act1_boss_profile == "ceremonial" and floor >= 10:
                value += 0.9
            if act1_boss_profile == "kin" and floor >= 11:
                value += 1.2
        elif "shop" in room or room in {"merchant", "$", "s"}:
            value = 2.65 if state.gold >= 150 else (0.35 if state.gold < 110 else 1.95)
            if purge_target is not None and state.gold >= 75:
                value += 0.95
            if floor >= 11 and state.gold >= 120:
                value += 1.25
            if floor >= 13 and hp_ratio < 0.85:
                value += 0.75
            if act == 2 and state.gold >= 100:
                value += 0.85
            if hard_act1_boss and floor >= 10 and hp_ratio < 0.9:
                value += 0.9
            if act1_boss_profile in {"ceremonial", "kin"} and floor >= 10 and state.gold >= 90:
                value += 0.6
            if act1_boss_profile == "kin" and floor >= 10:
                value += 0.45
            if act1_boss_profile == "vantom" and floor >= 9 and state.gold >= 80:
                value += 0.35
        elif "treasure" in room or "chest" in room or room == "t":
            value = 1.9
        elif "event" in room or room == "?" or "question" in room:
            if act == 2:
                value = 2.45 if hp_ratio >= 0.55 else 1.55
            elif floor <= 10 and hp_ratio >= 0.55:
                value = 1.3
            elif floor <= 10:
                value = 1.7
            else:
                value = 2.25 if hp_ratio < 0.8 else 1.65
        elif "monster" in room or "combat" in room or room == "m":
            if act == 2:
                value = 0.95 if hp_ratio >= 0.78 else -0.45
                if floor <= 5 and hp_ratio >= 0.85:
                    value += 0.45
            elif floor <= 10 and hp_ratio >= 0.55:
                value = 1.45
            elif floor <= 10:
                value = 0.95
            else:
                value = 0.55
            if floor >= 13 and hp_ratio < 0.7:
                value -= 1.9
            if floor >= 11 and hp_ratio < 0.55:
                value -= 1.35
            if floor >= 14 and hp_ratio < 0.85:
                value -= 1.1
            if hard_act1_boss and floor >= 11 and hp_ratio < 0.9:
                value -= 0.95
        elif "boss" in room or room == "b":
            value = 4.0
        else:
            value = 1.0
        has_elite_next = any("elite" in next_room for next_room in next_rooms)
        if act == 2 and has_elite_next:
            value -= 2.4
            if elite_readiness < 2.2:
                value -= 1.6
            if hp_ratio < 0.95:
                value -= 1.6
        if has_elite_next and hp_ratio < 0.78 and elite_readiness < 1.35:
            value -= 1.8
        if has_elite_next and floor >= 10 and hp_ratio < 0.9:
            value -= 3.8
        if has_elite_next and floor >= 13:
            value -= 2.5
            if hp_ratio < 0.97:
                value -= 3.5
        if hard_act1_boss and has_elite_next and floor >= 10:
            value -= 1.8
        if act1_boss_profile == "ceremonial" and has_elite_next and floor >= 10 and hp_ratio < 0.94:
            value -= 1.2
        if act1_boss_profile == "kin" and has_elite_next and floor >= 11:
            value -= 1.7
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


def choose_card_reward(
    cards: list[dict[str, Any]],
    deck: list[dict[str, Any]] | None,
    can_skip: bool,
    state: GameStateView | None = None,
) -> dict[str, Any] | None:
    indexed = [card for card in cards if isinstance(card, dict) and card.get("index") is not None]
    if not indexed:
        return None
    act, floor = _state_act_floor(state)
    boss_key = _state_boss_key(state)
    scored = sorted(
        ((estimate_card_reward_score(card, deck, act=act, floor=floor, boss_key=boss_key), card) for card in indexed),
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


def choose_purge_targets(deck: list[dict[str, Any]], count: int = 1) -> list[dict[str, Any]]:
    indexed = [card for card in deck if isinstance(card, dict) and card.get("index") is not None]
    if not indexed:
        return []

    def score(card: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
        return (
            0 if is_negative_card(card) else 1,
            0 if is_basic_strike(card) else 1,
            0 if is_basic_defend(card) else 1,
            0 if is_low_impact_card(card) else 1,
            _safe_int(card.get("cost"), 9),
            _safe_int(card.get("index"), 9999),
        )

    return sorted(indexed, key=score)[: max(0, count)]


def choose_purge_target(deck: list[dict[str, Any]]) -> dict[str, Any] | None:
    targets = choose_purge_targets(deck, 1)
    return targets[0] if targets else None


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


def choose_upgrade_targets(cards: list[dict[str, Any]], count: int = 1, state: GameStateView | None = None) -> list[dict[str, Any]]:
    indexed = [card for card in cards if isinstance(card, dict) and card.get("index") is not None]
    if not indexed:
        return []
    deck = [card for card in cards if isinstance(card, dict)]
    archetype_scores = _deck_archetype_scores(deck)
    archetype_anchors = _deck_archetype_anchors(deck)
    act, floor = _state_act_floor(state)

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

    def score(card: dict[str, Any]) -> tuple[int, int, int, float, float, int, int]:
        return (
            core_priority(card),
            1 if is_basic_strike(card) or is_basic_defend(card) else 0,
            1 if bool(card.get("upgraded")) else 0,
            0 if card.get("after_upgrade") else 1,
            -archetype_priority(card),
            -estimate_card_reward_score(card, cards, act=act, floor=floor),
            _safe_int(card.get("cost"), 99),
            _safe_int(card.get("index"), 9999),
        )

    return sorted(indexed, key=score)[: max(0, count)]


def choose_upgrade_target(cards: list[dict[str, Any]]) -> dict[str, Any] | None:
    targets = choose_upgrade_targets(cards, 1)
    return targets[0] if targets else None


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
    potion_id = str(potion.get("id") or "").strip().upper()
    if potion_id in _UNSUPPORTED_HEADLESS_POTION_IDS:
        return -100.0
    description = _text(potion.get("description"))
    target_type = _text(potion.get("target_type"))
    score = _rarity_value(potion.get("rarity")) * 0.6 + 1.0
    score += min(sum(1 for token in _UTILITY_TOKENS if token in description), 4) * 0.35
    if "enemy" in target_type:
        score += 0.5
    if any(token in description for token in ("heal", "block", "strength", "\u6062\u590d", "\u683c\u6321", "\u529b\u91cf")):
        score += 0.5
    if any(token in description for token in ("everyone", "all characters", "\u6240\u6709\u4eba")):
        score -= 4.0
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
        if any(token in text for token in ("amalgamator", "combine", "combine strikes", "junglemaze", "jungle maze", "safety in numbers")):
            score -= 8.0
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
