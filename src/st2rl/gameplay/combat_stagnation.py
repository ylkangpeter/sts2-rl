# -*- coding: utf-8 -*-
"""Combat stagnation profiles tuned by encounter identity."""

from dataclasses import dataclass
from typing import Any

from st2rl.gameplay.types import GameStateView


@dataclass(frozen=True, slots=True)
class CombatStagnationProfile:
    """Threshold profile for combat deadlock handling."""

    profile_name: str
    room_type: str
    enemy_key: str
    no_action_proceed_threshold: int
    no_action_abort_threshold: int
    stuck_warn_threshold: int
    stuck_abort_threshold: int


@dataclass(frozen=True, slots=True)
class _ProfileOverride:
    profile_name: str
    room_tokens: tuple[str, ...]
    any_tokens: tuple[str, ...]
    all_tokens: tuple[str, ...] = ()
    act_min: int = 0
    act_max: int = 0
    no_action_proceed_threshold: int = 0
    no_action_abort_threshold: int = 0
    stuck_warn_threshold: int = 0
    stuck_abort_threshold: int = 0


_SPECIAL_OVERRIDES: tuple[_ProfileOverride, ...] = (
    _ProfileOverride(
        profile_name="boss_kin",
        room_tokens=("boss",),
        any_tokens=("the_kin_boss", "同族神官", "同族信徒"),
        act_min=1,
        act_max=1,
        no_action_proceed_threshold=8,
        no_action_abort_threshold=34,
        stuck_warn_threshold=125,
        stuck_abort_threshold=330,
    ),
    _ProfileOverride(
        profile_name="boss_vantom",
        room_tokens=("boss",),
        any_tokens=("vantom", "vantom_boss", "墨影幻灵"),
        act_min=1,
        act_max=1,
        no_action_proceed_threshold=7,
        no_action_abort_threshold=30,
        stuck_warn_threshold=120,
        stuck_abort_threshold=300,
    ),
    _ProfileOverride(
        profile_name="boss_ceremonial_beast",
        room_tokens=("boss",),
        any_tokens=("ceremonial_beast_boss", "ceremonial beast", "仪式兽"),
        act_min=1,
        act_max=1,
        no_action_proceed_threshold=7,
        no_action_abort_threshold=30,
        stuck_warn_threshold=118,
        stuck_abort_threshold=300,
    ),
    _ProfileOverride(
        profile_name="boss_insatiable",
        room_tokens=("boss",),
        any_tokens=("insatiable", "sand", "sandpit", "沙虫", "沙坑"),
        act_min=1,
        no_action_proceed_threshold=8,
        no_action_abort_threshold=36,
        stuck_warn_threshold=130,
        stuck_abort_threshold=340,
    ),
    _ProfileOverride(
        profile_name="elite_old_statue",
        room_tokens=("elite",),
        any_tokens=("旧日雕像", "old statue"),
        act_min=1,
        act_max=1,
        no_action_proceed_threshold=6,
        no_action_abort_threshold=24,
        stuck_warn_threshold=102,
        stuck_abort_threshold=250,
    ),
    _ProfileOverride(
        profile_name="elite_donis_bird",
        room_tokens=("elite",),
        any_tokens=("多尼斯异鸟", "donis"),
        act_min=1,
        act_max=1,
        no_action_proceed_threshold=6,
        no_action_abort_threshold=23,
        stuck_warn_threshold=100,
        stuck_abort_threshold=245,
    ),
    _ProfileOverride(
        profile_name="elite_common_pack",
        room_tokens=("elite",),
        any_tokens=("leader", "book of stabbing", "sentries", "lagavulin", "nob", "头目", "哨兵", "拉格", "贵族"),
        act_min=1,
        no_action_proceed_threshold=5,
        no_action_abort_threshold=20,
        stuck_warn_threshold=90,
        stuck_abort_threshold=220,
    ),
    _ProfileOverride(
        profile_name="monster_raider_pack",
        room_tokens=("monster", "combat"),
        any_tokens=("劫掠者", "raider", "刺客", "追踪手", "斧手", "弩手", "暴徒"),
        act_min=1,
        no_action_proceed_threshold=4,
        no_action_abort_threshold=14,
        stuck_warn_threshold=72,
        stuck_abort_threshold=175,
    ),
    _ProfileOverride(
        profile_name="monster_construct_eye_pack",
        room_tokens=("monster", "combat"),
        any_tokens=("立柱构造体", "利齿之眼", "墨宝", "雾菇", "construct", "eye"),
        act_min=1,
        no_action_proceed_threshold=4,
        no_action_abort_threshold=15,
        stuck_warn_threshold=76,
        stuck_abort_threshold=182,
    ),
    _ProfileOverride(
        profile_name="monster_snare_pack",
        room_tokens=("monster", "combat"),
        any_tokens=("蛇行扼杀者", "藤蔓蹒跚者", "strangler", "vine"),
        act_min=1,
        no_action_proceed_threshold=4,
        no_action_abort_threshold=14,
        stuck_warn_threshold=70,
        stuck_abort_threshold=170,
    ),
    _ProfileOverride(
        profile_name="monster_easy_single",
        room_tokens=("monster", "combat"),
        any_tokens=(
            "jaw worm",
            "louse",
            "slime",
            "cultist",
            "树枝史莱姆",
            "树叶史莱姆",
            "毛绒伏地虫",
            "缩小甲虫",
            "小啃兽",
            "飞蝇菌子",
            "颚虫",
            "虱子",
            "史莱姆",
            "邪教徒",
        ),
        act_min=1,
        no_action_proceed_threshold=2,
        no_action_abort_threshold=8,
        stuck_warn_threshold=45,
        stuck_abort_threshold=100,
    ),
    _ProfileOverride(
        profile_name="boss_late_act_default",
        room_tokens=("boss",),
        any_tokens=(),
        act_min=2,
        no_action_proceed_threshold=9,
        no_action_abort_threshold=38,
        stuck_warn_threshold=145,
        stuck_abort_threshold=380,
    ),
    _ProfileOverride(
        profile_name="elite_late_act_default",
        room_tokens=("elite",),
        any_tokens=(),
        act_min=2,
        no_action_proceed_threshold=6,
        no_action_abort_threshold=24,
        stuck_warn_threshold=105,
        stuck_abort_threshold=260,
    ),
    _ProfileOverride(
        profile_name="monster_late_act_default",
        room_tokens=("monster", "combat"),
        any_tokens=(),
        act_min=2,
        no_action_proceed_threshold=4,
        no_action_abort_threshold=16,
        stuck_warn_threshold=85,
        stuck_abort_threshold=210,
    ),
)

_BOSS_HINT_TOKENS: tuple[str, ...] = (
    "boss",
    "vantom",
    "the_kin_boss",
    "ceremonial_beast_boss",
    "insatiable",
    "神官",
    "幻灵",
    "仪式兽",
    "沙虫",
)

_ELITE_HINT_TOKENS: tuple[str, ...] = (
    "elite",
    "旧日雕像",
    "多尼斯异鸟",
    "leader",
    "book of stabbing",
    "sentries",
    "lagavulin",
    "nob",
    "贵族",
    "哨兵",
)


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _state_room_type(state: GameStateView) -> str:
    context = state.raw.get("context") or {}
    return _text(context.get("room_type") or state.raw.get("room_type") or "")


def _state_act(state: GameStateView) -> int:
    context = state.raw.get("context") or {}
    try:
        value = context.get("act", state.raw.get("act", 0))
        if value is None or value == "":
            return 0
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _state_enemy_key(state: GameStateView) -> str:
    parts: list[str] = []
    for enemy in state.living_enemies():
        if not isinstance(enemy, dict):
            continue
        enemy_id = _text(enemy.get("id"))
        enemy_name = _text(enemy.get("name"))
        if enemy_id:
            parts.append(enemy_id)
        if enemy_name:
            parts.append(enemy_name)
    parts = sorted(set(parts))
    return " ".join(parts)


def _infer_room_type(room_type: str, enemy_key: str) -> str:
    if any(token in room_type for token in ("boss", "elite", "monster", "combat")):
        return room_type
    if any(token in enemy_key for token in _BOSS_HINT_TOKENS):
        return "boss"
    if any(token in enemy_key for token in _ELITE_HINT_TOKENS):
        return "elite"
    if enemy_key:
        return "monster"
    return room_type


def _base_profile(
    *,
    room_type: str,
    enemy_key: str,
    no_action_proceed_threshold: int,
    no_action_abort_threshold: int,
    stuck_warn_threshold: int,
    stuck_abort_threshold: int,
) -> CombatStagnationProfile:
    proceed = max(1, no_action_proceed_threshold)
    no_action_abort = max(proceed + 1, no_action_abort_threshold)
    warn = max(10, stuck_warn_threshold)
    stuck_abort = max(warn + 1, stuck_abort_threshold)
    profile_name = "default"

    if "boss" in room_type:
        profile_name = "boss_default"
        proceed = max(proceed, 5)
        no_action_abort = max(no_action_abort, 22)
        warn = max(warn, 90)
        stuck_abort = max(stuck_abort, 230)
    elif "elite" in room_type:
        profile_name = "elite_default"
        proceed = max(proceed, 4)
        no_action_abort = max(no_action_abort, 16)
        warn = max(warn, 72)
        stuck_abort = max(stuck_abort, 170)

    return CombatStagnationProfile(
        profile_name=profile_name,
        room_type=room_type,
        enemy_key=enemy_key,
        no_action_proceed_threshold=proceed,
        no_action_abort_threshold=no_action_abort,
        stuck_warn_threshold=warn,
        stuck_abort_threshold=stuck_abort,
    )


def _matches_override(room_type: str, enemy_key: str, act: int, override: _ProfileOverride) -> bool:
    if override.act_min > 0 and act < override.act_min:
        return False
    if override.act_max > 0 and act > override.act_max:
        return False
    if override.room_tokens and not any(token in room_type for token in override.room_tokens):
        return False
    if override.any_tokens and not any(token in enemy_key for token in override.any_tokens):
        return False
    if override.all_tokens and not all(token in enemy_key for token in override.all_tokens):
        return False
    return True


def build_combat_stagnation_profile(
    state: GameStateView | None,
    *,
    no_action_proceed_threshold: int,
    no_action_abort_threshold: int,
    stuck_warn_threshold: int,
    stuck_abort_threshold: int,
) -> CombatStagnationProfile:
    """Return a stagnation profile based on current combat encounter."""

    if state is None:
        return CombatStagnationProfile(
            profile_name="default",
            room_type="",
            enemy_key="",
            no_action_proceed_threshold=max(1, no_action_proceed_threshold),
            no_action_abort_threshold=max(2, no_action_abort_threshold),
            stuck_warn_threshold=max(10, stuck_warn_threshold),
            stuck_abort_threshold=max(20, stuck_abort_threshold),
        )

    room_type = _state_room_type(state)
    act = _state_act(state)
    enemy_key = _state_enemy_key(state)
    room_type = _infer_room_type(room_type, enemy_key)
    base = _base_profile(
        room_type=room_type,
        enemy_key=enemy_key,
        no_action_proceed_threshold=no_action_proceed_threshold,
        no_action_abort_threshold=no_action_abort_threshold,
        stuck_warn_threshold=stuck_warn_threshold,
        stuck_abort_threshold=stuck_abort_threshold,
    )

    for override in _SPECIAL_OVERRIDES:
        if not _matches_override(room_type, enemy_key, act, override):
            continue
        proceed = (
            override.no_action_proceed_threshold
            if override.no_action_proceed_threshold > 0
            else base.no_action_proceed_threshold
        )
        no_action_abort = (
            override.no_action_abort_threshold
            if override.no_action_abort_threshold > 0
            else base.no_action_abort_threshold
        )
        warn = override.stuck_warn_threshold if override.stuck_warn_threshold > 0 else base.stuck_warn_threshold
        stuck_abort = (
            override.stuck_abort_threshold
            if override.stuck_abort_threshold > 0
            else base.stuck_abort_threshold
        )
        no_action_abort = max(no_action_abort, proceed + 1)
        stuck_abort = max(stuck_abort, warn + 1)
        return CombatStagnationProfile(
            profile_name=override.profile_name,
            room_type=room_type,
            enemy_key=enemy_key,
            no_action_proceed_threshold=proceed,
            no_action_abort_threshold=no_action_abort,
            stuck_warn_threshold=warn,
            stuck_abort_threshold=stuck_abort,
        )

    return base
