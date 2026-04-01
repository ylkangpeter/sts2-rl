# -*- coding: utf-8 -*-
"""Reusable game mechanics models for planning, simulation, and RL features."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

UnknownRoomOutcome = Literal["Monster", "Treasure", "Shop", "Event"]

UNKNOWN_ROOM_OUTCOMES: tuple[UnknownRoomOutcome, ...] = (
    "Monster",
    "Treasure",
    "Shop",
    "Event",
)


@dataclass(frozen=True, slots=True)
class UnknownRoomWeights:
    """Weighted outcome snapshot for a single `?` room."""

    monster: int
    treasure: int
    shop: int
    event: int

    def as_dict(self) -> dict[str, int]:
        return {
            "Monster": self.monster,
            "Treasure": self.treasure,
            "Shop": self.shop,
            "Event": self.event,
        }

    def total_weight(self) -> int:
        return self.monster + self.treasure + self.shop + self.event


@dataclass(slots=True)
class UnknownRoomMechanics:
    """Tracks weighted `?` room outcomes.

    Rule model:
    - Each outcome has its own escalation track.
    - If an outcome does not happen, its weight advances to the next tier.
    - If an outcome happens, its weight resets to the first tier for the next `?`.

    This matches the user-provided example:
    - first `?`: M10 / T2 / S3 / E85
    - if first `?` became Monster, next `?`: M10 / T4 / S6 / E70
    """

    schedules: dict[UnknownRoomOutcome, tuple[int, ...]] = field(
        default_factory=lambda: {
            "Monster": (10, 20, 30, 40),
            "Treasure": (2, 4, 6, 8),
            "Shop": (3, 6, 9, 12),
            "Event": (85, 70, 55, 40),
        }
    )
    miss_counts: dict[UnknownRoomOutcome, int] = field(
        default_factory=lambda: {outcome: 0 for outcome in UNKNOWN_ROOM_OUTCOMES}
    )

    def current_weights(self) -> UnknownRoomWeights:
        return UnknownRoomWeights(
            monster=self._weight_for("Monster"),
            treasure=self._weight_for("Treasure"),
            shop=self._weight_for("Shop"),
            event=self._weight_for("Event"),
        )

    def record_outcome(self, outcome: UnknownRoomOutcome) -> UnknownRoomWeights:
        if outcome not in self.schedules:
            raise ValueError(f"Unsupported `?` outcome: {outcome}")

        for name in UNKNOWN_ROOM_OUTCOMES:
            if name == outcome:
                self.miss_counts[name] = 0
                continue
            max_index = len(self.schedules[name]) - 1
            self.miss_counts[name] = min(self.miss_counts[name] + 1, max_index)
        return self.current_weights()

    def sample_outcome(self, rng: random.Random) -> UnknownRoomOutcome:
        weights = self.current_weights().as_dict()
        pick = rng.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]
        return pick  # type: ignore[return-value]

    def preview_after(self, outcome: UnknownRoomOutcome) -> UnknownRoomWeights:
        clone = UnknownRoomMechanics(
            schedules={key: tuple(values) for key, values in self.schedules.items()},
            miss_counts=dict(self.miss_counts),
        )
        return clone.record_outcome(outcome)

    def reset(self) -> None:
        for outcome in UNKNOWN_ROOM_OUTCOMES:
            self.miss_counts[outcome] = 0

    def snapshot(self) -> dict[str, object]:
        return {
            "weights": self.current_weights().as_dict(),
            "miss_counts": dict(self.miss_counts),
        }

    def _weight_for(self, outcome: UnknownRoomOutcome) -> int:
        schedule = self.schedules[outcome]
        index = min(self.miss_counts[outcome], len(schedule) - 1)
        return int(schedule[index])


@dataclass(slots=True)
class EliteEncounterMechanics:
    """Tracks elite uniqueness within an act.

    Assumption:
    - "每一层的精英怪之间不会重复" is modeled as "within the same act, elites should
      not repeat until the pool is exhausted".
    """

    seen_by_act: dict[int, set[str]] = field(default_factory=dict)

    def available_elites(self, act: int, elite_pool: list[str]) -> list[str]:
        seen = self.seen_by_act.setdefault(act, set())
        available = [elite_id for elite_id in elite_pool if elite_id not in seen]
        return available or list(elite_pool)

    def record_elite(self, act: int, elite_id: str) -> None:
        self.seen_by_act.setdefault(act, set()).add(elite_id)

    def choose_elite(self, act: int, elite_pool: list[str], rng: random.Random) -> str:
        available = self.available_elites(act, elite_pool)
        if not available:
            raise ValueError("Elite pool is empty")
        elite_id = rng.choice(available)
        self.record_elite(act, elite_id)
        return elite_id

    def reset_act(self, act: int) -> None:
        self.seen_by_act.pop(act, None)

    def snapshot(self) -> dict[int, list[str]]:
        return {act: sorted(seen) for act, seen in self.seen_by_act.items()}


@dataclass(slots=True)
class TreasureMechanics:
    """Treasure-room mechanics placeholder.

    The structure is intentionally lightweight for now so later treasure rules can
    be added without changing callers.
    """

    seen_treasures_by_act: dict[int, list[str]] = field(default_factory=dict)

    def record_treasure(self, act: int, treasure_id: str) -> None:
        self.seen_treasures_by_act.setdefault(act, []).append(treasure_id)

    def snapshot(self) -> dict[int, list[str]]:
        return {act: list(treasures) for act, treasures in self.seen_treasures_by_act.items()}


@dataclass(slots=True)
class MonsterMechanics:
    """Monster encounter mechanics container."""

    elite: EliteEncounterMechanics = field(default_factory=EliteEncounterMechanics)

    def snapshot(self) -> dict[str, object]:
        return {
            "elite": self.elite.snapshot(),
        }


@dataclass(slots=True)
class GameplayMechanics:
    """Top-level mechanics state shared by planning, policy, and telemetry."""

    unknown_room: UnknownRoomMechanics = field(default_factory=UnknownRoomMechanics)
    treasure: TreasureMechanics = field(default_factory=TreasureMechanics)
    monster: MonsterMechanics = field(default_factory=MonsterMechanics)

    def reset_for_new_run(self) -> None:
        self.unknown_room.reset()
        self.treasure = TreasureMechanics()
        self.monster = MonsterMechanics()

    def snapshot(self) -> dict[str, object]:
        return {
            "unknown_room": self.unknown_room.snapshot(),
            "treasure": self.treasure.snapshot(),
            "monster": self.monster.snapshot(),
        }
