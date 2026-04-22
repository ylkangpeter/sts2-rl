# -*- coding: utf-8 -*-
"""Canonical action and state types used by reusable game flow logic."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def card_needs_enemy_target(card: Dict[str, Any]) -> bool:
    """Return whether a card requires a living enemy target."""
    target_type = str(card.get("target_type") or "").strip().lower()
    return target_type in {"anyenemy", "enemy"}


@dataclass(slots=True)
class FlowAction:
    """Protocol-agnostic action selected by the flow logic."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GameStateView:
    """Canonical state view shared by flow logic and protocol adapters."""

    raw: Dict[str, Any]
    decision: str
    player: Dict[str, Any]
    hand: list[Dict[str, Any]]
    draw_pile: list[Dict[str, Any]]
    discard_pile: list[Dict[str, Any]]
    exhaust_pile: list[Dict[str, Any]]
    enemies: list[Dict[str, Any]]
    cards: list[Dict[str, Any]]
    options: list[Dict[str, Any]]
    choices: list[Dict[str, Any]]
    bundles: list[Dict[str, Any]]
    relics: list[Dict[str, Any]]
    potions: list[Dict[str, Any]]
    map_nodes: list[Dict[str, Any]]
    full_map: Dict[str, Any]
    round: Any
    turn: Any
    energy: int
    max_energy: int
    can_skip: bool
    min_select: int
    max_select: int
    selection_prompt: str
    game_over: bool
    victory: bool

    @classmethod
    def from_raw(cls, raw_state: Dict[str, Any], *, decision_alias: Optional[Dict[str, str]] = None) -> "GameStateView":
        state = dict(raw_state or {})
        raw_decision = str(state.get("decision") or "")
        decision = decision_alias.get(raw_decision, raw_decision) if decision_alias else raw_decision
        player = dict(state.get("player") or {})
        hp = int(player.get("hp") or 0)
        max_hp = int(player.get("max_hp") or 0)
        defeated = bool(player and max_hp > 0 and hp <= 0)
        return cls(
            raw=state,
            decision=decision,
            player=player,
            hand=list(state.get("hand") or []),
            draw_pile=list(state.get("draw_pile") or state.get("drawPile") or []),
            discard_pile=list(state.get("discard_pile") or state.get("discardPile") or []),
            exhaust_pile=list(state.get("exhaust_pile") or state.get("exhaustPile") or []),
            enemies=list(state.get("enemies") or []),
            cards=list(state.get("cards") or []),
            options=list(state.get("options") or []),
            choices=list(state.get("choices") or []),
            bundles=list(state.get("bundles") or []),
            relics=list(state.get("relics") or []),
            potions=list(state.get("potions") or []),
            map_nodes=list(state.get("map") or []),
            full_map=dict(state.get("full_map") or {}),
            round=state.get("round"),
            turn=state.get("turn"),
            energy=int(state.get("energy") or 0),
            max_energy=int(state.get("max_energy") or 0),
            can_skip=bool(state.get("can_skip", True)),
            min_select=int(state.get("min_select", 1) or 0),
            max_select=int(state.get("max_select", 1) or 0),
            selection_prompt=str(state.get("selection_prompt") or state.get("prompt") or ""),
            game_over=bool(state.get("game_over") or raw_decision == "game_over" or decision == "game_over" or defeated),
            victory=bool(state.get("victory", False)),
        )

    @property
    def hp(self) -> int:
        return int(self.player.get("hp") or 0)

    @property
    def max_hp(self) -> int:
        return int(self.player.get("max_hp") or 0)

    @property
    def block(self) -> int:
        return int(self.player.get("block") or 0)

    @property
    def gold(self) -> int:
        return int(self.player.get("gold") or 0)

    @property
    def deck_size(self) -> int:
        return int(self.player.get("deck_size") or 0)

    def living_enemies(self) -> list[Dict[str, Any]]:
        return [enemy for enemy in self.enemies if enemy.get("hp", 0) > 0]

    def playable_cards(self) -> list[Dict[str, Any]]:
        living_enemies = self.living_enemies()
        playable: list[Dict[str, Any]] = []
        for card in self.hand:
            if not card.get("can_play"):
                continue
            if card.get("cost", 99) > self.energy:
                continue
            if card_needs_enemy_target(card) and not living_enemies:
                continue
            playable.append(card)
        return playable

    def fingerprint(self) -> str:
        enemy_sig = tuple((enemy.get("index"), enemy.get("hp"), enemy.get("block")) for enemy in self.enemies)
        hand_sig = tuple((card.get("id"), card.get("index"), card.get("can_play"), card.get("cost")) for card in self.hand)
        return str(
            (
                self.decision,
                self.round,
                self.turn,
                self.energy,
                self.hp,
                self.gold,
                self.block,
                len(self.hand),
                enemy_sig,
                hand_sig,
            )
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "round": self.round,
            "turn": self.turn,
            "energy": self.energy,
            "max_energy": self.max_energy,
            "player_hp": self.hp,
            "player_max_hp": self.max_hp,
            "player_block": self.block,
            "gold": self.gold,
            "deck_size": self.deck_size,
            "hand_size": len(self.hand),
            "selection_prompt": self.selection_prompt,
            "draw_pile_count": int(self.raw.get("draw_pile_count") or len(self.draw_pile) or 0),
            "discard_pile_count": int(self.raw.get("discard_pile_count") or len(self.discard_pile) or 0),
            "exhaust_pile_count": int(self.raw.get("exhaust_pile_count") or len(self.exhaust_pile) or 0),
            "enemy_count": len(self.enemies),
            "living_enemies": len(self.living_enemies()),
            "enemy_hp": [(enemy.get("index"), enemy.get("hp"), enemy.get("block")) for enemy in self.enemies],
        }


@dataclass(slots=True)
class FlowRunResult:
    """Outcome for a single full-game flow run."""

    success: bool
    worker_id: int
    game_id: str
    seed: str
    steps: int
    decision: Optional[str] = None
    error: str = ""
    victory: bool = False
    total_reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "worker_id": self.worker_id,
            "game_id": self.game_id,
            "seed": self.seed,
            "steps": self.steps,
            "decision": self.decision,
            "error": self.error,
            "victory": self.victory,
            "total_reward": self.total_reward,
        }


@dataclass(slots=True)
class FlowRunSummary:
    """Aggregated outcome for parallel flow execution."""

    workers: int
    total_rounds: int
    total: int
    success: int
    fail: int
    victory: int
    defeat: int
    avg_steps: float
    min_steps: int
    max_steps: int
    avg_reward: float
    failed_runs: list[FlowRunResult] = field(default_factory=list)
