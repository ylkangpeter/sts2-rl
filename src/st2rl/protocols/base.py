# -*- coding: utf-8 -*-
"""Base protocol interface for backend-specific game communication."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from st2rl.gameplay.types import FlowAction, GameStateView


@dataclass(slots=True)
class ProtocolStartResult:
    """Normalized result returned by starting a new run."""

    game_id: str
    raw_state: Dict[str, Any]


@dataclass(slots=True)
class ProtocolStepResult:
    """Normalized result returned by a backend step call."""

    status: str
    state: Optional[Dict[str, Any]] = None
    reward: float = 0.0
    message: str = ""
    last_state: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class FlowProtocol(ABC):
    """Backend-specific adapter for full-game flow execution."""

    @abstractmethod
    def health_check(self, retries: int = 0) -> None:
        """Raise if the backend is unavailable."""

    @abstractmethod
    def start_game(self, character: str, seed: str) -> ProtocolStartResult:
        """Start a new run and return the session identifier and initial state."""

    @abstractmethod
    def get_state(self, game_id: str) -> Dict[str, Any]:
        """Fetch the current raw backend state for a game session."""

    @abstractmethod
    def step(self, game_id: str, action: FlowAction) -> ProtocolStepResult:
        """Execute one action and return a normalized step result."""

    @abstractmethod
    def close_game(self, game_id: str) -> None:
        """Close a running session."""

    @abstractmethod
    def adapt_state(self, raw_state: Dict[str, Any]) -> GameStateView:
        """Convert backend raw state into the canonical state view."""

    @abstractmethod
    def sanitize_action(self, state: GameStateView, action: FlowAction, rng: Any) -> FlowAction:
        """Adjust an action so it is valid for the current backend state."""

    @abstractmethod
    def recover_action_from_error(
        self,
        error_payload: Dict[str, Any],
        state: GameStateView,
    ) -> Optional[FlowAction]:
        """Map a backend error into a recovery action when possible."""
