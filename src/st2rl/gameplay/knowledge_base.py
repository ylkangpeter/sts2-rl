# -*- coding: utf-8 -*-
"""Helpers for loading structured Slay the Spire 2 site knowledge."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent / "data" / "sts2_site_knowledge.json"


@lru_cache(maxsize=1)
def load_site_knowledge() -> dict[str, Any]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Knowledge file not found: {DATA_PATH}")
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def get_act_knowledge(act_name: str) -> dict[str, Any] | None:
    return load_site_knowledge().get("acts", {}).get(act_name)


def get_enemy(enemy_id: str) -> dict[str, Any] | None:
    for enemy in load_site_knowledge().get("enemies", []):
        if enemy.get("id") == enemy_id:
            return enemy
    return None


def get_character(character_id: str) -> dict[str, Any] | None:
    for character in load_site_knowledge().get("characters", []):
        if character.get("id") == character_id:
            return character
    return None


def get_card(card_id: str) -> dict[str, Any] | None:
    for card in load_site_knowledge().get("cards", []):
        if card.get("id") == card_id:
            return card
    return None
