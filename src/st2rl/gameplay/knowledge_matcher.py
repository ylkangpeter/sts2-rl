# -*- coding: utf-8 -*-
"""Knowledge matching and feature extraction for runtime game objects."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from st2rl.gameplay.knowledge_base import load_site_knowledge
from st2rl.gameplay.types import GameStateView

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MISMATCH_LOG_PATH = _ROOT / "logs" / "knowledge_mismatches.jsonl"

RARITY_SCORE = {
    "basic": 0.2,
    "common": 0.4,
    "uncommon": 0.7,
    "rare": 1.0,
    "starter": 0.3,
    "boss": 1.0,
    "shop": 0.8,
    "event": 0.8,
    "ancient": 0.9,
    "curse": 0.0,
    "status": 0.0,
}


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("the ", "")
    text = text.replace("'", "")
    text = re.sub(r"card\.", "", text)
    text = re.sub(r"relic\.", "", text)
    text = re.sub(r"potion\.", "", text)
    text = re.sub(r"character\.", "", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)
    return text


def _normalized_tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    parts = re.split(r"[^0-9a-z\u4e00-\u9fff]+", text)
    return {part for part in parts if part}


def _description_overlap(left: Any, right: Any) -> float:
    left_tokens = _normalized_tokens(left)
    right_tokens = _normalized_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


@dataclass(slots=True)
class MatchResult:
    matched: bool
    category: str
    runtime_name: str
    knowledge_id: str | None = None
    knowledge: dict[str, Any] | None = None
    score: float = 0.0
    reasons: list[str] | None = None


class KnowledgeMatcher:
    """Resolve runtime objects against the scraped site knowledge base."""

    def __init__(self, mismatch_log_path: str | Path | None = None):
        self.knowledge = load_site_knowledge()
        self.mismatch_log_path = Path(mismatch_log_path or DEFAULT_MISMATCH_LOG_PATH)
        self._seen_mismatches: set[str] = set()
        self._negative_match_cache: dict[str, str] = {}
        self._indices = {
            "enemy": self._build_index(self.knowledge.get("enemies", []), ("id", "name_en", "name_zh")),
            "card": self._build_index(self.knowledge.get("cards", []), ("id", "name")),
            "relic": self._build_index(self.knowledge.get("relics", []), ("id", "name")),
            "potion": self._build_index(self.knowledge.get("potions", []), ("id", "name")),
            "character": self._build_index(self.knowledge.get("characters", []), ("id", "name")),
            "event": self._build_index(self.knowledge.get("events", []), ("id", "title")),
        }

    def _build_index(self, rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
        index: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            aliases = set()
            for key in keys:
                value = row.get(key)
                if value:
                    aliases.add(_normalize_text(value))
            raw_aliases = row.get("alias") or row.get("aliases") or []
            if isinstance(raw_aliases, (list, tuple, set)):
                for value in raw_aliases:
                    normalized = _normalize_text(value)
                    if normalized:
                        aliases.add(normalized)
            for alias in aliases:
                index.setdefault(alias, []).append(row)
        return index

    def _runtime_cache_key(self, category: str, runtime_obj: dict[str, Any]) -> str:
        payload: dict[str, Any] = {
            "category": category,
            "id": runtime_obj.get("id"),
            "name": runtime_obj.get("name"),
            "name_en": runtime_obj.get("name_en"),
            "name_zh": runtime_obj.get("name_zh"),
            "title": runtime_obj.get("title"),
            "description": runtime_obj.get("description"),
            "type": runtime_obj.get("type"),
            "cost": runtime_obj.get("cost"),
            "hp": runtime_obj.get("hp"),
            "max_hp": runtime_obj.get("max_hp"),
            "max_energy": runtime_obj.get("max_energy"),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _negative_cache_reason(self, category: str, runtime_obj: dict[str, Any]) -> str | None:
        return self._negative_match_cache.get(self._runtime_cache_key(category, runtime_obj))

    def _remember_negative_cache(self, category: str, runtime_obj: dict[str, Any], reason: str) -> None:
        self._negative_match_cache[self._runtime_cache_key(category, runtime_obj)] = reason

    def _runtime_aliases(self, runtime_obj: dict[str, Any]) -> tuple[list[str], list[str]]:
        primary_aliases: list[str] = []
        fallback_aliases: list[str] = []
        for key in ("id", "name", "name_en", "name_zh", "title"):
            value = runtime_obj.get(key)
            if value:
                normalized = _normalize_text(value)
                if normalized:
                    primary_aliases.append(normalized)
        if runtime_obj.get("id"):
            normalized = _normalize_text(str(runtime_obj["id"]).split(".")[-1])
            if normalized:
                primary_aliases.append(normalized)
        raw_aliases = runtime_obj.get("alias") or runtime_obj.get("aliases") or []
        if isinstance(raw_aliases, (list, tuple, set)):
            for value in raw_aliases:
                normalized = _normalize_text(value)
                if normalized:
                    fallback_aliases.append(normalized)
        primary_aliases = list(dict.fromkeys(primary_aliases))
        fallback_aliases = [alias for alias in dict.fromkeys(fallback_aliases) if alias not in primary_aliases]
        return primary_aliases, fallback_aliases

    def _lookup_candidates(self, category: str, runtime_obj: dict[str, Any]) -> list[dict[str, Any]]:
        primary_aliases, fallback_aliases = self._runtime_aliases(runtime_obj)
        candidates: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for alias_group in (primary_aliases, fallback_aliases):
            for alias in alias_group:
                for row in self._indices[category].get(alias, []):
                    row_id = str(row.get("id"))
                    if row_id not in seen_ids:
                        seen_ids.add(row_id)
                        candidates.append(row)
            if candidates:
                break
        return candidates

    def _log_mismatch(
        self,
        *,
        category: str,
        runtime_obj: dict[str, Any],
        state: GameStateView | None,
        reason: str,
        candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        signature = json.dumps(
            {
                "category": category,
                "reason": reason,
                "runtime_name": runtime_obj.get("name") or runtime_obj.get("title") or runtime_obj.get("id"),
                "runtime_id": runtime_obj.get("id"),
                "candidate_ids": [candidate.get("id") for candidate in candidates or []],
                "decision": state.decision if state else None,
                "floor": ((state.raw.get("context") or {}).get("floor") if state else None),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if signature in self._seen_mismatches:
            return
        self._seen_mismatches.add(signature)
        self.mismatch_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "category": category,
            "reason": reason,
            "runtime_object": runtime_obj,
            "state_context": {
                "decision": state.decision if state else None,
                "context": dict(state.raw.get("context") or {}) if state else {},
            },
            "candidates": candidates or [],
        }
        with self.mismatch_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _negative_match_result(self, category: str, runtime_obj: dict[str, Any], reason: str) -> MatchResult:
        runtime_name = str(
            runtime_obj.get("name") or runtime_obj.get("title") or runtime_obj.get("id") or ""
        )
        return MatchResult(False, category, runtime_name, reasons=[reason])

    def match_enemy(self, enemy: dict[str, Any], state: GameStateView | None = None) -> MatchResult:
        cached_reason = self._negative_cache_reason("enemy", enemy)
        if cached_reason is not None:
            return self._negative_match_result("enemy", enemy, cached_reason)
        candidates = self._lookup_candidates("enemy", enemy)
        if not candidates:
            reason = "enemy_not_found"
            self._remember_negative_cache("enemy", enemy, reason)
            self._log_mismatch(category="enemy", runtime_obj=enemy, state=state, reason=reason)
            return self._negative_match_result("enemy", enemy, reason)

        runtime_max_hp = int(enemy.get("max_hp") or enemy.get("hp") or 0)
        runtime_hp = int(enemy.get("hp") or 0)
        best: MatchResult | None = None
        mismatched: list[dict[str, Any]] = []
        runtime_name = str(enemy.get("name") or enemy.get("id") or "")
        normalized_runtime_name = _normalize_text(runtime_name)

        for candidate in candidates:
            reasons: list[str] = []
            score = 0.0
            name_matched = normalized_runtime_name in {
                _normalize_text(candidate.get("name_en")),
                _normalize_text(candidate.get("name_zh")),
                _normalize_text(candidate.get("id")),
            }
            if name_matched:
                score += 5.0
                score += min(len(candidate.get("moves", [])), 8) / 8.0
                result = MatchResult(True, "enemy", runtime_name, str(candidate.get("id")), candidate, score, reasons)
                if best is None or result.score > best.score:
                    best = result
                continue

            hp_min = int(candidate.get("hp_min") or 0)
            hp_max = int(candidate.get("hp_max") or 0)
            if runtime_max_hp > 0 and hp_max > 0:
                if runtime_max_hp < hp_min - 5 or runtime_max_hp > hp_max + 5:
                    reasons.append(
                        f"enemy_hp_mismatch runtime_max_hp={runtime_max_hp} knowledge_range={hp_min}-{hp_max}"
                    )
                else:
                    score += 3.0
            if runtime_hp > 0 and hp_max > 0 and runtime_hp > hp_max + 5:
                reasons.append(f"enemy_current_hp_above_knowledge runtime_hp={runtime_hp} knowledge_max={hp_max}")
            if reasons:
                mismatched.append(candidate)
                continue

            score += min(len(candidate.get("moves", [])), 8) / 8.0
            result = MatchResult(True, "enemy", runtime_name, str(candidate.get("id")), candidate, score, reasons)
            if best is None or result.score > best.score:
                best = result

        if best is None:
            reason = "enemy_name_matched_but_stats_mismatch"
            self._remember_negative_cache("enemy", enemy, reason)
            self._log_mismatch(
                category="enemy",
                runtime_obj=enemy,
                state=state,
                reason=reason,
                candidates=mismatched,
            )
            return self._negative_match_result("enemy", enemy, reason)
        return best

    def match_card(self, card: dict[str, Any], state: GameStateView | None = None) -> MatchResult:
        cached_reason = self._negative_cache_reason("card", card)
        if cached_reason is not None:
            return self._negative_match_result("card", card, cached_reason)
        candidates = self._lookup_candidates("card", card)
        if not candidates:
            reason = "card_not_found"
            self._remember_negative_cache("card", card, reason)
            self._log_mismatch(category="card", runtime_obj=card, state=state, reason=reason)
            return self._negative_match_result("card", card, reason)

        runtime_name = str(card.get("name") or card.get("id") or "")
        runtime_cost = card.get("cost")
        runtime_type = _normalize_text(card.get("type"))
        runtime_description = card.get("description")
        best: MatchResult | None = None
        mismatched: list[dict[str, Any]] = []

        for candidate in candidates:
            reasons: list[str] = []
            score = 0.0
            name_matched = _normalize_text(runtime_name) in {
                _normalize_text(candidate.get("name")),
                _normalize_text(candidate.get("name_zh")),
                _normalize_text(candidate.get("id")),
            }
            type_matched = not runtime_type or not candidate.get("card_type") or runtime_type == _normalize_text(candidate.get("card_type"))
            if name_matched and type_matched:
                score += 4.0
                if runtime_cost is not None and candidate.get("energy") is not None and int(runtime_cost) == int(candidate.get("energy")):
                    score += 1.0
                overlap = _description_overlap(runtime_description, candidate.get("description"))
                score += overlap
                result = MatchResult(True, "card", runtime_name, str(candidate.get("id")), candidate, score, reasons)
                if best is None or result.score > best.score:
                    best = result
                continue
            energy = candidate.get("energy")
            if runtime_cost is not None and energy is not None:
                if int(runtime_cost) != int(energy):
                    reasons.append(f"card_cost_mismatch runtime_cost={runtime_cost} knowledge_cost={energy}")
                else:
                    score += 2.0
            if runtime_type and candidate.get("card_type"):
                if runtime_type != _normalize_text(candidate.get("card_type")):
                    reasons.append(
                        f"card_type_mismatch runtime_type={card.get('type')} knowledge_type={candidate.get('card_type')}"
                    )
                else:
                    score += 2.0
            overlap = _description_overlap(runtime_description, candidate.get("description"))
            if runtime_description:
                if overlap < 0.08:
                    reasons.append(f"card_description_mismatch overlap={overlap:.3f}")
                else:
                    score += overlap
            if reasons:
                mismatched.append(candidate)
                continue
            result = MatchResult(True, "card", runtime_name, str(candidate.get("id")), candidate, score, reasons)
            if best is None or result.score > best.score:
                best = result

        if best is None:
            reason = "card_name_matched_but_stats_mismatch"
            self._remember_negative_cache("card", card, reason)
            self._log_mismatch(
                category="card",
                runtime_obj=card,
                state=state,
                reason=reason,
                candidates=mismatched,
            )
            return self._negative_match_result("card", card, reason)
        return best

    def match_relic(self, relic: dict[str, Any], state: GameStateView | None = None) -> MatchResult:
        cached_reason = self._negative_cache_reason("relic", relic)
        if cached_reason is not None:
            return self._negative_match_result("relic", relic, cached_reason)
        candidates = self._lookup_candidates("relic", relic)
        if not candidates:
            reason = "relic_not_found"
            self._remember_negative_cache("relic", relic, reason)
            self._log_mismatch(category="relic", runtime_obj=relic, state=state, reason=reason)
            return self._negative_match_result("relic", relic, reason)

        runtime_description = relic.get("description")
        runtime_name = str(relic.get("name") or relic.get("id") or "")
        best: MatchResult | None = None
        for candidate in candidates:
            score = 4.0
            if _normalize_text(runtime_name) in {
                _normalize_text(candidate.get("name")),
                _normalize_text(candidate.get("name_zh")),
                _normalize_text(candidate.get("id")),
            }:
                overlap = _description_overlap(runtime_description, candidate.get("description"))
                score += overlap
                result = MatchResult(True, "relic", runtime_name, str(candidate.get("id")), candidate, score, [])
                if best is None or result.score > best.score:
                    best = result
                continue
            overlap = _description_overlap(runtime_description, candidate.get("description"))
            if runtime_description and overlap < 0.06:
                continue
            score += overlap
            result = MatchResult(True, "relic", runtime_name, str(candidate.get("id")), candidate, score, [])
            if best is None or result.score > best.score:
                best = result
        if best is None:
            reason = "relic_name_matched_but_description_mismatch"
            self._remember_negative_cache("relic", relic, reason)
            self._log_mismatch(
                category="relic",
                runtime_obj=relic,
                state=state,
                reason=reason,
                candidates=candidates,
            )
            return self._negative_match_result("relic", relic, reason)
        return best

    def match_potion(self, potion: dict[str, Any], state: GameStateView | None = None) -> MatchResult:
        cached_reason = self._negative_cache_reason("potion", potion)
        if cached_reason is not None:
            return self._negative_match_result("potion", potion, cached_reason)
        candidates = self._lookup_candidates("potion", potion)
        if not candidates:
            reason = "potion_not_found"
            self._remember_negative_cache("potion", potion, reason)
            self._log_mismatch(category="potion", runtime_obj=potion, state=state, reason=reason)
            return self._negative_match_result("potion", potion, reason)

        runtime_description = potion.get("description")
        runtime_name = str(potion.get("name") or potion.get("id") or "")
        best: MatchResult | None = None
        for candidate in candidates:
            score = 4.0
            if _normalize_text(runtime_name) in {
                _normalize_text(candidate.get("name")),
                _normalize_text(candidate.get("name_zh")),
                _normalize_text(candidate.get("id")),
            }:
                overlap = _description_overlap(runtime_description, candidate.get("description"))
                score += overlap
                result = MatchResult(True, "potion", runtime_name, str(candidate.get("id")), candidate, score, [])
                if best is None or result.score > best.score:
                    best = result
                continue
            overlap = _description_overlap(runtime_description, candidate.get("description"))
            if runtime_description and overlap < 0.06:
                continue
            score += overlap
            result = MatchResult(True, "potion", runtime_name, str(candidate.get("id")), candidate, score, [])
            if best is None or result.score > best.score:
                best = result
        if best is None:
            reason = "potion_name_matched_but_description_mismatch"
            self._remember_negative_cache("potion", potion, reason)
            self._log_mismatch(
                category="potion",
                runtime_obj=potion,
                state=state,
                reason=reason,
                candidates=candidates,
            )
            return self._negative_match_result("potion", potion, reason)
        return best

    def match_character(self, state: GameStateView) -> MatchResult:
        runtime_character = {
            "id": state.raw.get("character") or state.player.get("id") or state.player.get("name"),
            "name": state.player.get("name"),
            "hp": state.max_hp,
            "gold": state.gold,
            "max_energy": state.max_energy,
        }
        cached_reason = self._negative_cache_reason("character", runtime_character)
        if cached_reason is not None:
            return self._negative_match_result("character", runtime_character, cached_reason)
        candidates = self._lookup_candidates("character", runtime_character)
        if not candidates:
            reason = "character_not_found"
            self._remember_negative_cache("character", runtime_character, reason)
            self._log_mismatch(
                category="character",
                runtime_obj=runtime_character,
                state=state,
                reason=reason,
            )
            return self._negative_match_result("character", runtime_character, reason)

        for candidate in candidates:
            if _normalize_text(runtime_character.get("name")) in {
                _normalize_text(candidate.get("name")),
                _normalize_text(candidate.get("name_zh")),
                _normalize_text(candidate.get("id")),
            }:
                return MatchResult(True, "character", str(runtime_character.get("name")), str(candidate.get("id")), candidate, 1.0, [])
            reasons: list[str] = []
            if state.max_hp and candidate.get("starting_hp") and state.max_hp < int(candidate["starting_hp"]) - 20:
                reasons.append("character_hp_too_low_for_candidate")
            if state.max_energy and candidate.get("max_energy") and int(state.max_energy) != int(candidate["max_energy"]):
                reasons.append("character_max_energy_mismatch")
            if not reasons:
                return MatchResult(True, "character", str(runtime_character.get("name")), str(candidate.get("id")), candidate, 1.0, [])

        reason = "character_name_matched_but_stats_mismatch"
        self._remember_negative_cache("character", runtime_character, reason)
        self._log_mismatch(
            category="character",
            runtime_obj=runtime_character,
            state=state,
            reason=reason,
            candidates=candidates,
        )
        return self._negative_match_result("character", runtime_character, reason)

    def summarize_state(self, state: GameStateView) -> dict[str, Any]:
        card_matches = [self.match_card(card, state) for card in state.hand]
        enemy_matches = [self.match_enemy(enemy, state) for enemy in state.enemies]
        relic_matches = [self.match_relic(relic, state) for relic in state.relics]
        potion_matches = [self.match_potion(potion, state) for potion in state.potions]
        character_match = self.match_character(state)

        matched_cards = [match.knowledge for match in card_matches if match.matched and match.knowledge]
        matched_enemies = [match.knowledge for match in enemy_matches if match.matched and match.knowledge]
        matched_relics = [match.knowledge for match in relic_matches if match.matched and match.knowledge]
        matched_potions = [match.knowledge for match in potion_matches if match.matched and match.knowledge]

        def ratio(matches: list[MatchResult]) -> float:
            return float(sum(1 for match in matches if match.matched)) / max(1, len(matches))

        def avg(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        def rarity_score_of(row: dict[str, Any]) -> float:
            rarity = _normalize_text(row.get("rarity"))
            return RARITY_SCORE.get(rarity, 0.5)

        def encounter_flag(enemy_row: dict[str, Any], target_tier: str) -> float:
            entries = enemy_row.get("appears_in", [])
            return 1.0 if any(_normalize_text(item.get("tier")) == _normalize_text(target_tier) for item in entries) else 0.0

        hand_types = [_normalize_text(card.get("card_type")) for card in matched_cards]
        enemy_hp_max = [float(enemy.get("hp_max") or 0) for enemy in matched_enemies]
        enemy_move_counts = [float(len(enemy.get("moves") or [])) for enemy in matched_enemies]
        enemy_consistency = []
        for runtime_enemy, match in zip(state.enemies, enemy_matches):
            if not match.matched or not match.knowledge:
                enemy_consistency.append(0.0)
                continue
            hp_min = float(match.knowledge.get("hp_min") or 0)
            hp_max = float(match.knowledge.get("hp_max") or 0)
            runtime_max_hp = float(runtime_enemy.get("max_hp") or runtime_enemy.get("hp") or 0)
            if hp_max <= 0 or runtime_max_hp <= 0:
                enemy_consistency.append(0.0)
            elif hp_min - 5 <= runtime_max_hp <= hp_max + 5:
                enemy_consistency.append(1.0)
            else:
                enemy_consistency.append(0.0)

        features = [
            1.0 if character_match.matched else 0.0,
            ratio(card_matches),
            ratio(enemy_matches),
            ratio(relic_matches),
            ratio(potion_matches),
            avg([float(card.get("energy") or 0) / 5.0 for card in matched_cards]),
            avg([rarity_score_of(card) for card in matched_cards]),
            avg([1.0 if card_type == "attack" else 0.0 for card_type in hand_types]),
            avg([1.0 if card_type == "skill" else 0.0 for card_type in hand_types]),
            avg([1.0 if card_type == "power" else 0.0 for card_type in hand_types]),
            avg([1.0 if card_type in ("status", "curse") else 0.0 for card_type in hand_types]),
            avg([hp / 250.0 for hp in enemy_hp_max]),
            (max(enemy_hp_max) / 300.0) if enemy_hp_max else 0.0,
            avg([count / 8.0 for count in enemy_move_counts]),
            avg([encounter_flag(enemy, "weak") for enemy in matched_enemies]),
            avg([encounter_flag(enemy, "monster") for enemy in matched_enemies]),
            avg([encounter_flag(enemy, "elite") for enemy in matched_enemies]),
            avg([encounter_flag(enemy, "boss") for enemy in matched_enemies]),
            avg(enemy_consistency),
            avg([rarity_score_of(relic) for relic in matched_relics]),
            avg([1.0 if "Shared" in (relic.get("relic_pools") or []) else 0.0 for relic in matched_relics]),
            avg(
                [
                    1.0
                    if any(pool in {"Event", "Shop"} for pool in (relic.get("relic_pools") or []))
                    else 0.0
                    for relic in matched_relics
                ]
            ),
            avg([rarity_score_of(potion) for potion in matched_potions]),
            avg([1.0 if _normalize_text(potion.get("usage")) == "combatonly" else 0.0 for potion in matched_potions]),
        ]

        return {
            "features": features,
            "matches": {
                "character": character_match,
                "cards": card_matches,
                "enemies": enemy_matches,
                "relics": relic_matches,
                "potions": potion_matches,
            },
        }
