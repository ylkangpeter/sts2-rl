# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.analyze_boss_floor import _aggregate
from scripts.training_dashboard import _build_act1_metrics
from st2rl.gameplay.enemy_intent_script import describe_enemy_intent, forecast_enemy_intent
from st2rl.gameplay.types import GameStateView


def _state_view(raw: dict) -> GameStateView:
    return GameStateView.from_raw(raw)


def test_forecast_enemy_intent_returns_structured_act1_rows() -> None:
    state = _state_view(
        {
            "decision": "combat_play",
            "round": 1,
            "energy": 3,
            "max_energy": 3,
            "player": {"hp": 80, "max_hp": 80, "block": 0, "deck": [], "gold": 99},
            "enemies": [
                {
                    "index": 0,
                    "id": "ENEMY.VANTOM_BOSS",
                    "name": "Vantom Boss",
                    "hp": 140,
                    "max_hp": 140,
                    "intent": "Attack",
                    "intent_damage": 9,
                }
            ],
            "context": {"act": 1, "floor": 17, "room_type": "Boss"},
            "hand": [],
        }
    )

    forecast = forecast_enemy_intent(state, horizon=3)

    assert forecast["matched_enemies"] == 1
    assert forecast["expected_damage_turn_plus_one"] >= 0
    assert forecast["predicted_two_turn_pressure"] >= forecast["expected_damage_turn_plus_one"]
    assert len(forecast["rows"]) == 1
    row = forecast["rows"][0]
    assert row["script"]
    assert "next_turn_forecast" in row
    assert len(row["short_horizon_forecast"]) == 3
    assert "intent_categories" in row["next_turn_forecast"]


def test_describe_enemy_intent_reports_mismatch_and_phase() -> None:
    state = _state_view(
        {
            "decision": "combat_play",
            "round": 1,
            "energy": 3,
            "max_energy": 3,
            "player": {"hp": 80, "max_hp": 80, "block": 0, "deck": [], "gold": 99},
            "enemies": [
                {
                    "index": 0,
                    "id": "ENEMY.KIN_PRIEST",
                    "name": "Kin Priest",
                    "hp": 44,
                    "max_hp": 44,
                    "intent": "Attack",
                    "intent_damage": 12,
                }
            ],
            "context": {"act": 1, "floor": 17, "room_type": "Boss"},
            "hand": [],
        }
    )

    snapshot = describe_enemy_intent(state, horizon=3)

    assert snapshot["matched_enemies"] == 1
    assert snapshot["mismatch_count"] == 1
    row = snapshot["rows"][0]
    assert row["mismatch"] is True
    assert row["expected_step"]["special_phase_tag"] == "setup"
    assert row["short_horizon_forecast"][1]["special_phase_tag"] == "pressure"


def test_build_act1_metrics_computes_rates_breakdown_and_entry_samples() -> None:
    rows = [
        {
            "game_id": "g3",
            "finished_at": "2026-05-01T00:03:00",
            "episode_index": 3,
            "boss_name": "The Kin",
            "act1_boss_attempt": True,
            "act1_boss_clear": False,
            "max_floor": 17,
            "final_floor": 17,
            "final_hp": 8,
            "max_hp": 80,
        },
        {
            "game_id": "g2",
            "finished_at": "2026-05-01T00:02:00",
            "episode_index": 2,
            "boss_name": "Ceremonial Beast",
            "act1_boss_attempt": True,
            "act1_boss_clear": True,
            "max_floor": 20,
            "final_floor": 19,
            "final_hp": 42,
            "max_hp": 80,
        },
        {
            "game_id": "g1",
            "finished_at": "2026-05-01T00:01:00",
            "episode_index": 1,
            "boss_name": "Vantom",
            "act1_boss_attempt": False,
            "act1_boss_clear": False,
            "max_floor": 6,
            "final_floor": 6,
            "final_hp": 60,
            "max_hp": 80,
        },
    ]
    session_details = {
        "g2": {
            "nodes": [
                {
                    "act": 1,
                    "floor": 17,
                    "room_type": "Boss",
                    "entry_state": {
                        "hp": 55,
                        "max_hp": 80,
                        "deck": [{"id": "CARD.STRIKE"}] * 14,
                        "potions": [{"id": "POTION.BLOCK"}, {"id": "POTION.STRENGTH"}],
                    },
                }
            ]
        }
    }

    metrics = _build_act1_metrics(rows, session_details=session_details, window=500)

    assert metrics["window"] == 3
    assert metrics["attempts"] == 2
    assert metrics["clears"] == 1
    assert metrics["act1_boss_reach_rate_500"] == 66.67
    assert metrics["act1_boss_clear_given_reach_rate_500"] == 50.0
    assert metrics["act1_boss_clear_rate_500"] == 33.33
    assert any(item["boss"] == "Ceremonial Beast" for item in metrics["bosses"])
    ceremonial = next(item for item in metrics["bosses"] if item["boss"] == "Ceremonial Beast")
    assert ceremonial["avg_entry_hp"] == 55.0
    assert ceremonial["avg_entry_deck_size"] == 14.0
    assert ceremonial["avg_entry_potions"] == 2.0
    assert metrics["failure_reasons"]["route_failure"] == 1


def test_boss_analysis_aggregate_tracks_mismatch_hotspots() -> None:
    results = [
        {
            "seed": "s1",
            "error": "",
            "boss_attempt": True,
            "boss_clear": False,
            "boss": "KIN_FOLLOWER+KIN_FOLLOWER+KIN_PRIEST",
            "entry_hp": 80,
            "boss_rounds": 7,
            "cards_played": 18,
            "avg_playable": 3.0,
            "hp_loss_cards": 0,
            "potions_used": 1,
            "intent_mismatch_count": 2,
            "deck": {},
            "played_cards": {},
            "trace": [
                {
                    "round": 3,
                    "intent_mismatches": [
                        {"enemy": "Kin Priest", "expected_phase": "scale"},
                        {"enemy": "Kin Priest", "expected_phase": "scale"},
                    ]
                }
            ],
        },
        {
            "seed": "s2",
            "error": "",
            "boss_attempt": True,
            "boss_clear": True,
            "boss": "VANTOM",
            "entry_hp": 76,
            "boss_rounds": 11,
            "cards_played": 32,
            "avg_playable": 3.4,
            "hp_loss_cards": 1,
            "potions_used": 0,
            "intent_mismatch_count": 0,
            "deck": {},
            "played_cards": {},
            "trace": [],
        },
    ]

    aggregate = _aggregate(results)

    kin = aggregate["by_boss_mismatch"]["KIN_FOLLOWER+KIN_FOLLOWER+KIN_PRIEST"]
    assert kin["mismatch_attempts"] == 1
    assert kin["total_mismatches"] == 2
    assert kin["top_rounds"][0]["round"] == "3"
    assert aggregate["mismatch_hotspots"][0]["boss"] == "KIN_FOLLOWER+KIN_FOLLOWER+KIN_PRIEST"
