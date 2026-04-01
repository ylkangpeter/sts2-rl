# -*- coding: utf-8 -*-
"""Build a structured Slay the Spire 2 knowledge base from slaythespire2.gg."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "src" / "st2rl" / "gameplay" / "data" / "sts2_site_knowledge.json"
BASE_URL = "https://slaythespire2.gg"
USER_AGENT = "Mozilla/5.0 (compatible; st2rl-knowledge-builder/1.0)"
LOCALE_PREFIXES = {
    "en": "",
    "zh": "/zh",
}

PAGE_CONFIGS = {
    "cards": {"path": "/cards", "anchor": '"category":"CARD","items":'},
    "relics": {"path": "/relics", "anchor": '"category":"RELIC","items":'},
    "potions": {"path": "/potions", "anchor": '"category":"POTION","items":'},
    "characters": {"path": "/characters", "anchor": '"category":"CHARACTER","items":'},
    "events": {"path": "/events", "anchor": '"events":'},
    "enemies": {"path": "/enemies", "anchor": '"enemies":'},
    "encounters": {"path": "/encounters", "anchor": '"encounters":'},
}


def fetch_html(path: str, *, locale: str = "en") -> str:
    prefix = LOCALE_PREFIXES.get(locale, "")
    response = requests.get(
        f"{BASE_URL}{prefix}{path}",
        timeout=30,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.text


def decode_next_stream(html: str) -> str:
    parts = [
        match.group(2)
        for match in re.finditer(r'self\.__next_f\.push\(\[(\d+),"(.*?)"\]\)</script>', html, re.S)
    ]
    return "".join(parts).encode("utf-8").decode("unicode_escape", errors="ignore")


def extract_json_array(text: str, anchor: str) -> list[dict[str, Any]]:
    anchor_index = text.find(anchor)
    if anchor_index < 0:
        raise ValueError(f"Anchor not found: {anchor}")

    start = text.find("[", anchor_index)
    if start < 0:
        raise ValueError(f"Array start not found for anchor: {anchor}")

    depth = 0
    in_string = False
    escape = False
    end = -1
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if end < 0:
        raise ValueError(f"Array end not found for anchor: {anchor}")

    return json.loads(text[start:end])


def repair_text(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: repair_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [repair_text(item) for item in value]
    if not isinstance(value, str):
        return value
    try:
        repaired = value.encode("latin1").decode("utf-8")
    except Exception:
        return value
    return repaired


def normalize_markup(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.replace("[gold]", "").replace("[/gold]", "")
    cleaned = cleaned.replace("[green]", "").replace("[/green]", "")
    cleaned = cleaned.replace("[blue]", "").replace("[/blue]", "")
    cleaned = cleaned.replace("[red]", "").replace("[/red]", "")
    cleaned = cleaned.replace("[purple]", "").replace("[/purple]", "")
    cleaned = cleaned.replace("[orange]", "").replace("[/orange]", "")
    cleaned = cleaned.replace("[aqua]", "").replace("[/aqua]", "")
    cleaned = cleaned.replace("[b]", "").replace("[/b]", "")
    cleaned = cleaned.replace("[i]", "").replace("[/i]", "")
    cleaned = cleaned.replace("[jitter]", "").replace("[/jitter]", "")
    cleaned = cleaned.replace("[sine]", "").replace("[/sine]", "")
    cleaned = cleaned.replace("\\n", "\n")
    cleaned = re.sub(r"\{Energy:energyIcons\((\d+)\)\}", r"Energy(\1)", cleaned)
    cleaned = re.sub(r"\s+\n", "\n", cleaned).strip()
    return cleaned


def load_page_arrays(locale: str = "en") -> dict[str, list[dict[str, Any]]]:
    arrays: dict[str, list[dict[str, Any]]] = {}
    for name, config in PAGE_CONFIGS.items():
        html = fetch_html(config["path"], locale=locale)
        decoded = decode_next_stream(html)
        rows = extract_json_array(decoded, config["anchor"])
        arrays[name] = repair_text(rows)
    return arrays


def build_aliases(*values: Any) -> list[str]:
    aliases: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                text = str(item or "").strip()
                if text and text not in aliases:
                    aliases.append(text)
            continue
        text = str(value).strip()
        if text and text not in aliases:
            aliases.append(text)
    return aliases


def merge_localized_rows(
    english_rows: list[dict[str, Any]],
    chinese_rows: list[dict[str, Any]],
    *,
    id_keys: tuple[str, ...],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    zh_index: dict[str, dict[str, Any]] = {}
    for row in chinese_rows:
        row_key = None
        for key in id_keys:
            value = row.get(key)
            if value:
                row_key = str(value)
                break
        if row_key:
            zh_index[row_key] = row

    merged: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for row in english_rows:
        row_key = None
        for key in id_keys:
            value = row.get(key)
            if value:
                row_key = str(value)
                break
        merged.append((row, zh_index.get(row_key) if row_key else None))
    return merged


def build_acts(encounters: list[dict[str, Any]], enemies_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    acts: dict[str, Any] = {}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for encounter in encounters:
        act = encounter.get("act", "UNKNOWN")
        grouped[act][encounter.get("tier", "unknown")].append(encounter)

    for act, tier_map in grouped.items():
        unique_enemy_ids: set[str] = set()
        encounter_rows = []
        for tier, entries in tier_map.items():
            for entry in entries:
                unique_enemy_ids.update(entry.get("monsterIds", []))
                encounter_rows.append(
                    {
                        "id": entry.get("id"),
                        "name_en": entry.get("nameEn"),
                        "name_zh": entry.get("nameZh"),
                        "tier": tier,
                        "section": entry.get("section"),
                        "monster_ids": entry.get("monsterIds", []),
                        "note": entry.get("note"),
                        "description": normalize_markup(entry.get("description")),
                    }
                )

        acts[act] = {
            "encounter_count": len(encounter_rows),
            "unique_enemy_count": len(unique_enemy_ids),
            "tiers": {
                tier: [
                    {
                        "id": item.get("id"),
                        "name_en": item.get("nameEn"),
                        "name_zh": item.get("nameZh"),
                        "monster_ids": item.get("monsterIds", []),
                    }
                    for item in tier_entries
                ]
                for tier, tier_entries in sorted(tier_map.items())
            },
            "enemy_summaries": [
                {
                    "id": enemy_id,
                    "name_en": enemies_by_id.get(enemy_id, {}).get("nameEn"),
                    "name_zh": enemies_by_id.get(enemy_id, {}).get("nameZh"),
                    "hp_min": enemies_by_id.get(enemy_id, {}).get("hpMin"),
                    "hp_max": enemies_by_id.get(enemy_id, {}).get("hpMax"),
                }
                for enemy_id in sorted(unique_enemy_ids)
            ],
        }

    return acts


def enrich_enemies(
    enemy_rows: list[tuple[dict[str, Any], dict[str, Any] | None]],
    encounters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enemy_to_encounters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for encounter in encounters:
        for monster_id in encounter.get("monsterIds", []):
            enemy_to_encounters[monster_id].append(
                {
                    "encounter_id": encounter.get("id"),
                    "encounter_name_en": encounter.get("nameEn"),
                    "encounter_name_zh": encounter.get("nameZh"),
                    "act": encounter.get("act"),
                    "tier": encounter.get("tier"),
                    "section": encounter.get("section"),
                }
            )

    rows = []
    for enemy, enemy_zh in enemy_rows:
        moves = []
        zh_moves = enemy_zh.get("moves", []) if isinstance(enemy_zh, dict) else []
        for index, move in enumerate(enemy.get("moves", [])):
            move_zh = zh_moves[index] if index < len(zh_moves) and isinstance(zh_moves[index], dict) else {}
            moves.append(
                {
                    "name_en": move.get("nameEn"),
                    "name_zh": move.get("nameZh") or move_zh.get("nameZh") or move_zh.get("nameEn"),
                    "intents": move.get("intents", []),
                }
            )

        rows.append(
            {
                "id": enemy.get("id"),
                "name_en": enemy.get("nameEn"),
                "name_zh": enemy.get("nameZh") or (enemy_zh or {}).get("nameZh") or (enemy_zh or {}).get("nameEn"),
                "hp_min": enemy.get("hpMin"),
                "hp_max": enemy.get("hpMax"),
                "has_image": enemy.get("hasImage"),
                "moves": moves,
                "appears_in": enemy_to_encounters.get(enemy.get("id"), []),
                "alias": build_aliases(
                    enemy.get("nameEn"),
                    enemy.get("nameZh"),
                    (enemy_zh or {}).get("nameZh"),
                    (enemy_zh or {}).get("nameEn"),
                    enemy.get("id"),
                ),
            }
        )
    return rows


def normalize_cards(card_rows: list[tuple[dict[str, Any], dict[str, Any] | None]]) -> list[dict[str, Any]]:
    return [
        {
            "id": card.get("id"),
            "name": card.get("name"),
            "name_en": card.get("name"),
            "name_zh": (card_zh or {}).get("name"),
            "character": card.get("character"),
            "rarity": card.get("rarity"),
            "energy": card.get("energy"),
            "card_type": card.get("cardType"),
            "description": normalize_markup(card.get("description")),
            "description_en": normalize_markup(card.get("description")),
            "description_zh": normalize_markup((card_zh or {}).get("description")),
            "description_upgraded": normalize_markup(card.get("descriptionUpgraded")),
            "description_upgraded_en": normalize_markup(card.get("descriptionUpgraded")),
            "description_upgraded_zh": normalize_markup((card_zh or {}).get("descriptionUpgraded")),
            "localization_key": card.get("localizationKey"),
            "data_source": card.get("dataSource"),
            "alias": build_aliases(
                card.get("name"),
                (card_zh or {}).get("name"),
                card.get("id"),
                card.get("localizationKey"),
            ),
        }
        for card, card_zh in card_rows
    ]


def normalize_relics(relic_rows: list[tuple[dict[str, Any], dict[str, Any] | None]]) -> list[dict[str, Any]]:
    return [
        {
            "id": relic.get("id"),
            "name": relic.get("name"),
            "name_en": relic.get("name"),
            "name_zh": (relic_zh or {}).get("name"),
            "rarity": relic.get("rarity"),
            "description": normalize_markup(relic.get("description")),
            "description_en": normalize_markup(relic.get("description")),
            "description_zh": normalize_markup((relic_zh or {}).get("description")),
            "relic_pools": relic.get("relicPools", []),
            "data_source": relic.get("dataSource"),
            "alias": build_aliases(relic.get("name"), (relic_zh or {}).get("name"), relic.get("id")),
        }
        for relic, relic_zh in relic_rows
    ]


def normalize_potions(potion_rows: list[tuple[dict[str, Any], dict[str, Any] | None]]) -> list[dict[str, Any]]:
    return [
        {
            "id": potion.get("id"),
            "name": potion.get("name"),
            "name_en": potion.get("name"),
            "name_zh": (potion_zh or {}).get("name"),
            "rarity": potion.get("rarity"),
            "usage": potion.get("usage"),
            "description": normalize_markup(potion.get("description")),
            "description_en": normalize_markup(potion.get("description")),
            "description_zh": normalize_markup((potion_zh or {}).get("description")),
            "data_source": potion.get("dataSource"),
            "alias": build_aliases(potion.get("name"), (potion_zh or {}).get("name"), potion.get("id")),
        }
        for potion, potion_zh in potion_rows
    ]


def normalize_characters(character_rows: list[tuple[dict[str, Any], dict[str, Any] | None]]) -> list[dict[str, Any]]:
    return [
        {
            "id": character.get("id"),
            "name": character.get("name"),
            "name_en": character.get("name"),
            "name_zh": (character_zh or {}).get("name"),
            "description": normalize_markup(character.get("description")),
            "description_en": normalize_markup(character.get("description")),
            "description_zh": normalize_markup((character_zh or {}).get("description")),
            "starting_hp": character.get("startingHp"),
            "starting_gold": character.get("startingGold"),
            "max_energy": character.get("maxEnergy"),
            "base_orb_slot_count": character.get("baseOrbSlotCount"),
            "starting_relic": character.get("startingRelic"),
            "starting_relic_id": character.get("startingRelicId"),
            "starting_deck": character.get("startingDeck", []),
            "unlocks_after_run_as": character.get("unlocksAfterRunAs"),
            "data_source": character.get("dataSource"),
            "alias": build_aliases(character.get("name"), (character_zh or {}).get("name"), character.get("id")),
        }
        for character, character_zh in character_rows
    ]


def normalize_events(event_rows: list[tuple[dict[str, Any], dict[str, Any] | None]]) -> list[dict[str, Any]]:
    return [
        {
            "id": event.get("id"),
            "key": event.get("key"),
            "title": event.get("title"),
            "title_en": event.get("title"),
            "title_zh": (event_zh or {}).get("title"),
            "description": normalize_markup(event.get("description")),
            "description_en": normalize_markup(event.get("description")),
            "description_zh": normalize_markup((event_zh or {}).get("description")),
            "description_plain": normalize_markup(event.get("descriptionPlain")),
            "description_plain_en": normalize_markup(event.get("descriptionPlain")),
            "description_plain_zh": normalize_markup((event_zh or {}).get("descriptionPlain")),
            "options": [
                {
                    "id": option.get("id"),
                    "title": option.get("title"),
                    "title_en": option.get("title"),
                    "title_zh": ((event_zh or {}).get("options", [{}] * len(event.get("options", [])))[index] or {}).get("title"),
                    "description": normalize_markup(option.get("description")),
                    "description_en": normalize_markup(option.get("description")),
                    "description_zh": normalize_markup(
                        ((event_zh or {}).get("options", [{}] * len(event.get("options", [])))[index] or {}).get("description")
                    ),
                }
                for index, option in enumerate(event.get("options", []))
            ],
            "image": event.get("image"),
            "alias": build_aliases(event.get("title"), (event_zh or {}).get("title"), event.get("id"), event.get("key")),
        }
        for event, event_zh in event_rows
    ]


def build_knowledge() -> dict[str, Any]:
    arrays_en = load_page_arrays("en")
    arrays_zh = load_page_arrays("zh")

    merged_cards = merge_localized_rows(arrays_en["cards"], arrays_zh["cards"], id_keys=("id", "localizationKey"))
    merged_relics = merge_localized_rows(arrays_en["relics"], arrays_zh["relics"], id_keys=("id",))
    merged_potions = merge_localized_rows(arrays_en["potions"], arrays_zh["potions"], id_keys=("id",))
    merged_characters = merge_localized_rows(arrays_en["characters"], arrays_zh["characters"], id_keys=("id",))
    merged_events = merge_localized_rows(arrays_en["events"], arrays_zh["events"], id_keys=("id", "key"))
    merged_enemies = merge_localized_rows(arrays_en["enemies"], arrays_zh["enemies"], id_keys=("id",))
    merged_encounters = merge_localized_rows(arrays_en["encounters"], arrays_zh["encounters"], id_keys=("id",))

    arrays = arrays_en
    enriched_enemies = enrich_enemies(merged_enemies, arrays["encounters"])
    enemies_by_id = {enemy["id"]: enemy for enemy in enriched_enemies}
    normalized_encounters = []
    for encounter, encounter_zh in merged_encounters:
        normalized_encounters.append(
            {
                **encounter,
                "nameZh": encounter.get("nameZh") or (encounter_zh or {}).get("nameZh") or (encounter_zh or {}).get("nameEn"),
                "descriptionZh": normalize_markup((encounter_zh or {}).get("description")),
                "alias": build_aliases(
                    encounter.get("nameEn"),
                    encounter.get("nameZh"),
                    (encounter_zh or {}).get("nameZh"),
                    encounter.get("id"),
                ),
            }
        )

    return {
        "metadata": {
            "source_name": "slaythespire2.gg",
            "source_base_url": BASE_URL,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "locales": list(LOCALE_PREFIXES.keys()),
            "notes": [
                "Data is scraped from page payloads exposed by slaythespire2.gg.",
                "Both English and Chinese localized pages are scraped and merged by stable ids when available.",
                "Encounter pools are grouped by act and tier, which is more stable than literal floor-by-floor map paths.",
                "Text markup like [gold] and color tags is normalized into plain text for model consumption.",
            ],
            "counts": {
                "cards": len(arrays_en["cards"]),
                "relics": len(arrays_en["relics"]),
                "potions": len(arrays_en["potions"]),
                "characters": len(arrays_en["characters"]),
                "events": len(arrays_en["events"]),
                "enemies": len(enriched_enemies),
                "encounters": len(arrays_en["encounters"]),
            },
        },
        "acts": build_acts(normalized_encounters, enemies_by_id),
        "encounters": normalized_encounters,
        "enemies": enriched_enemies,
        "cards": normalize_cards(merged_cards),
        "relics": normalize_relics(merged_relics),
        "potions": normalize_potions(merged_potions),
        "characters": normalize_characters(merged_characters),
        "events": normalize_events(merged_events),
    }


def main() -> None:
    knowledge = build_knowledge()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(knowledge, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote knowledge base to: {OUTPUT_PATH}")
    print(json.dumps(knowledge["metadata"]["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
