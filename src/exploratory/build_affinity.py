"""Build the champion-class to item-tag affinity table."""
from __future__ import annotations

import json
import sys

import pandas as pd

try:
    from src.exploratory import paths
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src.exploratory import paths


CLASS_ITEM_TAGS: dict[str, list[str]] = {
    "Marksman": ["Damage", "AttackSpeed", "CriticalStrike", "LifeSteal",
                 "OnHit", "ArmorPenetration"],
    "Fighter":  ["Damage", "AttackSpeed", "Health", "LifeSteal", "OnHit",
                 "ArmorPenetration", "AbilityHaste", "CooldownReduction"],
    "Assassin": ["Damage", "ArmorPenetration", "MagicPenetration", "Lethality",
                 "AbilityHaste", "SpellDamage"],
    "Mage":     ["SpellDamage", "MagicPenetration", "AbilityHaste", "Mana",
                 "ManaRegen", "CooldownReduction"],
    "Tank":     ["Health", "Armor", "MagicResist", "SpellBlock", "HealthRegen",
                 "Tenacity", "AbilityHaste"],
    "Support":  ["Health", "Armor", "SpellBlock", "MagicResist", "GoldPer",
                 "Vision", "ManaRegen", "AbilityHaste", "Aura", "Active"],
}


def rules_table() -> pd.DataFrame:
    rows = [{"champion_class": cls, "item_tag": tag}
            for cls, tags in CLASS_ITEM_TAGS.items() for tag in tags]
    return pd.DataFrame(rows)


def _champion_versions(season: str) -> list[tuple[str, "object"]]:
    """Return (version, path) pairs for each patch's champion.json, oldest first."""
    if season == "2025":
        files = paths.PIPELINE_DDRAGON.glob("15.*.json")
        pairs = [(f.stem, f) for f in files]
    else:
        files = (paths.TASK1_RAW / season).glob("champion_*.json")
        pairs = [(f.stem[len("champion") + 1:], f) for f in files]
    return sorted(pairs, key=lambda t: tuple(int(p) for p in t[0].split(".")))


def _item_tags_present(season: str, version: str) -> set[str]:
    path = paths.TASK1_RAW / season / f"item_{version}.json"
    data = json.loads(path.read_text(encoding="utf-8"))["data"]
    tags: set[str] = set()
    for entry in data.values():
        tags |= set(entry.get("tags", []) or [])
    return tags


def build_affinity(seasons) -> pd.DataFrame:
    rows = []
    for season in seasons:
        item_versions = {paths.oracle_patch_id(v): v
                         for v in _item_versions_on_disk(season)}
        for version, path in _champion_versions(season):
            patch = paths.oracle_patch_id(version)
            champ_data = json.loads(path.read_text(encoding="utf-8"))["data"]
            present = _item_tags_present(season, item_versions[patch]) \
                if patch in item_versions else None
            for name, entry in champ_data.items():
                classes = entry.get("tags", []) or []
                relevant: list[str] = []
                for cls in classes:
                    for tag in CLASS_ITEM_TAGS.get(cls, []):
                        if (present is None or tag in present) and tag not in relevant:
                            relevant.append(tag)
                rows.append({
                    "season": season, "patch": patch,
                    "champion": entry.get("id", name),
                    "champion_classes": ";".join(classes),
                    "relevant_item_tags": ";".join(relevant),
                    "n_relevant_tags": len(relevant),
                })
    return pd.DataFrame(rows)


def _item_versions_on_disk(season: str) -> list[str]:
    files = (paths.TASK1_RAW / season).glob("item_*.json")
    versions = [f.stem[len("item") + 1:] for f in files]
    return sorted(versions, key=lambda s: tuple(int(p) for p in s.split(".")))


def main() -> None:
    seasons = sorted(paths.SEASONS.keys())
    paths.TASK2.mkdir(parents=True, exist_ok=True)

    rules = rules_table()
    affinity = build_affinity(seasons)

    rules.to_csv(paths.TASK2 / "class_item_tag_rules.csv", index=False, encoding="utf-8")
    affinity.to_csv(paths.TASK2 / "champion_class_item_affinity.csv",
                    index=False, encoding="utf-8")

    print("=== Task 2: champion-class item affinity (quarantine) ===")
    print("  class_item_tag_rules.csv          rows=%d classes=%d"
          % (len(rules), rules["champion_class"].nunique()))
    print("  champion_class_item_affinity.csv  rows=%d patches=%d champions=%d"
          % (len(affinity), affinity["patch"].nunique(),
             affinity["champion"].nunique()))
    print("  mean relevant tags/champion: %.2f" % affinity["n_relevant_tags"].mean())


if __name__ == "__main__":
    main()
