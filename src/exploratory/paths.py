"""Paths for the item, rune and build-usage inputs."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXPLORATORY = ROOT / "data" / "builds"

TASK1 = EXPLORATORY / "items_runes"
TASK1_RAW = TASK1 / "raw" / "ddragon"          # per-season subfolders: 2024/, 2025/
TASK2 = EXPLORATORY / "class_affinity"

PIPELINE_DDRAGON = ROOT / "data" / "raw" / "ddragon"

CDN_TMPL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/{kind}.json"
VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"

SEASONS = {"2024": "14", "2025": "15"}


def oracle_patch_id(version: str) -> str:
    """Map a Data Dragon version to Oracle/DDragon client patch id."""
    major, minor = version.split(".")[:2]
    return f"{major}.{int(minor):02d}"
