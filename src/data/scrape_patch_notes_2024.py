"""Scrape champion changes from the 2024 Riot patch notes, in the same schema as the 2025 scraper.

Usage:
    python -m src.data.scrape_patch_notes_2024
"""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict

import requests
from bs4 import BeautifulSoup

try:
    from src import config
    from src.data.scrape_patch_notes import (
        HEADERS, DELAY, PatchData, extract_champion_entries,
    )
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.data.scrape_patch_notes import (
        HEADERS, DELAY, PatchData, extract_champion_entries,
    )

BASE_URL = "https://www.leagueoflegends.com/en-us/news/game-updates/patch-{slug}-notes/"

PATCH_SLUGS = [f"14-{i}" for i in range(1, 25)]

URL_OVERRIDES = {
    "14-13": "https://www.leagueoflegends.com/en-us/news/game-updates/lol-patch-14-13-notes/",
}

OUT_JSON = config.PATCH_NOTES_DIR / "lol_patches_2024.json"
OUT_CSV = config.PATCH_NOTES_DIR / "lol_patches_2024.csv"


def normalise_version(slug: str) -> str:
    """'14-1' -> '14.1', '14-10' -> '14.10' (Riot's 2024 public patch labels)."""
    year, minor = slug.split("-")
    return f"{year}.{minor}"


def scrape_patch(slug: str) -> PatchData:
    version = normalise_version(slug)
    url = URL_OVERRIDES.get(slug, BASE_URL.format(slug=slug))
    patch = PatchData(patch_version=version, url=url, release_date="unknown")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        patch.error = f"HTTP {resp.status_code}"
        print(f"  x {version}: {patch.error}")
        return patch
    except requests.exceptions.RequestException as e:
        patch.error = str(e)
        print(f"  x {version}: {patch.error}")
        return patch

    soup = BeautifulSoup(resp.text, "html.parser")
    time_tag = soup.find("time")
    if time_tag:
        patch.release_date = time_tag.get("datetime", "unknown")

    patch.champion_entries = extract_champion_entries(soup)
    patch.scrape_success = True

    total_changes = sum(len(c.changes) for c in patch.champion_entries)
    buffs = sum(1 for c in patch.champion_entries if c.overall_direction == "buff")
    nerfs = sum(1 for c in patch.champion_entries if c.overall_direction == "nerf")
    adjs = sum(1 for c in patch.champion_entries if c.overall_direction == "adjustment")
    print(f"  ok {version}: {len(patch.champion_entries)} champions, "
          f"{total_changes} changes ({buffs} up, {nerfs} down, {adjs} adj)")
    return patch


def save_json(patches: list, path) -> None:
    data = []
    for p in patches:
        data.append({
            "patch_version": p.patch_version,
            "url": p.url,
            "release_date": p.release_date,
            "scrape_success": p.scrape_success,
            "error": p.error,
            "champions": [
                {
                    "champion": e.champion,
                    "overall_direction": e.overall_direction,
                    "changes": [asdict(c) for c in e.changes],
                }
                for e in p.champion_entries
            ],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved -> {path}")


def save_csv(patches: list, path) -> None:
    """Granular CSV: one row per individual change (matches 2025 columns)."""
    rows = []
    for p in patches:
        if not p.scrape_success:
            continue
        for entry in p.champion_entries:
            for change in entry.changes:
                rows.append({
                    "patch_version": p.patch_version,
                    "release_date": p.release_date,
                    "champion": entry.champion,
                    "overall_direction": entry.overall_direction,
                    "ability": change.ability,
                    "stat": change.stat,
                    "before": change.before,
                    "after": change.after,
                    "direction": change.direction,
                    "raw_line": change.raw_line,
                })
    if not rows:
        print("No data to write to CSV.")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved  -> {path}  ({len(rows)} rows)")


def print_summary(patches: list) -> None:
    print("\n" + "=" * 60)
    print("SCRAPE SUMMARY - LoL 2024 (Season 14) Patches")
    print("=" * 60)
    ok = [p for p in patches if p.scrape_success]
    bad = [p for p in patches if not p.scrape_success]
    print(f"  Patches scraped:  {len(ok)}/{len(patches)}")
    if bad:
        print(f"  Failed patches:   {[p.patch_version for p in bad]}")
    total_champs = sum(len(p.champion_entries) for p in ok)
    total_changes = sum(sum(len(c.changes) for c in p.champion_entries) for p in ok)
    print(f"  Total champion-patch entries: {total_champs}")
    print(f"  Total individual changes:     {total_changes}")
    print("=" * 60)


def main() -> None:
    config.ensure_dirs()
    print("LoL 2024 Patch Notes Scraper")
    print(f"Targeting {len(PATCH_SLUGS)} patches: 14.1 -> 14.24\n")

    all_patches = []
    for slug in PATCH_SLUGS:
        all_patches.append(scrape_patch(slug))
        time.sleep(DELAY)

    save_json(all_patches, OUT_JSON)
    save_csv(all_patches, OUT_CSV)
    print_summary(all_patches)


if __name__ == "__main__":
    main()
