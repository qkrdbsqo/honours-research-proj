"""Scrape champion changes from the 2025 Riot patch notes."""

import requests
import json
import csv
import time
import re
from bs4 import BeautifulSoup, Tag
from dataclasses import dataclass, field, asdict
from typing import Optional


BASE_URL = "https://www.leagueoflegends.com/en-us/news/game-updates/patch-{patch}-notes/"

PATCH_SLUGS = (
    ["25-s1-1", "25-s1-2", "2025-s1-3"]          # patches 25.S1.1–25.S1.3 (note: S1.3 uses full year in URL)
    + [f"25-{str(i).zfill(2)}" for i in range(4, 25)]  # patches 25.04–25.24
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
DELAY = 2  # seconds between requests

EXCLUDED_SECTIONS = {
    "aram", "arena", "brawl", "swiftplay", "tft", "teamfight tactics",
    "items", "runes", "systems", "system", "bugfixes", "bug fixes",
    "upcoming skins", "skin", "skins", "patch highlights",
    "esports", "client", "battle pass", "ranked",
    "modes", "objectives", "minions", "monsters", "jungle changes",
    "summoner spells", "early surrender",
}

CHAMPION_SECTION_HEADERS = {"champions", "champion"}


@dataclass
class IndividualChange:
    """One change to one champion stat."""
    champion: str
    ability: str          # "Passive", "Q", "W", "E", "R", "Stats", or "Other"
    stat: str             # e.g. "Damage", "Cooldown", "Mana Cost"
    before: str
    after: str
    direction: str        # "buff", "nerf", "neutral"
    raw_line: str

@dataclass
class ChampionEntry:
    """All changes to one champion in one patch."""
    champion: str
    overall_direction: str    # "buff", "nerf", "adjustment", "rework"
    changes: list = field(default_factory=list)

@dataclass
class PatchData:
    patch_version: str
    url: str
    release_date: str
    champion_entries: list = field(default_factory=list)
    scrape_success: bool = False
    error: Optional[str] = None


def normalise_version(slug: str) -> str:
    """Convert a patch-note URL slug to Riot's public patch label, e.g. '25-04' to '25.04'."""
    parts = slug.split("-")
    year = parts[0][2:] if parts[0].startswith("20") and len(parts[0]) == 4 else parts[0]
    if len(parts) == 3 and parts[1].lower() == "s1":
        return f"{year}.S1.{parts[2]}"
    return f"{year}.{parts[1]}"


def is_excluded_section(text: str) -> bool:
    """Check if a section header refers to a non-Champion / non-SR section."""
    t = text.strip().lower()
    if t in EXCLUDED_SECTIONS:
        return True
    return any(t.startswith(s) or s in t for s in EXCLUDED_SECTIONS)


def is_champion_section(text: str) -> bool:
    return text.strip().lower() in CHAMPION_SECTION_HEADERS


def parse_ability(text: str) -> str:
    """Pull out which ability the change belongs to from a header or label."""
    t = text.lower().strip()
    if "passive" in t:
        return "Passive"
    if re.search(r'\b\(?q\)?\b', t) or t.startswith("q "):
        return "Q"
    if re.search(r'\b\(?w\)?\b', t) or t.startswith("w "):
        return "W"
    if re.search(r'\b\(?e\)?\b', t) or t.startswith("e "):
        return "E"
    if re.search(r'\b\(?r\)?\b', t) or "ultimate" in t or " ult" in t:
        return "R"
    if "base stat" in t or "stats" in t:
        return "Stats"
    return "Other"


def parse_change_line(line: str) -> Optional[dict]:
    """Parse a single '⇒'-style change line into structured fields."""
    line = line.strip().lstrip("•-– ").strip()
    if "⇒" not in line and "→" not in line:
        return None

    arrow = "⇒" if "⇒" in line else "→"

    if ":" in line:
        stat, values = line.split(":", 1)
        stat = stat.strip()
    else:
        stat = "Change"
        values = line

    if arrow not in values:
        return None
    before, after = [v.strip() for v in values.split(arrow, 1)]

    direction = "neutral"
    nums_before = re.findall(r'[\d.]+', before)
    nums_after  = re.findall(r'[\d.]+', after)
    if nums_before and nums_after:
        try:
            b = float(nums_before[0])
            a = float(nums_after[0])
            if a > b:
                direction = "buff"
            elif a < b:
                direction = "nerf"
        except ValueError:
            pass

    return {
        "stat":      stat[:60],          # truncate very long labels
        "before":    before[:80],
        "after":     after[:80],
        "direction": direction,
        "raw_line":  line[:200]
    }


def overall_direction(changes: list) -> str:
    """Aggregate change direction for a whole champion."""
    if not changes:
        return "unknown"
    buffs = sum(1 for c in changes if c.direction == "buff")
    nerfs = sum(1 for c in changes if c.direction == "nerf")
    if buffs > nerfs * 1.5:
        return "buff"
    if nerfs > buffs * 1.5:
        return "nerf"
    return "adjustment"


def find_main_content(soup: BeautifulSoup) -> Optional[Tag]:
    """Locate the article body (Riot wraps it in <main> or a content div)."""
    for selector in ["main", "article", "div[class*='article']", "div[class*='content']"]:
        el = soup.select_one(selector)
        if el and len(el.get_text(strip=True)) > 500:
            return el
    return soup.body or soup


def extract_champion_entries(soup: BeautifulSoup) -> list:
    """Walk the page in order and collect champion changes by section."""
    main = find_main_content(soup)
    if not main:
        return []

    in_champion_section = False
    current_champion: Optional[str] = None
    current_ability  = "Other"
    champion_buffer: dict = {}   # champion_name → list[IndividualChange]

    for el in main.descendants:
        if not isinstance(el, Tag):
            continue

        if el.name == "h2":
            text = el.get_text(strip=True)
            if is_champion_section(text):
                in_champion_section = True
                current_champion = None
            elif is_excluded_section(text):
                in_champion_section = False
                current_champion = None
            else:
                in_champion_section = False
                current_champion = None
            continue

        if not in_champion_section:
            continue

        if el.name == "h3":
            text = el.get_text(strip=True)
            if not text or len(text) > 30 or re.search(r'[\d#\[\]{}|/\\]', text):
                continue
            if is_excluded_section(text):
                current_champion = None
                continue
            current_champion = text
            current_ability  = "Other"
            champion_buffer.setdefault(current_champion, [])
            continue

        if el.name == "h4" and current_champion:
            current_ability = parse_ability(el.get_text(strip=True))
            continue

        if el.name in ("li", "p") and current_champion:
            text = el.get_text(separator=" ", strip=True)
            if not text or ("⇒" not in text and "→" not in text):
                continue

            ability_for_line = current_ability
            if ability_for_line == "Other":
                inferred = parse_ability(text[:30])
                if inferred != "Other":
                    ability_for_line = inferred

            parsed = parse_change_line(text)
            if not parsed:
                continue

            champion_buffer[current_champion].append(IndividualChange(
                champion=current_champion,
                ability=ability_for_line,
                stat=parsed["stat"],
                before=parsed["before"],
                after=parsed["after"],
                direction=parsed["direction"],
                raw_line=parsed["raw_line"]
            ))

    entries = []
    for champ, changes in champion_buffer.items():
        if not changes:
            continue
        entries.append(ChampionEntry(
            champion=champ,
            overall_direction=overall_direction(changes),
            changes=changes
        ))
    return entries


def scrape_patch(slug: str) -> PatchData:
    version = normalise_version(slug)
    url = BASE_URL.format(patch=slug)

    patch = PatchData(patch_version=version, url=url, release_date="unknown")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        patch.error = f"HTTP {resp.status_code}"
        print(f"  ✗ {version}: {patch.error}")
        return patch
    except requests.exceptions.RequestException as e:
        patch.error = str(e)
        print(f"  ✗ {version}: {patch.error}")
        return patch

    soup = BeautifulSoup(resp.text, "html.parser")

    time_tag = soup.find("time")
    if time_tag:
        patch.release_date = time_tag.get("datetime", "unknown")

    patch.champion_entries = extract_champion_entries(soup)
    patch.scrape_success   = True

    total_changes = sum(len(c.changes) for c in patch.champion_entries)
    buffs   = sum(1 for c in patch.champion_entries if c.overall_direction == "buff")
    nerfs   = sum(1 for c in patch.champion_entries if c.overall_direction == "nerf")
    adjusts = sum(1 for c in patch.champion_entries if c.overall_direction == "adjustment")
    print(f"  ✓ {version}: {len(patch.champion_entries)} champions, "
          f"{total_changes} changes ({buffs}↑ {nerfs}↓ {adjusts}~)")
    return patch


def save_json(patches: list, path: str):
    data = []
    for p in patches:
        data.append({
            "patch_version":  p.patch_version,
            "url":            p.url,
            "release_date":   p.release_date,
            "scrape_success": p.scrape_success,
            "error":          p.error,
            "champions": [
                {
                    "champion":          e.champion,
                    "overall_direction": e.overall_direction,
                    "changes":           [asdict(c) for c in e.changes]
                }
                for e in p.champion_entries
            ]
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON saved → {path}")


def save_csv(patches: list, path: str):
    """Granular CSV: one row per individual change."""
    rows = []
    for p in patches:
        if not p.scrape_success:
            continue
        for entry in p.champion_entries:
            for change in entry.changes:
                rows.append({
                    "patch_version":     p.patch_version,
                    "release_date":      p.release_date,
                    "champion":          entry.champion,
                    "overall_direction": entry.overall_direction,
                    "ability":           change.ability,
                    "stat":              change.stat,
                    "before":            change.before,
                    "after":             change.after,
                    "direction":         change.direction,
                    "raw_line":          change.raw_line,
                })

    if not rows:
        print("No data to write to CSV.")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"✓ CSV saved  → {path}  ({len(rows)} rows)")


def print_summary(patches: list):
    print("\n" + "="*60)
    print("SCRAPE SUMMARY — LoL 2025 Patches")
    print("="*60)
    ok = [p for p in patches if p.scrape_success]
    bad = [p for p in patches if not p.scrape_success]
    print(f"  Patches scraped:  {len(ok)}/{len(patches)}")
    if bad:
        print(f"  Failed patches:   {[p.patch_version for p in bad]}")

    total_champs  = sum(len(p.champion_entries) for p in ok)
    total_changes = sum(sum(len(c.changes) for c in p.champion_entries) for p in ok)
    print(f"  Total champion-patch entries: {total_champs}")
    print(f"  Total individual changes:     {total_changes}")
    print("="*60)


def main():
    print("LoL 2025 Patch Notes Scraper")
    print(f"Targeting {len(PATCH_SLUGS)} patches: 25.S1.1 → 25.24\n")

    all_patches = []
    for slug in PATCH_SLUGS:
        all_patches.append(scrape_patch(slug))
        time.sleep(DELAY)

    save_json(all_patches, "lol_patches_2025.json")
    save_csv(all_patches,  "lol_patches_2025.csv")
    print_summary(all_patches)


if __name__ == "__main__":
    main()
