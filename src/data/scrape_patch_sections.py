"""Scrape the Items, Runes and Systems sections of the patch notes.

Usage:
    python -m src.data.scrape_patch_sections --season 2025
    python -m src.data.scrape_patch_sections --all
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from src import config
    from src.data import scrape_patch_notes as s25
    from src.data import scrape_patch_notes_2024 as s24
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.data import scrape_patch_notes as s25
    from src.data import scrape_patch_notes_2024 as s24

WANTED_SECTIONS = {"items", "runes", "systems", "system"}

OTHER_MODES = {"aram", "arena", "brawl", "swiftplay", "tft",
               "teamfight tactics"}


def season_sources(season: str) -> tuple[list[str], dict[str, str], str]:
    """(slugs, url_overrides, base_url_template) for a season."""
    if season == "2024":
        return list(s24.PATCH_SLUGS), dict(s24.URL_OVERRIDES), s24.BASE_URL
    return list(s25.PATCH_SLUGS), {}, s25.BASE_URL


def section_kind(header: str) -> str | None:
    """Canonical section name, or None if this header is not wanted."""
    t = header.strip().lower()
    if any(m in t for m in OTHER_MODES):
        return None
    for want in ("items", "runes", "systems"):
        if t == want or t.startswith(want) or want in t:
            return want
    if t == "system":
        return "systems"
    return None


def extract_sections(soup: BeautifulSoup) -> list[dict]:
    """Walk the page and pull every wanted section's entries."""
    main = s25.find_main_content(soup)
    if main is None:
        return []

    sections: list[dict] = []
    current: dict | None = None
    current_entry: dict | None = None

    for node in main.find_all(["h2", "h3", "h4", "li", "p"]):
        name = node.get_text(" ", strip=True)
        if not name:
            continue

        if node.name == "h2":
            kind = section_kind(name)
            current = ({"section": kind, "header": name, "entries": []}
                       if kind else None)
            current_entry = None
            if current:
                sections.append(current)
            continue

        if current is None:
            continue

        if node.name in ("h3", "h4"):
            current_entry = {"element_name": name, "raw_text": "",
                             "changes": [], "parsed": False}
            current["entries"].append(current_entry)
            continue

        if current_entry is None:
            current_entry = {"element_name": current["section"],
                             "raw_text": "", "changes": [], "parsed": False}
            current["entries"].append(current_entry)

        current_entry["raw_text"] = (
            current_entry["raw_text"] + " " + name).strip()
        parsed = s25.parse_change_line(name)
        if parsed:
            current_entry["changes"].append(parsed)
            current_entry["parsed"] = True

    for sec in sections:
        sec["entries"] = [e for e in sec["entries"]
                           if e["raw_text"] or e["changes"]]

    return sections


def scrape_season(season: str) -> list[dict]:
    slugs, overrides, base = season_sources(season)
    out: list[dict] = []
    for slug in slugs:
        url = overrides.get(slug) or base.format(
            **({"slug": slug} if season == "2024" else {"patch": slug}))
        try:
            resp = requests.get(url, headers=s25.HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  [fail] {slug}: {exc}")
            out.append({"slug": slug, "url": url, "sections": [],
                        "error": str(exc)})
            time.sleep(s25.DELAY)
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        sections = extract_sections(soup)
        n_entries = sum(len(s["entries"]) for s in sections)
        n_parsed = sum(1 for s in sections for e in s["entries"] if e["parsed"])
        print(f"  {slug}: {len(sections)} sections, {n_entries} entries, "
              f"{n_parsed} parsed")
        out.append({"slug": slug, "url": url, "sections": sections})
        time.sleep(s25.DELAY)
    return out


def _patch_for_slug(season: str, slug: str) -> str:
    """Slug -> client patch, via the same normaliser the scrapers use."""
    version = (s24.normalise_version(slug) if season == "2024"
               else s25.normalise_version(slug))
    vmap = json.loads(config.VERSION_MAP.read_text(encoding="utf-8"))
    return vmap.get(version, version)


def coverage_report(season: str, pages: list[dict]) -> pd.DataFrame:
    """One row per (season, patch, section)."""
    rows = []
    for page in pages:
        patch = _patch_for_slug(season, page["slug"])
        for sec in page.get("sections", []):
            entries = sec["entries"]
            rows.append({
                "season": season, "patch": patch, "section": sec["section"],
                "entries": len(entries),
                "parsed": sum(1 for e in entries if e["parsed"]),
            })
    cols = ["season", "patch", "section", "entries", "parsed"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df.assign(parse_rate=pd.Series(dtype=float))
    agg = df.groupby(["season", "patch", "section"], as_index=False)[
        ["entries", "parsed"]].sum()
    agg["parse_rate"] = agg.apply(
        lambda r: (r["parsed"] / r["entries"]) if r["entries"] else 0.0,
        axis=1)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", choices=config.SEASONS)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    seasons = config.SEASONS if args.all else [args.season]
    if not seasons or seasons == [None]:
        ap.error("pass --season <year> or --all")

    config.ensure_dirs()
    reports = []
    for season in seasons:
        print(f"\n== scraping {season} sections ==")
        pages = scrape_season(season)
        for page in pages:
            page["patch"] = _patch_for_slug(season, page["slug"])
        path = config.PATCH_SECTIONS_JSON[season]
        path.write_text(json.dumps(pages, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"  -> {path}")
        reports.append(coverage_report(season, pages))

    rep = pd.concat(reports, ignore_index=True)
    rep.to_csv(config.SECTION_COVERAGE_CSV, index=False)
    print(f"\n== coverage ==")
    for section, grp in rep.groupby("section"):
        print(f"  {section:<10} entries={int(grp['entries'].sum()):<5} "
              f"parsed={int(grp['parsed'].sum()):<5} "
              f"rate={grp['parsed'].sum() / max(grp['entries'].sum(), 1):.1%}")
    print(f"  -> {config.SECTION_COVERAGE_CSV}")


if __name__ == "__main__":
    main()
