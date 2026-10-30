"""Download Data Dragon champion.json for every patch in both seasons.

Usage:
    python -m src.data.fetch_ddragon          # one version per patch
    python -m src.data.fetch_ddragon --all    # every sub-version
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict

import requests

try:
    from src import config
except ModuleNotFoundError:  # allow running as a loose script from src/data/
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config

HEADERS = {"User-Agent": "lol-patch-prediction/0.1 (honours research)"}
DELAY = 0.5  # polite gap between champion.json downloads


def fetch_versions() -> list[str]:
    """Return every Data Dragon version on each season's major line, newest first."""
    resp = requests.get(config.DDRAGON_VERSIONS_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    versions = resp.json()
    majors = set(config.SEASON_MAJORS.values())
    return [v for v in versions if v.split(".")[0] in majors]


def canonical_versions(versions: list[str]) -> list[str]:
    """Keep the lowest sub-version per major.minor (the initial patch release)."""
    by_patch: dict[str, list[str]] = defaultdict(list)
    for v in versions:
        major, minor = v.split(".")[:2]
        by_patch[f"{major}.{minor}"].append(v)

    chosen = []
    for patch, subs in by_patch.items():
        subs_sorted = sorted(subs, key=lambda s: tuple(int(p) for p in s.split(".")))
        chosen.append(subs_sorted[0])
    return sorted(chosen, key=lambda s: tuple(int(p) for p in s.split(".")), reverse=True)


def download_champion_json(version: str) -> dict:
    url = config.DDRAGON_CHAMPION_URL.format(version=version)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true",
        help="download every sub-version on each season's line, not just the "
             "canonical per-patch release",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-download even if the local JSON already exists",
    )
    args = parser.parse_args()

    config.ensure_dirs()

    print("Fetching Data Dragon version list ...")
    all_versions = fetch_versions()
    for season, major in config.SEASON_MAJORS.items():
        n = sum(1 for v in all_versions if v.split(".")[0] == major)
        print(f"  {n} versions on the {major}.x line ({season})")

    targets = all_versions if args.all else canonical_versions(all_versions)
    print(f"Downloading champion.json for {len(targets)} version(s)\n")

    for version in targets:
        out = config.DDRAGON_DIR / f"{version}.json"
        if out.exists() and not args.force:
            print(f"  [-] {version}: cached, skipping")
            continue
        try:
            data = download_champion_json(version)
        except requests.RequestException as exc:
            print(f"  [FAIL] {version}: {exc}")
            continue
        out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        n_champs = len(data.get("data", {}))
        print(f"  [ok] {version}: {n_champs} champions -> {out.name}")
        time.sleep(DELAY)

    print("\nDone.")


if __name__ == "__main__":
    main()
