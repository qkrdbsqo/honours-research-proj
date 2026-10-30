"""Download Data Dragon item.json and runesReforged.json for every 2024 and 2025 patch.

Usage:
    python -m src.exploratory.fetch_ddragon_items_runes
    python -m src.exploratory.fetch_ddragon_items_runes --force
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict

import requests

try:
    from src.exploratory import paths
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src.exploratory import paths

HEADERS = {"User-Agent": "lol-patch-prediction/0.1 (honours research)"}
DELAY = 0.5  # polite gap between downloads


def fetch_versions() -> list[str]:
    resp = requests.get(paths.VERSIONS_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def canonical_versions(versions: list[str], major: str) -> list[str]:
    """Lowest sub-version per major.minor on the given major line, oldest first."""
    by_patch: dict[str, list[str]] = defaultdict(list)
    for v in versions:
        parts = v.split(".")
        if len(parts) >= 2 and parts[0] == major and parts[1].isdigit():
            by_patch[f"{parts[0]}.{parts[1]}"].append(v)
    chosen = []
    for _, subs in by_patch.items():
        subs_sorted = sorted(subs, key=lambda s: tuple(int(p) for p in s.split(".")))
        chosen.append(subs_sorted[0])
    return sorted(chosen, key=lambda s: tuple(int(p) for p in s.split(".")))


def download(version: str, kind: str, dest, force: bool) -> str:
    """Download {kind}.json for one version into dest/."""
    out = dest / f"{kind}_{version}.json"
    if out.exists() and not force:
        return "skip"
    url = paths.CDN_TMPL.format(version=version, kind=kind)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    out.write_text(json.dumps(resp.json(), ensure_ascii=False), encoding="utf-8")
    time.sleep(DELAY)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    all_versions = fetch_versions()
    for season, major in paths.SEASONS.items():
        dest = paths.TASK1_RAW / season
        dest.mkdir(parents=True, exist_ok=True)
        versions = canonical_versions(all_versions, major)
        kinds = ["item", "runesReforged"]
        if season == "2024":
            kinds.append("champion")
        print("[%s] major %s.x : %d canonical patches" % (season, major, len(versions)))
        for kind in kinds:
            n_ok = n_skip = 0
            for v in versions:
                status = download(v, kind, dest, args.force)
                n_ok += status == "ok"
                n_skip += status == "skip"
            print("  %-14s downloaded=%d skipped=%d" % (kind, n_ok, n_skip))
    print("done. raw jsons under: %s" % paths.TASK1_RAW)


if __name__ == "__main__":
    main()
