"""Per-champion base-stat changes between consecutive patches.

Usage:
    python -m src.features.stat_deltas
"""
from __future__ import annotations

import json
import sys

import pandas as pd

try:
    from src import config
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config

STAT_KEYS = [
    "hp", "hpperlevel", "mp", "mpperlevel", "movespeed",
    "armor", "armorperlevel", "spellblock", "spellblockperlevel", "attackrange",
    "hpregen", "hpregenperlevel", "mpregen", "mpregenperlevel", "crit",
    "critperlevel", "attackdamage", "attackdamageperlevel", "attackspeedperlevel",
    "attackspeed",
]
DELTA_COLS = [f"{k}_delta" for k in STAT_KEYS]


def file_to_patch(stem: str) -> str:
    """Map a Data Dragon filename stem to the canonical client patch label."""
    major, minor = stem.split(".")[:2]
    return f"{int(major)}.{int(minor):02d}"


def load_patch_stats() -> dict[str, dict[str, dict[str, float]]]:
    """Return ``{patch_label: {champion_id: {stat: value}}}`` for every snapshot across every season."""
    by_patch: dict[str, dict[str, dict[str, float]]] = {}
    for major in config.SEASON_MAJORS.values():
        for path in config.DDRAGON_DIR.glob(f"{major}.*.json"):
            patch = file_to_patch(path.stem)
            data = json.loads(path.read_text(encoding="utf-8"))["data"]
            by_patch[patch] = {cid: meta["stats"] for cid, meta in data.items()}
    return by_patch


def _sorted_patches(patches) -> list[str]:
    return sorted(patches, key=lambda s: tuple(int(p) for p in s.split(".")))


def compute_deltas() -> pd.DataFrame:
    """Build the per-(champion, patch) stat-delta matrix."""
    by_patch = load_patch_stats()
    if not by_patch:
        raise FileNotFoundError(
            "No Data Dragon JSON found. Run `python -m src.data.fetch_ddragon` first."
        )
    id_to_name = json.loads(config.CHAMPION_NAMES.read_text(encoding="utf-8"))

    patches = _sorted_patches(by_patch.keys())
    rows: list[dict] = []
    for prev, curr in zip(patches, patches[1:]):
        if prev.split(".")[0] != curr.split(".")[0]:
            print(f"  [break] skipping delta {prev} -> {curr} "
                  "(season boundary)")
            continue
        prev_stats, curr_stats = by_patch[prev], by_patch[curr]
        for cid, stats in curr_stats.items():
            if cid not in prev_stats:
                continue  # new champion, nothing to diff against
            prev_s = prev_stats[cid]
            deltas = {k: float(stats.get(k, 0)) - float(prev_s.get(k, 0))
                      for k in STAT_KEYS}
            n_changed = sum(1 for v in deltas.values() if v != 0)
            n_up = sum(1 for v in deltas.values() if v > 0)
            n_down = sum(1 for v in deltas.values() if v < 0)
            net = (n_up > n_down) - (n_up < n_down)  # -> +1 / 0 / -1
            row = {
                "season": config.season_of(curr),
                "champion": id_to_name.get(cid, cid),
                "patch": curr,
                **{f"{k}_delta": deltas[k] for k in STAT_KEYS},
                "n_stats_changed": n_changed,
                "net_direction": net,
            }
            rows.append(row)

    df = pd.DataFrame(rows, columns=["season", "champion", "patch", *DELTA_COLS,
                                     "n_stats_changed", "net_direction"])

    unmapped = sorted({r["champion"] for r in rows
                       if r["champion"] not in id_to_name.values()})
    if unmapped:
        print(f"  [warn] {len(unmapped)} champion ids had no display name: "
              f"{unmapped[:5]}")

    return df.sort_values(["patch", "champion"]).reset_index(drop=True)


def main() -> None:
    config.ensure_dirs()
    df = compute_deltas()
    df.to_csv(config.STAT_DELTAS_CSV, index=False)

    changed = df[df["n_stats_changed"] > 0]
    print(f"stat_deltas: {len(df):,} champion-patch rows "
          f"({df['patch'].nunique()} patches with a predecessor)")
    print(f"  rows with >=1 base-stat change: {len(changed):,} "
          f"({len(changed) / len(df):.1%})")
    print(f"  buffs (net +1): {(df['net_direction'] == 1).sum():,} | "
          f"nerfs (net -1): {(df['net_direction'] == -1).sum():,}")
    print(f"  -> {config.STAT_DELTAS_CSV}")


if __name__ == "__main__":
    main()
