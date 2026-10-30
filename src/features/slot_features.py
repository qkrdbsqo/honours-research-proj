"""Draft-slot features: changes to the champions that compete for the same positions.

Usage:
    python -m src.features.slot_features
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    from src import config
    from src.data import align, load_oracle
    from src.features import item_features as itf
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.data import align, load_oracle
    from src.features import item_features as itf

PLAYING_POSITIONS = ("top", "jng", "mid", "bot", "sup")

_CHANGE_COLS = ["n_buffs", "n_nerfs", "n_changes_total", "sum_signed_pct"]


def position_shares() -> pd.DataFrame:
    """(patch, position, champion) -> picks and within-position pick share."""
    raw = load_oracle.load_raw(usecols=["gameid", "patch", "position", "champion"])
    raw = raw[raw["position"].isin(PLAYING_POSITIONS)].copy()
    raw["champion"] = raw["champion"].map(align.normalise_champion_name)
    g = (raw.groupby(["patch", "position", "champion"])
            .size().rename("picks").reset_index())
    g["share"] = g["picks"] / g.groupby(["patch", "position"])["picks"].transform("sum")
    return g


def _own_position_weights(shares_prev: pd.DataFrame) -> dict[str, dict[str, float]]:
    """champion -> {position: fraction of that champion's P-1 games}."""
    tot = shares_prev.groupby("champion")["picks"].transform("sum")
    w = shares_prev.assign(w=shares_prev["picks"] / tot)
    out: dict[str, dict[str, float]] = {}
    for champ, pos, weight in zip(w["champion"], w["position"], w["w"]):
        out.setdefault(champ, {})[pos] = float(weight)
    return out


def build_slot_features(aligned: pd.DataFrame | None = None,
                        shares: pd.DataFrame | None = None,
                        notes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features for requested rows, independent of rivals' current pick counts."""
    if shares is None:
        shares = position_shares()
    if aligned is None:
        aligned = pd.read_csv(config.PWDS_CSV, dtype={"patch": str},
                              usecols=["champion", "patch"])
    if notes is None:
        notes = pd.read_csv(config.PATCH_NOTE_FEATURES_CSV, dtype={"patch": str})
    notes = notes.copy()
    notes["champion"] = notes["champion"].map(align.normalise_champion_name)
    if notes.duplicated(["champion", "patch"]).any():
        raise ValueError("duplicate champion-patch keys in patch-note features")
    changes = notes.set_index(["champion", "patch"])[_CHANGE_COLS].fillna(0.0)
    prev = itf._prev_patch_map()

    rows = []
    for patch in sorted(aligned["patch"].unique()):
        champs_here = aligned.loc[aligned["patch"] == patch, "champion"].unique()
        q = prev.get(patch)
        if q is None:
            for champ in champs_here:
                rows.append(_zero_row(champ, patch))
            continue

        prev_shares = shares[shares["patch"] == q]
        if prev_shares.empty:
            for champ in champs_here:
                rows.append(_zero_row(champ, patch))
            continue

        own_w = _own_position_weights(prev_shares)
        by_pos = {p: prev_shares[prev_shares["position"] == p]
                            .set_index("champion")["share"]
                  for p in prev_shares["position"].unique()}

        for champ in champs_here:
            if champ not in own_w:
                rows.append(_zero_row(champ, patch))
                continue
            acc = dict.fromkeys(
                ["slot_n_changed", "slot_n_buffs", "slot_n_nerfs",
                 "slot_net_signed_pct"], 0.0)
            max_share = 0.0
            for pos, weight in own_w[champ].items():
                rivals = by_pos[pos].drop(index=champ, errors="ignore")
                total = float(rivals.sum())
                if total <= 0:
                    continue
                rivals = rivals / total          # renormalise after self-drop
                for rival, share in rivals.items():
                    key = (rival, patch)
                    if key not in changes.index:
                        continue
                    r = changes.loc[key]
                    w = weight * float(share)
                    changed = 1.0 if float(r["n_changes_total"]) > 0 else 0.0
                    acc["slot_n_changed"] += w * changed
                    acc["slot_n_buffs"] += w * float(r["n_buffs"])
                    acc["slot_n_nerfs"] += w * float(r["n_nerfs"])
                    acc["slot_net_signed_pct"] += w * float(r["sum_signed_pct"])
                    if changed:
                        max_share = max(max_share, float(share))
            rows.append({"champion": champ, "patch": patch,
                         **acc,
                         "slot_max_rival_share_changed": max_share,
                         "has_slot_lag": 1})

    return pd.DataFrame(rows, columns=["champion", "patch"] + config.SLOT_FEATURE_COLS)


def _zero_row(champion: str, patch: str) -> dict:
    """No P-1 information: every channel is a real zero, flagged by has_slot_lag."""
    row = {"champion": champion, "patch": patch}
    for col in config.SLOT_FEATURE_COLS:
        row[col] = 0.0
    row["has_slot_lag"] = 0
    return row


def main() -> None:
    config.ensure_dirs()
    df = build_slot_features()
    df.to_csv(config.SLOT_FEATURES_CSV, index=False)
    te = df[df["patch"].isin(config.TEST_PATCHES)]
    nz = float((te[["slot_n_changed", "slot_n_buffs", "slot_n_nerfs"]]
                .abs().sum(axis=1) > 0).mean())
    print(f"slot_features: {len(df):,} rows x {df.shape[1]} cols "
          f"-> {config.SLOT_FEATURES_CSV}")
    print(f"  has_slot_lag: {100 * df['has_slot_lag'].mean():.1f}% of all rows")
    print(f"  test rows with non-zero slot pressure: {100 * nz:.1f}%")
    counts = te.groupby("patch")["slot_n_changed"].nunique()
    print(f"  distinct slot_n_changed values per test patch: {counts.to_dict()}")


if __name__ == "__main__":
    main()
