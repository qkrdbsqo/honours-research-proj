"""Recompute the direction and magnitude columns of patch_note_features.csv with the semantic encoder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src import config
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src import config

from src.data import align
from src.exploratory import semantic_changes as sc
from src.features import patch_note_features as pnf

RECOMPUTED = ["n_buffs", "n_nerfs", "mean_pct_change", "sum_signed_pct", "max_buff_pct",
              "max_nerf_pct", "n_parseable_changes", "is_adjustment", "buff_ratio"]


def features_for_champion(changes: list[dict]) -> dict:
    base = pnf._features_for_champion(changes)
    enc = [sc.encode_line(c.get("ability"), c.get("stat"), c.get("before", ""), c.get("after", ""))
           for c in changes]
    n_buffs = sum(e["sem_direction"] == "buff" for e in enc)
    n_nerfs = sum(e["sem_direction"] == "nerf" for e in enc)
    pcts = [e["signed_power"] for e in enc if not np.isnan(e["signed_power"])]
    buffs, nerfs = [p for p in pcts if p > 0], [p for p in pcts if p < 0]
    n_total = base["n_changes_total"]
    base.update({
        "n_buffs": n_buffs, "n_nerfs": n_nerfs,
        "mean_pct_change": sum(pcts) / len(pcts) if pcts else 0.0,
        "sum_signed_pct": sum(pcts),
        "max_buff_pct": max(buffs) if buffs else 0.0,
        "max_nerf_pct": min(nerfs) if nerfs else 0.0,
        "n_parseable_changes": len(pcts),
        "is_adjustment": int(n_buffs > 0 and n_nerfs > 0),
        "buff_ratio": n_buffs / n_total if n_total else 0.0,
    })
    return base


def build() -> pd.DataFrame:
    """Mirror of patch_note_features.build_patch_note_features, semantic columns."""
    vmap = json.loads(config.VERSION_MAP.read_text(encoding="utf-8"))
    rows = []
    for season in config.SEASONS:
        notes = json.loads((config.PATCH_NOTES_DIR / f"lol_patches_{season}.json").read_text(encoding="utf-8"))
        for entry in notes:
            patch = vmap.get(entry["patch_version"])
            if patch is None:
                continue
            for champ in entry["champions"]:
                rows.append({"season": season,
                             "champion": align.normalise_champion_name(champ["champion"]),
                             "patch": patch, **features_for_champion(champ.get("changes", []))})
    df = pd.DataFrame(rows, columns=["season", "champion", "patch", *pnf.FEATURE_COLS])
    return df.sort_values(["patch", "champion"]).reset_index(drop=True)
