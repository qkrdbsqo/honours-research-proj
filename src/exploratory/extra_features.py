"""The 19 extra features used by C1X and the CL control."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src import config
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src import config

KEYS = ["champion", "patch"]
EXTRA_COLS = ["lag_pr", "lag_br", "lag_wr", "lag_picks", "lag_total_games",
              "lag_n_teams", "lag_top_team_share", "lag_hhi", "lag_n_leagues",
              "lag_position_switched", "prior_pro_freq", "lag2_pwds", "has_lag2",
              "release_avail", "sem_power_ult", "sem_power_basic", "sem_power_passive",
              "sem_power_stats", "season_2025"]


NO_PATCH_EXTRAS = ["lag_pr", "lag_br", "lag_wr", "lag_picks", "lag_total_games",
                   "lag_n_teams", "lag_top_team_share", "lag_hhi", "lag_n_leagues",
                   "lag_position_switched", "prior_pro_freq", "lag2_pwds", "has_lag2", "season_2025"]
CONTROL_FEATURES = ["lag_pwds_presence_heavy", "has_lag"] + NO_PATCH_EXTRAS


def _prev_map(k: int = 1) -> dict[str, str]:
    """patch -> the patch k steps earlier in ALL_PATCHES, same season only."""
    ps = config.ALL_PATCHES
    return {p: ps[i - k] for i, p in enumerate(ps)
            if i >= k and config.season_of(ps[i - k]) == config.season_of(p)}


def _shift(df: pd.DataFrame, cols: list[str], k: int, prefix: str) -> pd.DataFrame:
    """Values of ``cols`` from k patches earlier, keyed by the later patch."""
    inv = {v: p for p, v in _prev_map(k).items()}
    s = df[KEYS + cols].copy()
    s["patch"] = s["patch"].map(inv)
    s = s.dropna(subset=["patch"])
    return s.rename(columns={c: prefix + c for c in cols})


def build(aligned: pd.DataFrame | None = None) -> pd.DataFrame:
    """Extras for the scored (champion, patch) rows of the aligned dataset."""
    d = pd.read_csv(config.ALIGNED_DATASET_CSV, dtype={"patch": str}) if aligned is None else aligned
    return _assemble(d, d[KEYS + ["season"]].copy())


def build_full(aligned: pd.DataFrame | None = None) -> pd.DataFrame:
    """The same extras for every (champion, patch) key that any component covers."""
    d = pd.read_csv(config.ALIGNED_DATASET_CSV, dtype={"patch": str}) if aligned is None else aligned
    parts = _components(d)
    keys = pd.concat([part[KEYS] for part in parts], ignore_index=True).drop_duplicates()
    keys = keys[keys["patch"].isin(config.ALL_PATCHES)].copy()
    keys["season"] = keys["patch"].map(config.season_of)
    return _assemble(d, keys.reset_index(drop=True), parts)


def _components(d: pd.DataFrame) -> list[pd.DataFrame]:
    from src.evaluation import emergence as em
    from src.evaluation import tier_anatomy as ta
    from src.exploratory import release_avail as ra
    from src.exploratory import semantic_changes as sc

    a = _shift(d, ["pr", "br", "wr", "picks", "total_games"], 1, "lag_")
    c = ta.concentration().rename(columns={"n_teams": "n_teams", "top_team_share": "top_team_share"})
    c = _shift(c, ["n_teams", "top_team_share", "hhi", "n_leagues"], 1, "lag_")
    pos = ta.main_position()
    p1 = _shift(pos, ["position"], 1, "p1_")
    p2 = _shift(pos, ["position"], 2, "p2_")
    sw = p1.merge(p2, on=KEYS, how="inner")
    sw["lag_position_switched"] = (sw["p1_position"] != sw["p2_position"]).astype(int)
    l2 = _shift(d, ["pwds_presence_heavy"], 2, "lag2_").rename(
        columns={"lag2_pwds_presence_heavy": "lag2_pwds"})
    l2["has_lag2"] = 1
    pf = em.build_cells()[KEYS + ["prior_freq"]].rename(columns={"prior_freq": "prior_pro_freq"})
    rel = ra.avail_flags()[KEYS + [ra.FLAG]].rename(columns={ra.FLAG: "release_avail"})
    sem = sc.features()[KEYS + ["sem_power_ult", "sem_power_basic", "sem_power_passive", "sem_power_stats"]]
    return [a, c, sw[KEYS + ["lag_position_switched"]], l2, pf, rel, sem]


def _assemble(d: pd.DataFrame, base: pd.DataFrame, parts: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    out = base
    for part in parts if parts is not None else _components(d):
        out = out.merge(part, on=KEYS, how="left", validate="one_to_one")
    out["season_2025"] = (out["season"].astype(str) == "2025").astype(int)
    out[EXTRA_COLS] = out[EXTRA_COLS].fillna(0.0)
    return out[KEYS + EXTRA_COLS]
