"""Paired MAE differences against a baseline, origin-cluster bootstrap and Holm families."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.evaluation import compare as cmp

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["champion", "patch"]
N_BOOT = 20_000


def read(path):
    return pd.read_csv(path, dtype={"patch": str})


def paired_frame(base, candidate, subset):
    assert not base.duplicated(KEYS).any() and not candidate.duplicated(KEYS).any()
    j = base.merge(candidate, on=KEYS, suffixes=("_b", "_c"), validate="one_to_one")
    assert np.allclose(j.y_true_b, j.y_true_c, atol=1e-7, rtol=0)
    flags = cmp.changed_flags()
    j = j.merge(flags, on=KEYS, validate="one_to_one")
    j = j[j.has_lag_b == 1]
    origins = sorted(j.patch.unique())
    if subset == "changed_has_lag":
        j = j[j.changed == 1]
    elif subset == "has_lag_dense":
        data = read(ROOT / "data/processed/aligned_dataset.csv")
        dense = data.groupby("patch").total_games.first()
        j = j[j.patch.isin(dense[dense >= config.SPARSE_PATCH_MIN_GAMES].index)]
    elif subset in ("item_affected", "any_affected"):
        j = j[j[subset] == 1]
    elif subset != "has_lag":
        raise ValueError(subset)
    j["loss_diff"] = np.abs(j.y_true_b - j.y_pred_c) - np.abs(j.y_true_b - j.y_pred_b)
    j.attrs["origins"] = origins
    return j


def cluster_result(frame, block=1, n_boot=N_BOOT, seed=config.RANDOM_SEED):
    """Row-weighted MAE difference; resample whole origins, not individual rows."""
    grouped = frame.groupby("patch").loss_diff.agg(["sum", "count"]).sort_index()
    if "origins" in frame.attrs:
        grouped = grouped.reindex(frame.attrs["origins"], fill_value=0)
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    g = len(grouped)
    if g < 2:
        raise ValueError("at least two origins are required")
    rng = np.random.default_rng(seed)
    def draw_indices(n_draws):
        if block == 1:
            return rng.integers(g, size=(n_draws, g))
        chunks = []
        for season in sorted({p.split(".")[0] for p in grouped.index}):
            positions = np.flatnonzero([p.startswith(season + ".") for p in grouped.index])
            size = len(positions)
            starts = rng.integers(size, size=(n_draws, int(np.ceil(size / block))))
            local = ((starts[..., None] + np.arange(block)) % size).reshape(n_draws, -1)[:, :size]
            chunks.append(positions[local])
        return np.concatenate(chunks, axis=1)
    idx = draw_indices(n_boot)
    den = counts[idx].sum(axis=1)
    empty_redraws = 0
    while (den == 0).any():
        bad = den == 0
        empty_redraws += int(bad.sum())
        idx[bad] = draw_indices(int(bad.sum()))
        den = counts[idx].sum(axis=1)
        if empty_redraws > 100 * n_boot:
            raise ValueError("bootstrap subset too sparse")
    stats = sums[idx].sum(axis=1) / den
    lo, hi = np.percentile(stats, [2.5, 97.5])
    p = min(1., 2 * min(np.mean(stats <= 0), np.mean(stats >= 0)))
    return {"n": len(frame), "n_origins": g, "empty_redraws": empty_redraws,
            "mae_diff": float(frame.loss_diff.mean()),
            "ci_lo": lo, "ci_hi": hi, "p_value": p,
            "verdict": "better" if hi < 0 else "worse" if lo > 0 else "tie"}


def apply_family(frame):
    frame = frame.copy()
    counted = frame.counted_in_family.astype(bool)
    frame["holm_significant"] = False
    frame.loc[counted, "holm_significant"] = cmp.holm_bonferroni(
        frame.loc[counted, "p_value"], resolution=2 / N_BOOT)
    frame["m"] = int(counted.sum())
    for i, r in frame[frame.aliases.fillna("").ne("")].iterrows():
        source = frame[(frame.config == r.aliases) & (frame.model == r.model)
                       & (frame.subset == r.subset) & (frame.target == r.target)]
        assert len(source) == 1
        frame.loc[i, "holm_significant"] = bool(source.iloc[0].holm_significant)
    return frame
