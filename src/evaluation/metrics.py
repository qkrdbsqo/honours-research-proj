"""MAE, per-patch Spearman correlation, bootstrap intervals, paired bootstrap and Holm correction."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from src import config
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config


def align_prediction_rows(base, candidate):
    """Join by row identity, rejecting duplicate, missing or conflicting outcomes."""
    keys = ["champion", "patch"]
    if base[keys].isna().any().any() or candidate[keys].isna().any().any():
        raise ValueError("prediction keys must not be missing")
    joined = base.merge(candidate, on=keys, how="left", validate="one_to_one",
                        suffixes=("", "_candidate"), indicator="_prediction_match")
    if (joined["_prediction_match"] != "both").any():
        raise ValueError("candidate predictions are missing baseline rows")
    if not np.allclose(joined["y_true"], joined["y_true_candidate"], atol=1e-7, rtol=0):
        raise ValueError("paired predictions have different targets")
    return joined.drop(columns=["y_true_candidate", "_prediction_match"])


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def spearman_pooled(y_true, y_pred) -> float:
    """Spearman rho over all rows pooled together."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    if len(yt) < 2 or np.std(yt) == 0 or np.std(yp) == 0:
        return float("nan")
    rho, _ = spearmanr(yt, yp)
    return float(rho)


def spearman_per_patch(df: pd.DataFrame,
                       y_true_col: str = "y_true",
                       y_pred_col: str = "y_pred") -> tuple[float, dict[str, float]]:
    """Mean of per-patch Spearman rho (the headline ranking metric)."""
    per: dict[str, float] = {}
    for patch, g in df.groupby("patch"):
        if len(g) < 2:
            continue
        yt, yp = g[y_true_col].to_numpy(), g[y_pred_col].to_numpy()
        if np.std(yt) == 0 or np.std(yp) == 0:
            continue
        rho, _ = spearmanr(yt, yp)
        if not np.isnan(rho):
            per[str(patch)] = float(rho)
    mean_rho = float(np.mean(list(per.values()))) if per else float("nan")
    return mean_rho, per


def _rng(seed):
    return np.random.default_rng(config.RANDOM_SEED if seed is None else seed)


def _pctl(stats: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_mae_ci(y_true, y_pred, n_boot=None, seed=None, alpha=0.05):
    """MAE with a percentile bootstrap CI."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    n = len(yt)
    point = mae(yt, yp)
    if n < 2:
        return point, float("nan"), float("nan")
    n_boot = config.N_BOOTSTRAP if n_boot is None else n_boot
    ae = np.abs(yt - yp)
    idx = _rng(seed).integers(0, n, size=(n_boot, n))
    stats = ae[idx].mean(axis=1)
    lo, hi = _pctl(stats, alpha)
    return point, lo, hi


def paired_bootstrap_mae_diff(y_true, pred_a, pred_b, n_boot=None, seed=None,
                              alpha=0.05, return_stats=False):
    """Paired bootstrap of MAE(a) - MAE(b)."""
    yt = np.asarray(y_true, float)
    ae_a = np.abs(yt - np.asarray(pred_a, float))
    ae_b = np.abs(yt - np.asarray(pred_b, float))
    n = len(yt)
    diff = float(ae_a.mean() - ae_b.mean())
    if n < 2:
        empty = np.array([])
        return (diff, float("nan"), float("nan"), empty) if return_stats \
            else (diff, float("nan"), float("nan"))
    n_boot = config.N_BOOTSTRAP if n_boot is None else n_boot
    idx = _rng(seed).integers(0, n, size=(n_boot, n))
    stats = ae_a[idx].mean(axis=1) - ae_b[idx].mean(axis=1)
    lo, hi = _pctl(stats, alpha)
    return (diff, lo, hi, stats) if return_stats else (diff, lo, hi)


def bootstrap_two_sided_pvalue(stats: np.ndarray) -> float:
    """Two-sided percentile-bootstrap p-value for a paired difference of zero."""
    n = len(stats)
    if n == 0:
        return float("nan")
    p_le = float(np.mean(stats <= 0))
    p_ge = float(np.mean(stats >= 0))
    return float(min(1.0, 2 * min(p_le, p_ge)))


def bootstrap_per_patch_spearman_ci(df: pd.DataFrame, n_boot=None, seed=None,
                                    alpha=0.05, y_true_col="y_true",
                                    y_pred_col="y_pred"):
    """Mean per-patch Spearman rho with a stratified bootstrap CI."""
    point, _ = spearman_per_patch(df, y_true_col, y_pred_col)
    groups = [(g[y_true_col].to_numpy(), g[y_pred_col].to_numpy())
              for _, g in df.groupby("patch") if len(g) >= 2]
    if not groups:
        return point, float("nan"), float("nan")
    n_boot = config.N_BOOTSTRAP if n_boot is None else n_boot
    rng = _rng(seed)
    stats = []
    for _ in range(n_boot):
        rhos = []
        for yt, yp in groups:
            m = len(yt)
            idx = rng.integers(0, m, m)
            yt_b, yp_b = yt[idx], yp[idx]
            if np.std(yt_b) == 0 or np.std(yp_b) == 0:
                continue
            r, _ = spearmanr(yt_b, yp_b)
            if not np.isnan(r):
                rhos.append(r)
        if rhos:
            stats.append(float(np.mean(rhos)))
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = _pctl(np.asarray(stats), alpha)
    return point, lo, hi
