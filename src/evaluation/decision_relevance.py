"""Ranking metrics and a breakdown of where the MAE gain over persistence comes from.

Usage:
    python -m src.evaluation.decision_relevance
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from src import config
    from src.evaluation import metrics as ev
    from src.evaluation import patch_size as ps
    from src.evaluation import rolling_origin as ro
    from src.models import _shared as sh
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.evaluation import metrics as ev
    from src.evaluation import patch_size as ps
    from src.evaluation import rolling_origin as ro
    from src.models import _shared as sh

CONFIG = "C1"
TOP_K = 10
OUT = config.OUTPUTS / "tables" / "diagnostics"
MODELS = ("persistence", "shrink", "ridge", "xgb")
FROZEN = tuple(config.TEST_PATCHES)


def predictions() -> pd.DataFrame:
    """(champion, patch, y_true, one column per model), has-lag rows only."""
    j = ps.causal_shrink(ps._rolling_join())        # persistence + xgb + shrunk
    r = pd.read_csv(config.OUTPUTS / "predictions" / f"ROLL-{CONFIG}"
                    / "ridge_preds.csv", dtype={"patch": str})
    out = j[["champion", "patch", "y_true_p", "y_pred_p", "shrunk", "y_pred_x"]]
    out = out.rename(columns={"y_true_p": "y_true", "y_pred_p": "persistence",
                              "shrunk": "shrink", "y_pred_x": "xgb"})
    out = ev.align_prediction_rows(out, r[["champion", "patch", "y_true", "y_pred"]].rename(
        columns={"y_pred": "ridge"}))
    return out.reset_index(drop=True)


def _order_metrics(g: pd.DataFrame, model: str, k: int = TOP_K) -> dict:
    """rho, precision@k and displacement@k for one origin."""
    yt, yp = g["y_true"].to_numpy(), g[model].to_numpy()
    if len(g) < k + 1 or np.std(yt) == 0 or np.std(yp) == 0:
        return {}
    rho = spearmanr(yt, yp).statistic
    true_rank = pd.Series(-yt).rank(method="first")
    pred_rank = pd.Series(-yp).rank(method="first")
    top_true = set(np.argsort(-yt, kind="stable")[:k])
    top_pred = set(np.argsort(-yp, kind="stable")[:k])
    idx = list(top_true)
    return {"rho": float(rho),
            "precision_at_k": len(top_true & top_pred) / k,
            "displacement_at_k": float(
                np.abs(pred_rank.to_numpy()[idx] - true_rank.to_numpy()[idx]).mean()),
            "n": len(g)}


def ranking_table(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-origin order metrics, and the pooled table with paired tests."""
    rows = []
    for patch, g in p.groupby("patch"):
        for m in MODELS:
            d = _order_metrics(g, m)
            if d:
                rows.append({"patch": patch, "model": m, **d})
    per_origin = pd.DataFrame(rows)

    pooled = []
    for m in MODELS:
        d = per_origin[per_origin["model"] == m]
        base = per_origin[per_origin["model"] == "persistence"]
        merged = d.merge(base, on="patch", suffixes=("", "_p"))
        row = {"model": m, "n_origins": len(d),
               "mae": ev.mae(p["y_true"], p[m]),
               "rho": d["rho"].mean(),
               "precision_at_k": d["precision_at_k"].mean(),
               "displacement_at_k": d["displacement_at_k"].mean(),
               "sd_ratio": float(np.std(p[m]) / np.std(p["y_true"])),
               "corr_with_persistence": float(np.corrcoef(p[m], p["persistence"])[0, 1])}
        for metric in ("rho", "precision_at_k"):
            diff = merged[metric] - merged[f"{metric}_p"]
            lo, hi, pv = _boot_mean(diff.to_numpy())
            row[f"d_{metric}"] = float(diff.mean())
            row[f"d_{metric}_lo"], row[f"d_{metric}_hi"] = lo, hi
            row[f"d_{metric}_p"] = pv
        pooled.append(row)
    return per_origin, pd.DataFrame(pooled)


def _boot_mean(x: np.ndarray, n_boot: int | None = None, seed: int | None = None):
    """Bootstrap CI and two-sided p for a mean difference over origins."""
    n_boot = config.N_BOOTSTRAP if n_boot is None else n_boot
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    stats = x[idx].mean(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi), ev.bootstrap_two_sided_pvalue(stats)


def with_strata(p: pd.DataFrame) -> pd.DataFrame:
    """Add the three ex-ante strata and the per-row gain over persistence."""
    p = p.copy()
    p["gain_vs_persistence"] = (p["y_true"] - p["persistence"]).abs() \
        - (p["y_true"] - p["xgb"]).abs()
    p["gain_vs_shrink"] = (p["y_true"] - p["shrink"]).abs() \
        - (p["y_true"] - p["xgb"]).abs()

    p["lag_quartile"] = pd.qcut(p["persistence"], 4,
                                labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])

    changed = ro.add_changed_flag(
        pd.read_csv(config.ALIGNED_DATASET_CSV, dtype={"patch": str}))
    p = p.merge(changed[["champion", "patch", "changed"]],
                on=["champion", "patch"], how="left")
    p["changed"] = p["changed"].fillna(0).astype(int)

    sh.set_config(CONFIG)
    games = sh.games_by_patch()
    order = [q for q in config.ALL_PATCHES if q in games]
    prev = {q: order[i - 1] for i, q in enumerate(order) if i > 0}
    p["prev_games"] = p["patch"].map(lambda q: games.get(prev.get(q, ""), np.nan))
    p["prev_games_tercile"] = pd.qcut(p["prev_games"], 3,
                                      labels=["thin", "medium", "dense"])
    return p


def strata_table(p: pd.DataFrame) -> pd.DataFrame:
    """Mean gain and share of the total, per stratum, with a cluster CI."""
    total = p["gain_vs_persistence"].sum()
    rows = []
    for name, col in (("lag quartile", "lag_quartile"),
                      ("patch touched the champion", "changed"),
                      ("games in the previous patch", "prev_games_tercile")):
        for level, g in p.groupby(col, observed=True):
            per_origin = g.groupby("patch")["gain_vs_persistence"].agg(["sum", "count"])
            lo, hi, pv = _boot_cluster(per_origin["sum"].to_numpy(),
                                       per_origin["count"].to_numpy())
            rows.append({"stratum": name, "level": str(level), "n": len(g),
                         "share_of_rows_pct": 100 * len(g) / len(p),
                         "mae_persistence": ev.mae(g["y_true"], g["persistence"]),
                         "mae_xgb": ev.mae(g["y_true"], g["xgb"]),
                         "mean_gain": g["gain_vs_persistence"].mean(),
                         "ci_lo": lo, "ci_hi": hi, "p_value": pv,
                         "share_of_total_gain_pct": 100 * g["gain_vs_persistence"].sum() / total})
    return pd.DataFrame(rows)


def _boot_cluster(sums: np.ndarray, counts: np.ndarray):
    """CI and p for a mean gain, resampling origins rather than rows."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.integers(0, len(sums), size=(config.N_BOOTSTRAP, len(sums)))
    stats = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi), ev.bootstrap_two_sided_pvalue(stats)


def champion_table(p: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-champion gain, its concentration, and whether it repeats in time."""
    per_champ = (p.groupby("champion")
                 .agg(rows=("gain_vs_persistence", "size"),
                      total_gain=("gain_vs_persistence", "sum"),
                      mean_gain=("gain_vs_persistence", "mean"))
                 .sort_values("total_gain", ascending=False).reset_index())

    total = per_champ["total_gain"].sum()
    top_decile = max(1, len(per_champ) // 10)
    gained = per_champ[per_champ["total_gain"] > 0]

    origins = sorted(p["patch"].unique(), key=ro._patch_key)
    half = len(origins) // 2
    first = p[p["patch"].isin(origins[:half])].groupby("champion")["gain_vs_persistence"].sum()
    second = p[p["patch"].isin(origins[half:])].groupby("champion")["gain_vs_persistence"].sum()
    both = pd.concat([first.rename("first"), second.rename("second")], axis=1).dropna()
    split_half = spearmanr(both["first"], both["second"]).statistic

    summary = {
        "n_champions": len(per_champ),
        "champions_with_positive_gain_pct": 100 * len(gained) / len(per_champ),
        "top_decile_share_of_gain_pct": 100 * per_champ["total_gain"].head(top_decile).sum() / total,
        "top_decile_n": top_decile,
        "split_half_spearman": float(split_half),
        "split_half_n": len(both),
        "first_half_origins": f"{origins[0]}..{origins[half - 1]}",
        "second_half_origins": f"{origins[half]}..{origins[-1]}",
    }
    return per_champ, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = predictions()
    print(f"[decision] {len(p)} has-lag rows over {p['patch'].nunique()} origins "
          f"(rolling origin, {CONFIG})")

    per_origin, pooled = ranking_table(p)
    per_origin.to_csv(OUT / "decision_ranking_per_origin.csv", index=False)
    pooled.to_csv(OUT / "decision_ranking.csv", index=False)
    print("\n=== RQ-A: does lower error mean better ranking? (per-origin means)")
    print(pooled[["model", "mae", "rho", "precision_at_k", "displacement_at_k",
                  "sd_ratio", "d_rho", "d_rho_p", "d_precision_at_k",
                  "d_precision_at_k_p"]].round(4).to_string(index=False))

    frozen = p[p["patch"].isin(FROZEN)]
    _, pooled_frozen = ranking_table(frozen)
    pooled_frozen.insert(0, "window", "frozen")
    pooled_frozen.to_csv(OUT / "decision_ranking_frozen.csv", index=False)
    print("\n--- the same on the frozen window only (n=%d)" % len(frozen))
    print(pooled_frozen[["model", "mae", "rho", "precision_at_k"]].round(4).to_string(index=False))

    s = with_strata(p)
    strata = strata_table(s)
    strata.to_csv(OUT / "decision_gain_by_stratum.csv", index=False)
    print("\n=== RQ-B: where does the gain come from?")
    print(strata[["stratum", "level", "n", "mean_gain", "ci_lo", "ci_hi",
                  "p_value", "share_of_total_gain_pct"]].round(5).to_string(index=False))

    per_champ, summary = champion_table(s)
    per_champ.to_csv(OUT / "decision_gain_by_champion.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT / "decision_gain_concentration.csv", index=False)
    print("\n=== RQ-B: concentration, and does it repeat?")
    for k, v in summary.items():
        print(f"  {k}: {v if isinstance(v, str) else round(v, 4)}")
    print("\n  most helped:", ", ".join(per_champ.head(5)["champion"]))
    print("  most hurt:  ", ", ".join(per_champ.tail(5)["champion"]))


if __name__ == "__main__":
    main()
