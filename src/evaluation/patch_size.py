"""Relate patch size to meta change and to model gains, including the shrink-only baseline.

Usage:
    python -m src.evaluation.patch_size
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from src import config
    from src.evaluation import metrics as ev
    from src.evaluation import noise_floor as nf
    from src.models import _shared as sh
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.evaluation import metrics as ev
    from src.evaluation import noise_floor as nf
    from src.models import _shared as sh

ROLL_CONFIG = "C1"
N_SIM = 400
SHRINK_GRID = np.arange(0.40, 1.21, 0.01)
OUT = config.OUTPUTS / "tables" / "diagnostics"

TARGET = f"pwds_{config.DEFAULT_PWDS_WEIGHTS}"
LAG = f"lag_pwds_{config.DEFAULT_PWDS_WEIGHTS}"


def _order():
    return {p: i for i, p in enumerate(config.ALL_PATCHES)}


def _with_lag_counts() -> pd.DataFrame:
    """Every has-lag row joined to its own previous-patch pick/ban/win counts."""
    d = pd.read_csv(config.ALIGNED_DATASET_CSV, dtype={"patch": str})
    order = _order()
    d["oi"] = d["patch"].map(order)
    prev = d[["champion", "oi", "pr", "br", "wr",
              "picks", "ban_games", "total_games"]].copy()
    prev["oi"] += 1
    prev.columns = ["champion", "oi", "pr_l", "br_l", "wr_l",
                    "picks_l", "ban_l", "tg_l"]
    m = d.merge(prev, on=["champion", "oi"], how="left")
    return m[(m["has_lag"] == 1) & m["pr_l"].notna()].copy()


def patch_table(n_sim: int = N_SIM, seed: int | None = None) -> pd.DataFrame:
    """One row per patch: how big it was, and how much the meta really moved."""
    d = pd.read_csv(config.ALIGNED_DATASET_CSV, dtype={"patch": str})
    m = _with_lag_counts()
    order = _order()
    rows = []
    for patch, g in m.groupby("patch"):
        if len(g) < 5:
            continue
        full = d[d["patch"] == patch]
        changed = ((full["note_changed"].fillna(0) != 0)
                   | (full["stat_changed"].fillna(0) != 0))
        observed = float((g[TARGET] - g[LAG]).abs().mean())
        noise = nf.error_budget(g.reset_index(drop=True), n_sim=n_sim,
                                seed=seed)["noise_only_persistence"]
        rank_rho = (spearmanr(g[TARGET], g[LAG])[0] if len(g) > 2 else np.nan)
        rows.append({
            "patch": patch, "oi": order[patch],
            "games": int(full["total_games"].iloc[0]),
            "champs_scored": len(full),
            "champs_changed": int(changed.sum()),
            "pct_champs_changed": float(100 * changed.mean()),
            "total_changes": float(full["n_changes_total"].fillna(0).sum()),
            "item_classes_changed": float(full["item_class_n_changed"].fillna(0).max()),
            "rank_stability_rho": rank_rho,
            "observed_churn": observed,
            "noise_churn": noise,
            "real_churn": observed - noise,
        })
    return pd.DataFrame(rows).sort_values("oi").reset_index(drop=True)


def size_vs_churn(pt: pd.DataFrame) -> pd.DataFrame:
    """Patch size against churn, raw and noise-corrected, side by side."""
    rows = []
    for col in ("champs_changed", "pct_champs_changed", "total_changes",
                "item_classes_changed", "games"):
        for target in ("observed_churn", "real_churn"):
            rho, p = spearmanr(pt[col], pt[target])
            rows.append({"predictor": col, "churn_measure": target,
                         "spearman_rho": rho, "p_value": p,
                         "significant": p < 0.05})
    return pd.DataFrame(rows)


def _rolling_join() -> pd.DataFrame:
    p = pd.read_csv(config.OUTPUTS / "predictions" / f"ROLL-{ROLL_CONFIG}"
                    / "persistence_preds.csv", dtype={"patch": str})
    x = pd.read_csv(config.OUTPUTS / "predictions" / f"ROLL-{ROLL_CONFIG}"
                    / "xgb_preds.csv", dtype={"patch": str})
    j = p.merge(x, on=["champion", "patch"], suffixes=("_p", "_x"))
    return j[j["has_lag_p"] == 1].copy()


def causal_shrink(j: pd.DataFrame) -> pd.DataFrame:
    """Add a shrink-only prediction that uses no patch information."""
    order = _order()
    hist = pd.read_csv(config.ALIGNED_DATASET_CSV, dtype={"patch": str})
    hist["oi"] = hist["patch"].map(order)
    hist = hist[hist["has_lag"] == 1]
    out = []
    for patch, g in j.groupby("patch"):
        tr = hist[hist["oi"] < order[patch]]
        mean = float(tr[LAG].mean())
        errs = [float(np.abs(tr[TARGET] - (mean + k * (tr[LAG] - mean))).mean())
                for k in SHRINK_GRID]
        k = float(SHRINK_GRID[int(np.argmin(errs))])
        g = g.copy()
        g["shrunk"] = mean + k * (g["y_pred_p"] - mean)
        g["shrink_k"] = k
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _paired(y, a, b) -> dict:
    diff, lo, hi, stats = ev.paired_bootstrap_mae_diff(y, a, b, return_stats=True)
    return {"mae_diff": diff, "ci_lo": lo, "ci_hi": hi,
            "p_value": ev.bootstrap_two_sided_pvalue(stats),
            "verdict": "better" if hi < 0 else "worse" if lo > 0 else "tie"}


def shrinkage_decomposition(z: pd.DataFrame) -> pd.DataFrame:
    """Split the model's gain over persistence into shrinkage and the rest."""
    y = z["y_true_p"].to_numpy()
    mae_p = ev.mae(y, z["y_pred_p"])
    mae_s = ev.mae(y, z["shrunk"])
    mae_x = ev.mae(y, z["y_pred_x"])
    total = mae_p - mae_x
    from_shrink = mae_p - mae_s
    rows = [
        {"comparison": "shrink-only vs persistence", "n": len(z),
         "mae_a": mae_s, "mae_b": mae_p, **_paired(y, z["shrunk"], z["y_pred_p"])},
        {"comparison": "xgb vs persistence", "n": len(z),
         "mae_a": mae_x, "mae_b": mae_p, **_paired(y, z["y_pred_x"], z["y_pred_p"])},
        {"comparison": "xgb vs shrink-only", "n": len(z),
         "mae_a": mae_x, "mae_b": mae_s, **_paired(y, z["y_pred_x"], z["shrunk"])},
    ]
    out = pd.DataFrame(rows)
    out["total_gain_over_persistence"] = total
    out["gain_from_shrinkage"] = from_shrink
    out["gain_from_shrinkage_pct"] = 100 * from_shrink / total
    out["gain_beyond_shrinkage_pct"] = 100 * (total - from_shrink) / total
    return out


def gain_by_group(z: pd.DataFrame, pt: pd.DataFrame, col: str,
                  labels: list[str]) -> pd.DataFrame:
    """Model gain split into terciles of a per-patch quantity."""
    pt = pt.copy()
    pt["group"] = pd.qcut(pt[col], 3, labels=labels)
    zz = z.merge(pt[["patch", "group", col]], on="patch")
    rows = []
    for label in labels:
        g = zz[zz["group"] == label]
        if g.empty:
            continue
        y = g["y_true_p"].to_numpy()
        rows.append({"split_by": col, "group": label,
                     "patches": g["patch"].nunique(), "n_rows": len(g),
                     f"median_{col}": float(g[col].median()),
                     "mae_persistence": ev.mae(y, g["y_pred_p"]),
                     "mae_shrink_only": ev.mae(y, g["shrunk"]),
                     "mae_xgb": ev.mae(y, g["y_pred_x"]),
                     **{f"xgb_vs_persistence_{k}": v
                        for k, v in _paired(y, g["y_pred_x"], g["y_pred_p"]).items()},
                     **{f"xgb_vs_shrink_{k}": v
                        for k, v in _paired(y, g["y_pred_x"], g["shrunk"]).items()}})
    return pd.DataFrame(rows)


def main() -> None:
    assert sh.active_target() == "level", \
        "patch-size diagnostics are defined on the PWDS level target"
    OUT.mkdir(parents=True, exist_ok=True)

    pt = patch_table()
    pt.to_csv(OUT / "patch_size_and_churn.csv", index=False)
    print(f"[patch_size] {len(pt)} scored patches\n")
    print("1. HOW STABLE IS THE META BETWEEN ADJACENT PATCHES?")
    print(f"   rank correlation P vs P-1: median {pt['rank_stability_rho'].median():.3f}"
          f"  min {pt['rank_stability_rho'].min():.3f}"
          f"  max {pt['rank_stability_rho'].max():.3f}")

    sc = size_vs_churn(pt)
    sc.to_csv(OUT / "patch_size_vs_churn.csv", index=False)
    print("\n2. DO BIGGER PATCHES MOVE THE META MORE?")
    for target in ("observed_churn", "real_churn"):
        print(f"   against {target}:")
        for _, r in sc[sc["churn_measure"] == target].iterrows():
            flag = "  <-- significant" if r["significant"] else ""
            print(f"     {r['predictor']:22s} rho {r['spearman_rho']:+.3f}"
                  f"  p={r['p_value']:.3f}{flag}")

    z = causal_shrink(_rolling_join())
    dec = shrinkage_decomposition(z)
    dec.to_csv(OUT / "patch_shrinkage_decomposition.csv", index=False)
    print("\n3. WHAT IS THE MODEL'S ADVANTAGE ACTUALLY MADE OF?")
    print(f"   shrink factor k: median {z['shrink_k'].median():.2f}"
          f" (fitted causally, on earlier patches only)")
    print(dec[["comparison", "n", "mae_a", "mae_b", "mae_diff",
               "ci_lo", "ci_hi", "p_value", "verdict"]].to_string(index=False))
    row = dec.iloc[0]
    print(f"\n   of xgb's {row['total_gain_over_persistence']:.5f} gain over persistence,"
          f" {row['gain_from_shrinkage_pct']:.0f}% is plain shrinkage"
          f" and {row['gain_beyond_shrinkage_pct']:.0f}% is everything else")

    by_size = gain_by_group(z, pt, "champs_changed", ["small", "medium", "LARGE"])
    by_churn = gain_by_group(z, pt, "real_churn",
                             ["meta quiet", "medium", "META MOVED"])
    pd.concat([by_size, by_churn]).to_csv(OUT / "patch_gain_by_group.csv", index=False)
    cols = ["group", "patches", "n_rows", "mae_persistence", "mae_xgb",
            "xgb_vs_persistence_mae_diff", "xgb_vs_persistence_p_value",
            "xgb_vs_shrink_mae_diff", "xgb_vs_shrink_p_value"]
    print("\n   BY PATCH SIZE (champions Riot changed)")
    print(by_size[cols].to_string(index=False))
    print("\n   BY REAL META CHURN (noise removed)")
    print(by_churn[cols].to_string(index=False))
    print(f"\n[patch_size] tables -> {OUT}")


if __name__ == "__main__":
    main()
