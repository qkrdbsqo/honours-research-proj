"""Leave-one-block-out feature audit used to select the C5 feature set.

Usage:
    python -m src.evaluation.degeneracy
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    from src import config
    from src.models import _shared as sh
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.models import _shared as sh

CORR_CLUSTER_THRESHOLD = 0.95


def blocks_for(config_id: str) -> dict[str, list[str]]:
    """Named blocks restricted to the features this config actually uses."""
    feats = set(config.features_for(config_id))
    out = {}
    for name, cols in config.FEATURE_BLOCKS.items():
        present = [c for c in cols if c in feats]
        if present:
            out[name] = present
    return out


def static_screen() -> pd.DataFrame:
    """Per-feature degeneracy indicators, computed without fitting anything."""
    df = pd.read_csv(config.ALIGNED_DATASET_CSV, dtype={"patch": str})
    df = df[df["patch"].isin(config.TRAIN_PATCHES)].copy()
    all_feats = sorted({c for cols in config.FEATURE_BLOCKS.values() for c in cols
                        if c in df.columns})
    block_of = {c: n for n, cols in config.FEATURE_BLOCKS.items() for c in cols}

    sub = df[all_feats].fillna(0.0)
    corr = sub.corr().abs()
    rows = []
    for f in all_feats:
        col = sub[f]
        per_patch_distinct = df.groupby("patch")[f].nunique()
        partners = [g for g in all_feats
                    if g != f and corr.loc[f, g] >= CORR_CLUSTER_THRESHOLD]
        rows.append({
            "feature": f,
            "block": block_of.get(f, "?"),
            "nonzero_rate": float((col != 0).mean()),
            "variance": float(col.var()),
            "median_distinct_within_patch": float(per_patch_distinct.median()),
            "patch_constant": bool((per_patch_distinct <= 1).all()),
            "corr_partners": ";".join(partners),
            "n_corr_partners": len(partners),
        })
    return pd.DataFrame(rows)


def _cv_ridge(folds, feats: list[str]) -> float:
    """Mean validation MAE, sweeping config.RIDGE_ALPHAS for THIS subset."""
    best = np.inf
    for alpha in config.RIDGE_ALPHAS:
        maes = []
        for ft, fv in folds:
            scaler = StandardScaler().fit(ft[feats])
            m = Ridge(alpha=alpha, random_state=config.RANDOM_SEED)
            m.fit(scaler.transform(ft[feats]), ft[sh.TARGET])
            maes.append(sh.mae(fv[sh.TARGET].to_numpy(),
                               m.predict(scaler.transform(fv[feats]))))
        best = min(best, float(np.mean(maes)))
    return best


def _cv_xgb(folds, feats: list[str]) -> float:
    """Mean validation MAE over a reduced XGB grid, re-swept per subset."""
    best = np.inf
    for lr in (0.05, 0.1):
        for n_est in (50, 100):
            for reg_lambda in (1, 5, 10):
                maes = []
                for ft, fv in folds:
                    m = XGBRegressor(max_depth=3, learning_rate=lr,
                                     n_estimators=n_est, reg_lambda=reg_lambda,
                                     random_state=config.RANDOM_SEED,
                                     verbosity=0)
                    m.fit(ft[feats], ft[sh.TARGET])
                    maes.append(sh.mae(fv[sh.TARGET].to_numpy(), m.predict(fv[feats])))
                best = min(best, float(np.mean(maes)))
    return best


def block_ablation(config_id: str) -> pd.DataFrame:
    """Drop each block in turn; report the change in mean validation MAE."""
    sh.set_config(config_id)
    df = sh.load_dataset()
    train, _ = sh.train_test_split(df)
    folds = list(sh.walk_forward_folds(train))
    feats = config.features_for(config_id)
    blocks = blocks_for(config_id)

    base_ridge = _cv_ridge(folds, feats)
    base_xgb = _cv_xgb(folds, feats)

    rows = []
    for name, cols in blocks.items():
        keep = [f for f in feats if f not in cols]
        if not keep:
            continue
        r = _cv_ridge(folds, keep)
        x = _cv_xgb(folds, keep)
        rows.append({
            "config": config_id, "block": name, "n_cols": len(cols),
            "ridge_val_mae": r, "ridge_delta": r - base_ridge,
            "xgb_val_mae": x, "xgb_delta": x - base_xgb,
            "delta_val_mae": float(np.mean([r - base_ridge, x - base_xgb])),
            "degenerate": bool(r < base_ridge and x < base_xgb),
        })
    out = pd.DataFrame(rows)
    out.insert(3, "base_ridge_val_mae", base_ridge)
    out.insert(4, "base_xgb_val_mae", base_xgb)
    return out


def lean_features(config_id: str = "C4") -> list[str]:
    """Features that survive the audit (the C6 feature set)."""
    tbl = block_ablation(config_id)
    drop = set()
    for _, r in tbl.iterrows():
        if r["degenerate"]:
            drop.update(config.FEATURE_BLOCKS[r["block"]])
    return [f for f in config.features_for(config_id) if f not in drop]


def main() -> None:
    assert sh.active_target() == "level", (
        f"degeneracy.main() writes level-target paths but the active target "
        f"is {sh.active_target()!r}")
    config.ensure_dirs()

    screen = static_screen()
    screen.to_csv(config.FEATURE_SCREEN_CSV, index=False)
    print("=== F1 static screen ===")
    flagged = screen[(screen["patch_constant"]) | (screen["nonzero_rate"] < 0.05)]
    print(f"  {len(flagged)} of {len(screen)} features are patch-constant or "
          f"fire on < 5% of rows:")
    for _, r in flagged.iterrows():
        print(f"    {r['feature']:<32} block={r['block']:<12} "
              f"nonzero={100 * r['nonzero_rate']:5.1f}%  "
              f"patch_constant={r['patch_constant']}")
    print(f"  -> {config.FEATURE_SCREEN_CSV}\n")

    print("=== F2 leave-one-block-out (CV validation MAE, never test) ===")
    tables = []
    for cid in config.ABLATION_CONFIGS:
        print(f"\n  -- {cid}")
        t = block_ablation(cid)
        tables.append(t)
        for _, r in t.sort_values("delta_val_mae").iterrows():
            tag = "  <- DEGENERATE" if r["degenerate"] else ""
            print(f"    drop {r['block']:<12} ({int(r['n_cols']):2d} cols)  "
                  f"ridge {r['ridge_delta']:+.5f}  xgb {r['xgb_delta']:+.5f}{tag}")
    pd.concat(tables, ignore_index=True).to_csv(config.BLOCK_ABLATION_CSV, index=False)
    print(f"\n  -> {config.BLOCK_ABLATION_CSV}")

    lean = lean_features("C4")
    print(f"\n=== C6 lean feature set: {len(lean)} of "
          f"{len(config.features_for('C4'))} C4 features survive ===")
    print(f"  {lean}")


if __name__ == "__main__":
    main()
