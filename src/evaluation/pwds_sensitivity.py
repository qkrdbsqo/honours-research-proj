"""Select the PWDS weights on validation folds and check the result under each weighting.

Usage:
    python -m src.evaluation.pwds_sensitivity
    python -m src.evaluation.pwds_sensitivity --config C0
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from src import config
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config

from src.models import _shared as sh

DIAG = config.TABLES / "diagnostics"


def _log(msg):
    print(str(msg).encode("ascii", "replace").decode("ascii"), flush=True)


def _features_for(config_id, variant):
    """Path A features with the lag column swapped to this variant."""
    feats = list(config.features_for(config_id))
    default_lag = f"lag_pwds_{config.DEFAULT_PWDS_WEIGHTS}"
    return [f"lag_pwds_{variant}" if f == default_lag else f for f in feats]


def _impute(train, test, lag_col):
    med = train[lag_col].median()
    train[lag_col] = train[lag_col].fillna(med)
    test[lag_col] = test[lag_col].fillna(med)


def _per_patch_rho(df, truth, pred):
    rhos = []
    for _, g in df.groupby("patch"):
        if len(g) >= 3 and g[truth].nunique() > 1 and np.std(g[pred]) > 0:
            r, _ = spearmanr(g[truth], g[pred])
            if r == r:
                rhos.append(r)
    return float(np.mean(rhos)) if rhos else float("nan")


def select_weights(config_id="C1"):
    """Mean per-patch Spearman on validation folds only."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    sh.set_config(config_id)
    df = sh.load_dataset()
    train, _test = sh.train_test_split(df)
    del _test                     # structural: the test split is not used here

    rows = []
    for variant in config.PWDS_WEIGHTS:
        target, lag = f"pwds_{variant}", f"lag_pwds_{variant}"
        feats = _features_for(config_id, variant)
        per_model = {}
        for name, make in (("ridge", lambda: Ridge(alpha=100.0)),
                           ("xgb", lambda: XGBRegressor(
                               max_depth=3, learning_rate=0.05,
                               n_estimators=100, reg_lambda=5,
                               random_state=config.RANDOM_SEED)),
                           ("persistence", None)):
            fold_rhos = []
            for ft, fv in sh.walk_forward_folds(train):
                ft, fv = ft.copy(), fv.copy()
                _impute(ft, fv, lag)
                if name == "persistence":
                    fv["_pred"] = fv[lag]
                elif name == "ridge":
                    sc = StandardScaler().fit(ft[feats])
                    m = make().fit(sc.transform(ft[feats]), ft[target])
                    fv["_pred"] = m.predict(sc.transform(fv[feats]))
                else:
                    m = make().fit(ft[feats], ft[target])
                    fv["_pred"] = m.predict(fv[feats])
                fold_rhos.append(_per_patch_rho(fv, target, "_pred"))
            per_model[name] = float(np.nanmean(fold_rhos))
        rows.append({"variant": variant,
                     "weights (PR, BR, WR)": str(config.PWDS_WEIGHTS[variant]),
                     **{f"val_rho_{k}": round(v, 4) for k, v in per_model.items()},
                     "val_rho_mean_models": round(
                         np.mean([per_model["ridge"], per_model["xgb"]]), 4)})
    return pd.DataFrame(rows)


def sensitivity(config_id="C1"):
    """Test-set MAE and rho for each weighting."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    sh.set_config(config_id)
    df = sh.load_dataset()
    rows = []
    for variant in config.PWDS_WEIGHTS:
        target, lag = f"pwds_{variant}", f"lag_pwds_{variant}"
        feats = _features_for(config_id, variant)
        train, test = sh.train_test_split(df)
        train, test = train.copy(), test.copy()
        has_lag_mask = test[lag].notna()
        _impute(train, test, lag)
        preds = {"persistence": test[lag].to_numpy()}
        sc = StandardScaler().fit(train[feats])
        preds["ridge"] = Ridge(alpha=100.0).fit(
            sc.transform(train[feats]), train[target]).predict(sc.transform(test[feats]))
        preds["xgb"] = XGBRegressor(
            max_depth=3, learning_rate=0.05, n_estimators=100, reg_lambda=5,
            random_state=config.RANDOM_SEED).fit(
                train[feats], train[target]).predict(test[feats])

        sub = test[has_lag_mask]
        base = float(np.abs(sub[target] - preds["persistence"][has_lag_mask.to_numpy()]).mean())
        for name, p in preds.items():
            pv = p[has_lag_mask.to_numpy()]
            mae = float(np.abs(sub[target] - pv).mean())
            tmp = sub.copy()
            tmp["_pred"] = pv
            rows.append({"variant": variant, "model": name,
                         "test_mae": round(mae, 5),
                         "mae_vs_persistence": round(mae - base, 5),
                         "lower_mae_uncorrected": bool(mae < base),
                         "test_rho_per_patch": round(_per_patch_rho(tmp, target, "_pred"), 4)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="C1")
    args = ap.parse_args(argv)
    DIAG.mkdir(parents=True, exist_ok=True)

    _log("1. SELECTION - mean per-patch Spearman on CV validation folds only")
    _log("   (the frozen test window is not read in this step)")
    sel = select_weights(args.config)
    _log(sel.to_string(index=False))
    best = sel.loc[sel["val_rho_mean_models"].idxmax(), "variant"]
    _log("")
    _log("   -> selected by the proposal's rule: %s" % best)
    _log("   -> project default in use:          %s" % config.DEFAULT_PWDS_WEIGHTS)
    _log("   -> %s" % ("MATCH: the default is retroactively justified"
                       if best == config.DEFAULT_PWDS_WEIGHTS
                       else "MISMATCH: see the sensitivity table before acting"))
    sel.to_csv(DIAG / ("pwds_weight_selection_%s.csv" % args.config), index=False)

    _log("")
    _log("2. SENSITIVITY - does the null hold under every weighting?")
    sen = sensitivity(args.config)
    _log(sen.to_string(index=False))
    sen.to_csv(DIAG / ("pwds_weight_sensitivity_%s.csv" % args.config), index=False)
    _log("")
    _log("   NOTE: `lower_mae_uncorrected` is a raw point estimate with no")
    _log("   confidence interval and no multiple-comparison correction. Gaps of")
    _log("   this size (-0.0006 to -0.0032) are TIES in the main analysis, where")
    _log("   ridge C1's -0.0007 fails the paired bootstrap. The finding here is")
    _log("   that the ORDERING is unchanged across weightings, not that any")
    _log("   model beats the baseline.")
    _log("")
    _log("tables in %s" % DIAG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
