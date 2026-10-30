"""XGBoost with walk-forward hyperparameter selection and feature importance.

Usage:
    python -m src.models.xgb --config C0
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

try:
    from src import config
    from src.models import _shared as sh
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.models import _shared as sh


GRID = config.XGB_GRID


def _make_model(params) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        **params,
    )


def select_params(train) -> dict:
    """Walk-forward CV grid search: params with lowest mean validation MAE."""
    print("[xgb] walk-forward CV grid search")
    feats = sh.features()
    folds = list(sh.walk_forward_folds(train))
    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"  {len(combos)} param combinations x {len(folds)} folds")

    best_params, best_mae = None, np.inf
    for values in combos:
        params = dict(zip(keys, values))
        fold_maes = []
        for ft, fv in folds:
            model = _make_model(params)
            model.fit(ft[feats], ft[sh.TARGET])
            pred = model.predict(fv[feats])
            fold_maes.append(sh.mae(fv[sh.TARGET].to_numpy(), pred))
        avg = float(np.mean(fold_maes))
        if avg < best_mae:
            best_mae, best_params = avg, params
    print(f"  selected {best_params}  (mean val MAE={best_mae:.4f})")
    return best_params


def importance_path():
    """Where the gain table goes, namespaced by config AND target."""
    return config.xgb_feature_importance_csv(
        f"{sh.target_prefix()}{sh.active_config()}")


def export_importance(model: XGBRegressor) -> None:
    """Gain-based feature importance for every Path A feature (0 if unused)."""
    feats = sh.features()
    gain = model.get_booster().get_score(importance_type="gain")
    rows = [{"feature": f, "gain": float(gain.get(f, 0.0))} for f in feats]
    imp = (pd.DataFrame(rows)
           .sort_values("gain", ascending=False)
           .reset_index(drop=True))
    config.ensure_dirs()
    out = importance_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out, index=False)
    print("  top 10 features by gain:")
    for _, r in imp.head(10).iterrows():
        print(f"    {r['feature']:<28} {r['gain']:.3f}")
    print(f"  -> {out}")


def main(config_id: str = "C0") -> None:
    sh.set_config(config_id)
    feats = sh.features()

    df = sh.load_dataset()
    train, test = sh.train_test_split(df)
    best_params = select_params(train)
    sh.save_hyperparams("xgb", best_params)

    sh.impute_lag(train, test)
    model = _make_model(best_params)
    model.fit(train[feats], train[sh.TARGET])
    y_pred = model.predict(test[feats])

    preds = test[["champion", "patch", "has_lag"]].copy()
    preds["y_true"] = test[sh.TARGET].to_numpy()
    preds["y_pred"] = y_pred

    sh.report("xgb", preds)
    sh.save_predictions("xgb", preds)
    export_importance(model)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="C0", choices=list(config.ABLATION_CONFIGS))
    main(ap.parse_args().config)
