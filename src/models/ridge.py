"""Ridge regression with walk-forward hyperparameter selection.

Usage:
    python -m src.models.ridge --config C0
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from src import config
    from src.models import _shared as sh
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.models import _shared as sh

ALPHAS = config.RIDGE_ALPHAS


def _fit_predict(alpha, ft, fv):
    """Standardise on ft, fit Ridge(alpha), return predictions on fv."""
    feats = sh.features()
    scaler = StandardScaler().fit(ft[feats])
    Xt = scaler.transform(ft[feats])
    Xv = scaler.transform(fv[feats])
    model = Ridge(alpha=alpha, random_state=config.RANDOM_SEED)
    model.fit(Xt, ft[sh.TARGET])
    return model.predict(Xv)


def select_alpha(train) -> float:
    """Walk-forward CV: pick alpha with lowest mean validation MAE."""
    print("[ridge] walk-forward CV")
    folds = list(sh.walk_forward_folds(train))
    best_alpha, best_mae = None, np.inf
    for alpha in ALPHAS:
        fold_maes = []
        for ft, fv in folds:
            pred = _fit_predict(alpha, ft, fv)
            fold_maes.append(sh.mae(fv[sh.TARGET].to_numpy(), pred))
        avg = float(np.mean(fold_maes))
        marker = ""
        if avg < best_mae:
            best_mae, best_alpha, marker = avg, alpha, "  <- best"
        print(f"  alpha={alpha:<8} mean val MAE={avg:.4f}{marker}")
    print(f"  selected alpha={best_alpha}")
    return best_alpha


def main(config_id: str = "C0") -> None:
    sh.set_config(config_id)
    feats = sh.features()

    df = sh.load_dataset()
    train, test = sh.train_test_split(df)
    best_alpha = select_alpha(train)
    sh.save_hyperparams("ridge", {"alpha": best_alpha})

    sh.impute_lag(train, test)
    scaler = StandardScaler().fit(train[feats])
    model = Ridge(alpha=best_alpha, random_state=config.RANDOM_SEED)
    model.fit(scaler.transform(train[feats]), train[sh.TARGET])
    y_pred = model.predict(scaler.transform(test[feats]))

    preds = test[["champion", "patch", "has_lag"]].copy()
    preds["y_true"] = test[sh.TARGET].to_numpy()
    preds["y_pred"] = y_pred

    sh.report("ridge", preds)
    sh.save_predictions("ridge", preds)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="C0", choices=list(config.ABLATION_CONFIGS))
    main(ap.parse_args().config)
