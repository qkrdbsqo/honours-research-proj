"""Frozen BERT with numeric-only scaling (bert_ctrl), the control for bert_ft.

Usage:
    python -m src.models.bert_control --config C1
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    from src import config
    from src.models import _shared as sh
    from src.models import bert_regressor as br
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.models import _shared as sh
    from src.models import bert_regressor as br

MODEL_NAME = "bert_ctrl"


class NumericOnlyScaler:
    """Standardise the last two columns only; pass the 768 CLS dims through."""

    def fit(self, X: np.ndarray) -> "NumericOnlyScaler":
        num = X[:, -2:]
        self.mean_ = num.mean(axis=0)
        scale = num.std(axis=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = X.copy()
        out[:, -2:] = (out[:, -2:] - self.mean_) / self.scale_
        return out


def select_lr(train: pd.DataFrame, emb, key_to_row) -> tuple[float, int]:
    """Walk-forward CV, identical to bert_regressor.select_lr but for the scaler."""
    print(f"[{MODEL_NAME}] walk-forward CV (numeric-only standardisation)")
    folds = list(sh.walk_forward_folds(train))
    best_lr, best_mae, best_epochs = None, np.inf, br.MAX_EPOCHS
    for lr in br.LEARNING_RATES:
        fold_maes, fold_epochs = [], []
        for ft, fv in folds:
            scaler = NumericOnlyScaler().fit(br.make_X(ft, emb, key_to_row))
            X_tr = scaler.transform(br.make_X(ft, emb, key_to_row)).astype(np.float32)
            X_val = scaler.transform(br.make_X(fv, emb, key_to_row)).astype(np.float32)
            _, vmae, vep = br._train_head(
                X_tr, ft[sh.TARGET].to_numpy(), lr, br.MAX_EPOCHS,
                X_val, fv[sh.TARGET].to_numpy())
            fold_maes.append(vmae)
            fold_epochs.append(vep)
        avg = float(np.mean(fold_maes))
        marker = ""
        if avg < best_mae:
            best_mae, best_lr = avg, lr
            best_epochs = int(round(float(np.mean(fold_epochs))))
            marker = "  <- best"
        print(f"  lr={lr:<7g} mean val MAE={avg:.4f}  "
              f"best epochs/fold={[int(e) for e in fold_epochs]}{marker}")
    print(f"  selected lr={best_lr:g}, final-fit epochs={best_epochs}")
    return best_lr, best_epochs


def main(config_id: str = "C1") -> None:
    sh.set_config(config_id)

    df = sh.load_dataset()
    train, test = sh.train_test_split(df)
    emb, key_to_row = br.load_embeddings()
    missing = [(c, p) for c, p in zip(df["champion"], df["patch"])
               if (c, p) not in key_to_row]
    if missing:
        raise RuntimeError(f"{len(missing)} rows lack a cached embedding, "
                           f"e.g. {missing[:3]}. Rebuild with "
                           f"'python -m src.features.text_features --force'.")

    best_lr, epochs = select_lr(train, emb, key_to_row)
    sh.save_hyperparams(MODEL_NAME, {"learning_rate": best_lr, "epochs": epochs})

    sh.impute_lag(train, test)
    scaler = NumericOnlyScaler().fit(br.make_X(train, emb, key_to_row))
    X_tr = scaler.transform(br.make_X(train, emb, key_to_row)).astype(np.float32)
    X_te = scaler.transform(br.make_X(test, emb, key_to_row)).astype(np.float32)
    model, _, _ = br._train_head(X_tr, train[sh.TARGET].to_numpy(), best_lr, epochs)
    y_pred = br._predict(model, X_te)

    preds = test[["champion", "patch", "has_lag"]].copy()
    preds["y_true"] = test[sh.TARGET].to_numpy()
    preds["y_pred"] = y_pred

    sh.report(MODEL_NAME, preds)
    sh.save_predictions(MODEL_NAME, preds)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="C1", choices=list(config.ABLATION_CONFIGS))
    args = ap.parse_args()
    main(args.config)
