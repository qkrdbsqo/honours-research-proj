"""Linear head on frozen BERT [CLS] embeddings (bert).

Usage:
    python -m src.models.bert_regressor --config C0
"""
from __future__ import annotations

import copy
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

try:
    from src import config
    from src.models import _shared as sh
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.models import _shared as sh

LEARNING_RATES = config.BERT_LR_GRID
MAX_EPOCHS = config.BERT_MAX_EPOCHS
PATIENCE = config.BERT_PATIENCE
INPUT_DIM = config.BERT_HIDDEN + 2   # 768 CLS dims + lag + has_lag


def _embedding_paths() -> tuple:
    """(embeddings.npy, index.csv) for the active config's text variant."""
    if sh.cfg()["text"] == "enriched":
        return config.CLS_EMBEDDINGS_ENRICHED_NPY, config.CLS_INDEX_ENRICHED_CSV
    return config.CLS_EMBEDDINGS_NPY, config.CLS_INDEX_CSV


def load_embeddings() -> tuple[np.ndarray, dict[tuple[str, str], int]]:
    """Load the cached [CLS] matrix and a (champion, patch) -> row-index map."""
    emb_path, idx_path = _embedding_paths()
    from src.features.text_features import load_embedding_cache
    return load_embedding_cache(emb_path, idx_path)


def make_X(df: pd.DataFrame, emb: np.ndarray,
           key_to_row: dict[tuple[str, str], int]) -> np.ndarray:
    """Assemble the raw 770-d input [CLS | lag | has_lag] in ``df`` row order."""
    rows = [key_to_row[(c, p)] for c, p in zip(df["champion"], df["patch"])]
    cls = emb[rows]                                        # (n, 768)
    lag = df[sh.LAG].to_numpy().reshape(-1, 1)             # (n, 1)
    has = df["has_lag"].to_numpy().reshape(-1, 1)          # (n, 1)
    return np.hstack([cls, lag, has]).astype(np.float32)


def _train_head(X_tr: np.ndarray, y_tr: np.ndarray, lr: float,
                epochs: int,
                X_val: np.ndarray | None = None,
                y_val: np.ndarray | None = None) -> tuple[nn.Linear, float, int]:
    """Train nn.Linear(770, 1) with Adam + MSE, full-batch, deterministic."""
    torch.manual_seed(config.RANDOM_SEED)
    model = nn.Linear(INPUT_DIM, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(X_tr)
    yt = torch.from_numpy(y_tr.astype(np.float32)).reshape(-1, 1)
    Xv = torch.from_numpy(X_val) if X_val is not None else None

    best_mae, best_epoch, best_state, wait = np.inf, epochs, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss_fn(model(Xt), yt).backward()
        opt.step()

        if Xv is None:
            continue
        model.eval()
        with torch.no_grad():
            pv = model(Xv).numpy().ravel()
        vmae = sh.mae(y_val, pv)
        if vmae < best_mae - 1e-9:
            best_mae, best_epoch, wait = vmae, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_mae), best_epoch


def _predict(model: nn.Linear, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy().ravel()


def select_lr(train: pd.DataFrame, emb, key_to_row) -> tuple[float, int]:
    """Walk-forward CV: pick LR by mean best validation MAE."""
    print("[bert] walk-forward CV")
    folds = list(sh.walk_forward_folds(train))
    best_lr, best_mae, best_epochs = None, np.inf, MAX_EPOCHS
    for lr in LEARNING_RATES:
        fold_maes, fold_epochs = [], []
        for ft, fv in folds:
            scaler = StandardScaler().fit(make_X(ft, emb, key_to_row))
            X_tr = scaler.transform(make_X(ft, emb, key_to_row)).astype(np.float32)
            X_val = scaler.transform(make_X(fv, emb, key_to_row)).astype(np.float32)
            _, vmae, vep = _train_head(
                X_tr, ft[sh.TARGET].to_numpy(), lr, MAX_EPOCHS,
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


def main(config_id: str = "C0") -> None:
    sh.set_config(config_id)

    df = sh.load_dataset()
    train, test = sh.train_test_split(df)
    emb, key_to_row = load_embeddings()
    missing = [(c, p) for c, p in zip(df["champion"], df["patch"])
               if (c, p) not in key_to_row]
    if missing:
        raise RuntimeError(f"{len(missing)} rows lack a cached embedding, "
                           f"e.g. {missing[:3]}. Rebuild with "
                           f"'python -m src.features.text_features --force'.")

    best_lr, epochs = select_lr(train, emb, key_to_row)
    sh.save_hyperparams("bert", {"learning_rate": best_lr, "epochs": epochs})

    sh.impute_lag(train, test)
    scaler = StandardScaler().fit(make_X(train, emb, key_to_row))
    X_tr = scaler.transform(make_X(train, emb, key_to_row)).astype(np.float32)
    X_te = scaler.transform(make_X(test, emb, key_to_row)).astype(np.float32)
    model, _, _ = _train_head(X_tr, train[sh.TARGET].to_numpy(), best_lr, epochs)
    y_pred = _predict(model, X_te)

    preds = test[["champion", "patch", "has_lag"]].copy()
    preds["y_true"] = test[sh.TARGET].to_numpy()
    preds["y_pred"] = y_pred

    sh.report("bert", preds)
    sh.save_predictions("bert", preds)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="C0", choices=list(config.ABLATION_CONFIGS))
    main(ap.parse_args().config)
