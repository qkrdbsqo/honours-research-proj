"""Shared helpers for the model scripts: configs, data loading, folds, imputation and output paths."""
from __future__ import annotations

import json
import sys
from typing import Iterator

import numpy as np
import pandas as pd

try:
    from src import config
    from src.evaluation import metrics as ev
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.evaluation import metrics as ev

LEVEL_TARGET = f"pwds_{config.DEFAULT_PWDS_WEIGHTS}"
DELTA_TARGET = f"pwds_delta_{config.DEFAULT_PWDS_WEIGHTS}"
LAG = f"lag_pwds_{config.DEFAULT_PWDS_WEIGHTS}"
TARGETS = ("level", "delta")

TARGET = LEVEL_TARGET
_ACTIVE_TARGET = "level"


def set_target(name: str) -> None:
    """Select the 'level' or 'delta' target."""
    if name not in TARGETS:
        raise KeyError(f"unknown target {name!r}, expected one of {TARGETS}")
    global TARGET, _ACTIVE_TARGET, _GAMES_CACHE
    _ACTIVE_TARGET = name
    TARGET = LEVEL_TARGET if name == "level" else DELTA_TARGET
    _GAMES_CACHE = None


def active_target() -> str:
    return _ACTIVE_TARGET


def target_prefix() -> str:
    """Output namespace prefix for the active target ('' or 'D-')."""
    return "D-" if _ACTIVE_TARGET == "delta" else ""

_ACTIVE = "C0"


def set_config(config_id: str) -> None:
    """Select the ablation config all subsequent calls operate under."""
    if config_id not in config.ABLATION_CONFIGS:
        raise KeyError(f"unknown ablation config {config_id!r}")
    global _ACTIVE, _GAMES_CACHE
    _ACTIVE = config_id
    _GAMES_CACHE = None       # game counts are per-config (seasons differ)


def active_config() -> str:
    return _ACTIVE


def cfg() -> dict:
    return config.ABLATION_CONFIGS[_ACTIVE]


def features() -> list[str]:
    """Feature list for the active config."""
    return config.features_for(_ACTIVE)


def load_dataset() -> pd.DataFrame:
    """Aligned modelling matrix, filtered to the active config's seasons."""
    df = pd.read_csv(config.ALIGNED_DATASET_CSV, dtype={"patch": str})
    df = df[df["season"].astype(str).isin(cfg()["seasons"])].copy()
    order = {p: i for i, p in enumerate(
        sorted(df["patch"].unique(),
               key=lambda s: tuple(int(x) for x in s.split("."))))}
    df["patch_idx"] = df["patch"].map(order)
    if _ACTIVE_TARGET == "delta":
        df = df[df["has_lag"] == 1].copy()
    return df.sort_values(["patch_idx", "champion"]).reset_index(drop=True)


def train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split by patch LABEL."""
    train = df[df["patch"].isin(config.TRAIN_PATCHES)].copy()
    test = df[df["patch"].isin(config.TEST_PATCHES)].copy()
    return train, test


def impute_lag(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Fill lag PWDS NaNs in place with the *training-set* median."""
    med = float(train[LAG].median())
    train[LAG] = train[LAG].fillna(med)
    test[LAG] = test[LAG].fillna(med)
    return med


def walk_forward_folds(train: pd.DataFrame,
                       fold_set: str | None = None
                       ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (fold_train, fold_val) for each walk-forward fold."""
    folds = config.FOLD_SETS[fold_set or cfg()["folds"]]
    if ((train["has_lag"] == 0) & train[LAG].notna()).any():
        raise ValueError("CV requires raw lag NaNs; impute only after model selection")
    order = {p: i for i, p in enumerate(config.ALL_PATCHES)}
    for cutoff, val_patches in folds:
        cut = order[cutoff]
        ft = train[train["patch"].map(order) <= cut].copy()
        fv = train[train["patch"].isin(val_patches)].copy()
        if ft.empty or fv.empty:
            continue
        med = float(ft[LAG].median())
        ft[LAG] = ft[LAG].fillna(med)
        fv[LAG] = fv[LAG].fillna(med)
        yield ft, fv


mae = ev.mae
spearman_pooled = ev.spearman_pooled
spearman_per_patch = ev.spearman_per_patch


_GAMES_CACHE: dict[str, int] | None = None


def games_by_patch() -> dict[str, int]:
    """Map patch -> professional games that patch (patch-level field)."""
    global _GAMES_CACHE
    if _GAMES_CACHE is None:
        df = load_dataset()
        _GAMES_CACHE = (df.groupby("patch")["total_games"].first()
                        .astype(int).to_dict())
    return _GAMES_CACHE


def report(name: str, preds: pd.DataFrame,
           subsets: tuple[str, ...] = ("full", "has_lag")) -> None:
    """Print MAE (with 95% bootstrap CI) + Spearman for a predictions frame."""
    games = games_by_patch()
    print(f"\n[{name}] test-set metrics")
    for subset in subsets:
        if subset == "has_lag":
            d = preds[preds["has_lag"] == 1]
            label = "has_lag subset"
        else:
            d = preds
            label = "full test"
        if d.empty:
            continue
        m, lo, hi = ev.bootstrap_mae_ci(d["y_true"].to_numpy(), d["y_pred"].to_numpy())
        rho_pp, per = spearman_per_patch(d)
        dense = [v for p, v in per.items()
                 if games.get(p, 0) >= config.SPARSE_PATCH_MIN_GAMES]
        rho_dense = float(np.mean(dense)) if dense else float("nan")
        rho_pool = spearman_pooled(d["y_true"].to_numpy(), d["y_pred"].to_numpy())
        print(f"  {label:<14} n={len(d):<4} MAE={m:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"rho(per-patch)={rho_pp:.4f}  rho(dense-only)={rho_dense:.4f}  "
              f"rho(pooled)={rho_pool:.4f}")

    _, per = spearman_per_patch(preds)
    print("  per-patch (full test):")
    for patch in sorted(per):
        n = int((preds["patch"] == patch).sum())
        g = games.get(patch, 0)
        flag = "  [SPARSE]" if g < config.SPARSE_PATCH_MIN_GAMES else ""
        print(f"    {patch}  n={n:<3} games={g:<4} rho={per[patch]:+.3f}{flag}")


def save_predictions(name: str, preds: pd.DataFrame) -> None:
    """Write predictions to outputs/predictions/[D-]<config>/<name>_preds.csv."""
    out_dir = config.PREDICTIONS / f"{target_prefix()}{_ACTIVE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["champion", "patch", "y_true", "y_pred", "has_lag"]
    out = out_dir / f"{name}_preds.csv"
    preds[cols].to_csv(out, index=False)
    print(f"  -> {out}")


def save_hyperparams(name: str, params: dict) -> None:
    """Record CV-selected hyperparameters, namespaced by target + config."""
    config.ensure_dirs()
    path = config.SELECTED_HYPERPARAMS_JSON
    data: dict = {}
    if path.exists():
        data = json.loads(path.read_text())
    key = f"{target_prefix()}{_ACTIVE}"
    data.setdefault(key, {})[name] = params
    path.write_text(json.dumps(data, indent=2))
    print(f"  selected hyperparameters -> {path} [{key}/{name}]")
