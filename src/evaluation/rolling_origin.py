"""Rolling-origin evaluation: each patch is predicted once from all earlier patches.

Usage:
    python -m src.evaluation.rolling_origin --config C1
    python -m src.evaluation.rolling_origin --all
"""
from __future__ import annotations

import itertools
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    from src import config
    from src.evaluation import compare as cmp
    from src.evaluation import metrics as ev
    from src.models import _shared as sh
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.evaluation import compare as cmp
    from src.evaluation import metrics as ev
    from src.models import _shared as sh

MIN_TRAIN_PATCHES = 12
INNER_VAL_PATCHES = 3
MODELS = ("persistence", "ridge", "xgb")
PAIRED_SUBSETS = ("has_lag", "has_lag_dense", "changed_has_lag")
CONFIGS = ("C1", "C4")

OUT_TABLES = config.OUTPUTS / "tables" / "rolling_origin"


def _pred_dir(config_id: str):
    return config.OUTPUTS / "predictions" / f"ROLL-{config_id}"


def _patch_key(p: str) -> tuple[int, ...]:
    return tuple(int(x) for x in p.split("."))


def _inner_folds(train: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Two expanding walk-forward folds inside an origin's training window."""
    patches = sorted(train["patch"].unique(), key=_patch_key)
    folds = []
    for back in (0, INNER_VAL_PATCHES):
        end = len(patches) - back
        start = end - INNER_VAL_PATCHES
        if start <= 0:
            continue
        ft = train[train["patch"].isin(patches[:start])].copy()
        fv = train[train["patch"].isin(patches[start:end])].copy()
        if ft.empty or fv.empty:
            continue
        med = float(ft[sh.LAG].median())
        ft[sh.LAG] = ft[sh.LAG].fillna(med)
        fv[sh.LAG] = fv[sh.LAG].fillna(med)
        folds.append((ft, fv))
    return folds


def _ridge_fit_predict(alpha, ft, fv, feats):
    scaler = StandardScaler().fit(ft[feats])
    model = Ridge(alpha=alpha, random_state=config.RANDOM_SEED)
    model.fit(scaler.transform(ft[feats]), ft[sh.TARGET])
    return model.predict(scaler.transform(fv[feats]))


def _xgb_model(params):
    return XGBRegressor(objective="reg:squarederror",
                        random_state=config.RANDOM_SEED, n_jobs=-1,
                        verbosity=0, **params)


def _select_ridge(folds, feats) -> float:
    if not folds:
        return config.RIDGE_ALPHAS[0]
    best, best_mae = None, np.inf
    for alpha in config.RIDGE_ALPHAS:
        m = float(np.mean([ev.mae(fv[sh.TARGET].to_numpy(),
                                  _ridge_fit_predict(alpha, ft, fv, feats))
                           for ft, fv in folds]))
        if m < best_mae:
            best, best_mae = alpha, m
    return best


def _select_xgb(folds, feats) -> dict:
    keys = list(config.XGB_GRID)
    combos = list(itertools.product(*(config.XGB_GRID[k] for k in keys)))
    if not folds:
        return dict(zip(keys, combos[0]))
    best, best_mae = None, np.inf
    for values in combos:
        params = dict(zip(keys, values))
        maes = []
        for ft, fv in folds:
            model = _xgb_model(params)
            model.fit(ft[feats], ft[sh.TARGET])
            maes.append(ev.mae(fv[sh.TARGET].to_numpy(), model.predict(fv[feats])))
        m = float(np.mean(maes))
        if m < best_mae:
            best, best_mae = params, m
    return best


def training_rows(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    """Rows strictly before ``origin``."""
    order = {p: i for i, p in enumerate(config.ALL_PATCHES)}
    return df[df["patch"].map(order) < order[origin]].copy()


def run_origin(df: pd.DataFrame, origin: str) -> tuple[pd.DataFrame, dict]:
    """Fit every model on patches strictly before ``origin`` and predict it."""
    train = training_rows(df, origin)
    test = df[df["patch"] == origin].copy()
    feats = sh.features()

    folds = _inner_folds(train)
    alpha = _select_ridge(folds, feats)
    xgb_params = _select_xgb(folds, feats)

    med = float(train[sh.LAG].median())
    train[sh.LAG] = train[sh.LAG].fillna(med)
    test_imputed = test.copy()
    test_imputed[sh.LAG] = test_imputed[sh.LAG].fillna(med)

    base = test[["champion", "patch", "has_lag"]].copy()
    base["y_true"] = test[sh.TARGET].to_numpy()
    rows = []

    real = test["has_lag"].to_numpy() == 1
    p = base[real].copy()
    p["y_pred"] = test.loc[real, sh.LAG].to_numpy()
    p["model"] = "persistence"
    rows.append(p)

    r = base.copy()
    r["y_pred"] = _ridge_fit_predict(alpha, train, test_imputed, feats)
    r["model"] = "ridge"
    rows.append(r)

    model = _xgb_model(xgb_params)
    model.fit(train[feats], train[sh.TARGET])
    x = base.copy()
    x["y_pred"] = model.predict(test_imputed[feats])
    x["model"] = "xgb"
    rows.append(x)

    chosen = {"origin": origin, "n_train": len(train), "n_test": len(test),
              "n_train_patches": int(train["patch"].nunique()),
              "ridge_alpha": alpha,
              **{f"xgb_{k}": v for k, v in xgb_params.items()}}
    return pd.concat(rows, ignore_index=True), chosen


def origins_for(df: pd.DataFrame) -> list[str]:
    """Patches with at least MIN_TRAIN_PATCHES of history before them."""
    return sorted(df["patch"].unique(), key=_patch_key)[MIN_TRAIN_PATCHES:]


def add_changed_flag(df: pd.DataFrame) -> pd.DataFrame:
    """``changed`` = direct patch-note change or Data Dragon base-stat delta."""
    df = df.copy()
    df["changed"] = ((df["note_changed"].fillna(0) != 0)
                     | (df["stat_changed"].fillna(0) != 0)).astype(int)
    return df


def _slice(preds: pd.DataFrame, subset: str, sparse: set[str]) -> pd.DataFrame:
    if subset == "full":
        return preds
    if subset == "has_lag":
        return preds[preds["has_lag"] == 1]
    if subset == "has_lag_dense":
        return preds[(preds["has_lag"] == 1) & (~preds["patch"].isin(sparse))]
    if subset == "changed_has_lag":
        return preds[(preds["has_lag"] == 1) & (preds["changed"] == 1)]
    raise KeyError(subset)


def main(config_id: str = "C1") -> None:
    assert sh.active_target() == "level", (
        "rolling origin is a level-target analysis; it must never run under the "
        "delta target")
    sh.set_config(config_id)
    df = add_changed_flag(sh.load_dataset())

    origins = origins_for(df)
    print(f"[rolling_origin] config {config_id}: {len(origins)} origins "
          f"({origins[0]} .. {origins[-1]}), "
          f"min {MIN_TRAIN_PATCHES} training patches")

    all_preds, chosen, per_origin = [], [], []
    games = sh.games_by_patch()
    t0 = time.time()
    for i, origin in enumerate(origins, 1):
        preds, params = run_origin(df, origin)
        preds = preds.merge(df[["champion", "patch", "changed"]],
                            on=["champion", "patch"], how="left")
        all_preds.append(preds)
        chosen.append(params)
        row = {"origin": origin, "games": games.get(origin, 0),
               "n_train_patches": params["n_train_patches"],
               "n_train": params["n_train"], "n_test": params["n_test"]}
        for m in MODELS:
            d = preds[(preds["model"] == m) & (preds["has_lag"] == 1)]
            row[f"mae_{m}"] = ev.mae(d["y_true"], d["y_pred"]) if len(d) else np.nan
        per_origin.append(row)
        print(f"  [{i:>2}/{len(origins)}] {origin}: "
              f"train {params['n_train']:>4} rows / "
              f"{params['n_train_patches']:>2} patches -> "
              f"test {params['n_test']:>3} rows | "
              f"persistence {row['mae_persistence']:.5f} "
              f"ridge {row['mae_ridge']:.5f} xgb {row['mae_xgb']:.5f} "
              f"| {time.time() - t0:.0f}s", flush=True)

    pooled = pd.concat(all_preds, ignore_index=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    _pred_dir(config_id).mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        cols = ["champion", "patch", "y_true", "y_pred", "has_lag"]
        pooled[pooled["model"] == m][cols].to_csv(
            _pred_dir(config_id) / f"{m}_preds.csv", index=False)
    pd.DataFrame(per_origin).to_csv(
        OUT_TABLES / f"per_origin_{config_id}.csv", index=False)
    pd.DataFrame(chosen).to_csv(
        OUT_TABLES / f"hyperparameters_{config_id}.csv", index=False)

    sparse = {p for p, g in games.items() if g < config.SPARSE_PATCH_MIN_GAMES}
    report(pooled, config_id, sparse)


def report(pooled: pd.DataFrame, config_id: str, sparse: set[str]) -> None:
    """Pooled metrics + paired bootstrap against persistence, own Holm family."""
    rows = []
    for subset in ("full",) + PAIRED_SUBSETS:
        for m in MODELS:
            d = _slice(pooled[pooled["model"] == m], subset, sparse)
            if d.empty:
                continue
            point, lo, hi = ev.bootstrap_mae_ci(d["y_true"], d["y_pred"])
            rows.append({"config": config_id, "subset": subset, "model": m,
                         "n": len(d), "n_patches": int(d["patch"].nunique()),
                         "mae": point, "mae_ci_lo": lo, "mae_ci_hi": hi,
                         "rho_per_patch": ev.spearman_per_patch(d)[0]})
    metrics = pd.DataFrame(rows)
    metrics_path = OUT_TABLES / f"pooled_{config_id}.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"\n[rolling_origin] pooled metrics -> {metrics_path}")
    print(metrics.to_string(index=False))

    paired = []
    for subset in PAIRED_SUBSETS:
        base = _slice(pooled[pooled["model"] == "persistence"], subset, sparse)
        for m in ("ridge", "xgb"):
            cand = _slice(pooled[pooled["model"] == m], subset, sparse)
            j = base.merge(cand, on=["champion", "patch"], suffixes=("_p", "_m"))
            if j.empty:
                continue
            diff, lo, hi, stats = ev.paired_bootstrap_mae_diff(
                j["y_true_p"], j["y_pred_m"], j["y_pred_p"], return_stats=True)
            paired.append({"config": config_id, "subset": subset, "model": m,
                           "n": len(j), "mae_diff_vs_persistence": diff,
                           "ci_lo": lo, "ci_hi": hi,
                           "p_value": ev.bootstrap_two_sided_pvalue(stats),
                           "verdict": ("better" if hi < 0 else
                                       "worse" if lo > 0 else "tie")})
    pdf = pd.DataFrame(paired)
    res = cmp.p_resolution()
    pdf["n_tests"] = len(pdf)
    pdf["p_resolution"] = res
    pdf["holm_significant"] = cmp.holm_bonferroni(pdf["p_value"], resolution=res)
    paired_path = OUT_TABLES / f"paired_{config_id}.csv"
    pdf.to_csv(paired_path, index=False)
    print(f"\n[rolling_origin] paired vs persistence "
          f"(own Holm family, m={len(pdf)}) -> {paired_path}")
    print(pdf.to_string(index=False))


def decompose_against_frozen_window(config_id: str = "C1") -> pd.DataFrame:
    """Compare rolling and fixed-split results on the same 335 test rows."""
    test_patches = set(config.TEST_PATCHES)
    rp = pd.read_csv(_pred_dir(config_id) / "persistence_preds.csv", dtype={"patch": str})
    rx = pd.read_csv(_pred_dir(config_id) / "xgb_preds.csv", dtype={"patch": str})
    j = rp.merge(rx, on=["champion", "patch"], suffixes=("_p", "_x"))
    j = j[j["has_lag_p"] == 1]

    rows = []
    slices = [("frozen_test_window", j["patch"].isin(test_patches)),
              ("all_other_origins", ~j["patch"].isin(test_patches)),
              ("pooled", pd.Series(True, index=j.index))]
    for label, mask in slices:
        d = j[mask]
        yt = d["y_true_p"].to_numpy()
        diff, lo, hi, stats = ev.paired_bootstrap_mae_diff(
            yt, d["y_pred_x"], d["y_pred_p"], return_stats=True)
        rows.append({"config": config_id, "slice": label, "n": len(d),
                     "mae_persistence": ev.mae(yt, d["y_pred_p"]),
                     "mae_xgb": ev.mae(yt, d["y_pred_x"]),
                     "mae_diff_vs_persistence": diff, "ci_lo": lo, "ci_hi": hi,
                     "p_value": ev.bootstrap_two_sided_pvalue(stats)})
    out = pd.DataFrame(rows)
    fz = out[out["slice"] == "frozen_test_window"].iloc[0]
    pooled_diff = float(out[out["slice"] == "pooled"]["mae_diff_vs_persistence"].iloc[0])
    out["frozen_ci_contains_pooled_estimate"] = bool(
        fz["ci_lo"] <= pooled_diff <= fz["ci_hi"])
    return out


def pooled_family_holm(configs=CONFIGS) -> pd.DataFrame:
    """Holm correction of the rolling tests pooled with the fixed-split family."""
    main = pd.read_csv(config.OUTPUTS / "tables" / "paired_vs_persistence.csv")
    counted = main[main["counted_in_family"]]
    roll = pd.concat([pd.read_csv(OUT_TABLES / f"paired_{c}.csv") for c in configs],
                     ignore_index=True)
    cols = ["config", "model", "subset", "p_value", "verdict"]
    pool = pd.concat([counted[cols].assign(source="frozen"),
                      roll[cols].assign(source="rolling")], ignore_index=True)
    res = p_res = cmp.p_resolution()
    pool["n_tests"] = len(pool)
    pool["p_resolution"] = p_res
    pool["holm_significant"] = cmp.holm_bonferroni(pool["p_value"], resolution=res)
    return pool


def cross_checks(configs=CONFIGS) -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    dec = pd.concat([decompose_against_frozen_window(c) for c in configs],
                    ignore_index=True)
    dec.to_csv(OUT_TABLES / "frozen_window_decomposition.csv", index=False)
    print()
    print("[rolling_origin] OBJECTION 1 -- is it just the refitting?")
    print(dec.to_string(index=False))

    pool = pooled_family_holm(configs)
    pool.to_csv(OUT_TABLES / "pooled_family_holm.csv", index=False)
    sig = pool[pool["holm_significant"]]
    print()
    print(f"[rolling_origin] OBJECTION 2 -- pooled into the full family "
          f"(m={len(pool)})")
    print(sig.groupby(["source", "verdict"]).size().to_string())
    print("rolling rows:")
    print(pool[pool["source"] == "rolling"][
        ["config", "model", "subset", "p_value", "verdict",
         "holm_significant"]].to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="C1", choices=list(config.ABLATION_CONFIGS))
    ap.add_argument("--all", action="store_true",
                    help=f"run every config in {CONFIGS}")
    ap.add_argument("--cross-checks", action="store_true",
                    help="re-run only the two decisive cross-checks over "
                         "existing predictions (no refitting)")
    a = ap.parse_args()
    if a.cross_checks:
        cross_checks()
    else:
        for c in (CONFIGS if a.all else (a.config,)):
            main(c)
        cross_checks(CONFIGS if a.all else (a.config,))
