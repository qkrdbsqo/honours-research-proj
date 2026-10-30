"""Block-target rolling origin: predict the next B patches as one pooled patch.

Usage:
    python -m src.exploratory.rolling_block            # B = 1, 2, 3
    python -m src.exploratory.rolling_block --block 3
    python -m src.exploratory.rolling_block --target delta
"""
from __future__ import annotations

import functools
import sys
import time

import numpy as np
import pandas as pd

try:
    from src import config
    from src.data import align, load_oracle
    from src.evaluation import compare as cmp
    from src.evaluation import metrics as ev
    from src.evaluation import rolling_origin as ro
    from src.evaluation.patch_size import SHRINK_GRID
    from src.features import pwds
    from src.models import _shared as sh
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.data import align, load_oracle
    from src.evaluation import compare as cmp
    from src.evaluation import metrics as ev
    from src.evaluation import rolling_origin as ro
    from src.evaluation.patch_size import SHRINK_GRID
    from src.features import pwds
    from src.models import _shared as sh

CONFIG = "C1"
BLOCK_SIZES = (1, 2, 3)
TARGET = sh.TARGET
LAG = sh.LAG
FEATS = [c for c in config.features_for(CONFIG) if c not in (LAG, "has_lag")]
MODELS = ("persistence", "shrink", "ridge", "xgb")
COMPARISONS = (("ridge", "persistence"), ("xgb", "persistence"),
               ("shrink", "persistence"), ("ridge", "shrink"), ("xgb", "shrink"))
OUT_TABLES = config.OUTPUTS / "tables" / "rolling_block"

_MAX = {"max_buff_pct", "is_adjustment", "passive_changed", "q_changed",
        "w_changed", "e_changed", "r_changed", "base_stats_changed"}
_MIN = {"max_nerf_pct"}  # stored negative


def _pred_dir(b: int):
    return config.OUTPUTS / "predictions" / f"ROLLB-{sh.target_prefix()}{CONFIG}-B{b}"


def _suffix() -> str:
    return "_delta" if sh.active_target() == "delta" else ""


ORDER = {p: i for i, p in enumerate(config.ALL_PATCHES)}


@functools.lru_cache(maxsize=None)
def scored_patches() -> tuple[str, ...]:
    p = pd.read_csv(config.PWDS_CSV, dtype={"patch": str}, usecols=["patch"])
    return tuple(sorted(p["patch"].unique(), key=ORDER.get))


@functools.lru_cache(maxsize=None)
def blocks(b: int) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """(start, patches) for every run of B consecutive scored patches in one season."""
    ps = scored_patches()
    out = []
    for i in range(len(ps) - b + 1):
        run = ps[i:i + b]
        if (len({config.season_of(p) for p in run}) == 1
                and [ORDER[p] for p in run] == list(range(ORDER[run[0]], ORDER[run[0]] + b))):
            out.append((run[0], tuple(run)))
    return tuple(out)


@functools.lru_cache(maxsize=None)
def _oracle():
    df = load_oracle.load_raw(usecols=["gameid", "patch", "participantid", "position",
                                       "champion", "result", *pwds.BAN_COLS])
    return load_oracle.player_rows(df), load_oracle.team_rows(df)


def _score(players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """pwds.compute_pwds arithmetic, one row per (patch label, champion)."""
    games = pwds._games_per_patch(players)
    m = pwds._pick_stats(players).merge(pwds._ban_stats(teams),
                                        on=["patch", "champion"], how="left")
    m["ban_games"] = m["ban_games"].fillna(0).astype(int)
    m["total_games"] = m["patch"].map(games).astype(int)
    m = m[m["picks"] >= config.MIN_GAMES].copy()
    w_pr, w_br, w_wr = config.PWDS_WEIGHTS[config.DEFAULT_PWDS_WEIGHTS]
    m[TARGET] = (w_pr * m["picks"] / m["total_games"]
                 + w_br * m["ban_games"] / m["total_games"]
                 + w_wr * (m["wins"] / m["picks"] - config.WR_CENTRE))
    m["champion"] = m["champion"].map(align.normalise_champion_name)
    return m[["champion", "patch", "total_games", "picks", TARGET]]


def block_pwds(b: int) -> pd.DataFrame:
    """Block target: every block's patches relabelled to the block start, then scored."""
    players, teams = _oracle()
    pl, tm = [], []
    for start, patches in blocks(b):
        pl.append(players[players["patch"].isin(patches)].assign(patch=start))
        tm.append(teams[teams["patch"].isin(patches)].assign(patch=start))
    return _score(pd.concat(pl, ignore_index=True), pd.concat(tm, ignore_index=True))


@functools.lru_cache(maxsize=None)
def _per_patch_features() -> pd.DataFrame:
    deltas = pd.read_csv(config.STAT_DELTAS_CSV, dtype={"patch": str})
    notes = pd.read_csv(config.PATCH_NOTE_FEATURES_CSV, dtype={"patch": str})
    for f in (deltas, notes):
        f["champion"] = f["champion"].map(align.normalise_champion_name)
        f.drop(columns=[c for c in ("season",) if c in f.columns], inplace=True)
    f = deltas.merge(notes, on=["champion", "patch"], how="outer")
    return f[["champion", "patch"] + FEATS].fillna(0)


def block_features(b: int) -> pd.DataFrame:
    f = _per_patch_features()
    rules = {c: ("max" if c in _MAX else "min" if c in _MIN else "sum") for c in FEATS}
    out = []
    for start, patches in blocks(b):
        g = f[f["patch"].isin(patches)].groupby("champion")[FEATS].agg(rules)
        out.append(g.reset_index().assign(patch=start))
    out = pd.concat(out, ignore_index=True)
    n = out["n_changes_total"]
    out["buff_ratio"] = np.where(n > 0, out["n_buffs"] / n.where(n > 0, 1), 0.0)
    k = out["n_parseable_changes"]
    out["mean_pct_change"] = np.where(k > 0, out["sum_signed_pct"] / k.where(k > 0, 1), 0.0)
    out["is_adjustment"] = ((out["n_buffs"] > 0) & (out["n_nerfs"] > 0)).astype(int)
    return out


def build_dataset(b: int) -> pd.DataFrame:
    """One row per (champion, block): target, stale lag, block change features."""
    df = block_pwds(b).merge(block_features(b), on=["champion", "patch"], how="left")
    df[FEATS] = df[FEATS].fillna(0)
    ps = scored_patches()
    prev = {p: ps[i - 1] for i, p in enumerate(ps)
            if i > 0 and config.season_of(p) == config.season_of(ps[i - 1])}
    lag = pd.read_csv(config.PWDS_CSV, dtype={"patch": str})[["champion", "patch", TARGET]]
    lag["champion"] = lag["champion"].map(align.normalise_champion_name)
    lag = lag.rename(columns={TARGET: LAG})
    lag["patch"] = lag["patch"].map({v: k for k, v in prev.items()})
    df = df.merge(lag.dropna(subset=["patch"]), on=["champion", "patch"], how="left")
    df["has_lag"] = df[LAG].notna().astype(int)
    df["block_end"] = df["patch"].map({s: ps[-1] for s, ps in blocks(b)})
    df["changed"] = ((df["n_changes_total"] > 0) | (df["n_stats_changed"] > 0)).astype(int)
    df[sh.DELTA_TARGET] = df[TARGET] - df[LAG]
    df["patch_idx"] = df["patch"].map(ORDER)
    return df.sort_values(["patch_idx", "champion"]).reset_index(drop=True)


def training_rows(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    """Blocks that END strictly before the test block starts."""
    return df[df["block_end"].map(ORDER) < ORDER[origin]].copy()


def origins_for(df: pd.DataFrame) -> list[str]:
    starts = sorted(df["patch"].unique(), key=ORDER.get)
    if sh.active_target() == "delta":  # season-start blocks have no lag, so no delta rows
        starts = [s for s in starts if df.loc[df["patch"] == s, "has_lag"].any()]
    return [s for s in starts if training_rows(df, s)["patch"].nunique() >= ro.MIN_TRAIN_PATCHES]


def _inner_folds(train: pd.DataFrame):
    """ro._inner_folds with the block-aware cut: fold-train blocks END before the validation block starts."""
    starts = sorted(train["patch"].unique(), key=ORDER.get)
    folds = []
    for back in (0, ro.INNER_VAL_PATCHES):
        end = len(starts) - back
        start = end - ro.INNER_VAL_PATCHES
        if start <= 0:
            continue
        fv = train[train["patch"].isin(starts[start:end])].copy()
        ft = train[train["block_end"].map(ORDER) < ORDER[starts[start]]].copy()
        if ft.empty or fv.empty:
            continue
        med = float(ft[LAG].median())
        ft[LAG] = ft[LAG].fillna(med)
        fv[LAG] = fv[LAG].fillna(med)
        folds.append((ft, fv))
    return folds


def _shrink(train: pd.DataFrame, lag_values: np.ndarray) -> tuple[np.ndarray, float]:
    """patch_size.causal_shrink on this origin's own training blocks."""
    tr = train[train["has_lag"] == 1]
    mean = float(tr[LAG].mean())
    errs = [float(np.abs(tr[TARGET] - (mean + k * (tr[LAG] - mean))).mean())
            for k in SHRINK_GRID]
    k = float(SHRINK_GRID[int(np.argmin(errs))])
    return mean + k * (lag_values - mean), k


def run_origin(df: pd.DataFrame, origin: str) -> tuple[pd.DataFrame, dict]:
    """ro.run_origin with the block cut, plus the shrink-only baseline."""
    train = training_rows(df, origin)
    test = df[df["patch"] == origin].copy()
    feats = FEATS + [LAG, "has_lag"]
    delta = sh.active_target() == "delta"
    if delta:
        train = train[train["has_lag"] == 1].copy()
        test = test[test["has_lag"] == 1].copy()
    fit_col = sh.TARGET  # level or delta column, per sh.set_target

    folds = _inner_folds(train)
    alpha = ro._select_ridge(folds, feats)
    xgb_params = ro._select_xgb(folds, feats)

    med = float(train[LAG].median())
    train[LAG] = train[LAG].fillna(med)
    test_imp = test.copy()
    test_imp[LAG] = test_imp[LAG].fillna(med)

    base = test[["champion", "patch", "block_end", "has_lag"]].copy()
    base["y_true"] = test[TARGET].to_numpy()
    rows = []
    real = test["has_lag"].to_numpy() == 1
    p = base[real].copy()
    p["y_pred"] = test.loc[real, LAG].to_numpy()
    p["model"] = "persistence"
    rows.append(p)
    s = base[real].copy()
    s["y_pred"], k = _shrink(train, test.loc[real, LAG].to_numpy())
    s["model"] = "shrink"
    rows.append(s)
    back = test[LAG].to_numpy() if delta else 0.0  # reconstruct to the level scale
    r = base.copy()
    r["y_pred"] = ro._ridge_fit_predict(alpha, train, test_imp, feats) + back
    r["model"] = "ridge"
    rows.append(r)
    model = ro._xgb_model(xgb_params)
    model.fit(train[feats], train[fit_col])
    x = base.copy()
    x["y_pred"] = model.predict(test_imp[feats]) + back
    x["model"] = "xgb"
    rows.append(x)

    chosen = {"origin": origin, "n_train": len(train), "n_test": len(test),
              "n_train_patches": int(train["patch"].nunique()),
              "ridge_alpha": alpha, "shrink_k": k,
              **{f"xgb_{k_}": v for k_, v in xgb_params.items()}}
    return pd.concat(rows, ignore_index=True), chosen


def cluster_paired(j: pd.DataFrame, a: str, b: str, n_boot=None, seed=None) -> dict:
    """Paired MAE(a) - MAE(b), bootstrapping origins (the ``patch`` column)."""
    d = np.abs(j["y_true"] - j[a]) - np.abs(j["y_true"] - j[b])
    g = pd.DataFrame({"d": d.to_numpy(), "c": j["patch"].to_numpy()}).groupby("c")["d"].agg(["sum", "count"])
    dsum, cnt = g["sum"].to_numpy(), g["count"].to_numpy()
    n_boot = config.N_BOOTSTRAP if n_boot is None else n_boot
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    idx = rng.integers(0, len(dsum), size=(n_boot, len(dsum)))
    stats = dsum[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"n": int(cnt.sum()), "n_origins": len(dsum),
            "mae_diff": float(dsum.sum() / cnt.sum()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": ev.bootstrap_two_sided_pvalue(stats),
            "verdict": "better" if hi < 0 else "worse" if lo > 0 else "tie"}


def _slice(preds, subset, sparse):
    if subset == "has_lag":
        return preds[preds["has_lag"] == 1]
    if subset == "has_lag_dense":
        return preds[(preds["has_lag"] == 1) & (~preds["patch"].isin(sparse))]
    if subset == "changed_has_lag":
        return preds[(preds["has_lag"] == 1) & (preds["changed"] == 1)]
    raise KeyError(subset)


def run_block(b: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = build_dataset(b)
    origins = origins_for(df)
    games = df.groupby("patch")["total_games"].first().to_dict()
    print(f"[rolling_block] B={b}: {len(df)} rows over {df['patch'].nunique()} blocks; "
          f"{len(origins)} origins ({origins[0]} .. {origins[-1]})", flush=True)
    all_preds, chosen, per_origin = [], [], []
    t0 = time.time()
    for i, origin in enumerate(origins, 1):
        preds, params = run_origin(df, origin)
        preds = preds.merge(df[["champion", "patch", "changed"]], on=["champion", "patch"], how="left")
        all_preds.append(preds)
        chosen.append(params)
        row = {"block_size": b, "origin": origin, "block_end": preds["block_end"].iloc[0],
               "games": games[origin], "n_train": params["n_train"], "n_test": params["n_test"],
               "shrink_k": params["shrink_k"]}
        for m in MODELS:
            d = preds[(preds["model"] == m) & (preds["has_lag"] == 1)]
            row[f"mae_{m}"] = ev.mae(d["y_true"], d["y_pred"]) if len(d) else np.nan
        per_origin.append(row)
        print(f"  [{i:>2}/{len(origins)}] {origin}..{row['block_end']}: "
              f"persistence {row['mae_persistence']:.5f} shrink {row['mae_shrink']:.5f} "
              f"ridge {row['mae_ridge']:.5f} xgb {row['mae_xgb']:.5f} | {time.time() - t0:.0f}s",
              flush=True)

    pooled = pd.concat(all_preds, ignore_index=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    _pred_dir(b).mkdir(parents=True, exist_ok=True)
    cols = ["champion", "patch", "block_end", "y_true", "y_pred", "has_lag"]
    for m in MODELS:
        pooled[pooled["model"] == m][cols].to_csv(_pred_dir(b) / f"{m}_preds.csv", index=False)
    pd.DataFrame(per_origin).to_csv(OUT_TABLES / f"per_origin_B{b}{_suffix()}.csv", index=False)
    pd.DataFrame(chosen).to_csv(OUT_TABLES / f"hyperparameters_B{b}{_suffix()}.csv", index=False)

    sparse = {p for p, g in games.items() if g < config.SPARSE_PATCH_MIN_GAMES}
    metrics, paired = [], []
    for subset in ro.PAIRED_SUBSETS:
        wide = None
        for m in MODELS:
            d = _slice(pooled[pooled["model"] == m], subset, sparse)
            metrics.append({"block_size": b, "subset": subset, "model": m, "n": len(d),
                            "n_origins": int(d["patch"].nunique()),
                            "mae": ev.mae(d["y_true"], d["y_pred"]),
                            "rho_per_patch": ev.spearman_per_patch(d)[0]})
            d = d[["champion", "patch", "y_true", "y_pred"]].rename(columns={"y_pred": m})
            wide = d if wide is None else wide.merge(d, on=["champion", "patch", "y_true"])
        for a, ref in COMPARISONS:
            paired.append({"block_size": b, "subset": subset, "model": a, "baseline": ref,
                           **cluster_paired(wide, a, ref)})
    metrics = pd.DataFrame(metrics)
    paired = pd.DataFrame(paired)
    metrics.to_csv(OUT_TABLES / f"pooled_B{b}{_suffix()}.csv", index=False)
    paired.to_csv(OUT_TABLES / f"paired_B{b}{_suffix()}.csv", index=False)
    print(metrics.to_string(index=False))
    print(paired.to_string(index=False), flush=True)
    return metrics, paired


def main(sizes=BLOCK_SIZES, target: str = "level") -> None:
    sh.set_target(target)
    sh.set_config(CONFIG)
    metrics, paired = [], []
    for b in sizes:
        m, p = run_block(b)
        metrics.append(m)
        paired.append(p)
    paired = pd.concat(paired, ignore_index=True)
    res = cmp.p_resolution()
    paired["n_tests"] = len(paired)
    paired["p_resolution"] = res
    paired["holm_significant"] = cmp.holm_bonferroni(paired["p_value"], resolution=res)
    paired.to_csv(OUT_TABLES / f"paired_all{_suffix()}.csv", index=False)
    summary = (pd.concat(metrics, ignore_index=True)
               .pivot_table(index=["subset", "block_size"], columns="model", values="mae")
               .reset_index())
    summary.to_csv(OUT_TABLES / f"summary{_suffix()}.csv", index=False)
    print(f"\n[rolling_block] one Holm family over every block size (m={len(paired)})")
    print(paired.to_string(index=False))
    print("\n[rolling_block] has-lag MAE by block size")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    args = sys.argv[1:]
    sizes = (int(args[args.index("--block") + 1]),) if "--block" in args else BLOCK_SIZES
    main(sizes, args[args.index("--target") + 1] if "--target" in args else "level")
