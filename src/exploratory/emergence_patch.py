"""Emergence models with patch-note features and release/rework flags."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src import config
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src import config

from src.data import align
from src.evaluation import compare as cmp
from src.evaluation import emergence as em
from src.exploratory import release_flags as rf

OUT = rf.OUT
PATCH_COLS = ["changed", "n_changes_total", "n_buffs", "n_nerfs"]
BASE = ["pro_prev", "prior_freq"]
SPECS = {
    "A_prior": BASE,
    "B_patch": BASE + PATCH_COLS,
    "C_flags": BASE + list(rf.FLAG_COLS),
    "D_both": BASE + PATCH_COLS + list(rf.FLAG_COLS),
}
COMPARE = (("B_patch", "A_prior"), ("C_flags", "A_prior"), ("D_both", "A_prior"))
MIN_PRIOR = 12
N_BOOT = 2000


def patch_features() -> pd.DataFrame:
    """Per (champion, patch) change counts."""
    notes = pd.read_csv(config.PATCH_NOTE_FEATURES_CSV, dtype={"patch": str})
    stats = pd.read_csv(config.STAT_DELTAS_CSV, dtype={"patch": str})
    for f in (notes, stats):
        f["champion"] = f["champion"].map(align.normalise_champion_name)
    f = notes[["champion", "patch", "n_changes_total", "n_buffs", "n_nerfs"]].merge(
        stats[["champion", "patch", "n_stats_changed"]], on=["champion", "patch"], how="outer").fillna(0)
    f["changed"] = ((f["n_changes_total"] > 0) | (f["n_stats_changed"] > 0)).astype(int)
    return f[["champion", "patch"] + PATCH_COLS]


def build(cells: pd.DataFrame | None = None) -> pd.DataFrame:
    cells = em.build_cells() if cells is None else cells
    df = cells.merge(patch_features(), on=["champion", "patch"], how="left")
    df[PATCH_COLS] = df[PATCH_COLS].fillna(0)
    fl = rf.flags()[["champion", "patch", "released"] + list(rf.FLAG_COLS)]
    df = df.merge(fl, on=["champion", "patch"], how="left")
    if df["released"].isna().any():
        missing = sorted(df.loc[df["released"].isna(), "champion"].unique())
        raise ValueError(f"champions without Data Dragon flags: {missing}")
    df["released"] = df["released"].astype(int)
    for c in rf.FLAG_COLS:
        df[c] = df[c].astype(int)
    df["patch_idx"] = df["patch"].map(rf.ORDER)
    return df


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(train[feats])
    m = LogisticRegression(max_iter=2000).fit(sc.transform(train[feats]), train["pro_now"])
    return m.predict_proba(sc.transform(test[feats]))[:, 1]


def frozen(df: pd.DataFrame) -> pd.DataFrame:
    train = df[~df["patch"].isin(config.TEST_PATCHES)]
    test = df[df["patch"].isin(config.TEST_PATCHES)].copy()
    for name, feats in SPECS.items():
        test[name] = _fit_predict(train, test, feats)
    return test


def rolling(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for origin in config.ALL_PATCHES[MIN_PRIOR:]:
        i = rf.ORDER[origin]
        train = df[df["patch_idx"] < i]
        test = df[df["patch"] == origin].copy()
        if test.empty or train["pro_now"].nunique() < 2:
            continue
        for name, feats in SPECS.items():
            test[name] = _fit_predict(train, test, feats)
        out.append(test)
    return pd.concat(out, ignore_index=True)


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def _ap(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def boot_dauc(dead: pd.DataFrame, a: str, b: str, cluster: bool, seed=None) -> dict:
    """AUC(b) - AUC(a) on dead-zone rows; bootstrap rows or origin clusters."""
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    y = dead["pro_now"].to_numpy()
    sa, sb = dead[a].to_numpy(), dead[b].to_numpy()
    point = _auc(y, sb) - _auc(y, sa)
    if cluster:
        groups = [np.flatnonzero(dead["patch"].to_numpy() == g) for g in dead["patch"].unique()]
    diffs = []
    for _ in range(N_BOOT):
        if cluster:
            idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        else:
            idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(_auc(y[idx], sb[idx]) - _auc(y[idx], sa[idx]))
    d = np.array(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"delta_auc": point, "ci_lo": float(lo), "ci_hi": float(hi), "p_value": float(p),
            "n_boot_used": len(d)}


def score(pred: pd.DataFrame, evaluation: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dead = pred[pred["pro_prev"] == 0]
    y = dead["pro_now"].to_numpy()
    metrics = [{"evaluation": evaluation, "model": m, "n_dead_zone": len(dead),
                "n_emergences": int(y.sum()), "n_origins": int(dead["patch"].nunique()),
                "auc_all": _auc(pred["pro_now"].to_numpy(), pred[m].to_numpy()),
                "auc_dead_zone": _auc(y, dead[m].to_numpy()),
                "ap_dead_zone": _ap(y, dead[m].to_numpy())} for m in SPECS]
    tests = [{"evaluation": evaluation, "model": b, "baseline": a,
              **boot_dauc(dead, a, b, cluster=(evaluation == "rolling"))} for b, a in COMPARE]
    return pd.DataFrame(metrics), pd.DataFrame(tests)


def descriptive(pred: pd.DataFrame, evaluation: str) -> pd.DataFrame:
    dead = pred[pred["pro_prev"] == 0]
    rows = []
    for col in ("changed", "recent_release", "recent_rework"):
        for v in (1, 0):
            d = dead[dead[col] == v]
            rows.append({"evaluation": evaluation, "group": f"{col}={v}", "n": len(d),
                         "emergences": int(d["pro_now"].sum()),
                         "emergence_rate": float(d["pro_now"].mean()) if len(d) else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = em.build_cells()
    df_all = build(cells)
    df = df_all[df_all["released"] == 1].copy()
    print(f"cells {len(df_all)}; dropped pre-release {int((df_all.released == 0).sum())}; kept {len(df)}")

    uni = []
    for label, d in (("original", df_all), ("released_only", df)):
        t = frozen(d)
        dz = t[t["pro_prev"] == 0]
        uni.append({"universe": label, "n_dead_zone": len(dz), "n_emergences": int(dz.pro_now.sum()),
                    "auc_dead_zone_A": _auc(dz.pro_now.to_numpy(), dz["A_prior"].to_numpy())})
    uni = pd.DataFrame(uni)
    print(uni.to_string(index=False))

    fz, rl = frozen(df), rolling(df)
    m1, t1 = score(fz, "frozen")
    m2, t2 = score(rl, "rolling")
    metrics = pd.concat([m1, m2], ignore_index=True)
    tests = pd.concat([t1, t2], ignore_index=True)
    res = 2 / N_BOOT
    tests["m"] = len(tests)
    tests["p_resolution"] = res
    tests["holm_significant"] = cmp.holm_bonferroni(tests["p_value"], resolution=res)
    tests["verdict"] = np.where(tests["ci_lo"] > 0, "better", np.where(tests["ci_hi"] < 0, "worse", "tie"))
    desc = pd.concat([descriptive(fz, "frozen"), descriptive(rl, "rolling")], ignore_index=True)
    rel = rl[(rl["recent_release"] == 1) | (rl["recent_rework"] == 1)][
        ["champion", "patch", "pro_prev", "pro_now", "picks_now", "recent_release", "recent_rework"] + list(SPECS)]

    pd.set_option("display.width", 220)
    print(metrics.round(4).to_string(index=False))
    print(tests.round(5).to_string(index=False))
    print(desc.round(3).to_string(index=False))
    print(rel.round(3).to_string(index=False))
    uni.to_csv(OUT / "emergence_universe.csv", index=False)
    metrics.to_csv(OUT / "emergence_metrics.csv", index=False)
    tests.to_csv(OUT / "emergence_tests.csv", index=False)
    desc.to_csv(OUT / "emergence_descriptive.csv", index=False)
    rel.to_csv(OUT / "emergence_release_rework_cells.csv", index=False)
    fz.to_csv(OUT / "emergence_frozen_preds.csv", index=False)
    rl.to_csv(OUT / "emergence_rolling_preds.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
