"""Block-target rolling origin with release/rework flags."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src import config
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src import config

from src.evaluation import compare as cmp
from src.evaluation import metrics as ev
from src.exploratory import release_flags as rf
from src.exploratory import rolling_block as rb
from src.exploratory import rolling_block_persistence as rbp
from src.models import _shared as sh

OUT = rf.OUT / "rolling_block"
FLAGS = list(rf.FLAG_COLS)
FITTED = ("ridge", "xgb")


def block_flags(b: int) -> pd.DataFrame:
    fl = rf.flags()
    out = []
    for start, patches in rb.blocks(b):
        g = fl[fl["patch"].isin(patches)].groupby("champion")[FLAGS].max()
        out.append(g.reset_index().assign(patch=start))
    return pd.concat(out, ignore_index=True)


def build(b: int) -> pd.DataFrame:
    df = rb.build_dataset(b)
    df = df.merge(block_flags(b), on=["champion", "patch"], how="left", validate="one_to_one")
    df[FLAGS] = df[FLAGS].fillna(0).astype(int)
    return df


def run_origin(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    base_feats = list(rb.FEATS)
    p0, _ = rb.run_origin(df, origin)
    try:
        rb.FEATS = base_feats + FLAGS
        p1, _ = rb.run_origin(df, origin)
    finally:
        rb.FEATS = base_feats
    p1 = p1[p1["model"].isin(FITTED)].assign(model=lambda d: d["model"] + "_flags")
    return pd.concat([p0, p1], ignore_index=True)


def wide(pooled: pd.DataFrame) -> pd.DataFrame:
    keys = ["champion", "patch", "block_end", "has_lag", "y_true"]
    w = pooled.pivot_table(index=keys, columns="model", values="y_pred").reset_index()
    w.columns.name = None
    return w


def run(b: int, target: str, max_origins: int | None = None) -> pd.DataFrame:
    sh.set_target(target)
    sh.set_config(rb.CONFIG)
    OUT.mkdir(parents=True, exist_ok=True)
    df = build(b)
    origins = rb.origins_for(df)[:max_origins]
    print(f"[flags] B={b} {target}: {len(origins)} origins; recent_release rows "
          f"{int(df.recent_release.sum())}, recent_rework rows {int(df.recent_rework.sum())}", flush=True)
    parts, t0 = [], time.time()
    for i, o in enumerate(origins, 1):
        parts.append(run_origin(df, o))
        print(f"  [{i}/{len(origins)}] {o} {time.time() - t0:.0f}s", flush=True)
    w = wide(pd.concat(parts, ignore_index=True))
    w = w.merge(df[["champion", "patch", "changed"] + FLAGS], on=["champion", "patch"], how="left")
    w["block_size"], w["target"] = b, target
    if max_origins is None:
        w.to_csv(OUT / f"preds_B{b}_{target}.csv", index=False)
    return w


def compare_rows(w: pd.DataFrame, b: int, target: str) -> tuple[list, list]:
    tests, desc = [], []
    pops = {"has_lag": w[w["has_lag"] == 1],
            "changed": w[(w["has_lag"] == 1) & (w["changed"] == 1)]}
    if target == "level":
        pops["all"] = w
    for pop, d in pops.items():
        for m in FITTED:
            tests.append({"block_size": b, "target": target, "population": pop,
                          "model": f"{m}_flags", "baseline": m,
                          "mae_model": ev.mae(d["y_true"], d[f"{m}_flags"]),
                          "mae_baseline": ev.mae(d["y_true"], d[m]),
                          **rb.cluster_paired(d, f"{m}_flags", m)})
    for flag in FLAGS:
        for pop, d in (("has_lag", w[w["has_lag"] == 1]), ("all", w)):
            if target == "delta" and pop == "all":
                continue
            d = d[d[flag] == 1]
            row = {"block_size": b, "target": target, "population": pop, "rows": flag, "n": len(d)}
            for m in FITTED:
                row[f"mae_{m}"] = ev.mae(d["y_true"], d[m]) if len(d) else np.nan
                row[f"mae_{m}_flags"] = ev.mae(d["y_true"], d[f"{m}_flags"]) if len(d) else np.nan
            desc.append(row)
    return tests, desc


def vs_block_persist(w: pd.DataFrame, b: int, target: str) -> list:
    if b == 1:
        return []
    lag = rbp.block_lag_table(b)
    d = w[w["has_lag"] == 1].merge(lag, on=["champion", "patch"], how="inner", validate="one_to_one")
    rows = []
    for m in ("xgb", "xgb_flags", "ridge", "ridge_flags", "shrink", "persistence"):
        rows.append({"block_size": b, "target": target, "model": m, "baseline": "block_persist",
                     **rb.cluster_paired(d.assign(block_persist=d[rbp.BLOCK_LAG]), m, "block_persist")})
    return rows


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    sizes = (int(argv[argv.index("--block") + 1]),) if "--block" in argv else (1, 3)
    targets = (argv[argv.index("--target") + 1],) if "--target" in argv else ("level", "delta")
    max_o = int(argv[argv.index("--max-origins") + 1]) if "--max-origins" in argv else None
    reuse = "--reuse" in argv
    tests, desc, bp, means = [], [], [], []
    for b in sizes:
        for t in targets:
            path = OUT / f"preds_B{b}_{t}.csv"
            if reuse and path.exists():
                w = pd.read_csv(path, dtype={"patch": str, "block_end": str})
            else:
                w = run(b, t, max_o)
            if max_o is not None:
                print(w.groupby("has_lag")[[c for c in w.columns if c in ("persistence", "shrink", "ridge", "ridge_flags", "xgb", "xgb_flags")]]
                      .apply(lambda g: (g.sub(w.loc[g.index, "y_true"], axis=0)).abs().mean()))
                continue
            sh.set_target(t)
            tt, dd = compare_rows(w, b, t)
            tests += tt
            desc += dd
            bp += vs_block_persist(w, b, t)
            h = w[w["has_lag"] == 1]
            means.append({"block_size": b, "target": t, "n_has_lag": len(h),
                          **{f"mae_{m}": ev.mae(h["y_true"], h[m]) for m in
                             ("persistence", "shrink", "ridge", "ridge_flags", "xgb", "xgb_flags")}})
    if max_o is not None:
        return 0
    tests = pd.DataFrame(tests)
    res = cmp.p_resolution()
    tests["m"], tests["p_resolution"] = len(tests), res
    tests["holm_significant"] = cmp.holm_bonferroni(tests["p_value"], resolution=res)
    pd.set_option("display.width", 250)
    for name, t in (("flag_tests", tests), ("flag_rows", pd.DataFrame(desc)),
                    ("vs_block_persist", pd.DataFrame(bp)), ("has_lag_mae", pd.DataFrame(means))):
        t.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n== {name}\n" + t.round(6).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
