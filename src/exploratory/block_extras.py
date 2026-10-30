"""Block-target rolling origin for C1X and CL, with tests against CL and shrink-only."""
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
from src.exploratory import extra_features as xf
from src.exploratory import rolling_block as rb
from src.exploratory import rolling_block_persistence as rbp
from src.models import _shared as sh

KEYS = ["champion", "patch"]
BLOCK_SUM = ["sem_power_ult", "sem_power_basic", "sem_power_passive", "sem_power_stats"]
BLOCK_MAX = ["release_avail"]
LAGGED = [c for c in xf.EXTRA_COLS if c not in BLOCK_SUM + BLOCK_MAX]


def out_dir() -> Path:
    return config.ROOT / "semv2" / "block_extras"


def block_extras(b: int, per_patch: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for start, patches in rb.blocks(b):
        at = per_patch[per_patch["patch"] == start][["champion"] + LAGGED]
        inb = per_patch[per_patch["patch"].isin(patches)].groupby("champion").agg(
            {**{c: "sum" for c in BLOCK_SUM}, **{c: "max" for c in BLOCK_MAX}}).reset_index()
        rows.append(at.merge(inb, on="champion", how="outer").assign(patch=start))
    out = pd.concat(rows, ignore_index=True)
    out[xf.EXTRA_COLS] = out[xf.EXTRA_COLS].fillna(0.0)
    return out


def build(b: int, per_patch: pd.DataFrame) -> pd.DataFrame:
    df = rb.build_dataset(b)
    df = df.merge(block_extras(b, per_patch), on=KEYS, how="left", validate="one_to_one")
    df[xf.EXTRA_COLS] = df[xf.EXTRA_COLS].fillna(0.0)
    return df


FEATS = {"C1X": lambda base: list(base) + list(xf.EXTRA_COLS),
         "C1L": lambda base: list(xf.NO_PATCH_EXTRAS)}


def run(b: int, target: str, cfg: str, per_patch: pd.DataFrame, max_origins: int | None = None) -> pd.DataFrame:
    sh.set_target(target)
    sh.set_config(rb.CONFIG)
    dest = out_dir() / f"preds_{cfg}_B{b}_{target}.csv"
    if dest.exists() and max_origins is None:
        return pd.read_csv(dest, dtype={"patch": str, "block_end": str})
    df = build(b, per_patch)
    origins = rb.origins_for(df)[:max_origins]
    base = list(rb.FEATS)
    parts, t0 = [], time.time()
    try:
        rb.FEATS = FEATS[cfg](base)
        for i, o in enumerate(origins, 1):
            p, _ = rb.run_origin(df, o)
            parts.append(p[p["model"].isin(["ridge", "xgb"])])
            print(f"  [{cfg} B={b} {target}] {i}/{len(origins)} {o} {time.time() - t0:.0f}s", flush=True)
    finally:
        rb.FEATS = base
        sh.set_target("level")
    p = pd.concat(parts, ignore_index=True)
    p = p.merge(df[KEYS + ["changed"]], on=KEYS, how="left")
    if max_origins is None:
        out_dir().mkdir(parents=True, exist_ok=True)
        p.to_csv(dest, index=False)
    return p


def _ref(b: int, target: str) -> pd.DataFrame:
    d = config.OUTPUTS / "predictions" / f"ROLLB-{'D-' if target == 'delta' else ''}{rb.CONFIG}-B{b}"
    frames = []
    for m in ("persistence", "shrink", "ridge", "xgb"):
        x = pd.read_csv(d / f"{m}_preds.csv", dtype={"patch": str, "block_end": str})
        frames.append(x.assign(model=m))
    return pd.concat(frames, ignore_index=True)


def wide(b: int, target: str, per_patch: pd.DataFrame) -> pd.DataFrame:
    ref = _ref(b, target)
    w = ref.pivot_table(index=KEYS + ["has_lag", "y_true"], columns="model", values="y_pred").reset_index()
    w.columns.name = None
    for cfg in ("C1X", "C1L"):
        p = run(b, target, cfg, per_patch)
        q = p.pivot_table(index=KEYS, columns="model", values="y_pred").reset_index()
        q.columns = KEYS + [f"{m}_{cfg}" for m in q.columns[2:]]
        w = w.merge(q, on=KEYS, how="inner", validate="one_to_one")
        if "changed" not in w:
            w = w.merge(p.drop_duplicates(KEYS)[KEYS + ["changed"]], on=KEYS, how="left")
    return w


def main(sizes=(1, 2, 3)) -> int:
    per_patch = xf.build_full()
    out_dir().mkdir(parents=True, exist_ok=True)
    per_patch.to_csv(out_dir() / "extra_features_full.csv", index=False)
    tests, desc, mae = [], [], []
    for b in sizes:
        lag = rbp.block_lag_table(b) if b > 1 else None
        for target in ("level", "delta"):
            w = wide(b, target, per_patch)
            pops = {"has_lag": w[w["has_lag"] == 1], "changed": w[(w["has_lag"] == 1) & (w["changed"] == 1)]}
            for pop, d in pops.items():
                mae.append({"block_size": b, "target": target, "population": pop, "n": len(d),
                            **{f"mae_{c}": ev.mae(d["y_true"], d[c]) for c in
                               ("persistence", "shrink", "ridge", "xgb", "ridge_C1X", "xgb_C1X",
                                "ridge_C1L", "xgb_C1L")}})
                for m in ("ridge", "xgb"):
                    for a, ref, fam in ((f"{m}_C1X", f"{m}_C1L", True), (f"{m}_C1L", "shrink", True),
                                        (f"{m}_C1X", m, False), (f"{m}_C1L", "persistence", False)):
                        r = {"block_size": b, "target": target, "population": pop, "model": a,
                             "baseline": ref, **rb.cluster_paired(d, a, ref)}
                        (tests if fam else desc).append(r)
                    if lag is not None:
                        dd = d.merge(lag, on=KEYS, how="inner", validate="one_to_one")
                        dd = dd.assign(block_persist=dd[rbp.BLOCK_LAG])
                        desc.append({"block_size": b, "target": target, "population": pop,
                                     "model": f"{m}_C1L", "baseline": "block_persist",
                                     **rb.cluster_paired(dd, f"{m}_C1L", "block_persist")})
    t = pd.DataFrame(tests)
    res = cmp.p_resolution()
    t["m"], t["p_resolution"] = len(t), res
    t["holm_significant"] = cmp.holm_bonferroni(t["p_value"], resolution=res)
    out_dir().mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir() / "tests.csv", index=False)
    pd.DataFrame(desc).to_csv(out_dir() / "descriptive.csv", index=False)
    pd.DataFrame(mae).to_csv(out_dir() / "mae.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(mae).round(5).to_string(index=False))
    print(t.round(6).to_string(index=False))
    print(pd.DataFrame(desc).round(6).to_string(index=False))
    return 0
