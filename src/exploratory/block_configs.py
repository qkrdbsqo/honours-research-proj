"""Block-target rolling origin for C2, C4, C2X and C4X."""
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

from src.data import align
from src.evaluation import compare as cmp
from src.evaluation import metrics as ev
from src.exploratory import block_extras as be
from src.exploratory import extra_features as xf
from src.exploratory import rolling_block as rb
from src.exploratory import rolling_block_persistence as rbp
from src.features import item_features as itf
from src.features import slot_features as sf
from src.models import _shared as sh

KEYS = ["champion", "patch"]
HELD = {"C3": "C3 block is not run: its build-usage routing reads usage inside the block"}
CLASS_COLS = list(config.ITEM_CLASS_FEATURE_COLS) + list(config.SYSTEM_FEATURE_COLS)
SLOT_COLS = list(config.SLOT_FEATURE_COLS)
BASE = {"C2": CLASS_COLS, "C4": SLOT_COLS}
CONFIGS = ("C2", "C4")


def out_dir() -> Path:
    return config.ROOT / "semv2" / "block_configs"


def check_config(cfg: str) -> str:
    base = cfg[:-1] if cfg.endswith("X") else cfg
    if base in HELD:
        raise SystemExit(f"refusing {cfg}: {HELD[base]}")
    if base not in BASE:
        raise SystemExit(f"unknown config {cfg}; block_configs supports {sorted(BASE)} and their X versions")
    return base


def class_block(b: int) -> pd.DataFrame:
    f = pd.read_csv(config.ITEM_FEATURES_CSV, dtype={"patch": str})
    f["champion"] = f["champion"].map(align.normalise_champion_name)
    out = []
    for start, patches in rb.blocks(b):
        g = f[f["patch"].isin(patches)].groupby("champion")[CLASS_COLS].sum()
        out.append(g.reset_index().assign(patch=start))
    return pd.concat(out, ignore_index=True)


def slot_block(b: int, rows: pd.DataFrame | None = None, shares: pd.DataFrame | None = None,
               notes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Slot pressure for block rows: P-1 shares fixed before the block, rival changes summed."""
    if rows is None:
        rows = rb.block_pwds(b)[KEYS]
    if shares is None:
        shares = sf.position_shares()
    if notes is None:
        notes = pd.read_csv(config.PATCH_NOTE_FEATURES_CSV, dtype={"patch": str})
    notes = notes.copy()
    notes["champion"] = notes["champion"].map(align.normalise_champion_name)
    changes = notes.set_index(KEYS)[sf._CHANGE_COLS].fillna(0.0)
    prev = itf._prev_patch_map()
    out = []
    for start, patches in rb.blocks(b):
        champs = rows.loc[rows["patch"] == start, "champion"].unique()
        q = prev.get(start)
        prev_shares = shares[shares["patch"] == q] if q is not None else shares.iloc[0:0]
        if prev_shares.empty:
            out += [{**sf._zero_row(c, start)} for c in champs]
            continue
        own_w = sf._own_position_weights(prev_shares)
        by_pos = {p: prev_shares[prev_shares["position"] == p].set_index("champion")["share"]
                  for p in prev_shares["position"].unique()}
        for champ in champs:
            if champ not in own_w:
                out.append(sf._zero_row(champ, start))
                continue
            acc = dict.fromkeys(["slot_n_changed", "slot_n_buffs", "slot_n_nerfs", "slot_net_signed_pct"], 0.0)
            max_share = 0.0
            for pos, weight in own_w[champ].items():
                rivals = by_pos[pos].drop(index=champ, errors="ignore")
                total = float(rivals.sum())
                if total <= 0:
                    continue
                rivals = rivals / total
                for rival, share in rivals.items():
                    for p in patches:
                        key = (rival, p)
                        if key not in changes.index:
                            continue
                        r = changes.loc[key]
                        w = weight * float(share)
                        changed = 1.0 if float(r["n_changes_total"]) > 0 else 0.0
                        acc["slot_n_changed"] += w * changed
                        acc["slot_n_buffs"] += w * float(r["n_buffs"])
                        acc["slot_n_nerfs"] += w * float(r["n_nerfs"])
                        acc["slot_net_signed_pct"] += w * float(r["sum_signed_pct"])
                        if changed:
                            max_share = max(max_share, float(share))
            out.append({"champion": champ, "patch": start, **acc,
                        "slot_max_rival_share_changed": max_share, "has_slot_lag": 1})
    return pd.DataFrame(out, columns=KEYS + SLOT_COLS)


def build(cfg: str, b: int, per_patch: pd.DataFrame | None) -> pd.DataFrame:
    base = check_config(cfg)
    df = rb.build_dataset(b)
    extra = class_block(b) if base == "C2" else slot_block(b, rows=df[KEYS])
    df = df.merge(extra, on=KEYS, how="left", validate="one_to_one")
    df[BASE[base]] = df[BASE[base]].fillna(0.0)
    if cfg.endswith("X"):
        df = df.merge(be.block_extras(b, per_patch), on=KEYS, how="left", validate="one_to_one")
        df[xf.EXTRA_COLS] = df[xf.EXTRA_COLS].fillna(0.0)
    return df


def feats(cfg: str, c1_feats: list[str]) -> list[str]:
    base = check_config(cfg)
    f = list(c1_feats) + list(BASE[base])
    return f + list(xf.EXTRA_COLS) if cfg.endswith("X") else f


def run(cfg: str, b: int, target: str, per_patch: pd.DataFrame | None,
        max_origins: int | None = None) -> pd.DataFrame:
    check_config(cfg)
    sh.set_target(target)
    sh.set_config(rb.CONFIG)
    dest = out_dir() / f"preds_{cfg}_B{b}_{target}.csv"
    if dest.exists() and max_origins is None:
        return pd.read_csv(dest, dtype={"patch": str, "block_end": str})
    df = build(cfg, b, per_patch)
    origins = rb.origins_for(df)[:max_origins]
    c1 = list(rb.FEATS)
    parts, t0 = [], time.time()
    try:
        rb.FEATS = feats(cfg, c1)
        for i, o in enumerate(origins, 1):
            p, _ = rb.run_origin(df, o)
            parts.append(p[p["model"].isin(["ridge", "xgb"])])
            print(f"  [{cfg} B={b} {target}] {i}/{len(origins)} {o} {time.time() - t0:.0f}s", flush=True)
    finally:
        rb.FEATS = c1
        sh.set_target("level")
    p = pd.concat(parts, ignore_index=True)
    p = p.merge(df[KEYS + ["changed"]], on=KEYS, how="left")
    if max_origins is None:
        out_dir().mkdir(parents=True, exist_ok=True)
        p.to_csv(dest, index=False)
    return p


def _wide(p: pd.DataFrame, tag: str) -> pd.DataFrame:
    q = p.pivot_table(index=KEYS, columns="model", values="y_pred").reset_index()
    q.columns = KEYS + [f"{m}_{tag}" for m in q.columns[2:]]
    return q


def main(configs=CONFIGS, sizes=(1, 2, 3), smoke: bool = False) -> int:
    for c in configs:
        check_config(c)
    full = out_dir().parent / "block_extras" / "extra_features_full.csv"
    per_patch = (pd.read_csv(full, dtype={"patch": str}) if full.exists()
                 else xf.build_full())
    if smoke:
        for c in configs:
            for cfg in (c, c + "X"):
                p = run(cfg, 1, "level", per_patch, max_origins=2)
                print(f"smoke {cfg}: {len(p)} prediction rows, features {len(feats(cfg, rb.FEATS))}")
        return 0
    tests, desc, mae = [], [], []
    for b in sizes:
        lag = rbp.block_lag_table(b) if b > 1 else None
        for target in ("level", "delta"):
            w = be._ref(b, target).pivot_table(index=KEYS + ["has_lag", "y_true"], columns="model",
                                               values="y_pred").reset_index()
            w.columns.name = None
            c1l = be.run(b, target, "C1L", per_patch)
            w = w.merge(_wide(c1l, "C1L"), on=KEYS, how="inner", validate="one_to_one")
            w = w.merge(c1l.drop_duplicates(KEYS)[KEYS + ["changed"]], on=KEYS, how="left")
            for c in configs:
                for cfg in (c, c + "X"):
                    w = w.merge(_wide(run(cfg, b, target, per_patch), cfg), on=KEYS, how="inner",
                                validate="one_to_one")
            pops = {"has_lag": w[w["has_lag"] == 1], "changed": w[(w["has_lag"] == 1) & (w["changed"] == 1)]}
            for pop, d in pops.items():
                cols = ["persistence", "shrink", "ridge", "xgb", "ridge_C1L", "xgb_C1L"] + \
                       [f"{m}_{cfg}" for c in configs for cfg in (c, c + "X") for m in ("ridge", "xgb")]
                mae.append({"block_size": b, "target": target, "population": pop, "n": len(d),
                            **{f"mae_{k}": ev.mae(d["y_true"], d[k]) for k in cols}})
                for c in configs:
                    for m in ("ridge", "xgb"):
                        tests.append({"block_size": b, "target": target, "population": pop,
                                      "model": f"{m}_{c}X", "baseline": f"{m}_{c}",
                                      **rb.cluster_paired(d, f"{m}_{c}X", f"{m}_{c}")})
                        if pop != "has_lag":
                            continue
                        for a, ref in ((f"{m}_{c}", m), (f"{m}_{c}X", f"{m}_C1L")):
                            desc.append({"block_size": b, "target": target, "population": pop,
                                         "model": a, "baseline": ref, **rb.cluster_paired(d, a, ref)})
                        if lag is not None:
                            dd = d.merge(lag, on=KEYS, how="inner", validate="one_to_one")
                            dd = dd.assign(block_persist=dd[rbp.BLOCK_LAG])
                            for a in (f"{m}_{c}", f"{m}_{c}X"):
                                desc.append({"block_size": b, "target": target, "population": pop,
                                             "model": a, "baseline": "block_persist",
                                             **rb.cluster_paired(dd, a, "block_persist")})
    t = pd.DataFrame(tests)
    assert len(t) == 24 * len(configs), len(t)
    res = cmp.p_resolution()
    t["m"], t["p_resolution"] = len(t), res
    t["holm_significant"] = cmp.holm_bonferroni(t["p_value"], resolution=res)
    out_dir().mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir() / "tests.csv", index=False)
    pd.DataFrame(desc).to_csv(out_dir() / "descriptive.csv", index=False)
    pd.DataFrame(mae).to_csv(out_dir() / "mae.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(mae).round(5).to_string(index=False))
    print(t.round(6).to_string(index=False))
    print(pd.DataFrame(desc).round(6).to_string(index=False))
    return 0
