"""Release-availability flag for each champion and patch."""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src import config
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src import config

from src.evaluation import compare as cmp
from src.evaluation import emergence as em
from src.exploratory import emergence_patch as ep
from src.exploratory import release_flags as rf
from src.exploratory import rolling_block_flags as rbf

OUT = rf.OUT / "followup_avail"
FLAG = "release_avail"


def avail_flags() -> pd.DataFrame:
    fl = rf.flags().copy()
    s = fl["patches_since_release"]
    fl[FLAG] = ((s >= 1) & (s <= rf.WINDOW)).fillna(False).astype(int)
    return fl[["champion", "patch", "released", FLAG]]


SPECS = {"A_prior": ep.BASE,
         "B_patch": ep.BASE + ep.PATCH_COLS,
         "C2_avail": ep.BASE + [FLAG],
         "D2_patch_avail": ep.BASE + ep.PATCH_COLS + [FLAG]}
COMPARE = (("C2_avail", "A_prior"), ("D2_patch_avail", "A_prior"))


def emergence() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = ep.build(em.build_cells())
    df = df.merge(avail_flags()[["champion", "patch", FLAG]], on=["champion", "patch"],
                  how="left", validate="one_to_one")
    df = df[df["released"] == 1].copy()
    old_specs, old_compare = ep.SPECS, ep.COMPARE
    try:
        ep.SPECS, ep.COMPARE = SPECS, COMPARE
        fz, rl = ep.frozen(df), ep.rolling(df)
        m1, t1 = ep.score(fz, "frozen")
        m2, t2 = ep.score(rl, "rolling")
    finally:
        ep.SPECS, ep.COMPARE = old_specs, old_compare
    tests = pd.concat([t1, t2], ignore_index=True)
    res = 2 / ep.N_BOOT
    tests["m"], tests["p_resolution"] = len(tests), res
    tests["holm_significant"] = cmp.holm_bonferroni(tests["p_value"], resolution=res)
    tests["verdict"] = np.where(tests["ci_lo"] > 0, "better", np.where(tests["ci_hi"] < 0, "worse", "tie"))
    cells = rl[(rl[FLAG] == 1) & (rl["pro_prev"] == 0)][
        ["champion", "patch", "pro_now", "picks_now"] + list(SPECS)]
    return pd.concat([m1, m2], ignore_index=True), tests, cells


def _block_flags(b: int) -> pd.DataFrame:
    fl = avail_flags()
    out = []
    for start, patches in rbf.rb.blocks(b):
        g = fl[fl["patch"].isin(patches)].groupby("champion")[[FLAG]].max()
        out.append(g.reset_index().assign(patch=start))
    return pd.concat(out, ignore_index=True)


def _run(b: int, target: str, max_origins: int | None = None) -> pd.DataFrame:
    """rolling_block_flags.run without its first log line."""
    import time
    rbf.sh.set_target(target)
    rbf.sh.set_config(rbf.rb.CONFIG)
    rbf.OUT.mkdir(parents=True, exist_ok=True)
    df = rbf.build(b)
    origins = rbf.rb.origins_for(df)[:max_origins]
    print(f"[avail] B={b} {target}: {len(origins)} origins; {FLAG} rows {int(df[FLAG].sum())}", flush=True)
    parts, t0 = [], time.time()
    for i, o in enumerate(origins, 1):
        parts.append(rbf.run_origin(df, o))
        print(f"  [{i}/{len(origins)}] {o} {time.time() - t0:.0f}s", flush=True)
    w = rbf.wide(pd.concat(parts, ignore_index=True))
    w = w.merge(df[["champion", "patch", "changed"] + rbf.FLAGS], on=["champion", "patch"], how="left")
    w["block_size"], w["target"] = b, target
    if max_origins is None:
        w.to_csv(rbf.OUT / f"preds_B{b}_{target}.csv", index=False)
    return w


@contextmanager
def configure_block():
    """Point rolling_block_flags at the new flag and the follow-up folder."""
    old = (rbf.FLAGS, rbf.OUT, rbf.block_flags, rbf.run)
    rbf.FLAGS, rbf.OUT = [FLAG], OUT / "rolling_block"
    rbf.block_flags, rbf.run = _block_flags, _run
    try:
        yield
    finally:
        rbf.FLAGS, rbf.OUT, rbf.block_flags, rbf.run = old


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    OUT.mkdir(parents=True, exist_ok=True)
    if "--block-only" not in argv:
        metrics, tests, cells = emergence()
        pd.set_option("display.width", 220)
        print(metrics.round(4).to_string(index=False))
        print(tests.round(5).to_string(index=False))
        print(cells.round(3).to_string(index=False))
        metrics.to_csv(OUT / "emergence_metrics.csv", index=False)
        tests.to_csv(OUT / "emergence_tests.csv", index=False)
        cells.to_csv(OUT / "emergence_avail_cells.csv", index=False)
    if "--emergence-only" not in argv:
        with configure_block():
            rest = [a for a in argv if a not in ("--block-only", "--emergence-only")]
            return rbf.main(rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
