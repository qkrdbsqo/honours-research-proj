"""Compare block-target models with block_persist, which copies the previous block.

Usage:
    python -m src.exploratory.rolling_block_persistence
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    from src import config
    from src.evaluation import compare as cmp
    from src.evaluation import metrics as ev
    from src.exploratory import rolling_block as rb
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.evaluation import compare as cmp
    from src.evaluation import metrics as ev
    from src.exploratory import rolling_block as rb

BLOCK_LAG = "block_lag"
SOURCES = (("ROLLB-", "persistence", "persistence"),
           ("ROLLB-", "shrink", "shrink"),
           ("ROLLB-", "ridge", "ridge"),
           ("ROLLB-", "xgb", "xgb"),
           ("ROLLB-D-", "ridge", "ridge_delta"),
           ("ROLLB-D-", "xgb", "xgb_delta"))
OPTIONAL_SOURCES = (("ROLLB-", "bert", "bert"),
                    ("ROLLB-", "bert_ctrl", "bert_ctrl"),
                    ("ROLLB-", "bert_ft", "bert_ft"),
                    ("ROLLB-D-", "bert_ft", "bert_ft_delta"))
OPTIONAL_C3 = (("bert", "bert_C3"), ("bert_ctrl", "bert_ctrl_C3"),
               ("bert_ft", "bert_ft_C3"))
CONTROLS = ("bert_ctrl", "bert_ctrl_C3")
BASELINE = "block_persist"


def _pred_path(prefix: str, stem: str, b: int):
    return (config.OUTPUTS / "predictions" / f"{prefix}{rb.CONFIG}-B{b}"
            / f"{stem}_preds.csv")


def _c3_path(stem: str, b: int):
    return config.OUTPUTS / "predictions" / f"ROLLB-C3-B{b}" / f"{stem}_preds.csv"


def sources(b: int) -> tuple:
    """The required Path A runs, plus whichever Path B runs exist for B=b."""
    plain = tuple(s for s in OPTIONAL_SOURCES if _pred_path(s[0], s[1], b).exists())
    c3 = tuple(("ROLLB-C3/", stem, col) for stem, col in OPTIONAL_C3
               if _c3_path(stem, b).exists())
    return SOURCES + plain + c3


def comparisons(b: int) -> tuple:
    return tuple((m, BASELINE) for _, _, m in sources(b))
SUBSETS = ("has_lag", "changed_has_lag")
OUT = rb.OUT_TABLES


def previous_block(b: int) -> dict[str, str]:
    """test-block start -> start of the block that ends on the patch just before it."""
    ps = rb.scored_patches()
    end_to_start = {pats[-1]: start for start, pats in rb.blocks(b)}
    out = {}
    for start, _ in rb.blocks(b):
        i = ps.index(start)
        if i == 0 or config.season_of(ps[i - 1]) != config.season_of(start):
            continue
        prev_end = ps[i - 1]
        if prev_end in end_to_start:
            out[start] = end_to_start[prev_end]
    return out


def block_lag_table(b: int) -> pd.DataFrame:
    """(champion, patch=test-block start, block_lag)."""
    bp = rb.block_pwds(b)[["champion", "patch", rb.TARGET]]
    prev = previous_block(b)
    inv = {}
    for test_start, prev_start in prev.items():
        inv.setdefault(prev_start, []).append(test_start)
    rows = []
    for prev_start, tests in inv.items():
        g = bp[bp["patch"] == prev_start]
        for t in tests:
            rows.append(g.assign(patch=t))
    out = pd.concat(rows, ignore_index=True).rename(columns={rb.TARGET: BLOCK_LAG})
    return out[["champion", "patch", BLOCK_LAG]]


def wide_table(b: int) -> pd.DataFrame:
    wide = None
    for prefix, stem, col in sources(b):
        path = (_c3_path(stem, b) if prefix == "ROLLB-C3/"
                else _pred_path(prefix, stem, b))
        d = pd.read_csv(path, dtype={"patch": str, "block_end": str})
        d = d[d["has_lag"] == 1][["champion", "patch", "y_true", "y_pred"]].rename(columns={"y_pred": col})
        wide = d if wide is None else ev.align_prediction_rows(wide, d)
    lag = block_lag_table(b)
    wide = wide.merge(lag, on=["champion", "patch"], how="inner", validate="one_to_one")
    wide[BASELINE] = wide[BLOCK_LAG]
    feats = rb.build_dataset(b)[["champion", "patch", "changed"]]
    return wide.merge(feats, on=["champion", "patch"], how="left", validate="one_to_one")


def run(b: int):
    wide = wide_table(b)
    persist = pd.read_csv(config.OUTPUTS / "predictions" / f"ROLLB-{rb.CONFIG}-B{b}"
                          / "persistence_preds.csv")
    n_all = int((persist["has_lag"] == 1).sum())
    metrics, paired = [], []
    for subset in SUBSETS:
        d = wide if subset == "has_lag" else wide[wide["changed"] == 1]
        for m in [BASELINE] + [c for _, _, c in sources(b)]:
            metrics.append({"block_size": b, "subset": subset, "model": m, "n": len(d),
                            "n_origins": int(d["patch"].nunique()),
                            "mae": ev.mae(d["y_true"], d[m]),
                            "rho_per_block": ev.spearman_per_patch(d, y_pred_col=m)[0]})
        for a, ref in comparisons(b):
            paired.append({"block_size": b, "subset": subset, "model": a, "baseline": ref,
                           **rb.cluster_paired(d, a, ref)})
    print(f"[block_persistence] B={b}: {len(wide)} of {n_all} has-lag rows carry a block lag")
    return pd.DataFrame(metrics), pd.DataFrame(paired), len(wide), n_all


def main(sizes=rb.BLOCK_SIZES) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ms, ps, cov = [], [], []
    for b in sizes:
        m, p, n, n_all = run(b)
        ms.append(m)
        ps.append(p)
        cov.append({"block_size": b, "rows_with_block_lag": n, "has_lag_rows": n_all})
    metrics = pd.concat(ms, ignore_index=True)
    paired = pd.concat(ps, ignore_index=True)
    res = cmp.p_resolution()
    counted = (paired["block_size"] > 1) & (~paired["model"].isin(CONTROLS))
    paired["counted_in_family"] = counted
    paired["n_tests"] = int(counted.sum())
    paired["p_resolution"] = res
    paired["holm_significant"] = False
    paired.loc[counted, "holm_significant"] = cmp.holm_bonferroni(
        paired.loc[counted, "p_value"], resolution=res)
    ident = (paired["block_size"] == 1) & (paired["model"] == "persistence")
    paired.loc[ident, ["mae_diff", "ci_lo", "ci_hi"]] = 0.0
    paired.loc[ident, ["p_value", "verdict"]] = [1.0, "identical"]
    metrics.to_csv(OUT / "block_persistence_pooled.csv", index=False)
    paired.to_csv(OUT / "block_persistence_paired.csv", index=False)
    pd.DataFrame(cov).to_csv(OUT / "block_persistence_coverage.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(cov).to_string(index=False))
    print(metrics.pivot_table(index=["subset", "block_size"], columns="model", values="mae").to_string())
    print(paired.to_string(index=False))


if __name__ == "__main__":
    args = sys.argv[1:]
    main((int(args[args.index("--block") + 1]),) if "--block" in args else rb.BLOCK_SIZES)
