"""Cross-model comparison: bootstrap intervals, paired tests against persistence and Holm-corrected families.

Usage:
    python -m src.evaluation.compare                  # level target
    python -m src.evaluation.compare --target delta   # difference target
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    from src import config
    from src.evaluation import metrics as ev
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.evaluation import metrics as ev

MODELS = ["persistence", "ridge", "xgb", "bert", "bert_ctrl", "bert_ft",
          "bert_ft_fallback"]
TEXT_MODELS = ("bert", "bert_ctrl", "bert_ft", "bert_ft_fallback")

PAIRED_CONTROLS = ("bert_ctrl",)
PAIRED_MODELS = tuple(m for m in MODELS
                      if m != "persistence" and m not in PAIRED_CONTROLS)
KEYS = ["champion", "patch"]

SUBSETS = ("full", "has_lag", "has_lag_dense", "changed", "changed_has_lag",
           "item_affected", "any_affected")


def load_all(config_id: str, prefix: str = "") -> dict[str, pd.DataFrame]:
    """Load predictions for one ablation config, merged with the eval flags."""
    from src.models.run_ablation import BERT_ALIASES

    runner = "run_delta" if prefix == "D-" else "run_ablation"
    out = {}
    flags = changed_flags()
    for m in MODELS:
        src = (BERT_ALIASES.get(config_id, config_id)
               if m in TEXT_MODELS else config_id)
        p = config.PREDICTIONS / f"{prefix}{src}" / f"{m}_preds.csv"
        if p.exists():
            df = pd.read_csv(p, dtype={"patch": str})
            out[m] = df.merge(flags, on=KEYS, how="left", validate="one_to_one")
            for c in ("changed", "item_affected", "any_affected"):
                out[m][c] = out[m][c].fillna(0).astype(int)
        else:
            print(f"  (missing {p} - run "
                  f"'python -m src.models.{runner} --config {src}')")
    return out


def changed_flags() -> pd.DataFrame:
    """(champion, patch) -> changed / item_affected / any_affected flags."""
    df = pd.read_csv(config.ALIGNED_DATASET_CSV, dtype={"patch": str})
    changed = ((df["note_changed"].fillna(0) != 0)
               | (df["stat_changed"].fillna(0) != 0)).astype(int)
    if "item_affected" not in df.columns:
        raise KeyError(
            "aligned_dataset.csv is missing the 'item_affected' column, which "
            "the item_affected/any_affected subsets and dead_zone_by_config() "
            "require. Check the pipeline stage that adds it (item/rune feature "
            "merge) before re-running compare.py.")
    item_aff = df["item_affected"].fillna(0).astype(int)
    return pd.DataFrame({
        "champion": df["champion"], "patch": df["patch"],
        "changed": changed, "item_affected": item_aff,
        "any_affected": ((changed | item_aff) > 0).astype(int),
    })


def _slice(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Row subset for evaluation."""
    if subset == "has_lag":
        return df[df["has_lag"] == 1]
    if subset == "changed":
        return df[df["changed"] == 1]
    if subset == "changed_has_lag":
        return df[(df["changed"] == 1) & (df["has_lag"] == 1)]
    if subset == "has_lag_dense":
        from src.models import _shared as sh
        games = sh.games_by_patch()
        dense = [p for p in df["patch"].unique()
                 if games.get(p, 0) >= config.SPARSE_PATCH_MIN_GAMES]
        return df[(df["has_lag"] == 1) & (df["patch"].isin(dense))]
    if subset == "item_affected":
        return df[(df["item_affected"] == 1) & (df["has_lag"] == 1)]
    if subset == "any_affected":
        return df[(df["any_affected"] == 1) & (df["has_lag"] == 1)]
    return df  # "full"


def per_model_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """MAE + per-patch rho (each with 95% CI) for every model x subset."""
    rows = []
    for name, df in preds.items():
        for subset in SUBSETS:
            if name == "persistence" and subset in ("full", "changed"):
                continue
            d = _slice(df, subset)
            if len(d) < 2:
                continue
            mae_pt, mae_lo, mae_hi = ev.bootstrap_mae_ci(d["y_true"], d["y_pred"])
            rho_pt, rho_lo, rho_hi = ev.bootstrap_per_patch_spearman_ci(d)
            rows.append({
                "model": name, "subset": subset, "n": len(d),
                "mae": mae_pt, "mae_lo": mae_lo, "mae_hi": mae_hi,
                "rho_per_patch": rho_pt, "rho_lo": rho_lo, "rho_hi": rho_hi,
            })
    return pd.DataFrame(rows)


def paired_vs_persistence(preds: dict[str, pd.DataFrame],
                          subset: str = "has_lag") -> pd.DataFrame:
    """Paired MAE(model) - MAE(persistence) with 95% CI on a given subset."""
    if "persistence" not in preds:
        return pd.DataFrame()
    base = (_slice(preds["persistence"], subset)[KEYS + ["y_true", "y_pred"]]
            .rename(columns={"y_pred": "persist"}))
    rows = []
    for name in PAIRED_MODELS:
        if name not in preds:
            continue
        m = ev.align_prediction_rows(base, preds[name][KEYS + ["y_true", "y_pred"]])
        if len(m) < 2:
            continue
        diff, lo, hi, stats = ev.paired_bootstrap_mae_diff(
            m["y_true"], m["y_pred"], m["persist"], return_stats=True)
        p_value = ev.bootstrap_two_sided_pvalue(stats)
        significant = not (lo < 0 < hi)
        rows.append({
            "model": name, "subset": subset, "n": len(m),
            "mae_diff_vs_persistence": diff, "ci_lo": lo, "ci_hi": hi,
            "verdict": ("better" if hi < 0 else "worse" if lo > 0 else "tie")
            if significant else "tie",
            "p_value": p_value,
        })
    return pd.DataFrame(rows)


def p_resolution(n_boot: int | None = None) -> float:
    """Smallest non-zero gap between two attainable bootstrap p-values."""
    return 2.0 / (config.N_BOOTSTRAP if n_boot is None else n_boot)


def holm_bonferroni(p_values, alpha: float = 0.05, resolution: float = 0.0):
    """Holm-Bonferroni step-down correction, refusing unresolvable decisions."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    sig = np.zeros(m, dtype=bool)
    order = np.argsort(p)
    for rank, idx in enumerate(order):  # rank is 0-indexed -> (m - rank) is (m - i + 1)
        if p[idx] + resolution <= alpha / (m - rank):
            sig[idx] = True
        else:
            break  # step-down: first failure stops further rejections
    return sig


def subset_sizes() -> pd.DataFrame:
    """Test-set subset sizes per config, from the SUBSETS/_slice() flags."""
    rows = []
    for cid in config.ABLATION_CONFIGS:
        preds = load_all(cid)
        if "ridge" not in preds:
            continue
        d = preds["ridge"]
        for subset in SUBSETS:
            rows.append({"config": cid, "subset": subset,
                         "n": len(_slice(d, subset))})
    return pd.DataFrame(rows)


def dead_zone_by_config() -> pd.DataFrame:
    """Test rows carrying a change signal available to each config's features."""
    df = pd.read_csv(config.ALIGNED_DATASET_CSV, dtype={"patch": str})
    test = df[df["patch"].isin(config.TEST_PATCHES)].copy()
    if "item_affected" not in test.columns:
        raise KeyError(
            "aligned_dataset.csv is missing the 'item_affected' column, "
            "required by dead_zone_by_config() for C3's signal definition.")
    path_a_signal = (test["note_changed"] > 0) | (test["stat_changed"] > 0)
    item_signal = test["item_affected"].fillna(0).astype(int) == 1
    has_lag = test["has_lag"] == 1

    rows = []
    for cid in config.ABLATION_CONFIGS:
        signal = (path_a_signal | item_signal) if cid == "C3" else path_a_signal
        n_total = len(test)
        n_signal = int(signal.sum())
        rows.append({
            "config": cid,
            "n_test_rows": n_total,
            "n_with_change_signal": n_signal,
            "pct_with_change_signal": round(100 * n_signal / n_total, 1),
            "n_with_change_signal_has_lag": int((signal & has_lag).sum()),
        })
    return pd.DataFrame(rows)


PAIRED_SUBSETS = ("has_lag", "has_lag_dense", "changed_has_lag",
                  "item_affected", "any_affected")
PAIRED_HEADERS = {
    "has_lag": "has_lag subset",
    "has_lag_dense": f"has_lag, dense patches only (>= {config.SPARSE_PATCH_MIN_GAMES} games)",
    "changed_has_lag": "changed-champion subset",
    "item_affected": "item/rune-affected subset (champion-routed)",
    "any_affected": "any-affected subset (changed | item_affected)",
}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=("level", "delta"), default="level")
    args = ap.parse_args()

    config.ensure_dirs()
    if args.target == "delta":
        from src.models import _shared as sh
        sh.set_target("delta")
        cfg_ids = list(config.DELTA_CONFIGS)
        prefix = "D-"
        out_tbl, out_paired = config.DELTA_COMPARISON_CSV, config.DELTA_PAIRED_CSV
    else:
        cfg_ids = list(config.ABLATION_CONFIGS)
        prefix = ""
        out_tbl, out_paired = config.MODEL_COMPARISON_CSV, config.PAIRED_COMPARISON_CSV

    all_tbl = []
    all_paired = []
    for cid in cfg_ids:
        print(f"\n########## Config {cid} "
              f"({config.ABLATION_CONFIGS[cid]}) ##########")
        preds = load_all(cid, prefix)
        if "persistence" not in preds:
            print(f"  Nothing to compare for {cid} (persistence predictions missing).")
            continue

        tbl = per_model_table(preds)
        tbl.insert(0, "config", cid)
        all_tbl.append(tbl)

        print(f"\n=== Per-model metrics (95% bootstrap CI) - {cid} ===")
        for _, r in tbl.iterrows():
            print(f"  {r['model']:<12} {r['subset']:<14} n={int(r['n']):<4} "
                  f"MAE={r['mae']:.4f} [{r['mae_lo']:.4f}, {r['mae_hi']:.4f}]   "
                  f"rho={r['rho_per_patch']:.4f} [{r['rho_lo']:.4f}, {r['rho_hi']:.4f}]")

        for subset in PAIRED_SUBSETS:
            header = PAIRED_HEADERS[subset]
            print(f"\n=== Paired MAE difference vs persistence "
                  f"({header}) - {cid} ===")
            print("    negative = model beats 'nothing changes'; CI crossing 0 = a tie")
            pv = paired_vs_persistence(preds, subset)
            if pv.empty:
                print("    (no rows)")
                continue
            pv.insert(0, "config", cid)
            all_paired.append(pv)
            for _, r in pv.iterrows():
                tag = "" if r["verdict"] == "tie" else f"  *{r['verdict'].upper()}*"
                print(f"  {r['model']:<12} n={int(r['n']):<4} "
                      f"dMAE={r['mae_diff_vs_persistence']:+.4f} "
                      f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  ({r['verdict']}){tag}")

    if all_tbl:
        pd.concat(all_tbl, ignore_index=True).to_csv(out_tbl, index=False)

    if all_paired:
        paired_tbl = pd.concat(all_paired, ignore_index=True)
        from src.models.run_ablation import BERT_ALIASES
        paired_tbl["aliases"] = [
            BERT_ALIASES.get(c, c) if m in TEXT_MODELS else c
            for c, m in zip(paired_tbl["config"], paired_tbl["model"])]
        paired_tbl["counted_in_family"] = paired_tbl["aliases"] == paired_tbl["config"]

        counted = paired_tbl["counted_in_family"].to_numpy()
        n_tests = int(counted.sum())
        paired_tbl["n_tests"] = n_tests
        res = p_resolution()
        paired_tbl["p_resolution"] = res
        paired_tbl["holm_significant"] = False
        paired_tbl.loc[counted, "holm_significant"] = holm_bonferroni(
            paired_tbl.loc[counted, "p_value"].to_numpy(), resolution=res)

        thr = np.full(len(paired_tbl), np.nan)
        cidx = np.flatnonzero(counted)
        cp = paired_tbl.loc[counted, "p_value"].to_numpy()
        for rank, j in enumerate(np.argsort(cp)):
            thr[cidx[j]] = 0.05 / (n_tests - rank)
        paired_tbl["holm_threshold"] = thr
        paired_tbl["holm_resolvable"] = (
            (paired_tbl["p_value"] + res <= paired_tbl["holm_threshold"])
            | (paired_tbl["p_value"] - res > paired_tbl["holm_threshold"]))

        by_key = {(r["config"], r["model"], r["subset"]): r["holm_significant"]
                  for _, r in paired_tbl[counted].iterrows()}
        for i, r in paired_tbl[~counted].iterrows():
            paired_tbl.at[i, "holm_significant"] = by_key.get(
                (r["aliases"], r["model"], r["subset"]), False)

        paired_tbl["note"] = ""
        non_tie = paired_tbl["verdict"] != "tie"
        paired_tbl.loc[non_tie, "note"] = (
            "UNCORRECTED single-test verdict (multiple comparisons not "
            "adjusted here) -- see holm_significant")
        paired_tbl.loc[~counted, "note"] = (
            paired_tbl.loc[~counted, "aliases"].map(
                lambda a: f"REPORTED BY REFERENCE: identical to {a}'s run "
                          "(same text variant), counted once in the family "
                          "-- design spec section 12"))
        paired_tbl.to_csv(out_paired, index=False)

        n_aliased = int((~counted).sum())
        print(f"\n=== Multiple-comparison check: Holm-Bonferroni across "
              f"all {n_tests} paired tests in the {args.target} family "
              f"(alpha=0.05) ===")
        if n_aliased:
            print(f"    ({len(paired_tbl)} rows reported, {n_aliased} of them "
                  f"aliased duplicates counted once at their source -- "
                  f"family size is {n_tests})")
        flagged = paired_tbl[non_tie]
        if flagged.empty:
            print("    (every paired verdict was already 'tie'; nothing to correct)")
        else:
            for _, r in flagged.iterrows():
                print(f"  {r['config']}/{r['model']:<12} {r['subset']:<16} "
                      f"verdict={r['verdict']:<6} p={r['p_value']:.4f}  "
                      f"holm_significant={r['holm_significant']}")

    print(f"\n  -> {out_tbl}")
    print(f"  -> {out_paired}")

    if args.target == "delta":
        print("\n  (subset-size and dead-zone tables are level-target "
              "artefacts; not regenerated under --target delta)")
        return

    sizes = subset_sizes()
    sizes.to_csv(config.SUBSET_SIZES_CSV, index=False)
    print("\n=== Subset sizes per config (identical by construction -- see "
          "subset_sizes() docstring) ===")
    pivot = sizes.pivot(index="subset", columns="config", values="n")
    pivot = pivot.reindex(SUBSETS)
    print(pivot.to_string())

    dz = dead_zone_by_config()
    dz.to_csv(config.DEAD_ZONE_CSV, index=False)
    print("\n=== Dead-zone closure by config (the real per-config quantity) ===")
    print(dz.to_string(index=False))

    print(f"  -> {config.SUBSET_SIZES_CSV}")
    print(f"  -> {config.DEAD_ZONE_CSV}")


if __name__ == "__main__":
    main()
