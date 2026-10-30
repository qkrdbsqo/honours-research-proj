"""Predict whether a champion absent from pro play on the previous patch appears on the next one.

Usage:
    python -m src.evaluation.emergence
"""
from __future__ import annotations

import argparse
import glob
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src import config
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config

DIAG = config.TABLES / "diagnostics"
N_BOOT = 2000


def _log(msg):
    print(str(msg).encode("ascii", "replace").decode("ascii"), flush=True)


def load_presence():
    """(champion, patch) -> appeared in a professional game."""
    from src.data.align import normalise_champion_name as norm
    frames = [pd.read_csv(f, dtype={"patch": str},
                          usecols=["patch", "champion", "position"],
                          low_memory=False)
              for f in sorted(glob.glob(str(config.ROOT / "data" / "raw" /
                                            "oracle" / "*.csv")))]
    o = pd.concat(frames)
    o = o[(o["position"] != "team") & o["champion"].notna()].copy()
    o["champion"] = [norm(c) for c in o["champion"]]
    return o


def build_cells():
    """One row per (champion, patch) with causal presence-history features."""
    o = load_presence()
    pro = {(c, p) for c, p in zip(o["champion"], o["patch"])}
    picks = o.groupby(["champion", "patch"]).size().to_dict()

    d = pd.read_csv(config.ALIGNED_DATASET_CSV, dtype={"patch": str})
    ordered = (d[["patch", "patch_idx"]].drop_duplicates()
               .sort_values("patch_idx")["patch"].tolist())
    scored = {(c, p): h for c, p, h in
              zip(d["champion"], d["patch"], d["has_lag"])}
    universe = sorted({c for c, _ in pro})

    rows = []
    for i, patch in enumerate(ordered):
        if i == 0:
            continue
        prev, past = ordered[i - 1], ordered[:i]
        total_picks = sum(v for (c, p), v in picks.items() if p == patch)
        for champ in universe:
            seen_before = [(champ, p) in pro for p in past]
            last_seen = next((i - 1 - j for j, p in
                              enumerate(reversed(past)) if (champ, p) in pro),
                             None)
            rows.append({
                "champion": champ, "patch": patch,
                "pro_prev": int((champ, prev) in pro),
                "prior_freq": float(np.mean(seen_before)) if past else 0.0,
                "patches_since_seen": (i - 1 - last_seen) if last_seen is not None else i,
                "pro_now": int((champ, patch) in pro),
                "picks_now": picks.get((champ, patch), 0),
                "patch_total_picks": total_picks,
                "in_dataset": int((champ, patch) in scored),
                "has_lag": int(scored.get((champ, patch), 0) == 1),
            })
    return pd.DataFrame(rows)


def _boot_auc_diff(y, s_a, s_b, seed=None):
    """Paired bootstrap on the AUC difference (b - a)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    n, diffs = len(y), []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], s_b[idx]) - roc_auc_score(y[idx], s_a[idx]))
    d = np.array(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(np.mean(d)), float(lo), float(hi), float(p)


def predictability(cells):
    """Fit on the project's training patches, evaluate on the frozen window."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score

    train = cells[~cells["patch"].isin(config.TEST_PATCHES)]
    test = cells[cells["patch"].isin(config.TEST_PATCHES)]
    cold = (test["pro_prev"] == 0).to_numpy()
    y = test["pro_now"].to_numpy()

    specs = [(["pro_prev"], "persistence analogue (played last patch)"),
             (["pro_prev", "prior_freq"], "+ prior professional frequency"),
             (["pro_prev", "prior_freq", "patches_since_seen"], "+ recency")]
    rows, scores = [], {}
    for feats, label in specs:
        sc = StandardScaler().fit(train[feats])
        m = LogisticRegression(max_iter=2000).fit(sc.transform(train[feats]),
                                                  train["pro_now"])
        s = m.predict_proba(sc.transform(test[feats]))[:, 1]
        scores[label] = s
        rows.append({
            "model": label,
            "auc_all": round(float(roc_auc_score(y, s)), 4),
            "auc_dead_zone": round(float(roc_auc_score(y[cold], s[cold])), 4),
            "ap_dead_zone": round(float(average_precision_score(y[cold], s[cold])), 4),
        })
    res = pd.DataFrame(rows)

    base = scores[specs[0][1]][cold]
    best = scores[specs[1][1]][cold]
    d, lo, hi, p = _boot_auc_diff(y[cold], base, best)
    gain = {"delta_auc_dead_zone": round(d, 4), "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4), "p_value": round(p, 5),
            "n_dead_zone": int(cold.sum()), "n_emergences": int(y[cold].sum())}
    return res, gain, (y, cold, scores, specs)


def blindness(cells):
    """How much of emergence the pipeline's target definition cannot see."""
    test = cells[cells["patch"].isin(config.TEST_PATCHES)]
    em = test[(test["pro_prev"] == 0) & (test["pro_now"] == 1)]
    pick_share = (em["picks_now"].sum() /
                  test.groupby("patch")["patch_total_picks"].first().sum())
    return {
        "emergence_cases": int(len(em)),
        "entering_aligned_dataset": int(em["in_dataset"].sum()),
        "with_has_lag_1": int(em["has_lag"].sum()),
        "excluded_by_5_pick_minimum": int(len(em) - em["in_dataset"].sum()),
        "share_of_test_window_picks_pct": round(100 * float(pick_share), 2),
        "median_picks_when_emerging": float(em["picks_now"].median()),
    }


def plot(y, cold, scores, specs):
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for feats, label in specs:
        s = scores[label][cold]
        fpr, tpr, _ = roc_curve(y[cold], s)
        ax.plot(fpr, tpr, linewidth=1.8,
                label="%s (AUC %.3f)" % (label, roc_auc_score(y[cold], s)))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Predicting champion emergence\n"
                 "(champions absent from pro play on the previous patch, n=%d)"
                 % int(cold.sum()), fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, linestyle=":")
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    p = config.FIGURES / "emergence_roc.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log("  wrote %s" % p)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args(argv)
    DIAG.mkdir(parents=True, exist_ok=True)
    cells = build_cells()
    _log("cells: %d (champion x patch), test window %d"
         % (len(cells), int(cells["patch"].isin(config.TEST_PATCHES).sum())))

    res, gain, curve = predictability(cells)
    _log("")
    _log("1. IS EMERGENCE PREDICTABLE?  (fit on train patches, frozen test window)")
    _log(res.to_string(index=False))
    _log("")
    _log("   gain over the persistence analogue, dead zone (n=%d, %d emergences):"
         % (gain["n_dead_zone"], gain["n_emergences"]))
    _log("   dAUC %+.4f  95%% CI [%+.4f, %+.4f]  p=%.5f"
         % (gain["delta_auc_dead_zone"], gain["ci_lo"], gain["ci_hi"],
            gain["p_value"]))

    bl = blindness(cells)
    _log("")
    _log("2. WHAT THE PIPELINE CANNOT SEE")
    for k, v in bl.items():
        _log("   %-34s %s" % (k, v))

    res.to_csv(DIAG / "emergence_predictability.csv", index=False)
    pd.DataFrame([gain]).to_csv(DIAG / "emergence_gain.csv", index=False)
    pd.DataFrame([bl]).to_csv(DIAG / "emergence_blindness.csv", index=False)
    plot(*curve)
    _log("")
    _log("tables in %s" % DIAG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
