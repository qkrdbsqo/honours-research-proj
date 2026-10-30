"""Break the rolling gain down by lag quartile: real movement, position changes and team concentration.

Usage:
    python -m src.evaluation.tier_anatomy
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config
from src.data import align, load_oracle
from src.evaluation import decision_relevance as dr
from src.evaluation import noise_floor as nf

OUT = config.OUTPUTS / "tables" / "diagnostics"
QLAB = ["Q1 lowest", "Q2", "Q3", "Q4 highest"]
N_SIM = 1500


def rolling_rows() -> pd.DataFrame:
    """Rolling-origin has-lag rows, carrying their own previous-patch counts."""
    d = pd.read_csv(config.ALIGNED_DATASET_CSV, dtype={"patch": str})
    order = {p: i for i, p in enumerate(config.ALL_PATCHES)}
    d["oi"] = d["patch"].map(order)
    prev = d[["champion", "oi", "pr", "br", "wr",
              "picks", "ban_games", "total_games"]].copy()
    prev["oi"] += 1
    prev.columns = ["champion", "oi", "pr_l", "br_l", "wr_l",
                    "picks_l", "ban_l", "tg_l"]
    m = d.merge(prev, on=["champion", "oi"], how="left")

    keep = dr.with_strata(dr.predictions())[
        ["champion", "patch", "lag_quartile", "changed", "gain_vs_persistence"]]
    t = m.merge(keep, on=["champion", "patch"], how="inner", suffixes=("", "_s"))
    t = t[t["pr_l"].notna()].copy()
    t["abs_move"] = (t["pwds_presence_heavy"] - t["lag_pwds_presence_heavy"]).abs()
    return t.reset_index(drop=True)


def noise_by_quartile(t: pd.DataFrame, n_sim: int = N_SIM) -> pd.DataFrame:
    rows = []
    for q in QLAB:
        sub = t[t["lag_quartile"].astype(str) == q]
        if sub.empty:
            continue
        b = nf.error_budget(sub.reset_index(drop=True), n_sim=n_sim)
        rows.append({"lag_quartile": q, "median_picks": float(sub["picks"].median()),
                     **b})
    out = pd.DataFrame(rows)
    if "observed_mae" in out and "real_movement" in out:
        out["real_share_pct"] = 100 * out["real_movement"] / out["observed_mae"]
    return out


def main_position() -> pd.DataFrame:
    """Modal position per (champion, patch), read empirically from Oracle."""
    raw = load_oracle.load_raw()
    pl = load_oracle.player_rows(raw)
    pl = pl[pl["position"].notna() & pl["champion"].notna()]
    pl["champion"] = pl["champion"].map(align.normalise_champion_name)
    g = pl.groupby(["champion", "patch", "position"]).size().rename("n").reset_index()
    idx = g.groupby(["champion", "patch"])["n"].idxmax()
    return g.loc[idx, ["champion", "patch", "position"]].reset_index(drop=True)


def position_churn(t: pd.DataFrame) -> pd.DataFrame:
    pos = main_position()
    order = {p: i for i, p in enumerate(config.ALL_PATCHES)}
    pos["oi"] = pos["patch"].map(order)
    prev = pos.copy()
    prev["oi"] += 1
    prev = prev.rename(columns={"position": "position_l"})[["champion", "oi", "position_l"]]
    pos = pos.merge(prev, on=["champion", "oi"], how="left")

    k = t.copy()
    k["champion"] = k["champion"].map(align.normalise_champion_name)
    k = k.merge(pos[["champion", "patch", "position", "position_l"]],
                on=["champion", "patch"], how="left")
    k = k[k["position"].notna() & k["position_l"].notna()].copy()
    k["switched"] = (k["position"] != k["position_l"]).astype(int)
    out = (k.groupby(k["lag_quartile"].astype(str))
             .agg(n=("switched", "size"),
                  switched_pct=("switched", lambda s: 100 * s.mean()),
                  mean_abs_move=("abs_move", "mean"))
             .reindex(QLAB).reset_index().rename(columns={"index": "lag_quartile"}))
    sw = k[k["switched"] == 1]["abs_move"]
    st = k[k["switched"] == 0]["abs_move"]
    u = stats.mannwhitneyu(sw, st, alternative="two-sided")
    out.attrs["switch_move"] = (float(sw.mean()), float(st.mean()), float(u.pvalue))
    return out, k


def concentration() -> pd.DataFrame:
    raw = load_oracle.load_raw()
    pl = load_oracle.player_rows(raw)
    pl = pl[pl["champion"].notna() & pl["teamname"].notna()]
    pl["champion"] = pl["champion"].map(align.normalise_champion_name)
    g = pl.groupby(["champion", "patch"])
    out = g.agg(picks_obs=("teamname", "size"),
                n_teams=("teamname", "nunique"),
                n_leagues=("league", "nunique")).reset_index()
    top = (pl.groupby(["champion", "patch", "teamname"]).size()
             .rename("n").reset_index()
             .groupby(["champion", "patch"])["n"].max().rename("top_team_picks")
             .reset_index())
    out = out.merge(top, on=["champion", "patch"], how="left")
    out["top_team_share"] = out["top_team_picks"] / out["picks_obs"]
    out["hhi_teams"] = np.nan
    hhi = (pl.groupby(["champion", "patch", "teamname"]).size().rename("n")
             .reset_index())
    hhi["tot"] = hhi.groupby(["champion", "patch"])["n"].transform("sum")
    hhi["sq"] = (hhi["n"] / hhi["tot"]) ** 2
    h = hhi.groupby(["champion", "patch"])["sq"].sum().rename("hhi").reset_index()
    out = out.drop(columns=["hhi_teams"]).merge(h, on=["champion", "patch"], how="left")
    return out


def _partial_spearman(x, y, z):
    """Spearman of x and y after removing a continuous rank-linear effect of z."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a):
        b = np.polyfit(rz, a, 1)
        return a - (b[0] * rz + b[1])
    return stats.spearmanr(resid(rx), resid(ry))


def concentration_by_quartile(t: pd.DataFrame):
    c = concentration()
    k = t.copy()
    k["champion"] = k["champion"].map(align.normalise_champion_name)
    k = k.merge(c, on=["champion", "patch"], how="left")
    k = k[k["n_teams"].notna()].copy()
    out = (k.groupby(k["lag_quartile"].astype(str))
             .agg(n=("n_teams", "size"),
                  median_picks=("picks_obs", "median"),
                  median_teams=("n_teams", "median"),
                  mean_top_team_share=("top_team_share", "mean"),
                  mean_hhi=("hhi", "mean"),
                  median_leagues=("n_leagues", "median"))
             .reindex(QLAB).reset_index().rename(columns={"index": "lag_quartile"}))
    tests = []
    for col in ("top_team_share", "hhi"):
        raw_r = stats.spearmanr(k[col], k["abs_move"])
        par_r = _partial_spearman(k[col], k["abs_move"], k["picks_obs"])
        tests.append({"concentration_measure": col, "n": len(k),
                      "rho_with_abs_move": raw_r.statistic, "p_raw": raw_r.pvalue,
                      "partial_rho_given_picks": par_r.statistic,
                      "p_partial": par_r.pvalue})
    return out, pd.DataFrame(tests), k


def _budget_deff(t, n_sim, seed, deff):
    """error_budget with variance inflated by a design effect."""
    if deff == 1:
        return nf.error_budget(t, n_sim=n_sim, seed=seed)
    rng = np.random.default_rng(seed)
    g1, g2, p, b, w = nf._pooled_truth(t)
    e1 = np.maximum(np.round(g1 / deff).astype(int), 1)
    e2 = np.maximum(np.round(g2 / deff).astype(int), 1)
    persist = np.empty(n_sim)
    for i in range(n_sim):
        k1, n1 = rng.binomial(e1, p), rng.binomial(e1, b)
        k2, n2 = rng.binomial(e2, p), rng.binomial(e2, b)
        v1 = rng.binomial(np.maximum(k1, 1), w)
        v2 = rng.binomial(np.maximum(k2, 1), w)
        persist[i] = np.abs(nf._pwds(k1, n1, v1, e1) - nf._pwds(k2, n2, v2, e2)).mean()
    target = f"pwds_{config.DEFAULT_PWDS_WEIGHTS}"
    lag = f"lag_pwds_{config.DEFAULT_PWDS_WEIGHTS}"
    obs = float((t[target] - t[lag]).abs().mean())
    pe = float(persist.mean())
    return {"observed_mae": obs, "noise_only_persistence": pe,
            "real_movement_pct": 100 * (obs - pe) / obs}


def robustness(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def run(label, groups, n_sim=1500, seed=None, deff=1):
        for q, sub in groups:
            b = _budget_deff(sub.reset_index(drop=True), n_sim,
                             config.RANDOM_SEED if seed is None else seed, deff)
            rows.append({"check": label, "group": str(q), "n": len(sub),
                         "real_share_pct": b["real_movement_pct"]})
    quart = [(q, t[t["lag_quartile"].astype(str) == q]) for q in QLAB]
    for sd in (1, 2, 3, 4, 5):
        run(f"seed={sd}", quart, seed=sd)
    for ns in (500, 4000):
        run(f"n_sim={ns}", quart, n_sim=ns)
    for de in (1.5, 2.0, 3.0):
        run(f"design_effect={de}", quart, deff=de)
    for k, lab in ((3, "tercile"), (5, "quintile")):
        cut = pd.qcut(t["lag_pwds_presence_heavy"], k, labels=[f"{lab} {i+1}" for i in range(k)])
        run(f"cut={lab}s", [(c, t[cut == c]) for c in cut.cat.categories])
    return pd.DataFrame(rows)


def cluster_bootstrap(t: pd.DataFrame, n_boot: int = 200, n_sim: int = 150) -> pd.DataFrame:
    """Bootstrap over origins, since rows in one patch share its games."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    patches = t["patch"].unique()
    by_patch = {p: t[t["patch"] == p] for p in patches}
    out = {q: [] for q in QLAB}
    for bi in range(n_boot):
        pick = rng.choice(patches, size=len(patches), replace=True)
        bt = pd.concat([by_patch[p] for p in pick], ignore_index=True)
        for q in QLAB:
            sub = bt[bt["lag_quartile"].astype(str) == q].reset_index(drop=True)
            if len(sub) < 30:
                continue
            b = nf.error_budget(sub, n_sim=n_sim, seed=bi)
            out[q].append(b["real_movement_pct"])
    rows = []
    for q in QLAB:
        a = np.array(out[q])
        rows.append({"lag_quartile": q, "n_boot": len(a), "median": np.median(a),
                     "ci_lo": np.percentile(a, 2.5), "ci_hi": np.percentile(a, 97.5)})
    res = pd.DataFrame(rows)
    m = np.array([out[q][:min(map(len, out.values()))] for q in QLAB])
    res.attrs["monotone_share"] = float(np.mean(np.all(np.diff(m, axis=0) > 0, axis=0)))
    res.attrs["q1_lt_q4_share"] = float(np.mean(m[0] < m[3]))
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    t = rolling_rows()
    print(f"population: {len(t)} rows, {t['patch'].nunique()} patches")

    print("\n=== A. sampling noise by tier ===")
    a = noise_by_quartile(t)
    print(a.to_string(index=False, float_format=lambda v: f"{v: .5f}"))
    a.to_csv(OUT / "tier_noise_by_quartile.csv", index=False)

    print("\n=== B. position churn by tier ===")
    b, bk = position_churn(t)
    print(b.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    m_sw, m_st, p_sw = b.attrs["switch_move"]
    print(f"mean |dPWDS| switched {m_sw:.5f} vs stayed {m_st:.5f}  (p={p_sw:.4g})")
    b.to_csv(OUT / "tier_position_churn.csv", index=False)

    print("\n=== C. team / league concentration by tier ===")
    c, ctests, _ = concentration_by_quartile(t)
    print(c.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    print()
    print(ctests.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    c.to_csv(OUT / "tier_concentration.csv", index=False)
    ctests.to_csv(OUT / "tier_concentration_tests.csv", index=False)


def main_robust() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = rolling_rows()
    r = robustness(t)
    r.to_csv(OUT / "tier_noise_robustness.csv", index=False)
    piv = r.pivot_table(index="check", columns="group", values="real_share_pct", sort=False)
    print(piv.to_string(float_format=lambda v: f"{v:6.1f}"))
    cb = cluster_bootstrap(t)
    cb.to_csv(OUT / "tier_noise_bootstrap.csv", index=False)
    print()
    print(cb.to_string(index=False, float_format=lambda v: f"{v:6.1f}"))
    print(f"strict Q1<Q2<Q3<Q4 in {100*cb.attrs['monotone_share']:.0f}% of resamples; "
          f"Q1<Q4 in {100*cb.attrs['q1_lt_q4_share']:.0f}%")


if __name__ == "__main__":
    if "--robust" in sys.argv:
        main_robust()
    else:
        main()
