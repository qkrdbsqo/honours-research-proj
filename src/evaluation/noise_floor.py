"""Simulate a static meta to estimate how much PWDS movement is sampling noise.

Usage:
    python -m src.evaluation.noise_floor
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    from src import config
    from src.models import _shared as sh
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.models import _shared as sh

N_SIM = 4000
OUT = config.OUTPUTS / "tables" / "diagnostics"
_w = config.PWDS_WEIGHTS[config.DEFAULT_PWDS_WEIGHTS]
W = {"pr": _w[0], "br": _w[1], "wr": _w[2]}


def test_rows() -> pd.DataFrame:
    """Frozen-test has-lag rows, joined to their own previous-patch counts."""
    d = pd.read_csv(config.ALIGNED_DATASET_CSV, dtype={"patch": str})
    order = {p: i for i, p in enumerate(config.ALL_PATCHES)}
    d["oi"] = d["patch"].map(order)
    prev = d[["champion", "oi", "pr", "br", "wr",
              "picks", "ban_games", "total_games"]].copy()
    prev["oi"] += 1
    prev.columns = ["champion", "oi", "pr_l", "br_l", "wr_l",
                    "picks_l", "ban_l", "tg_l"]
    m = d.merge(prev, on=["champion", "oi"], how="left")
    t = m[m["patch"].isin(config.TEST_PATCHES)
          & (m["has_lag"] == 1)
          & m["pr_l"].notna()].copy()
    t["changed"] = ((t["note_changed"].fillna(0) != 0)
                    | (t["stat_changed"].fillna(0) != 0)).astype(int)
    return t.reset_index(drop=True)


def _pooled_truth(t: pd.DataFrame):
    """Pooled rate estimates, used as the true rates under the static-meta null."""
    g1 = t["total_games"].to_numpy().astype(int)
    g2 = t["tg_l"].to_numpy().astype(int)
    p = ((t["picks"] + t["picks_l"]) / (g1 + g2)).to_numpy()
    b = ((t["ban_games"] + t["ban_l"]) / (g1 + g2)).to_numpy()
    w = ((t["wr"] * t["picks"] + t["wr_l"] * t["picks_l"])
         / (t["picks"] + t["picks_l"])).to_numpy()
    return g1, g2, p, b, w


def _pwds(picks, bans, wins, games):
    safe = np.maximum(picks, 1)
    return (W["pr"] * picks / games + W["br"] * bans / games
            + W["wr"] * (wins / safe - 0.5))


def error_budget(t: pd.DataFrame, n_sim: int = N_SIM,
                 seed: int | None = None) -> dict:
    """Split observed persistence error into irreducible / lag-noise / real."""
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    g1, g2, p, b, w = _pooled_truth(t)
    truth = W["pr"] * p + W["br"] * b + W["wr"] * (w - 0.5)

    oracle = np.empty(n_sim)
    persist = np.empty(n_sim)
    for i in range(n_sim):
        k1, n1 = rng.binomial(g1, p), rng.binomial(g1, b)
        k2, n2 = rng.binomial(g2, p), rng.binomial(g2, b)
        v1 = rng.binomial(np.maximum(k1, 1), w)
        v2 = rng.binomial(np.maximum(k2, 1), w)
        pw1 = _pwds(k1, n1, v1, g1)
        pw2 = _pwds(k2, n2, v2, g2)
        oracle[i] = np.abs(pw1 - truth).mean()
        persist[i] = np.abs(pw1 - pw2).mean()

    target = f"pwds_{config.DEFAULT_PWDS_WEIGHTS}"
    lag = f"lag_pwds_{config.DEFAULT_PWDS_WEIGHTS}"
    observed = float((t[target] - t[lag]).abs().mean())
    o, pe = float(oracle.mean()), float(persist.mean())
    return {"n": len(t), "observed_mae": observed,
            "noise_only_persistence": pe, "oracle_floor": o,
            "irreducible": o, "lag_copy_noise": pe - o, "real_movement": observed - pe,
            "irreducible_pct": 100 * o / observed,
            "lag_copy_noise_pct": 100 * (pe - o) / observed,
            "real_movement_pct": 100 * (observed - pe) / observed,
            "noise_only_ci_lo": float(np.percentile(persist, 2.5)),
            "noise_only_ci_hi": float(np.percentile(persist, 97.5))}


def component_shares(t: pd.DataFrame) -> pd.DataFrame:
    """Which of pick/ban/win rate actually carries the movement."""
    rows = []
    contrib = {"pr": W["pr"] * (t["pr"] - t["pr_l"]),
               "br": W["br"] * (t["br"] - t["br_l"]),
               "wr": W["wr"] * (t["wr"] - t["wr_l"])}
    total = sum(c.abs().mean() for c in contrib.values())
    for name, c in contrib.items():
        rows.append({"component": name, "weight_in_pwds": W[name],
                     "mean_abs_contribution": float(c.abs().mean()),
                     "share_of_movement_pct": float(100 * c.abs().mean() / total)})
    return pd.DataFrame(rows)


def by_sample_size(t: pd.DataFrame) -> pd.DataFrame:
    """Movement against how many times the champion was actually picked."""
    t = t.copy()
    t["min_picks"] = t[["picks", "picks_l"]].min(axis=1)
    t["abs_move"] = (t[f"pwds_{config.DEFAULT_PWDS_WEIGHTS}"]
                     - t[f"lag_pwds_{config.DEFAULT_PWDS_WEIGHTS}"]).abs()
    t["abs_move_wr"] = (t["wr"] - t["wr_l"]).abs()
    t["quartile"] = pd.qcut(t["min_picks"], 4,
                            labels=["Q1 fewest", "Q2", "Q3", "Q4 most"])
    g = t.groupby("quartile", observed=True).agg(
        n=("abs_move", "size"), median_picks=("min_picks", "median"),
        mean_abs_move=("abs_move", "mean"),
        mean_abs_move_win_rate=("abs_move_wr", "mean")).reset_index()
    return g


def by_changed(t: pd.DataFrame, n_sim: int = 2000) -> pd.DataFrame:
    """Changed versus unchanged champions, with sampling noise removed from both."""
    rows = []
    for label, sub in [("changed", t[t["changed"] == 1]),
                       ("unchanged", t[t["changed"] == 0]),
                       ("all", t)]:
        b = error_budget(sub.reset_index(drop=True), n_sim=n_sim)
        rows.append({"population": label, **b})
    out = pd.DataFrame(rows)
    ch, un = out.iloc[0], out.iloc[1]
    obs_mass = {"changed": ch["n"] * ch["observed_mae"],
                "unchanged": un["n"] * un["observed_mae"]}
    real_mass = {"changed": ch["n"] * ch["real_movement"],
                 "unchanged": un["n"] * un["real_movement"]}
    out["share_of_observed_movement_pct"] = [
        100 * obs_mass["changed"] / sum(obs_mass.values()),
        100 * obs_mass["unchanged"] / sum(obs_mass.values()), 100.0]
    out["share_of_REAL_movement_pct"] = [
        100 * real_mass["changed"] / sum(real_mass.values()),
        100 * real_mass["unchanged"] / sum(real_mass.values()), 100.0]
    return out


def volume_requirement(t: pd.DataFrame, multipliers=(1, 2, 4, 8, 16, 32, 64),
                       n_sim: int = 200, seed: int | None = None) -> pd.DataFrame:
    """How many professional games per patch would the signal need to surface?"""
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    g1, g2, p, b, w = _pooled_truth(t)
    truth = W["pr"] * p + W["br"] * b + W["wr"] * (w - 0.5)
    real = error_budget(t, n_sim=n_sim, seed=seed)["real_movement"]

    rows = []
    for mult in multipliers:
        G1, G2 = (g1 * mult).astype(int), (g2 * mult).astype(int)
        oracle, persist = np.empty(n_sim), np.empty(n_sim)
        for i in range(n_sim):
            k1, n1 = rng.binomial(G1, p), rng.binomial(G1, b)
            k2, n2 = rng.binomial(G2, p), rng.binomial(G2, b)
            v1 = rng.binomial(np.maximum(k1, 1), w)
            v2 = rng.binomial(np.maximum(k2, 1), w)
            pw1 = _pwds(k1, n1, v1, G1)
            oracle[i] = np.abs(pw1 - truth).mean()
            persist[i] = np.abs(pw1 - _pwds(k2, n2, v2, G2)).mean()
        observed = real + float(persist.mean())
        rows.append({"games_multiplier": mult,
                     "median_games_per_patch": int(np.median(G1)),
                     "oracle_floor": float(oracle.mean()),
                     "persistence_mae": observed,
                     "real_movement": real,
                     "real_share_pct": 100 * real / observed,
                     "max_possible_model_gain": observed - float(oracle.mean())})
    return pd.DataFrame(rows)


def main() -> None:
    assert sh.active_target() == "level", \
        "the noise floor is defined on the PWDS level, not on the delta target"
    OUT.mkdir(parents=True, exist_ok=True)
    t = test_rows()
    print(f"[noise_floor] frozen test window, has-lag rows: n={len(t)}, "
          f"{N_SIM} simulated seasons under a static-meta null")

    budget = error_budget(t)
    pd.DataFrame([budget]).to_csv(OUT / "noise_error_budget.csv", index=False)
    print("\nERROR BUDGET (mean |error| in PWDS)")
    print(f"  observed persistence MAE                {budget['observed_mae']:.5f}")
    print(f"  irreducible measurement noise           {budget['irreducible']:.5f}"
          f"  {budget['irreducible_pct']:5.1f}%")
    print(f"  noise imported by copying a noisy lag   {budget['lag_copy_noise']:.5f}"
          f"  {budget['lag_copy_noise_pct']:5.1f}%")
    print(f"  genuinely real meta movement            {budget['real_movement']:.5f}"
          f"  {budget['real_movement_pct']:5.1f}%")

    comp = component_shares(t)
    comp.to_csv(OUT / "noise_component_shares.csv", index=False)
    print("\nWHERE THE MOVEMENT COMES FROM")
    print(comp.to_string(index=False))

    size = by_sample_size(t)
    size.to_csv(OUT / "noise_by_sample_size.csv", index=False)
    print("\nMOVEMENT vs SAMPLE SIZE (meta movement would not shrink; this does)")
    print(size.to_string(index=False))

    ch = by_changed(t)
    ch.to_csv(OUT / "noise_by_changed.csv", index=False)
    print("\nFINDING 6, WITH NOISE REMOVED FROM BOTH SIDES")
    print(ch[["population", "n", "observed_mae", "real_movement",
              "real_movement_pct", "share_of_observed_movement_pct",
              "share_of_REAL_movement_pct"]].to_string(index=False))
    vol = volume_requirement(t)
    vol.to_csv(OUT / "noise_volume_requirement.csv", index=False)
    print()
    print("WHAT WOULD IT TAKE? real movement is fixed; only the noise shrinks")
    print(vol.to_string(index=False))

    print(f"\n[noise_floor] tables -> {OUT}")


if __name__ == "__main__":
    main()
