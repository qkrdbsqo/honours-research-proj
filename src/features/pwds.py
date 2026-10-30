"""Presence-weighted dominance score (PWDS), the target variable.

Usage:
    python -m src.features.pwds
"""
from __future__ import annotations

import sys

import pandas as pd

try:
    from src import config
    from src.data import load_oracle
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.data import load_oracle

BAN_COLS = ["ban1", "ban2", "ban3", "ban4", "ban5"]


def _games_per_patch(players: pd.DataFrame) -> pd.Series:
    """Distinct games per patch (denominator for PR and BR)."""
    return players.groupby("patch")["gameid"].nunique()


def _pick_stats(players: pd.DataFrame) -> pd.DataFrame:
    """Per (patch, champion): games picked and wins."""
    grp = players.groupby(["patch", "champion"])
    out = grp.agg(picks=("gameid", "nunique"), wins=("result", "sum"))
    return out.reset_index()


def _ban_stats(teams: pd.DataFrame) -> pd.DataFrame:
    """Per (patch, champion): distinct games the champion was banned in."""
    long = (teams[["patch", "gameid", *BAN_COLS]]
            .melt(id_vars=["patch", "gameid"], value_vars=BAN_COLS,
                  value_name="champion").dropna(subset=["champion"]))
    long = long[long["champion"].str.strip().astype(bool)]  # drop empty bans
    bans = (long.groupby(["patch", "champion"])["gameid"].nunique()
            .reset_index(name="ban_games"))
    return bans


def compute_pwds() -> pd.DataFrame:
    df = load_oracle.load_raw(usecols=[
        "gameid", "patch", "participantid", "position", "champion", "result",
        *BAN_COLS,
    ])
    players = load_oracle.player_rows(df)
    teams = load_oracle.team_rows(df)

    games = _games_per_patch(players)
    picks = _pick_stats(players)
    bans = _ban_stats(teams)

    m = picks.merge(bans, on=["patch", "champion"], how="left")
    m["ban_games"] = m["ban_games"].fillna(0).astype(int)
    m["total_games"] = m["patch"].map(games).astype(int)

    m = m[m["picks"] >= config.MIN_GAMES].copy()

    m["pr"] = m["picks"] / m["total_games"]
    m["br"] = m["ban_games"] / m["total_games"]
    m["wr"] = m["wins"] / m["picks"]
    m["wr_centred"] = m["wr"] - config.WR_CENTRE

    for name, (w_pr, w_br, w_wr) in config.PWDS_WEIGHTS.items():
        m[f"pwds_{name}"] = w_pr * m["pr"] + w_br * m["br"] + w_wr * m["wr_centred"]

    cols = ["champion", "patch", "total_games", "picks", "ban_games",
            "pr", "br", "wr", "wr_centred",
            *[f"pwds_{n}" for n in config.PWDS_WEIGHTS]]
    m = m[cols].sort_values(["patch", f"pwds_{config.DEFAULT_PWDS_WEIGHTS}"],
                             ascending=[True, False]).reset_index(drop=True)

    m["season"] = m["patch"].map(config.season_of)
    front = ["season", "champion", "patch"]
    m = m[front + [c for c in m.columns if c not in front]]
    return m


def main() -> None:
    config.ensure_dirs()
    df = compute_pwds()
    df.to_csv(config.PWDS_CSV, index=False)

    default = f"pwds_{config.DEFAULT_PWDS_WEIGHTS}"
    print(f"pwds: {len(df):,} scored champion-patch rows over "
          f"{df['patch'].nunique()} patches "
          f"(min {config.MIN_GAMES} picks)")
    print(f"  rows per patch: min {df.groupby('patch').size().min()}, "
          f"max {df.groupby('patch').size().max()}, "
          f"median {int(df.groupby('patch').size().median())}")
    top = df.nlargest(5, default)[["patch", "champion", "pr", "br", "wr", default]]
    print(f"\n  top 5 by {default}:")
    for _, r in top.iterrows():
        print(f"    {r['patch']} {r['champion']:<16} "
              f"PR={r['pr']:.2f} BR={r['br']:.2f} WR={r['wr']:.2f} "
              f"-> {r[default]:.3f}")
    print(f"\n  -> {config.PWDS_CSV}")


if __name__ == "__main__":
    main()
