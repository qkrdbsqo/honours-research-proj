"""Load and filter the Oracle's Elixir match CSVs.

Usage:
    python -m src.data.load_oracle
"""
from __future__ import annotations

import sys

import pandas as pd

try:
    from src import config
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config

BAN_COLS = ["ban1", "ban2", "ban3", "ban4", "ban5"]


def load_raw(usecols: list[str] | None = None,
             seasons: list[str] | None = None) -> pd.DataFrame:
    """Load and concatenate the Oracle's Elixir CSVs for the given seasons."""
    seasons = seasons or config.SEASONS
    frames = []
    for season in seasons:
        path = config.ORACLE_CSVS[season]
        df = pd.read_csv(path, usecols=usecols, low_memory=False,
                         dtype={"patch": str})
        if "patch" in df.columns:
            df["patch"] = df["patch"].apply(_normalise_patch)
            dropped = df["patch"].isin(config.DROP_PATCHES)
            if dropped.any():
                print(f"  {season}: dropped {int(dropped.sum()):,} rows on "
                      f"{sorted(set(df.loc[dropped, 'patch']))} "
                      f"(out-of-season spillover)")
                df = df[~dropped].copy()
        df["season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _normalise_patch(value) -> str | float:
    """Render a patch label as a stable 'MM.NN' string."""
    if pd.isna(value):
        return value
    s = str(value).strip()
    if "." not in s:
        return s
    major, minor = s.split(".", 1)
    return f"{int(major):d}.{int(minor):02d}"


def player_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for individual champions (participantid 1–10)."""
    return df[df["participantid"].between(1, 10)].copy()


def team_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Team-level rows (participantid 100 / 200) carrying the ban columns."""
    return df[df["position"] == "team"].copy()


def unique_patches(df: pd.DataFrame) -> list[str]:
    """Sorted unique patch labels present in the data."""
    patches = df["patch"].dropna().unique().tolist()
    return sorted(patches, key=lambda s: tuple(int(p) for p in str(s).split(".")))


def games_per_patch(df: pd.DataFrame) -> pd.Series:
    """Number of distinct games per patch."""
    return df.groupby("patch")["gameid"].nunique().sort_index()


def main() -> None:
    print("Loading Oracle's Elixir (all seasons) ...")
    df = load_raw()
    print(f"  {len(df):,} rows, {df['gameid'].nunique():,} games")
    for season in config.SEASONS:
        sub = df[df["season"] == season]
        print(f"  {season}: {len(sub):,} rows, {sub['gameid'].nunique():,} games, "
              f"{sub['patch'].nunique()} patches")

    patches = unique_patches(df)
    print(f"\nPatches ({len(patches)}): {', '.join(patches)}")

    print("\nGames per patch:")
    for patch, n in games_per_patch(df).items():
        print(f"  {patch}: {n:,} games")

    players = player_rows(df)
    print(f"\nPlayer rows: {len(players):,} | unique champions: "
          f"{players['champion'].nunique()}")


if __name__ == "__main__":
    main()
