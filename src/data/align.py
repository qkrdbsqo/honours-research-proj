"""Align match data, Data Dragon stats and patch notes by patch and champion, and write the modelling dataset.

Usage:
    python -m src.data.align
"""
from __future__ import annotations

import json
import re
import sys

try:
    from src import config
    from src.data import load_oracle
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.data import load_oracle


def build_version_map() -> dict[str, str]:
    """Map every scraped patch-note version to its client patch."""
    version_map: dict[str, str] = {}
    for season in config.SEASONS:
        major = config.SEASON_MAJORS[season]
        path = config.PATCH_NOTES_DIR / f"lol_patches_{season}.json"
        notes = json.loads(path.read_text(encoding="utf-8"))
        for entry in notes:
            key = entry["patch_version"]
            if key in version_map:
                raise ValueError(f"duplicate patch-note version {key!r}")
            pattern = r"14\.(\d+)" if season == "2024" else r"25\.(?:S1\.)?(\d+)"
            match = re.fullmatch(pattern, key)
            if match is None or not 1 <= int(match[1]) <= 24:
                raise ValueError(f"unexpected patch-note version {key!r} for {season}")
            value = f"{major}.{int(match[1]):02d}"
            if value in version_map.values():
                raise ValueError(f"duplicate client patch {value!r}")
            version_map[key] = value
    return version_map


def write_version_map() -> dict[str, str]:
    config.ensure_dirs()
    vmap = build_version_map()
    config.VERSION_MAP.write_text(json.dumps(vmap, indent=2), encoding="utf-8")
    print(f"version_map.json: {len(vmap)} patches -> {config.VERSION_MAP}")
    return vmap


def _latest_ddragon_file(season: str | None = None):
    """Newest champion.json snapshot, optionally restricted to one season."""
    majors = ([config.SEASON_MAJORS[season]] if season
              else list(config.SEASON_MAJORS.values()))
    files = [p for m in majors for p in config.DDRAGON_DIR.glob(f"{m}.*.json")]
    files.sort(key=lambda p: tuple(int(x) for x in p.stem.split(".")))
    return files[-1] if files else None


def build_champion_names() -> dict[str, str]:
    """Map Data Dragon champion ids to display names."""
    latest = _latest_ddragon_file()
    if latest is None:
        raise FileNotFoundError(
            "No Data Dragon JSON found. Run `python -m src.data.fetch_ddragon` first."
        )
    data = json.loads(latest.read_text(encoding="utf-8"))["data"]
    return {cid: meta["name"] for cid, meta in data.items()}


def write_champion_names() -> dict[str, str]:
    config.ensure_dirs()
    names = build_champion_names()
    config.CHAMPION_NAMES.write_text(json.dumps(names, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"champion_names.json: {len(names)} champions -> {config.CHAMPION_NAMES}")
    return names


MOJIBAKE_MARKER = "â€"


def _repair_mojibake(name: str) -> str:
    """Undo a UTF-8-read-as-cp1252 round trip, if that is what happened."""
    if MOJIBAKE_MARKER not in name:
        return name
    try:
        repaired = name.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name
    return repaired if MOJIBAKE_MARKER not in repaired else name


def normalise_champion_name(name: str) -> str:
    """Canonicalise a champion name for cross-source matching."""
    name = _repair_mojibake(name)
    return (name.replace("’", "'").replace("‘", "'")
                .replace("′", "'").strip())


def validate_alignment(vmap: dict[str, str], dd_names: dict[str, str]) -> None:
    """Cross-check champion names across Oracle, Data Dragon, and patch notes."""
    oracle_champs = {
        normalise_champion_name(c) for c in
        load_oracle.player_rows(load_oracle.load_raw(usecols=["participantid", "position", "champion"]))
        ["champion"].dropna().unique()
    }
    dd_display = {normalise_champion_name(n) for n in dd_names.values()}

    notes = json.loads((config.PATCH_NOTES_DIR / "lol_patches_2025.json")
                       .read_text(encoding="utf-8"))
    note_champs = {normalise_champion_name(c["champion"])
                   for entry in notes for c in entry["champions"]}

    print("\n== Champion-name coverage ==")
    print(f"  Oracle's Elixir : {len(oracle_champs)} unique champions")
    print(f"  Data Dragon     : {len(dd_display)} champions")
    print(f"  Patch notes     : {len(note_champs)} champions mentioned")

    note_unmatched = sorted(note_champs - oracle_champs)
    if note_unmatched:
        print(f"\n  Patch-note champions NOT matching an Oracle name "
              f"({len(note_unmatched)}) — need name normalisation:")
        for c in note_unmatched:
            print(f"    - {c!r}")
    else:
        print("\n  All patch-note champion names match Oracle names. [ok]")

    dd_unmatched = sorted(oracle_champs - dd_display)
    if dd_unmatched:
        print(f"\n  Oracle champions NOT in Data Dragon display names "
              f"({len(dd_unmatched)}):")
        for c in dd_unmatched:
            print(f"    - {c!r}")

    print("\n== Patch coverage ==")
    oracle_patches = set(load_oracle.unique_patches(
        load_oracle.load_raw(usecols=["patch", "gameid"])))
    mapped = set(vmap.values())
    print(f"  Oracle patches        : {len(oracle_patches)}")
    print(f"  Mapped from notes      : {len(mapped)}")
    missing = sorted(oracle_patches - mapped)
    extra = sorted(mapped - oracle_patches)
    if missing:
        print(f"  In Oracle but no notes: {missing}")
    if extra:
        print(f"  Mapped but not in Oracle: {extra}")
    if not missing and not extra:
        print("  Patch sets align exactly. [ok]")


def _sorted_patches(patches) -> list[str]:
    return sorted(patches, key=lambda s: tuple(int(p) for p in str(s).split(".")))


def build_aligned_dataset() -> "pd.DataFrame":
    """Merge PWDS targets + lag PWDS + stat deltas + patch-note features into one modelling matrix."""
    import pandas as pd

    pwds = pd.read_csv(config.PWDS_CSV, dtype={"patch": str})
    deltas = pd.read_csv(config.STAT_DELTAS_CSV, dtype={"patch": str})
    notes = pd.read_csv(config.PATCH_NOTE_FEATURES_CSV, dtype={"patch": str})

    pwds["champion"] = pwds["champion"].map(normalise_champion_name)
    deltas["champion"] = deltas["champion"].map(normalise_champion_name)
    notes["champion"] = notes["champion"].map(normalise_champion_name)

    for frame in (deltas, notes):
        frame.drop(columns=[c for c in ("season",) if c in frame.columns],
                   inplace=True)

    weight_cols = [f"pwds_{n}" for n in config.PWDS_WEIGHTS]

    patches = _sorted_patches(pwds["patch"].unique())
    patch_idx = {p: i for i, p in enumerate(patches)}
    prev_patch = {
        p: patches[i - 1] for i, p in enumerate(patches)
        if i > 0 and p.split(".")[0] == patches[i - 1].split(".")[0]
    }

    lag = pwds[["champion", "patch", *weight_cols]].copy()
    lag = lag.rename(columns={c: f"lag_{c}" for c in weight_cols})
    lag["patch"] = lag["patch"].map({v: k for k, v in prev_patch.items()})
    lag = lag.dropna(subset=["patch"])

    df = pwds.merge(lag, on=["champion", "patch"], how="left", validate="one_to_one")
    df = df.merge(deltas, on=["champion", "patch"], how="left", validate="one_to_one")
    df = df.merge(notes, on=["champion", "patch"], how="left", validate="one_to_one")

    delta_cols = [c for c in deltas.columns if c not in ("champion", "patch")]
    df[delta_cols] = df[delta_cols].fillna(0)
    note_cols = [c for c in notes.columns if c not in ("champion", "patch")]
    df[note_cols] = df[note_cols].fillna(0)
    df["note_changed"] = (df["n_changes_total"] > 0).astype(int)
    df["stat_changed"] = (df["n_stats_changed"] > 0).astype(int)

    items = pd.read_csv(config.ITEM_FEATURES_CSV, dtype={"patch": str})
    items["champion"] = items["champion"].map(normalise_champion_name)
    items = items.drop(columns=[c for c in ("season",) if c in items.columns])

    df = df.merge(items, on=["champion", "patch"], how="left",
                  indicator="_item_merge", validate="one_to_one")
    unmatched = df.loc[df["_item_merge"] == "left_only", ["champion", "patch"]]
    if not unmatched.empty:
        examples = unmatched.head(5).to_dict("records")
        raise ValueError(
            f"{len(unmatched)} aligned rows have no matching item_features.csv "
            f"row on (champion, patch) -- examples: {examples}")
    df = df.drop(columns=["_item_merge"])

    item_cols = (config.ITEM_CLASS_FEATURE_COLS
                + config.ITEM_CHAMP_FEATURE_COLS
                + config.RUNE_CHAMP_FEATURE_COLS
                + config.SYSTEM_FEATURE_COLS)
    df[item_cols] = df[item_cols].fillna(0)
    df["item_affected"] = ((df["item_champ_n_changed"] > 0)
                           | (df["rune_champ_n_changed"] > 0)).astype(int)

    if not config.SLOT_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"{config.SLOT_FEATURES_CSV} not found. Build order: "
            "`python -m src.features.slot_features` must run after PWDS and "
            "patch-note features, before `python -m src.data.align "
            "--merge`.")
    slots = pd.read_csv(config.SLOT_FEATURES_CSV, dtype={"patch": str})
    slots["champion"] = slots["champion"].map(normalise_champion_name)
    df = df.merge(slots, on=["champion", "patch"], how="left",
                  indicator="_slot_merge", validate="one_to_one")
    unmatched = df.loc[df["_slot_merge"] == "left_only", ["champion", "patch"]]
    if not unmatched.empty:
        examples = unmatched.head(5).to_dict("records")
        raise ValueError(
            f"{len(unmatched)} aligned rows have no matching slot_features.csv "
            f"row on (champion, patch) -- examples: {examples}")
    df = df.drop(columns=["_slot_merge"])
    df[config.SLOT_FEATURE_COLS] = df[config.SLOT_FEATURE_COLS].fillna(0)

    df["has_lag"] = df[f"lag_pwds_{config.DEFAULT_PWDS_WEIGHTS}"].notna().astype(int)

    tgt = f"pwds_{config.DEFAULT_PWDS_WEIGHTS}"
    lag = f"lag_pwds_{config.DEFAULT_PWDS_WEIGHTS}"
    df[f"pwds_delta_{config.DEFAULT_PWDS_WEIGHTS}"] = df[tgt] - df[lag]

    df["patch_idx"] = df["patch"].map(patch_idx)

    df["season"] = df["patch"].map(config.season_of)
    front = ["season", "champion", "patch", "patch_idx"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest].sort_values(["patch_idx", "champion"]).reset_index(drop=True)
    return df


def write_aligned_dataset() -> "pd.DataFrame":
    config.ensure_dirs()
    df = build_aligned_dataset()
    df.to_csv(config.ALIGNED_DATASET_CSV, index=False)
    print(f"aligned_dataset: {len(df):,} rows x {df.shape[1]} cols "
          f"over {df['patch'].nunique()} patches -> {config.ALIGNED_DATASET_CSV}")
    print(f"  rows with a lag PWDS (persistence-usable): {int(df['has_lag'].sum()):,}"
          f" ({df['has_lag'].mean():.1%})")
    print(f"  rows with a base-stat change: {int(df['stat_changed'].sum()):,}")
    print(f"  rows with a patch-note entry: {int(df['note_changed'].sum()):,}"
          f" ({df['note_changed'].mean():.1%})")
    any_aff = ((df['note_changed'] | df['stat_changed']
                | df['item_affected']) > 0)
    test_mask = df["patch"].isin(config.TEST_PATCHES)
    df_test = df[test_mask]
    any_aff_test = any_aff[test_mask]
    print(f"  rows item-affected (full)      : {int(df['item_affected'].sum()):,}"
          f" ({df['item_affected'].mean():.1%})")
    print(f"  rows item-affected (test only) : {int(df_test['item_affected'].sum()):,}"
          f" ({df_test['item_affected'].mean():.1%})")
    print(f"  rows any-affected  (full)      : {int(any_aff.sum()):,} ({any_aff.mean():.1%})")
    print(f"  rows any-affected  (test only) : {int(any_aff_test.sum()):,}"
          f" ({any_aff_test.mean():.1%})")
    for season in config.SEASONS:
        sub = df[df["season"] == season]
        print(f"  {season}: {len(sub):,} rows, {sub['patch'].nunique()} patches, "
              f"has_lag {sub['has_lag'].mean():.1%}")
    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", action="store_true",
                        help="Phase 2C.1 / 2D.2: build aligned_dataset.csv "
                             "(needs pwds.csv + stat_deltas.csv + "
                             "patch_note_features.csv)")
    args = parser.parse_args()

    if args.merge:
        write_aligned_dataset()
        return

    vmap = write_version_map()
    dd_names = write_champion_names()
    validate_alignment(vmap, dd_names)


if __name__ == "__main__":
    main()
