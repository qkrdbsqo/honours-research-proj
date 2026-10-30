"""Item and rune change features: class-level, champion-routed and system counts.

Usage:
    python -m src.features.item_features
"""
from __future__ import annotations

import json
import sys

import pandas as pd

try:
    from src import config
    from src.data import align
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.data import align

GOLD_FIELDS = {"gold_base", "gold_total"}

DIRECTION_SIGN = {"up": +1, "down": -1, "buff": +1, "nerf": -1}


def field_sign(field: str, direction: str) -> int:
    """Signed impact of a change: +1 buff, -1 nerf, 0 unknown/neutral."""
    base = DIRECTION_SIGN.get(str(direction).strip().lower(), 0)
    if base == 0:
        return 0
    return -base if field in GOLD_FIELDS else base


def element_key(name: str) -> str:
    """Canonical join key for an item or rune name."""
    return align.normalise_champion_name(str(name)).lower().strip()


def _from_ddragon() -> pd.DataFrame:
    """Item changes from the Data Dragon per-patch diffs."""
    df = pd.read_csv(config.ITEM_DELTAS_SRC, dtype={"patch": str,
                                                    "prev_patch": str})
    df["sign"] = [field_sign(f, d)
                  for f, d in zip(df["field"], df["direction"])]
    before = df["before"].abs().replace(0, pd.NA)
    df["magnitude"] = (df["delta"].abs() / before).fillna(1.0).clip(upper=10.0)
    df["kind"] = "item"
    df["element_key"] = df["name"].map(element_key)
    df["source"] = "ddragon"
    return df[["season", "patch", "kind", "name", "element_key",
               "sign", "magnitude", "source"]].rename(
                   columns={"name": "element_name"})


RUNE_NAME_ALIASES: dict[str, str] = {
    "unflinching rune": "unflinching",
    "unsealed spellbook (smite)": "unsealed spellbook",
    "bloodline": "legend: bloodline",
    "biscuit of everlasting will": "biscuit delivery",
}


def _valid_rune_keys() -> set[str]:
    """Keys of every real rune name in the Data Dragon rune snapshots."""
    df = pd.read_csv(config.RUNES_BY_PATCH_SRC, dtype={"patch": str})
    return set(df["rune_name"].map(element_key))


def _is_valid_rune(key: str, valid_keys: set[str]) -> bool:
    """Whether a key is a real rune, allowing for scraper name variants and stat shards."""
    if key in valid_keys:
        return True
    if RUNE_NAME_ALIASES.get(key) in valid_keys:
        return True
    return key.endswith(" shard")


def _from_sections() -> pd.DataFrame:
    """Item and rune changes from the scraped Items/Runes sections."""
    rows: list[dict] = []
    for season in config.SEASONS:
        path = config.PATCH_SECTIONS_JSON[season]
        if not path.exists():
            print(f"  [warn] {path.name} missing -- run "
                  "src.data.scrape_patch_sections first")
            continue
        for page in json.loads(path.read_text(encoding="utf-8")):
            patch = page.get("patch")
            for sec in page.get("sections", []):
                kind = {"items": "item", "runes": "rune"}.get(sec["section"])
                if kind is None:
                    continue          # systems handled separately
                for entry in sec["entries"]:
                    changes = entry["changes"] or [{}]
                    for ch in changes:
                        rows.append({
                            "season": int(season), "patch": patch, "kind": kind,
                            "element_name": entry["element_name"],
                            "element_key": element_key(entry["element_name"]),
                            "sign": DIRECTION_SIGN.get(
                                str(ch.get("direction", "")).lower(), 0),
                            "magnitude": 1.0,
                            "source": "patch_notes",
                        })
    df = pd.DataFrame(rows, columns=["season", "patch", "kind",
                                     "element_name", "element_key",
                                     "sign", "magnitude", "source"])
    rune_mask = df["kind"] == "rune"
    if rune_mask.any():
        valid_keys = _valid_rune_keys()
        is_valid = df["element_key"].map(lambda k: _is_valid_rune(k, valid_keys))
        junk_mask = rune_mask & ~is_valid
        if junk_mask.any():
            dropped = sorted(df.loc[junk_mask, "element_name"].unique())
            print(f"  [warn] dropping {int(junk_mask.sum())} non-rune rows "
                  f"from the Runes section (name not found in "
                  f"{config.RUNES_BY_PATCH_SRC.name}):")
            for name in dropped:
                print(f"    - {name}")
        df = df[~junk_mask].reset_index(drop=True)
    return df


def build_change_table() -> pd.DataFrame:
    """Merged item/rune change table across both sources."""
    dd, sec = _from_ddragon(), _from_sections()
    both = pd.concat([dd, sec], ignore_index=True)
    key = ["patch", "kind", "element_key"]
    authoritative = set(map(tuple, sec[key].drop_duplicates().to_numpy()))
    keep = [not (tuple(r) in authoritative and s == "ddragon")
            for r, s in zip(both[key].to_numpy(), both["source"])]
    out = both[keep].copy()
    dd_keys = set(map(tuple, dd[key].drop_duplicates().to_numpy()))
    sec_keys = set(map(tuple, sec[key].drop_duplicates().to_numpy()))
    genuine_both = dd_keys & sec_keys
    out.loc[[tuple(r) in genuine_both for r in out[key].to_numpy()],
            "source"] = "both"
    return out.sort_values(["patch", "kind", "element_name"]).reset_index(drop=True)


def system_changes_by_patch() -> pd.DataFrame:
    """Number of system changes per patch."""
    rows: list[dict] = []
    for season in config.SEASONS:
        path = config.PATCH_SECTIONS_JSON[season]
        if not path.exists():
            continue
        for page in json.loads(path.read_text(encoding="utf-8")):
            n = sum(len(s["entries"]) for s in page.get("sections", [])
                    if s["section"] == "systems")
            rows.append({"patch": page.get("patch"),
                         "patch_n_system_changes": n})
    return pd.DataFrame(rows).drop_duplicates("patch")


def _distinct_item_changes(changes: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct changed item or rune, so routing counts items rather than change rows."""
    g = (changes.groupby(["patch", "kind", "element_key"], as_index=False)
                 ["sign"].sum())
    g["net_sign"] = g["sign"].map(lambda s: (s > 0) - (s < 0))
    return g[["patch", "kind", "element_key", "net_sign"]]


def build_class_features(changes: pd.DataFrame) -> pd.DataFrame:
    """Route item changes onto champions via class -> item-tag affinity."""
    id_to_name = json.loads(config.CHAMPION_NAMES.read_text(encoding="utf-8"))

    aff = pd.read_csv(config.CLASS_AFFINITY_SRC, dtype={"patch": str})
    items = pd.read_csv(config.ITEMS_BY_PATCH_SRC, dtype={"patch": str})
    items["element_key"] = items["name"].map(element_key)
    tags = (items[["patch", "element_key", "tags"]].drop_duplicates(
        ["patch", "element_key"]))

    distinct = _distinct_item_changes(changes[changes["kind"] == "item"])
    ch = distinct.merge(tags, on=["patch", "element_key"], how="left")
    ch["tag_set"] = ch["tags"].fillna("").str.split(";").map(
        lambda xs: {t.strip() for t in xs if t.strip()})

    by_patch: dict[str, list[tuple[set, int]]] = {}
    for patch, grp in ch.groupby("patch"):
        by_patch[patch] = list(zip(grp["tag_set"], grp["net_sign"]))

    rows: list[dict] = []
    for r in aff.itertuples(index=False):
        want = {t.strip() for t in str(r.relevant_item_tags).split(";")
                if t.strip()}
        hits = [s for tset, s in by_patch.get(r.patch, []) if tset & want]
        champ = align.normalise_champion_name(
            id_to_name.get(r.champion, r.champion))
        rows.append({
            "season": str(r.season), "patch": r.patch,
            "champion": champ,
            "item_class_n_changed": len(hits),
            "item_class_n_buffs": sum(1 for s in hits if s > 0),
            "item_class_n_nerfs": sum(1 for s in hits if s < 0),
            "item_class_net_signed": float(sum(hits)),
        })
    return pd.DataFrame(rows)


def _prev_patch_map() -> dict[str, str]:
    """patch -> previous patch, never crossing the season boundary."""
    order = config.ALL_PATCHES
    return {p: order[i - 1] for i, p in enumerate(order)
            if i > 0 and p.split(".")[0] == order[i - 1].split(".")[0]}


def build_champ_features(changes: pd.DataFrame) -> pd.DataFrame:
    """Route item/rune changes onto champions through their actual pro builds."""
    use = pd.read_csv(config.LEAGUEPEDIA_USAGE_SRC,
                       dtype={"patch": str, "patch_client": str})
    use = use.drop(columns=["patch"]).rename(columns={"patch_client": "patch"})
    use["champion"] = use["champion"].map(align.normalise_champion_name)
    use["element_key"] = use["element_name"].map(element_key)

    prev = _prev_patch_map()
    nxt = {v: k for k, v in prev.items()}
    use["target_patch"] = use["patch"].map(nxt)
    use = use.dropna(subset=["target_patch"])

    ch = (_distinct_item_changes(changes)
          .rename(columns={"patch": "target_patch"}))
    m = use.merge(ch, on=["target_patch", "kind", "element_key"], how="inner")

    out: dict[tuple, dict] = {}
    for (patch, champ, kind), grp in m.groupby(["target_patch", "champion",
                                                "kind"]):
        pref = "item_champ" if kind == "item" else "rune_champ"
        rec = out.setdefault((patch, champ), {})
        rec[f"{pref}_n_changed"] = len(grp)
        rec[f"{pref}_n_buffs"] = int((grp["net_sign"] > 0).sum())
        rec[f"{pref}_n_nerfs"] = int((grp["net_sign"] < 0).sum())
        rec[f"{pref}_exposure"] = float(grp["share"].sum())
        if kind == "item":
            rec["item_champ_net_signed_weighted"] = float(
                (grp["net_sign"] * grp["share"]).sum())
            rec["item_champ_max_share_changed"] = float(grp["share"].max())

    seen = set(map(tuple, use[["target_patch", "champion"]]
                   .drop_duplicates().to_numpy()))

    rows = []
    for (patch, champ), rec in out.items():
        rows.append({"patch": patch, "champion": champ,
                     "has_build_lag": 1, **rec})
    df = pd.DataFrame(rows)
    lagged = pd.DataFrame([{"patch": p, "champion": c, "has_build_lag": 1}
                           for p, c in seen])
    df = lagged.merge(df.drop(columns=["has_build_lag"]),
                      on=["patch", "champion"], how="left")
    df["season"] = df["patch"].map(config.season_of)
    return df


def build_all() -> pd.DataFrame:
    """Assemble every item/rune/system feature, keyed (season, patch, champion)."""
    changes = build_change_table()
    changes.to_csv(config.ITEM_CHANGES_CSV, index=False)

    cls = build_class_features(changes)
    champ = build_champ_features(changes)
    sysf = system_changes_by_patch()

    df = cls.merge(champ.drop(columns=["season"]),
                   on=["patch", "champion"], how="left")
    df = df.merge(sysf, on="patch", how="left")

    numeric = (config.ITEM_CLASS_FEATURE_COLS + config.ITEM_CHAMP_FEATURE_COLS
               + config.RUNE_CHAMP_FEATURE_COLS + config.SYSTEM_FEATURE_COLS)
    for col in numeric:
        if col not in df.columns:
            df[col] = 0.0
    df[numeric] = df[numeric].fillna(0)
    df["has_build_lag"] = df["has_build_lag"].astype(int)

    front = ["season", "patch", "champion"]
    return df[front + numeric].sort_values(["patch", "champion"]).reset_index(
        drop=True)


def main() -> None:
    config.ensure_dirs()
    df = build_all()
    df.to_csv(config.ITEM_FEATURES_CSV, index=False)
    print(f"item_features: {len(df):,} rows x {df.shape[1]} cols over "
          f"{df['patch'].nunique()} patches")
    print(f"  class channel fires on : "
          f"{int((df['item_class_n_changed'] > 0).sum()):,} rows "
          f"({(df['item_class_n_changed'] > 0).mean():.1%})")
    print(f"  champ channel fires on : "
          f"{int((df['item_champ_n_changed'] > 0).sum()):,} rows "
          f"({(df['item_champ_n_changed'] > 0).mean():.1%})")
    print(f"  rune channel fires on  : "
          f"{int((df['rune_champ_n_changed'] > 0).sum()):,} rows "
          f"({(df['rune_champ_n_changed'] > 0).mean():.1%})")
    print(f"  rows with a P-1 build  : {int(df['has_build_lag'].sum()):,} "
          f"({df['has_build_lag'].mean():.1%})")
    print(f"  -> {config.ITEM_FEATURES_CSV}")


if __name__ == "__main__":
    main()
