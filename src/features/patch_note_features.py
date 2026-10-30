"""Patch-note features: change counts, signed magnitudes, adjustment flags and ability flags.

Usage:
    python -m src.features.patch_note_features
"""
from __future__ import annotations

import json
import sys

import pandas as pd

try:
    from src import config
    from src.data import align
    from src.data.align import normalise_champion_name
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.data import align
    from src.data.align import normalise_champion_name

FEATURE_COLS = list(config.PATCH_NOTE_FEATURE_COLS)


def _numeric_segment(value: str) -> str:
    """Take the leading numeric portion of a stat value."""
    s = str(value).strip()
    for sep in ("+", "("):
        idx = s.find(sep)
        if idx != -1:
            s = s[:idx]
    return s.strip()


def _parse_ranks(value: str) -> list[float] | None:
    """Parse a (possibly per-rank) value to a list of floats, or None if it isn't cleanly numeric."""
    seg = _numeric_segment(value)
    if not seg:
        return None
    out: list[float] = []
    for part in seg.split("/"):
        token = part.strip().rstrip("%").replace(",", "").strip()
        if not token:
            return None
        try:
            out.append(float(token))
        except ValueError:
            return None
    return out or None


def _pct_change(before: str, after: str) -> float | None:
    """Signed percentage change between two stat values."""
    b = _parse_ranks(before)
    a = _parse_ranks(after)
    if b is None or a is None or len(b) != len(a):
        return None
    pcts = [(av - bv) / abs(bv) for bv, av in zip(b, a) if bv != 0]
    if not pcts:
        return None
    return sum(pcts) / len(pcts)


def _features_for_champion(changes: list[dict]) -> dict:
    """Compute the numeric feature dict for one champion's list of changes."""
    n_buffs = sum(1 for c in changes if c.get("direction") == "buff")
    n_nerfs = sum(1 for c in changes if c.get("direction") == "nerf")
    n_total = len(changes)
    abilities = {str(c.get("ability", "")).strip() for c in changes}
    abilities.discard("")

    pcts = [p for p in (_pct_change(c.get("before", ""), c.get("after", ""))
                        for c in changes) if p is not None]
    buffs = [p for p in pcts if p > 0]
    nerfs = [p for p in pcts if p < 0]

    def touched(*names: str) -> int:
        return int(any(a in abilities for a in names))

    return {
        "n_buffs": n_buffs,
        "n_nerfs": n_nerfs,
        "n_changes_total": n_total,
        "n_abilities_changed": len(abilities),
        "mean_pct_change": sum(pcts) / len(pcts) if pcts else 0.0,
        "sum_signed_pct": sum(pcts),
        "max_buff_pct": max(buffs) if buffs else 0.0,
        "max_nerf_pct": min(nerfs) if nerfs else 0.0,   # stored negative
        "n_parseable_changes": len(pcts),
        "is_adjustment": int(n_buffs > 0 and n_nerfs > 0),
        "buff_ratio": n_buffs / n_total if n_total else 0.0,
        "passive_changed": touched("Passive"),
        "q_changed": touched("Q"),
        "w_changed": touched("W"),
        "e_changed": touched("E"),
        "r_changed": touched("R"),
        "base_stats_changed": touched("Stats", "Base Stats"),
    }


def build_patch_note_features() -> pd.DataFrame:
    """One row per (champion, patch) that has a patch-note entry, for every season."""
    vmap = json.loads(config.VERSION_MAP.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for season in config.SEASONS:
        path = config.PATCH_NOTES_DIR / f"lol_patches_{season}.json"
        notes = json.loads(path.read_text(encoding="utf-8"))
        for entry in notes:
            patch = vmap.get(entry["patch_version"])
            if patch is None:
                print(f"  [warn] no version_map entry for "
                      f"{entry['patch_version']!r} -- skipped")
                continue
            for champ in entry["champions"]:
                feats = _features_for_champion(champ.get("changes", []))
                rows.append({
                    "season": season,
                    "champion": align.normalise_champion_name(champ["champion"]),
                    "patch": patch,
                    **feats,
                })
    df = pd.DataFrame(rows, columns=["season", "champion", "patch", *FEATURE_COLS])
    return df.sort_values(["patch", "champion"]).reset_index(drop=True)


def main() -> None:
    config.ensure_dirs()
    df = build_patch_note_features()
    df.to_csv(config.PATCH_NOTE_FEATURES_CSV, index=False)

    print(f"patch_note_features: {len(df):,} champion-patch rows "
          f"({df['patch'].nunique()} patches)")
    print(f"  total changes captured: {int(df['n_changes_total'].sum()):,} "
          f"(buffs {int(df['n_buffs'].sum()):,} | nerfs {int(df['n_nerfs'].sum()):,})")
    print(f"  rows with >=1 parseable magnitude: "
          f"{int((df['n_parseable_changes'] > 0).sum()):,} | "
          f"adjustments (buff+nerf): {int(df['is_adjustment'].sum()):,}")
    print(f"  parseable changes: {int(df['n_parseable_changes'].sum()):,} "
          f"of {int(df['n_changes_total'].sum()):,} total "
          f"({df['n_parseable_changes'].sum() / df['n_changes_total'].sum():.1%})")
    print(f"  -> {config.PATCH_NOTE_FEATURES_CSV}")


if __name__ == "__main__":
    main()
