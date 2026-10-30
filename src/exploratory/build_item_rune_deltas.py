"""Build per-patch item and rune tables from Data Dragon and diff consecutive patches."""
from __future__ import annotations

import json
import sys

import pandas as pd

try:
    from src.exploratory import paths
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src.exploratory import paths


def _versions_on_disk(season: str, kind: str) -> list[str]:
    dest = paths.TASK1_RAW / season
    files = dest.glob(f"{kind}_*.json")
    versions = [f.stem[len(kind) + 1:] for f in files]
    return sorted(versions, key=lambda s: tuple(int(p) for p in s.split(".")))


def _load(season: str, kind: str, version: str) -> dict:
    path = paths.TASK1_RAW / season / f"{kind}_{version}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _item_numeric_fields(entry: dict) -> dict:
    """Numeric fields for one item: gold.* plus every numeric stat."""
    fields: dict[str, float] = {}
    gold = entry.get("gold", {}) or {}
    for k in ("base", "total"):
        if isinstance(gold.get(k), (int, float)):
            fields[f"gold_{k}"] = float(gold[k])
    for sk, sv in (entry.get("stats", {}) or {}).items():
        if isinstance(sv, (int, float)):
            fields[f"stat_{sk}"] = float(sv)
    return fields


def build_items(seasons) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_patch_rows = []
    delta_rows = []
    for season in seasons:
        versions = _versions_on_disk(season, "item")
        prev: dict | None = None
        prev_patch: str | None = None
        for v in versions:
            patch = paths.oracle_patch_id(v)
            data = _load(season, "item", v)["data"]
            cur: dict[str, dict] = {}
            for iid, entry in data.items():
                fields = _item_numeric_fields(entry)
                cur[iid] = {
                    "name": entry.get("name", ""),
                    "tags": ";".join(entry.get("tags", []) or []),
                    "purchasable": bool((entry.get("gold", {}) or {}).get("purchasable", False)),
                    "fields": fields,
                }
                row = {
                    "season": season, "patch": patch, "version": v,
                    "item_id": iid, "name": cur[iid]["name"],
                    "tags": cur[iid]["tags"], "purchasable": cur[iid]["purchasable"],
                }
                row.update(fields)
                by_patch_rows.append(row)
            if prev is not None:
                delta_rows.extend(_diff_items(prev, cur, season, patch, prev_patch))
            prev, prev_patch = cur, patch
    items_by_patch = pd.DataFrame(by_patch_rows)
    item_deltas = pd.DataFrame(delta_rows)
    return items_by_patch, item_deltas


def _diff_items(prev, cur, season, patch, prev_patch) -> list[dict]:
    rows = []
    for iid, c in cur.items():
        p = prev.get(iid)
        if p is None:
            rows.append(_delta_row(season, patch, prev_patch, iid, c["name"],
                                   "item", None, None, "added"))
            continue
        keys = set(p["fields"]) | set(c["fields"])
        for f in sorted(keys):
            before = p["fields"].get(f)
            after = c["fields"].get(f)
            if before == after:
                continue
            if before is None:
                direction = "added"
            elif after is None:
                direction = "removed"
            elif after > before:
                direction = "up"
            else:
                direction = "down"
            rows.append(_delta_row(season, patch, prev_patch, iid, c["name"],
                                   f, before, after, direction))
    for iid, p in prev.items():
        if iid not in cur:
            rows.append(_delta_row(season, patch, prev_patch, iid, p["name"],
                                   "item", None, None, "removed"))
    return rows


def _delta_row(season, patch, prev_patch, iid, name, field, before, after, direction):
    delta = None
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        delta = after - before
    return {
        "season": season, "patch": patch, "prev_patch": prev_patch,
        "item_id": iid, "name": name, "field": field,
        "before": before, "after": after, "delta": delta, "direction": direction,
    }


def build_runes(seasons) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_patch_rows = []
    delta_rows = []
    for season in seasons:
        versions = _versions_on_disk(season, "runesReforged")
        prev: dict | None = None
        prev_patch: str | None = None
        for v in versions:
            patch = paths.oracle_patch_id(v)
            trees = _load(season, "runesReforged", v)
            cur: dict[str, dict] = {}
            for tree in trees:
                for si, slot in enumerate(tree.get("slots", [])):
                    for rune in slot.get("runes", []):
                        rid = str(rune["id"])
                        rec = {
                            "tree_id": tree["id"], "tree_name": tree["name"],
                            "tree_key": tree["key"], "slot_index": si,
                            "rune_id": rid, "rune_key": rune.get("key", ""),
                            "rune_name": rune.get("name", ""),
                            "short_desc": rune.get("shortDesc", ""),
                        }
                        cur[rid] = rec
                        by_patch_rows.append({"season": season, "patch": patch,
                                              "version": v, **rec})
            if prev is not None:
                delta_rows.extend(_diff_runes(prev, cur, season, patch, prev_patch))
            prev, prev_patch = cur, patch
    runes_by_patch = pd.DataFrame(by_patch_rows)
    rune_deltas = pd.DataFrame(delta_rows)
    return runes_by_patch, rune_deltas


def _diff_runes(prev, cur, season, patch, prev_patch) -> list[dict]:
    rows = []
    for rid, c in cur.items():
        p = prev.get(rid)
        if p is None:
            rows.append(_rune_delta(season, patch, prev_patch, c, "added", None, None))
            continue
        if c["rune_name"] != p["rune_name"]:
            rows.append(_rune_delta(season, patch, prev_patch, c, "renamed",
                                    p["rune_name"], c["rune_name"]))
        if (c["tree_id"], c["slot_index"]) != (p["tree_id"], p["slot_index"]):
            rows.append(_rune_delta(
                season, patch, prev_patch, c, "moved",
                f"{p['tree_key']}/slot{p['slot_index']}",
                f"{c['tree_key']}/slot{c['slot_index']}"))
        if c["short_desc"] != p["short_desc"]:
            rows.append(_rune_delta(season, patch, prev_patch, c, "desc_changed",
                                    None, None))
    for rid, p in prev.items():
        if rid not in cur:
            rows.append(_rune_delta(season, patch, prev_patch, p, "removed", None, None))
    return rows


def _rune_delta(season, patch, prev_patch, rec, change_type, before, after):
    return {
        "season": season, "patch": patch, "prev_patch": prev_patch,
        "rune_id": rec["rune_id"], "rune_name": rec["rune_name"],
        "tree_key": rec["tree_key"], "change_type": change_type,
        "before": before, "after": after,
    }


def main() -> None:
    seasons = sorted(paths.SEASONS.keys())
    paths.TASK1.mkdir(parents=True, exist_ok=True)

    items_by_patch, item_deltas = build_items(seasons)
    runes_by_patch, rune_deltas = build_runes(seasons)

    out = {
        "items_by_patch.csv": items_by_patch,
        "item_deltas.csv": item_deltas,
        "runes_by_patch.csv": runes_by_patch,
        "rune_deltas.csv": rune_deltas,
    }
    for name, df in out.items():
        df.to_csv(paths.TASK1 / name, index=False, encoding="utf-8")

    print("=== Task 1: item/rune tables (quarantine) ===")
    for name, df in out.items():
        patches = df["patch"].nunique() if "patch" in df else 0
        print("  %-20s rows=%-7d patches=%d" % (name, len(df), patches))
    if not item_deltas.empty:
        print("  item_deltas by direction:",
              item_deltas["direction"].value_counts().to_dict())
    if not rune_deltas.empty:
        print("  rune_deltas by change_type:",
              rune_deltas["change_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
