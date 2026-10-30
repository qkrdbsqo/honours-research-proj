"""Semantic encoding of champion changes: direction and size by gameplay meaning."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src import config
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src import config

from src.data import align

OUT = config.OUTPUTS / "semantic_20261004"
SEM_COLS = ["sem_n_buffs", "sem_n_nerfs", "sem_net_power", "sem_power_ult",
            "sem_power_basic", "sem_power_passive", "sem_power_stats", "sem_n_ambiguous"]
REPLACED = ["n_buffs", "n_nerfs", "mean_pct_change", "sum_signed_pct",
            "max_buff_pct", "max_nerf_pct", "buff_ratio", "is_adjustment"]

_LOWER = re.compile(r"cooldown(?!\s*(refund|reduction))|cost|cast time|windup|delay|recharge|lockout|penalty"
                    r"|charge time|time spent|time to|travel time|falloff|timer", re.I)
_AMBIG = re.compile(r"threshold|requirement|damage taken|speed reduction", re.I)
_NUM = re.compile(r"\d*\.\d+|\d+")


def polarity(stat: str) -> int:
    """+1 higher is better, -1 lower is better, 0 ambiguous."""
    s = str(stat)
    if _AMBIG.search(s):
        return 0
    if _LOWER.search(s):
        return -1
    return 1


def _label(text: str) -> str:
    return re.sub(r"[^a-z]", "", re.sub(_NUM, "", text.lower()))


def components(text: str) -> list[tuple[str, list[float], bool]]:
    """[(label, numbers, has_percent)]: the base value first, then labelled parts."""
    text = str(text)
    parts = re.split(r"\(|\)|\s\+\s", text)
    out = []
    for i, p in enumerate(parts):
        nums = [float(x) for x in _NUM.findall(p)]
        if not nums:
            continue
        out.append(("__base__" if not out else _label(p), nums, "%" in p))
    return out


def _rel(b: list[float], a: list[float]) -> float:
    if len(b) == len(a):
        diffs = [((y - x) / x) if x != 0 else float(np.sign(y)) for x, y in zip(b, a)]
        r = float(np.mean(diffs))
    else:
        mb, ma = float(np.mean(b)), float(np.mean(a))
        r = (ma - mb) / mb if mb != 0 else float(np.sign(ma))
    return float(np.clip(r, -1.0, 1.0))


def relative_change(before: str, after: str) -> float | None:
    """Mean relative change over changed, comparable components; None if none."""
    cb, ca = components(before), components(after)
    if not cb or not ca:
        return None
    if cb[0][0] == "__base__" and ca[0][0] == "__base__" and cb[0][2] != ca[0][2]:
        return None  # units differ on the base value
    bmap = {lab: n for lab, n, _ in cb}
    rels, compared = [], 0
    for lab, n, _ in ca:
        if lab in bmap:
            compared += 1
            if bmap[lab] != n:
                rels.append(_rel(bmap[lab], n))
    if compared == 0:
        return None
    return float(np.mean(rels)) if rels else 0.0


def encode_line(ability: str, stat: str, before: str, after: str) -> dict:
    pol = polarity(stat)
    rel = relative_change(before, after)
    if pol == 0 or rel is None:
        return {"polarity": pol, "rel_change": rel, "signed_power": np.nan, "sem_direction": "ambiguous"}
    p = pol * rel
    d = "buff" if p > 0 else "nerf" if p < 0 else "none"
    return {"polarity": pol, "rel_change": rel, "signed_power": p, "sem_direction": d}


def _group(ability: str) -> str:
    a = str(ability)
    return {"R": "ult", "Passive": "passive", "Stats": "stats"}.get(a, "basic")


def load_lines() -> pd.DataFrame:
    vmap = json.loads(config.VERSION_MAP.read_text())
    rows = []
    for year in (2024, 2025):
        for p in json.loads((config.PATCH_NOTES_DIR / f"lol_patches_{year}.json").read_text(encoding="utf-8")):
            patch = vmap.get(p["patch_version"])
            for c in p.get("champions", []):
                for ch in c.get("changes", []):
                    rows.append({"patch": patch, "champion": align.normalise_champion_name(c["champion"]),
                                 "ability": ch.get("ability"), "stat": ch.get("stat"),
                                 "before": ch.get("before"), "after": ch.get("after"),
                                 "stored_direction": ch.get("direction")})
    d = pd.DataFrame(rows)
    enc = pd.DataFrame([encode_line(r.ability, r.stat, r.before, r.after) for r in d.itertuples()])
    d = pd.concat([d, enc], axis=1)
    d["group"] = d["ability"].map(_group)
    return d


def features(lines: pd.DataFrame | None = None) -> pd.DataFrame:
    d = load_lines() if lines is None else lines
    d = d.dropna(subset=["patch"])
    sp = d["signed_power"]
    d = d.assign(_buff=(d["sem_direction"] == "buff").astype(int),
                 _nerf=(d["sem_direction"] == "nerf").astype(int),
                 _amb=(d["sem_direction"] == "ambiguous").astype(int),
                 _p=sp.fillna(0.0))
    g = d.groupby(["champion", "patch"])
    out = pd.DataFrame({
        "sem_n_buffs": g["_buff"].sum(), "sem_n_nerfs": g["_nerf"].sum(),
        "sem_net_power": g["_p"].sum(), "sem_n_ambiguous": g["_amb"].sum()})
    for grp in ("ult", "basic", "passive", "stats"):
        out[f"sem_power_{grp}"] = d[d["group"] == grp].groupby(["champion", "patch"])["_p"].sum()
    out = out.fillna(0.0).reset_index()
    return out[["champion", "patch"] + SEM_COLS]


def audit_agreement(lines: pd.DataFrame | None = None) -> dict:
    """Compare semantic and stored directions with the cooldown audit's rule."""
    d = load_lines() if lines is None else lines
    a = pd.read_csv(config.DATA / "cooldown_audit.csv",
                    dtype={"patch": str})
    a = a[a["resolved_numeric_direction"] & a["audit_direction"].isin(["buff", "nerf"])]
    vmap = json.loads(config.VERSION_MAP.read_text())
    a["patch"] = a["patch"].map(lambda p: vmap.get(p, p))
    a["champion"] = a["champion"].map(align.normalise_champion_name)
    keys = ["patch", "champion", "ability", "stat", "before", "after"]
    m = a.merge(d[keys + ["sem_direction"]].drop_duplicates(keys), on=keys, how="left")
    matched = m["sem_direction"].notna()
    return {"audit_resolved_buff_nerf": len(a), "matched": int(matched.sum()),
            "semantic_agrees": int((m.loc[matched, "sem_direction"] == m.loc[matched, "audit_direction"]).sum()),
            "stored_agrees": int((m.loc[matched, "stored_direction"] == m.loc[matched, "audit_direction"]).sum())}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_lines()
    f = features(d)
    ag = audit_agreement(d)
    print(d["sem_direction"].value_counts().to_string())
    print(pd.crosstab(d["stored_direction"], d["sem_direction"]).to_string())
    print(ag)
    d.to_csv(OUT / "lines.csv", index=False)
    f.to_csv(OUT / "features.csv", index=False)
    pd.DataFrame([ag]).to_csv(OUT / "audit_agreement.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
