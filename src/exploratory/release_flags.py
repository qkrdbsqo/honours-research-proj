"""Champion release and rework flags from Data Dragon."""
from __future__ import annotations

import functools
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src import config
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src import config

WINDOW = 3
FLAG_COLS = ("recent_release", "recent_rework")
OUT = config.OUTPUTS / "release_flags_20261004"
ORDER = {p: i for i, p in enumerate(config.ALL_PATCHES)}


def _version_key(path: str) -> tuple[int, int]:
    v = Path(path).stem.split(".")
    return int(v[0]), int(v[1])


def _label(major: int, minor: int) -> str:
    return f"{major}.{minor:02d}"


def event_patch(major: int, minor: int) -> str:
    """First ALL_PATCHES label at or after Data Dragon version major.minor."""
    target = (major, minor)
    for p in config.ALL_PATCHES:
        a, b = (int(x) for x in p.split("."))
        if (a, b) >= target:
            return p
    raise ValueError(f"no scored patch at or after {major}.{minor}")


@functools.lru_cache(maxsize=None)
def _names() -> dict:
    return json.loads(config.CHAMPION_NAMES.read_text(encoding="utf-8"))


def _display(cid: str) -> str:
    from src.data.align import normalise_champion_name as norm
    return norm(_names().get(cid, cid))


@functools.lru_cache(maxsize=None)
def events() -> pd.DataFrame:
    files = sorted(glob.glob(str(config.DDRAGON_DIR / "*.json")), key=_version_key)
    if len(files) < 2:
        raise FileNotFoundError("Data Dragon snapshots missing")
    rows, prev = [], None
    for f in files:
        data = json.loads(Path(f).read_text(encoding="utf-8"))["data"]
        major, minor = _version_key(f)
        if prev is not None:
            for cid in sorted(set(data) - set(prev)):
                rows.append({"champion": _display(cid), "kind": "release",
                             "ddragon_version": _label(major, minor),
                             "event_patch": event_patch(major, minor)})
            for cid in sorted(set(data) & set(prev)):
                a, b = prev[cid], data[cid]
                if a.get("title") != b.get("title") or a.get("info") != b.get("info"):
                    rows.append({"champion": _display(cid), "kind": "rework",
                                 "ddragon_version": _label(major, minor),
                                 "event_patch": event_patch(major, minor)})
        prev = data
    return pd.DataFrame(rows, columns=["champion", "kind", "ddragon_version", "event_patch"])


@functools.lru_cache(maxsize=None)
def _universe() -> tuple[str, ...]:
    files = sorted(glob.glob(str(config.DDRAGON_DIR / "*.json")), key=_version_key)
    ids = set()
    for f in files:
        ids |= set(json.loads(Path(f).read_text(encoding="utf-8"))["data"])
    return tuple(sorted({_display(c) for c in ids}))


@functools.lru_cache(maxsize=None)
def flags(window: int = WINDOW) -> pd.DataFrame:
    ev = events()
    rel = {r.champion: ORDER[r.event_patch] for r in ev[ev["kind"] == "release"].itertuples()}
    rew: dict[str, list[int]] = {}
    for r in ev[ev["kind"] == "rework"].itertuples():
        rew.setdefault(r.champion, []).append(ORDER[r.event_patch])
    rows = []
    for champ in _universe():
        for p, i in ORDER.items():
            since = i - rel[champ] if champ in rel else np.nan
            rw = [i - j for j in rew.get(champ, []) if 0 <= i - j]
            rows.append({
                "champion": champ, "patch": p,
                "released": int(champ not in rel or since >= 0),
                "patches_since_release": since,
                "recent_release": int(champ in rel and 0 <= since <= window),
                "recent_rework": int(bool(rw) and min(rw) <= window),
            })
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = events()
    fl = flags()
    print(ev.to_string(index=False))
    print(f"flag rows: {len(fl)}; recent_release {int(fl.recent_release.sum())}; "
          f"recent_rework {int(fl.recent_rework.sum())}; unreleased cells {int((fl.released == 0).sum())}")
    ev.to_csv(OUT / "events.csv", index=False)
    fl.to_csv(OUT / "flags.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
