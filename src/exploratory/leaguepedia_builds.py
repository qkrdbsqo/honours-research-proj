"""Download professional item and rune usage per champion and patch from Leaguepedia's Cargo API.

Needs a Leaguepedia bot login in secrets/lp_bot.json.

Usage:
    python -m src.exploratory.leaguepedia_builds --smoke
    python -m src.exploratory.leaguepedia_builds --pilot --patch 14.1
    python -m src.exploratory.leaguepedia_builds --full
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

try:
    from src.exploratory import paths
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src.exploratory import paths

UA = "lol-patch-prediction/1.0 (contact: ttongkking2003@gmail.com)"

PAGE = 500                # Cargo hard max rows per request
REQUEST_SPACING = 1.5     # polite gap between authenticated requests (seconds)
BACKOFFS = [30, 60, 120]  # exponential backoff on ratelimited / 429
MAX_RETRIES_PER_PAGE = 6  # cap per page before aborting (headroom for long runs)

TASK3 = paths.EXPLORATORY / "leaguepedia"
CACHE = TASK3 / "cache" / "builds"     # cache/builds/patch=<p>/offset=<n>.json
CRED_FILE = paths.ROOT / "secrets" / "lp_bot.json"

_REQS = [0]   # running network-request counter (shown in logs)


class CargoError(RuntimeError):
    """A Cargo page failed after the retry cap."""


def _log(msg: str) -> None:
    print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _load_credentials() -> tuple[str, str]:
    """Return (username, password) from env or the gitignored file, else stop."""
    u = os.environ.get("LP_BOT_USER")
    p = os.environ.get("LP_BOT_PASS")
    if u and p:
        return u, p
    if CRED_FILE.exists():
        try:
            d = json.loads(CRED_FILE.read_text(encoding="utf-8"))
        except ValueError:
            d = {}
        u, p = d.get("username", ""), d.get("password", "")
        placeholder = (not u or not p
                       or "YourWikiUsername" in u or "PASTE_BOT_PASSWORD" in p)
        if not placeholder:
            return u, p
    _log("STOP: no Leaguepedia bot credentials found.")
    _log("  Provide them one of two ways (do NOT paste the password anywhere public):")
    _log("    1) env vars: set LP_BOT_USER and LP_BOT_PASS, or")
    _log("    2) fill %s" % CRED_FILE)
    _log("       {\"username\": \"WikiName@BotName\", \"password\": \"<bot password>\"}")
    sys.exit(3)


def make_client():
    """Authenticate once; return an EsportsClient."""
    from river_mwclient.esports_client import EsportsClient
    from river_mwclient.auth_credentials import AuthCredentials
    user, pw = _load_credentials()
    creds = AuthCredentials(username=user, password=pw)
    try:
        ec = EsportsClient("lol", credentials=creds, clients_useragent=UA)
    except Exception as e:
        msg = str(e).replace(pw, "***") if pw else str(e)
        raise CargoError("login failed: %s: %s" % (type(e).__name__, msg[:200]))
    _log("authenticated to Leaguepedia as bot user (session established).")
    return ec


FIELDS = ("SG.Patch=Patch,SG.DateTime_UTC=Date,SP.Link=Player,"
          "SP.Champion=Champion,SP.Role=Role,SP.Items=Items,SP.Runes=Runes")


def _query(ec, patch: str, offset: int) -> list[dict]:
    """One authenticated cargo page (list of row dicts) with backoff/retry."""
    import mwclient.errors as mwerr
    params = dict(
        tables="ScoreboardPlayers=SP,ScoreboardGames=SG",
        join_on="SP.GameId=SG.GameId", fields=FIELDS,
        where="SG.Patch='%s'" % patch,
        order_by="SG.DateTime_UTC, SP.Link",
        limit=PAGE, offset=offset,
    )
    last = ""
    for attempt in range(1, MAX_RETRIES_PER_PAGE + 1):
        _REQS[0] += 1
        try:
            rows = ec.cargo_client.query(**params)   # returns list of 'title' dicts
            time.sleep(REQUEST_SPACING)
            return rows
        except mwerr.APIError as e:
            code = getattr(e, "code", "") or ""
            last = "%s: %s" % (code, str(getattr(e, "info", ""))[:120])
            transient = code in ("ratelimited", "maxlag", "readonly") or "429" in str(e)
            if not transient:
                raise CargoError("cargo API error code=%s (%s)" % (code, last))
        except Exception as e:  # network / transport
            last = "%s: %s" % (type(e).__name__, str(e)[:120])
        wait = BACKOFFS[min(attempt - 1, len(BACKOFFS) - 1)]
        _log("  [%s] offset=%d attempt=%d/%d transient (%s) -> wait %ds (req_total=%d)"
             % (patch, offset, attempt, MAX_RETRIES_PER_PAGE, last, wait, _REQS[0]))
        time.sleep(wait)
    raise CargoError("patch=%s offset=%d failed after %d attempts (%s)"
                     % (patch, offset, MAX_RETRIES_PER_PAGE, last))


def fetch_patch(ec, patch: str) -> list[dict]:
    """All pro player-rows for one Patch string, offset-paged + cached."""
    safe = patch.replace(".", "_").replace(" ", "")
    pdir = CACHE / ("patch=%s" % safe)
    pdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    offset = 0
    while True:
        cfile = pdir / ("offset=%d.json" % offset)
        if cfile.exists():
            page = json.loads(cfile.read_text(encoding="utf-8"))
            _log("  [%s] offset=%d rows=%d CACHED (req_total=%d)"
                 % (patch, offset, len(page), _REQS[0]))
        else:
            page = _query(ec, patch, offset)
            cfile.write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")
            _log("  [%s] offset=%d rows=%d FETCHED (req_total=%d)"
                 % (patch, offset, len(page), _REQS[0]))
        rows.extend(page)
        if len(page) < PAGE:
            break
        offset += PAGE
    return rows


def _norm(name: str) -> str:
    name = (name or "").replace("’", "'").lower()
    return re.sub(r"[^a-z0-9]", "", name)


def _latest(season: str, kind: str):
    files = sorted((paths.TASK1_RAW / season).glob("%s_*.json" % kind),
                   key=lambda f: tuple(int(x) for x in f.stem[len(kind) + 1:].split(".")))
    return files[-1] if files else None


def build_id_maps() -> tuple[dict, dict]:
    item_map, rune_map = {}, {}
    for season in paths.SEASONS:
        f = _latest(season, "item")
        if f:
            for iid, v in json.loads(f.read_text(encoding="utf-8"))["data"].items():
                item_map.setdefault(_norm(v.get("name", "")), iid)
        rf = _latest(season, "runesReforged")
        if rf:
            for tree in json.loads(rf.read_text(encoding="utf-8")):
                rune_map.setdefault(_norm(tree["name"]), tree["id"])
                for slot in tree["slots"]:
                    for r in slot["runes"]:
                        rune_map.setdefault(_norm(r["name"]), r["id"])
    return item_map, rune_map


def _map_one(name: str, id_map: dict, unmatched: set) -> str:
    i = id_map.get(_norm(name))
    if i is None:
        unmatched.add(name)
        return "?"
    return str(i)


def split_items(raw: str) -> list[str]:
    return [t.strip() for t in (raw or "").split(";") if t.strip()]


def split_runes(raw: str) -> list[str]:
    return [t.strip() for t in re.split(r"[;,]", raw or "") if t.strip()]


def _version_map() -> dict:
    f = paths.ROOT / "data" / "mappings" / "version_map.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _lp_2024() -> list[str]:
    vers = [f.stem[len("champion") + 1:]
            for f in (paths.TASK1_RAW / "2024").glob("champion_14.*.json")]
    out = []
    for v in sorted(vers, key=lambda s: tuple(int(x) for x in s.split("."))):
        maj, minr = v.split(".")[:2]
        out.append("%s.%d" % (maj, int(minr)))          # "14.1" .. "14.24"
    return out


def project_patches() -> list[str]:
    """Leaguepedia patch strings for both seasons (2024 then 2025)."""
    return _lp_2024() + list(_version_map().keys())


def patch_client_map() -> dict:
    """Leaguepedia patch string -> project client patch (for later joining)."""
    m = dict(_version_map())
    for lp in _lp_2024():
        maj, minr = lp.split(".")
        m[lp] = "%s.%02d" % (maj, int(minr))
    return m


def aggregate_usage(rows, item_map, rune_map, client_map):
    games = defaultdict(int)
    item_ct = defaultdict(Counter)
    rune_ct = defaultdict(Counter)
    unmatched_items, unmatched_runes = set(), set()
    for r in rows:
        key = (r.get("Patch"), r.get("Champion"))
        if key[0] is None or key[1] is None:
            continue
        games[key] += 1
        for nm in set(split_items(r.get("Items", ""))):
            item_ct[key][nm] += 1
        for nm in set(split_runes(r.get("Runes", ""))):
            rune_ct[key][nm] += 1
    out = []
    for key in sorted(games):
        patch, champ = key
        g = games[key]
        pc = client_map.get(patch, "")
        for nm, c in item_ct[key].most_common():
            out.append({"patch": patch, "patch_client": pc, "champion": champ,
                        "kind": "item", "element_name": nm,
                        "element_id": _map_one(nm, item_map, unmatched_items),
                        "count": c, "games": g, "share": round(c / g, 4)})
        for nm, c in rune_ct[key].most_common():
            out.append({"patch": patch, "patch_client": pc, "champion": champ,
                        "kind": "rune", "element_name": nm,
                        "element_id": _map_one(nm, rune_map, unmatched_runes),
                        "count": c, "games": g, "share": round(c / g, 4)})
    return out, unmatched_items, unmatched_runes


def raw_player_rows(rows, client_map):
    return [{"patch": r.get("Patch"), "patch_client": client_map.get(r.get("Patch"), ""),
             "date": r.get("Date"), "player": r.get("Player"),
             "champion": r.get("Champion"), "role": r.get("Role"),
             "items": r.get("Items"), "runes": r.get("Runes")} for r in rows]


def write_csv(rows, name):
    TASK3.mkdir(parents=True, exist_ok=True)
    out = TASK3 / name
    if not rows:
        out.write_text("", encoding="utf-8")
        _log("  wrote empty %s" % name)
        return
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    _log("  wrote %s (%d rows)" % (name, len(rows)))


def write_unmatched(items, runes):
    (TASK3 / "unmatched_names.txt").write_text(
        "UNMATCHED ITEM NAMES (%d):\n%s\n\nUNMATCHED RUNE NAMES (%d):\n%s\n"
        % (len(items), "\n".join(sorted(items)), len(runes), "\n".join(sorted(runes))),
        encoding="utf-8")
    _log("  wrote unmatched_names.txt (items=%d runes=%d)" % (len(items), len(runes)))


def run(ec, patches, tag):
    item_map, rune_map = build_id_maps()
    _log("id maps: items=%d runes=%d | patches to pull: %d"
         % (len(item_map), len(rune_map), len(patches)))
    all_rows = []
    for i, patch in enumerate(patches, 1):
        _log("[patch %d/%d] %s" % (i, len(patches), patch))
        all_rows.extend(fetch_patch(ec, patch))
    _log("[%s] total player-rows=%d patches=%d champions=%d (network requests=%d)"
         % (tag, len(all_rows), len({r.get("Patch") for r in all_rows}),
            len({r.get("Champion") for r in all_rows}), _REQS[0]))
    client_map = patch_client_map()
    usage, ui, ur = aggregate_usage(all_rows, item_map, rune_map, client_map)
    prefix = "pilot_" if tag == "pilot" else ""
    write_csv(usage, prefix + "leaguepedia_usage_by_champion_patch.csv")
    write_csv(raw_player_rows(all_rows, client_map), prefix + "leaguepedia_player_rows.csv")
    write_unmatched(ui, ur)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="one authenticated test query")
    g.add_argument("--pilot", action="store_true", help="one patch, end to end")
    g.add_argument("--full", action="store_true", help="all project patches (long)")
    ap.add_argument("--patch", default="25.14",
                    help="Leaguepedia patch for --pilot (default 25.14 = client 15.14)")
    args = ap.parse_args()

    try:
        ec = make_client()
        if args.smoke:
            _log("=== SMOKE TEST (authenticated cargoquery, Patch=14.1, limit=5) ===")
            rows = ec.cargo_client.query(
                tables="ScoreboardPlayers=SP,ScoreboardGames=SG",
                join_on="SP.GameId=SG.GameId", fields=FIELDS,
                where="SG.Patch='14.1'", order_by="SG.DateTime_UTC, SP.Link", limit=5)
            _log("rows returned: %d" % len(rows))
            for r in rows[:5]:
                _log("  " + json.dumps(r, ensure_ascii=False)[:200])
        elif args.pilot:
            _log("=== Task 3 PILOT (Leaguepedia, patch %s) ===" % args.patch)
            run(ec, [args.patch], "pilot")
        else:
            ps = project_patches()
            _log("=== Task 3 FULL (Leaguepedia): %d patches ===" % len(ps))
            run(ec, ps, "full")
    except CargoError as e:
        _log("")
        _log("ABORTED: %s" % str(e)[:280])
        _log("Run is RESUMABLE - cached pages are kept; re-run the same command.")
        sys.exit(2)


if __name__ == "__main__":
    main()
