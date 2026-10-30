"""Frozen BERT [CLS] embeddings of patch-note text, plain and enriched.

Usage:
    python -m src.features.text_features
    python -m src.features.text_features --variant plain
    python -m src.features.text_features --variant enriched --force
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

try:
    from src import config
    from src.data import align
    from src.features import item_features as itf
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.data import align
    from src.features import item_features as itf

NO_CHANGE_TEXT = "No changes."
MAX_TOKENS = 256
BATCH_SIZE = 32
VARIANTS = ("plain", "enriched")


def _paths(variant: str) -> tuple:
    """(texts_json, embeddings_npy, index_csv) for a text variant."""
    if variant == "enriched":
        return (config.PATCH_TEXTS_ENRICHED_JSON,
                config.CLS_EMBEDDINGS_ENRICHED_NPY,
                config.CLS_INDEX_ENRICHED_CSV)
    return (config.PATCH_TEXTS_JSON, config.CLS_EMBEDDINGS_NPY,
            config.CLS_INDEX_CSV)


def load_embedding_cache(emb_path, idx_path):
    """Reject incomplete or ambiguous caches before any model consumes them."""
    idx = pd.read_csv(idx_path, dtype={"patch": str})
    emb = np.load(emb_path, allow_pickle=False)
    if (idx[["champion", "patch"]].isna().any().any()
            or idx.duplicated(["champion", "patch"]).any()
            or emb.shape != (len(idx), config.BERT_HIDDEN)
            or not np.isfinite(emb).all()):
        raise ValueError(f"invalid embedding/index cache: {emb_path}, {idx_path}")
    return emb, {(c, p): i for i, (c, p) in enumerate(zip(idx.champion, idx.patch))}


def _render_change(c: dict) -> str:
    """Render one change as a short clause."""
    ability = (c.get("ability") or "").strip()
    stat = (c.get("stat") or "").strip()
    before = (c.get("before") or "").strip()
    after = (c.get("after") or "").strip()
    direction = (c.get("direction") or "").strip()
    head = " ".join(p for p in (ability, stat) if p) or "stat"
    clause = head
    if before or after:
        clause += f" from {before} to {after}"
    if direction:
        clause += f" ({direction})"
    return clause


def _render_champion(entry_champ: dict) -> str:
    """All changes for one champion in one patch -> a single blurb."""
    name = entry_champ.get("champion", "").strip()
    direction = (entry_champ.get("overall_direction") or "").strip()
    clauses = [_render_change(c) for c in entry_champ.get("changes", [])]
    head = f"{name} {direction}".strip() + "."
    if not clauses:
        return head
    return head + " " + " ".join(s + "." for s in clauses)


def _section_text_by_patch() -> dict[str, dict[str, str]]:
    """{patch: {element_key: raw_text}} from the scraped Items/Runes sections."""
    out: dict[str, dict[str, str]] = {}
    for season in config.SEASONS:
        path = config.PATCH_SECTIONS_JSON[season]
        if not path.exists():
            continue
        for page in json.loads(path.read_text(encoding="utf-8")):
            patch = page.get("patch")
            bucket = out.setdefault(patch, {})
            for sec in page.get("sections", []):
                if sec["section"] not in ("items", "runes"):
                    continue
                for entry in sec["entries"]:
                    key = itf.element_key(entry["element_name"])
                    txt = entry["raw_text"].strip()
                    if txt:
                        bucket[key] = (bucket.get(key, "") + " " + txt).strip()
    return out


def _lagged_builds() -> dict[tuple[str, str], set[str]]:
    """{(patch, champion): {element_key built at P-1}}."""
    use = pd.read_csv(config.LEAGUEPEDIA_USAGE_SRC, dtype={"patch": str, "patch_client": str})
    nxt = {v: k for k, v in itf._prev_patch_map().items()}
    out: dict[tuple[str, str], set[str]] = {}
    for r in use.itertuples(index=False):
        target = nxt.get(r.patch_client)
        if target is None:
            continue
        champ = align.normalise_champion_name(r.champion)
        out.setdefault((target, champ), set()).add(
            itf.element_key(r.element_name))
    return out


def _routed_section_text(patch: str, champion: str,
                         sections: dict, builds: dict) -> str:
    """Item/rune section text relevant to one champion on one patch."""
    built = builds.get((patch, champion), set())
    if not built:
        return ""
    parts = [txt for key, txt in sections.get(patch, {}).items()
             if key in built]
    return " ".join(parts)


def build_patch_texts(universe: pd.DataFrame,
                      variant: str = "plain") -> dict[str, dict[str, str]]:
    """Build {client_patch: {champion: blurb}} for every scored champion-patch."""
    vmap = json.loads(config.VERSION_MAP.read_text(encoding="utf-8"))

    per_patch: dict[str, dict[str, str]] = {}
    for season in config.SEASONS:
        path = config.PATCH_NOTES_DIR / f"lol_patches_{season}.json"
        for entry in json.loads(path.read_text(encoding="utf-8")):
            patch = vmap.get(entry["patch_version"])
            if patch is None:
                continue
            bucket = per_patch.setdefault(patch, {})
            for champ in entry.get("champions", []):
                name = align.normalise_champion_name(champ.get("champion", ""))
                if name:
                    bucket[name] = _render_champion(champ)

    sections = _section_text_by_patch() if variant == "enriched" else {}
    builds = _lagged_builds() if variant == "enriched" else {}

    texts: dict[str, dict[str, str]] = {}
    for row in universe[["patch", "champion"]].itertuples(index=False):
        patch, champ = row.patch, align.normalise_champion_name(row.champion)
        blurb = per_patch.get(patch, {}).get(champ, NO_CHANGE_TEXT)
        if variant == "enriched":
            extra = _routed_section_text(patch, champ, sections, builds)
            if extra:
                blurb = (("" if blurb == NO_CHANGE_TEXT else blurb + " ")
                         + extra)
        texts.setdefault(patch, {})[champ] = blurb
    return texts


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Frozen bert-base-uncased -> (len(texts), 768) [CLS] vectors."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(config.RANDOM_SEED)
    tok = AutoTokenizer.from_pretrained(config.BERT_MODEL)
    model = AutoModel.from_pretrained(config.BERT_MODEL)
    model.eval()

    out = np.empty((len(texts), config.BERT_HIDDEN), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=MAX_TOKENS, return_tensors="pt")
            cls = model(**enc).last_hidden_state[:, 0, :]  # [CLS] token
            out[start:start + len(batch)] = cls.numpy()
            print(f"  embedded {min(start + BATCH_SIZE, len(texts)):>5}/"
                  f"{len(texts)}", end="\r")
    print()
    return out


def build_embeddings(variant: str = "plain", force: bool = False) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    config.ensure_dirs()
    texts_json, embeddings_npy, index_csv = _paths(variant)
    if embeddings_npy.exists() and not force:
        load_embedding_cache(embeddings_npy, index_csv)
        if not texts_json.exists():
            raise ValueError(f"missing text cache: {texts_json}; rebuild with --force")
        print(f"[-] {embeddings_npy.name} exists; pass --force to recompute.")
        return

    universe = pd.read_csv(config.PWDS_CSV, dtype={"patch": str})[["patch", "champion"]]
    texts_map = build_patch_texts(universe, variant=variant)
    texts_json.write_text(
        json.dumps(texts_map, indent=1, ensure_ascii=False), encoding="utf-8")

    index_rows, texts = [], []
    for patch in sorted(texts_map, key=lambda s: tuple(int(p) for p in s.split("."))):
        for champ in sorted(texts_map[patch]):
            index_rows.append({"champion": champ, "patch": patch})
            texts.append(texts_map[patch][champ])

    n_changed = sum(1 for t in texts if t != NO_CHANGE_TEXT)
    print(f"[{variant}] Embedding {len(texts):,} champion-patch blurbs "
          f"({n_changed:,} with patch-note changes, "
          f"{len(texts) - n_changed:,} 'No changes') with {config.BERT_MODEL} ...")

    emb = _embed_texts(texts)
    np.save(embeddings_npy, emb)
    pd.DataFrame(index_rows).to_csv(index_csv, index=False)

    print(f"  embeddings: {emb.shape} -> {embeddings_npy}")
    print(f"  index:      {len(index_rows)} rows -> {index_csv}")
    print(f"  texts:      -> {texts_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("plain", "enriched", "both"),
                        default="both", help="which text variant(s) to build")
    parser.add_argument("--force", action="store_true",
                        help="recompute even if the cache already exists")
    args = parser.parse_args()
    variants = VARIANTS if args.variant == "both" else (args.variant,)
    for variant in variants:
        build_embeddings(variant=variant, force=args.force)


if __name__ == "__main__":
    main()
