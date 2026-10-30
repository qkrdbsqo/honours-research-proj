"""BERT models on the block target: frozen, scaling control and fine-tuned.

Usage:
    python -m src.exploratory.rolling_block_text --stage embed --block 3
    python -m src.exploratory.rolling_block_text --stage run --model bert --block 3
    python -m src.exploratory.rolling_block_text --stage run --model bert_ctrl --block 3
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from src import config
    from src.evaluation import metrics as ev
    from src.exploratory import rolling_block as rb
    from src.features import text_features as tf
    from src.models import _shared as sh
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.evaluation import metrics as ev
    from src.exploratory import rolling_block as rb
    from src.features import text_features as tf
    from src.models import _shared as sh

CONFIG = rb.CONFIG                     # C1: the block runs' Path A config
TEXT_CONFIGS = ("C1", "C3")
ALIASED = {"C0": "C1", "C2": "C1", "C4": "C1", "C6": "C1"}
MODELS = ("bert", "bert_ctrl", "bert_ft")
CACHE = config.DATA / "block_text"
PARTS = config.ROOT / "cache" / "rolling_block_text"   # gitignored, resumable
OUT_TABLES = rb.OUT_TABLES
SMOKE_EPOCHS = 1


def _frozen_modules():
    from src.models import bert_control as bc
    from src.models import bert_regressor as br
    return br, bc


def variant_of(cfg: str) -> str:
    """Text variant for a config: "plain" or "enriched"."""
    if cfg in ALIASED:
        raise ValueError(
            f"{cfg}'s text is identical to {ALIASED[cfg]}'s, so a {cfg} run would "
            f"reproduce it bit-for-bit. Run {ALIASED[cfg]} and report {cfg} by "
            f"reference, as the frozen split does (run_ablation.BERT_ALIASES).")
    if cfg not in TEXT_CONFIGS:
        raise ValueError(f"unknown config {cfg!r}, expected one of {TEXT_CONFIGS}")
    return config.ABLATION_CONFIGS[cfg]["text"]


def texts_path(b: int, cfg: str = CONFIG):
    return CACHE / f"block_texts_B{b}_{variant_of(cfg)}.json"


def embeddings_path(b: int, cfg: str = CONFIG):
    return CACHE / f"cls_embeddings_B{b}_{variant_of(cfg)}.npy"


def index_path(b: int, cfg: str = CONFIG):
    return CACHE / f"cls_index_B{b}_{variant_of(cfg)}.csv"


def pred_dir(b: int, cfg: str = CONFIG):
    """``ROLLB-[D-]<cfg>-B<b>``."""
    return config.OUTPUTS / "predictions" / f"ROLLB-{sh.target_prefix()}{cfg}-B{b}"


def part_dir(b: int, model: str, cfg: str = CONFIG):
    """Per-origin results, so a stopped sweep resumes instead of restarting."""
    prefix = "D-" if sh.active_target() == "delta" else ""
    return PARTS / f"{prefix}{cfg}-B{b}" / model


def part_path(b: int, model: str, origin: str, cfg: str = CONFIG):
    return part_dir(b, model, cfg) / f"{origin}.csv"


def hyperparams_path(b: int, model: str, cfg: str = CONFIG):
    suffix = "_delta" if sh.active_target() == "delta" else ""
    return OUT_TABLES / f"hyperparameters_B{b}_{cfg}_{model}{suffix}.csv"


def block_text(per_patch: dict, champion: str, patches: tuple[str, ...]) -> str:
    """The champion's blurbs for a block's patches, joined in patch order."""
    parts = [per_patch.get((champion, p), tf.NO_CHANGE_TEXT) for p in patches]
    said = [t for t in parts if t != tf.NO_CHANGE_TEXT]
    return " ".join(said) if said else tf.NO_CHANGE_TEXT


def build_block_texts(b: int, cfg: str = CONFIG) -> dict[str, dict[str, str]]:
    """{block start: {champion: text}} over exactly the block dataset's rows."""
    patches_of = dict(rb.blocks(b))
    df = rb.build_dataset(b)
    universe = pd.DataFrame(
        [(champ, patch) for start, champ in zip(df["patch"], df["champion"])
         for patch in patches_of[start]], columns=["champion", "patch"]
    ).drop_duplicates()
    raw = tf.build_patch_texts(universe, variant=variant_of(cfg))
    per_patch = {(champ, patch): text for patch, champs in raw.items()
                 for champ, text in champs.items()}
    out: dict[str, dict[str, str]] = {}
    for start, champ in zip(df["patch"], df["champion"]):
        out.setdefault(start, {})[champ] = block_text(per_patch, champ,
                                                      patches_of[start])
    return out


def build_embeddings(b: int, force: bool = False, cfg: str = CONFIG) -> None:
    """Write the block texts, then their frozen [CLS] matrix and row index."""
    CACHE.mkdir(parents=True, exist_ok=True)
    if embeddings_path(b, cfg).exists() and not force:
        load_embeddings(b, cfg)
        print(f"[block_text] {cfg} B={b}: cache exists, nothing to do "
              f"(--force to rebuild)")
        return

    texts = build_block_texts(b, cfg)
    texts_path(b, cfg).write_text(json.dumps(texts, indent=1, ensure_ascii=False),
                             encoding="utf-8")
    rows = [(champ, start) for start, champs in texts.items() for champ in champs]
    blurbs = [texts[start][champ] for champ, start in rows]
    n_empty = sum(t == tf.NO_CHANGE_TEXT for t in blurbs)
    print(f"[block_text] {cfg} ({variant_of(cfg)}) B={b}: {len(rows)} rows, "
          f"{n_empty} ({n_empty / len(rows):.1%}) are \"No changes.\"")

    emb = tf._embed_texts(blurbs)          # frozen bert-base-uncased [CLS]
    np.save(embeddings_path(b, cfg), emb)
    idx = pd.DataFrame(rows, columns=["champion", "patch"])
    idx.to_csv(index_path(b, cfg), index=False)
    embeddings_path(b, cfg).with_suffix(".json").write_text(
        json.dumps(_cache_hashes(b, cfg), sort_keys=True), encoding="utf-8")
    print(f"[block_text] wrote {embeddings_path(b, cfg).name} {emb.shape}")


def load_embeddings(b: int, cfg: str = CONFIG):
    """(matrix, (champion, block start) -> row) for the block cache."""
    if not embeddings_path(b, cfg).exists():
        raise RuntimeError(
            f"no block embedding cache for {cfg} B={b}. Build it first: "
            f"python -m src.exploratory.rolling_block_text --stage embed "
            f"--block {b} --config {cfg}")
    receipt = embeddings_path(b, cfg).with_suffix(".json")
    if not receipt.exists() or json.loads(receipt.read_text()) != _cache_hashes(b, cfg):
        raise ValueError("Unverified or changed block cache; rebuild in an isolated run directory.")
    texts = json.loads(texts_path(b, cfg).read_text(encoding="utf-8"))
    if texts != build_block_texts(b, cfg):
        raise ValueError("Stale block texts; rebuild embeddings before training.")
    return tf.load_embedding_cache(embeddings_path(b, cfg), index_path(b, cfg))


def _cache_hashes(b, cfg):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (texts_path(b, cfg), embeddings_path(b, cfg), index_path(b, cfg))}


def _run_signature(df, model, b, cfg, epoch_cap, emb, key_to_row, texts):
    """Bind resumed origins to their inputs, settings, code and environment."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    settings = dict(model=model, block=b, config=cfg, target=sh.active_target(),
                    epoch_cap=epoch_cap, seed=config.RANDOM_SEED,
                    columns=list(df.columns),
                    packages={p: importlib.metadata.version(p) for p in
                              ("numpy", "pandas", "scikit-learn", "torch", "transformers")})
    h.update(json.dumps(settings, sort_keys=True).encode())
    if emb is not None:
        h.update(emb.tobytes())
        h.update(repr(sorted(key_to_row.items())).encode())
    if texts is not None:
        h.update(repr(sorted(texts.items())).encode())
    for module in ("config.py", "exploratory/rolling_block.py", "exploratory/rolling_block_text.py",
                   "models/_shared.py", "models/bert_regressor.py", "models/bert_control.py",
                   "models/bert_finetune.py"):
        h.update((config.ROOT / "src" / module).read_bytes())
    return h.hexdigest()


def _scaler_for(model: str):
    """`bert` standardises all 770 dims; `bert_ctrl` only the 2 numeric ones."""
    _, bc = _frozen_modules()
    return StandardScaler() if model == "bert" else bc.NumericOnlyScaler()


def _frozen_origin(model: str, df: pd.DataFrame, origin: str, emb, key_to_row,
                   smoke: bool) -> tuple[pd.DataFrame, dict]:
    br, _ = _frozen_modules()
    train, test = rb.training_rows(df, origin), df[df["patch"] == origin].copy()
    delta = sh.active_target() == "delta"
    if delta:                      # the delta is undefined without a real lag
        train = train[train["has_lag"] == 1].copy()
        test = test[test["has_lag"] == 1].copy()

    grid = br.LEARNING_RATES[:1] if smoke else br.LEARNING_RATES
    cap = 50 if smoke else br.MAX_EPOCHS
    best = (None, np.inf, cap)
    for lr in grid:
        maes, epochs = [], []
        for ft, fv in rb._inner_folds(train):
            scaler = _scaler_for(model).fit(br.make_X(ft, emb, key_to_row))
            X_tr = scaler.transform(br.make_X(ft, emb, key_to_row)).astype(np.float32)
            X_val = scaler.transform(br.make_X(fv, emb, key_to_row)).astype(np.float32)
            _, vmae, vep = br._train_head(X_tr, ft[sh.TARGET].to_numpy(), lr, cap,
                                          X_val, fv[sh.TARGET].to_numpy())
            maes.append(vmae)
            epochs.append(vep)
        avg = float(np.mean(maes)) if maes else np.inf
        if avg < best[1]:
            best = (lr, avg, int(round(float(np.mean(epochs)))))
    lr, _, epochs = best

    med = float(train[sh.LAG].median())
    train[sh.LAG] = train[sh.LAG].fillna(med)
    test_imp = test.copy()
    test_imp[sh.LAG] = test_imp[sh.LAG].fillna(med)
    scaler = _scaler_for(model).fit(br.make_X(train, emb, key_to_row))
    X_tr = scaler.transform(br.make_X(train, emb, key_to_row)).astype(np.float32)
    X_te = scaler.transform(br.make_X(test_imp, emb, key_to_row)).astype(np.float32)
    head, _, _ = br._train_head(X_tr, train[sh.TARGET].to_numpy(), lr, epochs)
    back = test_imp[sh.LAG].to_numpy() if delta else 0.0

    preds = test[["champion", "patch", "block_end", "has_lag"]].copy()
    preds["y_true"] = test[rb.TARGET].to_numpy()
    preds["y_pred"] = br._predict(head, X_te) + back
    return preds, {"origin": origin, "n_train": len(train), "n_test": len(test),
                   "learning_rate": lr, "epochs": epochs, "device": "cpu"}


def _ft_origin(df: pd.DataFrame, origin: str, texts, smoke: bool,
               epoch_cap: int | None = None) -> tuple[pd.DataFrame, dict]:
    from transformers import AutoTokenizer          # torch only on this path

    from src.models import bert_finetune as bft

    train, test = rb.training_rows(df, origin), df[df["patch"] == origin].copy()
    delta = sh.active_target() == "delta"
    if delta:
        train = train[train["has_lag"] == 1].copy()
        test = test[test["has_lag"] == 1].copy()

    tok = AutoTokenizer.from_pretrained(config.BERT_MODEL)
    epochs = epoch_cap or (SMOKE_EPOCHS if smoke else config.BERT_FT_EPOCHS)
    grid = config.BERT_FT_LR_GRID[:1] if smoke else config.BERT_FT_LR_GRID

    med = float(train[sh.LAG].median())
    best = (None, np.inf, epochs)
    for lr in grid:
        maes, best_epochs = [], []
        for ft, fv in rb._inner_folds(train):
            _, vmae, vep, _, _ = bft._train(ft, fv, texts, tok, lr, epochs)
            maes.append(vmae)
            best_epochs.append(vep)
        avg = float(np.mean(maes)) if maes else np.inf
        if avg < best[1]:
            best = (lr, avg, int(round(float(np.mean(best_epochs)))))
    lr, _, fit_epochs = best

    train[sh.LAG] = train[sh.LAG].fillna(med)
    test_imp = test.copy()
    test_imp[sh.LAG] = test_imp[sh.LAG].fillna(med)
    model, _, _, _, scaler = bft._train(train, None, texts, tok, lr, fit_epochs)
    back = test_imp[sh.LAG].to_numpy() if delta else 0.0

    preds = test[["champion", "patch", "block_end", "has_lag"]].copy()
    preds["y_true"] = test[rb.TARGET].to_numpy()
    preds["y_pred"] = bft._predict(model, test_imp, texts, tok, scaler) + back
    return preds, {"origin": origin, "n_train": len(train), "n_test": len(test),
                   "learning_rate": lr, "epochs": fit_epochs,
                   "device": str(bft.resolve_device()),
                   "epoch_cap": epochs}


def run(model: str, b: int, target: str = "level", smoke: bool = False,
        max_origins: int | None = None, epoch_cap: int | None = None,
        redo: bool = False, cfg: str = CONFIG) -> pd.DataFrame:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}, expected one of {MODELS}")
    if max_origins is not None and max_origins < 1:
        raise ValueError("max_origins must be positive")
    if epoch_cap is not None and epoch_cap < 1:
        raise ValueError("epoch_cap must be positive")
    variant_of(cfg)                      # refuses an alias before any work
    sh.set_target(target)
    sh.set_config(CONFIG)                # Path A plumbing stays on the block config

    df = rb.build_dataset(b)
    origins = rb.origins_for(df)
    if smoke:
        origins = origins[:1]
    elif max_origins:
        origins = origins[:max_origins]

    if model == "bert_ft":
        texts = {(c, p): t
                 for p, champs in json.loads(
                     texts_path(b, cfg).read_text(encoding="utf-8")).items()
                 for c, t in champs.items()} if texts_path(b, cfg).exists() else None
        if texts is None:
            raise RuntimeError(
                f"no block texts for {cfg} B={b}. Build them first: --stage "
                f"embed --block {b} --config {cfg} (the texts are written "
                f"before the embeddings, so this also covers the fine-tuned "
                f"path).")
        expected = {(c, p): t for p, cs in build_block_texts(b, cfg).items() for c, t in cs.items()}
        if texts != expected:
            raise ValueError("Stale block texts; rebuild in an isolated run directory.")
        emb = key_to_row = None
    else:
        emb, key_to_row = load_embeddings(b, cfg)
        texts = None

    if not smoke:
        if max_origins is None:
            pass
        part_dir(b, model, cfg).mkdir(parents=True, exist_ok=True)
    signature = _run_signature(df, model, b, cfg, epoch_cap, emb, key_to_row, texts)
    done = [o for o in origins if not smoke and part_path(b, model, o, cfg).exists()
            and part_path(b, model, o, cfg).with_suffix(".json").exists()
            and not redo]
    if done:
        print(f"[block_text] resuming: {len(done)} of {len(origins)} origins "
              f"already on disk ({done[0]} .. {done[-1]})")

    print(f"[block_text] {model} B={b} target={target}: {len(origins)} origins"
          f"{' (SMOKE, nothing written)' if smoke else ''}", flush=True)
    all_preds, chosen, t0 = [], [], time.time()
    for i, origin in enumerate(origins, 1):
        if origin in done:
            path = part_path(b, model, origin, cfg)
            params = json.loads(path.with_suffix(".json").read_text())
            if (params.get("run_signature") != signature or
                    params.get("prediction_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
                raise ValueError(f"Incompatible or damaged checkpoint {path}; use an isolated rerun.")
            all_preds.append(pd.read_csv(path, dtype={"patch": str, "block_end": str}))
            chosen.append(params)
            continue
        if model == "bert_ft":
            preds, params = _ft_origin(df, origin, texts, smoke, epoch_cap)
        else:
            preds, params = _frozen_origin(model, df, origin, emb, key_to_row, smoke)
        if not smoke:      # written before the next origin starts, so a crash
            preds.to_csv(part_path(b, model, origin, cfg), index=False)   # costs one origin
            params["run_signature"] = signature
            params["prediction_sha256"] = hashlib.sha256(part_path(b, model, origin, cfg).read_bytes()).hexdigest()
            part_path(b, model, origin, cfg).with_suffix(".json").write_text(
                json.dumps(params), encoding="utf-8")
        all_preds.append(preds)
        chosen.append(params)
        d = preds[preds["has_lag"] == 1]
        print(f"  [{i:>2}/{len(origins)}] {origin}: MAE {ev.mae(d['y_true'], d['y_pred']):.5f}"
              f"  lr={params['learning_rate']:g}  {time.time() - t0:.0f}s", flush=True)

    pooled = pd.concat(all_preds, ignore_index=True)
    d = pooled[pooled["has_lag"] == 1]
    print(f"[block_text] {model} B={b}: pooled has-lag MAE "
          f"{ev.mae(d['y_true'], d['y_pred']):.6f} over {len(d)} rows, "
          f"rho {ev.spearman_per_patch(d)[0]:.4f}")
    if smoke:
        print("[block_text] SMOKE: wiring only, nothing written, not a result.")
        return pooled
    if max_origins:
        print(f"[block_text] PARTIAL sweep ({len(origins)} origins): nothing "
              f"written. Rerun without --max-origins to finish; what just ran "
              f"is cached in {part_dir(b, model, cfg)}.")
        return pooled

    pred_dir(b, cfg).mkdir(parents=True, exist_ok=True)
    cols = ["champion", "patch", "block_end", "y_true", "y_pred", "has_lag"]
    pooled[cols].to_csv(pred_dir(b, cfg) / f"{model}_preds.csv", index=False)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(chosen).to_csv(hyperparams_path(b, model, cfg), index=False)
    print(f"[block_text] wrote {pred_dir(b, cfg) / f'{model}_preds.csv'}")
    return pooled


def main() -> None:
    ap = argparse.ArgumentParser(description="Block-target Path B runs.")
    ap.add_argument("--stage", choices=("embed", "run"), required=True)
    ap.add_argument("--block", type=int, default=3, choices=rb.BLOCK_SIZES)
    ap.add_argument("--model", choices=MODELS, default="bert")
    ap.add_argument("--config", default=CONFIG, choices=TEXT_CONFIGS,
                    help="text variant: C1 plain (C2/C4/C6 alias it), C3 enriched")
    ap.add_argument("--target", choices=("level", "delta"), default="level")
    ap.add_argument("--force", action="store_true", help="rebuild the cache")
    ap.add_argument("--smoke", action="store_true",
                    help="one origin, minimal budget, writes nothing")
    ap.add_argument("--max-origins", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="bert_ft epoch cap override")
    ap.add_argument("--redo", action="store_true",
                    help="recompute origins already cached in cache/rolling_block_text/")
    a = ap.parse_args()
    if a.stage == "embed":
        build_embeddings(a.block, a.force, a.config)
    else:
        run(a.model, a.block, a.target, a.smoke, a.max_origins, a.epochs, a.redo,
            a.config)


if __name__ == "__main__":
    main()
