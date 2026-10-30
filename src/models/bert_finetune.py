"""Fine-tuned BERT regressor (bert_ft).

Usage:
    python -m src.models.bert_finetune --config C1
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

try:
    from src import config
    from src.models import _shared as sh
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.models import _shared as sh

MODEL_NAME = "bert_ft"
FALLBACK_MODEL_NAME = "bert_ft_fallback"


def diagnostics_path():
    """Where the per-fold fit diagnostics go, namespaced by config AND target."""
    return config.bert_ft_diagnostics_csv(
        f"{sh.target_prefix()}{sh.active_config()}")


def _text_path(config_id: str):
    """The same text variant the frozen path (bert_regressor) uses."""
    if config.ABLATION_CONFIGS[config_id]["text"] == "enriched":
        return config.PATCH_TEXTS_ENRICHED_JSON
    return config.PATCH_TEXTS_JSON


def load_texts(config_id: str) -> dict[tuple[str, str], str]:
    """(champion, patch) -> patch-note blurb."""
    raw = json.loads(_text_path(config_id).read_text(encoding="utf-8"))
    return {(champ, patch): text
            for patch, champs in raw.items()
            for champ, text in champs.items()}


def resolve_device() -> torch.device:
    """The device to train on: CUDA when available, else CPU."""
    forced = os.environ.get("BERT_FT_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_device(model) -> torch.device:
    """Where ``model``'s parameters live, defaulting to CPU."""
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed torch on all devices and pin cuDNN to deterministic algorithms."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BertRegressor(nn.Module):
    """Encoder + 2 numeric features -> 1 output."""

    def __init__(self, frozen_layers: int = 0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.BERT_MODEL)
        self.head = nn.Linear(config.BERT_HIDDEN + 2, 1)
        if frozen_layers > 0:
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
            for layer in self.encoder.encoder.layer[:frozen_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask, numeric):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]          # [CLS]
        return self.head(torch.cat([cls, numeric], dim=1))


def _numeric_frame(df: pd.DataFrame) -> np.ndarray:
    """The two numeric features, in the frozen path's column order."""
    return np.column_stack([df[sh.LAG].to_numpy(dtype=float),
                            df["has_lag"].to_numpy(dtype=float)])


def _numeric_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """Scaler for [lag_pwds, has_lag], fit on training rows only."""
    return StandardScaler().fit(_numeric_frame(train_df))


def _make_batches(df: pd.DataFrame, texts, tokenizer, shuffle: bool, seed: int,
                  scaler: StandardScaler):
    """Length-sorted dynamic-padding batches."""
    missing = [(c, p) for c, p in zip(df["champion"], df["patch"])
               if (c, p) not in texts]
    if missing:
        raise KeyError(
            f"{len(missing)} rows have no patch-note text, e.g. {missing[:3]}. "
            "Rebuild with 'python -m src.features.text_features'. Substituting "
            "'No changes.' here would fabricate model input.")

    numeric_all = scaler.transform(_numeric_frame(df))
    items = []
    for i, (champ, patch) in enumerate(zip(df["champion"], df["patch"])):
        enc = tokenizer(texts[(champ, patch)],
                        truncation=True, max_length=config.BERT_FT_MAX_TOKENS)
        items.append((enc["input_ids"], numeric_all[i].tolist()))

    order = np.argsort([len(i[0]) for i in items], kind="stable")
    batches = [order[i:i + config.BERT_FT_BATCH]
               for i in range(0, len(order), config.BERT_FT_BATCH)]
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(batches)                            # batch ORDER only

    y_all = df[sh.TARGET].to_numpy()
    out = []
    for b in batches:
        ids = [items[i][0] for i in b]
        width = max(len(x) for x in ids)
        input_ids = torch.tensor([x + [tokenizer.pad_token_id] * (width - len(x))
                                  for x in ids])
        attn = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids])
        numeric = torch.tensor([items[i][1] for i in b], dtype=torch.float32)
        y = torch.tensor(y_all[b], dtype=torch.float32).reshape(-1, 1)
        out.append((input_ids, attn, numeric, y, b))
    return out


def _evaluate(model, batches, device=None) -> float:
    model.eval()
    dev = device if device is not None else model_device(model)
    errs = []
    with torch.no_grad():
        for ids, attn, num, y, _ in batches:
            p = model(ids.to(dev), attn.to(dev),
                      num.to(dev)).cpu().numpy().ravel()
            errs.extend(np.abs(y.numpy().ravel() - p))
    return float(np.mean(errs))


def _optimizer(model, lr: float):
    """AdamW with discriminative LRs: encoder at ``lr``, head at the larger ``config.BERT_FT_HEAD_LR``."""
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    head = [p for p in model.head.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [{"params": enc, "lr": lr},
         {"params": head, "lr": config.BERT_FT_HEAD_LR}],
        weight_decay=config.BERT_FT_WEIGHT_DECAY)


def _train(train_df, val_df, texts, tokenizer, lr, epochs, frozen_layers=0):
    """Returns (model, best_val_mae, best_epoch, train_mae_at_best, scaler)."""
    device = resolve_device()
    seed_everything(config.RANDOM_SEED)
    model = BertRegressor(frozen_layers).to(device)
    opt = _optimizer(model, lr)
    scaler = _numeric_scaler(train_df)
    tb = _make_batches(train_df, texts, tokenizer, True, config.RANDOM_SEED,
                       scaler)
    total = max(1, len(tb) * epochs)
    sched = get_linear_schedule_with_warmup(
        opt, int(config.BERT_FT_WARMUP_FRAC * total), total)
    loss_fn = nn.MSELoss()

    vb = (_make_batches(val_df, texts, tokenizer, False, config.RANDOM_SEED,
                        scaler)
          if val_df is not None else None)
    best = (np.inf, epochs, float("nan"))
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for ids, attn, num, y, _ in tb:
            opt.zero_grad()
            loss_fn(model(ids.to(device), attn.to(device), num.to(device)),
                    y.to(device)).backward()
            opt.step()
            sched.step()
        if vb is None:
            continue
        v = _evaluate(model, vb, device)
        t = _evaluate(model, tb, device)
        print(f"    epoch {epoch}: train MAE={t:.5f}  val MAE={v:.5f}", flush=True)
        if v < best[0]:
            best = (v, epoch, t)
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best[0], best[1], best[2], scaler


def _predict(model, df, texts, tokenizer, scaler) -> np.ndarray:
    """Predictions in ``df`` row order (batches carry positional indices)."""
    batches = _make_batches(df, texts, tokenizer, False, config.RANDOM_SEED,
                            scaler)
    out = np.zeros(len(df))
    model.eval()
    dev = model_device(model)
    with torch.no_grad():
        for ids, attn, num, _, idx in batches:
            out[idx] = model(ids.to(dev), attn.to(dev),
                             num.to(dev)).cpu().numpy().ravel()
    return out


def _write_diagnostics(rows: list[dict]) -> None:
    out = diagnostics_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  -> {out}")


def main(config_id: str = "C1", fallback: bool = False,
         epoch_cap: int | None = None) -> None:
    """Fine-tune BERT under one ablation config."""
    sh.set_config(config_id)
    cap = int(epoch_cap or config.BERT_FT_EPOCHS)
    frozen_layers = config.BERT_FT_FALLBACK_FROZEN_LAYERS if fallback else 0
    name = FALLBACK_MODEL_NAME if fallback else MODEL_NAME

    texts = load_texts(config_id)
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL)

    df = sh.load_dataset()
    train, test = sh.train_test_split(df)
    device = resolve_device()
    print(f"[{name}] walk-forward CV "
          f"(target={sh.active_target()}, config={config_id}, "
          f"frozen_layers={frozen_layers}, device={device.type})")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
        print("  NOTE: CUDA results are NOT bit-identical to CPU results. Every "
              "bert_ft config being compared must be run on the SAME device.")
    folds = list(sh.walk_forward_folds(train))
    diagnostics: list[dict] = []
    best_lr, best_mae, best_epochs = None, np.inf, cap
    for lr in config.BERT_FT_LR_GRID:
        maes, epochs_at_best, gaps = [], [], []
        for k, (ft, fv) in enumerate(folds, 1):
            print(f"  lr={lr:g} fold {k}/{len(folds)} "
                  f"(train n={len(ft)}, val n={len(fv)})", flush=True)
            _, v, e, t, _ = _train(ft, fv, texts, tokenizer, lr,
                                   cap, frozen_layers)
            ratio = v / t if t > 0 else float("nan")
            maes.append(v)
            epochs_at_best.append(e)
            gaps.append(ratio)
            diagnostics.append({"target": sh.active_target(),
                                "config": config_id,
                                "model": name,
                                "frozen_layers": frozen_layers,
                                "lr": lr, "fold": k,
                                "train_mae": t, "val_mae": v,
                                "val_over_train": ratio,
                                "best_epoch": e})
        avg = float(np.mean(maes))
        print(f"  lr={lr:g} mean val MAE={avg:.5f}  "
              f"mean val/train ratio={np.nanmean(gaps):.2f}")
        if avg < best_mae:
            best_mae, best_lr = avg, lr
            best_epochs = int(round(float(np.mean(epochs_at_best))))

    diag = pd.DataFrame(diagnostics)
    gap = float(diag[diag["lr"] == best_lr]["val_over_train"].mean())
    triggered = bool(gap > config.BERT_FT_OVERFIT_GAP_TRIGGER)
    diag["overfit_gap_at_selected_lr"] = gap
    diag["overfit_trigger"] = config.BERT_FT_OVERFIT_GAP_TRIGGER
    diag["overfit_trigger_fired"] = int(triggered)
    _write_diagnostics(diag.to_dict("records"))

    if triggered and not fallback:
        print("  " + "!" * 66)
        print(f"  !! PRE-REGISTERED OVERFIT TRIGGER FIRED for {config_id}: mean "
              f"val/train MAE ratio {gap:.2f} > "
              f"{config.BERT_FT_OVERFIT_GAP_TRIGGER}")
        print("  !! The fallback is NOT applied automatically. Full "
              "fine-tuning remains the reported")
        print("  !! headline; run the pre-registered fallback separately and "
              "report BOTH:")
        print(f"  !!     python -m src.models.bert_finetune --config "
              f"{config_id} --fallback")
        print("  " + "!" * 66)
    elif not fallback:
        print(f"  overfit check: mean val/train MAE ratio {gap:.2f} <= "
              f"{config.BERT_FT_OVERFIT_GAP_TRIGGER} (trigger not fired)")

    print(f"  selected lr={best_lr:g}, epochs={best_epochs}, "
          f"frozen_layers={frozen_layers}")
    sh.save_hyperparams(name, {"learning_rate": best_lr,
                               "epochs": best_epochs,
                               "frozen_layers": frozen_layers,
                               "val_over_train_gap": gap,
                               "overfit_trigger_fired": triggered,
                               "device": device.type,
                               "epoch_cap": cap,
                               "torch_version": torch.__version__})

    sh.impute_lag(train, test)
    model, _, _, _, scaler = _train(train, None, texts, tokenizer, best_lr,
                                    best_epochs, frozen_layers)
    y_pred = _predict(model, test, texts, tokenizer, scaler)

    preds = test[["champion", "patch", "has_lag"]].copy()
    preds["y_true"] = test[sh.TARGET].to_numpy()
    preds["y_pred"] = y_pred
    sh.report(name, preds)
    sh.save_predictions(name, preds)


def run(config_id: str, target: str = "level", fallback: bool = False,
        epoch_cap: int | None = None) -> None:
    """Run under an explicit target, always handing back the level target."""
    try:
        sh.set_target(target)
        main(config_id, fallback, epoch_cap)
    finally:
        sh.set_target("level")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="C1", choices=list(config.ABLATION_CONFIGS))
    ap.add_argument("--target", default="level", choices=list(sh.TARGETS),
                    help="'level' (default, published) or 'delta' (work item D)")
    ap.add_argument("--fallback", action="store_true",
                    help="run the pre-registered partially-frozen fallback "
                         "(spec section 11, risk 2) as a separate model row")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override config.BERT_FT_EPOCHS for THIS run. Use "
                         "when the acceptance test fails (best epoch == cap in "
                         "a majority of folds); the cap used is recorded in "
                         "selected_hyperparameters.json so a published cell "
                         "always carries the budget that produced it.")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="exit without training if this (target, config)'s "
                         "predictions already exist. For resuming a "
                         "multi-hour sweep after an interruption -- each "
                         "config is a self-contained unit of work, but there "
                         "is no checkpointing WITHIN a config, so an "
                         "interrupted config restarts from scratch.")
    args = ap.parse_args()

    pause = config.OUTPUTS / ".pause_sweep"
    if pause.exists():
        print(f"[{MODEL_NAME}] PAUSED: {pause} exists, refusing to start "
              f"{args.target}/{args.config}.")
        print(f"[{MODEL_NAME}] Delete that file to resume "
              f"(and pass --skip-if-done so finished configs are not re-run).")
        sys.exit(0)

    if args.skip_if_done:
        sh.set_target(args.target)
        sh.set_config(args.config)
        name = FALLBACK_MODEL_NAME if args.fallback else MODEL_NAME
        done = (config.PREDICTIONS / f"{sh.target_prefix()}{args.config}"
                / f"{name}_preds.csv")
        sh.set_target("level")
        if done.exists():
            print(f"[{name}] {args.target}/{args.config} already done "
                  f"({done}) - skipping")
            sys.exit(0)

    run(args.config, args.target, args.fallback, args.epochs)
