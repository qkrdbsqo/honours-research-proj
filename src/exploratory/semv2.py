"""Rerun the structured pipeline under the semantic or historical encoding.

`setup` creates a working copy of the repository; every other stage runs inside that copy.

Usage:
    python -m src.exploratory.semv2 setup [--encoding historical]
    python -m src.exploratory.semv2 all
    python -m src.exploratory.semv2 extras_c2 | extras_cfg | block_configs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src import config
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src import config

ROOT = config.ROOT
INFO = "SEMV2_INFO.json"
WORK = ROOT / "semv2"
REF = WORK / "reference"
LOGS = WORK / "logs"
TABLES = WORK / "tables"
KEYS = ["champion", "patch"]

COPY_TREES = ["src", "tests", "data/raw/oracle", "data/raw/ddragon", "data/raw/patch_notes",
              "data/mappings", "data/processed"]
COPY_FILES = ["requirements.txt", "data/cooldown_audit.csv"]
REFERENCE = ["outputs/tables/paired_vs_persistence.csv", "outputs/tables/paired_vs_persistence_delta.csv"]
BERT_MODELS = ("bert", "bert_ctrl", "bert_ft")
BERT_ALIASES = {"C2": "C1", "C4": "C1", "C6": "C1"}


def _log(msg: str) -> None:
    print(str(msg).encode("ascii", "replace").decode("ascii"), flush=True)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def in_copy() -> bool:
    return "-semv2" in ROOT.name and (ROOT / INFO).exists()


def require_copy() -> None:
    if not in_copy():
        raise SystemExit(f"refusing to run: {ROOT} is not a semv2 copy (run 'setup' first, "
                         "then run stages from inside the copy)")


def info() -> dict:
    return json.loads((ROOT / INFO).read_text())


def setup(encoding: str, dest: Path | None, force: bool) -> Path:
    if "-semv2" in ROOT.name:
        raise SystemExit("run setup from the real repository, not from a copy")
    dest = dest or ROOT.parent / (ROOT.name + ("-semv2" if encoding == "semantic" else "-semv2-hist"))
    if dest.exists() and not force:
        raise SystemExit(f"{dest} exists; pass --force to add missing files (nothing is deleted)")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.npy")
    for rel in COPY_TREES:
        if (ROOT / rel).exists():
            shutil.copytree(ROOT / rel, dest / rel, ignore=ignore, dirs_exist_ok=True)
    for rel in COPY_FILES:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest / rel)
    for rel in REFERENCE:
        src, out = ROOT / rel, dest / "semv2" / "reference" / rel
        if not src.exists():
            src = ROOT / "data" / "reference" / Path(rel).name
        if src.is_dir():
            shutil.copytree(src, out, dirs_exist_ok=True)
        elif src.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
    copied = []
    saved = ROOT / "results" / encoding / "predictions"
    for f in sorted(saved.glob("*/bert*_preds.csv")):
        out = dest / "outputs/predictions" / f.parent.name / f.name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        copied.append({"from": str(f.relative_to(ROOT)), "to": str(out.relative_to(dest))})
    meta = {"encoding": encoding, "source_root": str(ROOT), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version.split()[0], "copied_predictions": copied,
            "input_md5": {rel: _md5(ROOT / rel) for rel in
                          ("data/processed/patch_note_features.csv", "data/processed/aligned_dataset.csv",
                           "data/processed/pwds.csv", "data/processed/stat_deltas.csv")}}
    (dest / INFO).write_text(json.dumps(meta, indent=2))
    _log(f"[semv2] copy ready at {dest} ({encoding}); {len(copied)} canonical BERT prediction files copied")
    subprocess.run([sys.executable, "-m", "src.exploratory.semv2", "build"], cwd=dest, check=True)
    return dest


def build() -> None:
    from src.exploratory import semantic_dropin as sd
    from src.features import slot_features as sf
    from src.data import align
    require_copy()
    WORK.mkdir(exist_ok=True)
    enc = info()["encoding"]
    pnf_path = config.PATCH_NOTE_FEATURES_CSV
    hist = pd.read_csv(pnf_path, dtype={"patch": str})
    keep = ROOT / "data/processed/patch_note_features_historical.csv"
    if not keep.exists():
        hist.to_csv(keep, index=False)
    hist = pd.read_csv(keep, dtype={"patch": str})
    receipt = {"encoding": enc}
    if enc == "semantic":
        new = sd.build()
        assert len(new) == len(hist), (len(new), len(hist))
        a, b = hist.set_index(KEYS).sort_index(), new.set_index(KEYS).sort_index()
        assert a.index.equals(b.index)
        unchanged = [c for c in a.columns if c not in sd.RECOMPUTED and c != "season"]
        for c in unchanged:
            assert np.allclose(a[c].astype(float), b[c].astype(float)), c
        receipt["patch_note_cells_changed"] = {c: int((~np.isclose(a[c].astype(float), b[c].astype(float))).sum())
                                               for c in sd.RECOMPUTED}
        new.to_csv(pnf_path, index=False)
    else:
        hist.to_csv(pnf_path, index=False)
    sf.main()
    align.write_aligned_dataset()
    receipt["aligned_md5"] = _md5(config.ALIGNED_DATASET_CSV)
    receipt["patch_note_md5"] = _md5(pnf_path)
    (WORK / "build_receipt.json").write_text(json.dumps(receipt, indent=2))
    _log(json.dumps(receipt, indent=2))


def _run(label: str, fn) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    _log(f"[semv2] {label}: started")
    t0 = time.time()
    with (LOGS / f"{label}.log").open("w") as log, redirect_stdout(log):
        fn()
    _log(f"[semv2] {label}: done in {time.time() - t0:.0f}s")


def frozen() -> None:
    from src.models import _shared as sh, persistence, ridge, xgb
    require_copy()
    jobs = [("level", c) for c in config.ABLATION_CONFIGS] + [("delta", c) for c in config.DELTA_CONFIGS]
    for target, c in jobs:
        for name, mod in (("persistence", persistence), ("ridge", ridge), ("xgb", xgb)):
            key = ("D-" if target == "delta" else "") + c
            if (config.PREDICTIONS / key / f"{name}_preds.csv").exists() and (WORK / "frozen_done" / f"{key}_{name}").exists():
                continue

            def go(mod=mod, c=c, target=target):
                sh.set_target(target)
                try:
                    sh.set_config(c)
                    mod.main(c)
                finally:
                    sh.set_target("level")
            _run(f"frozen_{key}_{name}", go)
            (WORK / "frozen_done").mkdir(parents=True, exist_ok=True)
            (WORK / "frozen_done" / f"{key}_{name}").write_text("ok")


def _pred(target: str, c: str, model: str) -> pd.DataFrame:
    src = BERT_ALIASES.get(c, c) if model in BERT_MODELS else c
    key = ("D-" if target == "delta" else "") + src
    p = config.PREDICTIONS / key / f"{model}_preds.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p, dtype={"patch": str})


def families() -> None:
    from src.evaluation import metrics as ev
    from src.evaluation import paired as pc
    require_copy()
    TABLES.mkdir(parents=True, exist_ok=True)
    for target, suffix, expected in (("level", "", 90), ("delta", "_delta", 50)):
        template = pd.read_csv(REF / f"outputs/tables/paired_vs_persistence{suffix}.csv")
        assert int(template.counted_in_family.sum()) == expected
        for method in ("row", "patch"):
            rows = []
            for r in template.itertuples():
                j = pc.paired_frame(_pred(target, r.config, "persistence"), _pred(target, r.config, r.model), r.subset)
                if method == "row":
                    diff, lo, hi, stats = ev.paired_bootstrap_mae_diff(
                        j.y_true_b, j.y_pred_c, j.y_pred_b, n_boot=pc.N_BOOT, return_stats=True)
                    res = {"n": len(j), "mae_diff": diff, "ci_lo": lo, "ci_hi": hi,
                           "p_value": ev.bootstrap_two_sided_pvalue(stats),
                           "verdict": "better" if hi < 0 else "worse" if lo > 0 else "tie"}
                else:
                    res = pc.cluster_result(j)
                rows.append({"target": target, "config": r.config, "model": r.model, "subset": r.subset,
                             "against": "persistence", "method": method,
                             "aliases": "" if r.counted_in_family else r.aliases,
                             "counted_in_family": r.counted_in_family, **res})
            t = pc.apply_family(pd.DataFrame(rows))
            assert t.m.iloc[0] == expected
            t.to_csv(TABLES / f"frozen_{target}_{method}_paired.csv", index=False)
            _log(f"[semv2] family {target}/{method}: m={expected}, "
                 f"Holm-significant {int(t[t.counted_in_family.astype(bool)].holm_significant.sum())}")
    metrics = []
    for target, configs in (("level", config.ABLATION_CONFIGS), ("delta", config.DELTA_CONFIGS)):
        for c in configs:
            for model in ("persistence", "ridge", "xgb", "bert", "bert_ctrl", "bert_ft"):
                if target == "delta" and model == "bert_ctrl":
                    continue
                p = _pred(target, c, model)
                for subset in ("full", "has_lag"):
                    if model == "persistence" and subset == "full":
                        continue
                    d = p if subset == "full" else p[p.has_lag == 1]
                    metrics.append({"target": target, "config": c, "model": model, "subset": subset,
                                    "n": len(d), "mae": ev.mae(d.y_true, d.y_pred),
                                    "rho_per_patch": ev.spearman_per_patch(d)[0]})
    pd.DataFrame(metrics).to_csv(TABLES / "frozen_metrics.csv", index=False)


def rolling() -> None:
    from src.evaluation import rolling_origin as ro
    require_copy()
    for c in tuple(ro.CONFIGS) + ("C3",):
        if (ro._pred_dir(c) / "xgb_preds.csv").exists():
            continue
        _run(f"rolling_{c}", lambda c=c: ro.main(c))


def block() -> None:
    from src.exploratory import rolling_block as rb
    from src.exploratory import rolling_block_persistence as rbp
    require_copy()
    for target in ("level", "delta"):
        _run(f"block_{target}", lambda target=target: rb.main(rb.BLOCK_SIZES, target))
    _run("block_persistence", rbp.main)


def diagnostics() -> None:
    from src.evaluation import decision_relevance as dr
    from src.evaluation import tier_anatomy as ta
    require_copy()
    _run("decision_relevance", dr.main)
    _run("tier_anatomy", ta.main)


EXTRA_CONFIG = "C1X"
CONTROL_CONFIG = "C1L"
FEATURE_OVERRIDES: dict[str, list[str]] = {}


def _register_extras() -> Path:
    """Register config C1X for this process and point the dataset at a merged copy."""
    from src.exploratory import extra_features as xf
    merged = WORK / "aligned_with_extras.csv"
    if not merged.exists():
        x = xf.build()
        x.to_csv(WORK / "extra_features.csv", index=False)
        d = pd.read_csv(config.ALIGNED_DATASET_CSV, dtype={"patch": str})
        d.merge(x, on=KEYS, how="left", validate="one_to_one").to_csv(merged, index=False)
    config.ABLATION_CONFIGS[EXTRA_CONFIG] = {**config.ABLATION_CONFIGS["C1"],
                                             "extra_features": list(xf.EXTRA_COLS)}
    config.ABLATION_CONFIGS[CONTROL_CONFIG] = {**config.ABLATION_CONFIGS["C1"], "extra_features": []}
    FEATURE_OVERRIDES.setdefault(CONTROL_CONFIG, list(xf.CONTROL_FEATURES))
    if not getattr(config.features_for, "_semv2", False):
        original = config.features_for

        def features_for(config_id, _orig=original):
            return list(FEATURE_OVERRIDES[config_id]) if config_id in FEATURE_OVERRIDES else _orig(config_id)
        features_for._semv2 = True
        config.features_for = features_for
    config.ALIGNED_DATASET_CSV = merged
    return merged


def extras() -> None:
    from src.models import _shared as sh, persistence, ridge, xgb
    from src.evaluation import rolling_origin as ro
    from src.evaluation import paired as pc
    from src.exploratory import extra_features as xf
    from src.evaluation import compare as cmp
    require_copy()
    original = config.ALIGNED_DATASET_CSV
    _register_extras()
    try:
        for cfg in (EXTRA_CONFIG, CONTROL_CONFIG):
            for target in ("level", "delta"):
                for name, mod in (("persistence", persistence), ("ridge", ridge), ("xgb", xgb)):
                    key = ("D-" if target == "delta" else "") + cfg
                    if (config.PREDICTIONS / key / f"{name}_preds.csv").exists():
                        continue

                    def go(mod=mod, target=target, cfg=cfg):
                        sh.set_target(target)
                        try:
                            sh.set_config(cfg)
                            mod.main(cfg)
                        finally:
                            sh.set_target("level")
                    _run(f"extras_{key}_{name}", go)
            if not (ro._pred_dir(cfg) / "xgb_preds.csv").exists():
                sh.set_target("level")
                _run(f"extras_rolling_{cfg}", lambda cfg=cfg: ro.main(cfg))
    finally:
        config.ALIGNED_DATASET_CSV = original
    rows = []
    for target in ("level", "delta"):
        for model in ("ridge", "xgb"):
            for subset in ("has_lag", "changed_has_lag"):
                j = pc.paired_frame(_pred(target, "C1", model), _pred(target, EXTRA_CONFIG, model), subset)
                rows.append({"evaluation": "frozen", "target": target, "model": model, "subset": subset,
                             **pc.cluster_result(j)})
    for model in ("ridge", "xgb"):
        for subset in ("has_lag", "changed_has_lag"):
            base = pd.read_csv(ro._pred_dir("C1") / f"{model}_preds.csv", dtype={"patch": str})
            cand = pd.read_csv(ro._pred_dir(EXTRA_CONFIG) / f"{model}_preds.csv", dtype={"patch": str})
            j = pc.paired_frame(base, cand, subset)
            rows.append({"evaluation": "rolling", "target": "level", "model": model, "subset": subset,
                         **pc.cluster_result(j)})
    ctrl = []
    for cand_cfg, base_cfg in ((CONTROL_CONFIG, "persistence"), (EXTRA_CONFIG, CONTROL_CONFIG)):
        for target in ("level", "delta"):
            for model in ("ridge", "xgb"):
                base = _pred(target, CONTROL_CONFIG, "persistence") if base_cfg == "persistence" \
                    else _pred(target, base_cfg, model)
                j = pc.paired_frame(base, _pred(target, cand_cfg, model), "has_lag")
                ctrl.append({"evaluation": "frozen", "target": target, "model": model,
                             "candidate": cand_cfg, "baseline": base_cfg, **pc.cluster_result(j)})
        for model in ("ridge", "xgb"):
            base = pd.read_csv(ro._pred_dir(CONTROL_CONFIG if base_cfg != "persistence" else CONTROL_CONFIG)
                               / f"{'persistence' if base_cfg == 'persistence' else model}_preds.csv",
                               dtype={"patch": str})
            cand = pd.read_csv(ro._pred_dir(cand_cfg) / f"{model}_preds.csv", dtype={"patch": str})
            j = pc.paired_frame(base, cand, "has_lag")
            ctrl.append({"evaluation": "rolling", "target": "level", "model": model,
                         "candidate": cand_cfg, "baseline": base_cfg, **pc.cluster_result(j)})
    pd.DataFrame(ctrl).to_csv(TABLES / "extras_controls_posthoc.csv", index=False)
    _log("post hoc controls (descriptive, uncorrected):")
    _log(pd.DataFrame(ctrl).round(6).to_string(index=False))
    t = pd.DataFrame(rows)
    t["m"] = len(t)
    t["holm_significant"] = cmp.holm_bonferroni(t["p_value"], resolution=2 / pc.N_BOOT)
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "extras_vs_C1.csv", index=False)
    x = pd.read_csv(WORK / "extra_features.csv", dtype={"patch": str})[KEYS + ["lag_picks"]]
    desc = []
    for model in ("ridge", "xgb"):
        base = pd.read_csv(ro._pred_dir("C1") / f"{model}_preds.csv", dtype={"patch": str})
        cand = pd.read_csv(ro._pred_dir(EXTRA_CONFIG) / f"{model}_preds.csv", dtype={"patch": str})
        j = base.merge(cand, on=KEYS, suffixes=("_b", "_c")).merge(x, on=KEYS)
        j = j[j.has_lag_b == 1]
        j["third"] = pd.qcut(j.lag_picks.rank(method="first"), 3, labels=["thin", "middle", "thick"])
        for third, g in j.groupby("third", observed=True):
            desc.append({"model": model, "lag_picks_third": third, "n": len(g),
                         "mae_C1": float(np.abs(g.y_true_b - g.y_pred_b).mean()),
                         "mae_C1X": float(np.abs(g.y_true_b - g.y_pred_c).mean())})
    pd.DataFrame(desc).to_csv(TABLES / "extras_by_lag_picks.csv", index=False)
    imp_path = config.TABLES / EXTRA_CONFIG / "xgb_feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        name_col = imp.columns[0]
        imp = imp.sort_values(imp.columns[1], ascending=False).reset_index(drop=True)
        imp["rank"] = imp.index + 1
        imp["is_extra"] = imp[name_col].isin(xf.EXTRA_COLS)
        imp.to_csv(TABLES / "extras_importance.csv", index=False)
        _log(imp.head(15).to_string(index=False))
    _log(t.round(6).to_string(index=False))
    _log(pd.DataFrame(desc).round(6).to_string(index=False))


EXTRAS_CFG_DOC = """C3X = C3 + 19 extras, C4X = C4 + 19 extras (extra_features.EXTRA_COLS).

Post hoc: requested after the C1X/C1L results were seen. C1L needs no C3/C4
version: it uses no patch features, so it is the same model for every config.

FAMILY (Holm, one family, 24 tests; patch-cluster bootstrap frozen, origin-cluster rolling)
  {C3X vs C3, C4X vs C4} x [frozen {ridge, xgb} x {level, delta} x {has_lag, changed}
                            + rolling {ridge, xgb} x {has_lag, changed}]
DESCRIPTIVE (uncorrected): C3X vs C1L and C4X vs C1L (patch information on top of
the lagged stats), frozen level/delta and rolling, has_lag.

PRE-REGISTERED PREDICTIONS (written 2026-10-18, before any C3X/C4X fit)
  W1  has_lag: C3X and C4X have lower MAE than their base config in all 8 has_lag
      cells, and at least 5 of the 8 survive Holm (C1X vs C1: 5 of 6).
  W2  changed_has_lag: no Holm-significant cell.
  W3  C3X vs C1L and C4X vs C1L, xgb: every frozen cell within +-0.001 and the
      rolling cell between -0.0012 and 0 (C1X vs C1L rolling xgb was -0.00078).
  W4  rolling xgb has-lag MAE of C3X and C4X within 0.0005 of C1X (0.03359).
"""
MORE_CONFIGS = {"C3X": "C3", "C4X": "C4"}
EXTRAS_C2_DOC = """C2X = C2 + the 19 extras (post hoc, set up and run 2026-10-18 after the C3X/C4X
results). Its own 12-test Holm family, {C2X vs C2} x [frozen {ridge, xgb} x {level,
delta} x {has_lag, changed} + rolling {ridge, xgb} x {has_lag, changed}], so the
C3X/C4X family of 24 stays as reported. Base runs that did not exist yet (frozen
delta C2, rolling C2) are fitted here first.

PRE-REGISTERED PREDICTIONS (written 2026-10-18, before any C2X fit)
  V1  has_lag: C2X has lower MAE than C2 in all 6 has_lag cells; at least 4 of the
      6 survive Holm.
  V2  changed_has_lag: no Holm-significant cell.
  V3  C2X vs C1L, xgb: every frozen cell within +-0.001 and the rolling cell
      between -0.0012 and 0.
  V4  rolling xgb has-lag MAE of C2X within 0.0005 of C1X.
"""
C2_CONFIGS = {"C2X": "C2"}


def _register_more(configs=None) -> None:
    from src.exploratory import extra_features as xf
    _register_extras()
    for name, base in (configs or MORE_CONFIGS).items():
        config.ABLATION_CONFIGS[name] = {**config.ABLATION_CONFIGS[base],
                                         "extra_features": list(config.ABLATION_CONFIGS[base]["extra_features"])
                                         + list(xf.EXTRA_COLS)}


def extras_cfg(configs=None, tag: str = "") -> None:
    configs = configs or MORE_CONFIGS
    from src.models import _shared as sh, persistence, ridge, xgb
    from src.evaluation import rolling_origin as ro
    from src.evaluation import paired as pc
    from src.evaluation import compare as cmp
    require_copy()
    original = config.ALIGNED_DATASET_CSV
    _register_more(configs)
    try:
        for base in sorted(set(configs.values())):   # base runs the earlier stages may not have made
            for target in ("level", "delta"):
                for name, mod in (("persistence", persistence), ("ridge", ridge), ("xgb", xgb)):
                    key = ("D-" if target == "delta" else "") + base
                    if (config.PREDICTIONS / key / f"{name}_preds.csv").exists():
                        continue

                    def go(mod=mod, target=target, cfg=base):
                        sh.set_target(target)
                        try:
                            sh.set_config(cfg)
                            mod.main(cfg)
                        finally:
                            sh.set_target("level")
                    _run(f"extras_cfg_base_{key}_{name}", go)
            if not (ro._pred_dir(base) / "xgb_preds.csv").exists():
                sh.set_target("level")
                _run(f"extras_cfg_rolling_{base}", lambda cfg=base: ro.main(cfg))
        for cfg in configs:
            for target in ("level", "delta"):
                for name, mod in (("persistence", persistence), ("ridge", ridge), ("xgb", xgb)):
                    key = ("D-" if target == "delta" else "") + cfg
                    if (config.PREDICTIONS / key / f"{name}_preds.csv").exists():
                        continue

                    def go(mod=mod, target=target, cfg=cfg):
                        sh.set_target(target)
                        try:
                            sh.set_config(cfg)
                            mod.main(cfg)
                        finally:
                            sh.set_target("level")
                    _run(f"extras_cfg_{key}_{name}", go)
            if not (ro._pred_dir(cfg) / "xgb_preds.csv").exists():
                sh.set_target("level")
                _run(f"extras_cfg_rolling_{cfg}", lambda cfg=cfg: ro.main(cfg))
    finally:
        config.ALIGNED_DATASET_CSV = original

    def rolling_pred(cfg, model):
        return pd.read_csv(ro._pred_dir(cfg) / f"{model}_preds.csv", dtype={"patch": str})

    rows, desc, mae = [], [], []
    for cand, base in configs.items():
        for target in ("level", "delta"):
            for model in ("ridge", "xgb"):
                for subset in ("has_lag", "changed_has_lag"):
                    j = pc.paired_frame(_pred(target, base, model), _pred(target, cand, model), subset)
                    rows.append({"pair": f"{cand} vs {base}", "evaluation": "frozen", "target": target,
                                 "model": model, "subset": subset, **pc.cluster_result(j)})
                j = pc.paired_frame(_pred(target, CONTROL_CONFIG, model), _pred(target, cand, model), "has_lag")
                desc.append({"pair": f"{cand} vs {CONTROL_CONFIG}", "evaluation": "frozen", "target": target,
                             "model": model, "subset": "has_lag", **pc.cluster_result(j)})
        for model in ("ridge", "xgb"):
            for subset in ("has_lag", "changed_has_lag"):
                j = pc.paired_frame(rolling_pred(base, model), rolling_pred(cand, model), subset)
                rows.append({"pair": f"{cand} vs {base}", "evaluation": "rolling", "target": "level",
                             "model": model, "subset": subset, **pc.cluster_result(j)})
            j = pc.paired_frame(rolling_pred(CONTROL_CONFIG, model), rolling_pred(cand, model), "has_lag")
            desc.append({"pair": f"{cand} vs {CONTROL_CONFIG}", "evaluation": "rolling", "target": "level",
                         "model": model, "subset": "has_lag", **pc.cluster_result(j)})
    for key in ("C1", "C1X", "C1L") + tuple(k for kv in configs.items() for k in kv[::-1]):
        for target in ("level", "delta"):
            for model in ("persistence", "ridge", "xgb"):
                q = _pred(target, key, model)
                q = q[q.has_lag == 1]
                mae.append({"config": key, "evaluation": "frozen", "target": target, "model": model,
                            "mae": float((q.y_true - q.y_pred).abs().mean())})
        for model in ("persistence", "ridge", "xgb"):
            q = rolling_pred(key, model)
            q = q[q.has_lag == 1]
            mae.append({"config": key, "evaluation": "rolling", "target": "level", "model": model,
                        "mae": float((q.y_true - q.y_pred).abs().mean())})
    t = pd.DataFrame(rows)
    assert len(t) == 12 * len(configs), len(t)
    t["m"] = len(t)
    t["holm_significant"] = cmp.holm_bonferroni(t["p_value"], resolution=2 / pc.N_BOOT)
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / f"extras_cfg{tag}_tests.csv", index=False)
    pd.DataFrame(desc).to_csv(TABLES / f"extras_cfg{tag}_descriptive.csv", index=False)
    pd.DataFrame(mae).to_csv(TABLES / f"extras_cfg{tag}_mae.csv", index=False)
    _log(pd.DataFrame(mae).pivot_table(index=["evaluation", "target", "model"], columns="config",
                                       values="mae").round(5).to_string())
    _log(t.round(6).to_string(index=False))
    _log(pd.DataFrame(desc).round(6).to_string(index=False))


def block_configs_stage(smoke: bool = False) -> None:
    """C2, C4 and their X versions on the block target (block_configs.py)."""
    from src.exploratory import block_configs as bc
    require_copy()
    if not (WORK / "block_extras" / "preds_C1L_B3_delta.csv").exists():
        raise SystemExit("run the `block_extras` stage first (C1L block predictions are the reference)")
    _run("block_configs" + ("_smoke" if smoke else ""), lambda: bc.main(smoke=smoke))


def block_extras_stage() -> None:
    """C1X and C1L on the block target, B = 1, 2, 3 (block_extras.py)."""
    from src.exploratory import block_extras as be
    require_copy()
    if not (WORK / "extra_features.csv").exists():
        raise SystemExit("run the `extras` stage first (it writes semv2/extra_features.csv)")
    _run("block_extras", be.main)


STAGES = ("build", "frozen", "families", "rolling", "block", "diagnostics", "extras", "block_extras")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("setup", "all", "extras_cfg", "extras_c2", "block_configs") + STAGES)
    ap.add_argument("--smoke", action="store_true", help="block_configs: 2-origin wiring check, writes nothing")
    ap.add_argument("--encoding", choices=("semantic", "historical"), default="semantic")
    ap.add_argument("--dest", type=Path)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.stage == "setup":
        setup(a.encoding, a.dest, a.force)
        return 0
    require_copy()
    run = {"build": build, "frozen": frozen, "families": families, "rolling": rolling, "block": block,
           "diagnostics": diagnostics, "extras": extras, "extras_cfg": extras_cfg,
           "extras_c2": lambda: extras_cfg(C2_CONFIGS, "_c2"),
           "block_configs": lambda: block_configs_stage(a.smoke),
           "block_extras": block_extras_stage}
    for s in (STAGES[1:] if a.stage == "all" else (a.stage,)):
        run[s]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
