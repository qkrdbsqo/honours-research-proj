"""Run every model under one or all configurations.

Usage:
    python -m src.models.run_ablation --config C0
    python -m src.models.run_ablation --all
"""
from __future__ import annotations

import argparse
import sys

try:
    from src import config
    from src.models import _shared as sh
    from src.models import persistence, ridge, xgb, bert_regressor
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.models import _shared as sh
    from src.models import persistence, ridge, xgb, bert_regressor

RUNNERS = {
    "persistence": persistence.main,
    "ridge": ridge.main,
    "xgb": xgb.main,
    "bert": bert_regressor.main,
}

BERT_ALIASES = {"C2": "C1", "C4": "C1", "C6": "C1"}


def run_config(config_id: str) -> None:
    print(f"\n{'=' * 70}\n== ablation config {config_id} "
          f"({', '.join(config.ABLATION_CONFIGS[config_id]['seasons'])}, "
          f"{config.ABLATION_CONFIGS[config_id]['folds']} folds, "
          f"{config.ABLATION_CONFIGS[config_id]['text']} text)\n{'=' * 70}")
    for name, runner in RUNNERS.items():
        if name == "bert" and config_id in BERT_ALIASES:
            print(f"\n-- bert: skipped, identical to "
                  f"{BERT_ALIASES[config_id]} (same text variant)")
            continue
        print(f"\n-- {name}")
        sh.set_config(config_id)
        runner(config_id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=list(config.ABLATION_CONFIGS))
    ap.add_argument("--all", action="store_true", help="run every config")
    args = ap.parse_args()
    if args.all:
        for cid in config.ABLATION_CONFIGS:
            run_config(cid)
    elif args.config:
        run_config(args.config)
    else:
        ap.error("pass --config <id> or --all")


if __name__ == "__main__":
    main()