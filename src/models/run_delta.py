"""Run every model on the difference target.

Usage:
    python -m src.models.run_delta --config C1
    python -m src.models.run_delta --all
"""
from __future__ import annotations

import argparse
import sys

try:
    from src import config
    from src.models import _shared as sh
    from src.models import run_ablation
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.models import _shared as sh
    from src.models import run_ablation


def run_config(config_id: str) -> None:
    print(f"\n{'=' * 70}\n== DELTA target, config {config_id}\n{'=' * 70}")
    try:
        for name, runner in run_ablation.RUNNERS.items():
            if name == "bert" and config_id in run_ablation.BERT_ALIASES:
                print(f"\n-- bert: skipped, identical to "
                      f"{run_ablation.BERT_ALIASES[config_id]} (same text variant)")
                continue
            print(f"\n-- {name}")
            sh.set_target("delta")
            sh.set_config(config_id)
            runner(config_id)
    finally:
        sh.set_target("level")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=config.DELTA_CONFIGS)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        for cid in config.DELTA_CONFIGS:
            run_config(cid)
    elif args.config:
        run_config(args.config)
    else:
        ap.error("pass --config <id> or --all")


if __name__ == "__main__":
    main()
