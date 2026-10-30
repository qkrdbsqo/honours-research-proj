"""Persistence baseline: predict the previous patch's PWDS.

Usage:
    python -m src.models.persistence --config C0
"""
from __future__ import annotations

import sys

try:
    from src import config
    from src.models import _shared as sh
except ModuleNotFoundError:  # allow running as a loose script
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src import config
    from src.models import _shared as sh


def main(config_id: str = "C0") -> None:
    sh.set_config(config_id)

    df = sh.load_dataset()
    _, test = sh.train_test_split(df)

    d = test[test["has_lag"] == 1].copy()

    preds = d[["champion", "patch", "has_lag"]].copy()
    preds["y_true"] = d[sh.TARGET].to_numpy()
    if sh.active_target() == "delta":
        preds["y_pred"] = 0.0
    else:
        preds["y_pred"] = d[sh.LAG].to_numpy()

    sh.report("persistence", preds, subsets=("has_lag",))
    sh.save_predictions("persistence", preds)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="C0", choices=list(config.ABLATION_CONFIGS))
    main(ap.parse_args().config)
