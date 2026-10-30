"""Paths, constants, feature lists, data splits and hyperparameter grids."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
ORACLE_DIR = RAW / "oracle"
DDRAGON_DIR = RAW / "ddragon"
PATCH_NOTES_DIR = RAW / "patch_notes"

PROCESSED = DATA / "processed"
MAPPINGS = DATA / "mappings"

OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
PREDICTIONS = OUTPUTS / "predictions"

SEASONS: list[str] = ["2024", "2025"]
SEASON_MAJORS: dict[str, str] = {"2024": "14", "2025": "15"}

ORACLE_CSVS: dict[str, Path] = {
    "2024": ORACLE_DIR / "2024_LoL_esports_match_data_from_OraclesElixir.csv",
    "2025": ORACLE_DIR / "2025_LoL_esports_match_data_from_OraclesElixir.csv",
}
ORACLE_CSV = ORACLE_CSVS["2025"]      # back-compat alias
SEASON_MAJOR = "15"                    # back-compat alias

SEASON_PATCHES: dict[str, list[str]] = {
    "2024": [f"14.{m:02d}" for m in range(1, 24)],
    "2025": [f"15.{m:02d}" for m in range(1, 25)],
}
ALL_PATCHES: list[str] = SEASON_PATCHES["2024"] + SEASON_PATCHES["2025"]
DROP_PATCHES: set[str] = {"13.24"}


def season_of(patch: str) -> str:
    """'14.07' -> '2024'."""
    major = str(patch).split(".")[0]
    return {v: k for k, v in SEASON_MAJORS.items()}[major]


VERSION_MAP = MAPPINGS / "version_map.json"
CHAMPION_NAMES = MAPPINGS / "champion_names.json"

STAT_DELTAS_CSV = PROCESSED / "stat_deltas.csv"
PATCH_NOTE_FEATURES_CSV = PROCESSED / "patch_note_features.csv"
PATCH_TEXTS_JSON = PROCESSED / "patch_texts.json"
PWDS_CSV = PROCESSED / "pwds.csv"
ALIGNED_DATASET_CSV = PROCESSED / "aligned_dataset.csv"
CLS_EMBEDDINGS_NPY = PROCESSED / "cls_embeddings.npy"
CLS_INDEX_CSV = PROCESSED / "cls_index.csv"   # (champion, patch) row order of the .npy

SELECTED_HYPERPARAMS_JSON = TABLES / "selected_hyperparameters.json"
MODEL_COMPARISON_CSV = TABLES / "model_comparison.csv"
PAIRED_COMPARISON_CSV = TABLES / "paired_vs_persistence.csv"
DELTA_COMPARISON_CSV = TABLES / "model_comparison_delta.csv"
DELTA_PAIRED_CSV = TABLES / "paired_vs_persistence_delta.csv"
DELTA_CONFIGS: list[str] = ["C1", "C3", "C4"]
POWER_ANALYSIS_CSV = TABLES / "power_analysis.csv"
FEATURE_SCREEN_CSV = TABLES / "feature_degeneracy_screen.csv"
BLOCK_ABLATION_CSV = TABLES / "block_ablation.csv"


def xgb_feature_importance_csv(config_id: str) -> Path:
    """Gain-based XGBoost feature-importance table, namespaced per ablation config (same pattern as _shared.save_predictions)."""
    return TABLES / config_id / "xgb_feature_importance.csv"


def bert_ft_diagnostics_csv(config_id: str) -> Path:
    """Per-fold fine-tuning diagnostics (train/val MAE gap), namespaced."""
    return TABLES / config_id / "bert_ft_fit_diagnostics.csv"

PATCH_SECTIONS_JSON: dict[str, Path] = {
    s: PATCH_NOTES_DIR / f"patch_sections_{s}.json" for s in SEASONS
}
SECTION_COVERAGE_CSV = TABLES / "section_scrape_coverage.csv"

ITEM_CHANGES_CSV = PROCESSED / "item_changes.csv"
ITEM_FEATURES_CSV = PROCESSED / "item_features.csv"
SLOT_FEATURES_CSV = PROCESSED / "slot_features.csv"

PATCH_TEXTS_ENRICHED_JSON = PROCESSED / "patch_texts_enriched.json"
CLS_EMBEDDINGS_ENRICHED_NPY = PROCESSED / "cls_embeddings_enriched.npy"
CLS_INDEX_ENRICHED_CSV = PROCESSED / "cls_index_enriched.csv"

BASELINE_DIR = OUTPUTS / "baseline_2025only"

BASELINE_PRE_CONVERGENCE_DIR = OUTPUTS / "baseline_pre_convergence"

SUBSET_SIZES_CSV = TABLES / "subset_size_comparison.csv"
DEAD_ZONE_CSV = TABLES / "dead_zone_by_config.csv"
GENERALISATION_CSV = TABLES / "generalisation_2024_to_2025.csv"
GENERALISATION_PAIRED_CSV = TABLES / "generalisation_paired_vs_persistence.csv"
GENERALISATION_LIKE_FOR_LIKE_CSV = TABLES / "generalisation_like_for_like.csv"

EXPLORATORY_BUILDS = DATA / "builds"
ITEM_DELTAS_SRC = EXPLORATORY_BUILDS / "items_runes" / "item_deltas.csv"
ITEMS_BY_PATCH_SRC = EXPLORATORY_BUILDS / "items_runes" / "items_by_patch.csv"
RUNES_BY_PATCH_SRC = (EXPLORATORY_BUILDS / "items_runes"
                      / "runes_by_patch.csv")
CLASS_AFFINITY_SRC = (EXPLORATORY_BUILDS / "class_affinity"
                      / "champion_class_item_affinity.csv")
LEAGUEPEDIA_USAGE_SRC = (EXPLORATORY_BUILDS / "leaguepedia"
                         / "leaguepedia_usage_by_champion_patch.csv")

DDRAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMPION_URL = (
    "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
)


MIN_GAMES = 5            # minimum games for a champion-patch to be scored
WR_CENTRE = 0.5          # win rate is centred here

PWDS_WEIGHTS = {
    "equal": (0.33, 0.33, 0.34),
    "presence_heavy": (0.40, 0.40, 0.20),
    "ban_dominant": (0.25, 0.50, 0.25),
}
DEFAULT_PWDS_WEIGHTS = "presence_heavy"   # fixed before the test set was used

RANDOM_SEED = 42
BERT_MODEL = "bert-base-uncased"
BERT_HIDDEN = 768

RIDGE_ALPHAS: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
XGB_GRID: dict[str, list] = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [50, 100, 200],
    "reg_lambda": [1, 5, 10],
}
BERT_LR_GRID: list[float] = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
BERT_MAX_EPOCHS = 5000
BERT_PATIENCE = 200

BERT_FT_LR_GRID: list[float] = [2e-5, 3e-5, 5e-5]
BERT_FT_EPOCHS = 12              # best-epoch restore on fold validation MAE
BERT_FT_BATCH = 16
BERT_FT_WEIGHT_DECAY = 0.01
BERT_FT_HEAD_LR = 1e-2
BERT_FT_WARMUP_FRAC = 0.10
BERT_FT_MAX_TOKENS = 256         # only 0.2% (plain) / 3.3% (enriched) exceed this
BERT_FT_OVERFIT_GAP_TRIGGER = 2.0
BERT_FT_FALLBACK_FROZEN_LAYERS = 8

N_BOOTSTRAP = 20000       # resamples for 95% bootstrap CIs (metrics + comparisons)
SPARSE_PATCH_MIN_GAMES = 100

STAT_DELTA_COLS: list[str] = [
    "hp_delta", "hpperlevel_delta", "mp_delta", "mpperlevel_delta",
    "movespeed_delta", "armor_delta", "armorperlevel_delta", "spellblock_delta",
    "spellblockperlevel_delta", "attackrange_delta", "hpregen_delta",
    "hpregenperlevel_delta", "mpregen_delta", "mpregenperlevel_delta",
    "crit_delta", "critperlevel_delta", "attackdamage_delta",
    "attackdamageperlevel_delta", "attackspeedperlevel_delta", "attackspeed_delta",
    "n_stats_changed", "net_direction",
]

PATCH_NOTE_FEATURE_COLS: list[str] = [
    "n_buffs", "n_nerfs", "n_changes_total", "n_abilities_changed",
    "mean_pct_change", "sum_signed_pct", "max_buff_pct", "max_nerf_pct",
    "n_parseable_changes", "is_adjustment", "buff_ratio",
    "passive_changed", "q_changed", "w_changed", "e_changed",
    "r_changed", "base_stats_changed",
]

PATH_A_FEATURES: list[str] = (
    STAT_DELTA_COLS                                  # 22 Data Dragon base-stat cols
    + PATCH_NOTE_FEATURE_COLS                        # 17 patch-note-derived features
    + [f"lag_pwds_{DEFAULT_PWDS_WEIGHTS}", "has_lag"]  # temporal momentum + flag
)

ITEM_CLASS_FEATURE_COLS: list[str] = [
    "item_class_n_changed", "item_class_n_buffs",
    "item_class_n_nerfs", "item_class_net_signed",
]
ITEM_CHAMP_FEATURE_COLS: list[str] = [
    "item_champ_n_changed", "item_champ_n_buffs", "item_champ_n_nerfs",
    "item_champ_net_signed_weighted", "item_champ_exposure",
    "item_champ_max_share_changed", "has_build_lag",
]
RUNE_CHAMP_FEATURE_COLS: list[str] = [
    "rune_champ_n_changed", "rune_champ_n_buffs",
    "rune_champ_n_nerfs", "rune_champ_exposure",
]
SYSTEM_FEATURE_COLS: list[str] = ["patch_n_system_changes"]

SLOT_FEATURE_COLS: list[str] = [
    "slot_n_changed", "slot_n_buffs", "slot_n_nerfs",
    "slot_net_signed_pct", "slot_max_rival_share_changed", "has_slot_lag",
]

FEATURE_BLOCKS: dict[str, list[str]] = {
    "statdelta": STAT_DELTA_COLS,
    "counts": ["n_buffs", "n_nerfs", "n_changes_total", "n_abilities_changed"],
    "magnitudes": ["mean_pct_change", "sum_signed_pct", "max_buff_pct",
                   "max_nerf_pct", "n_parseable_changes"],
    "adjust": ["is_adjustment", "buff_ratio"],
    "flags": ["passive_changed", "q_changed", "w_changed", "e_changed",
              "r_changed", "base_stats_changed"],
    "lag": [f"lag_pwds_{DEFAULT_PWDS_WEIGHTS}", "has_lag"],
    "item_class": ITEM_CLASS_FEATURE_COLS,
    "item_champ": ITEM_CHAMP_FEATURE_COLS,
    "rune_champ": RUNE_CHAMP_FEATURE_COLS,
    "system": SYSTEM_FEATURE_COLS,
    "slot": SLOT_FEATURE_COLS,
}

LEAN_FEATURE_COLS: list[str] = [
    "n_buffs", "n_nerfs", "n_changes_total", "n_abilities_changed",
    "is_adjustment", "buff_ratio",
    "passive_changed", "q_changed", "w_changed", "e_changed",
    "r_changed", "base_stats_changed",
    "lag_pwds_presence_heavy", "has_lag",
    "slot_n_changed", "slot_n_buffs", "slot_n_nerfs",
    "slot_net_signed_pct", "slot_max_rival_share_changed", "has_slot_lag",
]

TRAIN_PATCHES: list[str] = (SEASON_PATCHES["2024"]
                            + [f"15.{m:02d}" for m in range(1, 18)])
TEST_PATCHES: list[str] = [f"15.{m:02d}" for m in range(18, 25)]

def _span(major: str, lo: int, hi: int) -> list[str]:
    return [f"{major}.{m:02d}" for m in range(lo, hi + 1)]


LEGACY_FOLDS: list[tuple[str, list[str]]] = [
    ("15.10", _span("15", 11, 13)),
    ("15.13", _span("15", 14, 16)),
    ("15.16", _span("15", 17, 17)),
]

EXTENDED_FOLDS: list[tuple[str, list[str]]] = [
    ("14.14", _span("14", 15, 17)),
    ("14.17", _span("14", 18, 20)),
    ("14.20", _span("14", 21, 23)),
    ("14.23", _span("15", 1, 3)),
    ("15.03", _span("15", 4, 6)),
    ("15.06", _span("15", 7, 9)),
    ("15.09", _span("15", 10, 12)),
    ("15.12", _span("15", 13, 15)),
    ("15.15", _span("15", 16, 17)),
]

GENERALISATION_FOLDS: list[tuple[str, list[str]]] = EXTENDED_FOLDS[:3]

FOLD_SETS: dict[str, list[tuple[str, list[str]]]] = {
    "legacy": LEGACY_FOLDS,
    "extended": EXTENDED_FOLDS,
    "generalisation": GENERALISATION_FOLDS,
}

ABLATION_CONFIGS: dict[str, dict] = {
    "C0": {"seasons": ["2025"], "folds": "legacy", "text": "plain",
           "extra_features": []},
    "C1": {"seasons": ["2024", "2025"], "folds": "extended", "text": "plain",
           "extra_features": []},
    "C2": {"seasons": ["2024", "2025"], "folds": "extended", "text": "plain",
           "extra_features": ITEM_CLASS_FEATURE_COLS + SYSTEM_FEATURE_COLS},
    "C3": {"seasons": ["2024", "2025"], "folds": "extended", "text": "enriched",
           "extra_features": (ITEM_CHAMP_FEATURE_COLS + RUNE_CHAMP_FEATURE_COLS
                              + SYSTEM_FEATURE_COLS)},
    "C4": {"seasons": ["2024", "2025"], "folds": "extended", "text": "plain",
           "extra_features": SLOT_FEATURE_COLS},
    "C6": {"seasons": ["2024", "2025"], "folds": "extended", "text": "plain",
           "extra_features": []},
}


def features_for(config_id: str) -> list[str]:
    """Full feature list for an ablation config."""
    if config_id == "C6":
        return list(LEAN_FEATURE_COLS)
    return PATH_A_FEATURES + ABLATION_CONFIGS[config_id]["extra_features"]


def ensure_dirs() -> None:
    """Create all output/processed directories if missing (idempotent)."""
    for d in (PROCESSED, MAPPINGS, DDRAGON_DIR, PATCH_NOTES_DIR,
              FIGURES, TABLES, PREDICTIONS, BASELINE_DIR, BASELINE_PRE_CONVERGENCE_DIR):
        d.mkdir(parents=True, exist_ok=True)


