# Balance Patches and Professional Champion Dominance in League of Legends

Code, data and results for the BSc Honours project *Semantic Patch Encoding and Historical-Information Controls* (COMS4057A, Wits).
**Author:** Yoon Bae Park · **Supervisors:** Branden Ingram, Pravesh Ranchod

Predicts a presence-weighted dominance score (PWDS) for 4,059 champion–patch rows over 47 pro patches (2024–2025), always against a persistence baseline (previous patch's score). The pipeline runs twice: with a rule-based **semantic** patch encoding and with the original **historical** parser.

```
PWDS = 0.40*pick_rate + 0.40*ban_rate + 0.20*(win_rate - 0.50)   (min. 5 picks)
```

## Layout

| Path | Contents |
|---|---|
| `src/data`, `src/features` | data collection/alignment; PWDS, stat-delta, patch-note, item/rune, draft-slot, BERT features |
| `src/models`, `src/evaluation` | persistence, ridge, XGBoost, BERT; metrics, paired bootstrap + Holm, rolling origin |
| `src/exploratory` | semantic encoder, extra features, block targets, rerun driver `semv2.py` |
| `data/` | patch notes, mappings, processed features, cached [CLS] embeddings, build tables |
| `results/semantic`, `results/historical` | `predictions/`, `tables/`, `block_*`, `features/` per encoding |

**Code keys:** paper C5 = `C6`; CL = `C1L`; C1X–C4X = `C1X`–`C4X`; prefixes `D-` (difference target), `ROLL-` (rolling origin), `ROLLB-…-B<B>` (block target).

## Main results (semantic encoding)

| Comparison | Result |
|---|---|
| Direction accuracy vs 134 hand labels | semantic 95.5%, historical 56.7% |
| Fixed test split (335 valid-lag rows) | no Holm-significant gain over persistence |
| Rolling origin, XGBoost C1 / CL vs persistence | −5.9% / −7.3% MAE |
| Patch features on top of CL (C1X–C4X) | further 2.1–3.0% of persistence MAE (uncorrected) |

## Setup

Python 3.12; `torch==2.3.1` CPU (CUDA 12.1 only for fine-tuned BERT), `numpy<2`, `transformers<5`.

```bash
py -3.12 -m venv .venv && .venv\Scripts\Activate.ps1
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Not included: Oracle's Elixir 2024/25 CSVs (download to `data/raw/oracle/`), Data Dragon JSONs (`src.data.fetch_ddragon`, `src.exploratory.fetch_ddragon_items_runes`), Leaguepedia player rows (needs a bot login in `secrets/`; aggregated table is included). No model weights are saved; scripts refit everything.

## Reproduce

Run as modules from the repo root.

```bash
# 1. Dataset
python -m src.features.pwds
python -m src.features.stat_deltas
python -m src.features.patch_note_features
python -m src.features.item_features
python -m src.features.slot_features
python -m src.data.align
python -m src.features.text_features            # embeddings already cached

# 2. Text models (fixed split, shared by both encodings)
python -m src.models.run_ablation --all
python -m src.models.bert_control  --config C1   # also C0, C3
python -m src.models.bert_finetune --config C1   # also C0, C3; GPU recommended

# 3. Semantic / historical runs (setup makes a sibling working copy)
python -m src.exploratory.semv2 setup [--encoding historical]
cd ../<repo>-semv2                               # or ../<repo>-semv2-hist
python -m src.exploratory.semv2 all              # then: extras_c2, extras_cfg, block_configs
```

Working-copy outputs (`outputs/`, `semv2/`) correspond to `results/semantic/` and `results/historical/`.

## Attribution

Build data from Leaguepedia (CC BY-SA 3.0); match data from Oracle's Elixir; champion/item/rune data from Riot's Data Dragon; patch notes from leagueoflegends.com. League of Legends is a trademark of Riot Games, Inc.; this project is not endorsed by Riot Games.
