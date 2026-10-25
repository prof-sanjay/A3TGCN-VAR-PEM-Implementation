# Architecture and experiment design

Ablation study of graph construction for A3T-GCN traffic forecasting on
LargeST-GBA 2021 (5-minute flow, 105,120 x 2,352; 2,056 sensors after the
near-copy filter). Only the graph (and, for E5, the horizon) changes between
experiments; data, split, preprocessing, model and training are shared.

## 1. Pipeline

```
datasets/LargeST GBA/  gba_his_raw_2021.h5, gba_rn_adj.npy, gba_meta.csv
        |
data/loader.py         float32 load, NaNs preserved
data/splitting.py      chronological 80 / 10 / 10
data/sensor_filter.py  near-copy sensors removed (decided on TRAIN rows)
data/preprocessing.py  max scaler fitted on observed TRAIN values
data/imputation.py     gap-band filling of model INPUTS only
                       (look-ahead only inside TRAIN; targets must be observed)
data/windows.py        valid window starts, batches by index
        |
graph construction (offline, TRAIN rows only)
  run_var.py           topology-aware ridge VAR      -> outputs/var_train/<run>/
  run_local_pem.py     local PEM on VAR residuals    -> outputs/pem/<run>/
  build_graph.py       graph artifact G[target, source] -> outputs/graphs/
        |
main.py                load graph -> A3T-GCN -> train / validate -> test once
  models/              A3T-GCN (original authors' code; unchanged)
  graph/normalization  D^-1/2 (A+I)^T D^-1/2 (original code)
  training/            trainer (Adam, optional clipping), LR scheduler, metrics
        |
results/<run>/         results.json, history.csv, test_predictions.npz, ...
compare_results.py     tables, significance tests, figures
```

Settings live in `config/config.py` (shared training settings in
`ExperimentConfig`; VAR / PEM settings and run names as module constants).

## 2. Conventions

- Graphs are stored as `G[target, source]` (`!= 0` means source -> target),
  the convention of VAR coefficients and PEM-LBN. `to_model_orientation()`
  transposes once before A3T-GCN, whose normalization transposes again, so
  every node aggregates from its sources. Applied identically to all graphs.
- Self-loops: removed from causal graphs; A3T-GCN adds `I` itself.
- Graph artifacts carry `train_end` and the sensor list; `main.load_graph()`
  refuses an artifact built on another split or sensor set.
- The test set is evaluated once, at the best-validation checkpoint.

## 3. Experiments (decisions of 2026-10-25)

Numbering changed: the former E2 (VAR + PEM) is now **E3**.

| Exp | Graph | Candidate links | PEM prior | Horizon |
|-----|-------|-----------------|-----------|---------|
| E0 | road adjacency | - | - | 5 min |
| E1 | VAR: sum_k abs(A_k) | road neighbours | - | 5 min |
| E2 | VAR-LiNGAM: sum_k abs(B_k) + abs(B_0) | road neighbours | - | 5 min |
| E3 | VAR + PEM: P * D | road neighbours | uniform | 5 min |
| E4 | VAR-LiNGAM + PEM: sum_k abs(B_k) + P * abs(B_0) | road neighbours | uniform | 5 min |
| E5 | E4 built per horizon from structural impulse responses | road neighbours | uniform | 15 / 30 / 60 min |
| E6 | as E4 | data-driven: most-correlated TRAIN sensors, no road information | uniform | 5 min |
| E7 | as E4 | road neighbours | road prior (0.7 / 0.3) | 5 min |

with B_k = (I - B_0) A_k (VAR-LiNGAM, Hyvarinen et al. 2010).

Baselines (not graph-construction ablations):

| Name | Method |
|------|--------|
| NOGRAPH | A3T-GCN with an all-zero graph: the operator becomes I, so each sensor sees only its own history (GRU + attention, shared weights). Same pipeline and training as E0. |
| HA | `baseline_ha.py`: mean of observed training values per (weekday, 5-minute slot, sensor); scored on the exact test targets of a reference run. |

E4 / E6 / E7 form a topology ladder: no road information (E6), roads choose
candidate links (E4), roads choose candidates and the prior (E7).

Decisions behind this table:

1. "Global" discovery over all 2,056 sensors at once (DirectLiNGAM or PEM on
   2,056 variables) is computationally infeasible; E6 is therefore named
   "data-driven neighbourhoods", not full global discovery.
2. Horizon conditioning = horizon-specific graphs (`causal/horizon.py`,
   `causal/horizon_graph.py`), one model per horizon; the A3T-GCN
   architecture is not changed. The matched comparison is E4's graph at the
   same horizons.
3. Additional baselines: Historical Average and a no-graph A3T-GCN (GRU with attention; NOGRAPH).
   STGCN / DCRNN / Graph WaveNet / PDFormer are out of scope; published
   LargeST numbers are not comparable (other year, split and horizon).
4. Seeds: 42 for the controlled ablation first; 7 and 123 as compute allows.

## 4. Status of components

| Component | State |
|-----------|-------|
| Data, split, filter, gap filling, windows | done, tested (`tests/test_windows.py`, `tests/test_imputation.py`) |
| A3T-GCN, trainer, LR scheduler | done (original model) |
| VAR (E1) | done |
| PEM, local PEM, road prior, standardisation (E3) | done, tested |
| VAR-LiNGAM | partial: `causal/var_lingam.py` returns B_0 only; B_k and local-unit execution missing |
| Horizon graphs | modules exist; multi-horizon training / evaluation missing |
| Data-driven neighbourhoods (E6) | missing |
| Per-epoch metrics file, run summary, consolidated tables | done (Phase 1; `tests/test_run_logger.py`, `summarize_runs.py`) |
| Experiment registry, graph checks, NOGRAPH / HA baselines | done (Phase 2; `experiments/registry.py`, `graph/adjacency.py`, `tests/test_registry_graphs.py`) |

## 5. Phase 3: shared training settings (validation only)

One E0 epoch takes ~17 min on a Kaggle T4, so 40 epochs do not fit in a
12-hour session (a session that exceeds the limit loses its outputs).
Protocol, fixed before any result:

1. Kaggle, E0, selection runs (test set NOT evaluated), `MAX_HOURS = 11`:
   - run A: 30 epochs, learning rate 0.001, milestone 20 (`E0_30ep_sel`)
   - run B: 30 epochs, learning rate 0.0005, milestone 20 (`E0_lr0.0005_30ep_sel`)
2. `python select_shared_settings.py <run A> <run B>`: best validation RMSE
   within 20 and within 30 epochs for each learning rate; the current
   setting (0.001, 20 epochs) is kept unless another is >= 2 % better.
3. The chosen run's test set is evaluated once from its checkpoint:
   `python -m implementation.main --evaluate-only <run> [--milestone 20]`.
   This is the E0 reference result.
4. The chosen settings are written into `config.py` and frozen. If they
   differ from 0.001 / 20 epochs, E1 and E3 are rerun with them.

Supporting options in `main.py`: `--max-hours`, `--milestones`,
`--early-stopping` (off by default), `--evaluate-only` (test set
evaluated at most once per run).
