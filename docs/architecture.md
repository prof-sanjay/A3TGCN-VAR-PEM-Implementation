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
2. `python selection/select_shared_settings.py <run A> <run B>`: best validation RMSE
   within 20 and within 30 epochs for each learning rate; the current
   setting (0.001, 20 epochs) is kept unless another is >= 2 % better.
3. The chosen run's test set is evaluated once from its checkpoint:
   `python -m implementation.main --evaluate-only <run> [--milestone 20]`.
   This is the E0 reference result.
4. The chosen settings are written into `config.py` and frozen. If they
   differ from 0.001 / 20 epochs, E1 and E3 are rerun with them.

Deviation (2026-10-26, option (a)): the final E0 run (lr 0.001, 20
epochs, test RMSE 29.76) showed no validation improvement after epoch 15,
so run A is not run. Its validation curve is the reference:
`python selection/select_shared_settings.py <run B> --reference-run
results/kaggle/output_E0_final`. If the reference is kept, that run is
the E0 result.

Result (2026-10-27, run B on Kaggle, from the run log; validation only):
reference lr 0.001 / 20 epochs best val RMSE 29.98 (epoch 15); lr 0.0005
best 30.50 within 20 epochs (1.7 % worse) and 29.75 within 30 epochs
(0.8 % better, below the 2 % margin). Decision: KEEP lr 0.001 / 20 epochs
(frozen; already in config.py). E0_final (test RMSE 29.76) is the E0
result; E1 stays valid. Run B's test set is not evaluated. The lower
rate was not more stable (spikes at epochs 5 and 19-20), only slower.

Supporting options in `main.py`: `--max-hours`, `--milestones`,
`--early-stopping` (off by default), `--evaluate-only` (test set
evaluated at most once per run).

## 6. Phase 5: VAR-LiNGAM (E2) and VAR-LiNGAM + PEM (E4)

- `run_local_lingam.py`: DirectLiNGAM (lingam package) on the VAR
  training residuals of each topology unit U_i = {i} + N_in(i) + N_out(i)
  (raw scale, complete rows only). Global B_0 by the home-unit rule:
  row i (the effects into sensor i) comes only from U_i.
  Output: `outputs/lingam/<LINGAM_RUN>/lingam_result.npz`.
- `causal/graph_builders.py`: B_k = (I - B_0) A_k with the same VAR's
  A_k; E2 = sum_k |B_k| + |B_0|; E4 = sum_k |B_k| + P * |B_0| (P from
  the selected PEM run weights only the contemporaneous term).
  B_k can contain two-step links beyond the road neighbours.
- `build_graph.py E2` / `E4` check that VAR, LiNGAM and PEM share the
  training split and sensor set, and that LiNGAM used the same VAR.
- Validated on a synthetic structural VAR with Laplace noise
  (`tests/test_var_lingam_graph.py`): B_0, causal order and B_1 recovered.

## 7. Phase 4: PEM setting for E3 (Stage A, training residuals only)

Sweeps on 60 sampled units (full rows and both halves of the training
rows): raw vs standardised residuals x uniform vs road prior x (nu0, nu1).
`pem_stage_a_report.py` adds, for every setting: split-half directed /
undirected Jaccard, chance-corrected stability, the share of edges from
the lower- to the higher-variance sensor, and the road-link share.

Findings (2026-10-25):
- Raw residuals: 72-74 % of edges point from the lower- to the
  higher-variance sensor in every setting (PEM's ordering follows sensor
  noise levels). Standardised residuals: 46-48 % (close to 0.5).
- The road prior changes stability by at most +0.02.
- Raw Jaccard rises with density even for random graphs; corrected for
  chance, all standardised settings are about equally stable (~0.31).

Selection rule AMENDED after the sweeps (approved by the author; report as
a post-hoc correction): rank by chance-corrected stability, among
uniform-prior settings at threshold 0.5 (the road prior is E7's
ingredient). Shortlist for Stage B, standardised residuals, uniform prior:
  - nu0 = 0.0017, nu1 = 1   (PEM paper defaults; dense, ~375 edges/unit)
  - nu0 = 0.05,   nu1 = 1   (sparse, ~94 edges/unit)
The original rule's shortlist is kept in `outputs/pem/stage_a_report.json`.
Stage B (validation training of both, in the Kaggle batch) makes the
final choice.

## 8. Folder layout (2026-10-26)

```
implementation/
  main.py               train / validate / test one experiment (E0-E7, NOGRAPH)
  run_var.py            VAR on training data            -> outputs/var_train/
  run_local_pem.py      local PEM on VAR residuals      -> outputs/pem/
  run_local_lingam.py   local VAR-LiNGAM                -> outputs/lingam/
  build_graph.py        graph artifacts E1-E4           -> outputs/graphs/
  baseline_ha.py        Historical Average baseline     -> results/HA/
  summarize_runs.py     consolidated metric tables      -> results/summary/
  compare_results.py    comparison tables, tests, figures
  selection/            tuning and selection: PEM sweep + Stage A report,
                        Phase 3 settings choice, LiNGAM row-subset check
  tools/                one-off data checks (missing gaps, sensor order /
                        alignment, GPU check)
  causal/ data/ graph/ models/ training/ experiments/ config/   library code
  tests/                all tests (python -m tests.<name> or python tests/<name>.py)
  notebooks/            Kaggle notebook
  docs/                 this document
  datasets/             LargeST GBA (not in git)
  outputs/              VAR / PEM / LiNGAM results and graphs (not in git)
  results/              kaggle/ (final runs), HA/, reports/ (workbooks,
                        figures), archive/ (earlier decision runs)
```

Scripts in selection/ and tools/ are run from implementation/, e.g.
`python selection/pem_stage_a_report.py`.

### LiNGAM row count (2026-10-26)

Full rows cost ~1,000 CPU hours (~4 days on 10 workers). Row-subset check
on 20 sampled units (`selection/compare_lingam_rows.py`, criteria fixed
beforehand: median relative B_0 difference <= 0.10, median support
Jaccard >= 0.80, sign agreement >= 95 %): 20,000 rows 0.54 / 0.50 / 98 %,
10,000 rows 0.53 / 0.45 / 97 %, so neither passes.

Because both subsets differ from the full rows by the same amount, the
full-row B_0 may itself be unstable. Split-half check (approved by the
author; rule fixed before the result): DirectLiNGAM on rows [0, 42048)
and [42048, 84096) of the same 20 units, compared with each other.
  - halves meet the original criteria  -> use all rows (4-day run);
  - otherwise -> B_0 is intrinsically unstable on this data (report as a
    finding); use the smallest row subset whose agreement with the full
    rows is at least as good as the agreement between the two halves.

## 9. Phase 6: E6 (data-driven neighbourhoods) and E7 (road prior)

Both use E4's formula, sum_k |B_k| + P * |B_0|; only one ingredient changes.

- E7: PEM with the road prior (eta 0.7 for a road link in either
  direction, 0.3 otherwise). VAR and LiNGAM are E4's (road candidates).
  Graph: `build_graph.py E7` (PEM run `PEM_RUN_E7`).
- E6: NO road information anywhere. `build_candidates.py --k 25` chooses,
  for every sensor, the 25 sensors whose 5-minute changes (first
  differences) are most correlated with its own on the TRAINING rows
  (`causal/candidates.py`; K = 25 = the road graph's mean in-degree, so E6
  has as many candidate links as E4). Then
  `run_var.py --candidates corr_k25 --save-residuals`; PEM and LiNGAM read
  the candidates from that VAR run, so all steps use the same links.
  Graph: `build_graph.py E6`.
- `build_graph.py` checks every input: E4 / E7 road candidates, E6
  data-driven candidates everywhere; E4 / E6 uniform prior, E7 road prior;
  PEM and LiNGAM on the same VAR run.
- Cost: E7 needs one extra full PEM run (~1 h); E6 needs its own VAR,
  PEM and LiNGAM runs (LiNGAM as costly as E4's).

E6 candidates (built 2026-10-26, `outputs/candidates/corr_k25.npz`):
51,400 links (road graph 51,686); 45 % of them are also road links;
median |corr| of 5-minute changes 0.17. Hubness: a few sensors are a
top-25 neighbour of hundreds of others (max 1,209; 55 sensors > 100).
Masking network-wide jump events did not remove it (max became 1,546),
so it is the known hubness of k-nearest-neighbour graphs on weak
similarities, not a data artefact. Units would reach 1,210 sensors
(~38,000 CPU h for LiNGAM). Fix: E6 units keep all of a sensor's own
sources and at most the 25 strongest sensors that use it as a source
(`build_causal_units(max_reverse_neighbors=25)`, `candidates.unit_options`).
Unit sizes: E6 mean 28.4 / max 43, road (E4) mean 31.6 / max 91. VAR is
unaffected (it regresses each sensor on its own 25 sources).

Split-half result (2026-10-26, 20 units, all rows of each half):
half 1 vs half 2: median relative B_0 difference 0.95 (max 4.07), support
Jaccard 0.31, sign agreement 86 %, causal-order tau 0.32 -- far below
the criteria and below both subsets' agreement with the full rows
(10,000 rows: 0.53 / 0.45 / 97 % / 0.67). By the rule fixed beforehand:
B_0 is intrinsically unstable on these residuals (a thesis finding; part
of it may be genuine change between the halves, Jan-Jun vs Jun-Nov), and
the smallest subset, 10,000 evenly spaced complete rows per unit, is used
(`LINGAM_MAX_ROWS = 10000`; ~20x faster). Caveat: a subset of the full
data is naturally closer to it than two disjoint halves are to each
other, so this comparison is lenient.

## 10. Phase 7: horizon conditioning (E5)

Forecast horizon h (5-minute steps): `ExperimentConfig.horizon`,
`main.py --horizon h`, notebook `HORIZON`. pre_len stays 1: the target
is the single timestep h steps after the last input (inputs s..s+11,
target s+11+h). Only the 12 input rows and the target row are checked
for validity (`data/windows.py`, `target_offset = h - 1`). Window counts
therefore differ slightly between horizons; all experiments at the same
horizon use identical windows. The A3T-GCN model is unchanged.

E5 graph for horizon h (`causal/graph_builders.horizon_graphs`,
`build_graph.py E5`, one file per horizon):
    Theta_h = Psi_h (I - P * B_0)^{-1}   (structural impulse responses,
                                          causal/horizon.py)
    G_h     = |Theta_h| off-diagonal, the 25 strongest sources per sensor
Psi_h are the VAR's reduced-form impulse responses; P (PEM) weights the
contemporaneous effects as in E4. |Theta_h| is dense, so each sensor
keeps its K = 25 strongest sources (`E5_TOP_K`, the road graph's mean
in-degree). Horizons: 3 / 6 / 12 (15 / 30 / 60 min).

Matched comparison: E4's graph trained at the same horizons
(`--experiment E4 --horizon h`); optionally E0 at the same horizons as
the road-graph reference. Cost: one Kaggle run per experiment and horizon.

`summarize_runs.py` reports every table per experiment AND horizon.

## 11. Full framework first (2026-10-26, author's decision)

To obtain the full-framework result (E5) before the other ablations:
- E5 also gets a 5-minute graph (horizon 1), so E5 at 5 min compares
  directly with the finished E0 run (test RMSE 29.76); horizons 3 / 6 / 12
  follow, with E0 at the same horizons if needed.
- PEM setting without Stage B: the PEM paper's defaults on standardised
  residuals (nu0 = 0.0017, nu1 = 1, uniform prior, threshold 0.5), one of
  the two Phase 4 shortlisted settings. E3 / E4 / E7 use the same PEM run.
- Shared training settings as E0_final (lr 0.001, 20 epochs); if Phase 3
  run B wins, E0 and E5 are rerun with the new settings.
- If E5's error is high, the ablations (E2, E4, E3, ...) locate the cause.

## 12. Phase 10: figures and tables

- `summarize_runs.py`: all_epoch_metrics, best_epoch_summary,
  multi_seed_summary (per experiment and horizon).
- `compare_results.py --runs E0=... E1=... E5=...`: the first run is the
  reference; tables (comparison, deltas, significance: Wilcoxon per sensor
  and Diebold-Mariano, graph comparison), figures 01-09; runs at several
  horizons are compared within each horizon (sub-folders h1/, h3/, ...)
  plus 10_error_by_horizon.
- `plot_graphs.py`: graph figures 11-15 (same 40 sensors on the map,
  in-degree, edge overlap, size, weights) with CSV tables.
- Colours: one fixed slot per experiment from the validated 8-slot
  palette (E0 blue, E1 orange, E2 aqua, E3 yellow, E4 magenta, E5 green,
  E6 violet, E7 red); NOGRAPH / HA neutral grey, dashed. Three slots are
  below 3:1 contrast on white, so every figure has visible labels and a
  CSV of its numbers.

## 13. PEM solver fallback amended (D5, 2026-10-26)

Full PEM run with the paper defaults (standardised, nu0 0.0017, nu1 1):
2,050 units finished (largest 85 sensors, <= 29 min); the 6 largest units
(77-91 sensors) ran ~8 h each without finishing. Cause: after an L-BFGS-B
line-search failure, D5 re-solved with SLSQP, whose cost grows with the
cube of the parameter count (~4,200 for 91 sensors). New order, same
PEM-MAP objective (`causal/pem.py`, tested in `tests/test_pem_fallback.py`):
  1. up to 3 warm L-BFGS-B restarts from the stopping point;
  2. SLSQP only for <= 1,500 parameters (~54 sensors);
  3. larger problems: the L-BFGS-B point is accepted only if the objective
     changed by <= the tolerance over the last restart (it has stopped
     decreasing); otherwise the unit fails visibly.
Each unit file records lbfgs_restarts / slsqp_retries / accepted_stalled.
Only the 6 unfinished units were recomputed; the 2,050 finished units
(2 of them via the old SLSQP retry) were kept.
