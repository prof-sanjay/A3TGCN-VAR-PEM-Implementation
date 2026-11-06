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

## 14. Result: full framework E5 vs base E0 (2026-10-27, seed 42, 5 min)

Test (same 5,183 windows x 2,056 sensors): E0 RMSE 29.76 / MAE 19.34;
E1 28.10 / 18.36; E5 28.17 / 18.47.
- E5 vs E0: -5.4 % RMSE, -4.5 % MAE; better on 62 % of sensors (Wilcoxon
  p ~ 1e-33), Diebold-Mariano p < 0.001. Significant improvement.
- E5 vs E1: tied (+0.2 % RMSE; better on 56 % of sensors, Wilcoxon p ~ 1e-4,
  but Diebold-Mariano p = 0.15; masked MAPE 11.4 % vs 10.5 %).
- Training instability: best epoch 4, then validation RMSE 78-112
  (epochs 6-15), partly recovering to 50.9 by epoch 20. The test score
  comes from a 4-epoch model, so it likely understates E5.
- Checked and rejected: graph-scale amplification. The A3T-GCN operator of
  E5 has spectral radius 1.00 and max row sum 4.2, within the stable
  graphs' range (E1 2.9, E0 5.5). As with the old E3 seed-42 run, the
  collapse is an optimisation failure, not a graph construction error.
- Next: E5 with another seed (7) to test robustness; then horizons.
Figures: results/reports/E5_vs_E0_E1/, results/reports/graphs/.

## 15. E3 graph amended: VAR weights on PEM edges (2026-10-28, author-approved)

- First E3 run (P * D, standardised PEM, seed 42): test RMSE 57.17 / MAE 39.66,
  worse than HA (54.92); validation unstable throughout (spikes to ~140).
  Kept as a reported result (results/kaggle/output_E3_final).
- Cause (diagnosed from the graph files): with the paper settings PEM's
  inclusion probabilities are ~0/1 (85% > 0.99; 87.5% of candidate pairs kept),
  so P * D is an unweighted graph with 75,885 edges (in-degree up to 132).
  After A3T-GCN's normalisation a sensor's own signal is a median 3% of its
  input, versus 11-39% for every graph that trained well (E0 11%, old E3 18%,
  E2/E4 24%, E1 39%, E5 h1 39%).
- Amendment: E3 = W * D, the E1 VAR weights W = sum_k |A_k| kept only on the
  directed PEM-LBN edges D (causal/graph_builders.var_pem_mask_graph, method
  "var_pem_mask"). E3 now differs from E1 by PEM's edge selection alone:
  32,873 of E1's 51,686 edges, 65% of the VAR weight, own-signal share 50%.
  File: E3__lag3_alpha1e+06__lag3_alpha1e+06__nu0_0.0017_..._std.npz.
  The former P * D files are refused as E3 (method check).
- Related: E4 = sum|B_k| + P * |B_0| differs from E2 in only 523 of 101,685
  entries (P ~ 1 on B_0's edges). Decision on running E4 postponed until the
  amended E3 validates the graph construction.

## 16. Result: E2 (VAR-LiNGAM) at 5 min, seed 42 (2026-10-29)

- Test RMSE 32.38 / MAE 21.88, best epoch 14; worse than E0 (29.76) and E1
  (28.10): better than E1 on only 15% of sensors (DM p < 0.001). Validation
  unstable (spikes at epochs 6, 8, 11, 15-19; 239 at epoch 19).
- Adding |B_0| doubles the graph (101,685 edges, in-degree mean 49, max 160,
  vs E1's 51,686) and lowers the own-signal share from 39% to 24%; B_0 itself
  is unstable across data halves (split-half Jaccard 0.31, section 6).
  Across runs so far, denser graphs train worse (E1 / E5 ~50k edges best).
- E5 uses the same B_0 but keeps only the top-25 sources per horizon and
  beats E2 on 85% of sensors.
- Report: results/reports/E2_seed42. First E3 (P * D) output renamed to
  results/kaggle/output_E3_PxD_final. Next: amended E3 (section 15).

## 17. Result: amended E3 (VAR weights on PEM edges), 5 min, seed 42 (2026-10-29)

- Test RMSE 28.30 / MAE 19.62 / masked MAPE 15.9, best epoch 7. Healthy until
  epoch 8 (validation 28.5 at epoch 7, close to E1's best 28.37), then
  validation collapsed (59-115 from epoch 9) and never recovered.
- vs E1 (28.10 / 18.36): E3 better on only 16% of sensors (DM p 1e-8).
  vs E0 (29.76 / 19.34): tie (DM p 0.17). E5 beats E3 on 80% of sensors.
- Reading: the amended graph is sound (the P * D failure is fixed), but
  restricting the VAR graph to PEM's edges (32,873 of 51,686) does not improve
  on E1 at 5 min. Caveat: one seed, and the run collapsed after epoch 8.
- Training collapses at the fixed lr 0.001 now occur in E2, E3 and E5 (seed 42).
  Correction (2026-11-03): E0 (epochs 2-6) and E1 (epochs 5-8, 16-19) also had
  validation spikes (history.csv); only NOGRAPH trained smoothly. Report as a
  limitation of the fixed protocol.
- Next: 60-min runs (E5, E1, E0 at horizon 12, seed 42). E4 (differs from E2
  in 1.6% of entries) recommended to be dropped; author's decision pending.

## 18. Result: E5 at 60 min (horizon 12), seed 42 (2026-11-02)

- Test RMSE 48.97 / MAE 32.88 / masked MAPE 20.0; best epoch 20 (last).
  Validation spiked at epochs 6-9 and recovered after the lr was halved
  (epoch 10). Windows: 69,504 / 10,449 / 5,172 (target 12 steps ahead).
- HA scored on the same targets (results/HA_h12): RMSE 54.89 / MAE 33.21.
  E5 has 11% lower RMSE but the same MAE (DM p 0.67); HA has lower MAE on 69%
  of sensors and is better at night (00-06 h: HA 17.5 vs E5 21.2), E5 is
  better 10-24 h. A one-hour input window cannot see the weekly profile HA
  uses; this is a property of the unchanged A3T-GCN input (seq_len 12).
- Notebook fix: cell 2b now builds the config with horizon=HORIZON (it printed
  the h1 file name; the run itself used the h12 graph).
- E1 / E0 at horizon 12 pending; they decide whether E5's horizon graphs help.

## 19. Result: E1 / E0 at 60 min and E0 seed 7 (2026-11-02)

Horizon 12 (60 min), seed 42, 5,172 test timesteps:

| Run | RMSE | MAE | Best epoch | vs E1 (sensors better) | MAE vs HA (DM p) |
|-----|------|-----|-----------|------------------------|------------------|
| E1  | 47.41 | 31.32 | 19 | -     | 0.016 (E1 better) |
| E5  | 48.97 | 32.88 | 20 | 34%   | 0.67 (tie) |
| E0  | 49.87 | 33.19 | 15 | 30%   | 0.98 (tie) |
| HA  | 54.89 | 33.21 | -  | -     | - |

- E1 beats E0 and E5 (DM p < 0.001 for both). E1 is the only model whose MAE
  beats HA at 60 min. E5's horizon-specific graphs do not help at 60 min either.
- E1's best epoch is 19 of 20 and the lr was never halved: it may be
  under-trained at 20 epochs (same budget for all runs, so the comparison holds).
- Validation spikes: E0 at epochs 3, 12, 18; E1 at epochs 5-7 (recovered).

Horizon 1, seed 7: E0 = 30.13 / 19.33 (best epoch 16; seed 42: 29.76 / 19.34).
Two-seed means (RMSE): E0 29.94, E1 27.41, E5 27.78. At seed 7, E1 beats E0
on 93% of sensors and E5 on 78% (DM p < 0.001).

Reading: across both seeds at 5 min and at 60 min, the VAR graph (E1) is the
best graph; the causal refinements (E2-E5) do not improve on it.
Reports: results/reports/h12_E0_E1_E5, results/reports/seed7_E0_E1_E5.

## 20. Result: NOGRAPH (5 min) and E1 / E5 at 30 min, seed 42 (2026-11-03)

NOGRAPH (A = 0, so the A3T-GCN operator is the identity: every sensor sees only
its own history, weights shared), horizon 1: test RMSE 27.58 / MAE 16.87, best
epoch 20 (last; validation noisy, 26.6-38.4, lr halved twice).

- Beats every seed-42 graph run at 5 min: E0 29.76, E1 28.10, E5 28.17
  (DM p < 0.001 each; better than E1 on 82% of sensors, E0 on 88%).
- Largest gain in daytime and at high-flow sensors (MAE 21.8 vs E1 24.4);
  E1 is better only at night (00-06 h).
- Consistent with the own-signal-share pattern (section 16): NOGRAPH is the
  limit (100% own signal). At 5 min the next value is mostly the sensor's own
  recent flow; graph mixing adds noise.
- Caveats: one seed; best epoch = last epoch with a noisy validation curve.

Horizon 6 (30 min): E1 37.47 / 23.58 (best epoch 20), E5 39.98 / 25.82 (best
epoch 15). E1 better on 86% of sensors (DM p < 0.001). Both collapsed several
times (E5 validation 108-179 at epochs 5-8, 10, 17-19). HA at 30 min
(results/HA_h6): 54.91 / 33.23; both models beat it (DM p < 1e-19).

Open question: does any graph beat NOGRAPH at 30 / 60 min? (NOGRAPH h6 / h12.)
Reports: results/reports/NOGRAPH_seed42, results/reports/h6_E1_E5.
compare_results.py fix: runs without a graph file (NOGRAPH) are skipped in the
graph table instead of crashing.


## 21. E8 / E8V: horizon-conditioned VAR-PEM without LiNGAM (2026-11-05)

Status: implemented and unit-tested; graphs and forecasting runs pending (Kaggle).
No result is claimed here: whether removing LiNGAM helps is what the runs decide.

**Why.** E5 builds Theta_h = Psi_h (I - P * B0)^-1. Tuning checks on validation
data (tune_e5_proxy.py, Kaggle, 2026-11-04) and the A3T-GCN results (E2 / E4 / E7
worse than E1; E5 not better than E1) suggest that B0 adds error, while P changes
almost nothing (on B0's non-zero entries P > 0.99 for 98.6%, so P * B0 removes
0.31% of B0's weight). E8 therefore removes LiNGAM and B0 completely and keeps
PEM.

**Architecture (E8).**

    traffic + road adjacency -> topology-aware VAR (lag 3, ridge 1e6) -> A_k
    VAR residuals (standardised) -> PEM (nu0 0.0017, nu1 1, tau 0.01, eta 0.5) -> P
    A_k -> impulse responses Psi_h (h = 1, 3, 6, 12)
    S_h = P (x) |Psi_h|, diagonal 0
    G_h = top-25 sources per target of S_h        (G[target, source])
    -> existing normalisation and transposition -> existing A3T-GCN (unchanged)

**Mathematics.**

    Psi_0 = I,   Psi_s = sum_{k=1}^{min(s,3)} A_k Psi_{s-k}
    S_h[i, j] = P[i, j] * |Psi_h[i, j]|  (i != j),   S_h[i, i] = 0
    G_h[i, j] = S_h[i, j] if j is among the 25 largest of S_h[i, .], else 0

Psi_h[target, source] follows A_k[target, source], so G_h has the same
[target, source] orientation as every other graph; main.py applies the single
transposition for all experiments. P is symmetric.

**E8V (ablation).** S_h = |Psi_h|: no P and no B0. It isolates the contribution
of PEM: E8 versus E8V differ only by P. With P = 1 everywhere E8 equals E8V
(tested), and E8V equals the existing E5 builder with B0 = 0 and P = 1 (tested
against causal/graph_builders.horizon_graphs, which is unchanged).

**Differences from E5.**

| | E5 | E8 |
|---|---|---|
| DirectLiNGAM, B0 | used | not used (no import, no file) |
| effect matrix | abs(Psi_h (I - P * B0)^-1) | P * abs(Psi_h) |
| role of P | scales B0 | weights (and masks) the VAR responses |
| top-K rule, K = 25, orientation, A3T-GCN | same | same |

E8 is a new formulation; it is not an algebraic variant of E5.

**PEM's role, stated precisely.** The PEM run (E4 / E5 setting) used only VAR
residuals, so its P does not depend on LiNGAM and is reused, not re-estimated.
P is non-zero only for pairs that share a local unit U_i = {i} + N_in(i) +
N_out(i) (about 3% of the matrix) and is close to 0 / 1 where non-zero. Hence
P * |Psi_h| mostly acts as a mask: it keeps the sources inside the PEM-supported
(about two-hop road) neighbourhood and leaves their weights nearly unchanged,
whereas |Psi_h| (E8V) is dense and can pick sources beyond it, especially at
long horizons. For h = 1, Psi_1 = A_1, which is non-zero only on road neighbours.
E8 versus E8V measures what this mask does.

**Configuration (frozen, identical to E1 / E5 runs).** LargeST GBA 2021, 2,056
sensors, split rows [0, 84096) / [84096, 94608) / [94608, 105120), input 12,
pre_len 1, horizons 1, 3, 6, 12; Adam lr 0.001, batch 8, 20 epochs, clip 5, GRU
100, lambda 0.0015, LR halved after 5 epochs without validation gain (min 1e-5),
seed 42, best-validation checkpoint, test evaluated once. All graph estimation
uses training rows only.

**Implementation.**

| File | Purpose |
|---|---|
| causal/var_pem_horizon.py | Psi_h, S_h, top-K graphs, diagnostics, LiNGAM access report (new) |
| build_e8_graphs.py | builds E8 and E8V graphs and writes diagnostics (new) |
| notebooks/e8-graphs.ipynb | runs the build on Kaggle CPU (new) |
| tests/test_var_pem_horizon.py | 12 tests (new) |
| experiments/registry.py, config/config.py | E8 and E8V entries (additive) |
| notebooks/ablation-study.ipynb | E8 / E8V graph names (additive) |

Experiment names: E8 (graph_method var_pem_horizon) and E8V (var_horizon).
Graph files: E8__h{h}__<VAR_RUN>__<PEM_RUN>.npz and E8V__h{h}__<VAR_RUN>.npz
(compressed artifacts, same fields as E5 plus horizon and top_k); the rest of
the build (VAR coefficients, P, Psi_h and S_h per horizon, configuration,
diagnostics) is in outputs/e8/.

**Validation checks.** (1) Psi_h equals the top-left block of the p-th power of
the VAR companion matrix (independent calculation), Psi_1 = A_1 and
Psi_2 = A_1 A_1 + A_2. (2) S_h is element-wise, absolute, non-negative, zero
diagonal; P outside [0, 1] or of the wrong shape is refused. (3) Each target
keeps at most 25 sources, all inside P's support, rows are targets, weights equal
the 25 largest S_h entries. (4) E8V equals the E5 builder with B0 = 0, P = 1;
E8 with P = 1 equals E8V. (5) The artifacts load through main.load_graph with the
right method and horizon; a wrong horizon, method or training split is refused.
(6) importing the E8 code imports no lingam module; the build writes an access
report (no lingam module imported, no lingam / B0 file opened) and stops if it is
not clean. The build also stops if Psi_1 differs from A_1, P is not symmetric,
a diagonal is non-zero, an in-degree exceeds 25, an E8 edge lies outside P's
support, or an E8 weight exceeds the E8V weight.

**Run order (Kaggle).** (1) notebooks/e8-graphs.ipynb (CPU, minutes): builds the
eight graph files and prints VAR / PEM / graph diagnostics. (2) Upload the
graphs to the artifacts dataset. (3) ablation-study.ipynb with EXPERIMENT E8 or
E8V and HORIZON 1, 3, 6, 12; compare with E1, E5 (and E5 seed repeats) with
compare_results.py. Runs of one configuration can differ by several RMSE points
(training instability; a repeat test is pending), so a conclusion needs repeated
runs; seed 42 is the protocol seed, repeats use 7 and 123.

### 21.x E8 prior study (added 2026-11-05)

The E8 graphs of section 21 use PEM's probabilities from the E4/E5 run (uniform prior 0.5, nu0 = 0.0017, nu1 = 1).
At those settings P is almost exactly 0 or 1 (1.2 % of the estimated pairs lie between 0.05 and 0.95 on a 60-unit slice),
so a prior can hardly change it: the E7 road prior moved 0.02 % of the weight. The prior study therefore compares priors
under one fixed, looser PEM setting (nu0 = 0.01, nu1 = 0.2; 40 % of P graded on the same slice), with everything else
unchanged (post-hoc amendment, approved by the user on 2026-11-05).

| Item | Definition |
|---|---|
| Priors | `uniform` (eta = 0.5), `pearson`, `spearman`, `partial` correlation of the VAR **training** residuals (pairwise-complete, no filling; partial uses a shrunk correlation matrix, lambda = 0.05) |
| Mapping to eta | percentile of abs(r) among each sensor's candidate pairs (pairs sharing a causal unit) mapped to [0.05, 0.95]; eta = max over the two directions; the same rule for every correlation |
| Graphs | E8 with the new P: `S_h = P (x) abs(Psi_h)`, top-25, h = 1, 3, 6, 12; files `E8__h{h}__<VAR_RUN>__<prior>_nu0_0.01_nu1_0.2.npz` |
| Reference | E8V (no PEM) is unchanged: it does not depend on P |
| Code | `causal/corr_prior.py`, `build_corr_prior.py`, `tools/prior_gate.py`, `run_local_pem.py --prior-file`, `build_e8_graphs.py --variant`, `notebooks/e8-prior-study.ipynb`, `ablation-study.ipynb` (`GRAPH_VARIANT`) |

Validity checks done before any training (training residuals, candidate pairs): split-half reliability of abs(r) 0.886
(Pearson), 0.897 (Spearman), 0.468 (partial); Pearson and Spearman agree at 0.94 (top-25 overlap 0.83). A correlation prior
is useful only if it changes P moderately while the data still decides most of it; the gate on the 60-unit slice is in
`tools/prior_gate.py`. Whether a prior improves forecasting is decided only by the A3T-GCN runs (E8 variants against E8V).

### 21.y E8C: coefficient-space refinement (added 2026-11-06; SUPERSEDED by 21.z the same day, kept for the record, not to be trained)

A proposed, PEM-inspired method (not the original PEM). It uses no VAR residuals, no road units and no LiNGAM.

- **Candidates M**: the non-zero support of the topology-aware VAR (A_k[target, source]; 51,686 edges, about 25 per target). Direction comes only from VAR.
- **Prior**: Pearson correlation of the original training flow (pairwise-complete), by default after subtracting each sensor's time-of-day profile (raw |r| has median 0.84 on the candidates, so it hardly discriminates). eta = 0.05 + 0.90 |R|, symmetric. Association, not causation.
- **Estimator** (`causal/var_coefficient_pem.py`): u = ||(A1,A2,A3)[i,j]||2; w_hat = argmin_{0<=w<=u} q/2 (w-u)^2 - lam log[(1-eta) f0(w) + eta f1(w)], f_q Laplace(nu_q). Separable per edge, solved by a grid plus parabolic refinement. All lags of an edge are scaled by w_hat/u. Posterior slab probability P = eta f1 / (eta f1 + (1-eta) f0); logit P = logit(eta) + log(nu0/nu1) + w_hat (1/nu0 - 1/nu1).
- **Scale**: s = median candidate u (training only); nu0 = s, nu1 = 5s, lam = 0.5 s nu0, q = 1 (ridge standard errors are not stored; only lam/q matters). Chosen from graph statistics on the training VAR (92% of P graded), before any forecasting result; the sensitivity grid is a, b, lam in {0.5,1}, {5,20}, {0.5,1,2}.
- **Graph**: S_h = M P |Psi_h| (Psi from the original A_k; `--psi-from W` is an ablation), diagonal 0, top-25 per target, G[target, source]. Files `E8__h{h}__<VAR_RUN>__<variant>.npz`, trained with `ablation-study.ipynb` (EXPERIMENT = "E8", GRAPH_VARIANT = variant).
- **Variants / controls**: `detrended` (proposed), `raw` (correlation of undetrended flow), `uniform` (eta = 0.5), `shuffle` (detrended eta shuffled among candidates, seed 42), `none` (P = 1: the mask alone), `psiW`. E8V is the no-mask, no-PEM control.
- **Code**: `build_e8c_graphs.py`, `causal/var_coefficient_pem.py`, `causal/flow_correlation.py`, `tests/test_var_coefficient_pem.py` (7 tests, incl. a synthetic true-edge test).
- **Gate (training data only)**: versus the uniform prior the detrended prior changes 6% of edges and 18-26% of edge weight (h = 1..12); the shuffled prior changes 10% and 24-36%. P is a function of |w_hat| and eta only, so part of its variation is coefficient magnitude, which |Psi_h| also carries.
- **Limitations**: nothing here shows better forecasting until the A3T-GCN runs; the mask drops the indirect (multi-step) pairs that E8V keeps; the VAR companion matrix has an eigenvalue of 1.05, so |Psi_h| grows with h; P is a plug-in slab probability, not the probability that an edge is real.

### 21.z E8 with a traffic-correlation prior (replaces E8C, 2026-11-06)

The original PEM is used (precision matrix, spike-and-slab prior), and the two sources are kept separate until the end:

- **VAR** gives the coefficients A_k (direction and size, through Psi_h) and the residuals (used only to estimate each unit's variance).
- **Prior**: Pearson correlation of the ORIGINAL detrended training flow (time-of-day mean removed; pairwise-complete), eta_jk = 0.05 + 0.90 |r_jk| for the pairs sharing a road causal unit, 0.5 elsewhere, symmetric. Association, not causation. Built by `build_corr_prior.py --source flow` (file `outputs/priors/corr_flow.npz`); control `--source flow --shuffle-seed 42` (`corr_flow_shuffle42.npz`, eta shuffled over the candidate pairs, distribution and symmetry kept).
- **PEM**: per causal unit (road), `run_local_pem.py --prior-file`, nu0 = 0.01, nu1 = 0.2, tau = 0.01, L-BFGS-B (nu change is post-hoc: the original 0.0017 / 1 saturates P, so no prior can matter). P = max over the units containing a pair (symmetric).
- **Combination**: S_h = P (x) |Psi_h|, diagonal 0, top-25 per target, G[target, source] (`build_e8_graphs.py --variant flow_nu0_0.01_nu1_0.2`). P weights edges only; direction and size come from VAR. No LiNGAM, no B0.
- **Run**: `notebooks/e8-prior-study.ipynb` with PRIOR = "flow" (then "uniform" and "flow_shuffle" as controls); needs the GBA h5 for the flow correlation. Train with `ablation-study.ipynb` (EXPERIMENT = "E8", GRAPH_VARIANT = variant). Controls: uniform prior, shuffled prior, E8V.
- **Figure**: `presentation/figures/e8_flowprior_pipeline.png`.
- **Limitations**: P is undirected; the correlation of the detrended flow partly reflects spatial proximity; no forecasting claim before the A3T-GCN runs (at least 3 seeds for finalists); the VAR companion matrix has an eigenvalue of 1.05.
