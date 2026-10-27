# Project brief: Horizon-Conditioned Probabilistic VAR-LiNGAM Causal Graphs for A3T-GCN Traffic Forecasting

*State as of 27 Oct 2026. Use this as the starting context ("prompt") for any assistant or collaborator who continues the work. Everything below is factual for this repository; numbers are from actual runs.*

---

## 1. Your role and the ground rules

You are continuing an M.Tech research project: an ablation study of graph construction for traffic forecasting. The author decides methodology; you implement, run, check and report. Rules that have held throughout:

1. **Do not change the forecasting model.** A3T-GCN is the original authors' code (`models/`, `graph/normalization.py`); only the graph given to it (and, for E5, the forecast horizon) changes between experiments.
2. **No leakage.** Everything data-dependent (scaler, sensor filter, gap-filling models, VAR, PEM, LiNGAM, candidate links) is fitted on TRAINING rows only. Validation is for selection; the **test set is evaluated exactly once per run**, at the best-validation checkpoint (`main.py` enforces this; `--evaluate-only` refuses a second evaluation).
3. **Selection rules are fixed before results exist**; any later change is an explicit, documented amendment approved by the author (two so far, see section 8).
4. **Report honestly**: failed runs, instabilities and rejected hypotheses are recorded, never hidden.
5. Work incrementally, test each change (`tests/`), and record decisions in `docs/architecture.md`.

---

## 2. The research question and the framework

**Question:** does a causal graph learned from traffic data, specifically a horizon-conditioned probabilistic VAR-LiNGAM graph, improve A3T-GCN over the physical road graph, and which ingredient is responsible?

**The full framework (E5)** builds, from training data only:

1. **Topology-aware VAR** (lag 3, ridge alpha 1e6): each sensor regressed on the last 3 steps of its road neighbours: x(t) = sum_k A_k x(t-k) + e(t).
2. **VAR-LiNGAM**: DirectLiNGAM on the VAR residuals e(t) gives the contemporaneous structural matrix B_0; the lagged structural effects are B_k = (I - B_0) A_k (Hyvarinen et al. 2010).
3. **PEM (Probabilistic Edge Modulation)**: a spike-and-slab precision-matrix MAP estimator with PEM-LBN ordering (Hwang, Lee, Oh, Park, UAI 2026; `Papers/PEM.pdf`), giving an edge inclusion probability P that weights the contemporaneous effects: B_0 -> P * B_0.
4. **Horizon conditioning**: structural impulse responses Theta_h = Psi_h (I - P*B_0)^-1 (Psi_h = the VAR's reduced-form impulse responses). Graph for horizon h = |Theta_h| without the diagonal, keeping each sensor's 25 strongest sources. One graph and one model per horizon (h = 1, 3, 6, 12 steps = 5, 15, 30, 60 min).

The graph is given to the unchanged A3T-GCN.

---

## 3. Data and shared setup

| Item | Value |
|---|---|
| Dataset | LargeST GBA 2021: `gba_his_raw_2021.h5` (105,120 x 2,352, 5-minute flow), `gba_rn_adj.npy`, `gba_meta.csv` (do not use PeMS-BAY) |
| Sensors | 2,056 after removing near-copies (>= 90 % identical values or corr >= 0.999 on training rows; cached `outputs/sensor_filter/sensor_filter_21d1a83e4d8775b6.json`) |
| Split (rows) | chronological 80/10/10: train [0, 84096), val [84096, 94608), test [94608, 105120) |
| Scaling | max of observed training values (999) |
| Missing values | gap-band filling of model INPUTS only (<= 30 min interpolation; 30 min - 2 h interpolation or neighbour regression; > 2 h neighbour regression; look-ahead only inside the training split). Targets must be originally observed. VAR / PEM / LiNGAM use observed values only. |
| Windows | seq_len 12 inputs, pre_len 1 target; horizon h puts the target h steps after the last input. Valid windows at h = 1: train 69,580, val 10,460, test 5,183 |
| Model | A3T-GCN (T-GCN cell + temporal attention), GRU 100, lambda 0.0015 |
| Training (frozen by Phase 3) | Adam, lr 0.001, gradient clipping 5, batch 8, 20 epochs, LR halved after 5 epochs without validation improvement (min 1e-5), seed 42; best-validation-RMSE checkpoint |
| Metrics | RMSE, MAE, masked MAPE (targets >= 10), R2, Accuracy (fraction within 10 %, project-specific), Accuracy_TGCN (1 - ||Y-Yhat||_F / ||Y||_F, the papers' definition); vehicles per 5 min, pooled over windows and sensors |
| Significance | Wilcoxon over per-sensor MAE; Diebold-Mariano over time (Newey-West) |
| Compute | Kaggle T4 for training (~17 min / epoch, ~5.8 h / run, 30 GPU-h / week); laptop (16 threads, 8 GB RAM, 4 GB GPU) for graph construction |

Graph convention everywhere: `G[target, source]` (row = target), transposed once before A3T-GCN so every sensor aggregates from its sources. Causal graphs have no self-loops; A3T-GCN adds the identity.

---

## 4. The experiments

| Exp | Graph | Candidate links | PEM prior | Horizon | Question it answers |
|---|---|---|---|---|---|
| **E0** (base) | road network | - | - | 5 min | reference |
| **E1** | VAR: sum_k abs(A_k) | road | - | 5 min | does VAR help? |
| **E2** | VAR-LiNGAM: sum_k abs(B_k) + abs(B_0) | road | - | 5 min | does LiNGAM add to VAR? |
| **E3** | VAR + PEM: P * D (PEM-LBN direction D) *(was called E2 before 25 Oct)* | road | uniform | 5 min | does PEM add to VAR? |
| **E4** | VAR-LiNGAM + PEM: sum_k abs(B_k) + P * abs(B_0) | road | uniform | 5 min | does PEM add to VAR-LiNGAM? |
| **E5** (full) | top-25 of abs(Theta_h) per sensor | road | uniform | 5 / 15 / 30 / 60 min | does horizon conditioning help? (vs E4 and E0 at the same horizon) |
| **E6** | as E4 | data-driven: each sensor's 25 most correlated (5-min changes) training sensors, no road information | uniform | 5 min | topology ladder: no road info |
| **E7** | as E4 | road | road prior (eta 0.7 if a road link in either direction, else 0.3) | 5 min | topology ladder: roads in candidates and prior |
| NOGRAPH | no graph (operator = I; GRU + attention per sensor) | - | - | 5 min | does any graph help? |
| HA | historical average per weekday x 5-min slot x sensor | - | - | 5 min | lower reference |

Topology ladder: E6 (no road info) -> E4 (roads choose candidates) -> E7 (roads choose candidates and the prior).

---

## 5. Component decisions and what we learned building them

**VAR (E1, base of E2-E7).** Lag 3, ridge alpha 1e6, road neighbours plus self, training rows only (chosen on validation one-step RMSE). Graph = sum_k abs(A_k): same 51,686 edges as the road graph, different weights.

**PEM (E3, and P for E4/E5/E7).**
- Faithful to the paper's objective, posterior and Algorithm 1; sped up (exact gradient, vectorised, L-BFGS-B with value-equivalent tolerance), run per local unit U_i = {i} + in- and out-neighbours, merged by max (P) and OR (D).
- **Finding:** on raw residuals 72-74 % of PEM's directed edges point from the lower- to the higher-variance sensor ("varsortability"; the paper assumes comparable error variances). **Standardised residuals fix it (46-48 %).**
- Raw split-half Jaccard rises with density even for random graphs; chance-corrected, all standardised settings are about equally stable (~0.31).
- Current setting: **standardised residuals, nu0 = 0.0017, nu1 = 1 (paper defaults), tau 0.01, uniform prior 0.5, threshold 0.5** (threshold 0.3 / 0.7 barely matter). Stage B (validation training of dense 0.0017/1 vs sparse 0.05/1) was skipped to get the full-framework result first.
- Solver: the 6 largest units (77-91 sensors) hung in the SLSQP fallback (~4,200 parameters); fallback amended (section 8). All 2,056 units now converge.

**VAR-LiNGAM (E2, E4, E5, E7).**
- DirectLiNGAM (`lingam` 1.13.0) per local unit on raw VAR residuals; global B_0 row i taken from the unit centred on sensor i ("home unit"). 15,883 contemporaneous links.
- **Finding: B_0 is intrinsically unstable on this data.** Two halves of the training rows (all rows each) agree poorly: median relative difference 0.95, edge Jaccard 0.31, sign agreement 86 %, causal-order tau 0.32 (part of this may be genuine change, Jan-Jun vs Jun-Nov).
- All rows would cost ~1,000 CPU-h; by the pre-fixed rule, **10,000 evenly spaced rows per unit** are used (agreement with full rows 0.53 / 0.45 / 97 % / 0.67, better than the halves; ~4 h on the laptop).

**E6 data-driven candidates.** Correlation of 5-minute changes (raw flows share the daily cycle), k = 25 (the road graph's mean in-degree): 51,400 links, 45 % of them road links, median abs(corr) 0.17. **Hubness:** a few sensors are top-25 neighbours of hundreds of others (max 1,209); not a data artefact. Units therefore keep all of a sensor's own sources and at most its 25 strongest reverse neighbours (mean unit 28.4, max 43; road units mean 31.6, max 91).

**E5 horizon graphs.** Built for h = 1, 3, 6, 12 (50,555-50,927 edges, ~24.7 sources per sensor). Graph files: `outputs/graphs/E5__h{h}__lag3_alpha1e+06__rows10000__lag3_alpha1e+06__nu0_0.0017_nu1_1_tau_0.01_thr_0.5_eta_0.5_L-BFGS-B_std.npz`.

Built graphs (in `outputs/graphs/`): E1; E2, E3, E4 for the current settings; E5 at all four horizons. **Not yet built: E6, E7.**

---

## 6. Results so far (test set, 5-minute forecasts, seed 42)

| Run | RMSE | MAE | masked MAPE | R2 | Acc. (10 %) | Best epoch | Status |
|---|---|---|---|---|---|---|---|
| HA | 54.92 | 33.24 | 19.9 % | 0.885 | 0.467 | - | done |
| **E0** (base) | **29.76** | **19.34** | 11.89 % | 0.966 | 0.611 | 15 | official E0 |
| **E1** (VAR) | **28.10** | **18.36** | 10.49 % | 0.970 | 0.639 | 15 | done (older run, no epoch_metrics file) |
| **E5** (full) | **28.17** | **18.47** | 11.41 % | 0.970 | 0.623 | **4** | done, unstable training |
| old E3 (raw PEM), seed 42 | 68.10 | 46.08 | 35.9 % | 0.822 | 0.296 | 2 | failed run (collapsed) |
| old E3 (raw PEM), seed 7 | 40.76 | 28.35 | 18.9 % | 0.936 | 0.455 | 20 | valid but worse; uses the old PEM setting |

Significance:
- **E1 vs E0:** better on 59 % of sensors (Wilcoxon p ~ 1e-30), Diebold-Mariano p < 0.001. VAR helps.
- **E5 vs E0:** -5.4 % RMSE, -4.5 % MAE; better on 62 % of sensors (p ~ 1e-33), DM p < 0.001. **The full framework significantly beats the base.**
- **E5 vs E1:** tied (+0.2 % RMSE; better on 56 % of sensors, p ~ 1e-4, but DM p = 0.15; worse MAPE).

**Training instability (key open issue).** E5's validation RMSE reached 28.4 at epoch 4, then collapsed to 78-112 (epochs 6-15) and only partly recovered (50.9 at epoch 20). Its test score therefore comes from a 4-epoch model, likely an under-estimate. Hypothesis "the E5 graph amplifies signals" was tested and **rejected** (A3T-GCN operator spectral radius 1.00 for all graphs; E5 max row sum 4.2, between E1 2.9 and E0 5.5). Same pattern as the old E3 seed-42 collapse: an optimisation failure. E0 also spikes early but recovers after the LR cut. Run-to-run noise with identical settings is ~2 % (E0: 30.42 earlier vs 29.76 now; GPU not bit-reproducible).

**Phase 3 (shared settings).** Reference lr 0.001 / 20 epochs, best validation 29.98; lr 0.0005 gave 30.50 (20 epochs) and 29.75 (30 epochs, +0.8 %, below the 2 % margin). Kept lr 0.001 / 20 epochs (`results/kaggle/phase3_decision.json`).

---

## 7. What is left

**Kaggle runs** (about 5.8 GPU-h each; set `MAX_HOURS` 1 h below the remaining weekly quota):

| Priority | Run | Purpose |
|---|---|---|
| 1 | **E5, seed 7** (h = 1) | is the collapse bad luck? how good is a stable E5 vs E1? |
| 2 | **E5 and E0 at 60 min** (h = 12) | does horizon conditioning help where it should matter most? |
| 3 | E5 and E0 at 15 / 30 min; E4 at 15 / 30 / 60 min | full horizon curve; E4 is the matched no-conditioning comparison |
| 4 | E2, E4 (5 min) | locate the gain / instability: LiNGAM, PEM |
| 5 | E3 (new PEM setting), NOGRAPH | PEM alone; does any graph help? |
| 6 | E6, E7 (after their graphs are built) | topology ladder |
| 7 | seeds 7 and 123 for the final set | mean +- std, error bars |
| optional | E1 rerun with new logging | per-epoch metrics file for E1 |

**Laptop computation still needed:**
- E7: one full PEM run with the road prior (`run_local_pem.py --prior road`), then `build_graph.py E7` (hours, mostly the largest units).
- E6: `run_var.py --candidates corr_k25 --save-residuals`, then `run_local_pem.py --var-run lag3_alpha1e+06__cand-corr_k25`, `run_local_lingam.py --var-run lag3_alpha1e+06__cand-corr_k25`, then `build_graph.py E6`.
- Optional: PEM Stage B (dense vs sparse) if PEM's setting becomes a question.

**Reporting:** rerun `compare_results.py` (first run = reference; several horizons give `10_error_by_horizon`), `plot_graphs.py`, `summarize_runs.py` as results arrive; then the final report (Phase 11).

**Limitations to state in the thesis:** single dataset and year; mostly single seed so far; LiNGAM's B_0 unstable between halves; PEM directions variance-driven unless standardised; E5's first run unstable; causal claims are predictive, not interventional; E6 is "data-driven neighbourhoods", not full global discovery (2,056-variable LiNGAM/PEM is infeasible); post-hoc amendments listed below.

---

## 8. Deviations and amendments (all recorded in `docs/architecture.md`)

1. Experiment renumbering (25 Oct): old E2 (VAR + PEM) is now E3.
2. PEM Stage A selection rule amended after the sweeps (raw Jaccard favours dense graphs) to chance-corrected stability, uniform prior, threshold 0.5. Approved by the author; report as post-hoc.
3. Phase 3: run A (lr 0.001, 30 epochs) skipped, since the E0 curve plateaued after epoch 15; E0_final's validation curve served as the reference.
4. LiNGAM: 10,000 rows per unit after the split-half check (rule fixed before the result).
5. PEM Stage B skipped; paper defaults used to get the full-framework result first.
6. PEM solver fallback (D5) amended: 3 warm L-BFGS-B restarts; SLSQP only for <= 1,500 parameters; larger problems accepted only once the objective stops changing. All 6 large units converged after one restart.
7. E5 includes a 5-minute graph (h = 1) so it compares directly with E0.

---

## 9. Where things are

```
implementation/                     (git: github.com/prof-sanjay/A3TGCN-VAR-PEM-Implementation)
  main.py                 train/validate/test one experiment: --experiment E0..E7|NOGRAPH --horizon --seed
                          --selection-only --evaluate-only RUN --milestones --max-hours --smoke
  run_var.py              VAR (--candidates road|corr_k25)      -> outputs/var_train/
  run_local_pem.py        local PEM (--standardise --prior)     -> outputs/pem/
  run_local_lingam.py     local VAR-LiNGAM (--max-rows)         -> outputs/lingam/
  build_candidates.py     E6 data-driven candidate links        -> outputs/candidates/
  build_graph.py          graph artifacts E1-E7                 -> outputs/graphs/
  baseline_ha.py          Historical Average                    -> results/HA/
  summarize_runs.py, compare_results.py, plot_graphs.py         -> results/reports/
  selection/              PEM sweep + Stage A report, Phase 3 choice, LiNGAM row check
  tools/                  data checks, GPU check
  config/config.py        ALL settings (shared training, VAR, PEM, LiNGAM, run names, FINAL_GRAPHS)
  experiments/registry.py E0-E7 + NOGRAPH definitions
  causal/ data/ graph/ models/ training/                       library code
  tests/                  python -m tests.<name>  (all passing)
  notebooks/ablation-study.ipynb   Kaggle notebook (settings cell + cell 2b check)
  docs/architecture.md    full design log, decisions and results (sections 1-14)
  results/kaggle/         downloaded runs: output_E0_final, output_E1_final, output_E5_final, ...
```

Environments: `traffic-gpu` (Python 3.10, TF 2.10; main.py, reports) and `traffic-causal` (Python 3.12; PEM, LiNGAM). Kaggle: push the repo, re-import the notebook, change only the settings cell (EXPERIMENT, HORIZON, SEED, MAX_HOURS), Save & Run All (Commit), download `output_<RUN_NAME>`. Cell 2b must say "verified"; if not, the pushed code is out of date. Long laptop jobs started from an assistant session stop when the session ends; run them in your own terminal when possible.

---

## 10. Open questions for the author

1. Is E5's 4-epoch result an under-estimate? The seed-7 run decides. If E5 collapses again, consider a documented stability measure applied to ALL experiments (e.g. lower gradient-clipping threshold), not to E5 alone.
2. Which ablations are essential for the thesis given the GPU quota (minimum: E5 seed 7, E0/E5 at 60 min, E4)?
3. Should PEM Stage B be run, or the paper-default setting be justified as is?
