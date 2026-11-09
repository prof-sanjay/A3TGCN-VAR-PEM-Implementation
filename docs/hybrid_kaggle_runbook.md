# Hybrid graphs: local build, verification, Kaggle GPU training

Status 2026-11-09: tooling is ready and tested. Items marked **[after proxy]** depend on the validation-proxy results
(`outputs/proxy/proxy_sample500_h*_hybrid.json`) and are deliberately left open.

## What is built where

| Stage | Where | Cost | Command |
|---|---|---|---|
| Predictive scores (once per horizon and pool size) | local CPU | about 7 min per horizon, cached in `outputs/proxy_graphs/pcscore__h<h>[_M<M>].npz` | done automatically by the builder |
| Weighted graph artifacts | local CPU | about 1 min per variant and horizon after the scores exist | `python build_hybrid_graphs.py --variant hyb_pc10 --horizons 1 6 12` |
| Verify | local | seconds | `python tools/verify_graph_artifacts.py outputs/hybrid_graphs` |
| Stage one upload folder | local | seconds | `python tools/stage_upload.py --src outputs/hybrid_graphs --variants hyb_pc10 vstd --horizons 1 --dest outputs/upload_hybrid` |
| A3T-GCN training | Kaggle GPU | about 5.4 h per run (20 epochs) | `notebooks/ablation-study.ipynb` |

Nothing from the VAR, PEM or proxy runs needs to be uploaded; training reads only the finished graph files.

## Variants (names are `GRAPH_VARIANT` values with `EXPERIMENT = "E8"`)

| Variant | Meaning |
|---|---|
| `vstd` | standardised VAR effect only (no pool additions) |
| `hyb_pc<k>` | standardised core + predictive-precision (PEM-style) additions, at least `k` of 25 slots reserved |
| `hyb_rand<k>` | same budget, random pool sources (control for `pc`) |
| `hyb_corr<k>` | same budget, largest marginal correlation (control for `pc`) |
| suffix `_M<M>` | candidate pool capped at the M most correlated sources per target (hyperparameter; whole pool when absent) |

Weights: core = standardised effect, additions = row-rescaled partial correlation, and one global factor makes the mean
non-zero weight equal to that of the E8V graph of the same horizon (so differences come from which edges exist, not from the
weight scale). Everything is estimated on training rows only.

## Kaggle settings (per run)

Inputs: the GBA dataset and the artifacts dataset (E8V files for the baseline, plus the staged hybrid files and both `sensor_filter_*.json`).
Accelerator: GPU. Use the same Kaggle image / TensorFlow version for every run that will be compared
(the E8V seed-42 run used TensorFlow 2.21.0, the other E8 runs 2.20.0).

```
EXPERIMENT    = "E8"
GRAPH_VARIANT = "<variant>"      # a string, with quotes
HORIZON       = <1 | 6 | 12>
SEED          = <42 | 7 | 123>
```
The notebook stops at once if the variant is malformed, if there is no GPU, or if the graph file is missing or does not match the run.

## Run list  **[after proxy]**

| Priority | Variant | Horizon | Seeds | Compare with |
|---|---|---|---|---|
| 1 | _best hybrid from the proxy_ | 1 | 42, 7 | E8V (26.39 / 25.87), `vstd` |
| 2 | _its random control (same budget)_ | 1 | 42 | the hybrid |
| 3 | `vstd` | 6 or 12 | 42 | E8V at that horizon |
| later | pool-size sweep `_M<M>` | decided on validation | | `hyb_pc<k>` with the whole pool |

GPU budget: each row-seed is about 5.4 GPU hours; two Kaggle accounts run in parallel.

## Reading the results

- Per variant: test RMSE, MAE, masked MAPE and the seed-to-seed gap; compare means against E8V on the same seeds.
- A component counts only if it beats its control by more than the seed gap (0.14 to 1.4 RMSE so far).
- The test set is scored once per run; no graph or budget is chosen on test numbers.
- Report the construction as a post-hoc amendment of E8, with the proxy and validation evidence that selected it.
