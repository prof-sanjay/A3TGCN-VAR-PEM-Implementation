"""
Build WEIGHTED, training-ready graph artifacts for the hybrid framework (standardised VAR core + pool additions).

    variant          core                                    additions (from the candidate pool C_i)
    vstd             top-25 standardised effect              none (VAR only; the "fix 1" ablation)
    hyb_pc<k>        top-(25-k) standardised effect          predictive-precision (PEM-style) score, fills the rest
    hyb_rand<k>      top-(25-k) standardised effect          random pool sources (control, same budget)
    hyb_corr<k>      top-(25-k) standardised effect          largest marginal detrended correlation (control)
    optional suffix  _M<M>  caps the candidate pool at the M most correlated candidates per target (hyperparameter)

Weights (non-negative, as A3T-GCN requires):
    core edge     a_h[i,j] = |Psi_h[i,j]| sigma_j / sigma_i          (VAR direction and size, standardised)
    added edge    kappa_i * rho[i,j],  rho = |partial correlation| of the forecasting relation, kappa_i rescales the
                  row's added weights to the median of its core weights (global fallback when the row has no core)
    global scale  the whole graph is multiplied by one constant so that its mean non-zero weight equals that of the
                  E8V graph of the same horizon. This keeps the identity-to-neighbour balance of A3T-GCN comparable,
                  so a difference between graphs comes from WHICH edges they contain, not from the weight scale.

Everything is estimated from TRAINING rows only (sigma, partial correlations); no validation or test row is read.

The files are written for EXPERIMENT = "E8" with GRAPH_VARIANT = <variant> in notebooks/ablation-study.ipynb:
    outputs/hybrid_graphs/E8__h{h}__lag3_alpha1e+06__<variant>.npz  (+ .json summary)
(The "E8" slot is only the plumbing that selects a graph file; these graphs are not the old P x |Psi| E8 graphs.)

Usage (from the implementation directory):
    python build_hybrid_graphs.py --variant hyb_pc10 --horizons 1 6 12
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

IMPLEMENTATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(IMPLEMENTATION_DIR))
sys.path.insert(0, str(IMPLEMENTATION_DIR / "tools"))

from graph.adjacency import check_model_graph  # noqa: E402
import build_hybrid_candidates as bhc  # noqa: E402
import graph_proxy  # noqa: E402

K = 25
VARIANT_PATTERN = re.compile(r"^(?P<rule>vstd|hyb_pc|hyb_rand|hyb_corr)(?P<k>\d*)(?:_M(?P<pool>\d+))?$")


def parse_variant(name):
    m = VARIANT_PATTERN.match(name)
    if not m:
        raise ValueError(f"unknown variant {name!r}: use vstd or hyb_{{pc,rand,corr}}<k_add>[_M<pool size>]")
    rule = m["rule"]
    if rule == "vstd":
        return "vstd", 0, None
    if not m["k"]:
        raise ValueError("hybrid variants need the number of reserved additions, e.g. hyb_pc10")
    k_add = int(m["k"])
    if not 0 < k_add < K:
        raise ValueError(f"k_add must be between 1 and {K - 1}")
    return rule.replace("hyb_", ""), k_add, int(m["pool"]) if m["pool"] else None


def weighted_assemble(effect, pool, rho, eta, k_add, rule, rng):
    """
    Returns (weights, is_core): weights[i, j] > 0 for the selected sources (G[target, source]), and the boolean core mask.
    Selection is identical to build_hybrid_candidates.assemble; weights follow the module docstring (before global scale).
    """
    n = effect.shape[0]
    weights = np.zeros((n, n), dtype=np.float64)
    is_core = np.zeros((n, n), dtype=bool)
    added = np.zeros((n, n), dtype=bool)
    for i in range(n):
        eff = effect[i]
        nz = np.flatnonzero(eff > 0)
        k_core = min(K - k_add, len(nz)) if rule != "vstd" else min(K, len(nz))
        core = nz[np.argsort(-eff[nz])[:k_core]] if k_core > 0 else np.array([], dtype=int)
        weights[i, core] = eff[core]
        is_core[i, core] = True
        room = K - len(core)
        if rule == "vstd" or room <= 0:
            continue
        cand = np.flatnonzero(pool[i])
        cand = cand[~np.isin(cand, core)]
        cand = cand[rho[i, cand] > 0]                 # an added edge needs a defined, positive partial correlation
        if len(cand) == 0:
            continue
        if rule == "rand":
            pick = rng.choice(cand, size=min(room, len(cand)), replace=False)
        elif rule == "corr":
            pick = cand[np.argsort(-eta[i, cand])[:room]]
        else:
            pick = cand[np.argsort(-rho[i, cand])[:room]]
        added[i, pick] = True
    # row-wise kappa: added weights are rescaled to the median of the row's core weights
    core_values = weights[is_core]
    global_core_median = float(np.median(core_values)) if core_values.size else 1.0
    added_values = rho[added]
    global_added_median = float(np.median(added_values)) if added_values.size else 1.0
    for i in np.flatnonzero(added.any(axis=1)):
        cols = np.flatnonzero(added[i])
        row_core = weights[i, is_core[i]]
        target = float(np.median(row_core)) if row_core.size else global_core_median
        median_rho = float(np.median(rho[i, cols]))
        kappa = target / median_rho if median_rho > 0 else target / global_added_median
        weights[i, cols] = kappa * rho[i, cols]
    np.fill_diagonal(weights, 0.0)
    np.fill_diagonal(is_core, False)
    return weights, is_core


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 12])
    parser.add_argument("--e8v-dir", default=str(IMPLEMENTATION_DIR / "outputs" / "artifacts_all"),
                        help="folder with the E8V graphs (weight-scale reference)")
    parser.add_argument("--out", default=str(IMPLEMENTATION_DIR / "outputs" / "hybrid_graphs"))
    args = parser.parse_args()

    rule, k_add, pool_size = parse_variant(args.variant)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = IMPLEMENTATION_DIR / "outputs"
    t0 = time.time()
    flow, sensors = graph_proxy.load_flow()
    train = np.asfortranarray(flow[:graph_proxy.TRAIN_END])            # training rows only
    sigma = np.nanstd(train, axis=0)
    sigma[sigma == 0] = 1.0
    scale = sigma[None, :] / sigma[:, None]
    e8 = base / "E8 flow" / "e8_prior_output" / "e8"
    support = np.load(e8 / "pem_probabilities.npz")["probabilities"] > 0
    eta = np.load(base / "priors" / "corr_flow.npz")["eta"].astype(np.float64)
    pool = bhc.candidate_pool(support, eta, pool_size)
    suffix = f"_M{pool_size}" if pool_size else ""
    n = len(sensors)

    for h in args.horizons:
        psi = np.load(e8 / f"psi_h{h}.npz")["psi"].astype(np.float64)
        effect = np.abs(psi) * scale
        np.fill_diagonal(effect, 0.0)
        cache = base / "proxy_graphs" / f"pcscore__h{h}{suffix}.npz"
        if rule != "vstd" and cache.exists():
            rho = np.load(cache)["score"].astype(np.float64)
        elif rule != "vstd":
            print(f"h={h}: computing predictive scores (about 7 minutes, cached for later variants)", flush=True)
            rho = bhc.predictive_scores(train, pool, h).astype(np.float64)
            np.savez_compressed(cache, score=rho.astype(np.float32))
        else:
            rho = np.zeros((n, n))
        weights, is_core = weighted_assemble(effect, pool, rho, eta, k_add, rule, np.random.default_rng(0))

        # global scale: mean non-zero weight equal to the E8V graph of this horizon
        e8v_path = Path(args.e8v_dir) / f"E8V__h{h}__lag3_alpha1e+06.npz"
        with np.load(e8v_path) as z:
            ref = np.asarray(z["adjacency"])
            assert np.array_equal(z["sensors"], sensors) and int(z["train_end"]) == graph_proxy.TRAIN_END
        factor = float(ref[ref > 0].mean() / weights[weights > 0].mean())
        weights = weights * factor

        diagnostics = check_model_graph(weights.astype(np.float32), n, allow_self_loops=False)
        degree = (weights > 0).sum(axis=1)
        assert degree.max() <= K and not np.diag(weights).any() and (weights >= 0).all()
        added_mask = (weights > 0) & ~is_core
        summary = {
            "experiment": "E8", "method": "var_pem_horizon", "horizon": h, "top_k": K, "variant": args.variant,
            "rule": rule, "k_add": k_add, "pool_size_cap": pool_size, "weight_scale_factor": factor,
            "edges": int((weights > 0).sum()), "core_edges": int(is_core.sum()), "added_edges": int(added_mask.sum()),
            "added_edges_outside_var_support": int((added_mask & (np.abs(psi) == 0)).sum()),
            "mean_weight_core": float(weights[is_core].mean()) if is_core.any() else None,
            "mean_weight_added": float(weights[added_mask].mean()) if added_mask.any() else None,
            "edge_overlap_with_E8V": float(((weights > 0) & (ref > 0)).sum() / max(((weights > 0) | (ref > 0)).sum(), 1)),
            "graph_stats": diagnostics,
        }
        name = f"E8__h{h}__lag3_alpha1e+06__{args.variant}"
        bhc_fields = dict(num_nodes=n, orientation="target_source", experiment="E8", method="var_pem_horizon",
                          train_end=graph_proxy.TRAIN_END, sensors=np.asarray(sensors, dtype=np.int64), horizon=h,
                          top_k=K, source_run=f"lag3_alpha1e+06__{args.variant}")
        np.savez_compressed(out / f"{name}.npz", adjacency=weights.astype(np.float32), **bhc_fields)
        (out / f"{name}.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        print(f"h={h:2d} {args.variant}: edges {summary['edges']} (core {summary['core_edges']}, added {summary['added_edges']}, "
              f"{summary['added_edges_outside_var_support']} outside the VAR support), scale x{factor:.3f}, "
              f"overlap with E8V {summary['edge_overlap_with_E8V']:.3f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
