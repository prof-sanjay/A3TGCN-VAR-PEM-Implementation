"""
Build WEIGHTED, training-ready graph artifacts for E9: knockoff-filter edge selection (FDR-controlled, no top-k).

Selection  : the edges chosen by tools/run_knockoff_selection.py (group knockoffs, knockoff+ threshold at level q) from the
             candidate pool of the causal units; at most 25 sources per target, some targets keep none.
Weights    (non-negative, as A3T-GCN requires), the same convention as build_hybrid_graphs.py:
    edge supported by the VAR (|Psi_h[i,j]| > 0)   a_h[i,j] = |Psi_h[i,j]| sigma_j / sigma_i      (VAR direction and size)
    edge added from the pool                       kappa * rho[i,j]; rho = |partial correlation| of the forecasting
                                                   relation (the same bounded magnitude as the hybrid graphs, cached in
                                                   outputs/proxy_graphs/pcscore__h<h>.npz), kappa ONE global constant that
                                                   puts the median added weight at the median VAR-supported weight (a
                                                   per-row constant was tried: the knockoff selection keeps edges whose
                                                   rho spans orders of magnitude inside a row, and it gave weights up to
                                                   5 next to a mean of 0.045). The knockoff filter decides WHICH
                                                   edges exist; rho only sets the size of the added ones. (The knockoff
                                                   statistic itself is an unbounded lasso entry penalty: it produced
                                                   weights above 12 next to a mean of 0.045, so it is not used as a weight.)
    global scale   one constant so that the mean non-zero weight equals that of the E8V graph of the same horizon, so a
                   difference between graphs comes from WHICH edges they contain, not from the weight scale.
Estimated from TRAINING rows only; no validation or test row is read.

The "method" stored in the file is "var_pem_horizon": the E8 slot of the training code only accepts that label (the knockoff
selection is recorded in the .json summary). File names follow the hybrid artifacts, for EXPERIMENT = "E8" with GRAPH_VARIANT = "e9_g50_q30" in ablation-study.ipynb:
    outputs/hybrid_graphs/E8__h{h}__lag3_alpha1e+06__e9_g50_q30.npz  (+ .json summary)

    python build_e9_graphs.py --variant e9_g50_q30 --horizons 1
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
import graph_proxy  # noqa: E402

K = 25
VARIANT_PATTERN = re.compile(r"^e9_(?P<tag>[gm]\d+)_q(?P<q>\d+)$")


def parse_variant(name):
    m = VARIANT_PATTERN.match(name)
    if not m:
        raise ValueError(f"unknown variant {name!r}: use e9_<tag>_q<level*100>, e.g. e9_g50_q30")
    return m["tag"], int(m["q"])


def weigh_selection(effect, selected, rho):
    """
    Returns (weights, in_var): weights[i, j] > 0 on `selected` (G[target, source]) and `in_var`, the mask of the selected
    edges that the VAR supports. An added edge needs a positive partial correlation rho (every one has, in practice).
    Weights follow the module docstring (before the global scale).
    """
    W = rho
    n = effect.shape[0]
    selected = selected.copy()
    np.fill_diagonal(selected, False)
    in_var = selected & (effect > 0)
    added = selected & ~in_var
    added &= W > 0
    selected = in_var | added
    weights = np.zeros((n, n), dtype=np.float64)
    weights[in_var] = effect[in_var]
    var_values = weights[in_var]
    core_median = float(np.median(var_values)) if var_values.size else 1.0
    added_median = float(np.median(W[added])) if added.any() else 1.0
    kappa = core_median / added_median                       # ONE constant: no row can blow up
    weights[added] = kappa * W[added]
    np.fill_diagonal(weights, 0.0)
    return weights, in_var


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1])
    parser.add_argument("--e8v-dir", default=str(IMPLEMENTATION_DIR / "outputs" / "artifacts_all"))
    parser.add_argument("--out", default=str(IMPLEMENTATION_DIR / "outputs" / "hybrid_graphs"))
    args = parser.parse_args()

    tag, q = parse_variant(args.variant)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = IMPLEMENTATION_DIR / "outputs"
    t0 = time.time()
    flow, sensors = graph_proxy.load_flow()
    sigma = np.nanstd(flow[:graph_proxy.TRAIN_END], axis=0)           # training rows only
    sigma[sigma == 0] = 1.0
    scale = sigma[None, :] / sigma[:, None]
    e8 = base / "E8 flow" / "e8_prior_output" / "e8"
    n = len(sensors)

    for h in args.horizons:
        ko_path = base / "knockoff" / f"ko_scores_{tag}_h{h}.npz"
        if not ko_path.exists():
            raise FileNotFoundError(f"{ko_path}: run tools/run_knockoff_selection.py --horizon {h} --tag _{tag} first (all targets)")
        ko = np.load(ko_path)
        assert np.array_equal(ko["sensors"], sensors) and len(ko["targets"]) == n, "knockoff scores must cover all targets"
        selected = ko[f"sel_q{q}"]
        rho = np.load(base / "proxy_graphs" / f"pcscore__h{h}.npz")["score"].astype(np.float64)
        psi = np.load(e8 / f"psi_h{h}.npz")["psi"].astype(np.float64)
        effect = np.abs(psi) * scale
        np.fill_diagonal(effect, 0.0)
        weights, in_var = weigh_selection(effect, selected, rho)

        e8v_path = Path(args.e8v_dir) / f"E8V__h{h}__lag3_alpha1e+06.npz"
        with np.load(e8v_path) as z:
            ref = np.asarray(z["adjacency"])
            assert np.array_equal(z["sensors"], sensors) and int(z["train_end"]) == graph_proxy.TRAIN_END
        factor = float(ref[ref > 0].mean() / weights[weights > 0].mean())
        weights = weights * factor

        diagnostics = check_model_graph(weights.astype(np.float32), n, allow_self_loops=False)
        degree = (weights > 0).sum(axis=1)
        assert degree.max() <= K and not np.diag(weights).any() and (weights >= 0).all()
        added_mask = (weights > 0) & ~in_var
        summary = {
            "experiment": "E8", "method": "var_pem_horizon", "selection": "knockoff_filter", "horizon": h, "top_k": K, "variant": args.variant,
            "knockoff_group_tag": tag, "fdr_level": q / 100, "weight_scale_factor": factor,
            "edges": int((weights > 0).sum()), "var_supported_edges": int(in_var.sum()), "added_edges": int(added_mask.sum()),
            "targets_without_sources": int((degree == 0).sum()), "sources_per_target_mean": float(degree.mean()),
            "mean_weight_var_edges": float(weights[in_var].mean()) if in_var.any() else None,
            "mean_weight_added": float(weights[added_mask].mean()) if added_mask.any() else None,
            "edge_overlap_with_E8V": float(((weights > 0) & (ref > 0)).sum() / max(((weights > 0) | (ref > 0)).sum(), 1)),
            "graph_stats": diagnostics,
        }
        name = f"E8__h{h}__lag3_alpha1e+06__{args.variant}"
        fields = dict(num_nodes=n, orientation="target_source", experiment="E8", method="var_pem_horizon",
                      train_end=graph_proxy.TRAIN_END, sensors=np.asarray(sensors, dtype=np.int64), horizon=h,
                      top_k=K, source_run=f"lag3_alpha1e+06__{args.variant}")
        np.savez_compressed(out / f"{name}.npz", adjacency=weights.astype(np.float32), **fields)
        (out / f"{name}.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        print(f"h={h:2d} {args.variant}: edges {summary['edges']} (VAR-supported {summary['var_supported_edges']}, added "
              f"{summary['added_edges']}), {summary['targets_without_sources']} targets without sources, "
              f"{summary['sources_per_target_mean']:.1f} sources/target, scale x{factor:.3f}, "
              f"overlap with E8V {summary['edge_overlap_with_E8V']:.3f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
