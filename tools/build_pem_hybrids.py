"""
Hybrid proxy graphs from the REAL PEM scores of tools/run_pem_forecast_relation.py.

Same assembly as build_hybrid_candidates.py (standardised-effect core + pool additions to a total of 25 sources), with the
additions ranked by the PEM-regularised partial correlation rho_pem instead of the shrunk-inverse stand-in.
Graphs exist for the sampled targets only (the proxy evaluates exactly those targets).

    python tools/build_pem_hybrids.py --horizon 1 --k-add 5 10 15
Output: outputs/proxy_graphs/hyb_pemflow<k>__h<h>.npz and hyb_pemuni<k>__h<h>.npz   (flow prior / uniform prior)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

IMPLEMENTATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPLEMENTATION_DIR))
sys.path.insert(0, str(IMPLEMENTATION_DIR / "tools"))

import build_hybrid_candidates as bhc  # noqa: E402
import graph_proxy  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--k-add", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--priors", nargs="+", default=["flow", "uniform"])
    parser.add_argument("--scores-dir", default=str(IMPLEMENTATION_DIR / "outputs" / "pem_forecast"))
    parser.add_argument("--out", default=str(IMPLEMENTATION_DIR / "outputs" / "proxy_graphs"))
    args = parser.parse_args()

    base = IMPLEMENTATION_DIR / "outputs"
    h = args.horizon
    flow, sensors = graph_proxy.load_flow()
    sigma = np.nanstd(flow[:graph_proxy.TRAIN_END], axis=0)
    sigma[sigma == 0] = 1.0
    scale = sigma[None, :] / sigma[:, None]
    e8 = base / "E8 flow" / "e8_prior_output" / "e8"
    psi = np.load(e8 / f"psi_h{h}.npz")["psi"].astype(np.float64)
    effect = np.abs(psi) * scale
    np.fill_diagonal(effect, 0.0)
    pool = np.load(e8 / "pem_probabilities.npz")["probabilities"] > 0
    np.fill_diagonal(pool, False)

    for prior in args.priors:
        z = np.load(Path(args.scores_dir) / f"pem_scores_{prior}_h{h}.npz")
        assert np.array_equal(z["sensors"], sensors)
        rho = z["rho"].astype(np.float64)
        slab = z["slab"].astype(np.float64)
        targets = z["targets"]
        used = rho[targets][pool[targets]]
        slab_used = slab[targets][pool[targets]]
        print(f"{prior}: pool pairs {used.size}; share of pool pairs with slab probability < 0.5: {(slab_used < 0.5).mean():.1%}, "
              f"> 0.99: {(slab_used > 0.99).mean():.1%}; median rho {np.median(used):.4f}", flush=True)
        short = "pemflow" if prior == "flow" else "pemuni"
        for k_add in args.k_add:
            g = bhc.assemble(effect, pool, rho, k_add, "pc", np.random.default_rng(0))
            g_rows = np.zeros_like(g)
            g_rows[targets] = g[targets]                         # rows outside the sample are never evaluated
            np.savez_compressed(Path(args.out) / f"hyb_{short}{k_add}__h{h}.npz", adjacency=g_rows.astype(np.float32),
                                sensors=sensors, train_end=graph_proxy.TRAIN_END)
            print(f"  hyb_{short}{k_add}: sources/target on the sample {g_rows[targets].sum(axis=1).mean():.1f}", flush=True)


if __name__ == "__main__":
    main()
