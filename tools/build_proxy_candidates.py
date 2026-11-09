"""
Build candidate graph-selection rules for the validation proxy (tools/graph_proxy.py).
Uses the stored impulse responses Psi_h and PEM probabilities P of the E8 prior study; nothing is re-estimated.

    V_std         top-K of |Psi_h[i,j]| * sigma_j / sigma_i              VAR effect in standardised units, no PEM
    E8std_<prior> top-K of P[i,j] * |Psi_h[i,j]| * sigma_j / sigma_i      the E8 rule on the standardised effect
    fuse<w>       top-K of (1-w) * a[i,j] + w * P[i,j]  inside the support of P, where a is the standardised
                  effect divided by the largest effect in its row. High-P pairs can enter with a small effect.

sigma = standard deviation of the observed TRAINING flow of each sensor. The files are written to
outputs/proxy_graphs/<name>__h<h>.npz and are NOT artifacts for training.

    python tools/build_proxy_candidates.py --horizons 1 6 12
"""

import argparse
import sys
from pathlib import Path

import numpy as np

IMPLEMENTATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPLEMENTATION_DIR))
sys.path.insert(0, str(IMPLEMENTATION_DIR / "tools"))

import graph_proxy  # noqa: E402  (loader, split constants)

K = 25


def topk(scores, k=K):
    out = np.zeros(scores.shape, dtype=bool)
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    out[rows, idx] = scores[rows, idx] > 0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 12])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.25, 0.5, 0.9])
    parser.add_argument("--out", default=str(IMPLEMENTATION_DIR / "outputs" / "proxy_graphs"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flow, sensors = graph_proxy.load_flow()
    sigma = np.nanstd(flow[:graph_proxy.TRAIN_END], axis=0)
    sigma[sigma == 0] = 1.0
    scale = sigma[None, :] / sigma[:, None]                     # sigma_j / sigma_i, G[target i, source j]

    base = IMPLEMENTATION_DIR / "outputs"
    p_flow = np.load(base / "E8 flow" / "e8_prior_output" / "e8" / "pem_probabilities.npz")["probabilities"].astype(np.float64)
    p_unif = np.load(base / "E8 Uniform" / "e8_prior_output" / "e8" / "pem_probabilities.npz")["probabilities"].astype(np.float64)

    for h in args.horizons:
        psi = np.load(base / "E8 flow" / "e8_prior_output" / "e8" / f"psi_h{h}.npz")["psi"].astype(np.float64)
        effect = np.abs(psi) * scale
        np.fill_diagonal(effect, 0.0)
        graphs = {
            "V_std": topk(effect),
            "E8std_flow": topk(p_flow * effect),
            "E8std_uniform": topk(p_unif * effect),
        }
        support = p_flow > 0
        a = np.where(support, effect, 0.0)
        a = a / np.maximum(a.max(axis=1, keepdims=True), 1e-12)
        for w in args.weights:
            graphs[f"fuse{int(round(w * 100))}"] = topk(np.where(support, (1 - w) * a + w * p_flow, 0.0))
        # ---- controls on the SAME candidate pool (pairs sharing a unit, i.e. the support of P) ----
        # They tell whether a gain of the fuse graphs comes from PEM or only from widening the pool.
        rng = np.random.default_rng(0)
        rand_support = np.zeros(support.shape, dtype=bool)
        for i in range(support.shape[0]):
            cols = np.flatnonzero(support[i])
            cols = cols[cols != i]
            rand_support[i, rng.choice(cols, size=min(K, len(cols)), replace=False)] = True
        graphs["ctrl_rand_support"] = rand_support                                   # random sources from the pool
        eta = np.load(base / "priors" / "corr_flow.npz")["eta"].astype(np.float64)
        graphs["ctrl_corr"] = topk(np.where(support, eta, 0.0))                      # correlation only, no PEM
        for w in args.weights:
            if abs(w - 0.9) < 1e-9:                                                  # same rule with the uniform-prior P
                graphs["fuse90_uniform"] = topk(np.where(support, (1 - w) * a + w * p_unif, 0.0))
        for name, g in graphs.items():
            np.fill_diagonal(g, False)
            np.savez_compressed(out / f"{name}__h{h}.npz", adjacency=g.astype(np.float32),
                                sensors=sensors, train_end=graph_proxy.TRAIN_END)
            print(f"h={h:2d} {name:14s} edges {int(g.sum()):6d}  sources/target {g.sum(axis=1).mean():5.1f}")


if __name__ == "__main__":
    main()
