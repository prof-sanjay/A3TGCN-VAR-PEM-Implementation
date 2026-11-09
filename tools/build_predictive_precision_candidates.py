"""
Candidate rule that models the FORECASTING RELATION itself with a precision matrix (quick proxy version).

For each target i and horizon h, the vector v = [ x_i(t+h), x_i(t), x_j(t) for j in the pool of i ] is formed on TRAINING
rows (complete rows only), where the pool of i is the set of sensors that share a causal unit with i (the support of
the PEM probabilities). Its correlation matrix R is shrunk, R_s = (1 - a) R + a I, and inverted, Omega = R_s^-1.
The conditional (partial) dependence of the future target on source j given ALL the other sources and the target's own
present value is

        score[i, j] = | Omega[y, j] | / sqrt( Omega[y, y] * Omega[j, j] ).

The top-K sources by this score form the graph (G[target, source]). This is the quantity a spike-and-slab precision
model (PEM) would turn into an inclusion probability; here it is used directly as a fast stand-in, to test whether the
RIGHT quantity (source needed to predict the target, given the other sources) selects better than random or marginal
correlation from the same pool. Direction comes from time order: the future target is regressed on current sources.

    python tools/build_predictive_precision_candidates.py --horizons 1 6 12
Output: outputs/proxy_graphs/pc25__h<h>.npz   (not a training artifact)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

IMPLEMENTATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPLEMENTATION_DIR))
sys.path.insert(0, str(IMPLEMENTATION_DIR / "tools"))

import graph_proxy  # noqa: E402

K = 25
SHRINK = 0.02


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 12])
    parser.add_argument("--out", default=str(IMPLEMENTATION_DIR / "outputs" / "proxy_graphs"))
    parser.add_argument("--shrink", type=float, default=SHRINK)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flow, sensors = graph_proxy.load_flow()
    train = np.asfortranarray(flow[:graph_proxy.TRAIN_END])
    p_flow = np.load(IMPLEMENTATION_DIR / "outputs" / "E8 flow" / "e8_prior_output" / "e8" / "pem_probabilities.npz")["probabilities"]
    pool = p_flow > 0
    n = len(sensors)

    for h in args.horizons:
        t0 = time.time()
        graph = np.zeros((n, n), dtype=bool)
        for i in range(n):
            cols = np.flatnonzero(pool[i])
            cols = cols[cols != i]
            if len(cols) == 0:
                continue
            # columns first (cheap contiguous gather), then contiguous row ranges: target rows h..TRAIN_END-1,
            # inputs at rows 0..TRAIN_END-h-1 (x_i(t) first, then the pool at time t)
            sub = train[:, np.concatenate([[i], cols])]
            v = np.column_stack([train[h:graph_proxy.TRAIN_END, i], sub[:graph_proxy.TRAIN_END - h]])
            v = v[np.isfinite(v).all(axis=1)]
            if len(v) < 1000:
                continue
            corr = np.corrcoef(v, rowvar=False)
            corr = np.nan_to_num(corr)
            shrunk = (1 - args.shrink) * corr + args.shrink * np.eye(corr.shape[0])
            omega = np.linalg.inv(shrunk)
            d = np.sqrt(np.abs(np.diag(omega)))
            score = np.abs(omega[0, 2:]) / (d[0] * d[2:])                  # target vs each pool source
            keep = np.argsort(-score)[:K]
            graph[i, cols[keep]] = True
            if (i + 1) % 200 == 0 or i + 1 == n:
                print(f"  h={h} target {i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
        np.fill_diagonal(graph, False)
        np.savez_compressed(out / f"pc25__h{h}.npz", adjacency=graph.astype(np.float32), sensors=sensors,
                            train_end=graph_proxy.TRAIN_END)
        print(f"h={h:2d} pc25: edges {int(graph.sum())}, sources/target {graph.sum(axis=1).mean():.1f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
