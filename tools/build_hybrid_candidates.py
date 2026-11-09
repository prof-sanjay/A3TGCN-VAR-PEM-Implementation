"""
Hybrid graphs for the validation proxy: a standardised-VAR core plus pool additions.

For every target i and horizon h:
    core   = the (K - k_add) sources with the largest standardised effect  a_h[i,j] = |Psi_h[i,j]| sigma_j / sigma_i
             (only sources with a non-zero effect; at h = 1 the road-restricted VAR gives about 20 per target)
    added  = sources from the candidate pool C_i that are not in the core, filling the remaining budget (K - |core|),
             chosen by one of three rules on the SAME pool:
               pc    predictive precision score (the stand-in for PEM on the forecasting relation)
               rand  random sources of the pool (seeded)
               corr  largest marginal detrended-flow correlation (no PEM)
The comparison pc vs rand vs corr at the same k_add shows whether PEM-style selection of additions adds anything.

Candidate pool C_i: the sensors that share a causal unit with i (the support of the PEM probabilities), about 67 per target
on average (median 60, range 0-173). It is a HYPERPARAMETER: --pool-size M keeps only the M pool sources with the largest
marginal detrended-flow correlation (None = the whole pool).

Weights are not needed by the proxy (it uses the support); they matter only for A3T-GCN.
Output: outputs/proxy_graphs/hyb_<rule><k_add>[_M<M>]__h<h>.npz   (not a training artifact)

    python tools/build_hybrid_candidates.py --horizons 1 6 12 --k-add 5 10 15
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


def candidate_pool(support, eta, pool_size):
    """Boolean pool matrix; optionally capped at the pool_size sources with the largest eta (= |r|) per target."""
    pool = support.copy()
    np.fill_diagonal(pool, False)
    if pool_size is None:
        return pool
    for i in range(pool.shape[0]):
        cols = np.flatnonzero(pool[i])
        if len(cols) > pool_size:
            keep = cols[np.argsort(-eta[i, cols])[:pool_size]]
            pool[i] = False
            pool[i, keep] = True
    return pool


def predictive_scores(train, pool, h):
    """score[i, j] = partial correlation of x_i(t+h) with x_j(t) given the other pool sources and x_i(t)."""
    n = pool.shape[0]
    score = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        cols = np.flatnonzero(pool[i])
        if len(cols) == 0:
            continue
        sub = train[:, np.concatenate([[i], cols])]
        v = np.column_stack([train[h:graph_proxy.TRAIN_END, i], sub[:graph_proxy.TRAIN_END - h]])
        v = v[np.isfinite(v).all(axis=1)]
        if len(v) < 1000:
            continue
        corr = np.nan_to_num(np.corrcoef(v, rowvar=False))
        omega = np.linalg.inv((1 - SHRINK) * corr + SHRINK * np.eye(corr.shape[0]))
        d = np.sqrt(np.abs(np.diag(omega)))
        score[i, cols] = np.abs(omega[0, 2:]) / (d[0] * d[2:])
    return score


def assemble(effect, pool, extra_score, k_add, rule, rng):
    """Core by standardised effect, additions by `rule` from the pool; total at most K sources per target."""
    n = effect.shape[0]
    graph = np.zeros((n, n), dtype=bool)
    for i in range(n):
        eff = effect[i]
        nz = np.flatnonzero(eff > 0)
        k_core = min(K - k_add, len(nz))
        core = nz[np.argsort(-eff[nz])[:k_core]] if k_core > 0 else np.array([], dtype=int)
        graph[i, core] = True
        room = K - len(core)
        cand = np.flatnonzero(pool[i])
        cand = cand[~np.isin(cand, core)]
        if room <= 0 or len(cand) == 0:
            continue
        if rule == "rand":
            pick = rng.choice(cand, size=min(room, len(cand)), replace=False)
        else:
            pick = cand[np.argsort(-extra_score[i, cand])[:room]]
        graph[i, pick] = True
    np.fill_diagonal(graph, False)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 12])
    parser.add_argument("--k-add", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--pool-size", type=int, default=None, help="cap of the candidate pool per target (None = whole pool)")
    parser.add_argument("--out", default=str(IMPLEMENTATION_DIR / "outputs" / "proxy_graphs"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = IMPLEMENTATION_DIR / "outputs"
    flow, sensors = graph_proxy.load_flow()
    train = np.asfortranarray(flow[:graph_proxy.TRAIN_END])
    sigma = np.nanstd(train, axis=0)
    sigma[sigma == 0] = 1.0
    scale = sigma[None, :] / sigma[:, None]
    e8 = base / "E8 flow" / "e8_prior_output" / "e8"
    support = np.load(e8 / "pem_probabilities.npz")["probabilities"] > 0
    eta = np.load(base / "priors" / "corr_flow.npz")["eta"].astype(np.float64)
    pool = candidate_pool(support, eta, args.pool_size)
    size = pool.sum(axis=1)
    suffix = f"_M{args.pool_size}" if args.pool_size else ""
    print(f"pool size per target: mean {size.mean():.1f}, median {np.median(size):.0f}, max {size.max()}  "
          f"(pool-size cap: {args.pool_size})", flush=True)

    for h in args.horizons:
        t0 = time.time()
        psi = np.load(e8 / f"psi_h{h}.npz")["psi"].astype(np.float64)
        effect = np.abs(psi) * scale
        np.fill_diagonal(effect, 0.0)
        cache = out / f"pcscore__h{h}{suffix}.npz"
        if cache.exists():
            score = np.load(cache)["score"]
        else:
            score = predictive_scores(train, pool, h)
            np.savez_compressed(cache, score=score)
        print(f"h={h:2d} predictive scores ready ({time.time() - t0:.0f}s)", flush=True)
        for k_add in args.k_add:
            for rule in ("pc", "rand", "corr"):
                rng = np.random.default_rng(0)
                extra = score if rule == "pc" else eta
                g = assemble(effect, pool, extra, k_add, rule, rng)
                np.savez_compressed(out / f"hyb_{rule}{k_add}{suffix}__h{h}.npz", adjacency=g.astype(np.float32),
                                    sensors=sensors, train_end=graph_proxy.TRAIN_END)
                print(f"h={h:2d} hyb_{rule}{k_add}{suffix}: edges {int(g.sum())}, sources/target {g.sum(axis=1).mean():.1f}", flush=True)


if __name__ == "__main__":
    main()
