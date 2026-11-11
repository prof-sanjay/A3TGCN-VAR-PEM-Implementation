"""
E9: knockoff-filter edge selection on the forecasting relation (FDR-controlled; replaces the PEM ranking of E8).

For every target i and horizon h, on TRAINING rows only:
    y      = x_i(tau)                      detrended (time-of-day mean removed) flow
    kept   = lags 0..2 of the target itself (always conditioned on, never selected)
    cand   = x_j(tau - h) for every pool source j (the same candidate pool as the E8 hybrids)
y and the candidates are residualised on the self lags, so a source is selected only if it predicts the part of the
target's future that its own history does NOT explain (the unexplained residual dependency). Second-order Gaussian
knockoff copies of the candidates are drawn (equicorrelated construction on the shrunk correlation matrix), a lasso
path is run on [candidates, knockoffs], and the knockoff statistic W_j compares when the real source enters the path
against its own knockoff. The knockoff+ threshold at level q then gives a selected-edge set whose false discovery
rate is controlled at q: edges are ADDED from the pool and REMOVED (never selected) by one mechanism, with no top-k.

Serial dependence: rows are taken every STRIDE steps (15 min apart) and the series are detrended, following the
time-series knockoffs literature (arXiv 2112.09851). The test period is never read.

    python tools/run_knockoff_selection.py --horizon 1 --sample 500 --q 0.1 0.2 --k-add 10

Outputs:
    outputs/knockoff/ko_scores_h<h>.npz                         W matrix, selections, diagnostics
    outputs/proxy_graphs/e9_q<q*100><s>__h<h>.npz               pure knockoff graph (selected edges only)
    outputs/proxy_graphs/e9h<k><s>__h<h>.npz                    VAR core + k additions ranked by W (budget of hyb_pc<k>)
where <s> = "" for a full run and "_s<sample>" never appears: sampled rows are simply zero outside the sample,
exactly like build_pem_hybrids.py (the proxy evaluates the sampled targets only).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import lars_path

IMPLEMENTATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPLEMENTATION_DIR))
sys.path.insert(0, str(IMPLEMENTATION_DIR / "tools"))

import build_hybrid_candidates as bhc  # noqa: E402
import graph_proxy  # noqa: E402

STRIDE = 3          # one row every 15 minutes against autocorrelation
SHRINK = 0.01       # ridge shrinkage of the candidate correlation matrix
MIN_ROWS = 2000


def detrend_inplace(flow):
    """Remove the per-sensor time-of-day mean (288 slots, TRAINING rows only)."""
    tod = np.arange(flow.shape[0]) % 288
    for slot in range(288):
        rows = np.flatnonzero(tod == slot)
        mean = np.nanmean(flow[rows[rows < graph_proxy.TRAIN_END]], axis=0)
        flow[rows] -= mean


def entry_alphas(alphas, coefs):
    """Z_j = the largest lasso penalty at which feature j enters the path (0 = never)."""
    nz = coefs != 0
    first = np.argmax(nz, axis=1)
    entered = nz.any(axis=1)
    return np.where(entered, alphas[np.minimum(first, len(alphas) - 1)], 0.0)


def make_groups(sigma, thr):
    """Average-linkage clusters of the candidates at |r| >= thr (None -> every candidate its own group)."""
    m = sigma.shape[0]
    if thr is None or m < 2:
        return np.arange(m)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    dist = 1.0 - np.abs(sigma)
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(np.clip((dist + dist.T) / 2, 0, None), checks=False), method="average")
    return fcluster(link, t=1.0 - thr, criterion="distance") - 1


def knockoff_target(flow, i, cand, h, rng, group_thr=None):
    """Returns (W_group, W_individual, n_rows, gamma, n_groups) for one target, or None if too few clean rows.
    W_group is the group statistic (max entry point of the group's real members against its knockoffs), copied to
    every member; W_individual is the per-source statistic used only to break ties inside a group."""
    lags = graph_proxy.LAGS
    start = h + lags - 1
    rows = np.arange(start, graph_proxy.TRAIN_END, STRIDE)
    y = flow[rows, i]
    self_lags = np.stack([flow[rows - h - l, i] for l in range(lags)], axis=1)
    X = flow[:, cand][rows - h]
    ok = np.isfinite(y) & np.isfinite(self_lags).all(axis=1) & np.isfinite(X).all(axis=1)
    if ok.sum() < MIN_ROWS:
        return None
    y, self_lags, X = y[ok], self_lags[ok], X[ok]
    n, m = X.shape

    # residualise y and the candidates on [1, self lags]
    S = np.column_stack([np.ones(len(y)), self_lags])
    beta, *_ = np.linalg.lstsq(S, np.column_stack([y, X]), rcond=None)
    R = np.column_stack([y, X]) - S @ beta
    yres, Xres = R[:, 0], R[:, 1:]
    sd = Xres.std(axis=0)
    sd[sd == 0] = 1.0
    Xres = Xres / sd

    # second-order equicorrelated Gaussian knockoffs
    sigma = (1 - SHRINK) * (Xres.T @ Xres) / n + SHRINK * np.eye(m)
    d = np.sqrt(np.diag(sigma))
    sigma = sigma / np.outer(d, d)
    Xres = Xres / d
    # group knockoffs (Dai and Barber): S = gamma * blockdiag(Sigma), gamma = min(1, 2 * lambda_min(Sigma, blockdiag));
    # with every source in its own group this is the equicorrelated construction used before
    from scipy.linalg import eigh as geigh
    grp = make_groups(sigma, group_thr)
    blockdiag = np.zeros_like(sigma)
    for g in np.unique(grp):
        idx = np.flatnonzero(grp == g)
        blockdiag[np.ix_(idx, idx)] = sigma[np.ix_(idx, idx)]
    gamma = min(1.0, 2.0 * float(geigh(sigma, blockdiag, eigvals_only=True)[0]))
    S = gamma * blockdiag
    w_eig, U = np.linalg.eigh(sigma)
    inv = (U / w_eig) @ U.T
    mean_ko = Xres - Xres @ inv @ S
    V = 2 * S - S @ inv @ S
    v_eig, Vu = np.linalg.eigh((V + V.T) / 2)
    C = Vu * np.sqrt(np.clip(v_eig, 1e-12, None))
    Xko = mean_ko + rng.standard_normal((n, m)) @ C.T

    Xaug = np.concatenate([Xres, Xko], axis=1)
    alphas, _, coefs = lars_path(Xaug, yres, method="lasso", max_iter=4 * m)
    Z = entry_alphas(alphas, coefs)
    z, zt = Z[:m], Z[m:]
    W_ind = np.where(z == zt, 0.0, np.maximum(z, zt) * np.sign(z - zt))
    W_grp = np.zeros(m)
    for g in np.unique(grp):
        idx = np.flatnonzero(grp == g)
        zg, ztg = z[idx].max(), zt[idx].max()
        W_grp[idx] = 0.0 if zg == ztg else max(zg, ztg) * np.sign(zg - ztg)
    return W_grp, W_ind, n, gamma, len(np.unique(grp))


def knockoff_threshold(W, q):
    """knockoff+ threshold: smallest t with (1 + #{W <= -t}) / #{W >= t} <= q."""
    ts = np.sort(np.unique(np.abs(W[W != 0])))
    for t in ts:
        neg, pos = (W <= -t).sum(), (W >= t).sum()
        if pos > 0 and (1 + neg) / pos <= q:
            return t
    return np.inf


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--q", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--k-add", type=int, nargs="+", default=[10])
    parser.add_argument("--sample", type=int, default=None, help="same seeded targets as graph_proxy --sample")
    parser.add_argument("--group-corr", type=float, default=None, help="group knockoffs: cluster candidates at |r| >= this")
    parser.add_argument("--pool-size", type=int, default=None, help="cap the candidate pool per target (largest marginal |r|)")
    parser.add_argument("--tag", default="", help="suffix for the graph names, e.g. g50")
    parser.add_argument("--cap", type=int, default=25, help="at most this many sources per target in the pure knockoff graphs")
    parser.add_argument("--out", default=str(IMPLEMENTATION_DIR / "outputs" / "proxy_graphs"))
    args = parser.parse_args()

    t0 = time.time()
    base = IMPLEMENTATION_DIR / "outputs"
    ko_dir = base / "knockoff"
    ko_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    h = args.horizon

    flow, sensors = graph_proxy.load_flow()
    n_sensors = len(sensors)
    sigma_flow = np.nanstd(flow[:graph_proxy.TRAIN_END], axis=0)
    sigma_flow[sigma_flow == 0] = 1.0
    scale = sigma_flow[None, :] / sigma_flow[:, None]
    e8 = base / "E8 flow" / "e8_prior_output" / "e8"
    psi = np.load(e8 / f"psi_h{h}.npz")["psi"].astype(np.float64)
    effect = np.abs(psi) * scale
    np.fill_diagonal(effect, 0.0)
    support = np.load(e8 / "pem_probabilities.npz")["probabilities"] > 0
    eta = np.load(base / "priors" / "corr_flow.npz")["eta"].astype(np.float64)
    pool = bhc.candidate_pool(support, eta, args.pool_size)
    tag = args.tag
    print(f"pool per target: mean {pool.sum(axis=1).mean():.1f}; group threshold {args.group_corr}", flush=True)

    detrend_inplace(flow)
    print(f"flow loaded and detrended in {time.time() - t0:.0f}s", flush=True)

    targets = np.arange(n_sensors)
    if args.sample:
        targets = np.sort(np.random.default_rng(1).choice(n_sensors, size=args.sample, replace=False))

    W = np.zeros((n_sensors, n_sensors), dtype=np.float32)       # group statistic (= individual when no groups)
    Wi = np.zeros((n_sensors, n_sensors), dtype=np.float32)      # individual statistic
    s_vals, rows_used, done = np.zeros(n_sensors), np.zeros(n_sensors, dtype=np.int64), np.zeros(n_sensors, dtype=bool)
    n_groups = np.zeros(n_sensors, dtype=np.int64)
    t1 = time.time()
    for k, i in enumerate(targets):
        cand = np.flatnonzero(pool[i])
        if len(cand) == 0:
            continue
        res = knockoff_target(flow, i, cand, h, np.random.default_rng(1000 + int(i)), args.group_corr)
        if res is None:
            continue
        W[i, cand], Wi[i, cand], rows_used[i], s_vals[i], n_groups[i] = res
        done[i] = True
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(targets)} targets ({time.time() - t1:.0f}s)", flush=True)

    sel = {}
    for q in args.q:
        g = np.zeros((n_sensors, n_sensors), dtype=bool)
        for i in targets:
            if not done[i]:
                continue
            cand = np.flatnonzero(pool[i])
            wg = W[i, cand].astype(np.float64)
            # one W per group: the threshold is computed over groups (each group counted once)
            if args.group_corr is None:
                keep = wg >= knockoff_threshold(wg, q)
            else:
                uniq, first = np.unique(np.round(wg, 12), return_index=True)
                T = knockoff_threshold(wg[first], q)
                keep = np.zeros(len(cand), dtype=bool)
                for val in np.unique(np.round(wg[wg >= T], 12)):  # one representative per selected group
                    members = np.flatnonzero(np.round(wg, 12) == val)
                    keep[members[np.argmax(Wi[i, cand][members])]] = True
            picked = cand[keep]
            if len(picked) > args.cap:                             # same 25-source budget as the hybrids
                picked = picked[np.argsort(-(W[i, picked] + 1e-4 * Wi[i, picked] / (np.abs(Wi).max() + 1e-12)))[:args.cap]]
            g[i, picked] = True
        sel[q] = g
        src = g[targets].sum(axis=1)
        name = f"e9{tag}_q{int(round(q * 100))}"
        np.savez_compressed(out / f"{name}__h{h}.npz", adjacency=g.astype(np.float32), sensors=sensors,
                            train_end=graph_proxy.TRAIN_END)
        print(f"{name}: selected edges {int(g.sum())}; sources/target on the evaluated targets "
              f"mean {src.mean():.1f}, median {np.median(src):.0f}, zero for {(src == 0).mean():.0%}", flush=True)

    for k_add in args.k_add:
        score = W.astype(np.float64) + 1e-4 * Wi.astype(np.float64) / (np.abs(Wi).max() + 1e-12)   # individual W breaks group ties
        g = bhc.assemble(effect, pool, score, k_add, "pc", np.random.default_rng(0))
        g_rows = np.zeros_like(g)
        g_rows[targets] = g[targets]
        np.savez_compressed(out / f"e9{tag}h{k_add}__h{h}.npz", adjacency=g_rows.astype(np.float32), sensors=sensors,
                            train_end=graph_proxy.TRAIN_END)
        print(f"e9{tag}h{k_add}: sources/target on the evaluated targets {g_rows[targets].sum(axis=1).mean():.1f}", flush=True)

    np.savez_compressed(ko_dir / f"ko_scores{tag}_h{h}.npz", W=W, Wi=Wi, done=done, s=s_vals.astype(np.float32),
                        n_groups=n_groups, rows=rows_used, sensors=sensors, targets=targets,
                        **{f"sel_q{int(round(q * 100))}": sel[q] for q in args.q})
    sd = s_vals[done]
    print(f"saved ko_scores_h{h}.npz  | s (knockoff power, 1 = full): median {np.median(sd):.3f}, "
          f"10% {np.percentile(sd, 10):.3f} | rows/target median {np.median(rows_used[done]):.0f} "
          f"| total {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
