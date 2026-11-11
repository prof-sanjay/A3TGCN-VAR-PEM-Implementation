"""
Real spike-and-slab PEM on the FORECASTING relation (replaces the shrunk-inverse stand-in of build_hybrid_candidates.py).

For each target i (a sample of targets) and horizon h, the vector
        v = ( x_i(t+h),  x_i(t),  x_j(t) for j in the candidate pool of i )
is formed on TRAINING rows (complete rows only) and standardised. PEM (causal/pem.py, the existing objective, exact gradient,
L-BFGS-B) estimates the MAP precision matrix Omega with the prior inclusion matrix eta:
        eta[y, j] = eta[x_i(t), j] = eta_ij,   eta[j, k] = eta_jk       (flow prior; 0.5 everywhere for the uniform prior)
        eta[y, x_i(t)] = 0.5, diagonal 0
The ordering step of PEM-LBN is NOT run: direction comes from time order (future target, current sources).
Outputs per (target, pool source j):
        rho_pem  = | Omega[y, j] | / sqrt( Omega[y, y] Omega[j, j] )      PEM-regularised partial correlation (ranking / weights)
        p_slab   = posterior slab probability of Omega[y, j]               (reported; saturates near 1 for many pairs)
Resumable: a checkpoint is written every few targets.

    python tools/run_pem_forecast_relation.py --horizon 1 --prior flow --sample 500 --workers 8
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

IMPLEMENTATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPLEMENTATION_DIR))
sys.path.insert(0, str(IMPLEMENTATION_DIR / "tools"))

TRAIN_END = 84096
NU0, NU1, TAU = 0.01, 0.2, 0.01           # the settings of the E8 prior study

def fit_target(i, train_path, pool_path, eta_path, prior, h):
    from causal.pem import PEM

    train = np.load(train_path, mmap_mode="r")
    pool = np.load(pool_path, mmap_mode="r")
    eta_m = np.load(eta_path, mmap_mode="r")
    cols = np.flatnonzero(pool[i])
    cols = cols[cols != i]
    if len(cols) == 0:
        return i, cols, np.zeros(0), np.zeros(0), 0, 0.0
    t0 = time.time()
    sub = np.asarray(train[:, np.concatenate([[i], cols])], dtype=np.float64)
    v = np.column_stack([np.asarray(train[h:TRAIN_END, i], dtype=np.float64), sub[:TRAIN_END - h]])
    v = v[np.isfinite(v).all(axis=1)]
    if len(v) < 1000:
        return i, cols, np.zeros(len(cols)), np.zeros(len(cols)), len(v), 0.0
    v = (v - v.mean(axis=0)) / np.where(v.std(axis=0) == 0, 1.0, v.std(axis=0))
    p = v.shape[1]                                          # [y, x_i(t), pool...]
    eta = np.full((p, p), 0.5)
    if prior == "flow":
        e_ij = np.asarray(eta_m[i, cols])
        eta[0, 2:] = e_ij
        eta[2:, 0] = e_ij
        eta[1, 2:] = e_ij
        eta[2:, 1] = e_ij
        eta[2:, 2:] = np.asarray(eta_m[np.ix_(cols, cols)])
    np.fill_diagonal(eta, 0.0)
    model = PEM(eta=eta, nu0=NU0, nu1=NU1, tau=TAU, threshold=0.5, max_iter=20000, tolerance=1e-6, optimizer="L-BFGS-B")
    samples = model._validate_samples(v)
    cov = model._make_positive_definite(model._sample_covariance(samples))
    omega = model._estimate_precision(cov, eta, samples.shape[0])
    d = np.sqrt(np.abs(np.diag(omega)))
    rho = np.abs(omega[0, 2:]) / (d[0] * d[2:])
    slab = np.array([model._posterior_inclusion_probability(omega[0, 2 + k], eta[0, 2 + k]) for k in range(len(cols))])
    return i, cols, rho, slab, len(v), time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--prior", choices=["flow", "uniform"], required=True)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only the first N sampled targets (timing test)")
    parser.add_argument("--out", default=str(IMPLEMENTATION_DIR / "outputs" / "pem_forecast"))
    args = parser.parse_args()

    import graph_proxy
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = IMPLEMENTATION_DIR / "outputs"

    # training rows as a float32 memory map shared by the workers
    train_path = base / "proxy_graphs" / "train_f32.npy"
    if not train_path.exists():
        flow, sensors = graph_proxy.load_flow()
        np.save(train_path, np.asarray(flow[:TRAIN_END], dtype=np.float32))
        del flow
    sensors = np.load(base / "var_train" / "lag3_alpha1e+06" / "sensors.npy")
    pool = np.load(base / "E8 flow" / "e8_prior_output" / "e8" / "pem_probabilities.npz")["probabilities"] > 0
    np.fill_diagonal(pool, False)
    pool_path, eta_path = base / "proxy_graphs" / "pool.npy", base / "proxy_graphs" / "eta_f64.npy"
    np.save(pool_path, pool)
    np.save(eta_path, np.load(base / "priors" / "corr_flow.npz")["eta"].astype(np.float64))
    targets = np.sort(np.random.default_rng(1).choice(len(sensors), size=args.sample, replace=False))   # same targets as the proxy
    if args.limit:
        targets = targets[: args.limit]

    tag = f"{args.prior}_h{args.horizon}"
    ckpt = out / f"checkpoint_{tag}.pkl"
    done = pickle.load(open(ckpt, "rb")) if ckpt.exists() else {}
    todo = [int(i) for i in targets if int(i) not in done]
    # largest pools first: the longest jobs start early and the workers stay balanced
    todo.sort(key=lambda i: -pool[i].sum())
    print(f"{tag}: {len(done)} done, {len(todo)} to do, {args.workers} workers", flush=True)
    t0 = time.time()
    from joblib import Parallel, delayed

    chunk = max(args.workers * 4, 8)
    for start in range(0, len(todo), chunk):
        batch = todo[start:start + chunk]
        results = Parallel(n_jobs=args.workers)(
            delayed(fit_target)(i, str(train_path), str(pool_path), str(eta_path), args.prior, args.horizon) for i in batch)
        for i, cols, rho, slab, rows, secs in results:
            done[i] = (cols, rho, slab, rows, secs)
        pickle.dump(done, open(ckpt, "wb"))
        print(f"  {len(done)}/{len(targets)} targets  ({time.time() - t0:.0f}s elapsed; slowest of the batch "
              f"{max(r[5] for r in results):.0f}s)", flush=True)

    n_all = len(sensors)
    rho_m = np.zeros((n_all, n_all), dtype=np.float32)
    slab_m = np.zeros((n_all, n_all), dtype=np.float32)
    for i, (cols, rho, slab, _, _) in done.items():
        rho_m[i, cols], slab_m[i, cols] = rho, slab
    np.savez_compressed(out / f"pem_scores_{tag}.npz", rho=rho_m, slab=slab_m, targets=targets, sensors=sensors,
                        prior=args.prior, nu0=NU0, nu1=NU1, tau=TAU)
    secs = np.array([v[4] for v in done.values()])
    print(f"saved pem_scores_{tag}.npz  total {(time.time() - t0) / 60:.1f} min; per-target mean {secs.mean():.0f}s, max {secs.max():.0f}s", flush=True)


if __name__ == "__main__":
    main()
