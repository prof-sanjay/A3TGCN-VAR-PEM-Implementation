"""
Build a correlation-based PEM prior from the VAR training residuals.

    python build_corr_prior.py --method pearson
    python build_corr_prior.py --method spearman
    python build_corr_prior.py --method partial [--shrinkage 0.05]

Writes outputs/priors/corr_<method>.npz (eta, sensors, train_end, ...) and a
.json summary. Training rows only; missing residuals are never filled.
Use the file with  run_local_pem.py --prior-file ... --tag ...
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from causal.candidates import load_candidates, unit_options
from causal.causal_units import build_causal_units
from causal.corr_prior import METHODS, correlation_matrix, percentile_eta
from causal.flow_correlation import cached_flow_correlation
from causal.var_coefficient_pem import correlation_to_eta
from config.config import VAR_RUN, ExperimentConfig

HERE = Path(__file__).resolve().parent


def candidate_pairs(units, n):
    """Boolean n x n matrix: the pairs that share a causal unit."""

    mask = np.zeros((n, n), dtype=bool)
    for unit in units:
        nodes = np.asarray(unit.nodes, dtype=np.int64)
        mask[np.ix_(nodes, nodes)] = True
    np.fill_diagonal(mask, False)
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--method", choices=METHODS, default="pearson",
                        help="residual source only; the flow source is always Pearson")
    parser.add_argument("--source", choices=("residual", "flow"), default="residual",
                        help="residual: correlation of the VAR residuals (percentile mapping); flow: Pearson correlation of the "
                             "ORIGINAL detrended training flow, eta = low + (high - low) |r| on the candidate pairs")
    parser.add_argument("--var-run", default=VAR_RUN)
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="flow source only: shuffle eta among the candidate pairs (control; keeps the distribution and symmetry)")
    parser.add_argument("--low", type=float, default=0.05)
    parser.add_argument("--high", type=float, default=0.95)
    parser.add_argument("--shrinkage", type=float, default=0.05, help="partial correlation only")
    parser.add_argument("--out-dir", default=str(HERE / "outputs" / "priors"))
    args = parser.parse_args()

    start = time.time()
    var_dir = HERE / "outputs" / "var_train" / args.var_run
    metadata = json.loads((var_dir / "metadata.json").read_text(encoding="utf-8"))
    sensors = np.load(var_dir / "sensors.npy")
    residuals = np.load(var_dir / "residuals.npy")                     # training rows only
    if residuals.shape != (metadata["train_end"], len(sensors)):
        raise ValueError("residuals do not match the VAR metadata.")

    config = ExperimentConfig()
    adjacency = np.asarray(np.load(config.adjacency_path, mmap_mode="r")[np.ix_(sensors, sensors)], dtype=np.float32)
    candidates = metadata.get("candidates", "road")
    unit_adjacency = load_candidates(candidates, sensors, metadata["train_end"], adjacency)
    units = build_causal_units(unit_adjacency, **unit_options(candidates))
    pairs = candidate_pairs(units, len(sensors))

    if args.source == "flow":
        corr, corr_file = cached_flow_correlation(True, sensors, int(metadata["train_end"]), out_dir_cache := Path(args.out_dir))
        eta_full = correlation_to_eta(corr, args.low, args.high)
        eta = np.where(pairs, eta_full, 0.5)
        np.fill_diagonal(eta, 0.0)
        args.method = "flow"
        if args.shuffle_seed is not None:
            upper = np.triu(pairs, 1)
            values = eta[upper].copy()
            np.random.default_rng(args.shuffle_seed).shuffle(values)
            eta = np.full_like(eta, 0.5)
            eta[upper] = values
            eta = np.where(upper, eta, np.where(upper.T, eta.T, 0.5))
            np.fill_diagonal(eta, 0.0)
            args.method = f"flow_shuffle{args.shuffle_seed}"
    else:
        corr = correlation_matrix(residuals, args.method, args.shrinkage)
        eta = percentile_eta(corr, pairs, args.low, args.high)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"corr_{args.method}.npz"
    np.savez_compressed(path, eta=eta.astype(np.float32), sensors=sensors, train_end=metadata["train_end"],
                        method=args.method, low=args.low, high=args.high, shrinkage=args.shrinkage,
                        var_run=args.var_run)

    values = eta[pairs]
    summary = {"method": args.method, "var_run": args.var_run, "train_end": int(metadata["train_end"]),
               "num_sensors": int(len(sensors)), "candidate_pair_entries": int(pairs.sum()),
               "mapping": (f"eta = {args.low} + ({args.high} - {args.low}) |r| of the detrended training flow" if args.source == "flow" else f"percentile of |r| among each row's candidate pairs -> [{args.low}, {args.high}], max over the two directions"),
               "shrinkage": args.shrinkage if args.method == "partial" else None,
               "eta_mean": float(values.mean()), "eta_min": float(values.min()), "eta_max": float(values.max()),
               "eta_symmetric": bool(np.allclose(eta, eta.T)), "seconds": round(time.time() - start, 1)}
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
