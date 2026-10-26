"""
Build data-driven candidate links for E6 (causal/candidates.py).

For every sensor: the K sensors whose 5-minute changes are most
correlated with its own on the TRAINING rows (no road information).
Saved as outputs/candidates/corr_k<K>.npz; then

    python run_var.py --candidates corr_k25 --save-residuals
    python run_local_pem.py --var-run <that VAR run> ...
    python run_local_lingam.py --var-run <that VAR run> ...

PEM and LiNGAM read the candidate name from the VAR run.

Run from the implementation directory:

    python build_candidates.py --k 25
"""

import argparse
import json

import numpy as np

from causal.candidates import correlation_neighbours, difference_correlation, save_candidates
from config.config import ExperimentConfig
from data.loader import load_gba_dataset
from data.sensor_filter import cached_duplicate_filter, select_sensors
from data.splitting import chronological_split
from pathlib import Path


IMPLEMENTATION_DIR = Path(__file__).resolve().parent


def main():

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--k", type=int, default=25)
    args = parser.parse_args()

    config = ExperimentConfig()
    traffic, road = load_gba_dataset(config.traffic_path, config.adjacency_path,
                                     num_timesteps=config.num_timesteps,
                                     num_nodes=config.num_nodes)

    train, val, _ = chronological_split(traffic, config.train_rate, config.val_rate)
    train_end = len(train)

    sensors, _ = cached_duplicate_filter(train, IMPLEMENTATION_DIR / "outputs" / "sensor_filter")
    train, road = select_sensors(traffic[:train_end], road, sensors)
    del traffic, val

    print(f"Training rows {train.shape}; correlating first differences...")
    corr = difference_correlation(train)
    del train

    candidates = correlation_neighbours(corr, args.k)

    # Comparison with the road graph (diagnostic only).
    road_links = np.asarray(road) != 0
    np.fill_diagonal(road_links, False)
    either = road_links | road_links.T
    chosen = candidates != 0

    stats = {
        "k": args.k,
        "candidate_links": int(chosen.sum()),
        "road_links": int(road_links.sum()),
        "share_of_candidates_that_are_road_links": float(either[chosen].mean()),
        "share_of_road_links_among_candidates": float(chosen[road_links].mean()),
        "reciprocal_pairs": int(np.count_nonzero(np.triu(chosen & chosen.T, 1))),
        "abs_corr_of_chosen_links": {
            "min": float(np.abs(corr[chosen]).min()),
            "median": float(np.median(np.abs(corr[chosen]))),
        },
        "out_degree_max": int(chosen.sum(axis=0).max()),
    }

    name = f"corr_k{args.k}"
    path = save_candidates(name, candidates, sensors, train_end, args.k)
    path.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
