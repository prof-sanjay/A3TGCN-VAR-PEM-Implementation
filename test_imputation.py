"""
Validation of data/imputation.py (gap-band filling).

Checks:
    1. Training-mode bands: short gap -> exact linear interpolation,
       medium gap -> neighbour regression where available else
       interpolation, long gap -> neighbour regression only, gaps at
       the split edges are not interpolated.
    2. Observed values and the input array are never changed.
    3. Validation/test mode never looks ahead: the value filled at
       time t is identical when every row after t is removed.
    4. Neighbour regression recovers a known linear relation and
       uses only observed neighbour values.
    5. Real LargeST-GBA (after the duplicate-sensor filter): fill
       counts per band and split, runtime.

Run from the implementation directory:

    python test_imputation.py
"""

import time
from pathlib import Path

import numpy as np

from data.imputation import (
    fill_split,
    fit_neighbour_models,
    gap_length_counts,
    NeighbourModel,
)
from data.loader import load_gba_dataset
from data.sensor_filter import duplicate_filter, select_sensors
from data.splitting import chronological_split


DATASET_DIR = Path(__file__).parent / "datasets" / "LargeST GBA"


def synthetic_case():
    """
    Sensor 0 = 2 * sensor 1 + 3 * sensor 2 + 10 (exact), plus gaps.
    """

    rng = np.random.default_rng(0)

    n = 400

    data = np.empty((n, 3), dtype=np.float32)
    data[:, 1] = rng.uniform(50, 100, n)
    data[:, 2] = rng.uniform(20, 40, n)
    data[:, 0] = 2 * data[:, 1] + 3 * data[:, 2] + 10

    adjacency = np.ones((3, 3), dtype=np.float32)

    return data, adjacency


def test_training_mode():

    print("\n1-2. TRAINING-MODE BANDS")

    data, adjacency = synthetic_case()

    models = fit_neighbour_models(
        data, adjacency, num_neighbours=2, min_fit_rows=10
    )

    model = models[0]

    assert np.allclose(model.coefficients, [2.0, 3.0], atol=1e-3)
    assert abs(model.intercept - 10.0) < 1e-1
    print(f"  [PASS] neighbour model recovers 2, 3, 10: "
          f"{np.round(model.coefficients, 4)}, {model.intercept:.4f}")

    gappy = data.copy()

    gappy[50:54, 0] = np.nan          # short (4)
    gappy[100:115, 0] = np.nan        # medium (15)
    gappy[105:108, 1] = np.nan        # neighbour missing inside it
    gappy[200:240, 0] = np.nan        # long (40)
    gappy[210:215, 2] = np.nan        # neighbour missing inside it
    gappy[0:3, 0] = np.nan            # short gap at the start edge

    before = gappy.copy()

    filled, counts = fill_split(gappy, models, look_ahead=True)

    assert np.array_equal(gappy, before, equal_nan=True)
    observed = np.isfinite(before)
    assert np.array_equal(filled[observed], before[observed])
    print("  [PASS] input unchanged, observed values unchanged")

    expected = np.interp(
        np.arange(50, 54), [49, 54], [data[49, 0], data[54, 0]]
    )
    assert np.allclose(filled[50:54, 0], expected)
    print("  [PASS] short gap = exact linear interpolation")

    assert np.all(np.isnan(filled[0:3, 0]))
    print("  [PASS] short gap at split start not interpolated (NaN)")

    neighbour_rows = [r for r in range(100, 115) if not 105 <= r < 108]
    assert np.allclose(filled[neighbour_rows, 0],
                       data[neighbour_rows, 0], atol=0.05)
    interp = np.interp(
        np.arange(105, 108), [99, 115], [data[99, 0], data[115, 0]]
    )
    assert np.allclose(filled[105:108, 0], interp)
    print("  [PASS] medium gap: neighbour where available, "
          "interpolation where a neighbour is missing")

    long_ok = [r for r in range(200, 240) if not 210 <= r < 215]
    assert np.allclose(filled[long_ok, 0], data[long_ok, 0], atol=0.05)
    assert np.all(np.isnan(filled[210:215, 0]))
    print("  [PASS] long gap: neighbour only; NaN where a neighbour "
          "is missing (no interpolation)")

    # The neighbour's own 3-step gap is a short gap of sensor 1,
    # so it is interpolated from sensor 1's own observed values.
    assert np.allclose(
        filled[105:108, 1],
        np.interp(np.arange(105, 108), [104, 108],
                  [data[104, 1], data[108, 1]]),
    )
    print("  [PASS] neighbour's own short gap interpolated from its "
          "own values (sensor 0's fill used only observed neighbours)")
    print(f"  counts: {counts}")


def test_no_look_ahead():

    print("\n3. VALIDATION/TEST MODE: NO LOOK-AHEAD")

    rng = np.random.default_rng(1)

    data, adjacency = synthetic_case()

    models = fit_neighbour_models(
        data, adjacency, num_neighbours=2, min_fit_rows=10
    )

    gappy = data.copy()
    gappy[rng.random(gappy.shape) < 0.03] = np.nan
    gappy[150:190, 0] = np.nan
    gappy[300:330, 1] = np.nan
    gappy[310:312, 2] = np.nan

    filled_full, counts = fill_split(gappy, models, look_ahead=False)

    mismatches = 0

    for t in range(len(gappy)):

        prefix_filled, _ = fill_split(
            gappy[: t + 1], models, look_ahead=False
        )

        if not np.array_equal(
            prefix_filled[t], filled_full[t], equal_nan=True
        ):
            mismatches += 1

    assert mismatches == 0, mismatches
    print(f"  [PASS] all {len(gappy)} timesteps: filled value identical "
          f"with every later row removed")

    observed = np.isfinite(gappy)
    assert np.array_equal(filled_full[observed], gappy[observed])
    assert counts["interpolation"] == 0
    print(f"  [PASS] no interpolation used; counts: {counts}")


def test_real_data():

    print("\n5. REAL LargeST-GBA 2021 (duplicate filter applied)")

    traffic, adjacency = load_gba_dataset(
        DATASET_DIR / "gba_his_raw_2021.h5",
        DATASET_DIR / "gba_rn_adj.npy",
        num_timesteps=105_120,
        num_nodes=2_352,
    )

    keep, _ = duplicate_filter(traffic[:84_096])
    traffic, adjacency = select_sensors(traffic, adjacency, keep)

    train, val, test = chronological_split(traffic, 0.8, 0.1)

    start = time.perf_counter()
    models = fit_neighbour_models(train, adjacency)
    print(f"  neighbour models: {len(models)} / {train.shape[1]} sensors, "
          f"fit in {time.perf_counter() - start:.0f}s")

    for name, split, look_ahead in [
        ("train", train, True),
        ("val", val, False),
        ("test", test, False),
    ]:

        start = time.perf_counter()

        filled, counts = fill_split(split, models, look_ahead)

        observed = np.isfinite(split)
        assert np.array_equal(filled[observed], split[observed])

        print(f"  {name:5s}: gaps {gap_length_counts(split)}")
        print(f"         filled {counts} "
              f"({time.perf_counter() - start:.0f}s)")

        del filled


def main():

    print("=" * 70)
    print("GAP-BAND IMPUTATION VALIDATION")
    print("=" * 70)

    test_training_mode()
    test_no_look_ahead()
    test_real_data()

    print("\n" + "=" * 70)
    print("ALL IMPUTATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
