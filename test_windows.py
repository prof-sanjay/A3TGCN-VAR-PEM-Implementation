"""
Validation of data/windows.py.

Checks:
    1. Synthetic data: index-based windows are IDENTICAL to
       create_sequences + filter_valid_sequences (main.py logic)
       for several seq_len / pre_len settings, including NaNs at
       the split edges.
    2. Batching gives the same batches, in the same order, as
       iterate_batches in main.py.
    3. Real LargeST-GBA subset: identical samples to the existing
       pipeline.
    4. Full LargeST-GBA 2021, 80/10/10 split: valid-window counts,
       no window crossing a split boundary, normalization does not
       change window validity, and the traffic array is unchanged.

Run from the implementation directory:

    python test_windows.py
"""

import hashlib
from pathlib import Path

import numpy as np

from data.loader import load_gba_dataset
from data.preprocessing import (
    create_sequences,
    fit_max_scaler,
    normalize_data,
)
from data.splitting import chronological_split
from data.windows import (
    gather_windows,
    iterate_window_batches,
    missing_value_report,
    valid_window_starts,
)


DATASET_DIR = Path(__file__).parent / "datasets" / "LargeST GBA"
TRAFFIC_PATH = DATASET_DIR / "gba_his_raw_2021.h5"
ADJACENCY_PATH = DATASET_DIR / "gba_rn_adj.npy"

SEQ_LEN = 12
PRE_LEN = 1
BATCH_SIZE = 8

# Measured independently in Phase 0 (direct HDF5 scan).
EXPECTED_VALID_WINDOWS = {
    "train": 52_471,
    "val": 10_139,
    "test": 4_750,
}


# ============================================================
# Reference implementation (copied from main.py, unchanged)
# ============================================================

def filter_valid_sequences(X, y):

    X_valid = np.all(np.isfinite(X), axis=(1, 2))
    y_valid = np.all(np.isfinite(y), axis=(1, 2))

    valid_mask = X_valid & y_valid

    return X[valid_mask], y[valid_mask]


def iterate_batches(X, y, batch_size):

    for start in range(0, len(X), batch_size):

        end = min(start + batch_size, len(X))

        yield X[start:end], y[start:end]


def reference_windows(data, seq_len, pre_len):

    X, y = create_sequences(data, seq_len, pre_len)

    return filter_valid_sequences(X, y)


def assert_identical_to_reference(data, seq_len, pre_len, label):

    X_ref, y_ref = reference_windows(data, seq_len, pre_len)

    starts = valid_window_starts(data, seq_len, pre_len)

    X_new, y_new = gather_windows(data, starts, seq_len, pre_len)

    assert X_new.shape == X_ref.shape, (label, X_new.shape, X_ref.shape)
    assert y_new.shape == y_ref.shape, (label, y_new.shape, y_ref.shape)
    assert X_new.dtype == X_ref.dtype == np.float32
    assert np.array_equal(X_new, X_ref), label
    assert np.array_equal(y_new, y_ref), label

    print(
        f"  [PASS] {label}: {len(starts)} valid windows identical "
        f"to reference"
    )

    return X_ref, y_ref, starts


def array_digest(array):

    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()
    ).hexdigest()


# ============================================================
# 1-2. Synthetic tests
# ============================================================

def test_synthetic():

    print("\n1. SYNTHETIC EQUIVALENCE")

    rng = np.random.default_rng(42)

    data = rng.normal(
        loc=200.0,
        scale=50.0,
        size=(400, 7),
    ).astype(np.float32)

    # Scattered NaNs, plus NaNs at the first and last timestep.
    data[rng.random(data.shape) < 0.01] = np.nan
    data[0, 3] = np.nan
    data[-1, 5] = np.nan
    data[150:160, 2] = np.nan

    before = array_digest(data)

    for seq_len, pre_len in [(12, 1), (12, 3), (1, 1), (5, 12)]:
        assert_identical_to_reference(
            data,
            seq_len,
            pre_len,
            f"seq_len={seq_len}, pre_len={pre_len}",
        )

    # Window entirely invalid.
    all_nan = np.full((20, 3), np.nan, dtype=np.float32)
    assert len(valid_window_starts(all_nan, 12, 1)) == 0
    print("  [PASS] all-NaN data gives zero valid windows")

    # Too short.
    try:
        valid_window_starts(data[:12], 12, 1)
        raise AssertionError("short data was accepted")
    except ValueError:
        print("  [PASS] too-short data raises ValueError")

    assert array_digest(data) == before
    print("  [PASS] input array unchanged")

    print("\n2. BATCH EQUIVALENCE")

    X_ref, y_ref = reference_windows(data, SEQ_LEN, PRE_LEN)
    starts = valid_window_starts(data, SEQ_LEN, PRE_LEN)

    reference_batches = list(iterate_batches(X_ref, y_ref, BATCH_SIZE))
    new_batches = list(
        iterate_window_batches(data, starts, SEQ_LEN, PRE_LEN, BATCH_SIZE)
    )

    assert len(reference_batches) == len(new_batches)

    for (X_a, y_a), (X_b, y_b) in zip(reference_batches, new_batches):
        assert np.array_equal(X_a, X_b)
        assert np.array_equal(y_a, y_b)

    print(
        f"  [PASS] {len(new_batches)} batches identical in content "
        f"and order (last batch size {len(new_batches[-1][0])})"
    )


# ============================================================
# 3. Real-data subset equivalence
# ============================================================

def test_real_subset(traffic):

    print("\n3. REAL-DATA SUBSET EQUIVALENCE (first 20,000 x 150)")

    subset = np.ascontiguousarray(traffic[:20_000, :150])

    assert_identical_to_reference(
        subset,
        SEQ_LEN,
        PRE_LEN,
        "raw subset",
    )

    max_value = fit_max_scaler(subset)

    assert_identical_to_reference(
        normalize_data(subset, max_value),
        SEQ_LEN,
        PRE_LEN,
        "normalized subset",
    )


# ============================================================
# 4. Full dataset
# ============================================================

def test_full_dataset(traffic):

    print("\n4. FULL LargeST-GBA 2021 (80/10/10)")

    before = array_digest(traffic)

    splits = dict(
        zip(
            ["train", "val", "test"],
            chronological_split(traffic, 0.8, 0.1),
        )
    )

    print(
        "  Split sizes:",
        {name: len(split) for name, split in splits.items()},
    )

    assert sum(len(split) for split in splits.values()) == len(traffic)

    starts_by_split = {}

    for name, split in splits.items():

        report = missing_value_report(split, SEQ_LEN, PRE_LEN)

        starts = valid_window_starts(split, SEQ_LEN, PRE_LEN)
        starts_by_split[name] = starts

        print(
            f"  {name:5s}: missing {report['missing_values']:6d} "
            f"({report['missing_percentage']:.4f}%), "
            f"timesteps with NaN {report['timesteps_with_missing']:6d}, "
            f"valid windows {report['valid_windows']:6d} / "
            f"{report['total_windows']} "
            f"({report['valid_window_percentage']:.1f}%)"
        )

        assert report["valid_windows"] == len(starts)
        assert len(starts) == EXPECTED_VALID_WINDOWS[name], (
            name,
            len(starts),
            EXPECTED_VALID_WINDOWS[name],
        )

        # No window may cross the end of its split.
        assert starts.max() + SEQ_LEN + PRE_LEN <= len(split)

        # Every gathered window really is finite (spot check).
        sample = starts[:: max(1, len(starts) // 500)]
        X, y = gather_windows(split, sample, SEQ_LEN, PRE_LEN)
        assert np.all(np.isfinite(X)) and np.all(np.isfinite(y))

    print("  [PASS] valid-window counts match Phase 0 measurement")
    print("  [PASS] no window crosses a split boundary")
    print("  [PASS] gathered windows are all finite")

    # Normalization must not change validity (checked on the
    # validation split to limit memory use).
    max_value = fit_max_scaler(splits["train"])
    val_norm = normalize_data(splits["val"], max_value)

    assert np.array_equal(
        valid_window_starts(val_norm, SEQ_LEN, PRE_LEN),
        starts_by_split["val"],
    )
    print(
        f"  [PASS] normalization (train max = {max_value}) "
        "does not change window validity"
    )

    assert array_digest(traffic) == before
    print("  [PASS] traffic array unchanged")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("WINDOW MODULE VALIDATION")
    print("=" * 70)

    test_synthetic()

    print("\nLoading LargeST-GBA 2021 (float32, NaNs preserved)...")

    traffic, _ = load_gba_dataset(
        TRAFFIC_PATH,
        ADJACENCY_PATH,
        num_timesteps=105_120,
        num_nodes=2_352,
    )

    print("  Traffic shape:", traffic.shape, traffic.dtype)

    test_real_subset(traffic)
    test_full_dataset(traffic)

    print("\n" + "=" * 70)
    print("ALL WINDOW TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
