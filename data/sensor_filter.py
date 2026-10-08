"""
Data-quality filter: sensors with duplicated (copied) series.

LargeST-GBA contains groups of sensors whose training series are
copies of each other (identical values on >= 90 % of the jointly
observed timesteps, sometimes 100 %, sometimes with a few altered
values or NaN positions), although
the sensors lie on different freeways tens of kilometres apart.
Such series cannot be independent measurements. They make local
VAR regressions singular and give PEM degenerate covariances
(fake edges between copies).

Ablation decision D9 (option c): every sensor that belongs to such
a group is excluded from E0, E1 and E2 alike.

The groups are identified on the TRAINING split only, so the
validation and test periods do not influence the sensor set.

Traffic values are never modified; sensors are only selected.
"""

import numpy as np


def find_duplicate_sensor_groups(
    train_data: np.ndarray,
    min_overlap: float = 0.5,
    min_identical_fraction: float = 0.9,
    candidate_correlation: float = 0.95,
    copy_correlation: float = 0.999,
) -> list[list[int]]:
    """
    Groups of sensors that are copies of each other in the training
    split.

    Two sensors are linked when they are jointly observed on at least
    ``min_overlap`` of the training rows and report EXACTLY the same
    value on at least ``min_identical_fraction`` of those rows.
    Independent loop detectors cannot report identical integer flows
    (about 200 vehicles per 5 min) in 90 % of intervals, so such
    pairs are copies, possibly with a few altered values or NaN
    positions. Exact copies (100 %) are a special case.

    Two sensors are also linked when their Pearson correlation on the
    rows where all sensors are observed is >= ``copy_correlation``
    (0.999). This catches copies that are rescaled (e.g. one sensor =
    0.75 x another) or perturbed by +-1-3 vehicles, which the
    identical-value rule cannot see; independent detectors, even
    adjacent ones, stay below this level (499 road-neighbour pairs
    reach 0.99, only 16 pairs of any kind reach 0.999).

    Linked sensors are merged into groups (connected components).

    Candidates are pairs with Pearson correlation >=
    ``candidate_correlation`` on the rows where all sensors are
    observed, plus all pairs that are exactly equal on those rows;
    every candidate is then verified on its own jointly observed rows.

    Parameters
    ----------
    train_data:
        Training traffic with shape [time, nodes]. NaNs preserved.

    min_overlap:
        Minimum fraction of training rows on which both sensors of
        a pair must be observed.

    Returns
    -------
    groups:
        List of groups (each a sorted list of >= 2 sensor indices),
        ordered by their smallest sensor index.
    """

    if train_data.ndim != 2:
        raise ValueError(
            f"Expected data with shape [time, nodes], "
            f"got {train_data.shape}"
        )

    num_timesteps, num_sensors = train_data.shape

    observed = np.isfinite(train_data)

    # ---------------------------------------------------------
    # Candidate pairs, computed on the rows where ALL sensors are
    # observed (a pre-filter only; nothing is decided here):
    #   (a) exactly equal on those rows (catches constant series),
    #   (b) correlation >= candidate_correlation.
    # ---------------------------------------------------------

    all_observed = np.all(observed, axis=1)

    # One float32 copy of the complete rows (~0.65 GB for GBA),
    # standardized in place; float32 is sufficient for a pre-filter.
    complete = np.array(
        train_data[all_observed],
        dtype=np.float32,
        copy=True,
    )

    candidates: set[tuple[int, int]] = set()

    buckets: dict[bytes, list[int]] = {}

    for sensor in range(num_sensors):
        buckets.setdefault(complete[:, sensor].tobytes(), []).append(sensor)

    for members in buckets.values():
        for a_index, a in enumerate(members):
            for b in members[a_index + 1:]:
                candidates.add((a, b))

    mean = complete.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = complete.std(axis=0, dtype=np.float64).astype(np.float32)

    complete -= mean
    complete /= np.where(std > 0, std, 1.0)

    correlation = (complete.T @ complete) / max(1, len(complete))

    correlation[:, std == 0] = 0.0
    correlation[std == 0, :] = 0.0

    del complete

    rows, cols = np.nonzero(
        np.triu(correlation >= candidate_correlation, k=1)
    )

    candidates.update(zip(rows.tolist(), cols.tolist()))

    copy_rows, copy_cols = np.nonzero(
        np.triu(correlation >= copy_correlation, k=1)
    )

    # ---------------------------------------------------------
    # Verify every candidate on its own jointly observed rows.
    # ---------------------------------------------------------

    parent = list(range(num_sensors))

    def find(sensor):

        while parent[sensor] != sensor:
            parent[sensor] = parent[parent[sensor]]
            sensor = parent[sensor]

        return sensor

    # Rescaled / perturbed copies (correlation rule).
    for sensor_a, sensor_b in zip(copy_rows.tolist(), copy_cols.tolist()):
        parent[find(sensor_b)] = find(sensor_a)

    pairs = np.array(sorted(candidates), dtype=np.int64).reshape(-1, 2)

    # Verified in chunks of pairs (vectorized; same test per pair).
    for begin in range(0, len(pairs), 256):

        a = pairs[begin:begin + 256, 0]
        b = pairs[begin:begin + 256, 1]

        both = observed[:, a] & observed[:, b]

        joint = both.sum(axis=0)

        same = (
            (train_data[:, a] == train_data[:, b]) & both
        ).sum(axis=0)

        linked = (
            (joint >= min_overlap * num_timesteps)
            & (same >= min_identical_fraction * joint)
        )

        for sensor_a, sensor_b in zip(a[linked], b[linked]):
            parent[find(int(sensor_b))] = find(int(sensor_a))

    components: dict[int, list[int]] = {}

    for sensor in range(num_sensors):
        components.setdefault(find(sensor), []).append(sensor)

    groups = [
        sorted(members)
        for members in components.values()
        if len(members) > 1
    ]

    return sorted(groups, key=lambda group: group[0])


def duplicate_filter(
    train_data: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Sensors to KEEP after removing every duplicate-group member.

    Returns
    -------
    keep:
        Ascending int64 array of retained sensor indices.

    report:
        JSON-serialisable description of the filter.
    """

    groups = find_duplicate_sensor_groups(train_data)

    excluded = np.array(
        sorted({sensor for group in groups for sensor in group}),
        dtype=np.int64,
    )

    keep = np.setdiff1d(
        np.arange(train_data.shape[1], dtype=np.int64),
        excluded,
    )

    report = {
        "rule": "exclude every sensor that reports exactly the same "
                "value as another sensor on >= 90% of their jointly "
                "observed training rows (joint observation >= 50% of "
                "training rows), or whose correlation with another "
                "sensor on fully observed training rows is >= 0.999",
        "num_sensors_before": int(train_data.shape[1]),
        "num_groups": len(groups),
        "num_excluded": int(len(excluded)),
        "num_sensors_after": int(len(keep)),
        "group_sizes": sorted(
            (len(group) for group in groups),
            reverse=True,
        ),
        "excluded_sensors": excluded.tolist(),
        "groups": groups,
    }

    return keep, report


def cached_duplicate_filter(
    train_data: np.ndarray,
    cache_dir,
) -> tuple[np.ndarray, dict]:
    """
    duplicate_filter() with an on-disk cache (the full computation
    takes ~10 min on the GBA training split).

    The cache key is the SHA-256 of the training array (shape, dtype
    and bytes, NaNs included) together with the filter settings, so
    any change of data, split or settings recomputes the filter.
    """

    import hashlib
    import inspect
    import json
    from pathlib import Path

    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(
            find_duplicate_sensor_groups
        ).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }

    digest = hashlib.sha256()
    digest.update(str(train_data.shape).encode())
    digest.update(str(train_data.dtype).encode())
    digest.update(json.dumps(defaults, sort_keys=True).encode())

    contiguous = np.ascontiguousarray(train_data)

    for start in range(0, len(contiguous), 8192):
        digest.update(contiguous[start:start + 8192].tobytes())

    key = digest.hexdigest()[:16]

    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"sensor_filter_{key}.json"

    if cache_file.exists():

        with open(cache_file, encoding="utf-8") as f:
            report = json.load(f)

        keep = np.array(report["kept_sensors"], dtype=np.int64)

        report["cache"] = f"loaded {cache_file.name}"

        return keep, report

    keep, report = duplicate_filter(train_data)

    report["kept_sensors"] = keep.tolist()
    report["settings"] = defaults

    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(report, f)

    report["cache"] = f"computed and saved {cache_file.name}"

    return keep, report


def select_sensors(
    traffic: np.ndarray,
    adjacency: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Restrict traffic columns and adjacency rows/columns to ``keep``.

    The order of the retained sensors is preserved.
    """

    keep = np.asarray(keep, dtype=np.int64)

    return (
        np.ascontiguousarray(traffic[:, keep]),
        np.ascontiguousarray(adjacency[np.ix_(keep, keep)]),
    )
