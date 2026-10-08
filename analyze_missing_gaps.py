import h5py
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TRAFFIC_PATH = (
    r"C:\My Folder\Projects\Horizon-Conditioned Probabilistic "
    r"VAR-LiNGAM Causal Graph for Traffic Forecasting\implementation"
    r"\datasets\LargeST GBA\gba_his_raw_2021.h5"
)

CHUNK_SIZE = 2000

# Aggregation intervals to investigate
RESAMPLING_INTERVALS = {
    "10 min": 2,
    "15 min": 3,
    "20 min": 4,
    "30 min": 6,
    "60 min": 12,
}


# ============================================================
# HELPER: CONSECUTIVE TRUE RUNS
# ============================================================

def consecutive_runs(mask):
    """
    Return lengths and start/end indices of consecutive True runs.

    Example:
        [False, True, True, False, True]
        -> [(1, 2, 2), (4, 4, 1)]
    """

    if len(mask) == 0:
        return []

    mask = np.asarray(mask, dtype=bool)

    padded = np.concatenate(([False], mask, [False]))

    changes = np.diff(padded.astype(np.int8))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1

    return [
        (start, end, end - start + 1)
        for start, end in zip(starts, ends)
    ]


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_missing_gaps(path, chunk_size=2000):

    print("=" * 90)
    print("GBA MISSING-DATA GAP ANALYSIS")
    print("=" * 90)

    # --------------------------------------------------------
    # OPEN DATASET
    # --------------------------------------------------------

    with h5py.File(path, "r") as f:

        print("\nHDF5 KEYS")
        print("-" * 90)
        print(list(f.keys()))

        traffic = f["traffic"]
        timestamps = f["timestamps"]

        n_timesteps, n_sensors = traffic.shape

        print("\nDATASET")
        print("-" * 90)
        print("Shape          :", traffic.shape)
        print("dtype          :", traffic.dtype)
        print("Number sensors :", n_sensors)
        print("Number steps   :", n_timesteps)

        # ----------------------------------------------------
        # TIMESTAMPS
        # ----------------------------------------------------

        raw_ts = timestamps[:]

        try:
            ts = pd.to_datetime(raw_ts)
        except Exception:
            ts = pd.to_datetime(raw_ts.astype(str))

        print("\nTIME RANGE")
        print("-" * 90)
        print("Start:", ts[0])
        print("End  :", ts[-1])

        # ----------------------------------------------------
        # BUILD SENSOR MISSING MASK
        # ----------------------------------------------------
        #
        # We only store one boolean mask per sensor at a time.
        # This avoids loading the entire traffic matrix.
        #
        # missing_by_sensor[sensor] will eventually contain
        # the full temporal missingness pattern for that sensor.
        # ----------------------------------------------------

        print("\nBUILDING MISSINGNESS MASK")
        print("-" * 90)

        # This is ~247 MB for 105120 x 2352 if bool is used.
        # It is acceptable on a normal research workstation,
        # but we avoid the traffic float64 matrix itself.
        missing_matrix = np.zeros(
            (n_timesteps, n_sensors),
            dtype=np.bool_
        )

        for start in range(0, n_timesteps, chunk_size):

            end = min(start + chunk_size, n_timesteps)

            data = traffic[start:end]

            missing_matrix[start:end] = ~np.isfinite(data)

            print(
                f"\rProcessed {end:,} / {n_timesteps:,} timesteps",
                end=""
            )

        print("\n")

        # ====================================================
        # 1. LONGEST GAP PER SENSOR
        # ====================================================

        print("=" * 90)
        print("1. LONGEST CONSECUTIVE MISSING GAP PER SENSOR")
        print("=" * 90)

        sensor_results = []

        for sensor in range(n_sensors):

            mask = missing_matrix[:, sensor]

            runs = consecutive_runs(mask)

            if runs:

                longest = max(runs, key=lambda x: x[2])

                start_idx, end_idx, length = longest

                sensor_results.append({
                    "sensor": sensor,
                    "longest_gap_steps": length,
                    "longest_gap_minutes": length * 5,
                    "start": ts[start_idx],
                    "end": ts[end_idx],
                    "total_missing": int(mask.sum()),
                    "missing_percentage": (
                        mask.sum() / n_timesteps * 100
                    ),
                })

            else:

                sensor_results.append({
                    "sensor": sensor,
                    "longest_gap_steps": 0,
                    "longest_gap_minutes": 0,
                    "start": None,
                    "end": None,
                    "total_missing": 0,
                    "missing_percentage": 0.0,
                })

        sensor_df = pd.DataFrame(sensor_results)

        sensor_df = sensor_df.sort_values(
            "longest_gap_steps",
            ascending=False
        )

        print("\nTOP 30 SENSORS BY LONGEST MISSING GAP")
        print("-" * 90)

        print(
            sensor_df.head(30).to_string(index=False)
        )

        # ====================================================
        # 2. GAP LENGTH DISTRIBUTION
        # ====================================================

        print("\n" + "=" * 90)
        print("2. MISSING GAP LENGTH DISTRIBUTION")
        print("=" * 90)

        all_gap_lengths = []

        for sensor in range(n_sensors):

            runs = consecutive_runs(
                missing_matrix[:, sensor]
            )

            for _, _, length in runs:
                all_gap_lengths.append(length)

        all_gap_lengths = np.asarray(
            all_gap_lengths,
            dtype=np.int64
        )

        print("\nTotal missing gaps:", len(all_gap_lengths))

        if len(all_gap_lengths) > 0:

            print(
                "Shortest gap:",
                all_gap_lengths.min(),
                "steps",
                f"({all_gap_lengths.min() * 5} minutes)"
            )

            print(
                "Longest gap:",
                all_gap_lengths.max(),
                "steps",
                f"({all_gap_lengths.max() * 5} minutes)"
            )

            print(
                "Mean gap:",
                f"{all_gap_lengths.mean():.3f}",
                "steps"
            )

            print(
                "Median gap:",
                np.median(all_gap_lengths),
                "steps"
            )

        # ----------------------------------------------------
        # EXACT GAP CATEGORIES
        # ----------------------------------------------------

        print("\nGAP LENGTH CATEGORIES")
        print("-" * 90)

        categories = [
            ("1 step", 1),
            ("2 steps", 2),
            ("3 steps", 3),
            ("4 steps", 4),
            ("5 steps", 5),
            ("6 steps", 6),
            ("12 steps", 12),
        ]

        for name, length in categories:

            count = np.sum(all_gap_lengths == length)

            print(
                f"{name:12s}: "
                f"{count:,} gaps"
            )

        print(
            "> 12 steps:",
            f"{np.sum(all_gap_lengths > 12):,} gaps"
        )

        print(
            "> 24 steps:",
            f"{np.sum(all_gap_lengths > 24):,} gaps"
        )

        print(
            "> 60 steps:",
            f"{np.sum(all_gap_lengths > 60):,} gaps"
        )

        # ====================================================
        # 3. LONGEST GLOBAL OUTAGE PERIODS
        # ====================================================

        print("\n" + "=" * 90)
        print("3. GLOBAL TEMPORAL OUTAGES")
        print("=" * 90)

        # Number of sensors missing at every timestep
        missing_per_timestep = missing_matrix.sum(axis=1)

        # Percentage of sensors missing
        missing_percentage = (
            missing_per_timestep / n_sensors * 100
        )

        # Find runs where at least one sensor is missing
        any_missing = missing_per_timestep > 0

        global_runs = consecutive_runs(any_missing)

        global_results = []

        for start_idx, end_idx, length in global_runs:

            total_missing_in_period = missing_per_timestep[
                start_idx:end_idx + 1
            ].sum()

            max_missing = missing_per_timestep[
                start_idx:end_idx + 1
            ].max()

            global_results.append({
                "start": ts[start_idx],
                "end": ts[end_idx],
                "steps": length,
                "duration_minutes": length * 5,
                "max_missing_sensors": int(max_missing),
                "total_missing_values": int(
                    total_missing_in_period
                ),
            })

        global_df = pd.DataFrame(global_results)

        if len(global_df) > 0:

            global_df = global_df.sort_values(
                "steps",
                ascending=False
            )

            print("\nTOP 30 GLOBAL MISSING PERIODS")
            print("-" * 90)

            print(
                global_df.head(30).to_string(index=False)
            )

        # ====================================================
        # 4. COMPLETELY MISSING TIMESTEPS
        # ====================================================

        print("\n" + "=" * 90)
        print("4. COMPLETELY MISSING TIMESTEPS")
        print("=" * 90)

        completely_missing = (
            missing_per_timestep == n_sensors
        )

        complete_runs = consecutive_runs(
            completely_missing
        )

        print(
            "Number of completely missing timesteps:",
            int(completely_missing.sum())
        )

        print(
            "Number of completely missing periods:",
            len(complete_runs)
        )

        if complete_runs:

            print("\nCompletely missing periods:")
            print("-" * 90)

            for start_idx, end_idx, length in complete_runs:

                print(
                    f"{ts[start_idx]}  ->  "
                    f"{ts[end_idx]}  | "
                    f"{length} steps | "
                    f"{length * 5} minutes"
                )

        # ====================================================
        # 5. RESAMPLING WINDOW ANALYSIS
        # ====================================================

        print("\n" + "=" * 90)
        print("5. RESAMPLING WINDOW ANALYSIS")
        print("=" * 90)

        print("""
For each possible aggregation interval we calculate:

    - total number of complete windows
    - windows containing at least one missing value
    - completely missing windows
    - percentage of affected windows

No values are imputed here.
This is only an analysis of what resampling WOULD do.
""")

        # ----------------------------------------------------
        # Only complete windows fitting inside the dataset
        # are considered.
        # ----------------------------------------------------

        for interval_name, samples_per_window in RESAMPLING_INTERVALS.items():

            usable_steps = (
                n_timesteps // samples_per_window
            )

            trimmed = usable_steps * samples_per_window

            window_mask = missing_matrix[:trimmed]

            reshaped = window_mask.reshape(
                usable_steps,
                samples_per_window,
                n_sensors
            )

            # Any missing value inside a window
            affected_windows = reshaped.any(axis=1)

            # Every observation missing for a sensor
            completely_missing_sensor_windows = reshaped.all(
                axis=1
            )

            # Any sensor completely missing within window
            any_completely_missing_sensor = (
                completely_missing_sensor_windows.any(axis=1)
            )

            # All sensors missing throughout window
            completely_missing_windows = (
                reshaped.all(axis=(1, 2))
            )

            total_windows = usable_steps

            affected_count = affected_windows.any(axis=1).sum()

            completely_missing_count = (
                completely_missing_windows.sum()
            )

            print("\n" + "-" * 90)
            print(f"{interval_name} ({samples_per_window} × 5-minute samples)")
            print("-" * 90)

            print(
                "Total windows:",
                f"{total_windows:,}"
            )

            print(
                "Windows with >=1 missing value:",
                f"{affected_count:,}",
                f"({affected_count / total_windows * 100:.6f}%)"
            )

            print(
                "Windows with at least one completely missing sensor:",
                f"{any_completely_missing_sensor.sum():,}",
                f"({any_completely_missing_sensor.mean() * 100:.6f}%)"
            )

            print(
                "Completely missing windows (all sensors):",
                f"{completely_missing_count:,}",
                f"({completely_missing_count / total_windows * 100:.6f}%)"
            )

        # ====================================================
        # 6. IMPORTANT SENSOR-LEVEL RESAMPLING ANALYSIS
        # ====================================================

        print("\n" + "=" * 90)
        print("6. SENSOR-LEVEL RESAMPLING IMPACT")
        print("=" * 90)

        for interval_name, samples_per_window in RESAMPLING_INTERVALS.items():

            usable_steps = (
                n_timesteps // samples_per_window
            )

            trimmed = (
                usable_steps * samples_per_window
            )

            reshaped = missing_matrix[
                :trimmed
            ].reshape(
                usable_steps,
                samples_per_window,
                n_sensors
            )

            # For each sensor/window:
            # True if at least one observation is missing
            affected = reshaped.any(axis=1)

            affected_sensor_windows = affected.sum()

            total_sensor_windows = (
                usable_steps * n_sensors
            )

            print(
                f"\n{interval_name}:"
            )

            print(
                "Sensor-windows affected by >=1 NaN:",
                f"{affected_sensor_windows:,} / "
                f"{total_sensor_windows:,}"
            )

            print(
                "Percentage:",
                f"{affected_sensor_windows / total_sensor_windows * 100:.6f}%"
            )

        # ====================================================
        # 7. SAVE SENSOR GAP REPORT
        # ====================================================

        output_file = "gba_sensor_missing_gap_report.csv"

        sensor_df.to_csv(
            output_file,
            index=False
        )

        print("\n" + "=" * 90)
        print("REPORT SAVED")
        print("=" * 90)

        print(
            "Sensor gap report:",
            output_file
        )

        print("\nAnalysis completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    analyze_missing_gaps(
        TRAFFIC_PATH,
        CHUNK_SIZE
    )