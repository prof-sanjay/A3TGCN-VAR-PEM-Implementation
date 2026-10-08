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

CHUNK_SIZE = 5000


# ============================================================
# HELPER
# ============================================================

def inspect_dataset(path, chunk_size=5000):

    print("=" * 80)
    print("GBA TRAFFIC DATA INSPECTION FOR RESAMPLING")
    print("=" * 80)

    with h5py.File(path, "r") as f:

        print("\nHDF5 KEYS")
        print("-" * 80)
        print(list(f.keys()))

        traffic = f["traffic"]
        timestamps = f["timestamps"]

        print("\nTRAFFIC DATASET")
        print("-" * 80)
        print("Shape      :", traffic.shape)
        print("dtype      :", traffic.dtype)
        print("Chunks     :", traffic.chunks)
        print("Compression:", traffic.compression)

        n_timesteps, n_sensors = traffic.shape

        # ----------------------------------------------------
        # TIMESTAMPS
        # ----------------------------------------------------

        print("\nTIMESTAMP INSPECTION")
        print("-" * 80)

        raw_ts = timestamps[:]

        print("Timestamp dtype:", raw_ts.dtype)
        print("First timestamp:", raw_ts[0])
        print("Last timestamp :", raw_ts[-1])

        # Convert timestamps
        try:
            ts = pd.to_datetime(raw_ts)
        except Exception:
            ts = pd.to_datetime(raw_ts.astype(str))

        print("Parsed first   :", ts[0])
        print("Parsed last    :", ts[-1])

        if len(ts) > 1:
            differences = ts[1:] - ts[:-1]

            print("\nTimestamp interval statistics:")
            print("Minimum interval:", differences.min())
            print("Maximum interval:", differences.max())
            print("Most common interval:")
            print(differences.value_counts().head())

            irregular = np.sum(differences != differences[0])

            print("Irregular intervals:", irregular)

        # ----------------------------------------------------
        # GLOBAL STATISTICS
        # ----------------------------------------------------

        total_values = n_timesteps * n_sensors

        nan_count = 0
        pos_inf_count = 0
        neg_inf_count = 0
        zero_count = 0
        finite_count = 0

        global_min = np.inf
        global_max = -np.inf

        # For numerical stability
        sum_values = 0.0
        sum_squared = 0.0

        # Percentile sample
        percentile_sample = []

        # Per-sensor statistics
        sensor_nan = np.zeros(n_sensors, dtype=np.int64)
        sensor_zero = np.zeros(n_sensors, dtype=np.int64)
        sensor_valid = np.zeros(n_sensors, dtype=np.int64)

        # Per-timestep statistics
        timestep_nan = np.zeros(n_timesteps, dtype=np.int64)
        timestep_valid = np.zeros(n_timesteps, dtype=np.int64)

        print("\nSCANNING DATA")
        print("-" * 80)

        for start in range(0, n_timesteps, chunk_size):

            end = min(start + chunk_size, n_timesteps)

            data = traffic[start:end]

            # ------------------------------------------------
            # Special values
            # ------------------------------------------------

            nan_mask = np.isnan(data)
            pos_inf_mask = np.isposinf(data)
            neg_inf_mask = np.isneginf(data)
            finite_mask = np.isfinite(data)
            zero_mask = data == 0

            nan_count += np.count_nonzero(nan_mask)
            pos_inf_count += np.count_nonzero(pos_inf_mask)
            neg_inf_count += np.count_nonzero(neg_inf_mask)
            zero_count += np.count_nonzero(zero_mask)

            finite_values = data[finite_mask]

            finite_count += finite_values.size

            if finite_values.size > 0:

                chunk_min = np.min(finite_values)
                chunk_max = np.max(finite_values)

                global_min = min(global_min, chunk_min)
                global_max = max(global_max, chunk_max)

                sum_values += np.sum(
                    finite_values.astype(np.float64)
                )

                sum_squared += np.sum(
                    finite_values.astype(np.float64) ** 2
                )

                # Keep a manageable random sample
                if finite_values.size > 0:

                    sample_size = min(
                        10000,
                        finite_values.size
                    )

                    indices = np.random.choice(
                        finite_values.size,
                        size=sample_size,
                        replace=False
                    )

                    percentile_sample.append(
                        finite_values[indices]
                    )

            # ------------------------------------------------
            # Per sensor
            # ------------------------------------------------

            sensor_nan += np.count_nonzero(
                nan_mask,
                axis=0
            )

            sensor_zero += np.count_nonzero(
                zero_mask,
                axis=0
            )

            sensor_valid += np.count_nonzero(
                finite_mask,
                axis=0
            )

            # ------------------------------------------------
            # Per timestep
            # ------------------------------------------------

            timestep_nan[start:end] = np.count_nonzero(
                nan_mask,
                axis=1
            )

            timestep_valid[start:end] = np.count_nonzero(
                finite_mask,
                axis=1
            )

            print(
                f"\rProcessed {end:,} / {n_timesteps:,} timesteps",
                end=""
            )

        print("\n")

        # ====================================================
        # GLOBAL RESULTS
        # ====================================================

        print("=" * 80)
        print("GLOBAL VALUE STATISTICS")
        print("=" * 80)

        print("Total values       :", f"{total_values:,}")
        print("Finite values      :", f"{finite_count:,}")

        print(
            "NaN values         :",
            f"{nan_count:,}",
            f"({nan_count / total_values * 100:.6f}%)"
        )

        print(
            "+Inf values        :",
            f"{pos_inf_count:,}",
            f"({pos_inf_count / total_values * 100:.6f}%)"
        )

        print(
            "-Inf values        :",
            f"{neg_inf_count:,}",
            f"({neg_inf_count / total_values * 100:.6f}%)"
        )

        print(
            "Zero values        :",
            f"{zero_count:,}",
            f"({zero_count / total_values * 100:.6f}%)"
        )

        print("Minimum            :", global_min)
        print("Maximum            :", global_max)

        if finite_count > 0:

            mean = sum_values / finite_count

            variance = (
                sum_squared / finite_count
                - mean ** 2
            )

            variance = max(variance, 0)

            std = np.sqrt(variance)

            print("Mean               :", mean)
            print("Std                :", std)

        # ====================================================
        # PERCENTILES
        # ====================================================

        print("\n" + "=" * 80)
        print("VALUE PERCENTILES")
        print("=" * 80)

        if percentile_sample:

            percentile_sample = np.concatenate(
                percentile_sample
            )

            percentiles = [
                0,
                1,
                5,
                10,
                25,
                50,
                75,
                90,
                95,
                99,
                99.5,
                99.9,
                100,
            ]

            values = np.percentile(
                percentile_sample,
                percentiles
            )

            for p, value in zip(percentiles, values):

                print(
                    f"P{p:<5}: {value:.6f}"
                )

        # ====================================================
        # PER-SENSOR STATISTICS
        # ====================================================

        print("\n" + "=" * 80)
        print("PER-SENSOR STATISTICS")
        print("=" * 80)

        sensor_missing_percentage = (
            sensor_nan / n_timesteps * 100
        )

        sensor_valid_percentage = (
            sensor_valid / n_timesteps * 100
        )

        print(
            "Minimum sensor missingness:",
            f"{sensor_missing_percentage.min():.6f}%"
        )

        print(
            "Maximum sensor missingness:",
            f"{sensor_missing_percentage.max():.6f}%"
        )

        print(
            "Mean sensor missingness:",
            f"{sensor_missing_percentage.mean():.6f}%"
        )

        print(
            "Median sensor missingness:",
            f"{np.median(sensor_missing_percentage):.6f}%"
        )

        worst_sensors = np.argsort(
            sensor_missing_percentage
        )[-20:][::-1]

        print("\n20 sensors with highest missingness:")

        for sensor in worst_sensors:

            print(
                f"Sensor {sensor:4d} : "
                f"{sensor_missing_percentage[sensor]:.6f}% missing "
                f"({sensor_nan[sensor]:,} values)"
            )

        # ====================================================
        # PER-TIMESTEP STATISTICS
        # ====================================================

        print("\n" + "=" * 80)
        print("PER-TIMESTEP STATISTICS")
        print("=" * 80)

        timestep_missing_percentage = (
            timestep_nan / n_sensors * 100
        )

        print(
            "Minimum timestep missingness:",
            f"{timestep_missing_percentage.min():.6f}%"
        )

        print(
            "Maximum timestep missingness:",
            f"{timestep_missing_percentage.max():.6f}%"
        )

        print(
            "Mean timestep missingness:",
            f"{timestep_missing_percentage.mean():.6f}%"
        )

        print(
            "Median timestep missingness:",
            f"{np.median(timestep_missing_percentage):.6f}%"
        )

        worst_timesteps = np.argsort(
            timestep_missing_percentage
        )[-20:][::-1]

        print("\n20 timesteps with highest missingness:")

        for idx in worst_timesteps:

            print(
                ts[idx],
                ":",
                f"{timestep_missing_percentage[idx]:.6f}% missing",
                f"({timestep_nan[idx]:,} sensors)"
            )

        # ====================================================
        # RESAMPLING FEASIBILITY
        # ====================================================

        print("\n" + "=" * 80)
        print("RESAMPLING FEASIBILITY")
        print("=" * 80)

        print("""
Original sampling interval:
    5 minutes

Possible aggregation intervals:

    10 minutes  = 2 samples
    15 minutes  = 3 samples
    20 minutes  = 4 samples
    30 minutes  = 6 samples
    60 minutes  = 12 samples

Before choosing one, we need to examine:

    1. Missing-value distribution
    2. Consecutive missing gaps
    3. Number of valid samples per aggregation window
    4. Effect of aggregation on temporal resolution
    5. Compatibility with the forecasting horizons
""")

        # ====================================================
        # HDF5 ATTRIBUTES
        # ====================================================

        print("\n" + "=" * 80)
        print("HDF5 ATTRIBUTES")
        print("=" * 80)

        for key, value in f.attrs.items():
            print(f"{key}: {value}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    inspect_dataset(
        TRAFFIC_PATH,
        CHUNK_SIZE
    )