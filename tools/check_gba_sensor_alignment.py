import h5py
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\My Folder\Projects\Horizon-Conditioned Probabilistic "
    r"VAR-LiNGAM Causal Graph for Traffic Forecasting\implementation"
)

DATASET_DIR = BASE_DIR / "datasets" / "LargeST GBA"

TRAFFIC_PATH = DATASET_DIR / "gba_his_raw_2021.h5"
ADJACENCY_PATH = DATASET_DIR / "gba_rn_adj.npy"

# We will automatically search for metadata files.
METADATA_EXTENSIONS = ["*.csv", "*.txt", "*.xlsx"]


# ============================================================
# HELPER
# ============================================================

def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ============================================================
# 1. INSPECT HDF5 SENSOR IDS
# ============================================================

def inspect_hdf5():

    print_header("1. HDF5 SENSOR INFORMATION")

    with h5py.File(TRAFFIC_PATH, "r") as f:

        print("HDF5 keys:")
        print(list(f.keys()))

        traffic = f["traffic"]

        sensor_ids = f["sensor_ids"][:]
        sensor_id2 = f["sensor_id2"][:]

        print("\nTraffic shape:")
        print(traffic.shape)

        print("\nsensor_ids:")
        print("Shape :", sensor_ids.shape)
        print("dtype :", sensor_ids.dtype)
        print("First 20:")
        print(sensor_ids[:20])

        print("\nsensor_id2:")
        print("Shape :", sensor_id2.shape)
        print("dtype :", sensor_id2.dtype)
        print("First 20:")
        print(sensor_id2[:20])

        print("\nAre sensor_ids and sensor_id2 identical?")

        try:
            identical = np.array_equal(sensor_ids, sensor_id2)
            print("Result:", identical)
        except Exception as e:
            print("Could not directly compare:", e)

        print("\nSensor ID uniqueness:")

        unique_ids = np.unique(sensor_ids)

        print("Total IDs :", len(sensor_ids))
        print("Unique IDs:", len(unique_ids))
        print("Duplicates:", len(sensor_ids) - len(unique_ids))

        return sensor_ids, sensor_id2


# ============================================================
# 2. INSPECT ADJACENCY
# ============================================================

def inspect_adjacency():

    print_header("2. ADJACENCY INFORMATION")

    adjacency = np.load(ADJACENCY_PATH)

    print("Path:", ADJACENCY_PATH)

    print("\nShape:")
    print(adjacency.shape)

    print("\ndtype:")
    print(adjacency.dtype)

    print("\nFinite:")
    print(np.all(np.isfinite(adjacency)))

    print("\nMinimum:", adjacency.min())
    print("Maximum:", adjacency.max())

    nonzero_total = np.count_nonzero(adjacency)
    diagonal_nonzero = np.count_nonzero(np.diag(adjacency))

    print("\nNon-zero entries:", nonzero_total)
    print("Non-zero diagonal entries:", diagonal_nonzero)

    off_diagonal = adjacency.copy()
    np.fill_diagonal(off_diagonal, 0)

    off_diagonal_nonzero = np.count_nonzero(off_diagonal)

    print(
        "Off-diagonal non-zero entries:",
        off_diagonal_nonzero
    )

    print("\nExpected GBA graph:")
    print("Nodes : 2352")
    print("Edges : 61246")
    print("District: 4")

    print("\nGraph size check:")

    if adjacency.shape == (2352, 2352):
        print("PASS: adjacency is 2352 x 2352")
    else:
        print("FAIL: unexpected adjacency shape")

    if off_diagonal_nonzero == 61246:
        print(
            "PASS: off-diagonal edge count = 61,246"
        )
    else:
        print(
            "WARNING: off-diagonal edge count =",
            off_diagonal_nonzero,
            "(expected 61,246)"
        )

    print("\nSymmetry check:")

    symmetric = np.allclose(
        adjacency,
        adjacency.T
    )

    print("Symmetric:", symmetric)

    return adjacency


# ============================================================
# 3. FIND METADATA FILE
# ============================================================

def find_metadata_files():

    print_header("3. SEARCHING FOR SENSOR METADATA")

    files = []

    for extension in METADATA_EXTENSIONS:

        files.extend(
            DATASET_DIR.glob(extension)
        )

    print("Possible metadata files:")

    if not files:

        print("No CSV/TXT/XLSX metadata file found.")

    else:

        for file in files:

            print(
                " -",
                file.name
            )

    return files


# ============================================================
# 4. INSPECT METADATA
# ============================================================

def inspect_metadata(metadata_files):

    print_header("4. METADATA INSPECTION")

    if not metadata_files:

        print(
            "No local metadata file was found."
        )

        print(
            "\nSTOPPING BEFORE ALIGNMENT CHECK."
        )

        return None

    # Use CSV first if multiple files exist.
    csv_files = [
        f for f in metadata_files
        if f.suffix.lower() == ".csv"
    ]

    if csv_files:

        metadata_path = csv_files[0]

    else:

        metadata_path = metadata_files[0]

    print("Using metadata file:")
    print(metadata_path)

    try:

        if metadata_path.suffix.lower() == ".csv":

            metadata = pd.read_csv(
                metadata_path
            )

        elif metadata_path.suffix.lower() == ".xlsx":

            metadata = pd.read_excel(
                metadata_path
            )

        else:

            metadata = pd.read_csv(
                metadata_path,
                sep=None,
                engine="python"
            )

    except Exception as e:

        print(
            "\nCould not read metadata:",
            repr(e)
        )

        return None

    print("\nMetadata shape:")
    print(metadata.shape)

    print("\nMetadata columns:")
    print(list(metadata.columns))

    print("\nFirst 10 rows:")
    print(
        metadata.head(10).to_string(
            index=False
        )
    )

    print("\nMissing values per column:")
    print(
        metadata.isna().sum()
    )

    return metadata


# ============================================================
# 5. FIND SENSOR ID COLUMN
# ============================================================

def identify_sensor_id_column(metadata):

    if metadata is None:
        return None

    candidates = [
        "ID",
        "id",
        "sensor_id",
        "sensor_ids",
        "Sensor_ID",
        "sensorID",
    ]

    for column in candidates:

        if column in metadata.columns:

            print(
                "\nDetected sensor ID column:",
                column
            )

            return column

    print(
        "\nCould not automatically identify "
        "the sensor ID column."
    )

    print(
        "Available columns:",
        list(metadata.columns)
    )

    return None


# ============================================================
# 6. COMPARE HDF5 IDS WITH METADATA IDS
# ============================================================

def compare_sensor_ids(
    hdf5_sensor_ids,
    metadata,
    metadata_id_column
):

    print_header("5. HDF5 ↔ METADATA SENSOR ALIGNMENT")

    if metadata is None:

        print("Skipped: metadata unavailable.")
        return

    if metadata_id_column is None:

        print(
            "Skipped: sensor ID column not identified."
        )

        return

    metadata_ids = metadata[
        metadata_id_column
    ].to_numpy()

    hdf5_ids = np.asarray(
        hdf5_sensor_ids
    )

    print("HDF5 sensor count:")
    print(len(hdf5_ids))

    print("Metadata sensor count:")
    print(len(metadata_ids))

    # Convert to strings to avoid int/string mismatch.
    hdf5_strings = np.asarray(
        [str(x).strip() for x in hdf5_ids]
    )

    metadata_strings = np.asarray(
        [str(x).strip() for x in metadata_ids]
    )

    print("\nUnique HDF5 IDs:")
    print(len(np.unique(hdf5_strings)))

    print("Unique metadata IDs:")
    print(len(np.unique(metadata_strings)))

    hdf5_set = set(hdf5_strings)
    metadata_set = set(metadata_strings)

    hdf5_not_metadata = (
        hdf5_set - metadata_set
    )

    metadata_not_hdf5 = (
        metadata_set - hdf5_set
    )

    print(
        "\nHDF5 IDs missing from metadata:",
        len(hdf5_not_metadata)
    )

    print(
        "Metadata IDs missing from HDF5:",
        len(metadata_not_hdf5)
    )

    if hdf5_not_metadata:

        print("\nFirst HDF5-only IDs:")
        print(
            list(hdf5_not_metadata)[:20]
        )

    if metadata_not_hdf5:

        print("\nFirst metadata-only IDs:")
        print(
            list(metadata_not_hdf5)[:20]
        )

    if (
        len(hdf5_not_metadata) == 0
        and len(metadata_not_hdf5) == 0
    ):

        print(
            "\nPASS: HDF5 and metadata contain "
            "the same sensor IDs."
        )

    # --------------------------------------------------------
    # ORDER CHECK
    # --------------------------------------------------------

    print("\nORDER CHECK")

    if len(hdf5_strings) == len(metadata_strings):

        same_order = np.array_equal(
            hdf5_strings,
            metadata_strings
        )

        print(
            "Same order:",
            same_order
        )

        if same_order:

            print(
                "PASS: HDF5 traffic columns and "
                "metadata rows are identically ordered."
            )

        else:

            print(
                "WARNING: same IDs but different order."
            )

            print(
                "This does NOT necessarily mean the "
                "dataset is invalid, but we must map "
                "the ordering explicitly before using "
                "the adjacency matrix."
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print_header(
        "LARGEST GBA SENSOR / GRAPH ALIGNMENT CHECK"
    )

    print("Dataset directory:")
    print(DATASET_DIR)

    print("\nTraffic file:")
    print(TRAFFIC_PATH)

    print("\nAdjacency file:")
    print(ADJACENCY_PATH)

    # 1
    sensor_ids, sensor_id2 = inspect_hdf5()

    # 2
    adjacency = inspect_adjacency()

    # 3
    metadata_files = find_metadata_files()

    # 4
    metadata = inspect_metadata(
        metadata_files
    )

    # 5
    metadata_id_column = identify_sensor_id_column(
        metadata
    )

    # 6
    compare_sensor_ids(
        sensor_ids,
        metadata,
        metadata_id_column
    )

    print_header("CHECK COMPLETED")

    print("""
Do NOT modify the dataset based on this script.

The next decision depends on the results:

1. HDF5 IDs = metadata IDs
2. Traffic-column order = metadata order
3. Adjacency node order = traffic-column order

Only after these are verified should we use the
adjacency matrix for topology-aware VAR and A3T-GCN.
""")