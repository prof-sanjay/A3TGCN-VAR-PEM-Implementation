"""
Validation of saved topology-aware VAR outputs.

Checks:
    1. Saved array shapes
    2. Data types
    3. Missing-value counts
    4. Residual coverage
    5. Coefficient sparsity
    6. Topology consistency
    7. Residual / valid-mask consistency
    8. Metadata consistency
"""

import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs") / "var"

COEFFICIENT_FILE = OUTPUT_DIR / "coefficients.npy"
RESIDUAL_FILE = OUTPUT_DIR / "residuals.npy"
VALID_MASK_FILE = OUTPUT_DIR / "valid_mask.npy"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

ADJACENCY_FILE = Path(
    r"C:\My Folder\Projects\Datasets\LargeSTGBA\gba_rn_adj.npy"
)


# ============================================================
# HELPERS
# ============================================================

def check_file(path: Path) -> None:
    """Verify that an expected output file exists."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def print_shape_and_dtype(name: str, array: np.ndarray) -> None:
    """Print basic array information."""

    print(f"  {name}")
    print(f"    Shape : {array.shape}")
    print(f"    Dtype : {array.dtype}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("VAR OUTPUT VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print("\nCHECKING OUTPUT FILES")

    for path in [
        COEFFICIENT_FILE,
        RESIDUAL_FILE,
        VALID_MASK_FILE,
        METADATA_FILE,
    ]:
        check_file(path)
        print(f"  FOUND: {path}")

    # --------------------------------------------------------
    # Load outputs
    # --------------------------------------------------------

    print("\nLOADING OUTPUTS")

    coefficients = np.load(COEFFICIENT_FILE)
    residuals = np.load(RESIDUAL_FILE)
    valid_mask = np.load(VALID_MASK_FILE)

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    print_shape_and_dtype(
        "Coefficients:",
        coefficients,
    )

    print_shape_and_dtype(
        "Residuals:",
        residuals,
    )

    print_shape_and_dtype(
        "Valid mask:",
        valid_mask,
    )

    # --------------------------------------------------------
    # Basic shape validation
    # --------------------------------------------------------

    print("\nSHAPE VALIDATION")

    shape_passed = True

    if coefficients.ndim != 3:
        print("  FAIL: coefficients must be 3-dimensional")
        shape_passed = False

    if residuals.ndim != 2:
        print("  FAIL: residuals must be 2-dimensional")
        shape_passed = False

    if valid_mask.ndim != 2:
        print("  FAIL: valid_mask must be 2-dimensional")
        shape_passed = False

    if coefficients.ndim == 3:

        if coefficients.shape[0] != 1:
            print(
                "  FAIL: expected lag dimension of 1 "
                f"for lag_order=1, got {coefficients.shape[0]}"
            )
            shape_passed = False

        if coefficients.shape[1] != coefficients.shape[2]:
            print(
                "  FAIL: coefficient matrix is not square"
            )
            shape_passed = False

    if residuals.ndim == 2 and valid_mask.ndim == 2:

        if residuals.shape != valid_mask.shape:
            print(
                "  FAIL: residuals and valid_mask shapes differ"
            )
            shape_passed = False

    if (
        coefficients.ndim == 3
        and residuals.ndim == 2
    ):

        num_nodes = coefficients.shape[1]

        if residuals.shape[1] != num_nodes:
            print(
                "  FAIL: number of residual nodes does not "
                "match coefficient matrix"
            )
            shape_passed = False

    print(
        f"  Shape validation: "
        f"{'PASSED' if shape_passed else 'FAILED'}"
    )

    if not shape_passed:
        raise RuntimeError(
            "Basic shape validation failed."
        )

    # --------------------------------------------------------
    # Residual / mask consistency
    # --------------------------------------------------------

    print("\nRESIDUAL / VALID-MASK CONSISTENCY")

    residual_finite = np.isfinite(residuals)

    mask_boolean = valid_mask.astype(bool)

    matching_entries = np.count_nonzero(
        residual_finite == mask_boolean
    )

    total_entries = residuals.size

    consistency_percentage = (
        100.0
        * matching_entries
        / total_entries
    )

    print(
        f"  Matching entries : "
        f"{matching_entries}"
    )

    print(
        f"  Total entries    : "
        f"{total_entries}"
    )

    print(
        f"  Consistency      : "
        f"{consistency_percentage:.6f}%"
    )

    mask_consistency_passed = (
        matching_entries == total_entries
    )

    print(
        f"  Validation       : "
        f"{'PASSED' if mask_consistency_passed else 'FAILED'}"
    )

    # --------------------------------------------------------
    # Residual coverage
    # --------------------------------------------------------

    print("\nRESIDUAL COVERAGE")

    valid_counts = np.sum(
        mask_boolean,
        axis=0,
    )

    num_timesteps = residuals.shape[0]

    coverage = (
        valid_counts
        / num_timesteps
        * 100.0
    )

    print(
        f"  Minimum valid samples : "
        f"{valid_counts.min()}"
    )

    print(
        f"  Maximum valid samples : "
        f"{valid_counts.max()}"
    )

    print(
        f"  Mean valid samples    : "
        f"{valid_counts.mean():.3f}"
    )

    print(
        f"  Minimum coverage      : "
        f"{coverage.min():.6f}%"
    )

    print(
        f"  Maximum coverage      : "
        f"{coverage.max():.6f}%"
    )

    print(
        f"  Mean coverage         : "
        f"{coverage.mean():.6f}%"
    )

    # --------------------------------------------------------
    # Residual NaN validation
    # --------------------------------------------------------

    print("\nRESIDUAL VALUES")

    finite_count = np.count_nonzero(
        np.isfinite(residuals)
    )

    nonfinite_count = np.count_nonzero(
        ~np.isfinite(residuals)
    )

    print(
        f"  Finite values    : "
        f"{finite_count}"
    )

    print(
        f"  Non-finite values: "
        f"{nonfinite_count}"
    )

    if finite_count > 0:

        finite_residuals = residuals[
            np.isfinite(residuals)
        ]

        print(
            f"  Mean             : "
            f"{finite_residuals.mean()}"
        )

        print(
            f"  Std              : "
            f"{finite_residuals.std()}"
        )

        print(
            f"  Minimum          : "
            f"{finite_residuals.min()}"
        )

        print(
            f"  Maximum          : "
            f"{finite_residuals.max()}"
        )

    # --------------------------------------------------------
    # Coefficient sparsity
    # --------------------------------------------------------

    print("\nCOEFFICIENT SPARSITY")

    coefficient_matrix = coefficients[0]

    nonzero_coefficients = np.count_nonzero(
        coefficient_matrix
    )

    total_coefficients = (
        coefficient_matrix.size
    )

    coefficient_density = (
        100.0
        * nonzero_coefficients
        / total_coefficients
    )

    print(
        f"  Total coefficients    : "
        f"{total_coefficients}"
    )

    print(
        f"  Non-zero coefficients : "
        f"{nonzero_coefficients}"
    )

    print(
        f"  Density               : "
        f"{coefficient_density:.6f}%"
    )

    # --------------------------------------------------------
    # Load adjacency
    # --------------------------------------------------------

    print("\nLOADING ORIGINAL ADJACENCY")

    if not ADJACENCY_FILE.exists():
        raise FileNotFoundError(
            f"Adjacency file not found: {ADJACENCY_FILE}"
        )

    adjacency = np.load(
        ADJACENCY_FILE,
        mmap_mode="r",
    )

    num_nodes = coefficient_matrix.shape[0]

    adjacency = np.asarray(
        adjacency[:num_nodes, :num_nodes],
        dtype=np.float32,
    )

    print(
        f"  Adjacency shape : "
        f"{adjacency.shape}"
    )

    print(
        f"  Non-zero entries: "
        f"{np.count_nonzero(adjacency)}"
    )

    # --------------------------------------------------------
    # Topology consistency
    # --------------------------------------------------------

    print("\nTOPOLOGY CONSISTENCY")

    forbidden_positions = (
        adjacency == 0
    )

    forbidden_nonzero = np.count_nonzero(
        coefficient_matrix[forbidden_positions]
        != 0
    )

    allowed_positions = (
        adjacency != 0
    )

    allowed_count = np.count_nonzero(
        allowed_positions
    )

    coefficient_count = np.count_nonzero(
        coefficient_matrix
    )

    print(
        f"  Allowed topology entries : "
        f"{allowed_count}"
    )

    print(
        f"  Non-zero VAR coefficients : "
        f"{coefficient_count}"
    )

    print(
        f"  Forbidden non-zero values : "
        f"{forbidden_nonzero}"
    )

    topology_passed = (
        forbidden_nonzero == 0
    )

    print(
        f"  Validation                : "
        f"{'PASSED' if topology_passed else 'FAILED'}"
    )

    # --------------------------------------------------------
    # Coefficient finite-value validation
    # --------------------------------------------------------

    print("\nCOEFFICIENT FINITE-VALUE CHECK")

    coefficient_finite = np.isfinite(
        coefficient_matrix
    )

    nonfinite_coefficients = np.count_nonzero(
        ~coefficient_finite
    )

    print(
        f"  Non-finite coefficients : "
        f"{nonfinite_coefficients}"
    )

    coefficients_finite_passed = (
        nonfinite_coefficients == 0
    )

    print(
        f"  Validation              : "
        f"{'PASSED' if coefficients_finite_passed else 'FAILED'}"
    )

    # --------------------------------------------------------
    # Metadata validation
    # --------------------------------------------------------

    print("\nMETADATA")

    metadata_checks = {
        "num_timesteps": residuals.shape[0],
        "num_nodes": residuals.shape[1],
        "coefficient_shape": list(
            coefficients.shape
        ),
        "residual_shape": list(
            residuals.shape
        ),
        "valid_mask_shape": list(
            valid_mask.shape
        ),
    }

    metadata_passed = True

    for key, expected_value in metadata_checks.items():

        actual_value = metadata.get(key)

        matches = (
            actual_value == expected_value
        )

        print(
            f"  {key}: "
            f"{'OK' if matches else 'MISMATCH'}"
        )

        if not matches:
            print(
                f"    Expected: {expected_value}"
            )
            print(
                f"    Actual  : {actual_value}"
            )
            metadata_passed = False

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    checks = {
        "Shape validation": shape_passed,
        "Residual/mask consistency": (
            mask_consistency_passed
        ),
        "Topology consistency": topology_passed,
        "Finite coefficients": (
            coefficients_finite_passed
        ),
        "Metadata consistency": metadata_passed,
    }

    all_passed = True

    for name, passed in checks.items():

        print(
            f"  {name:<32}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

        if not passed:
            all_passed = False

    print()

    if all_passed:
        print(
            "ALL VAR OUTPUT VALIDATION CHECKS PASSED"
        )
    else:
        print(
            "VAR OUTPUT VALIDATION FAILED"
        )
        raise RuntimeError(
            "One or more VAR output validation checks failed."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()