from __future__ import annotations

import numpy as np

from causal.horizon_graph import HorizonGraphBuilder


def test_basic_thresholding():
    print("\n" + "=" * 70)
    print("TEST 1: BASIC HORIZON GRAPH THRESHOLDING")
    print("=" * 70)

    effect_matrix = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.1],
            [0.2, 0.6, 0.0],
        ],
        dtype=np.float64,
    )

    builder = HorizonGraphBuilder(
        threshold=0.2
    )

    adjacency = builder.build_graph(
        effect_matrix
    )

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [0.2, 0.6, 0.0],
        ],
        dtype=np.float64,
    )

    print("Effect matrix:")
    print(effect_matrix)

    print("\nExpected adjacency:")
    print(expected)

    print("\nActual adjacency:")
    print(adjacency)

    assert np.allclose(
        adjacency,
        expected,
        atol=1e-10,
    )

    print("PASS")


def test_direction_is_preserved():
    print("\n" + "=" * 70)
    print("TEST 2: EDGE DIRECTION PRESERVATION")
    print("=" * 70)

    effect_matrix = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 0.4, 0.0],
        ],
        dtype=np.float64,
    )

    builder = HorizonGraphBuilder(
        threshold=0.1
    )

    adjacency = builder.build_graph(
        effect_matrix
    )

    # VAR-LiNGAM convention:
    #
    # A[target, source] != 0
    #
    # means:
    #
    # source -> target

    assert adjacency[1, 0] == 0.9
    assert adjacency[0, 1] == 0.0

    assert adjacency[2, 1] == 0.4
    assert adjacency[1, 2] == 0.0

    print("X1 -> X2 effect:", adjacency[1, 0])
    print("X2 -> X1 effect:", adjacency[0, 1])

    print("X2 -> X3 effect:", adjacency[2, 1])
    print("X3 -> X2 effect:", adjacency[1, 2])

    print("PASS")


def test_self_loops_are_removed():
    print("\n" + "=" * 70)
    print("TEST 3: SELF-LOOP REMOVAL")
    print("=" * 70)

    effect_matrix = np.array(
        [
            [0.9, 0.2, 0.1],
            [0.3, 0.8, 0.4],
            [0.2, 0.5, 0.7],
        ],
        dtype=np.float64,
    )

    builder = HorizonGraphBuilder(
        threshold=0.0
    )

    adjacency = builder.build_graph(
        effect_matrix
    )

    print("Adjacency:")
    print(adjacency)

    assert np.allclose(
        np.diag(adjacency),
        0.0,
    )

    print("Diagonal successfully removed.")
    print("PASS")


def test_multiple_horizons():
    print("\n" + "=" * 70)
    print("TEST 4: MULTIPLE HORIZON GRAPHS")
    print("=" * 70)

    effect_matrices = {
        1: np.array(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.0, 0.3, 0.0],
            ],
            dtype=np.float64,
        ),
        3: np.array(
            [
                [0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [0.2, 0.5, 0.0],
            ],
            dtype=np.float64,
        ),
        6: np.array(
            [
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.3, 0.4, 0.0],
            ],
            dtype=np.float64,
        ),
    }

    builder = HorizonGraphBuilder(
        threshold=0.25
    )

    result = builder.fit(
        effect_matrices
    )

    print("Available horizons:")
    print(sorted(result.adjacency_matrices.keys()))

    assert set(
        result.adjacency_matrices.keys()
    ) == {1, 3, 6}

    assert set(
        result.thresholds.keys()
    ) == {1, 3, 6}

    for horizon in [1, 3, 6]:
        adjacency = result.adjacency_matrices[horizon]

        print(
            f"\nHorizon {horizon}:"
        )
        print(adjacency)

        assert adjacency.shape == (3, 3)
        assert np.all(
            np.isfinite(adjacency)
        )
        assert np.all(
            adjacency >= 0.0
        )
        assert np.allclose(
            np.diag(adjacency),
            0.0,
        )

    print("PASS")


def test_threshold_zero_preserves_nonzero_effects():
    print("\n" + "=" * 70)
    print("TEST 5: ZERO THRESHOLD")
    print("=" * 70)

    effect_matrix = np.array(
        [
            [0.0, 0.05, 0.0],
            [0.8, 0.0, 0.01],
            [0.0, 0.4, 0.0],
        ],
        dtype=np.float64,
    )

    builder = HorizonGraphBuilder(
        threshold=0.0
    )

    adjacency = builder.build_graph(
        effect_matrix
    )

    expected = effect_matrix.copy()

    np.fill_diagonal(
        expected,
        0.0,
    )

    assert np.allclose(
        adjacency,
        expected,
        atol=1e-10,
    )

    print("All non-zero causal effects preserved.")
    print("PASS")


def main():
    print("=" * 70)
    print("HORIZON GRAPH BUILDER TEST")
    print("=" * 70)

    test_basic_thresholding()
    test_direction_is_preserved()
    test_self_loops_are_removed()
    test_multiple_horizons()
    test_threshold_zero_preserves_nonzero_effects()

    print("\n" + "=" * 70)
    print("PASS: All horizon graph tests completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()