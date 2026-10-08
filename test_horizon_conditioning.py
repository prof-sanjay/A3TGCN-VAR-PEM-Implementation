from __future__ import annotations

import numpy as np

from causal.horizon import StructuralIRF


def test_no_contemporaneous_effect():
    """
    Test:

        B0 = 0

    Therefore:

        Theta_h = Psi_h

    """

    print("\n" + "=" * 70)
    print("TEST 1: B0 = 0 -> Theta_h = Psi_h")
    print("=" * 70)

    # Three-node VAR(1)
    #
    # X1(t) depends on X1(t-1)
    # X2(t) depends on X1(t-1)
    # X3(t) depends on X2(t-1)

    M1 = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.4, 0.3, 0.0],
            [0.0, 0.2, 0.2],
        ],
        dtype=np.float64,
    )

    coefficients = M1[np.newaxis, :, :]

    B0 = np.zeros(
        (3, 3),
        dtype=np.float64,
    )

    model = StructuralIRF()

    result = model.fit(
        coefficients=coefficients,
        b0=B0,
        horizons=[1, 2, 3],
    )

    difference = np.max(
        np.abs(
            result.theta - result.psi
        )
    )

    print("\nMaximum |Theta - Psi|:")
    print(difference)

    assert np.allclose(
        result.theta,
        result.psi,
        atol=1e-10,
    )

    print("PASS")


def test_two_step_causal_propagation():
    """
    Verify multi-step propagation.

    Chain:

        X1 -> X2 -> X3

    with:

        M[1,0] = 0.5
        M[2,1] = 0.4

    Therefore:

        Psi_2[2,0]
            = 0.4 * 0.5
            = 0.2

    This demonstrates that a shock at X1 can affect X3
    through X2 even without a direct X1 -> X3 coefficient.
    """

    print("\n" + "=" * 70)
    print("TEST 2: TWO-STEP CAUSAL PROPAGATION")
    print("=" * 70)

    M1 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.4, 0.0],
        ],
        dtype=np.float64,
    )

    coefficients = M1[np.newaxis, :, :]

    B0 = np.zeros(
        (3, 3),
        dtype=np.float64,
    )

    model = StructuralIRF()

    result = model.fit(
        coefficients=coefficients,
        b0=B0,
        horizons=[1, 2],
    )

    expected_two_step_effect = (
        0.4 * 0.5
    )

    actual_two_step_effect = (
        result.theta[2, 2, 0]
    )

    print("\nExpected Theta_2[2,0]:")
    print(expected_two_step_effect)

    print("\nActual Theta_2[2,0]:")
    print(actual_two_step_effect)

    assert np.isclose(
        actual_two_step_effect,
        expected_two_step_effect,
        atol=1e-10,
    )

    # There should be no direct one-step X1 -> X3 effect.
    direct_effect = result.theta[
        1, 2, 0
    ]

    print("\nOne-step X1 -> X3 effect:")
    print(direct_effect)

    assert np.isclose(
        direct_effect,
        0.0,
        atol=1e-10,
    )

    print("PASS")


def test_contemporaneous_structural_effect():
    """
    Test the structural contemporaneous response.

    Model:

        X_t = B0 X_t + ...

    At horizon 0:

        Psi_0 = I

    Therefore:

        Theta_0 = (I - B0)^(-1)

    For:

        X1 -> X2

    with B0[1,0] = 0.5:

        Theta_0[1,0] = 0.5

    Horizon 0 is used here only to validate the structural
    transformation. The actual forecasting horizons remain
    positive (5, 15, 30, 45, 60 minutes).
    """

    print("\n" + "=" * 70)
    print("TEST 3: CONTEMPORANEOUS STRUCTURAL EFFECT")
    print("=" * 70)

    M1 = np.zeros(
        (2, 2),
        dtype=np.float64,
    )

    coefficients = M1[np.newaxis, :, :]

    B0 = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
        ],
        dtype=np.float64,
    )

    model = StructuralIRF()

    result = model.fit(
        coefficients=coefficients,
        b0=B0,
        horizons=[1],
    )

    # ------------------------------------------------------------
    # At horizon 0:
    #
    # Psi_0 = I
    #
    # Therefore:
    #
    # Theta_0 = (I - B0)^(-1)
    # ------------------------------------------------------------
    expected_structural = np.linalg.inv(
        np.eye(2) - B0
    )

    actual_structural = result.theta[0]

    print("\nExpected Theta_0:")
    print(expected_structural)

    print("\nActual Theta_0:")
    print(actual_structural)

    assert np.allclose(
        actual_structural,
        expected_structural,
        atol=1e-10,
    )

    # ------------------------------------------------------------
    # Because M1 = 0, there is no lagged propagation.
    #
    # Therefore:
    #
    # Theta_1 = 0
    # ------------------------------------------------------------
    expected_horizon_one = np.zeros(
        (2, 2),
        dtype=np.float64,
    )

    actual_horizon_one = result.theta[1]

    print("\nExpected Theta_1:")
    print(expected_horizon_one)

    print("\nActual Theta_1:")
    print(actual_horizon_one)

    assert np.allclose(
        actual_horizon_one,
        expected_horizon_one,
        atol=1e-10,
    )

    print("PASS")


def test_traffic_horizons():
    """
    Verify the horizons used in the traffic forecasting study.

    Sampling interval:

        5 minutes

    Therefore:

        1  -> 5 minutes
        3  -> 15 minutes
        6  -> 30 minutes
        9  -> 45 minutes
        12 -> 60 minutes
    """

    print("\n" + "=" * 70)
    print("TEST 4: TRAFFIC FORECASTING HORIZONS")
    print("=" * 70)

    M1 = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.2, 0.4, 0.0],
            [0.0, 0.1, 0.3],
        ],
        dtype=np.float64,
    )

    coefficients = M1[np.newaxis, :, :]

    B0 = np.zeros(
        (3, 3),
        dtype=np.float64,
    )

    horizons = [
        1,
        3,
        6,
        9,
        12,
    ]

    model = StructuralIRF()

    result = model.fit(
        coefficients=coefficients,
        b0=B0,
        horizons=horizons,
    )

    expected_mapping = {
        1: 5,
        3: 15,
        6: 30,
        9: 45,
        12: 60,
    }

    print("\nHorizon mapping:")

    for horizon in horizons:
        minutes = horizon * 5

        print(
            f"  {horizon:2d} step(s) -> "
            f"{minutes:2d} minutes"
        )

        assert minutes == (
            expected_mapping[horizon]
        )

        assert horizon in (
            result.effect_matrices
        )

        matrix = (
            result.effect_matrices[horizon]
        )

        assert matrix.shape == (
            3,
            3,
        )

        # No self-edge effect in the graph representation.
        assert np.allclose(
            np.diag(matrix),
            0.0,
        )

        # Effects must be non-negative because
        # E^(h) = |Theta_h|.
        assert np.all(
            matrix >= 0.0
        )

    print("\nPASS")


def test_stable_response_decay():
    """
    Verify that a stable VAR produces decreasing
    long-horizon effects.

    This is not a universal theorem for arbitrary
    coefficient matrices; this test simply uses a
    deliberately stable synthetic VAR.
    """

    print("\n" + "=" * 70)
    print("TEST 5: STABLE RESPONSE DECAY")
    print("=" * 70)

    M1 = np.array(
        [
            [0.5, 0.0],
            [0.2, 0.4],
        ],
        dtype=np.float64,
    )

    coefficients = M1[np.newaxis, :, :]

    B0 = np.zeros(
        (2, 2),
        dtype=np.float64,
    )

    horizons = [
        1,
        2,
        3,
        4,
        5,
    ]

    model = StructuralIRF()

    result = model.fit(
        coefficients=coefficients,
        b0=B0,
        horizons=horizons,
    )

    # Track the X1 -> X2 response.
    effects = [
        result.theta[
            horizon,
            1,
            0,
        ]
        for horizon in horizons
    ]

    print("\nX1 -> X2 structural responses:")

    for horizon, effect in zip(
        horizons,
        effects,
    ):
        print(
            f"  h={horizon}: {effect:.8f}"
        )

    # In this specific stable system the
    # cross-node response should decrease.
    for previous, current in zip(
        effects,
        effects[1:],
    ):
        assert abs(current) < abs(
            previous
        )

    print("PASS")


def main():
    print("=" * 70)
    print("HORIZON CONDITIONING / STRUCTURAL IRF TEST")
    print("=" * 70)

    test_no_contemporaneous_effect()

    test_two_step_causal_propagation()

    test_contemporaneous_structural_effect()

    test_traffic_horizons()

    test_stable_response_decay()

    print("\n" + "=" * 70)
    print(
        "PASS: All horizon conditioning tests completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()