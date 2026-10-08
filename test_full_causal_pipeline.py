from __future__ import annotations

import numpy as np

from causal.var import TopologyAwareVAR
from causal.var_lingam import VARLiNGAM
from causal.bootstrap_lingam import BootstrapLiNGAM
from causal.pem_prior import PEMPrior
from causal.pem import PEM
from causal.horizon import StructuralIRF
from causal.horizon_graph import HorizonGraphBuilder


# ============================================================
# CONFIGURATION
# ============================================================

NUM_SAMPLES = 3000
NUM_NODES = 3
RANDOM_STATE = 42

HORIZONS = [1, 3, 6, 9, 12]

NUM_BOOTSTRAP_SAMPLES = 10


# ============================================================
# SYNTHETIC TRAFFIC GENERATION
# ============================================================

def generate_synthetic_traffic(
    num_samples: int = NUM_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(random_state)

    # --------------------------------------------------------
    # True contemporaneous causal matrix B0
    #
    # B0[target, source] means:
    #
    # source -> target
    #
    # Therefore:
    #
    # X1 -> X2 = 0.8
    # X2 -> X3 = 0.6
    # --------------------------------------------------------

    B0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [0.0, 0.6, 0.0],
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # True temporal VAR coefficient B1
    # --------------------------------------------------------

    B1 = np.array(
        [
            [0.4, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Generate non-Gaussian noise
    # --------------------------------------------------------

    noise = rng.laplace(
        loc=0.0,
        scale=1.0,
        size=(num_samples, NUM_NODES),
    )

    X = np.zeros(
        (num_samples, NUM_NODES),
        dtype=np.float64,
    )

    identity = np.eye(NUM_NODES)

    # --------------------------------------------------------
    # Structural VAR:
    #
    # X_t = B0 X_t + B1 X_(t-1) + e_t
    #
    # Therefore:
    #
    # X_t =
    # (I - B0)^(-1)
    # [B1 X_(t-1) + e_t]
    # --------------------------------------------------------

    structural_inverse = np.linalg.inv(
        identity - B0
    )

    for t in range(num_samples):

        previous = (
            B1 @ X[t - 1]
            if t > 0
            else np.zeros(NUM_NODES)
        )

        X[t] = structural_inverse @ (
            previous + noise[t]
        )

    return X, B0, B1


# ============================================================
# MAIN PIPELINE TEST
# ============================================================

def main() -> None:

    print("=" * 70)
    print("FULL CAUSAL PIPELINE INTEGRATION TEST")
    print("=" * 70)

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 1: SYNTHETIC TRAFFIC GENERATION")
    print("=" * 70)

    traffic, true_B0, true_B1 = (
        generate_synthetic_traffic()
    )

    print("Traffic shape:", traffic.shape)

    print("True B0:")
    print(true_B0)

    print("True B1:")
    print(true_B1)

    assert traffic.shape == (
        NUM_SAMPLES,
        NUM_NODES,
    )

    assert np.all(
        np.isfinite(traffic)
    )

    print("PASS")

    # ========================================================
    # STEP 2
    # TOPOLOGY-AWARE VAR
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 2: TOPOLOGY-AWARE VAR")
    print("=" * 70)

    # --------------------------------------------------------
    # Undirected road topology for the synthetic network.
    #
    # Self loops are included by the VAR implementation.
    # --------------------------------------------------------

    topology = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    var_model = TopologyAwareVAR(
        lag_order=1,
        ridge_alpha=1e-5,
        use_gpu=True,
    )

    var_result = var_model.fit(
        data=traffic,
        adjacency=topology,
    )

    print(
        "Estimated VAR coefficient shape:",
        var_result.coefficients.shape,
    )

    print("Estimated B1:")
    print(var_result.coefficients[0])

    print(
        "Residual shape:",
        var_result.residuals.shape,
    )

    # --------------------------------------------------------
    # NOTE:
    #
    # valid_mask is currently an element-wise mask:
    # (time, node)
    #
    # Therefore sum(valid_mask) counts valid residual
    # elements, not complete time samples.
    # --------------------------------------------------------

    print(
        "Valid residual elements:",
        int(np.sum(var_result.valid_mask)),
    )

    assert var_result.coefficients.shape == (
        1,
        NUM_NODES,
        NUM_NODES,
    )

    assert var_result.residuals.shape == (
        NUM_SAMPLES,
        NUM_NODES,
    )

    print("PASS")

    # ========================================================
    # PREPARE COMPLETE RESIDUAL SAMPLES
    # ========================================================

    residuals = var_result.residuals

    valid_residuals = residuals[
        np.all(
            np.isfinite(residuals),
            axis=1,
        )
    ]

    print(
        "Complete residual sample shape:",
        valid_residuals.shape,
    )

    assert valid_residuals.shape[0] >= 2

    # ========================================================
    # STEP 3
    # BOOTSTRAP VAR-LiNGAM
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 3: BOOTSTRAP VAR-LiNGAM")
    print("=" * 70)

    bootstrap_model = BootstrapLiNGAM(
        num_bootstrap_samples=NUM_BOOTSTRAP_SAMPLES,
        random_state=RANDOM_STATE,
    )

    bootstrap_result = bootstrap_model.fit(
        residuals
    )

    print(
        "Bootstrap adjacency shape:",
        bootstrap_result.adjacency_matrices.shape,
    )

    print(
        "Valid bootstrap runs:",
        bootstrap_result.valid_runs,
    )

    print("Bootstrap edge support:")
    print(bootstrap_result.edge_support)

    assert (
        bootstrap_result.adjacency_matrices.shape
        == (
            NUM_BOOTSTRAP_SAMPLES,
            NUM_NODES,
            NUM_NODES,
        )
    )

    assert (
        bootstrap_result.valid_runs
        == NUM_BOOTSTRAP_SAMPLES
    )

    assert np.all(
        np.isfinite(
            bootstrap_result.edge_support
        )
    )

    print("PASS")

    # ========================================================
    # STEP 4
    # PEM PRIOR CONSTRUCTION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 4: PEM PRIOR CONSTRUCTION")
    print("=" * 70)

    prior_builder = PEMPrior()

    eta = prior_builder.calculate_eta(
        bootstrap_result.edge_support
    )

    print("PEM eta:")
    print(eta)

    assert eta.shape == (
        NUM_NODES,
        NUM_NODES,
    )

    assert np.all(
        np.isfinite(eta)
    )

    assert np.all(
        eta >= 0.0
    )

    assert np.all(
        eta <= 1.0
    )

    assert np.allclose(
        eta,
        eta.T,
    )

    print("PASS")

    # ========================================================
    # STEP 5
    # PEM
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 5: PEM")
    print("=" * 70)

    pem_model = PEM(
        eta=eta,
        threshold=0.5,
    )

    pem_result = pem_model.fit(
        valid_residuals
    )

    print("PEM precision:")
    print(
        pem_result.precision_matrix
    )

    print("PEM posterior inclusion:")
    print(
        pem_result.inclusion_probabilities
    )

    print("PEM adjacency:")
    print(
        pem_result.adjacency_matrix
    )

    print(
        "PEM causal order:",
        pem_result.causal_order,
    )

    assert pem_result.precision_matrix.shape == (
        NUM_NODES,
        NUM_NODES,
    )

    assert pem_result.covariance_matrix.shape == (
        NUM_NODES,
        NUM_NODES,
    )

    assert pem_result.inclusion_probabilities.shape == (
        NUM_NODES,
        NUM_NODES,
    )

    assert pem_result.adjacency_matrix.shape == (
        NUM_NODES,
        NUM_NODES,
    )

    assert np.all(
        np.isfinite(
            pem_result.inclusion_probabilities
        )
    )

    assert np.all(
        np.isfinite(
            pem_result.precision_matrix
        )
    )

    print("PASS")

    # ========================================================
    # REFERENCE VAR-LiNGAM
    #
    # We need the numerical contemporaneous B0 for the
    # structural IRF.
    #
    # PEM adjacency is binary and therefore must NOT be
    # directly used as the numerical B0.
    # ========================================================

    print()
    print("=" * 70)
    print("REFERENCE VAR-LiNGAM B0 FOR STRUCTURAL IRF")
    print("=" * 70)

    lingam_model = VARLiNGAM(
        random_state=RANDOM_STATE
    )

    lingam_result = lingam_model.fit(
        valid_residuals
    )

    print("Estimated numerical B0:")
    print(
        lingam_result.adjacency_matrix
    )

    print(
        "LiNGAM causal order:",
        lingam_result.causal_order,
    )

    assert lingam_result.adjacency_matrix.shape == (
        NUM_NODES,
        NUM_NODES,
    )

    assert np.all(
        np.isfinite(
            lingam_result.adjacency_matrix
        )
    )

    print("PASS")

    # ========================================================
    # STEP 6
    # STRUCTURAL IRF / HORIZON CONDITIONING
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 6: STRUCTURAL IRF / HORIZON CONDITIONING")
    print("=" * 70)

    horizons = HORIZONS

    irf_model = StructuralIRF()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Use numerical VAR-LiNGAM B0 here.
    #
    # Do NOT use:
    #
    #     pem_result.adjacency_matrix
    #
    # because that is a binary causal graph.
    #
    # The structural IRF requires numerical B0 values.
    # --------------------------------------------------------

    irf_result = irf_model.fit(
        coefficients=var_result.coefficients,
        b0=lingam_result.adjacency_matrix,
        horizons=horizons,
    )

    print(
        "Psi shape:",
        irf_result.psi.shape,
    )

    print(
        "Theta shape:",
        irf_result.theta.shape,
    )

    print(
        "Requested horizons:",
        horizons,
    )

    assert irf_result.psi.shape[0] == (
        max(horizons) + 1
    )

    assert irf_result.theta.shape[0] == (
        max(horizons) + 1
    )

    assert irf_result.psi.shape[1:] == (
        NUM_NODES,
        NUM_NODES,
    )

    assert irf_result.theta.shape[1:] == (
        NUM_NODES,
        NUM_NODES,
    )

    for horizon in horizons:

        effect_matrix = (
            irf_result.effect_matrices[horizon]
        )

        print(
            f"Horizon {horizon}:",
            effect_matrix,
        )

        assert effect_matrix.shape == (
            NUM_NODES,
            NUM_NODES,
        )

        assert np.all(
            np.isfinite(effect_matrix)
        )

        assert np.allclose(
            np.diag(effect_matrix),
            0.0,
        )

    print("PASS")

    # ========================================================
    # STEP 7
    # HORIZON-SPECIFIC CAUSAL GRAPHS
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 7: HORIZON-SPECIFIC CAUSAL GRAPHS")
    print("=" * 70)

    graph_builder = HorizonGraphBuilder(
        threshold=0.0
    )

    graph_result = graph_builder.fit(
        effect_matrices=irf_result.effect_matrices,
    )

    print(
        "Available graph horizons:",
        horizons,
    )

    for horizon in horizons:

        adjacency = (
            graph_result.adjacency_matrices[horizon]
        )

        print(
            f"A^({horizon}):",
            adjacency,
        )

        assert adjacency.shape == (
            NUM_NODES,
            NUM_NODES,
        )

        assert np.all(
            np.isfinite(adjacency)
        )

        assert np.allclose(
            np.diag(adjacency),
            0.0,
        )

    print("PASS")

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("FULL CAUSAL PIPELINE RESULT")
    print("=" * 70)

    print(
        """
    Traffic
        |
        v
    Topology-aware VAR
        |
        v
    VAR residuals
        |
        +-----------------------------+
        |                             |
        v                             v
    Bootstrap VAR-LiNGAM        Reference VAR-LiNGAM
        |                             |
        v                             v
    Edge support                    B0
        |
        v
    PEM prior eta
        |
        v
    PEM
        |
        v
    Probabilistic causal information
        |
        +-----------------------------+
                                      |
                                      v
                              Structural IRF
                                      |
                                      v
                            Horizon conditioning
                                      |
                                      v
                         Horizon-specific effects
                                      |
                                      v
                      Horizon-specific causal graphs
    """
    )

    print()
    print("=" * 70)
    print("PASS: Full causal pipeline completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()