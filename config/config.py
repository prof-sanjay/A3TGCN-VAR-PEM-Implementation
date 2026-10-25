from dataclasses import dataclass
from pathlib import Path


IMPLEMENTATION_DIR = Path(__file__).resolve().parent.parent

GBA_DIR = IMPLEMENTATION_DIR / "datasets" / "LargeST GBA"

GRAPH_DIR = IMPLEMENTATION_DIR / "outputs" / "graphs"


# ============================================================
# Selected component hyperparameters (validation only, 2026-10-08)
# ============================================================

# Topology-aware VAR (E1, E2): grid lag {1,2,3} x alpha
# {1e-5, 1e4, 1e6, 1e8}; best validation one-step RMSE 26.45.
VAR_LAG = 3
VAR_ALPHA = 1e6
VAR_RUN = f"lag{VAR_LAG}_alpha{VAR_ALPHA:g}"

# Local PEM (E2) on the VAR training residuals, uniform prior.
# Stage B (lr 0.001 + clip 5, 3 epochs): medium nu0/nu1 = 1e-4/2e-3
# best validation RMSE 48.2 (dense 63.9). Update if Stage B changes.
PEM_NU0 = 1e-4
PEM_NU1 = 2e-3
PEM_TAU = 0.01
PEM_THRESHOLD = 0.5
PEM_PRIOR_INCLUSION = 0.5      # used when PEM_PRIOR = "uniform"
PEM_OPTIMIZER = "L-BFGS-B"
PEM_MAX_ITER = 20000
PEM_TOLERANCE = 1e-6

# Prior over edges: "uniform" (eta = PEM_PRIOR_INCLUSION everywhere)
# or "road" (eta = PEM_PRIOR_HIGH where a road link exists in either
# direction, PEM_PRIOR_LOW otherwise; causal/pem_prior.road_prior_eta).
PEM_PRIOR = "uniform"
PEM_PRIOR_HIGH = 0.7
PEM_PRIOR_LOW = 0.3

# Standardise residuals per sensor (training mean / std) before PEM.
# If True, nu0 / nu1 must be re-tuned (precision scale changes).
PEM_STANDARDISE = False


def pem_run_tag(
    var_run=VAR_RUN,
    nu0=PEM_NU0,
    nu1=PEM_NU1,
    tau=PEM_TAU,
    threshold=PEM_THRESHOLD,
    prior=PEM_PRIOR,
    prior_inclusion=PEM_PRIOR_INCLUSION,
    prior_high=PEM_PRIOR_HIGH,
    prior_low=PEM_PRIOR_LOW,
    optimizer=PEM_OPTIMIZER,
    standardise=PEM_STANDARDISE,
):
    """
    Folder name of a PEM run under outputs/pem/ (used by
    run_local_pem.py, build_graph.py and FINAL_GRAPHS, so the three
    always agree). Uniform / unstandardised runs keep their original
    names.
    """

    if prior == "road":
        eta = f"road{prior_high:g}-{prior_low:g}"
    else:
        eta = f"{prior_inclusion:g}"

    return (
        f"{var_run}"
        f"__nu0_{nu0:g}_nu1_{nu1:g}_tau_{tau:g}"
        f"_thr_{threshold:g}_eta_{eta}"
        f"_{optimizer}"
        f"{'_std' if standardise else ''}"
    )


PEM_RUN = pem_run_tag()

# Final graph artifact per experiment (built by build_graph.py).
# Experiment numbering of 2026-10-25 (docs/architecture.md): the former
# E2 (VAR + PEM) is E3.
FINAL_GRAPHS = {
    "E1": f"E1__{VAR_RUN}.npz",
    "E3": f"E3__{PEM_RUN}.npz",
}


@dataclass
class ExperimentConfig:
    """
    Shared configuration for the E0-E7 ablation.

    Every field except ``experiment`` and ``graph_path`` must be
    identical across the experiments of one comparison.

    experiment:
        E0-E7, defined in experiments/registry.py
        (E0 road graph, E1 VAR, E3 VAR + PEM, ...).

    graph_path:
        .npz graph artifact built offline from the TRAINING split
        only. Must be None for E0; otherwise it defaults to
        FINAL_GRAPHS[experiment].
    """

    # Ablation switch
    experiment: str = "E0"
    graph_path: str | None = None

    # Reproducibility
    seed: int = 42

    # Model
    gru_units: int = 100
    seq_len: int = 12
    pre_len: int = 1

    # Training
    # Chosen on E0 validation (stability check, 2026-10-08): lr 0.005
    # (original code) gave 2-6x validation jumps on 2,056 sensors;
    # lr 0.001 + clip 5 trained smoothly (val RMSE 33.9 after 3 epochs).
    learning_rate: float = 0.001
    training_epoch: int = 20
    batch_size: int = 8

    # Chronological split (test = remainder = 10%)
    train_rate: float = 0.8
    val_rate: float = 0.1

    # Gradient clipping by global norm (None = original update)
    grad_clip_norm: float | None = 5.0

    # Regularization
    lambda_loss: float = 0.0015

    # Validation learning-rate scheduler
    lr_patience: int = 5
    lr_factor: float = 0.5
    lr_min: float = 1e-5

    # Early stopping on validation RMSE (None = off: every epoch is run
    # and the best checkpoint is kept, as in all runs so far).
    early_stopping_patience: int | None = None

    # Data (LargeST-GBA 2021, full dataset)
    traffic_path: str = str(GBA_DIR / "gba_his_raw_2021.h5")
    adjacency_path: str = str(GBA_DIR / "gba_rn_adj.npy")
    num_timesteps: int = 105_120
    num_nodes: int = 2_352

    # Data-quality filter (decision D9, defined on TRAIN only):
    #   "duplicates" : drop sensors with copied series
    #   "none"       : keep all sensors
    sensor_filter: str = "duplicates"

    # Missing values for A3T-GCN INPUTS (policy of 2026-10-08):
    #   "gap_bands" : gap-length-band filling (data/imputation.py);
    #                 targets must still be originally observed
    #   "none"      : no filling; windows with any NaN excluded
    imputation: str = "gap_bands"
    imputation_neighbours: int = 5

    # Outputs: results/<experiment>/seed_<seed>/ (main.py --run-name)
    results_dir: str = str(IMPLEMENTATION_DIR / "results")

    def validate(self) -> None:

        try:
            from ..experiments.registry import get_experiment
        except ImportError:  # imported as top-level "config.config"
            from experiments.registry import get_experiment

        spec = get_experiment(self.experiment)

        if not spec.ready:
            raise NotImplementedError(
                f"{self.experiment} ({spec.title}) is not implemented yet: "
                f"{spec.todo}"
            )

        # "road" (E0) and "none" (NOGRAPH) need no graph file.
        builtin_graph = spec.graph_method in ("road", "none")

        if builtin_graph and self.graph_path is not None:
            raise ValueError(
                f"{self.experiment} uses a built-in graph "
                f"({spec.graph_method}); graph_path must be None."
            )

        if not builtin_graph and self.graph_path is None:

            if self.experiment not in FINAL_GRAPHS:
                raise ValueError(
                    f"No final graph set for {self.experiment}; "
                    "pass --graph or add it to FINAL_GRAPHS."
                )

            # Default: the final graph of the selected method.
            self.graph_path = str(GRAPH_DIR / FINAL_GRAPHS[self.experiment])

        if self.graph_path is not None and not Path(self.graph_path).exists():
            raise FileNotFoundError(
                f"Graph artifact not found: {self.graph_path}"
            )

        if self.sensor_filter not in ("duplicates", "none"):
            raise ValueError(
                f"sensor_filter must be 'duplicates' or 'none', "
                f"got {self.sensor_filter}"
            )

        if self.imputation not in ("gap_bands", "none"):
            raise ValueError(
                f"imputation must be 'gap_bands' or 'none', "
                f"got {self.imputation}"
            )

    @property
    def run_dir(self) -> Path:
        return Path(self.results_dir) / self.experiment / f"seed_{self.seed}"
