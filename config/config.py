from dataclasses import dataclass
from pathlib import Path


IMPLEMENTATION_DIR = Path(__file__).resolve().parent.parent

GBA_DIR = IMPLEMENTATION_DIR / "datasets" / "LargeST GBA"


@dataclass
class ExperimentConfig:
    """
    Shared configuration for the E0 / E1 / E2 ablation.

    Every field except ``experiment`` and ``graph_path`` must be
    identical across E0, E1 and E2. The values are the ones that
    were previously hard-coded in main.py.

    experiment:
        "E0" : A3T-GCN on the road-network adjacency.
        "E1" : A3T-GCN on the topology-aware VAR graph.
        "E2" : A3T-GCN on the VAR + PEM probabilistic causal graph.

    graph_path:
        .npz graph artifact built offline from the TRAINING split
        only. Required for E1 and E2, must be None for E0.
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
    learning_rate: float = 0.005
    training_epoch: int = 20
    batch_size: int = 8

    # Chronological split (test = remainder = 10%)
    train_rate: float = 0.8
    val_rate: float = 0.1

    # Regularization
    lambda_loss: float = 0.0015

    # Validation learning-rate scheduler
    lr_patience: int = 5
    lr_factor: float = 0.5
    lr_min: float = 1e-5

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

    # Outputs: results/<experiment>/
    results_dir: str = str(IMPLEMENTATION_DIR / "results")

    def validate(self) -> None:

        if self.experiment not in ("E0", "E1", "E2"):
            raise ValueError(
                f"experiment must be E0, E1 or E2, got {self.experiment}"
            )

        if self.experiment == "E0" and self.graph_path is not None:
            raise ValueError(
                "E0 uses the road adjacency; graph_path must be None."
            )

        if self.experiment in ("E1", "E2") and self.graph_path is None:
            raise ValueError(
                f"{self.experiment} requires graph_path."
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
        return Path(self.results_dir) / self.experiment
