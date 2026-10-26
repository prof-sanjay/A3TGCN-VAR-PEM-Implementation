"""
Experiment registry: E0-E7 of the ablation (docs/architecture.md, section 3).

Every experiment uses the same data pipeline, A3T-GCN model, training
loop, metrics and logger (main.py). They differ only in

    graph_method   which graph artifact the model gets
    horizons       forecast horizons in 5-minute steps

ready = False means the graph builder for that experiment is not
implemented yet; main.py refuses to run it and names the phase.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:

    name: str
    title: str
    graph_method: str          # "road" or the artifact's method
    candidate_links: str
    prior: str
    horizons: tuple[int, ...]  # 5-minute steps (1 = 5 min)
    ready: bool
    todo: str = ""


EXPERIMENTS: dict[str, ExperimentSpec] = {
    spec.name: spec
    for spec in [
        ExperimentSpec(
            "E0", "A3T-GCN on the road graph",
            graph_method="road", candidate_links="-", prior="-",
            horizons=(1,), ready=True,
        ),
        ExperimentSpec(
            "E1", "A3T-GCN + VAR graph",
            graph_method="var", candidate_links="road neighbours", prior="-",
            horizons=(1,), ready=True,
        ),
        ExperimentSpec(
            "E2", "A3T-GCN + VAR-LiNGAM graph",
            graph_method="var_lingam", candidate_links="road neighbours", prior="-",
            horizons=(1,), ready=True,
        ),
        ExperimentSpec(
            "E3", "A3T-GCN + VAR + PEM graph (former E2)",
            graph_method="var_pem", candidate_links="road neighbours", prior="uniform",
            horizons=(1,), ready=True,
        ),
        ExperimentSpec(
            "E4", "A3T-GCN + VAR-LiNGAM + PEM graph",
            graph_method="var_lingam_pem", candidate_links="road neighbours", prior="uniform",
            horizons=(1,), ready=True,
        ),
        ExperimentSpec(
            "E5", "A3T-GCN + horizon-specific VAR-LiNGAM + PEM graphs",
            graph_method="var_lingam_pem_horizon", candidate_links="road neighbours",
            prior="uniform", horizons=(1, 3, 6, 12), ready=True,
        ),
        ExperimentSpec(
            "E6", "A3T-GCN + VAR-LiNGAM + PEM, data-driven neighbourhoods",
            graph_method="var_lingam_pem_datadriven",
            candidate_links="most-correlated training sensors (no road information)",
            prior="uniform", horizons=(1,), ready=True,
        ),
        ExperimentSpec(
            "E7", "A3T-GCN + VAR-LiNGAM + PEM with road prior",
            graph_method="var_lingam_pem_roadprior", candidate_links="road neighbours",
            prior="road (0.7 / 0.3)", horizons=(1,), ready=True,
        ),
        # Baseline: same model, data and training, but no spatial
        # information. With an all-zero graph A3T-GCN's operator is
        # D^-1/2 (0 + I) D^-1/2 = I, so every sensor is a GRU with
        # attention over its own history (weights shared by sensors).
        ExperimentSpec(
            "NOGRAPH", "A3T-GCN without a graph (own history only)",
            graph_method="none", candidate_links="none", prior="-",
            horizons=(1,), ready=True,
        ),
    ]
}


# Graph artifacts written before artifacts recorded their method carry
# only the OLD experiment label. Old E2 (VAR + PEM) is the new E3.
LEGACY_ARTIFACT_METHODS = {
    "E1": "var",
    "E2": "var_pem",
}


def get_experiment(name: str) -> ExperimentSpec:

    if name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment {name!r}; choose one of {sorted(EXPERIMENTS)}."
        )

    return EXPERIMENTS[name]


def artifact_method(artifact_files: dict) -> str:
    """Method of a graph artifact (new key, or legacy experiment label)."""

    if "method" in artifact_files:
        return str(artifact_files["method"])

    label = str(artifact_files["experiment"])

    if label not in LEGACY_ARTIFACT_METHODS:
        raise ValueError(f"Graph artifact without method and unknown label {label!r}.")

    return LEGACY_ARTIFACT_METHODS[label]
