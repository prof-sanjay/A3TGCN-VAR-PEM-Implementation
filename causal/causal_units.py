"""
Local causal-unit construction for topology-aware causal discovery.

A causal unit contains one target sensor and the sensors connected to it
through the road-network adjacency.

The incoming and outgoing topology neighborhoods are combined:

    U_i = {i} U N_out(i) U N_in(i)

The road topology is used only to restrict the candidate variables.
It does NOT determine causal direction.

This module does NOT modify traffic data and does NOT perform imputation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CausalUnit:
    """
    Represents one local causal-discovery unit.

    Attributes
    ----------
    target:
        Global sensor index of the target sensor.

    nodes:
        Global sensor indices included in this causal unit.

    incoming_neighbors:
        Sensors j for which adjacency[j, target] != 0.

    outgoing_neighbors:
        Sensors j for which adjacency[target, j] != 0.

    neighbors:
        Union of incoming and outgoing neighbors.
    """

    target: int
    nodes: np.ndarray
    incoming_neighbors: np.ndarray
    outgoing_neighbors: np.ndarray
    neighbors: np.ndarray


class CausalUnitBuilder:
    """
    Build local causal-discovery units from a road-network adjacency matrix.

    For each target sensor i:

        outgoing = {j : A[i, j] != 0}
        incoming = {j : A[j, i] != 0}

        neighbors = outgoing U incoming

        unit = {i} U neighbors

    The direction of the road adjacency is therefore NOT interpreted as
    causal direction.

    Parameters
    ----------
    include_self:
        Whether to include the target sensor in its own causal unit.

    max_unit_size:
        Optional maximum number of variables in a causal unit.

        If None, all topology-connected variables are retained.

        If specified, the strongest topology connections are retained
        according to the maximum absolute adjacency weight considering
        both incoming and outgoing connections.

    remove_self_loops:
        Whether diagonal adjacency entries should be ignored when
        identifying topology neighbors.
    """

    def __init__(
        self,
        include_self: bool = True,
        max_unit_size: int | None = None,
        remove_self_loops: bool = True,
    ) -> None:

        if max_unit_size is not None and max_unit_size < 1:
            raise ValueError(
                "max_unit_size must be >= 1 or None."
            )

        self.include_self = include_self
        self.max_unit_size = max_unit_size
        self.remove_self_loops = remove_self_loops

    def _validate_adjacency(
        self,
        adjacency: np.ndarray,
    ) -> None:
        """Validate the supplied adjacency matrix."""

        if not isinstance(adjacency, np.ndarray):
            raise TypeError(
                "adjacency must be a NumPy array."
            )

        if adjacency.ndim != 2:
            raise ValueError(
                "adjacency must be 2-dimensional, "
                f"got {adjacency.ndim}D."
            )

        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(
                "adjacency must be square. "
                f"Got shape {adjacency.shape}."
            )

        if not np.all(np.isfinite(adjacency)):
            raise ValueError(
                "adjacency contains NaN or infinite values."
            )

    def _get_neighbors(
        self,
        adjacency: np.ndarray,
        target: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get incoming, outgoing, and combined topology neighbors.
        """

        # ---------------------------------------------------------
        # Outgoing topology neighbors
        #
        # A[target, source] != 0
        # ---------------------------------------------------------

        outgoing = np.flatnonzero(
            adjacency[target] != 0
        )

        # ---------------------------------------------------------
        # Incoming topology neighbors
        #
        # A[source, target] != 0
        # ---------------------------------------------------------

        incoming = np.flatnonzero(
            adjacency[:, target] != 0
        )

        # ---------------------------------------------------------
        # Remove target itself.
        #
        # Self-loops are not treated as separate causal neighbors.
        # ---------------------------------------------------------

        if self.remove_self_loops:

            outgoing = outgoing[
                outgoing != target
            ]

            incoming = incoming[
                incoming != target
            ]

        # ---------------------------------------------------------
        # Union of incoming and outgoing neighbors.
        # ---------------------------------------------------------

        neighbors = np.union1d(
            incoming,
            outgoing,
        )

        # ---------------------------------------------------------
        # Optional unit-size restriction.
        #
        # We rank candidate neighbors by their strongest connection
        # to the target, considering both directions.
        # ---------------------------------------------------------

        if self.max_unit_size is not None:

            max_neighbors = self.max_unit_size - (
                1 if self.include_self else 0
            )

            if max_neighbors < 0:
                raise ValueError(
                    "max_unit_size must be at least 1 "
                    "when include_self=True."
                )

            if len(neighbors) > max_neighbors:

                weights = np.maximum(
                    np.abs(adjacency[target, neighbors]),
                    np.abs(adjacency[neighbors, target]),
                )

                order = np.argsort(
                    -weights,
                    kind="stable",
                )

                neighbors = neighbors[
                    order[:max_neighbors]
                ]

                neighbors = np.sort(neighbors)

        return (
            incoming.astype(np.int64),
            outgoing.astype(np.int64),
            neighbors.astype(np.int64),
        )

    def build(
        self,
        adjacency: np.ndarray,
    ) -> list[CausalUnit]:
        """
        Construct one local causal unit for every sensor.
        """

        self._validate_adjacency(adjacency)

        num_nodes = adjacency.shape[0]

        units: list[CausalUnit] = []

        for target in range(num_nodes):

            (
                incoming,
                outgoing,
                neighbors,
            ) = self._get_neighbors(
                adjacency=adjacency,
                target=target,
            )

            if self.include_self:

                nodes = np.concatenate(
                    (
                        np.array(
                            [target],
                            dtype=np.int64,
                        ),
                        neighbors,
                    )
                )

            else:

                nodes = neighbors.copy()

            units.append(
                CausalUnit(
                    target=target,
                    nodes=nodes,
                    incoming_neighbors=incoming,
                    outgoing_neighbors=outgoing,
                    neighbors=neighbors,
                )
            )

        return units

    @staticmethod
    def summary(
        units: list[CausalUnit],
    ) -> dict:
        """
        Calculate summary statistics for causal units.
        """

        if not units:
            raise ValueError(
                "units must not be empty."
            )

        sizes = np.array(
            [
                len(unit.nodes)
                for unit in units
            ],
            dtype=np.int64,
        )

        neighbor_counts = np.array(
            [
                len(unit.neighbors)
                for unit in units
            ],
            dtype=np.int64,
        )

        incoming_counts = np.array(
            [
                len(unit.incoming_neighbors)
                for unit in units
            ],
            dtype=np.int64,
        )

        outgoing_counts = np.array(
            [
                len(unit.outgoing_neighbors)
                for unit in units
            ],
            dtype=np.int64,
        )

        return {
            "num_units": len(units),

            "min_unit_size": int(
                np.min(sizes)
            ),

            "max_unit_size": int(
                np.max(sizes)
            ),

            "mean_unit_size": float(
                np.mean(sizes)
            ),

            "median_unit_size": float(
                np.median(sizes)
            ),

            "min_neighbors": int(
                np.min(neighbor_counts)
            ),

            "max_neighbors": int(
                np.max(neighbor_counts)
            ),

            "mean_neighbors": float(
                np.mean(neighbor_counts)
            ),

            "median_neighbors": float(
                np.median(neighbor_counts)
            ),

            "mean_incoming_neighbors": float(
                np.mean(incoming_counts)
            ),

            "mean_outgoing_neighbors": float(
                np.mean(outgoing_counts)
            ),
        }


def build_causal_units(
    adjacency: np.ndarray,
    include_self: bool = True,
    max_unit_size: int | None = None,
    remove_self_loops: bool = True,
) -> list[CausalUnit]:
    """
    Convenience function for constructing causal units.
    """

    builder = CausalUnitBuilder(
        include_self=include_self,
        max_unit_size=max_unit_size,
        remove_self_loops=remove_self_loops,
    )

    return builder.build(adjacency)