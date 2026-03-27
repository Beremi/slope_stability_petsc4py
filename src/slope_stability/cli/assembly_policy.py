"""Runner-side policies for choosing distributed assembly paths."""

from __future__ import annotations


def solver_supports_owned_distributed_matrices(solver_type: str) -> bool:
    """Return whether the configured solver can consume owned PETSc operators.

    ``DIRECT`` maps to the legacy SciPy direct solver, which expects local
    NumPy/SciPy matrices and cannot safely solve distributed PETSc submatrices.
    """

    return str(solver_type).strip().upper() != "DIRECT"


def use_owned_tangent_path(*, solver_type: str, mpi_distribute_by_nodes: bool) -> bool:
    """Return whether the runner should enable the owned constitutive path."""

    return bool(mpi_distribute_by_nodes and solver_supports_owned_distributed_matrices(solver_type))


def use_lightweight_mpi_elastic_path(*, solver_type: str, mpi_distribute_by_nodes: bool, constitutive_mode: str) -> bool:
    """Return whether the runner should assemble the elastic operator by owned rows."""

    return bool(
        use_owned_tangent_path(
            solver_type=solver_type,
            mpi_distribute_by_nodes=mpi_distribute_by_nodes,
        )
        and str(constitutive_mode).lower() != "global"
    )
