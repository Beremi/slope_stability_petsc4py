from __future__ import annotations

from slope_stability.cli.assembly_policy import use_lightweight_mpi_elastic_path, use_owned_tangent_path


def test_direct_solver_stays_off_owned_tangent_path() -> None:
    assert use_owned_tangent_path(solver_type="DIRECT", mpi_distribute_by_nodes=True) is False


def test_direct_solver_stays_off_lightweight_elastic_path() -> None:
    assert (
        use_lightweight_mpi_elastic_path(
            solver_type="DIRECT",
            mpi_distribute_by_nodes=True,
            constitutive_mode="overlap",
        )
        is False
    )


def test_petsc_solver_keeps_owned_paths_when_requested() -> None:
    assert use_owned_tangent_path(
        solver_type="PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        mpi_distribute_by_nodes=True,
    )
    assert use_lightweight_mpi_elastic_path(
        solver_type="PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        mpi_distribute_by_nodes=True,
        constitutive_mode="overlap",
    )


def test_global_constitutive_mode_disables_only_lightweight_elastic_path() -> None:
    assert use_owned_tangent_path(solver_type="PETSC_DIRECT", mpi_distribute_by_nodes=True)
    assert (
        use_lightweight_mpi_elastic_path(
            solver_type="PETSC_DIRECT",
            mpi_distribute_by_nodes=True,
            constitutive_mode="global",
        )
        is False
    )
