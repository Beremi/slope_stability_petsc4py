from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from slope_stability.core.config import LinearSolverConfig
from slope_stability.linear.solver import PetscMatlabExactDFGMRESSolver
from slope_stability.linear.solver import SolverFactory


def _tiny_solver(**options) -> PetscMatlabExactDFGMRESSolver:
    return PetscMatlabExactDFGMRESSolver(
        pc_type="JACOBI",
        q_mask=np.array([[True, True]], dtype=bool),
        coord=np.zeros((1, 2), dtype=np.float64),
        preconditioner_options=options,
    )


def _tiny_matrix(scale: float = 1.0) -> csr_matrix:
    base = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    return csr_matrix(scale * base)


def test_current_policy_rebuilds_every_setup() -> None:
    solver = _tiny_solver(
        pc_backend="jacobi",
        preconditioner_matrix_policy="current",
        preconditioner_rebuild_policy="every_newton",
    )
    A = _tiny_matrix()

    solver.setup_preconditioner(A, full_matrix=A)
    solver.release_iteration_resources()
    solver.setup_preconditioner(A, full_matrix=A)

    diagnostics = solver.get_preconditioner_diagnostics()
    assert diagnostics["preconditioner_rebuild_count"] == 2
    assert diagnostics["preconditioner_reuse_count"] == 0
    assert diagnostics["preconditioner_last_rebuild_reason"] == "current_policy"


def test_lagged_policy_reuses_until_success_trigger() -> None:
    solver = _tiny_solver(
        pc_backend="jacobi",
        preconditioner_matrix_policy="lagged",
        preconditioner_rebuild_policy="accepted_step",
    )
    A = _tiny_matrix()

    solver.setup_preconditioner(A, full_matrix=A)
    solver.release_iteration_resources()
    solver.setup_preconditioner(A, full_matrix=A)
    solver.notify_continuation_attempt(success=True)
    solver.release_iteration_resources()
    solver.setup_preconditioner(A, full_matrix=A)

    diagnostics = solver.get_preconditioner_diagnostics()
    assert diagnostics["preconditioner_rebuild_count"] == 2
    assert diagnostics["preconditioner_reuse_count"] == 1
    assert diagnostics["preconditioner_age_max"] >= 1
    assert diagnostics["preconditioner_last_rebuild_reason"] == "attempt_trigger"


def test_every_n_newton_policy_rebuilds_on_interval() -> None:
    solver = _tiny_solver(
        pc_backend="jacobi",
        preconditioner_matrix_policy="lagged",
        preconditioner_rebuild_policy="every_n_newton",
        preconditioner_rebuild_interval=2,
    )
    A = _tiny_matrix()

    solver.setup_preconditioner(A, full_matrix=A)
    solver.release_iteration_resources()
    solver.setup_preconditioner(A, full_matrix=A)
    solver.release_iteration_resources()
    solver.setup_preconditioner(A, full_matrix=A)

    diagnostics = solver.get_preconditioner_diagnostics()
    assert diagnostics["preconditioner_rebuild_count"] == 2
    assert diagnostics["preconditioner_reuse_count"] == 1


def test_deflation_basis_snapshot_restore_roundtrip() -> None:
    solver = _tiny_solver(pc_backend="jacobi")
    solver.expand_deflation_basis(np.array([1.0, 2.0], dtype=np.float64))
    snapshot = solver.get_deflation_basis_snapshot()
    solver.expand_deflation_basis(np.array([3.0, 4.0], dtype=np.float64))

    solver.restore_deflation_basis(snapshot)

    assert solver.deflation_basis.shape == (2, 1)
    assert np.allclose(solver.deflation_basis[:, 0], np.array([1.0, 2.0], dtype=np.float64))


def test_from_config_propagates_bddc_local_solver_options() -> None:
    config = LinearSolverConfig(
        solver_type="PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE",
        pc_backend="bddc",
        preconditioner_matrix_source="elastic",
        pc_bddc_dirichlet_ksp_type="preonly",
        pc_bddc_dirichlet_pc_type="ilu",
        pc_bddc_neumann_ksp_type="preonly",
        pc_bddc_neumann_pc_type="ilu",
        pc_bddc_coarse_ksp_type="preonly",
        pc_bddc_coarse_pc_type="gamg",
        pc_bddc_dirichlet_approximate=True,
        pc_bddc_neumann_approximate=True,
        pc_bddc_switch_static=True,
        pc_bddc_use_deluxe_scaling=False,
        pc_bddc_use_vertices=True,
        pc_bddc_use_edges=True,
        pc_bddc_use_faces=True,
        pc_bddc_use_change_of_basis=True,
        pc_bddc_use_change_on_faces=True,
        pc_bddc_check_level=2,
    )

    solver = SolverFactory.from_config(
        config,
        q_mask=np.array([[True, True]], dtype=bool),
        coord=np.zeros((1, 2), dtype=np.float64),
    )

    assert solver.preconditioner_options["preconditioner_matrix_source"] == "elastic"
    assert solver.preconditioner_options["pc_bddc_dirichlet_ksp_type"] == "preonly"
    assert solver.preconditioner_options["pc_bddc_dirichlet_pc_type"] == "ilu"
    assert solver.preconditioner_options["pc_bddc_neumann_ksp_type"] == "preonly"
    assert solver.preconditioner_options["pc_bddc_neumann_pc_type"] == "ilu"
    assert solver.preconditioner_options["pc_bddc_coarse_ksp_type"] == "preonly"
    assert solver.preconditioner_options["pc_bddc_coarse_pc_type"] == "gamg"
    assert solver.preconditioner_options["pc_bddc_dirichlet_approximate"] is True
    assert solver.preconditioner_options["pc_bddc_neumann_approximate"] is True
    assert solver.preconditioner_options["pc_bddc_switch_static"] is True
    assert solver.preconditioner_options["pc_bddc_use_deluxe_scaling"] is False
    assert solver.preconditioner_options["pc_bddc_use_vertices"] is True
    assert solver.preconditioner_options["pc_bddc_use_edges"] is True
    assert solver.preconditioner_options["pc_bddc_use_faces"] is True
    assert solver.preconditioner_options["pc_bddc_use_change_of_basis"] is True
    assert solver.preconditioner_options["pc_bddc_use_change_on_faces"] is True
    assert solver.preconditioner_options["pc_bddc_check_level"] == 2
