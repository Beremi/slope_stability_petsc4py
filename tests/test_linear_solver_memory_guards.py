from __future__ import annotations

import numpy as np

from slope_stability.linear.solver import PetscKSPFGMRESSolver


def test_petsc_solver_caps_deflation_basis_columns() -> None:
    q_mask = np.array([[True, False, True], [True, True, False]], dtype=bool)
    solver = PetscKSPFGMRESSolver(
        pc_type="HYPRE",
        q_mask=q_mask,
        coord=np.zeros((2, 3), dtype=np.float64),
        preconditioner_options={"max_deflation_basis_vectors": 3},
    )

    free_size = int(np.count_nonzero(q_mask))
    for idx in range(5):
        solver.expand_deflation_basis(np.full(free_size, float(idx), dtype=np.float64))

    assert solver.deflation_basis.shape == (q_mask.size, 3)
    assert np.allclose(solver.deflation_basis[q_mask.reshape(-1, order="F"), :], np.array([[2.0, 3.0, 4.0]] * free_size))
    assert np.allclose(solver.deflation_basis[~q_mask.reshape(-1, order="F"), :], 0.0)


def test_petsc_solver_copy_shares_basis_until_clone_changes_it() -> None:
    q_mask = np.array([[True, False, True], [True, True, False]], dtype=bool)
    solver = PetscKSPFGMRESSolver(
        pc_type="HYPRE",
        q_mask=q_mask,
        coord=np.zeros((2, 3), dtype=np.float64),
        preconditioner_options={"max_deflation_basis_vectors": 4},
    )

    free_size = int(np.count_nonzero(q_mask))
    solver.expand_deflation_basis(np.arange(free_size, dtype=np.float64))
    clone = solver.copy()

    assert clone.deflation_basis is solver.deflation_basis

    clone.expand_deflation_basis(np.full(free_size, 9.0, dtype=np.float64))

    assert clone.deflation_basis is not solver.deflation_basis
    assert solver.deflation_basis.shape == (q_mask.size, 1)
    assert clone.deflation_basis.shape == (q_mask.size, 2)


def test_petsc_solver_zero_max_deflation_disables_recycling_and_orthogonalization() -> None:
    q_mask = np.array([[True, False, True], [True, True, False]], dtype=bool)
    solver = PetscKSPFGMRESSolver(
        pc_type="HYPRE",
        q_mask=q_mask,
        coord=np.zeros((2, 3), dtype=np.float64),
        preconditioner_options={"max_deflation_basis_vectors": 0},
    )

    free_size = int(np.count_nonzero(q_mask))
    solver.expand_deflation_basis(np.arange(free_size, dtype=np.float64))

    assert solver.supports_dynamic_deflation_basis() is False
    assert solver.supports_a_orthogonalization() is False
    assert solver.deflation_basis.shape == (q_mask.size, 0)
