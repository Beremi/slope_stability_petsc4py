from __future__ import annotations

import numpy as np
import pytest

import slope_stability.linear.solver as solver_module
from slope_stability.linear.solver import PetscKSPFGMRESSolver
from slope_stability.linear.solver import PetscKSPMatlabDeflatedFGMRESSolver
from slope_stability.linear.solver import PetscMatlabExactDFGMRESSolver


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


def test_pmg_shell_apply_timing_does_not_force_full_manualmg_diagnostics() -> None:
    q_mask = np.array([[True, True], [True, True], [True, True]], dtype=bool)
    solver = PetscKSPFGMRESSolver(
        pc_type="HYPRE",
        q_mask=q_mask,
        coord=np.zeros((3, 2), dtype=np.float64),
        preconditioner_options={},
    )

    class _ExplodingManualMG:
        def diagnostics(self, *args, **kwargs):
            raise AssertionError("full manualmg diagnostics should not run on every apply")

    solver._pc_backend = "pmg_shell"
    solver._manualmg_context = _ExplodingManualMG()

    solver._record_preconditioner_apply_time(0.25)

    assert solver._preconditioner_diagnostics.preconditioner_apply_time_last == pytest.approx(0.25)
    assert solver._preconditioner_diagnostics.preconditioner_apply_time_total == pytest.approx(0.25)
    assert solver._manualmg_last_apply_info == {
        "manualmg_last_pc_apply_time_s": pytest.approx(0.25),
        "manualmg_last_phase": "solve",
    }


def test_pmg_apply_timing_does_not_force_full_pmg_diagnostics() -> None:
    q_mask = np.array([[True, True], [True, True], [True, True]], dtype=bool)
    solver = PetscKSPFGMRESSolver(
        pc_type="HYPRE",
        q_mask=q_mask,
        coord=np.zeros((3, 2), dtype=np.float64),
        preconditioner_options={},
    )

    solver._pc_backend = "pmg"
    solver._pmg_state = object()
    solver._pmg_collect_pc_diagnostics = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full pmg diagnostics should not run on every apply"))

    solver._record_preconditioner_apply_time(0.5)

    assert solver._preconditioner_diagnostics.preconditioner_apply_time_last == pytest.approx(0.5)
    assert solver._preconditioner_diagnostics.preconditioner_apply_time_total == pytest.approx(0.5)
    assert solver._pmg_last_apply_info == {
        "pmg_last_pc_apply_time_s": pytest.approx(0.5),
        "pmg_last_phase": "solve",
    }


def test_matlab_deflated_solver_uses_manualmg_context_directly_for_pmg_shell() -> None:
    q_mask = np.ones((3, 1), dtype=bool)
    solver = PetscKSPMatlabDeflatedFGMRESSolver(
        pc_type="HYPRE",
        q_mask=q_mask,
        coord=np.zeros((3, 1), dtype=np.float64),
        preconditioner_options={},
    )
    solver._pc_backend = "pmg_shell"

    class _DummyVec:
        def __init__(self, values):
            self.arr = np.array(values, dtype=np.float64)

        def set(self, value):
            self.arr.fill(float(value))

        def getArray(self, readonly=False):
            return self.arr

    class _ExplodingInnerKSP:
        def solve(self, rhs, out):
            raise AssertionError("pmg_shell path should bypass nested inner KSP solve")

    class _ManualMG:
        def __init__(self):
            self.calls = 0

        def apply(self, pc, rhs, out):
            self.calls += 1
            out.getArray(readonly=False)[...] = rhs.getArray(readonly=True) + 2.0

    solver._inner_ksp = _ExplodingInnerKSP()
    solver._manualmg_context = _ManualMG()
    rhs = _DummyVec([1.0, 2.0, 3.0])
    out = _DummyVec([0.0, 0.0, 0.0])

    solver._apply_inner_preconditioner_vecs(rhs, out)

    assert solver._manualmg_context.calls == 1
    assert np.allclose(out.getArray(readonly=True), np.array([3.0, 4.0, 5.0]))


def test_exact_dfgmres_pmg_shell_destroys_transient_preconditioner_vecs() -> None:
    q_mask = np.ones((3, 1), dtype=bool)
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="HYPRE",
        q_mask=q_mask,
        coord=np.zeros((3, 1), dtype=np.float64),
        preconditioner_options={},
    )
    solver._pc_backend = "pmg_shell"

    class _DummyVec:
        def __init__(self, size: int):
            self.arr = np.zeros(size, dtype=np.float64)
            self.destroyed = False

        def set(self, value):
            self.arr.fill(float(value))

        def getArray(self, readonly=False):
            return self.arr

        def destroy(self):
            self.destroyed = True

    class _DummyMat:
        def __init__(self, size: int):
            self.size = size
            self.created: list[_DummyVec] = []

        def createVecRight(self):
            vec = _DummyVec(self.size)
            self.created.append(vec)
            return vec

    class _ManualMG:
        def apply(self, pc, rhs, out):
            out.getArray(readonly=False)[...] = rhs.getArray(readonly=True) + 1.0

    solver._A_petsc = _DummyMat(3)
    solver._manualmg_context = _ManualMG()

    out = solver._apply_inner_preconditioner_local(np.array([1.0, 2.0, 3.0], dtype=np.float64))

    assert np.allclose(out, np.array([2.0, 3.0, 4.0], dtype=np.float64))
    assert len(solver._A_petsc.created) == 2
    assert all(vec.destroyed for vec in solver._A_petsc.created)


def test_exact_dfgmres_pmg_shell_solve_runs_cleanup_guard(monkeypatch) -> None:
    q_mask = np.ones((3, 1), dtype=bool)
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="HYPRE",
        q_mask=q_mask,
        coord=np.zeros((3, 1), dtype=np.float64),
        preconditioner_options={},
    )
    solver._pc_backend = "pmg_shell"
    solver._projector_dirty = False
    solver._diagnostics_enabled = False
    solver._inner_ksp = object()
    solver._ownership_range = (0, 3)
    solver._mpi_comm = None
    solver._restrict_solution = lambda x: np.asarray(x, dtype=np.float64)
    solver._prepare_rhs = lambda b, full_rhs=None: np.asarray(b, dtype=np.float64)

    class _DummyComm:
        def getSize(self):
            return 1

    class _DummyMat:
        def getComm(self):
            return _DummyComm()

    solver._A_petsc = _DummyMat()

    cleanup_calls: list[object] = []
    gc_calls: list[str] = []

    monkeypatch.setattr(solver_module, "dfgmres_matlab_exact", lambda *args, **kwargs: (np.array([1.0, 2.0, 3.0]), 2, np.array([1.0, 0.5, 0.1])))
    monkeypatch.setattr(solver_module, "_petsc_garbage_cleanup", lambda comm=None: cleanup_calls.append(comm))
    monkeypatch.setattr(solver_module.gc, "isenabled", lambda: True)
    monkeypatch.setattr(solver_module.gc, "disable", lambda: gc_calls.append("disable"))
    monkeypatch.setattr(solver_module.gc, "enable", lambda: gc_calls.append("enable"))
    monkeypatch.setattr(solver_module.gc, "collect", lambda: gc_calls.append("collect"))

    out = solver.solve(None, np.array([4.0, 5.0, 6.0], dtype=np.float64))

    assert np.allclose(out, np.array([1.0, 2.0, 3.0], dtype=np.float64))
    assert gc_calls == ["disable", "enable", "collect"]
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0].getSize() == 1
