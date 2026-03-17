"""Solver facade objects used by nonlinear and continuation modules."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from mpi4py import MPI

from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve

try:
    from petsc4py import PETSc
except Exception:  # pragma: no cover - optional when PETSc is unavailable
    PETSc = None

from ..core.config import LinearSolverConfig
from ..utils import (
    bddc_pc_coordinates_from_metadata,
    get_petsc_matrix_metadata,
    global_array_to_petsc_vec,
    owned_block_range,
    petsc_vec_to_global_array,
    q_to_free_indices,
    release_petsc_aij_matrix,
    to_petsc_aij_matrix,
    to_scipy_csr_from_petsc,
)
from .collector import IterationCollector
from .deflated_fgmres import dfgmres, dfgmres_matlab_exact, dfgmres_matlab_exact_distributed, dfgmres_matlab_exact_distributed_compiled
from .elasticity import impose_zero_dirichlet_full_system
from .orthogonalize import a_orthogonalize, a_orthogonalize_with_info, a_orthogonalize_with_local_metadata
from .preconditioners import attach_near_nullspace, build_preconditioner, make_near_nullspace_elasticity


PreconditionerFactory = Callable[[object], Callable[[np.ndarray], np.ndarray]]


@dataclass
class PreconditionerDiagnostics:
    pc_backend: str
    preconditioner_matrix_source: str
    preconditioner_matrix_policy: str
    preconditioner_rebuild_policy: str
    preconditioner_rebuild_interval: int
    preconditioner_rebuild_count: int = 0
    preconditioner_reuse_count: int = 0
    preconditioner_age_max: int = 0
    preconditioner_setup_time_total: float = 0.0
    preconditioner_apply_time_total: float = 0.0
    preconditioner_last_rebuild_reason: str = "initial"

    def as_dict(self) -> dict[str, object]:
        return {
            "pc_backend": str(self.pc_backend),
            "preconditioner_matrix_source": str(self.preconditioner_matrix_source),
            "preconditioner_matrix_policy": str(self.preconditioner_matrix_policy),
            "preconditioner_rebuild_policy": str(self.preconditioner_rebuild_policy),
            "preconditioner_rebuild_interval": int(self.preconditioner_rebuild_interval),
            "preconditioner_rebuild_count": int(self.preconditioner_rebuild_count),
            "preconditioner_reuse_count": int(self.preconditioner_reuse_count),
            "preconditioner_age_max": int(self.preconditioner_age_max),
            "preconditioner_setup_time_total": float(self.preconditioner_setup_time_total),
            "preconditioner_apply_time_total": float(self.preconditioner_apply_time_total),
            "preconditioner_last_rebuild_reason": str(self.preconditioner_last_rebuild_reason),
        }


def _mat_scale_add(alpha: float, A, beta: float, B):
    if alpha == 0 and beta == 1:
        return B
    if beta == 0 and alpha == 1:
        return A

    if PETSc is not None and isinstance(A, PETSc.Mat) and isinstance(B, PETSc.Mat):
        C = A.copy()
        C.scale(alpha)
        C.axpy(beta, B)
        C.assemble()
        return C

    if isinstance(A, np.ndarray) or issparse(A):
        return alpha * np.asarray(A) + beta * np.asarray(B)

    if hasattr(A, "__array__"):
        return alpha * np.asarray(A) + beta * np.asarray(B)

    raise TypeError("Unsupported matrix type for linear combination")


def _to_numpy_array(v) -> np.ndarray:
    if hasattr(v, "getArray"):
        return np.asarray(v.getArray(), dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def _matvec(A, x: np.ndarray) -> np.ndarray:
    if issparse(A):
        return A @ x
    if isinstance(A, np.ndarray):
        return A @ x
    if PETSc is not None and isinstance(A, PETSc.Mat):
        xv = global_array_to_petsc_vec(
            x,
            comm=A.getComm(),
            ownership_range=A.getOwnershipRange() if int(A.getComm().getSize()) > 1 else None,
            bsize=A.getBlockSize() or None,
        )
        y = A.createVecRight()
        A.mult(xv, y)
        return petsc_vec_to_global_array(y)
    if callable(A):
        return np.asarray(A(x), dtype=np.float64)
    if hasattr(A, "dot"):
        return np.asarray(A.dot(x), dtype=np.float64)
    raise TypeError("Unsupported matrix type")


def _coarse_deflation_correction(A, b: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if basis.size == 0:
        return np.zeros_like(b), np.asarray(b, dtype=np.float64)
    x0 = basis @ (basis.T @ np.asarray(b, dtype=np.float64))
    rhs = np.asarray(b, dtype=np.float64) - _matvec(A, x0)
    return x0, rhs


def _basis_diagnostics(basis: np.ndarray, A=None, *, basis_A: np.ndarray | None = None) -> dict[str, float | int]:
    raw = np.asarray(basis, dtype=np.float64)
    if raw.size == 0:
        rows = int(raw.shape[0]) if raw.ndim >= 1 else 0
        return {
            "basis_rows": rows,
            "basis_cols": 0,
            "diag_min": np.nan,
            "diag_max": np.nan,
            "negative_diag_count": 0,
            "offdiag_max_abs": 0.0,
            "offdiag_fro": 0.0,
            "signed_identity_fro": 0.0,
        }
    if raw.ndim == 1:
        raw = raw[:, None]
    if basis_A is None:
        basis_A = np.column_stack([_matvec(A, raw[:, j]) for j in range(raw.shape[1])])
    gram = raw.T @ np.asarray(basis_A, dtype=np.float64)
    diag = np.diag(gram).astype(np.float64, copy=False)
    signed = np.sign(diag)
    signed[signed == 0.0] = 1.0
    offdiag = gram - np.diag(diag)
    return {
        "basis_rows": int(raw.shape[0]),
        "basis_cols": int(raw.shape[1]),
        "diag_min": float(diag.min()) if diag.size else np.nan,
        "diag_max": float(diag.max()) if diag.size else np.nan,
        "negative_diag_count": int(np.count_nonzero(diag < 0.0)),
        "offdiag_max_abs": float(np.max(np.abs(offdiag))) if offdiag.size else 0.0,
        "offdiag_fro": float(np.linalg.norm(offdiag)),
        "signed_identity_fro": float(np.linalg.norm(gram - np.diag(signed))),
    }


class DirectSolver:
    """Direct solver wrapper using PETSc LU (or dense NumPy fallback)."""

    def __init__(self, *, factor_solver_type: str | None = None) -> None:
        self.iteration_collector = IterationCollector()
        self.instance_id = self.iteration_collector.register_instance()
        self._ksp = None
        self._A_petsc = None
        self.factor_solver_type = factor_solver_type

    def setup_preconditioner(self, A, *, preconditioning_matrix=None, **_kwargs):
        t0 = perf_counter()

        if PETSc is None:
            self._ksp = None
            self._A_petsc = None
            self.iteration_collector.store_preconditioner_time(self.instance_id, perf_counter() - t0)
            return

        if self._ksp is not None:
            self._ksp.destroy()
            self._ksp = None
        if self._A_petsc is not None:
            release_petsc_aij_matrix(self._A_petsc)
            self._A_petsc.destroy()
            self._A_petsc = None
        self._A_petsc = to_petsc_aij_matrix(A, comm=A.getComm() if hasattr(A, "getComm") else PETSc.COMM_SELF)
        self._ksp = PETSc.KSP().create(comm=self._A_petsc.getComm())
        self._ksp.setOperators(self._A_petsc)
        self._ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = self._ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        if self.factor_solver_type:
            pc.setFactorSolverType(self.factor_solver_type)
        self._ksp.setUp()
        self.iteration_collector.store_preconditioner_time(self.instance_id, perf_counter() - t0)

    def A_orthogonalize(self, A):
        # Not applicable for direct solver.
        return

    def solve(self, A, b):
        b = np.asarray(b, dtype=np.float64)
        t0 = perf_counter()

        if PETSc is not None and (isinstance(A, PETSc.Mat) or issparse(A)):
            if self._ksp is None or self._A_petsc is None:
                self.setup_preconditioner(A)
            rhs = PETSc.Vec().createWithArray(b, comm=self._A_petsc.getComm())
            x = self._A_petsc.createVecRight()
            x.set(0.0)
            self._ksp.solve(rhs, x)
            solution = np.asarray(x.getArray(readonly=False)).copy()
        elif issparse(A) or isinstance(A, np.ndarray):
            solution = spsolve(to_scipy_csr_from_petsc(A), b) if issparse(A) else np.linalg.solve(A, b)
        else:
            solution = np.linalg.solve(np.asarray(A), b)

        elapsed = perf_counter() - t0
        nit = int(self._ksp.getIterationNumber()) if self._ksp is not None else 1
        self.iteration_collector.store_iteration(self.instance_id, max(1, nit), elapsed)
        return solution

    def expand_deflation_basis(self, additional_vectors):
        # Direct solver does not use deflation.
        return

    def copy(self):
        clone = self.__class__(factor_solver_type=self.factor_solver_type)
        clone.iteration_collector = self.iteration_collector
        clone.instance_id = self.iteration_collector.register_instance()
        return clone

    def prefers_full_system_operator(self) -> bool:
        return False

    def preconditioner_requires_explicit_matrix(self) -> bool:
        return False

    def notify_continuation_attempt(self, *, success: bool) -> None:
        return

    def get_preconditioner_diagnostics(self) -> dict[str, object]:
        return {
            "pc_backend": "lu",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "preconditioner_rebuild_count": 0,
            "preconditioner_reuse_count": 0,
            "preconditioner_age_max": 0,
            "preconditioner_setup_time_total": 0.0,
            "preconditioner_apply_time_total": 0.0,
            "preconditioner_last_rebuild_reason": "direct",
        }

    def get_deflation_basis_snapshot(self):
        return None

    def restore_deflation_basis(self, _snapshot) -> None:
        return

    def release_iteration_resources(self) -> None:
        if self._ksp is not None:
            self._ksp.destroy()
            self._ksp = None
        if self._A_petsc is not None:
            release_petsc_aij_matrix(self._A_petsc)
            self._A_petsc.destroy()
            self._A_petsc = None


class ScipyDirectSolver:
    """Legacy direct solver wrapper using SciPy/NumPy factorization."""

    def __init__(self) -> None:
        self.iteration_collector = IterationCollector()
        self.instance_id = self.iteration_collector.register_instance()

    def setup_preconditioner(self, A, *, preconditioning_matrix=None, **_kwargs):
        self.iteration_collector.store_preconditioner_time(self.instance_id, 0.0)

    def A_orthogonalize(self, A):
        return

    def solve(self, A, b):
        b = np.asarray(b, dtype=np.float64)
        t0 = perf_counter()
        if issparse(A):
            solution = spsolve(to_scipy_csr_from_petsc(A), b)
        elif PETSc is not None and isinstance(A, PETSc.Mat):
            solution = spsolve(to_scipy_csr_from_petsc(A), b)
        elif isinstance(A, np.ndarray):
            solution = np.linalg.solve(A, b)
        else:
            solution = np.linalg.solve(np.asarray(A), b)
        self.iteration_collector.store_iteration(self.instance_id, 1, perf_counter() - t0)
        return solution

    def expand_deflation_basis(self, additional_vectors):
        return

    def copy(self):
        clone = self.__class__()
        clone.iteration_collector = self.iteration_collector
        clone.instance_id = self.iteration_collector.register_instance()
        return clone

    def prefers_full_system_operator(self) -> bool:
        return False

    def preconditioner_requires_explicit_matrix(self) -> bool:
        return False

    def notify_continuation_attempt(self, *, success: bool) -> None:
        return

    def get_preconditioner_diagnostics(self) -> dict[str, object]:
        return {
            "pc_backend": "scipy_direct",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "preconditioner_rebuild_count": 0,
            "preconditioner_reuse_count": 0,
            "preconditioner_age_max": 0,
            "preconditioner_setup_time_total": 0.0,
            "preconditioner_apply_time_total": 0.0,
            "preconditioner_last_rebuild_reason": "direct",
        }

    def get_deflation_basis_snapshot(self):
        return None

    def restore_deflation_basis(self, _snapshot) -> None:
        return

    def release_iteration_resources(self) -> None:
        return


class DeflatedFGMRESSolver:
    """Deflated Flexible GMRES solver.

    API is intentionally compatible with the legacy MATLAB ``DFGMRES`` object.
    """

    def __init__(
        self,
        preconditioner_builder: PreconditionerFactory,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        tolerance_deflation_basis: float = 1e-3,
        verbose: bool = False,
    ) -> None:
        self.deflation_basis: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.preconditioner_builder = preconditioner_builder
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.tolerance_deflation_basis = float(tolerance_deflation_basis)
        self.verbose = bool(verbose)
        self.preconditioner = None

        self.iteration_collector = IterationCollector()
        self.instance_id = self.iteration_collector.register_instance()
        self._diagnostics_enabled = False
        self._last_solve_info: dict[str, object] = {}
        self._last_orthogonalization_info: dict[str, object] = {}

    def setup_preconditioner(self, A, *, preconditioning_matrix=None, **_kwargs):
        t0 = perf_counter()
        self.preconditioner = self.setup_preconditioner_core(A)
        self.iteration_collector.store_preconditioner_time(self.instance_id, perf_counter() - t0)

    def setup_preconditioner_core(self, A):
        return self.preconditioner_builder(A)

    def A_orthogonalize(self, A):
        t0 = perf_counter()
        before_cols = int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else int(bool(self.deflation_basis.size))
        self.deflation_basis, basis_norms = a_orthogonalize_with_info(self.deflation_basis, A, self.tolerance_deflation_basis)
        elapsed = perf_counter() - t0
        self.iteration_collector.store_orthogonalization_time(self.instance_id, elapsed)
        if self._diagnostics_enabled:
            self._last_orthogonalization_info = {
                "time_s": float(elapsed),
                "basis_cols_before": int(before_cols),
                "basis_cols_after": int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else 0,
                "basis_norm_signs": np.sign(basis_norms).astype(np.float64).tolist(),
                **_basis_diagnostics(self.deflation_basis, A),
            }

    def _as_prec(self):
        if self.preconditioner is None:
            return lambda x: np.asarray(x, dtype=np.float64)
        return self.preconditioner

    def solve(self, A, b):
        t0 = perf_counter()
        rhs = np.asarray(b, dtype=np.float64)
        x, nit, res_hist = self.solve_core(A, rhs)
        elapsed = perf_counter() - t0
        self.iteration_collector.store_iteration(self.instance_id, int(nit), elapsed)
        if self._diagnostics_enabled:
            self._last_solve_info = {
                "iterations": int(nit),
                "time_s": float(elapsed),
                "basis_cols": int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else 0,
                "rhs_norm": float(np.linalg.norm(rhs)),
                "true_residual_history": np.asarray(res_hist, dtype=np.float64).tolist(),
            }
        if self.verbose:
            print(f"{nit}|", end="")
        return x

    def solve_core(self, A, b):
        preconditioner = self._as_prec()
        if self.deflation_basis.size == 0:
            basis = np.empty((np.asarray(b).size, 0), dtype=np.float64)
        else:
            basis = self.deflation_basis
        x, nit, res_hist = dfgmres(
            A,
            np.asarray(b, dtype=np.float64),
            preconditioner,
            basis,
            self.max_iterations,
            self.tolerance,
            np.zeros_like(np.asarray(b, dtype=np.float64)),
        )
        return x, nit, res_hist

    def expand_deflation_basis(self, additional_vectors):
        if additional_vectors is None:
            return
        v = np.asarray(additional_vectors, dtype=np.float64)
        if v.size == 0:
            return
        if v.ndim == 1:
            v = v[:, None]
        if self.deflation_basis.size == 0:
            self.deflation_basis = np.asarray(v, dtype=np.float64)
        else:
            self.deflation_basis = np.hstack((self.deflation_basis, v))

    def copy(self):
        clone = self.__class__(
            self.preconditioner_builder,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            tolerance_deflation_basis=self.tolerance_deflation_basis,
            verbose=self.verbose,
        )
        # Share the current basis until the clone appends/reorthogonalizes and rebinds it.
        clone.deflation_basis = self.deflation_basis
        clone.preconditioner = self.preconditioner
        clone.iteration_collector = self.iteration_collector
        clone.instance_id = self.iteration_collector.register_instance()
        clone._diagnostics_enabled = self._diagnostics_enabled
        return clone

    def enable_diagnostics(self, enabled: bool = True) -> None:
        self._diagnostics_enabled = bool(enabled)

    def get_last_solve_info(self) -> dict[str, object]:
        return dict(self._last_solve_info)

    def get_last_orthogonalization_info(self) -> dict[str, object]:
        return dict(self._last_orthogonalization_info)

    def prefers_full_system_operator(self) -> bool:
        return False

    def preconditioner_requires_explicit_matrix(self) -> bool:
        return False

    def notify_continuation_attempt(self, *, success: bool) -> None:
        return

    def get_preconditioner_diagnostics(self) -> dict[str, object]:
        return {
            "pc_backend": "python",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "preconditioner_rebuild_count": 0,
            "preconditioner_reuse_count": 0,
            "preconditioner_age_max": 0,
            "preconditioner_setup_time_total": 0.0,
            "preconditioner_apply_time_total": 0.0,
            "preconditioner_last_rebuild_reason": "python",
        }

    def get_deflation_basis_snapshot(self):
        return np.array(self.deflation_basis, dtype=np.float64, copy=True)

    def restore_deflation_basis(self, snapshot) -> None:
        if snapshot is None:
            self.deflation_basis = np.empty((0, 0), dtype=np.float64)
        else:
            self.deflation_basis = np.array(snapshot, dtype=np.float64, copy=True)

    def release_iteration_resources(self) -> None:
        return


class PetscKSPFGMRESSolver:
    """PETSc-native FGMRES solve with configurable PETSc preconditioner."""

    def __init__(
        self,
        pc_type: str = "GAMG",
        *,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        tolerance_deflation_basis: float = 1e-3,
        verbose: bool = False,
        q_mask: np.ndarray | None = None,
        coord: np.ndarray | None = None,
        preconditioner_options: dict | None = None,
    ) -> None:
        self.pc_type = str(pc_type).upper()
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.tolerance_deflation_basis = float(tolerance_deflation_basis)
        self.verbose = bool(verbose)
        self.q_mask = np.array([], dtype=bool) if q_mask is None else np.asarray(q_mask, dtype=bool)
        self.coord = None if coord is None else np.asarray(coord, dtype=np.float64)
        self.preconditioner_options = dict(preconditioner_options or {})
        self._full_system_preconditioner = bool(self.preconditioner_options.get("full_system_preconditioner", True))
        self.deflation_basis: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self._A_petsc = None
        self._P_petsc = None
        self._ksp = None
        self._near_nullspace = None
        self._near_nullspace_vecs = []
        self._owns_A_petsc = False
        self._owns_P_petsc = False
        self._using_full_system = False
        self._active_free_indices = np.array([], dtype=np.int64)
        self._ownership_range = None
        self._default_free_indices = (
            np.array([], dtype=np.int64) if self.q_mask.size == 0 else q_to_free_indices(self.q_mask)
        )
        self._options_prefix = f"petsc_linear_{id(self)}_"
        self._diagnostics_enabled = False
        self._last_solve_info: dict[str, object] = {}
        self._last_orthogonalization_info: dict[str, object] = {}
        self._pc_backend = self._normalize_pc_backend()
        self._preconditioner_matrix_source = self._normalize_preconditioner_matrix_source()
        self._preconditioner_matrix_policy = self._normalize_preconditioner_matrix_policy()
        self._preconditioner_rebuild_policy = self._normalize_preconditioner_rebuild_policy()
        self._preconditioner_rebuild_interval = self._normalize_preconditioner_rebuild_interval()
        self._preconditioner_rebuild_requested = True
        self._preconditioner_age = 0
        self._preconditioner_newton_calls_since_rebuild = 0
        self._preconditioner_diagnostics = PreconditionerDiagnostics(
            pc_backend=self._pc_backend,
            preconditioner_matrix_source=self._preconditioner_matrix_source,
            preconditioner_matrix_policy=self._preconditioner_matrix_policy,
            preconditioner_rebuild_policy=self._preconditioner_rebuild_policy,
            preconditioner_rebuild_interval=self._preconditioner_rebuild_interval,
        )

        self.iteration_collector = IterationCollector()
        self.instance_id = self.iteration_collector.register_instance()

    def _normalize_pc_backend(self) -> str:
        raw = self.preconditioner_options.get("pc_backend")
        if raw is not None:
            backend = str(raw).strip().lower()
            if backend in {"hypre", "gamg", "bddc", "jacobi", "none"}:
                return backend
        if self.pc_type == "HYPRE":
            return "hypre"
        if self.pc_type == "GAMG":
            return "gamg"
        if self.pc_type == "JACOBI":
            return "jacobi"
        return "none"

    def _normalize_preconditioner_matrix_policy(self) -> str:
        policy = str(self.preconditioner_options.get("preconditioner_matrix_policy", "current")).strip().lower()
        if policy not in {"current", "lagged"}:
            raise ValueError(f"Unsupported preconditioner_matrix_policy {policy!r}")
        if self._normalize_pc_backend() == "bddc" and policy == "lagged":
            return policy
        return policy

    def _normalize_preconditioner_matrix_source(self) -> str:
        source = str(self.preconditioner_options.get("preconditioner_matrix_source", "tangent")).strip().lower()
        if source not in {"tangent", "regularized", "elastic"}:
            raise ValueError(f"Unsupported preconditioner_matrix_source {source!r}")
        return source

    def _normalize_preconditioner_rebuild_policy(self) -> str:
        policy = str(self.preconditioner_options.get("preconditioner_rebuild_policy", "every_newton")).strip().lower()
        if policy not in {"every_newton", "every_n_newton", "accepted_step", "accepted_or_rejected_step"}:
            raise ValueError(f"Unsupported preconditioner_rebuild_policy {policy!r}")
        return policy

    def _normalize_preconditioner_rebuild_interval(self) -> int:
        try:
            interval = int(self.preconditioner_options.get("preconditioner_rebuild_interval", 1))
        except Exception:
            interval = 1
        return max(1, interval)

    def _record_preconditioner_setup_time(self, elapsed: float) -> None:
        self.iteration_collector.store_preconditioner_time(self.instance_id, elapsed)
        self._preconditioner_diagnostics.preconditioner_setup_time_total += float(elapsed)

    def _record_preconditioner_apply_time(self, elapsed: float) -> None:
        self._preconditioner_diagnostics.preconditioner_apply_time_total += float(elapsed)

    def _destroy_owned_petsc_matrix(self, A, owns: bool) -> None:
        if A is not None and owns:
            release_petsc_aij_matrix(A)
            A.destroy()

    def _matrix_signature(self, A) -> tuple[tuple[int, int], int, int, str] | None:
        if PETSc is None or A is None:
            return None
        if not isinstance(A, PETSc.Mat):
            return None
        return (
            tuple(int(v) for v in A.getSize()),
            int(A.getComm().getSize()),
            int(A.getBlockSize() or 1),
            str(A.getType()),
        )

    def _matrix_compatible(self, A, B) -> bool:
        sig_a = self._matrix_signature(A)
        sig_b = self._matrix_signature(B)
        return sig_a is not None and sig_a == sig_b

    def _mark_preconditioner_rebuilt(self, *, reason: str) -> None:
        self._preconditioner_diagnostics.preconditioner_rebuild_count += 1
        self._preconditioner_diagnostics.preconditioner_last_rebuild_reason = str(reason)
        self._preconditioner_diagnostics.preconditioner_age_max = max(
            int(self._preconditioner_diagnostics.preconditioner_age_max),
            int(self._preconditioner_age),
        )
        self._preconditioner_age = 0
        self._preconditioner_newton_calls_since_rebuild = 0
        self._preconditioner_rebuild_requested = False

    def _mark_preconditioner_reused(self) -> None:
        self._preconditioner_diagnostics.preconditioner_reuse_count += 1
        self._preconditioner_age += 1
        self._preconditioner_newton_calls_since_rebuild += 1
        self._preconditioner_diagnostics.preconditioner_age_max = max(
            int(self._preconditioner_diagnostics.preconditioner_age_max),
            int(self._preconditioner_age),
        )

    def _preconditioner_source_is_static(self) -> bool:
        return bool(self.preconditioner_requires_explicit_matrix() and self._preconditioner_matrix_source == "elastic")

    def _should_rebuild_preconditioner(self) -> tuple[bool, str]:
        if self._preconditioner_source_is_static():
            if self._P_petsc is None:
                return True, "initial"
            policy = self._preconditioner_rebuild_policy
            if policy == "every_n_newton":
                if self._preconditioner_newton_calls_since_rebuild + 1 >= self._preconditioner_rebuild_interval:
                    return True, "every_n_newton"
                return False, "elastic_static"
            if self._preconditioner_rebuild_requested:
                return True, "attempt_trigger"
            return False, "elastic_static"
        if self._preconditioner_matrix_policy == "current":
            return True, "current_policy"
        if self._P_petsc is None:
            return True, "initial"
        policy = self._preconditioner_rebuild_policy
        if policy == "every_newton":
            return True, "every_newton"
        if policy == "every_n_newton":
            if self._preconditioner_newton_calls_since_rebuild + 1 >= self._preconditioner_rebuild_interval:
                return True, "every_n_newton"
            return False, "lagged_reuse"
        if self._preconditioner_rebuild_requested:
            return True, "attempt_trigger"
        return False, "lagged_reuse"

    def preconditioner_requires_explicit_matrix(self) -> bool:
        return self._pc_backend == "bddc"

    def needs_preconditioning_matrix_refresh(self) -> bool:
        return False

    def notify_continuation_attempt(self, *, success: bool) -> None:
        if self._preconditioner_rebuild_policy == "accepted_step":
            if success:
                self._preconditioner_rebuild_requested = True
            return
        if self._preconditioner_rebuild_policy == "accepted_or_rejected_step":
            self._preconditioner_rebuild_requested = True

    def get_preconditioner_diagnostics(self) -> dict[str, object]:
        diagnostics = self._preconditioner_diagnostics.as_dict()
        diagnostics["preconditioner_age_current"] = int(self._preconditioner_age)
        return diagnostics

    def get_preconditioner_matrix_source(self) -> str:
        return str(self._preconditioner_matrix_source)

    def get_deflation_basis_snapshot(self):
        return np.array(self.deflation_basis, dtype=np.float64, copy=True)

    def restore_deflation_basis(self, snapshot) -> None:
        if snapshot is None:
            self.deflation_basis = np.empty((0, 0), dtype=np.float64)
        else:
            self.deflation_basis = np.array(snapshot, dtype=np.float64, copy=True)

    def _reset_petsc_objects(self) -> None:
        if self._ksp is not None:
            self._ksp.destroy()
            self._ksp = None
        self._destroy_owned_petsc_matrix(self._P_petsc, self._owns_P_petsc and self._P_petsc is not self._A_petsc)
        self._destroy_owned_petsc_matrix(self._A_petsc, self._owns_A_petsc)
        self._A_petsc = None
        self._P_petsc = None
        self._owns_A_petsc = False
        self._owns_P_petsc = False
        self._near_nullspace = None
        self._near_nullspace_vecs = []
        self._ownership_range = None
        self._last_solve_info = {}
        self._last_orthogonalization_info = {}

    def _active_free(self, free_indices: np.ndarray | None = None) -> np.ndarray:
        if free_indices is not None:
            return np.asarray(free_indices, dtype=np.int64)
        if self._active_free_indices.size:
            return self._active_free_indices
        return self._default_free_indices

    def _should_use_full_system(self, full_matrix) -> bool:
        return bool(self._full_system_preconditioner and full_matrix is not None and self.q_mask.size)

    def _prepare_operator(self, A, *, full_matrix=None, free_indices: np.ndarray | None = None):
        if self._should_use_full_system(full_matrix):
            self._using_full_system = True
            self._active_free_indices = self._active_free(free_indices)
            if PETSc is not None and isinstance(full_matrix, PETSc.Mat):
                return full_matrix
            zero_rhs = np.zeros(self.q_mask.shape, dtype=np.float64)
            A_operator, _, _ = impose_zero_dirichlet_full_system(full_matrix, zero_rhs, self.q_mask)
            return A_operator

        self._using_full_system = False
        self._active_free_indices = self._active_free(free_indices)
        return A

    def _default_hypre_options(self) -> dict[str, object]:
        if self.q_mask.size == 0:
            return {}
        dim = int(self.q_mask.shape[0])
        return {
            "pc_hypre_boomeramg_max_iter": 1,
            "pc_hypre_boomeramg_tol": 0.0,
            "pc_hypre_boomeramg_numfunctions": dim,
            "pc_hypre_boomeramg_nodal_coarsen": 4,
            "pc_hypre_boomeramg_nodal_coarsen_diag": 1,
            "pc_hypre_boomeramg_vec_interp_variant": 2,
            "pc_hypre_boomeramg_vec_interp_qmax": 4,
            "pc_hypre_boomeramg_vec_interp_smooth": True,
            "pc_hypre_boomeramg_coarsen_type": "PMIS",
            "pc_hypre_boomeramg_interp_type": "ext+i-mm",
            "pc_hypre_boomeramg_P_max": 4,
            "pc_hypre_boomeramg_strong_threshold": 0.5,
            "pc_hypre_boomeramg_grid_sweeps_all": 1,
            "pc_hypre_boomeramg_cycle_type": "V",
            "pc_hypre_boomeramg_agg_nl": 0,
        }

    def _distribution_enabled(self) -> bool:
        return bool(self.preconditioner_options.get("mpi_distribute_by_nodes", False))

    def _matrix_comm(self) -> PETSc.Comm:
        if self._distribution_enabled():
            return PETSc.COMM_WORLD
        return PETSc.COMM_SELF

    @staticmethod
    def _set_petsc_option(options, key: str, value) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            options[key] = "true" if value else "false"
        else:
            options[key] = value

    def _configure_prefixed_options(self, prefix: str) -> None:
        opts = PETSc.Options()
        defaults: dict[str, object] = {}
        if self.pc_type == "HYPRE" and self._using_full_system:
            defaults.update(self._default_hypre_options())
        skip_keys = {
            "threads",
            "print_level",
            "use_as_preconditioner",
            "factor_solver_type",
            "full_system_preconditioner",
            "null_space",
            "use_coordinates",
            "pc_backend",
            "preconditioner_matrix_policy",
            "preconditioner_rebuild_policy",
            "preconditioner_rebuild_interval",
            "recycle_preconditioner",
            "compiled_outer",
            "max_deflation_basis_vectors",
        }

        for key, value in defaults.items():
            if key not in self.preconditioner_options:
                self._set_petsc_option(opts, f"{prefix}{key}", value)

        for key, value in self.preconditioner_options.items():
            if key in skip_keys:
                continue
            if key.startswith(("pc_", "mg_", "ksp_", "mat_")):
                self._set_petsc_option(opts, f"{prefix}{key}", value)

    def _max_deflation_basis_vectors(self) -> int | None:
        raw = self.preconditioner_options.get("max_deflation_basis_vectors")
        if raw is None:
            return None
        try:
            value = int(raw)
        except Exception:
            return None
        return value if value > 0 else None

    def _expand_to_full_space(self, vectors: np.ndarray) -> np.ndarray:
        vec = np.asarray(vectors, dtype=np.float64)
        if vec.ndim == 1:
            vec = vec[:, None]
        if not (self._full_system_preconditioner and self.q_mask.size):
            return vec
        if vec.shape[0] == self.q_mask.size:
            return vec

        free_idx = self._active_free()
        if free_idx.size and vec.shape[0] == free_idx.size:
            full = np.zeros((self.q_mask.size, vec.shape[1]), dtype=np.float64)
            full[free_idx, :] = vec
            return full
        return vec

    def _prepare_rhs(self, b, *, full_rhs=None) -> np.ndarray:
        rhs_free = np.asarray(b, dtype=np.float64).reshape(-1)
        if not self._using_full_system:
            return rhs_free

        if full_rhs is None:
            rhs = np.zeros(self.q_mask.size, dtype=np.float64)
            rhs[self._active_free()] = rhs_free
        else:
            rhs_arr = np.asarray(full_rhs, dtype=np.float64)
            rhs = rhs_arr.reshape(-1, order="F").copy() if rhs_arr.ndim > 1 else rhs_arr.reshape(-1).copy()

        rhs[~self.q_mask.reshape(-1, order="F")] = 0.0
        return rhs

    def _restrict_solution(self, x: np.ndarray) -> np.ndarray:
        if not self._using_full_system:
            return x
        return np.asarray(x, dtype=np.float64)[self._active_free()]

    def enable_diagnostics(self, enabled: bool = True) -> None:
        self._diagnostics_enabled = bool(enabled)

    def get_last_solve_info(self) -> dict[str, object]:
        return dict(self._last_solve_info)

    def get_last_orthogonalization_info(self) -> dict[str, object]:
        return dict(self._last_orthogonalization_info)

    def _install_true_residual_monitor(
        self,
        rhs_total: np.ndarray,
        *,
        solution_offset: np.ndarray | None = None,
    ) -> tuple[list[float], list[float]]:
        reported_history: list[float] = []
        true_history: list[float] = []
        rhs_ref = np.asarray(rhs_total, dtype=np.float64).reshape(-1)
        rhs_norm = float(np.linalg.norm(rhs_ref))
        if rhs_norm == 0.0:
            rhs_norm = 1.0
        offset = None if solution_offset is None else np.asarray(solution_offset, dtype=np.float64).reshape(-1)

        self._ksp.cancelMonitor()

        def _monitor(ksp, it, rnorm):
            x_iter = petsc_vec_to_global_array(ksp.buildSolution())
            if offset is not None:
                x_iter = x_iter + offset
            resid = rhs_ref - _matvec(self._A_petsc, x_iter)
            reported_history.append(float(rnorm))
            true_history.append(float(np.linalg.norm(resid) / rhs_norm))

        self._ksp.setMonitor(_monitor)
        return reported_history, true_history

    def setup_preconditioner(
        self,
        A,
        *,
        full_matrix=None,
        free_indices: np.ndarray | None = None,
        preconditioning_matrix=None,
    ):
        t0 = perf_counter()
        if PETSc is None:
            raise RuntimeError("PETSc is required for KSPFGMRES solver types.")

        self._reset_petsc_objects()

        operator_matrix = self._prepare_operator(A, full_matrix=full_matrix, free_indices=free_indices)
        comm = self._matrix_comm()
        block_size = None
        ownership_range = None
        if self._using_full_system and self.q_mask.size:
            block_size = int(self.q_mask.shape[0])
        elif self.preconditioner_options.get("block_size") is not None:
            block_size = int(self.preconditioner_options["block_size"])
        if int(comm.getSize()) > 1 and block_size is not None and not (PETSc is not None and isinstance(operator_matrix, PETSc.Mat)):
            ownership_range = owned_block_range(operator_matrix.shape[0] // block_size, block_size, comm)
        self._A_petsc = to_petsc_aij_matrix(
            operator_matrix,
            comm=operator_matrix.getComm() if hasattr(operator_matrix, "getComm") else comm,
            block_size=block_size,
            ownership_range=ownership_range,
        )
        self._ownership_range = self._A_petsc.getOwnershipRange()

        if self._using_full_system and self.q_mask.size:
            self._A_petsc.setBlockSize(int(self.q_mask.shape[0]))

        null_space = self.preconditioner_options.get("null_space")
        if null_space is None and self.pc_type in {"GAMG", "HYPRE"} and self.coord is not None and self.q_mask.size:
            null_space = make_near_nullspace_elasticity(
                self.coord,
                q_mask=self.q_mask,
                center_coordinates=True,
                return_full=self._using_full_system,
            )

        if self.pc_type in {"GAMG", "HYPRE"}:
            self._A_petsc, self._near_nullspace, self._near_nullspace_vecs = attach_near_nullspace(
                self._A_petsc,
                null_space,
            )
        else:
            self._near_nullspace = None
            self._near_nullspace_vecs = []

        self._ksp = PETSc.KSP().create(comm=self._A_petsc.getComm())
        self._ksp.setOptionsPrefix(self._options_prefix)
        self._ksp.setOperators(self._A_petsc)
        self._ksp.setType(PETSc.KSP.Type.FGMRES)
        self._ksp.setInitialGuessNonzero(False)
        self._ksp.setTolerances(rtol=self.tolerance, atol=1e-30, max_it=self.max_iterations)
        self._configure_prefixed_options(self._options_prefix)
        pc = self._ksp.getPC()
        if self.pc_type == "GAMG":
            pc.setType(PETSc.PC.Type.GAMG)
            if self._using_full_system and self.coord is not None and self.preconditioner_options.get("use_coordinates", True):
                if int(self._A_petsc.getComm().getSize()) > 1:
                    dim = int(self.q_mask.shape[0])
                    r0, r1 = self._ownership_range
                    node0, node1 = r0 // dim, r1 // dim
                    pc.setCoordinates(self.coord[:, node0:node1].T.copy())
                else:
                    pc.setCoordinates(self.coord.T.copy())
        elif self.pc_type == "HYPRE":
            pc.setType(PETSc.PC.Type.HYPRE)
            pc.setHYPREType(str(self.preconditioner_options.get("pc_hypre_type", "boomeramg")))
        elif self.pc_type == "JACOBI":
            pc.setType(PETSc.PC.Type.JACOBI)
        else:
            pc.setType(PETSc.PC.Type.NONE)
        self._ksp.setFromOptions()
        self._ksp.setUp()
        self.iteration_collector.store_preconditioner_time(self.instance_id, perf_counter() - t0)

    def A_orthogonalize(self, A):
        t0 = perf_counter()
        A_ref = self._A_petsc if self._A_petsc is not None else A
        before_cols = int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else int(bool(self.deflation_basis.size))
        self.deflation_basis, basis_norms = a_orthogonalize_with_info(
            self.deflation_basis,
            A_ref,
            self.tolerance_deflation_basis,
        )
        elapsed = perf_counter() - t0
        self.iteration_collector.store_orthogonalization_time(self.instance_id, elapsed)
        if self._diagnostics_enabled:
            self._last_orthogonalization_info = {
                "time_s": float(elapsed),
                "basis_cols_before": int(before_cols),
                "basis_cols_after": int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else 0,
                "basis_norm_signs": np.sign(basis_norms).astype(np.float64).tolist(),
                **_basis_diagnostics(self.deflation_basis, A_ref),
            }

    def solve(self, A, b, *, full_rhs=None, free_indices: np.ndarray | None = None):
        if PETSc is None:
            raise RuntimeError("PETSc is required for KSPFGMRES solver types.")
        if self._ksp is None or self._A_petsc is None:
            self.setup_preconditioner(A, free_indices=free_indices)

        rhs_arr = self._prepare_rhs(b, full_rhs=full_rhs)
        basis = self.deflation_basis if self.deflation_basis.size else np.empty((rhs_arr.size, 0), dtype=np.float64)

        t0 = perf_counter()
        x0, rhs_eff = _coarse_deflation_correction(self._A_petsc, rhs_arr, basis)
        reported_history: list[float] = []
        true_history: list[float] = []
        if self._diagnostics_enabled:
            reported_history, true_history = self._install_true_residual_monitor(rhs_arr, solution_offset=x0)
        rhs = global_array_to_petsc_vec(
            rhs_eff,
            comm=self._A_petsc.getComm(),
            ownership_range=self._ownership_range if int(self._A_petsc.getComm().getSize()) > 1 else None,
            bsize=self._A_petsc.getBlockSize() or None,
        )
        delta = self._A_petsc.createVecRight()
        delta.set(0.0)
        self._ksp.solve(rhs, delta)

        x = x0 + petsc_vec_to_global_array(delta)
        nit = int(self._ksp.getIterationNumber())
        elapsed = perf_counter() - t0
        self.iteration_collector.store_iteration(self.instance_id, nit, elapsed)
        if self._diagnostics_enabled:
            resid = rhs_arr - _matvec(self._A_petsc, x)
            rhs_norm = float(np.linalg.norm(rhs_arr))
            if rhs_norm == 0.0:
                rhs_norm = 1.0
            if not true_history:
                true_history = [float(np.linalg.norm(resid) / rhs_norm)]
            self._last_solve_info = {
                "iterations": int(nit),
                "time_s": float(elapsed),
                "basis_cols": int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else 0,
                "rhs_norm": rhs_norm,
                "coarse_initial_guess_norm": float(np.linalg.norm(x0)),
                "reported_residual_history": list(reported_history),
                "true_residual_history": list(true_history),
                "true_residual_final": float(np.linalg.norm(resid) / rhs_norm),
            }
            self._ksp.cancelMonitor()
        if self.verbose:
            print(f"{nit}|", end="")
        return self._restrict_solution(x)

    def expand_deflation_basis(self, additional_vectors):
        if additional_vectors is None:
            return
        v = np.asarray(additional_vectors, dtype=np.float64)
        if v.size == 0:
            return
        v = self._expand_to_full_space(v)
        if self.deflation_basis.size == 0:
            self.deflation_basis = np.asarray(v, dtype=np.float64)
        else:
            self.deflation_basis = np.hstack((self.deflation_basis, v))
        max_cols = self._max_deflation_basis_vectors()
        if max_cols is not None and self.deflation_basis.ndim == 2 and self.deflation_basis.shape[1] > max_cols:
            self.deflation_basis = np.asarray(self.deflation_basis[:, -max_cols:], dtype=np.float64)

    def copy(self):
        clone = self.__class__(
            self.pc_type,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            tolerance_deflation_basis=self.tolerance_deflation_basis,
            verbose=self.verbose,
            q_mask=self.q_mask,
            coord=self.coord,
            preconditioner_options=self.preconditioner_options,
        )
        # Share the current basis until the clone appends/reorthogonalizes and rebinds it.
        clone.deflation_basis = self.deflation_basis
        clone.iteration_collector = self.iteration_collector
        clone.instance_id = self.iteration_collector.register_instance()
        clone._diagnostics_enabled = self._diagnostics_enabled
        clone._preconditioner_diagnostics = self._preconditioner_diagnostics
        return clone

    def prefers_full_system_operator(self) -> bool:
        return bool(self._full_system_preconditioner and self.q_mask.size)

    def release_iteration_resources(self) -> None:
        self._reset_petsc_objects()


class PetscKSPGMRESDeflationSolver(PetscKSPFGMRESSolver):
    """PETSc-native GMRES with PCDEFLATION and a user-supplied recycle space."""

    def __init__(
        self,
        pc_type: str = "GAMG",
        *,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        tolerance_deflation_basis: float = 1e-3,
        verbose: bool = False,
        q_mask: np.ndarray | None = None,
        coord: np.ndarray | None = None,
        preconditioner_options: dict | None = None,
    ) -> None:
        super().__init__(
            pc_type,
            tolerance=tolerance,
            max_iterations=max_iterations,
            tolerance_deflation_basis=tolerance_deflation_basis,
            verbose=verbose,
            q_mask=q_mask,
            coord=coord,
            preconditioner_options=preconditioner_options,
        )
        self._using_native_deflation = False
        self._deflation_space_mat = None
        self._deflation_space_buffers = None

    def _reset_petsc_objects(self) -> None:
        if self._deflation_space_mat is not None:
            release_petsc_aij_matrix(self._deflation_space_mat)
            self._deflation_space_mat.destroy()
            self._deflation_space_mat = None
        self._deflation_space_buffers = None
        self._using_native_deflation = False
        super()._reset_petsc_objects()

    def _basis_qr_tolerance(self) -> float:
        return max(1.0e-12, self.tolerance_deflation_basis * 1.0e-3)

    def _compress_basis(self, basis: np.ndarray) -> np.ndarray:
        raw = np.asarray(basis, dtype=np.float64)
        if raw.size == 0:
            rows = int(raw.shape[0]) if raw.ndim >= 1 else 0
            return np.empty((rows, 0), dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[:, None]

        col_norm = np.linalg.norm(raw, axis=0)
        keep = col_norm > self._basis_qr_tolerance()
        if not np.any(keep):
            return np.empty((raw.shape[0], 0), dtype=np.float64)

        q, r = np.linalg.qr(raw[:, keep], mode="reduced")
        diag = np.abs(np.diag(r))
        if diag.size == 0:
            return np.empty((raw.shape[0], 0), dtype=np.float64)
        keep_q = diag > (self._basis_qr_tolerance() * float(diag.max()))
        if not np.any(keep_q):
            return np.empty((raw.shape[0], 0), dtype=np.float64)
        return np.asfortranarray(q[:, keep_q], dtype=np.float64)

    def _build_deflation_space(self):
        basis = self._compress_basis(self.deflation_basis)
        self.deflation_basis = basis
        if basis.size == 0:
            return None, False

        comm = self._A_petsc.getComm()
        n_rows, n_basis = basis.shape
        import scipy.sparse as sp

        if int(comm.getSize()) == 1:
            W = to_petsc_aij_matrix(sp.csr_matrix(basis), comm=comm)
            self._deflation_space_buffers = None
        else:
            r0, r1 = self._ownership_range
            csr = sp.csr_matrix(basis[r0:r1, :])
            indptr = np.array(csr.indptr, dtype=PETSc.IntType, copy=True)
            indices = np.array(csr.indices, dtype=PETSc.IntType, copy=True)
            data = np.array(csr.data, dtype=np.float64, copy=True)
            W = PETSc.Mat().createAIJ(
                size=((r1 - r0, n_rows), (PETSc.DECIDE, n_basis)),
                csr=(indptr, indices, data),
                comm=comm,
            )
            W.assemble()
            self._deflation_space_buffers = (indptr, indices, data)
        self._deflation_space_mat = W
        return W, False

    def _configure_native_deflation_options(self, prefix: str) -> None:
        opts = PETSc.Options()
        defaults: dict[str, object] = {
            "pc_deflation_correction_factor": self.preconditioner_options.get("pc_deflation_correction_factor", 1.0),
            "pc_deflation_init_only": False,
        }

        if self.pc_type == "HYPRE" and self._using_full_system:
            for key, value in self._default_hypre_options().items():
                if key not in self.preconditioner_options:
                    self._set_petsc_option(opts, f"{prefix}deflation_{key}", value)

        for key, value in defaults.items():
            if key not in self.preconditioner_options:
                self._set_petsc_option(opts, f"{prefix}{key}", value)

        for key, value in self.preconditioner_options.items():
            if key in {
                "threads",
                "print_level",
                "use_as_preconditioner",
                "factor_solver_type",
                "full_system_preconditioner",
                "null_space",
                "use_coordinates",
                "pc_backend",
                "preconditioner_matrix_policy",
                "preconditioner_rebuild_policy",
                "preconditioner_rebuild_interval",
                "recycle_preconditioner",
                "compiled_outer",
                "max_deflation_basis_vectors",
            }:
                continue
            if key.startswith("pc_deflation_"):
                self._set_petsc_option(opts, f"{prefix}{key}", value)
            elif key.startswith("ksp_"):
                self._set_petsc_option(opts, f"{prefix}{key}", value)
            elif key.startswith(("pc_", "mg_", "mat_")):
                self._set_petsc_option(opts, f"{prefix}deflation_{key}", value)

    def _configure_inner_deflation_pc(self) -> None:
        pc = self._ksp.getPC()
        inner_pc = pc.getDeflationPC()
        if self.pc_type == "GAMG":
            inner_pc.setType(PETSc.PC.Type.GAMG)
            if self._using_full_system and self.coord is not None and self.preconditioner_options.get("use_coordinates", True):
                if int(self._A_petsc.getComm().getSize()) > 1:
                    dim = int(self.q_mask.shape[0])
                    r0, r1 = self._ownership_range
                    node0, node1 = r0 // dim, r1 // dim
                    inner_pc.setCoordinates(self.coord[:, node0:node1].T.copy())
                else:
                    inner_pc.setCoordinates(self.coord.T.copy())
        elif self.pc_type == "HYPRE":
            inner_pc.setType(PETSc.PC.Type.HYPRE)
            inner_pc.setHYPREType(str(self.preconditioner_options.get("pc_hypre_type", "boomeramg")))
        elif self.pc_type == "JACOBI":
            inner_pc.setType(PETSc.PC.Type.JACOBI)
        else:
            inner_pc.setType(PETSc.PC.Type.NONE)
        inner_pc.setFromOptions()
        inner_pc.setUp()

    def setup_preconditioner(
        self,
        A,
        *,
        full_matrix=None,
        free_indices: np.ndarray | None = None,
        preconditioning_matrix=None,
    ):
        t0 = perf_counter()
        if PETSc is None:
            raise RuntimeError("PETSc is required for native deflation KSP solver types.")

        self._reset_petsc_objects()

        operator_matrix = self._prepare_operator(A, full_matrix=full_matrix, free_indices=free_indices)
        comm = self._matrix_comm()
        block_size = None
        ownership_range = None
        if self._using_full_system and self.q_mask.size:
            block_size = int(self.q_mask.shape[0])
        elif self.preconditioner_options.get("block_size") is not None:
            block_size = int(self.preconditioner_options["block_size"])
        if int(comm.getSize()) > 1 and block_size is not None and not (PETSc is not None and isinstance(operator_matrix, PETSc.Mat)):
            ownership_range = owned_block_range(operator_matrix.shape[0] // block_size, block_size, comm)
        self._A_petsc = to_petsc_aij_matrix(
            operator_matrix,
            comm=operator_matrix.getComm() if hasattr(operator_matrix, "getComm") else comm,
            block_size=block_size,
            ownership_range=ownership_range,
        )
        self._ownership_range = self._A_petsc.getOwnershipRange()

        if self._using_full_system and self.q_mask.size:
            self._A_petsc.setBlockSize(int(self.q_mask.shape[0]))

        null_space = self.preconditioner_options.get("null_space")
        if null_space is None and self.pc_type in {"GAMG", "HYPRE"} and self.coord is not None and self.q_mask.size:
            null_space = make_near_nullspace_elasticity(
                self.coord,
                q_mask=self.q_mask,
                center_coordinates=True,
                return_full=self._using_full_system,
            )

        if self.pc_type in {"GAMG", "HYPRE"}:
            self._A_petsc, self._near_nullspace, self._near_nullspace_vecs = attach_near_nullspace(
                self._A_petsc,
                null_space,
            )
        else:
            self._near_nullspace = None
            self._near_nullspace_vecs = []

        self._ksp = PETSc.KSP().create(comm=self._A_petsc.getComm())
        self._ksp.setOptionsPrefix(self._options_prefix)
        self._ksp.setOperators(self._A_petsc)
        self._ksp.setType(PETSc.KSP.Type.GMRES)
        self._ksp.setInitialGuessNonzero(False)
        self._ksp.setTolerances(rtol=self.tolerance, atol=1e-30, max_it=self.max_iterations)

        basis_mat, basis_is_transpose = self._build_deflation_space()
        self._using_native_deflation = basis_mat is not None
        pc = self._ksp.getPC()
        if self._using_native_deflation:
            pc.setType(PETSc.PC.Type.DEFLATION)
            pc.setDeflationSpace(basis_mat, basis_is_transpose)
            pc.setDeflationInitOnly(False)
            if "pc_deflation_correction_factor" in self.preconditioner_options:
                pc.setDeflationCorrectionFactor(float(self.preconditioner_options["pc_deflation_correction_factor"]))
            else:
                pc.setDeflationCorrectionFactor(1.0)
            self._configure_native_deflation_options(self._options_prefix)
            self._ksp.setFromOptions()
            self._ksp.setUp()
            self._configure_inner_deflation_pc()
        else:
            self._configure_prefixed_options(self._options_prefix)
            if self.pc_type == "GAMG":
                pc.setType(PETSc.PC.Type.GAMG)
                if self._using_full_system and self.coord is not None and self.preconditioner_options.get("use_coordinates", True):
                    if int(self._A_petsc.getComm().getSize()) > 1:
                        dim = int(self.q_mask.shape[0])
                        r0, r1 = self._ownership_range
                        node0, node1 = r0 // dim, r1 // dim
                        pc.setCoordinates(self.coord[:, node0:node1].T.copy())
                    else:
                        pc.setCoordinates(self.coord.T.copy())
            elif self.pc_type == "HYPRE":
                pc.setType(PETSc.PC.Type.HYPRE)
                pc.setHYPREType(str(self.preconditioner_options.get("pc_hypre_type", "boomeramg")))
            elif self.pc_type == "JACOBI":
                pc.setType(PETSc.PC.Type.JACOBI)
            else:
                pc.setType(PETSc.PC.Type.NONE)
            self._ksp.setFromOptions()
            self._ksp.setUp()

        self.iteration_collector.store_preconditioner_time(self.instance_id, perf_counter() - t0)

    def A_orthogonalize(self, A):
        t0 = perf_counter()
        if self.deflation_basis.size:
            self.deflation_basis = self._compress_basis(self.deflation_basis)
        self.iteration_collector.store_orthogonalization_time(self.instance_id, perf_counter() - t0)

    def solve(self, A, b, *, full_rhs=None, free_indices: np.ndarray | None = None):
        if PETSc is None:
            raise RuntimeError("PETSc is required for native deflation KSP solver types.")
        if self._ksp is None or self._A_petsc is None:
            self.setup_preconditioner(A, free_indices=free_indices)

        rhs_arr = self._prepare_rhs(b, full_rhs=full_rhs)
        t0 = perf_counter()
        reported_history: list[float] = []
        true_history: list[float] = []
        if self._diagnostics_enabled:
            reported_history, true_history = self._install_true_residual_monitor(rhs_arr)
        rhs = global_array_to_petsc_vec(
            rhs_arr,
            comm=self._A_petsc.getComm(),
            ownership_range=self._ownership_range if int(self._A_petsc.getComm().getSize()) > 1 else None,
            bsize=self._A_petsc.getBlockSize() or None,
        )
        x = self._A_petsc.createVecRight()
        x.set(0.0)
        self._ksp.solve(rhs, x)

        nit = int(self._ksp.getIterationNumber())
        elapsed = perf_counter() - t0
        self.iteration_collector.store_iteration(self.instance_id, nit, elapsed)
        if self._diagnostics_enabled:
            x_total = petsc_vec_to_global_array(x)
            resid = rhs_arr - _matvec(self._A_petsc, x_total)
            rhs_norm = float(np.linalg.norm(rhs_arr))
            if rhs_norm == 0.0:
                rhs_norm = 1.0
            if not true_history:
                true_history = [float(np.linalg.norm(resid) / rhs_norm)]
            self._last_solve_info = {
                "iterations": int(nit),
                "time_s": float(elapsed),
                "basis_cols": int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else 0,
                "rhs_norm": rhs_norm,
                "coarse_initial_guess_norm": 0.0,
                "reported_residual_history": list(reported_history),
                "true_residual_history": list(true_history),
                "true_residual_final": float(np.linalg.norm(resid) / rhs_norm),
            }
            self._ksp.cancelMonitor()
        if self.verbose:
            print(f"{nit}|", end="")
        return self._restrict_solution(petsc_vec_to_global_array(x))

    def copy(self):
        clone = self.__class__(
            self.pc_type,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            tolerance_deflation_basis=self.tolerance_deflation_basis,
            verbose=self.verbose,
            q_mask=self.q_mask,
            coord=self.coord,
            preconditioner_options=self.preconditioner_options,
        )
        # Share the current basis until the clone appends/reorthogonalizes and rebinds it.
        clone.deflation_basis = self.deflation_basis
        clone.iteration_collector = self.iteration_collector
        clone.instance_id = self.iteration_collector.register_instance()
        return clone


class _ProjectedMatlabDeflationPC:
    """Python PC context applying MATLAB-style projected preconditioning."""

    def __init__(self, solver: "PetscKSPMatlabDeflatedFGMRESSolver") -> None:
        self.solver = solver
        self._tmp = None

    def apply(self, pc, x, y) -> None:
        if self._tmp is None:
            self._tmp = self.solver._A_petsc.createVecRight()
        self._tmp.set(0.0)
        self.solver._inner_ksp.solve(x, self._tmp)
        z_local = np.asarray(self._tmp.getArray(readonly=True), dtype=np.float64)
        y_local = self.solver._project_local_vector(z_local)
        y_arr = y.getArray(readonly=False)
        y_arr[...] = y_local


class PetscKSPMatlabDeflatedFGMRESSolver(PetscKSPFGMRESSolver):
    """PETSc FGMRES using a MATLAB-style projected preconditioner."""

    def __init__(
        self,
        pc_type: str = "GAMG",
        *,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        tolerance_deflation_basis: float = 1e-3,
        verbose: bool = False,
        q_mask: np.ndarray | None = None,
        coord: np.ndarray | None = None,
        preconditioner_options: dict | None = None,
    ) -> None:
        super().__init__(
            pc_type,
            tolerance=tolerance,
            max_iterations=max_iterations,
            tolerance_deflation_basis=tolerance_deflation_basis,
            verbose=verbose,
            q_mask=q_mask,
            coord=coord,
            preconditioner_options=preconditioner_options,
        )
        self._inner_ksp = None
        self._projected_pc_context = None
        self._basis_local = np.empty((0, 0), dtype=np.float64)
        self._basis_A_local = np.empty((0, 0), dtype=np.float64)
        self._basis_global = np.empty((0, 0), dtype=np.float64)
        self._basis_A_global = np.empty((0, 0), dtype=np.float64)
        self._mpi_comm = None
        self._last_basis_reorth_passes = 0
        self._projector_dirty = True

    def _reset_petsc_objects(self) -> None:
        if self._inner_ksp is not None:
            self._inner_ksp.destroy()
            self._inner_ksp = None
        self._projected_pc_context = None
        self._basis_local = np.empty((0, 0), dtype=np.float64)
        self._basis_A_local = np.empty((0, 0), dtype=np.float64)
        self._basis_global = np.empty((0, 0), dtype=np.float64)
        self._basis_A_global = np.empty((0, 0), dtype=np.float64)
        self._mpi_comm = None
        self._last_basis_reorth_passes = 0
        self._projector_dirty = True
        super()._reset_petsc_objects()

    def _deflation_reorth_passes(self) -> int:
        try:
            passes = int(self.preconditioner_options.get("deflation_reorth_passes", 1))
        except Exception:
            passes = 1
        return max(1, passes)

    def _refresh_projector_data(self, A_ref) -> None:
        basis = self.deflation_basis
        if basis.size == 0:
            self._basis_global = np.empty((0, 0), dtype=np.float64)
            self._basis_A_global = np.empty((0, 0), dtype=np.float64)
            self._basis_local = np.empty((0, 0), dtype=np.float64)
            self._basis_A_local = np.empty((0, 0), dtype=np.float64)
            self._last_basis_reorth_passes = 0
            self._projector_dirty = False
            return

        used_passes = 0
        distributed_local = PETSc is not None and isinstance(A_ref, PETSc.Mat) and int(A_ref.getComm().getSize()) > 1

        if distributed_local:
            basis_local = None
            kept_source_idx = None
            for _ in range(self._deflation_reorth_passes()):
                basis_local, _basis_norms, kept_source_idx = a_orthogonalize_with_local_metadata(
                    basis,
                    A_ref,
                    self.tolerance_deflation_basis,
                )
                used_passes += 1
                if basis_local.size == 0 or kept_source_idx.size == 0:
                    basis = np.empty((basis.shape[0], 0), dtype=np.float64)
                    break
                basis = np.asarray(basis[:, kept_source_idx], dtype=np.float64)
            self.deflation_basis = basis
        else:
            for _ in range(self._deflation_reorth_passes()):
                basis, _basis_norms = a_orthogonalize_with_info(basis, A_ref, self.tolerance_deflation_basis)
                used_passes += 1
                if basis.size == 0:
                    break
            self.deflation_basis = basis
            basis_local = None
            kept_source_idx = None

        if basis.size == 0:
            self._basis_global = np.empty((0, 0), dtype=np.float64)
            self._basis_A_global = np.empty((0, 0), dtype=np.float64)
            self._basis_local = np.empty((0, 0), dtype=np.float64)
            self._basis_A_local = np.empty((0, 0), dtype=np.float64)
            self._last_basis_reorth_passes = used_passes
            self._projector_dirty = False
            return

        r0, r1 = self._ownership_range if self._ownership_range is not None else (0, basis.shape[0])
        if distributed_local:
            x_vec = A_ref.createVecRight()
            y_vec = A_ref.createVecRight()
            x_arr = x_vec.getArray(readonly=False)
            self._basis_local = np.asarray(basis_local, dtype=np.float64)
            basis_A_local = np.empty((r1 - r0, self._basis_local.shape[1]), dtype=np.float64)
            for j in range(self._basis_local.shape[1]):
                x_arr[...] = self._basis_local[:, j]
                A_ref.mult(x_vec, y_vec)
                col_local = np.asarray(y_vec.getArray(readonly=True), dtype=np.float64).copy()
                basis_A_local[:, j] = col_local
            self._basis_A_local = basis_A_local
            if self._diagnostics_enabled:
                gathered_basis = self._mpi_comm.allgather(self._basis_local)
                gathered_basis_A = self._mpi_comm.allgather(self._basis_A_local)
                self._basis_global = np.vstack(gathered_basis) if gathered_basis else np.empty((0, 0), dtype=np.float64)
                self._basis_A_global = np.vstack(gathered_basis_A) if gathered_basis_A else np.empty((0, 0), dtype=np.float64)
            else:
                self._basis_global = np.empty((0, 0), dtype=np.float64)
                self._basis_A_global = np.empty((0, 0), dtype=np.float64)
        else:
            self._basis_local = basis[r0:r1, :]
            basis_A = np.column_stack([_matvec(A_ref, basis[:, j]) for j in range(basis.shape[1])])
            self._basis_A_local = basis_A[r0:r1, :]
            if self._diagnostics_enabled:
                self._basis_global = basis
                self._basis_A_global = basis_A
            else:
                self._basis_global = np.empty((0, 0), dtype=np.float64)
                self._basis_A_global = np.empty((0, 0), dtype=np.float64)
        self._last_basis_reorth_passes = used_passes
        self._projector_dirty = False

    def _ensure_projector_ready(self) -> None:
        if self._projector_dirty:
            A_ref = self._A_petsc if self._A_petsc is not None else None
            if A_ref is None:
                return
            self._refresh_projector_data(A_ref)

    def _project_local_vector(self, z_local: np.ndarray) -> np.ndarray:
        self._ensure_projector_ready()
        if self._basis_local.size == 0:
            return np.asarray(z_local, dtype=np.float64)
        coeff_local = self._basis_A_local.T @ np.asarray(z_local, dtype=np.float64)
        coeff = self._mpi_comm.allreduce(coeff_local, op=MPI.SUM)
        return np.asarray(z_local, dtype=np.float64) - self._basis_local @ coeff

    def _coarse_initial_guess_local(self, b_local: np.ndarray) -> np.ndarray:
        self._ensure_projector_ready()
        if self._basis_local.size == 0:
            return np.zeros_like(np.asarray(b_local, dtype=np.float64))
        coeff_local = self._basis_local.T @ np.asarray(b_local, dtype=np.float64)
        coeff = self._mpi_comm.allreduce(coeff_local, op=MPI.SUM)
        return self._basis_local @ coeff

    def expand_deflation_basis(self, additional_vectors):
        super().expand_deflation_basis(additional_vectors)
        self._projector_dirty = True

    def _configure_inner_pc(self) -> None:
        inner_pc = self._inner_ksp.getPC()
        if hasattr(inner_pc, "setOptionsPrefix"):
            inner_pc.setOptionsPrefix(f"{self._options_prefix}inner_")
        matrix_ref = self._P_petsc if self._P_petsc is not None else self._A_petsc
        if self._pc_backend == "gamg":
            inner_pc.setType(PETSc.PC.Type.GAMG)
            if (
                self._using_full_system
                and self.coord is not None
                and self.preconditioner_options.get("use_coordinates", True)
                and matrix_ref is not None
            ):
                if int(matrix_ref.getComm().getSize()) > 1:
                    dim = int(self.q_mask.shape[0])
                    r0, r1 = self._ownership_range
                    node0, node1 = r0 // dim, r1 // dim
                    inner_pc.setCoordinates(self.coord[:, node0:node1].T.copy())
                else:
                    inner_pc.setCoordinates(self.coord.T.copy())
        elif self._pc_backend == "hypre":
            inner_pc.setType(PETSc.PC.Type.HYPRE)
            inner_pc.setHYPREType(str(self.preconditioner_options.get("pc_hypre_type", "boomeramg")))
        elif self._pc_backend == "bddc":
            inner_pc.setType(PETSc.PC.Type.BDDC)
            metadata = get_petsc_matrix_metadata(matrix_ref)
            coordinates = bddc_pc_coordinates_from_metadata(matrix_ref)
            if coordinates is not None and self.preconditioner_options.get("use_coordinates", True):
                inner_pc.setCoordinates(np.asarray(coordinates, dtype=np.float64))
            field_is = metadata.get("bddc_field_is_local")
            if field_is:
                if not all(isinstance(v, PETSc.IS) for v in field_is):
                    field_is = tuple(
                        PETSc.IS().createGeneral(np.asarray(v, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
                        for v in field_is
                    )
                inner_pc.setBDDCDofsSplittingLocal(field_is)
            dirichlet = metadata.get("bddc_dirichlet_local")
            if dirichlet is not None:
                if not isinstance(dirichlet, PETSc.IS):
                    dirichlet = PETSc.IS().createGeneral(np.asarray(dirichlet, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
                inner_pc.setBDDCDirichletBoundariesLocal(dirichlet)
            adjacency = metadata.get("bddc_local_adjacency")
            if adjacency is not None:
                inner_pc.setBDDCLocalAdjacency(adjacency)
            primal_vertices = metadata.get("bddc_primal_vertices_local")
            if primal_vertices is not None:
                if not isinstance(primal_vertices, PETSc.IS):
                    primal_vertices = PETSc.IS().createGeneral(
                        np.asarray(primal_vertices, dtype=PETSc.IntType),
                        comm=PETSc.COMM_SELF,
                    )
                inner_pc.setBDDCPrimalVerticesLocalIS(primal_vertices)
        elif self._pc_backend == "jacobi":
            inner_pc.setType(PETSc.PC.Type.JACOBI)
        else:
            inner_pc.setType(PETSc.PC.Type.NONE)
        inner_pc.setFromOptions()
        inner_pc.setUp()

    def _configure_outer_ksp_only_options(self) -> None:
        opts = PETSc.Options()
        for key, value in self.preconditioner_options.items():
            if key.startswith("ksp_"):
                self._set_petsc_option(opts, f"{self._options_prefix}{key}", value)

    def _configure_inner_pc_only_options(self) -> None:
        opts = PETSc.Options()
        prefix = f"{self._options_prefix}inner_"
        for key, value in self.preconditioner_options.items():
            if key in {
                "threads",
                "print_level",
                "use_as_preconditioner",
                "factor_solver_type",
                "full_system_preconditioner",
                "null_space",
                "use_coordinates",
                "deflation_reorth_passes",
                "pc_backend",
                "preconditioner_matrix_policy",
                "preconditioner_rebuild_policy",
                "preconditioner_rebuild_interval",
                "recycle_preconditioner",
                "compiled_outer",
                "max_deflation_basis_vectors",
            }:
                continue
            if key.startswith(("pc_", "mg_", "mat_")):
                self._set_petsc_option(opts, f"{prefix}{key}", value)

    def _outer_presolve(self, ksp, b, x) -> None:
        x_arr = x.getArray(readonly=False)
        if self._basis_local.size == 0:
            x_arr[...] = 0.0
            return
        b_local = np.asarray(b.getArray(readonly=True), dtype=np.float64)
        x_arr[...] = self._coarse_initial_guess_local(b_local)

    def setup_preconditioner(
        self,
        A,
        *,
        full_matrix=None,
        free_indices: np.ndarray | None = None,
        preconditioning_matrix=None,
    ):
        t0 = perf_counter()
        if PETSc is None:
            raise RuntimeError("PETSc is required for MATLAB-like KSPFGMRES solver types.")

        self._reset_petsc_objects()

        operator_matrix = self._prepare_operator(A, full_matrix=full_matrix, free_indices=free_indices)
        comm = self._matrix_comm()
        block_size = None
        ownership_range = None
        if self._using_full_system and self.q_mask.size:
            block_size = int(self.q_mask.shape[0])
        elif self.preconditioner_options.get("block_size") is not None:
            block_size = int(self.preconditioner_options["block_size"])
        if int(comm.getSize()) > 1 and block_size is not None and not (PETSc is not None and isinstance(operator_matrix, PETSc.Mat)):
            ownership_range = owned_block_range(operator_matrix.shape[0] // block_size, block_size, comm)
        self._A_petsc = to_petsc_aij_matrix(
            operator_matrix,
            comm=operator_matrix.getComm() if hasattr(operator_matrix, "getComm") else comm,
            block_size=block_size,
            ownership_range=ownership_range,
        )
        self._ownership_range = self._A_petsc.getOwnershipRange()
        self._mpi_comm = self._A_petsc.getComm().tompi4py()

        if self._using_full_system and self.q_mask.size:
            self._A_petsc.setBlockSize(int(self.q_mask.shape[0]))

        null_space = self.preconditioner_options.get("null_space")
        if null_space is None and self.pc_type in {"GAMG", "HYPRE"} and self.coord is not None and self.q_mask.size:
            null_space = make_near_nullspace_elasticity(
                self.coord,
                q_mask=self.q_mask,
                center_coordinates=True,
                return_full=self._using_full_system,
            )

        if self.pc_type in {"GAMG", "HYPRE"}:
            self._A_petsc, self._near_nullspace, self._near_nullspace_vecs = attach_near_nullspace(
                self._A_petsc,
                null_space,
            )
        else:
            self._near_nullspace = None
            self._near_nullspace_vecs = []

        self._inner_ksp = PETSc.KSP().create(comm=self._A_petsc.getComm())
        self._inner_ksp.setOptionsPrefix(f"{self._options_prefix}inner_")
        self._inner_ksp.setOperators(self._A_petsc)
        self._inner_ksp.setType(PETSc.KSP.Type.PREONLY)
        self._configure_inner_pc_only_options()
        self._configure_inner_pc()
        self._inner_ksp.setFromOptions()
        self._inner_ksp.setUp()

        self._ksp = PETSc.KSP().create(comm=self._A_petsc.getComm())
        self._ksp.setOptionsPrefix(self._options_prefix)
        self._ksp.setOperators(self._A_petsc)
        self._ksp.setType(PETSc.KSP.Type.FGMRES)
        self._ksp.setInitialGuessNonzero(True)
        self._ksp.setTolerances(rtol=self.tolerance, atol=1e-30, max_it=self.max_iterations)
        self._configure_outer_ksp_only_options()
        self._projected_pc_context = _ProjectedMatlabDeflationPC(self)
        pc = self._ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(self._projected_pc_context)
        self._ksp.setPreSolve(self._outer_presolve)
        self._ksp.setFromOptions()
        self._ksp.setUp()
        self._refresh_projector_data(self._A_petsc)
        self.iteration_collector.store_preconditioner_time(self.instance_id, perf_counter() - t0)

    def A_orthogonalize(self, A):
        t0 = perf_counter()
        A_ref = self._A_petsc if self._A_petsc is not None else A
        before_cols = int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else int(bool(self.deflation_basis.size))
        self._refresh_projector_data(A_ref)
        elapsed = perf_counter() - t0
        self.iteration_collector.store_orthogonalization_time(self.instance_id, elapsed)
        if self._diagnostics_enabled:
            self._last_orthogonalization_info = {
                "time_s": float(elapsed),
                "basis_cols_before": int(before_cols),
                "basis_cols_after": int(self._basis_global.shape[1]) if self._basis_global.ndim == 2 else 0,
                "basis_reorth_passes": int(self._last_basis_reorth_passes),
                **_basis_diagnostics(self._basis_global, basis_A=self._basis_A_global),
            }

    def solve(self, A, b, *, full_rhs=None, free_indices: np.ndarray | None = None):
        if PETSc is None:
            raise RuntimeError("PETSc is required for MATLAB-like KSPFGMRES solver types.")
        if self._ksp is None or self._A_petsc is None:
            self.setup_preconditioner(A, free_indices=free_indices)

        rhs_arr = self._prepare_rhs(b, full_rhs=full_rhs)
        t0 = perf_counter()
        reported_history: list[float] = []
        true_history: list[float] = []
        if self._diagnostics_enabled:
            reported_history, true_history = self._install_true_residual_monitor(rhs_arr)
        rhs = global_array_to_petsc_vec(
            rhs_arr,
            comm=self._A_petsc.getComm(),
            ownership_range=self._ownership_range if int(self._A_petsc.getComm().getSize()) > 1 else None,
            bsize=self._A_petsc.getBlockSize() or None,
        )
        x = self._A_petsc.createVecRight()
        x.set(0.0)
        self._ksp.solve(rhs, x)
        nit = int(self._ksp.getIterationNumber())
        elapsed = perf_counter() - t0
        self.iteration_collector.store_iteration(self.instance_id, nit, elapsed)
        if self._diagnostics_enabled:
            x_total = petsc_vec_to_global_array(x)
            resid = rhs_arr - _matvec(self._A_petsc, x_total)
            rhs_norm = float(np.linalg.norm(rhs_arr))
            if rhs_norm == 0.0:
                rhs_norm = 1.0
            if not true_history:
                true_history = [float(np.linalg.norm(resid) / rhs_norm)]
            self._last_solve_info = {
                "iterations": int(nit),
                "time_s": float(elapsed),
                "basis_cols": int(self._basis_global.shape[1]) if self._basis_global.ndim == 2 else 0,
                "rhs_norm": rhs_norm,
                "coarse_initial_guess_norm": float(np.linalg.norm(self._coarse_initial_guess_local(np.asarray(rhs.getArray(readonly=True), dtype=np.float64)))),
                "reported_residual_history": list(reported_history),
                "true_residual_history": list(true_history),
                "true_residual_final": float(np.linalg.norm(resid) / rhs_norm),
            }
            self._ksp.cancelMonitor()
        if self.verbose:
            print(f"{nit}|", end="")
        return self._restrict_solution(petsc_vec_to_global_array(x))

    def copy(self):
        clone = self.__class__(
            self.pc_type,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            tolerance_deflation_basis=self.tolerance_deflation_basis,
            verbose=self.verbose,
            q_mask=self.q_mask,
            coord=self.coord,
            preconditioner_options=self.preconditioner_options,
        )
        # Share the current basis until the clone appends/reorthogonalizes and rebinds it.
        clone.deflation_basis = self.deflation_basis
        clone.iteration_collector = self.iteration_collector
        clone.instance_id = self.iteration_collector.register_instance()
        clone._diagnostics_enabled = self._diagnostics_enabled
        return clone


class PetscKSPMatlabDeflatedFGMRESReorthSolver(PetscKSPMatlabDeflatedFGMRESSolver):
    """MATLAB-like PETSc FGMRES with stronger Krylov orthogonality and repeated A-reorthogonalization."""

    def __init__(
        self,
        pc_type: str = "GAMG",
        *,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        tolerance_deflation_basis: float = 1e-3,
        verbose: bool = False,
        q_mask: np.ndarray | None = None,
        coord: np.ndarray | None = None,
        preconditioner_options: dict | None = None,
    ) -> None:
        opts = dict(preconditioner_options or {})
        opts.setdefault("ksp_gmres_classicalgramschmidt", True)
        opts.setdefault("ksp_gmres_cgs_refinement_type", "refine_always")
        opts.setdefault("deflation_reorth_passes", 3)
        super().__init__(
            pc_type,
            tolerance=tolerance,
            max_iterations=max_iterations,
            tolerance_deflation_basis=tolerance_deflation_basis,
            verbose=verbose,
            q_mask=q_mask,
            coord=coord,
            preconditioner_options=opts,
        )


class PetscMatlabExactDFGMRESSolver(PetscKSPMatlabDeflatedFGMRESSolver):
    """MATLAB-style DFGMRES outer iteration driven directly in Python.

    PETSc is used for:
    - the assembled operator storage
    - the inner PREONLY preconditioner apply
    - GAMG/HYPRE setup

    The outer Arnoldi/projection loop intentionally mirrors MATLAB's
    ``dfgmres_solver.m`` instead of PETSc's built-in KSP implementation.
    """

    def __init__(
        self,
        pc_type: str = "GAMG",
        *,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        tolerance_deflation_basis: float = 1e-3,
        verbose: bool = False,
        q_mask: np.ndarray | None = None,
        coord: np.ndarray | None = None,
        preconditioner_options: dict | None = None,
    ) -> None:
        super().__init__(
            pc_type,
            tolerance=tolerance,
            max_iterations=max_iterations,
            tolerance_deflation_basis=tolerance_deflation_basis,
            verbose=verbose,
            q_mask=q_mask,
            coord=coord,
            preconditioner_options=preconditioner_options,
        )
        self._matvec_in = None
        self._matvec_out = None
        self._prec_in = None
        self._prec_out = None

    def _reuse_preconditioner_enabled(self) -> bool:
        return bool(self.preconditioner_options.get("recycle_preconditioner", False))

    def _preserve_preconditioner_state_between_solves(self) -> bool:
        return bool(
            self._reuse_preconditioner_enabled()
            or self._preconditioner_matrix_policy == "lagged"
            or self._preconditioner_source_is_static()
        )

    def _clear_transient_vectors(self) -> None:
        self._matvec_in = None
        self._matvec_out = None
        self._prec_in = None
        self._prec_out = None

    def _reset_petsc_objects(self) -> None:
        self._clear_transient_vectors()
        super()._reset_petsc_objects()

    def release_iteration_resources(self) -> None:
        if self._preserve_preconditioner_state_between_solves():
            self._clear_transient_vectors()
            self._last_solve_info = {}
            self._last_orthogonalization_info = {}
            return
        self._reset_petsc_objects()

    def _default_null_space(self):
        null_space = self.preconditioner_options.get("null_space")
        if null_space is not None:
            return null_space
        if self._pc_backend not in {"gamg", "hypre"} or self.coord is None or not self.q_mask.size:
            return None
        return make_near_nullspace_elasticity(
            self.coord,
            q_mask=self.q_mask,
            center_coordinates=True,
            return_full=self._using_full_system,
        )

    def _configure_reuse_flags(self) -> None:
        if self._inner_ksp is None:
            return
        if hasattr(self._inner_ksp, "setReusePreconditioner"):
            self._inner_ksp.setReusePreconditioner(True)
        pc = self._inner_ksp.getPC()
        if hasattr(pc, "setReusePreconditioner"):
            pc.setReusePreconditioner(True)

    def _can_reuse_inner_ksp(self, A_petsc, P_petsc) -> bool:
        if self._inner_ksp is None:
            return False
        if self._inner_ksp.getType() != PETSc.KSP.Type.PREONLY:
            return False
        return self._matrix_compatible(A_petsc, self._A_petsc) and self._matrix_compatible(P_petsc, self._P_petsc)

    def _prepare_operator_matrix(self, A, *, full_matrix=None, free_indices: np.ndarray | None = None):
        operator_matrix = self._prepare_operator(A, full_matrix=full_matrix, free_indices=free_indices)
        comm = self._matrix_comm()
        block_size = None
        ownership_range = None
        if self._using_full_system and self.q_mask.size:
            block_size = int(self.q_mask.shape[0])
        elif self.preconditioner_options.get("block_size") is not None:
            block_size = int(self.preconditioner_options["block_size"])
        if int(comm.getSize()) > 1 and block_size is not None and not (
            PETSc is not None and isinstance(operator_matrix, PETSc.Mat)
        ):
            ownership_range = owned_block_range(operator_matrix.shape[0] // block_size, block_size, comm)
        A_petsc = to_petsc_aij_matrix(
            operator_matrix,
            comm=operator_matrix.getComm() if hasattr(operator_matrix, "getComm") else comm,
            block_size=block_size,
            ownership_range=ownership_range,
        )
        owns = not (PETSc is not None and isinstance(operator_matrix, PETSc.Mat))
        if self._using_full_system and self.q_mask.size:
            A_petsc.setBlockSize(int(self.q_mask.shape[0]))
        null_space = self._default_null_space()
        if self._pc_backend in {"gamg", "hypre"}:
            A_petsc, near_nullspace, near_nullspace_vecs = attach_near_nullspace(A_petsc, null_space)
        else:
            near_nullspace = None
            near_nullspace_vecs = []
        return A_petsc, owns, A_petsc.getOwnershipRange(), A_petsc.getComm().tompi4py(), near_nullspace, near_nullspace_vecs

    def _prepare_preconditioning_matrix(self, source_matrix):
        if source_matrix is None:
            raise ValueError("Explicit preconditioning matrix is required for this backend")
        if PETSc is not None and isinstance(source_matrix, PETSc.Mat):
            return source_matrix, False
        comm = self._matrix_comm()
        block_size = int(self.q_mask.shape[0]) if self._using_full_system and self.q_mask.size else None
        P_petsc = to_petsc_aij_matrix(source_matrix, comm=comm, block_size=block_size)
        return P_petsc, True

    def needs_preconditioning_matrix_refresh(self) -> bool:
        if not self.preconditioner_requires_explicit_matrix():
            return False
        rebuild, _reason = self._should_rebuild_preconditioner()
        return rebuild

    def setup_preconditioner(
        self,
        A,
        *,
        full_matrix=None,
        free_indices: np.ndarray | None = None,
        preconditioning_matrix=None,
    ):
        t0 = perf_counter()
        if PETSc is None:
            raise RuntimeError("PETSc is required for explicit MATLAB-style DFGMRES solver types.")
        new_A_petsc, new_A_owned, ownership_range, mpi_comm, near_nullspace, near_nullspace_vecs = self._prepare_operator_matrix(
            A,
            full_matrix=full_matrix,
            free_indices=free_indices,
        )

        rebuild_preconditioner, rebuild_reason = self._should_rebuild_preconditioner()
        new_P_petsc = self._P_petsc
        new_P_owned = self._owns_P_petsc
        if self.preconditioner_requires_explicit_matrix():
            if rebuild_preconditioner:
                new_P_petsc, new_P_owned = self._prepare_preconditioning_matrix(preconditioning_matrix)
            elif new_P_petsc is None:
                raise ValueError("BDDC backend requested lagged reuse without an initialized preconditioning matrix")
        else:
            if rebuild_preconditioner:
                if preconditioning_matrix is not None:
                    new_P_petsc, new_P_owned = self._prepare_preconditioning_matrix(preconditioning_matrix)
                elif self._preconditioner_matrix_policy == "current":
                    new_P_petsc = new_A_petsc
                    new_P_owned = False
                else:
                    new_P_petsc = new_A_petsc.copy()
                    new_P_owned = True
            elif new_P_petsc is None:
                new_P_petsc = new_A_petsc
                new_P_owned = False
                rebuild_preconditioner = True
                rebuild_reason = "initial"

        if self._using_full_system and self.q_mask.size and new_P_petsc is not None:
            try:
                new_P_petsc.setBlockSize(int(self.q_mask.shape[0]))
            except Exception:
                pass

        reuse_inner = self._can_reuse_inner_ksp(new_A_petsc, new_P_petsc)
        old_A_petsc = self._A_petsc
        old_A_owned = self._owns_A_petsc
        old_P_petsc = self._P_petsc
        old_P_owned = self._owns_P_petsc

        if not reuse_inner:
            self._reset_petsc_objects()
            self._inner_ksp = PETSc.KSP().create(comm=new_A_petsc.getComm())
            self._inner_ksp.setOptionsPrefix(f"{self._options_prefix}inner_")
            self._inner_ksp.setType(PETSc.KSP.Type.PREONLY)

        self._clear_transient_vectors()
        self._A_petsc = new_A_petsc
        self._owns_A_petsc = new_A_owned
        self._P_petsc = new_P_petsc
        self._owns_P_petsc = new_P_owned
        self._ownership_range = ownership_range
        self._mpi_comm = mpi_comm
        self._near_nullspace = near_nullspace
        self._near_nullspace_vecs = near_nullspace_vecs
        self._configure_inner_pc_only_options()
        self._inner_ksp.setOperators(self._A_petsc, self._P_petsc)
        self._configure_inner_pc()
        self._inner_ksp.setFromOptions()
        self._configure_reuse_flags()
        self._inner_ksp.setUp()

        if old_A_petsc is not None and old_A_petsc is not self._A_petsc:
            self._destroy_owned_petsc_matrix(old_A_petsc, old_A_owned)
        if old_P_petsc is not None and old_P_petsc is not self._P_petsc and old_P_petsc is not self._A_petsc:
            self._destroy_owned_petsc_matrix(old_P_petsc, old_P_owned)

        self._projector_dirty = True
        if rebuild_preconditioner:
            self._mark_preconditioner_rebuilt(reason=rebuild_reason)
        else:
            self._mark_preconditioner_reused()
        self._record_preconditioner_setup_time(perf_counter() - t0)

    def _apply_inner_preconditioner(self, x: np.ndarray) -> np.ndarray:
        if self._prec_in is None or self._prec_out is None:
            self._prec_in = self._A_petsc.createVecRight()
            self._prec_out = self._A_petsc.createVecRight()
        rhs_arr = self._prec_in.getArray(readonly=False)
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.size != rhs_arr.size:
            r0, r1 = self._ownership_range if self._ownership_range is not None else (0, rhs_arr.size)
            x_arr = x_arr[r0:r1]
        rhs_arr[...] = x_arr
        self._prec_out.set(0.0)
        t0 = perf_counter()
        self._inner_ksp.getPC().apply(self._prec_in, self._prec_out)
        self._record_preconditioner_apply_time(perf_counter() - t0)
        return petsc_vec_to_global_array(self._prec_out)

    def _apply_inner_preconditioner_local(self, x_local: np.ndarray) -> np.ndarray:
        if self._prec_in is None or self._prec_out is None:
            self._prec_in = self._A_petsc.createVecRight()
            self._prec_out = self._A_petsc.createVecRight()
        rhs_arr = self._prec_in.getArray(readonly=False)
        rhs_arr[...] = np.asarray(x_local, dtype=np.float64)
        self._prec_out.set(0.0)
        t0 = perf_counter()
        self._inner_ksp.getPC().apply(self._prec_in, self._prec_out)
        self._record_preconditioner_apply_time(perf_counter() - t0)
        return np.asarray(self._prec_out.getArray(readonly=True), dtype=np.float64).copy()

    def _petsc_matvec(self, x: np.ndarray) -> np.ndarray:
        if self._matvec_in is None or self._matvec_out is None:
            self._matvec_in = self._A_petsc.createVecRight()
            self._matvec_out = self._A_petsc.createVecLeft()
        x_arr = self._matvec_in.getArray(readonly=False)
        x_global = np.asarray(x, dtype=np.float64)
        if x_global.size != x_arr.size:
            r0, r1 = self._ownership_range if self._ownership_range is not None else (0, x_arr.size)
            x_global = x_global[r0:r1]
        x_arr[...] = x_global
        self._A_petsc.mult(self._matvec_in, self._matvec_out)
        return petsc_vec_to_global_array(self._matvec_out)

    def _petsc_matvec_local(self, x_local: np.ndarray) -> np.ndarray:
        if self._matvec_in is None or self._matvec_out is None:
            self._matvec_in = self._A_petsc.createVecRight()
            self._matvec_out = self._A_petsc.createVecLeft()
        x_arr = self._matvec_in.getArray(readonly=False)
        x_arr[...] = np.asarray(x_local, dtype=np.float64)
        self._A_petsc.mult(self._matvec_in, self._matvec_out)
        return np.asarray(self._matvec_out.getArray(readonly=True), dtype=np.float64).copy()

    def solve(self, A, b, *, full_rhs=None, local_rhs=None, free_indices: np.ndarray | None = None):
        if PETSc is None:
            raise RuntimeError("PETSc is required for explicit MATLAB-style DFGMRES solver types.")
        if self._inner_ksp is None or self._A_petsc is None:
            self.setup_preconditioner(A, free_indices=free_indices)
        self._ensure_projector_ready()

        t0 = perf_counter()
        timing_stats: dict[str, float] = {}

        use_distributed_local = bool(self._mpi_comm is not None and int(self._A_petsc.getComm().getSize()) > 1)
        rhs_arr = None

        def _timed_matvec(v: np.ndarray) -> np.ndarray:
            t_mat = perf_counter()
            out = self._petsc_matvec(v)
            timing_stats["matvec_s"] = timing_stats.get("matvec_s", 0.0) + (perf_counter() - t_mat)
            return out

        def _timed_prec(v: np.ndarray) -> np.ndarray:
            t_prec = perf_counter()
            out = self._apply_inner_preconditioner(v)
            timing_stats["preconditioner_apply_s"] = timing_stats.get("preconditioner_apply_s", 0.0) + (perf_counter() - t_prec)
            return out

        if use_distributed_local:
            r0, r1 = self._ownership_range
            if local_rhs is not None:
                rhs_local = np.asarray(local_rhs, dtype=np.float64).reshape(-1)
            else:
                rhs_arr = self._prepare_rhs(b, full_rhs=full_rhs)
                rhs_local = np.asarray(rhs_arr[r0:r1], dtype=np.float64)
            basis_local = self._basis_local if self._basis_local.size else np.empty((rhs_local.size, 0), dtype=np.float64)

            def _timed_matvec_local(v_local: np.ndarray) -> np.ndarray:
                t_mat = perf_counter()
                out = self._petsc_matvec_local(v_local)
                timing_stats["matvec_s"] = timing_stats.get("matvec_s", 0.0) + (perf_counter() - t_mat)
                return out

            def _timed_prec_local(v_local: np.ndarray) -> np.ndarray:
                t_prec = perf_counter()
                out = self._apply_inner_preconditioner_local(v_local)
                timing_stats["preconditioner_apply_s"] = timing_stats.get("preconditioner_apply_s", 0.0) + (perf_counter() - t_prec)
                return out

            compiled_outer = bool(self.preconditioner_options.get("compiled_outer", False))
            if compiled_outer:
                x_local, nit, res_hist = dfgmres_matlab_exact_distributed_compiled(
                    _timed_matvec_local,
                    rhs_local,
                    _timed_prec_local,
                    basis_local,
                    self.max_iterations,
                    self.tolerance,
                    self._mpi_comm,
                    None,
                    stats=timing_stats,
                )
            else:
                x_local, nit, res_hist = dfgmres_matlab_exact_distributed(
                    _timed_matvec_local,
                    rhs_local,
                    _timed_prec_local,
                    basis_local,
                    self.max_iterations,
                    self.tolerance,
                    self._mpi_comm,
                    None,
                    stats=timing_stats,
                )
            x_total = np.concatenate(self._mpi_comm.allgather(np.asarray(x_local, dtype=np.float64)))
            if self._diagnostics_enabled:
                rhs_arr = np.concatenate(self._mpi_comm.allgather(rhs_local))
        else:
            rhs_arr = self._prepare_rhs(b, full_rhs=full_rhs)
            basis = self.deflation_basis if self.deflation_basis.size else np.empty((rhs_arr.size, 0), dtype=np.float64)
            x_total, nit, res_hist = dfgmres_matlab_exact(
                _timed_matvec,
                rhs_arr,
                _timed_prec,
                basis,
                self.max_iterations,
                self.tolerance,
                None,
                stats=timing_stats,
            )
        elapsed = perf_counter() - t0
        self.iteration_collector.store_iteration(self.instance_id, int(nit), elapsed)
        if self._diagnostics_enabled:
            if rhs_arr is None:
                rhs_arr = self._prepare_rhs(b, full_rhs=full_rhs)
            basis = self.deflation_basis if self.deflation_basis.size else np.empty((rhs_arr.size, 0), dtype=np.float64)
            rhs_norm = float(np.linalg.norm(rhs_arr))
            if rhs_norm == 0.0:
                rhs_norm = 1.0
            resid = rhs_arr - self._petsc_matvec(x_total)
            coarse_guess = basis @ (basis.T @ rhs_arr) if basis.size else np.zeros_like(rhs_arr)
            self._last_solve_info = {
                "iterations": int(nit),
                "time_s": float(elapsed),
                "basis_cols": int(basis.shape[1]) if basis.ndim == 2 else 0,
                "rhs_norm": rhs_norm,
                "coarse_initial_guess_norm": float(np.linalg.norm(coarse_guess)),
                "reported_residual_history": np.asarray(res_hist, dtype=np.float64).tolist(),
                "true_residual_history": np.asarray(res_hist, dtype=np.float64).tolist(),
                "true_residual_final": float(np.linalg.norm(resid) / rhs_norm),
                "timings": {k: float(v) for k, v in timing_stats.items()},
            }
        if self.verbose:
            print(f"{nit}|", end="")
        return self._restrict_solution(x_total)

    def copy(self):
        clone = self.__class__(
            self.pc_type,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            tolerance_deflation_basis=self.tolerance_deflation_basis,
            verbose=self.verbose,
            q_mask=self.q_mask,
            coord=self.coord,
            preconditioner_options=self.preconditioner_options,
        )
        # Share the current basis until the clone appends/reorthogonalizes and rebinds it.
        clone.deflation_basis = self.deflation_basis
        clone.iteration_collector = self.iteration_collector
        clone.instance_id = self.iteration_collector.register_instance()
        clone._diagnostics_enabled = self._diagnostics_enabled
        clone._preconditioner_diagnostics = self._preconditioner_diagnostics
        return clone


class SolverFactory:
    """Factory for creating solver backends from a compact configuration."""

    @staticmethod
    def create(
        solver_type: str = "DFGMRES_GAMG",
        *,
        tolerance: float = 1e-1,
        max_iterations: int = 200,
        deflation_basis_tolerance: float = 1e-3,
        verbose: bool = False,
        q_mask: np.ndarray | None = None,
        coord: np.ndarray | None = None,
        preconditioner_options: dict | None = None,
    ):
        if preconditioner_options is None:
            preconditioner_options = {}

        solver_type = str(solver_type).upper()
        q_mask = np.array([], dtype=bool) if q_mask is None else np.asarray(q_mask, dtype=bool)

        def _pc_builder(kind: str):
            return lambda A: build_preconditioner(
                kind,
                A,
                q_mask=q_mask,
                coord=coord,
                **preconditioner_options,
            ).apply

        if solver_type == "DIRECT":
            return ScipyDirectSolver()

        if solver_type in {"PETSC_DIRECT", "KSPDIRECT", "KSPDIRECT_LU", "KSPPREONLY_LU"}:
            return DirectSolver(
                factor_solver_type=preconditioner_options.get("factor_solver_type"),
            )

        if solver_type == "FGMRES":
            return FGMRESSolver(
                _pc_builder("JACOBI"),
                tolerance,
                max_iterations,
                deflation_basis_tolerance,
                verbose,
            )

        if solver_type.startswith("PETSC_MATLAB_DFGMRES"):
            pc_type = "GAMG"
            if "HYPRE" in solver_type:
                pc_type = "HYPRE"
            if "JACOBI" in solver_type:
                pc_type = "JACOBI"
            return PetscMatlabExactDFGMRESSolver(
                pc_type,
                tolerance=tolerance,
                max_iterations=max_iterations,
                tolerance_deflation_basis=deflation_basis_tolerance,
                verbose=verbose,
                q_mask=q_mask,
                coord=coord,
                preconditioner_options=preconditioner_options,
            )

        if solver_type.startswith("KSPFGMRES"):
            if "MATLAB" in solver_type or "DFGMRES" in solver_type:
                pc_type = "GAMG"
                if "HYPRE" in solver_type:
                    pc_type = "HYPRE"
                if "JACOBI" in solver_type:
                    pc_type = "JACOBI"
                solver_cls = PetscKSPMatlabDeflatedFGMRESSolver
                if "REORTH" in solver_type:
                    solver_cls = PetscKSPMatlabDeflatedFGMRESReorthSolver
                return solver_cls(
                    pc_type,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    tolerance_deflation_basis=deflation_basis_tolerance,
                    verbose=verbose,
                    q_mask=q_mask,
                    coord=coord,
                    preconditioner_options=preconditioner_options,
                )
            pc_type = "GAMG"
            if "HYPRE" in solver_type:
                pc_type = "HYPRE"
            if "JACOBI" in solver_type:
                pc_type = "JACOBI"
            return PetscKSPFGMRESSolver(
                pc_type,
                tolerance=tolerance,
                max_iterations=max_iterations,
                tolerance_deflation_basis=deflation_basis_tolerance,
                verbose=verbose,
                q_mask=q_mask,
                coord=coord,
                preconditioner_options=preconditioner_options,
            )

        if "DEFLATION" in solver_type and solver_type.startswith("KSPGMRES"):
            pc_type = "GAMG"
            if "HYPRE" in solver_type:
                pc_type = "HYPRE"
            if "JACOBI" in solver_type:
                pc_type = "JACOBI"
            return PetscKSPGMRESDeflationSolver(
                pc_type,
                tolerance=tolerance,
                max_iterations=max_iterations,
                tolerance_deflation_basis=deflation_basis_tolerance,
                verbose=verbose,
                q_mask=q_mask,
                coord=coord,
                preconditioner_options=preconditioner_options,
            )

        # default to GAMG-preconditioned deflated FGMRES
        if "GAMG" in solver_type:
            return DeflatedFGMRESSolver(
                _pc_builder("GAMG"),
                tolerance=tolerance,
                max_iterations=max_iterations,
                tolerance_deflation_basis=deflation_basis_tolerance,
                verbose=verbose,
            )

        if "JACOBI" in solver_type:
            return DeflatedFGMRESSolver(
                _pc_builder("JACOBI"),
                tolerance=tolerance,
                max_iterations=max_iterations,
                tolerance_deflation_basis=deflation_basis_tolerance,
                verbose=verbose,
            )

        if "DIRECT" in solver_type:
            return DirectSolver()

        raise ValueError(f"Unsupported solver_type {solver_type!r}")

    @staticmethod
    def from_config(config: LinearSolverConfig, *, q_mask: np.ndarray | None = None, coord: np.ndarray | None = None):
        preconditioner_options = {
            "threads": config.threads,
            "print_level": config.print_level,
            "use_as_preconditioner": config.use_as_preconditioner,
            "pc_backend": config.pc_backend,
            "preconditioner_matrix_source": config.preconditioner_matrix_source,
            "preconditioner_matrix_policy": config.preconditioner_matrix_policy,
            "preconditioner_rebuild_policy": config.preconditioner_rebuild_policy,
            "preconditioner_rebuild_interval": config.preconditioner_rebuild_interval,
        }
        if config.factor_solver_type is not None:
            preconditioner_options["factor_solver_type"] = config.factor_solver_type
        if config.pc_gamg_process_eq_limit is not None:
            preconditioner_options["pc_gamg_process_eq_limit"] = config.pc_gamg_process_eq_limit
        if config.pc_gamg_threshold is not None:
            preconditioner_options["pc_gamg_threshold"] = config.pc_gamg_threshold
        if config.pc_gamg_aggressive_coarsening is not None:
            preconditioner_options["pc_gamg_aggressive_coarsening"] = config.pc_gamg_aggressive_coarsening
        if config.pc_gamg_aggressive_square_graph is not None:
            preconditioner_options["pc_gamg_aggressive_square_graph"] = config.pc_gamg_aggressive_square_graph
        if config.pc_gamg_aggressive_mis_k is not None:
            preconditioner_options["pc_gamg_aggressive_mis_k"] = config.pc_gamg_aggressive_mis_k
        if config.pc_hypre_coarsen_type is not None:
            preconditioner_options["pc_hypre_boomeramg_coarsen_type"] = config.pc_hypre_coarsen_type
        if config.pc_hypre_interp_type is not None:
            preconditioner_options["pc_hypre_boomeramg_interp_type"] = config.pc_hypre_interp_type
        if config.pc_hypre_strong_threshold is not None:
            preconditioner_options["pc_hypre_boomeramg_strong_threshold"] = config.pc_hypre_strong_threshold
        if config.pc_hypre_P_max is not None:
            preconditioner_options["pc_hypre_boomeramg_P_max"] = config.pc_hypre_P_max
        if config.pc_hypre_agg_nl is not None:
            preconditioner_options["pc_hypre_boomeramg_agg_nl"] = config.pc_hypre_agg_nl
        if config.pc_hypre_nongalerkin_tol is not None:
            preconditioner_options["pc_hypre_boomeramg_nongalerkin_tol"] = config.pc_hypre_nongalerkin_tol
        if config.pc_bddc_symmetric:
            preconditioner_options["pc_bddc_symmetric"] = True
        if config.pc_bddc_dirichlet_ksp_type is not None:
            preconditioner_options["pc_bddc_dirichlet_ksp_type"] = config.pc_bddc_dirichlet_ksp_type
        if config.pc_bddc_dirichlet_pc_type is not None:
            preconditioner_options["pc_bddc_dirichlet_pc_type"] = config.pc_bddc_dirichlet_pc_type
        if config.pc_bddc_neumann_ksp_type is not None:
            preconditioner_options["pc_bddc_neumann_ksp_type"] = config.pc_bddc_neumann_ksp_type
        if config.pc_bddc_neumann_pc_type is not None:
            preconditioner_options["pc_bddc_neumann_pc_type"] = config.pc_bddc_neumann_pc_type
        if config.pc_bddc_coarse_ksp_type is not None:
            preconditioner_options["pc_bddc_coarse_ksp_type"] = config.pc_bddc_coarse_ksp_type
        if config.pc_bddc_coarse_pc_type is not None:
            preconditioner_options["pc_bddc_coarse_pc_type"] = config.pc_bddc_coarse_pc_type
        if config.pc_bddc_dirichlet_approximate is not None:
            preconditioner_options["pc_bddc_dirichlet_approximate"] = config.pc_bddc_dirichlet_approximate
        if config.pc_bddc_neumann_approximate is not None:
            preconditioner_options["pc_bddc_neumann_approximate"] = config.pc_bddc_neumann_approximate
        if config.pc_bddc_switch_static is not None:
            preconditioner_options["pc_bddc_switch_static"] = config.pc_bddc_switch_static
        if config.pc_bddc_use_deluxe_scaling is not None:
            preconditioner_options["pc_bddc_use_deluxe_scaling"] = config.pc_bddc_use_deluxe_scaling
        if config.pc_bddc_use_vertices is not None:
            preconditioner_options["pc_bddc_use_vertices"] = config.pc_bddc_use_vertices
        if config.pc_bddc_use_edges is not None:
            preconditioner_options["pc_bddc_use_edges"] = config.pc_bddc_use_edges
        if config.pc_bddc_use_faces is not None:
            preconditioner_options["pc_bddc_use_faces"] = config.pc_bddc_use_faces
        if config.pc_bddc_use_change_of_basis is not None:
            preconditioner_options["pc_bddc_use_change_of_basis"] = config.pc_bddc_use_change_of_basis
        if config.pc_bddc_use_change_on_faces is not None:
            preconditioner_options["pc_bddc_use_change_on_faces"] = config.pc_bddc_use_change_on_faces
        if config.pc_bddc_check_level is not None:
            preconditioner_options["pc_bddc_check_level"] = config.pc_bddc_check_level
        if config.compiled_outer:
            preconditioner_options["compiled_outer"] = True
        if config.recycle_preconditioner:
            preconditioner_options["recycle_preconditioner"] = True
        return SolverFactory.create(
            config.solver_type,
            tolerance=config.tolerance,
            max_iterations=config.max_iterations,
            deflation_basis_tolerance=config.deflation_basis_tolerance,
            verbose=bool(config.verbose),
            q_mask=q_mask,
            coord=coord,
            preconditioner_options=preconditioner_options,
        )


class FGMRESSolver(DeflatedFGMRESSolver):
    """FGMRES variant with deflation disabled."""

    def expand_deflation_basis(self, _additional_vectors):
        return
