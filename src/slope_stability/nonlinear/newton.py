"""Nonlinear Newton strategies."""

from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np

from .damping import damping, damping_alg5
from ..utils import q_to_free_indices
from ..utils import extract_submatrix_free, release_petsc_aij_matrix, to_petsc_aij_matrix

try:  # pragma: no cover - PETSc optional in tests
    from petsc4py import PETSc
except Exception:  # pragma: no cover
    PETSc = None


def _to_float_matrix(U: np.ndarray) -> np.ndarray:
    return np.asarray(U, dtype=np.float64)


def _to_free_vector(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return arr.reshape(-1, order="F")[q_to_free_indices(Q)]


def _free_dot(a: np.ndarray, b: np.ndarray, Q: np.ndarray) -> float:
    return float(np.dot(_to_free_vector(a, Q), _to_free_vector(b, Q)))


def _free_norm(v: np.ndarray, Q: np.ndarray) -> float:
    return float(np.linalg.norm(_to_free_vector(v, Q)))


def _combine_matrices(alpha: float, A, beta: float, B):
    if PETSc is not None and isinstance(A, PETSc.Mat) and not isinstance(B, PETSc.Mat):
        B = to_petsc_aij_matrix(B, comm=A.getComm(), block_size=A.getBlockSize() or None)
    elif PETSc is not None and isinstance(B, PETSc.Mat) and not isinstance(A, PETSc.Mat):
        A = to_petsc_aij_matrix(A, comm=B.getComm(), block_size=B.getBlockSize() or None)
    if PETSc is not None and isinstance(A, PETSc.Mat) and isinstance(B, PETSc.Mat):
        C = A.copy()
        C.scale(float(alpha))
        C.axpy(float(beta), B)
        C.assemble()
        return C
    return alpha * A + beta * B


def _setup_linear_system(
    linear_system_solver,
    A_free,
    *,
    A_full=None,
    free_idx: np.ndarray | None = None,
    preconditioning_matrix=None,
) -> None:
    try:
        linear_system_solver.setup_preconditioner(
            A_free,
            full_matrix=A_full,
            free_indices=free_idx,
            preconditioning_matrix=preconditioning_matrix,
        )
    except TypeError:
        linear_system_solver.setup_preconditioner(A_free)


def _solve_linear_system(linear_system_solver, A_free, b_free, *, b_full=None, free_idx: np.ndarray | None = None):
    try:
        return linear_system_solver.solve(A_free, b_free, full_rhs=b_full, free_indices=free_idx)
    except TypeError:
        return linear_system_solver.solve(A_free, b_free)


def _solve_linear_system_local(
    linear_system_solver,
    A_free,
    b_free,
    *,
    b_full=None,
    local_rhs=None,
    free_idx: np.ndarray | None = None,
):
    try:
        return linear_system_solver.solve(
            A_free,
            b_free,
            full_rhs=b_full,
            local_rhs=local_rhs,
            free_indices=free_idx,
        )
    except TypeError:
        return _solve_linear_system(linear_system_solver, A_free, b_free, b_full=b_full, free_idx=free_idx)


def _prefers_full_system_operator(linear_system_solver, A_full) -> bool:
    if PETSc is None or not isinstance(A_full, PETSc.Mat):
        return False
    prefers = getattr(linear_system_solver, "prefers_full_system_operator", None)
    if callable(prefers):
        return bool(prefers())
    return False


def _release_iteration_resources(linear_system_solver) -> None:
    release = getattr(linear_system_solver, "release_iteration_resources", None)
    if callable(release):
        release()


def _needs_preconditioning_matrix_refresh(linear_system_solver) -> bool:
    fn = getattr(linear_system_solver, "needs_preconditioning_matrix_refresh", None)
    if callable(fn):
        return bool(fn())
    return False


def _collector_snapshot(linear_system_solver) -> dict[str, float | int]:
    collector = getattr(linear_system_solver, "iteration_collector", None)
    if collector is None:
        return {
            "iterations": 0,
            "solve_time": 0.0,
            "preconditioner_time": 0.0,
            "orthogonalization_time": 0.0,
        }
    return {
        "iterations": int(collector.get_total_iterations()),
        "solve_time": float(collector.get_total_solve_time()),
        "preconditioner_time": float(collector.get_total_preconditioner_time()),
        "orthogonalization_time": float(collector.get_total_orthogonalization_time()),
    }


def _collector_delta(before: dict[str, float | int], after: dict[str, float | int]) -> dict[str, float | int]:
    return {
        "iterations": int(after["iterations"]) - int(before["iterations"]),
        "solve_time": float(after["solve_time"]) - float(before["solve_time"]),
        "preconditioner_time": float(after["preconditioner_time"]) - float(before["preconditioner_time"]),
        "orthogonalization_time": float(after["orthogonalization_time"]) - float(before["orthogonalization_time"]),
    }


def _emit_progress(progress_callback: Callable[[dict], None] | None, *, event: str, **payload) -> None:
    if progress_callback is None:
        return
    progress_callback({"event": event, **payload})


def _requires_explicit_preconditioning_matrix(linear_system_solver) -> bool:
    fn = getattr(linear_system_solver, "preconditioner_requires_explicit_matrix", None)
    if callable(fn):
        return bool(fn())
    return False


def _preconditioning_matrix_source(linear_system_solver) -> str:
    fn = getattr(linear_system_solver, "get_preconditioner_matrix_source", None)
    if callable(fn):
        try:
            return str(fn()).strip().lower()
        except Exception:
            return "tangent"
    return "tangent"


def _explicit_preconditioning_matrix(constitutive_matrix_builder, linear_system_solver, *, regularization_r: float | None, K_elast=None):
    source = _preconditioning_matrix_source(linear_system_solver)
    if source == "elastic":
        fn = getattr(constitutive_matrix_builder, "build_bddc_elastic_matrix", None)
        if callable(fn):
            return fn()
        if K_elast is not None:
            return K_elast
        raise RuntimeError("Elastic preconditioning matrix requested, but no elastic matrix is available")
    if regularization_r is not None:
        fn = getattr(constitutive_matrix_builder, "build_bddc_regularized_matrix", None)
        if callable(fn):
            return fn(float(regularization_r))
    if source == "regularized":
        raise RuntimeError("Regularized preconditioning matrix requested, but no regularized builder is available")
    fn = getattr(constitutive_matrix_builder, "build_bddc_tangent_matrix", None)
    if callable(fn):
        return fn()
    raise RuntimeError("Linear solver requested an explicit preconditioning matrix, but the constitutive builder cannot provide one")


def _destroy_petsc_mat(A) -> None:
    if PETSc is not None and isinstance(A, PETSc.Mat):
        release_petsc_aij_matrix(A)
        A.destroy()


def _is_builder_cached_matrix(A, constitutive_matrix_builder) -> bool:
    if A is None or constitutive_matrix_builder is None:
        return False
    for attr in (
        "_owned_tangent_mat",
        "_owned_regularized_mat",
        "_bddc_tangent_mat",
        "_bddc_elastic_mat",
        "_bddc_regularized_mat",
    ):
        if getattr(constitutive_matrix_builder, attr, None) is A:
            return True
    return False


def _cleanup_pre_solve_iteration_mats(*, K_tangent, K_r, use_full_operator: bool, constitutive_matrix_builder=None) -> None:
    if not _is_builder_cached_matrix(K_tangent, constitutive_matrix_builder):
        _destroy_petsc_mat(K_tangent)
    if not bool(use_full_operator) and not _is_builder_cached_matrix(K_r, constitutive_matrix_builder):
        _destroy_petsc_mat(K_r)


def _local_owned_rows_from_field(field: np.ndarray, pattern) -> np.ndarray:
    row0, row1 = pattern.owned_row_range
    flat = np.asarray(field, dtype=np.float64).reshape(-1, order="F")
    return np.asarray(flat[row0:row1], dtype=np.float64)


def _local_owned_free_rows_from_field(field: np.ndarray, pattern) -> np.ndarray:
    local = _local_owned_rows_from_field(field, pattern)
    return np.asarray(local[np.asarray(pattern.owned_free_mask, dtype=bool)], dtype=np.float64)


def _dist_dot_local(x_local: np.ndarray, y_local: np.ndarray, comm) -> float:
    value = float(np.dot(np.asarray(x_local, dtype=np.float64).reshape(-1), np.asarray(y_local, dtype=np.float64).reshape(-1)))
    if comm is None or int(comm.Get_size()) == 1:
        return value
    if hasattr(comm, "allreduce"):
        return float(comm.allreduce(value))
    return float(comm.tompi4py().allreduce(value))


def _dist_norm_local(x_local: np.ndarray, comm) -> float:
    return float(np.sqrt(max(_dist_dot_local(x_local, x_local, comm), 0.0)))


def _build_regularized_if_available(constitutive_matrix_builder, *, lam=None, U, r: float):
    if lam is None:
        fn = getattr(constitutive_matrix_builder, "build_F_K_regularized_reduced", None)
        if callable(fn):
            return fn(U, r)
        return None
    fn = getattr(constitutive_matrix_builder, "build_F_K_regularized_all", None)
    if callable(fn):
        return fn(lam, U, r)
    return None


def _build_regularized_from_cached_if_available(constitutive_matrix_builder, r: float):
    fn = getattr(constitutive_matrix_builder, "build_K_regularized", None)
    if callable(fn):
        return fn(r)
    return None


def _ensure_tangent_matrix_for_regularization(
    constitutive_matrix_builder,
    U_it,
    *,
    K_tangent,
    use_free_build: bool,
):
    if K_tangent is not None:
        return K_tangent
    if use_free_build and _supports_free_builder(constitutive_matrix_builder, "build_F_K_tangent_reduced_free"):
        _unused_F_free, K_tangent = constitutive_matrix_builder.build_F_K_tangent_reduced_free(U_it)
        return K_tangent
    _unused_F, K_tangent = constitutive_matrix_builder.build_F_K_tangent_reduced(U_it)
    return K_tangent


def _supports_free_builder(constitutive_matrix_builder, name: str) -> bool:
    fn = getattr(constitutive_matrix_builder, name, None)
    return callable(fn) and getattr(constitutive_matrix_builder, "owned_tangent_pattern", None) is not None


def _supports_local_builder(constitutive_matrix_builder, name: str) -> bool:
    fn = getattr(constitutive_matrix_builder, name, None)
    return callable(fn) and getattr(constitutive_matrix_builder, "owned_tangent_pattern", None) is not None


def _local_comm_from_operator(A_full):
    if PETSc is None or not isinstance(A_full, PETSc.Mat):
        return None
    return A_full.getComm().tompi4py()


def newton(
    U_ini: np.ndarray,
    tol: float,
    it_newt_max: int,
    it_damp_max: int,
    r_min: float,
    K_elast,
    Q: np.ndarray,
    f: np.ndarray,
    constitutive_matrix_builder,
    linear_system_solver,
    *,
    progress_callback: Callable[[dict], None] | None = None,
):
    """Plain Newton solver for ``F(U) = f``.

    Returns ``(U_it, flag_N, it)``.
    """

    U_it = _to_float_matrix(U_ini)
    Q = np.asarray(Q, dtype=bool)
    shape = U_it.shape

    free_idx = q_to_free_indices(Q)
    if free_idx.size == 0:
        return U_it, 0, 0
    f_free = _to_free_vector(f, Q)

    norm_f = _free_norm(f, Q)
    if norm_f == 0.0:
        norm_f = 1.0

    it = 0
    flag_N = 0
    dU = np.zeros_like(U_it)
    r = float(r_min)
    compute_diffs = True

    while True:
        it += 1
        iter_t0 = perf_counter()
        snap_before_iter = _collector_snapshot(linear_system_solver)

        use_full_operator = _prefers_full_system_operator(linear_system_solver, K_elast)
        use_free_build = _supports_free_builder(constitutive_matrix_builder, "build_F_reduced_free")
        use_local_build = use_full_operator and _supports_local_builder(constitutive_matrix_builder, "build_F_reduced_local")
        comm = _local_comm_from_operator(K_elast) if use_local_build else None
        K_tangent = None
        K_r = None
        F = None
        F_local = None
        F_free_local = None
        f_free_local = None

        if compute_diffs:
            if use_local_build:
                constitutive_matrix_builder.constitutive_problem_stress_tangent(U_it)
                F_local = np.asarray(constitutive_matrix_builder.build_F_local(), dtype=np.float64).reshape(-1)
                F_free_local = np.asarray(constitutive_matrix_builder.build_F_free_local(), dtype=np.float64).reshape(-1)
                F_free = F_free_local
                K_r = constitutive_matrix_builder.build_K_regularized(r)
            elif use_full_operator and _supports_free_builder(constitutive_matrix_builder, "build_F_K_regularized_reduced_free"):
                F_free, K_r = constitutive_matrix_builder.build_F_K_regularized_reduced_free(U_it, r)
                F_free = np.asarray(F_free, dtype=np.float64).reshape(-1)
            elif use_full_operator:
                regularized_pair = _build_regularized_if_available(constitutive_matrix_builder, U=U_it, r=r)
                if regularized_pair is not None:
                    F, K_r = regularized_pair
                    F_free = _to_free_vector(F, Q)
                else:
                    F, K_tangent = constitutive_matrix_builder.build_F_K_tangent_reduced(U_it)
                    F_free = _to_free_vector(F, Q)
            elif _supports_free_builder(constitutive_matrix_builder, "build_F_K_tangent_reduced_free"):
                F_free, K_tangent = constitutive_matrix_builder.build_F_K_tangent_reduced_free(U_it)
                F_free = np.asarray(F_free, dtype=np.float64).reshape(-1)
            else:
                F, K_tangent = constitutive_matrix_builder.build_F_K_tangent_reduced(U_it)
                F_free = _to_free_vector(F, Q)
        else:
            if use_local_build:
                F_local = np.asarray(constitutive_matrix_builder.build_F_reduced_local(U_it), dtype=np.float64).reshape(-1)
                F_free_local = np.asarray(constitutive_matrix_builder.build_F_reduced_free_local(U_it), dtype=np.float64).reshape(-1)
                F_free = F_free_local
            elif use_free_build:
                F_free = np.asarray(constitutive_matrix_builder.build_F_reduced_free(U_it), dtype=np.float64).reshape(-1)
            else:
                F = constitutive_matrix_builder.build_F_reduced(U_it)
                F_free = _to_free_vector(F, Q)

        if use_local_build:
            f_free_local = _local_owned_free_rows_from_field(f, constitutive_matrix_builder.owned_tangent_pattern)
            criterion_abs = _dist_norm_local(F_free_local - f_free_local, comm)
        else:
            criterion_abs = float(np.linalg.norm(F_free - f_free))
        criterion = criterion_abs / norm_f

        if compute_diffs and criterion < tol:
            _emit_progress(
                progress_callback,
                event="newton_iteration",
                solver="newton",
                iteration=int(it),
                criterion=float(criterion_abs),
                rel_residual=float(criterion),
                alpha=None,
                r=float(r),
                linear_iterations=0,
                linear_solve_time=0.0,
                linear_preconditioner_time=0.0,
                linear_orthogonalization_time=0.0,
                iteration_wall_time=float(perf_counter() - iter_t0),
                tolerance=float(tol),
                status="converged",
            )
            _cleanup_pre_solve_iteration_mats(
                K_tangent=K_tangent,
                K_r=K_r,
                use_full_operator=use_full_operator,
                constitutive_matrix_builder=constitutive_matrix_builder,
            )
            break
        if use_local_build:
            f_local = _local_owned_rows_from_field(f, constitutive_matrix_builder.owned_tangent_pattern)
            rhs_local = f_local - F_local
            rhs = f_free_local - F_free_local
        else:
            rhs_local = None
            F_free_local = None
            rhs = f_free - F_free
        if K_r is None:
            cached_regularized = _build_regularized_from_cached_if_available(constitutive_matrix_builder, r) if use_full_operator else None
            if cached_regularized is not None:
                K_r = cached_regularized
            else:
                K_tangent = _ensure_tangent_matrix_for_regularization(
                    constitutive_matrix_builder,
                    U_it,
                    K_tangent=K_tangent,
                    use_free_build=use_free_build,
                )
                K_r = _combine_matrices(r, K_elast, 1.0 - r, K_tangent)
        preconditioning_matrix = None
        if use_full_operator and _requires_explicit_preconditioning_matrix(linear_system_solver):
            if _needs_preconditioning_matrix_refresh(linear_system_solver):
                preconditioning_matrix = _explicit_preconditioning_matrix(
                    constitutive_matrix_builder,
                    linear_system_solver,
                    regularization_r=r,
                    K_elast=K_elast,
                )
        K_free = None
        try:
            if use_full_operator:
                _setup_linear_system(
                    linear_system_solver,
                    K_r,
                    A_full=K_r,
                    free_idx=free_idx,
                    preconditioning_matrix=preconditioning_matrix,
                )
                if getattr(linear_system_solver, "supports_a_orthogonalization", lambda: True)():
                    linear_system_solver.A_orthogonalize(K_r)
                if use_local_build:
                    dU_free = _solve_linear_system_local(
                        linear_system_solver,
                        K_r,
                        rhs,
                        b_full=rhs_local,
                        local_rhs=rhs_local,
                        free_idx=free_idx,
                    )
                else:
                    dU_free = _solve_linear_system(linear_system_solver, K_r, rhs, free_idx=free_idx)
            else:
                K_free = extract_submatrix_free(K_r, free_idx)
                _setup_linear_system(linear_system_solver, K_free, A_full=K_r, free_idx=free_idx)
                if getattr(linear_system_solver, "supports_a_orthogonalization", lambda: True)():
                    linear_system_solver.A_orthogonalize(K_free)
                dU_free = _solve_linear_system(linear_system_solver, K_free, rhs, free_idx=free_idx)
        finally:
            _release_iteration_resources(linear_system_solver)
            _destroy_petsc_mat(K_free)
            if not _is_builder_cached_matrix(K_tangent, constitutive_matrix_builder):
                _destroy_petsc_mat(K_tangent)
            if not use_full_operator and not _is_builder_cached_matrix(K_r, constitutive_matrix_builder):
                _destroy_petsc_mat(K_r)
        iter_delta = _collector_delta(snap_before_iter, _collector_snapshot(linear_system_solver))

        dU = np.zeros(U_it.size, dtype=np.float64)
        dU[free_idx] = np.asarray(dU_free, dtype=np.float64)
        dU = dU.reshape(shape, order="F")

        alpha = damping(
            it_damp_max,
            U_it,
            dU,
            F,
            f,
            constitutive_matrix_builder,
            Q,
            F_free=F_free,
            f_free=f_free,
            F_local_free=F_free_local,
            f_local_free=f_free_local,
            dU_local_free=_local_owned_free_rows_from_field(dU, constitutive_matrix_builder.owned_tangent_pattern) if use_local_build else None,
            comm=comm,
        )

        _emit_progress(
            progress_callback,
            event="newton_iteration",
            solver="newton",
            iteration=int(it),
            criterion=float(criterion_abs),
            rel_residual=float(criterion),
            alpha=float(alpha),
            r=float(r),
            linear_iterations=int(iter_delta["iterations"]),
            linear_solve_time=float(iter_delta["solve_time"]),
            linear_preconditioner_time=float(iter_delta["preconditioner_time"]),
            linear_orthogonalization_time=float(iter_delta["orthogonalization_time"]),
            iteration_wall_time=float(perf_counter() - iter_t0),
            tolerance=float(tol),
            status="iterate",
        )

        compute_diffs = True
        if alpha < 1e-1:
            if alpha == 0.0:
                compute_diffs = False
                r *= 2.0
            else:
                r *= 2.0 ** 0.25
        else:
            if getattr(linear_system_solver, "supports_dynamic_deflation_basis", lambda: True)():
                linear_system_solver.expand_deflation_basis(_to_free_vector(dU, Q))
            if alpha > 0.5:
                r = max(r / np.sqrt(2.0), r_min)

        if alpha == 0.0 and r > 1.0:
            flag_N = 1
            break

        U_it = U_it + alpha * dU
        if np.isnan(criterion) or (it == it_newt_max):
            flag_N = 1
            break

    return U_it, flag_N, it


def newton_ind_ssr(
    U_ini: np.ndarray,
    omega: float,
    lambda_it: float,
    it_newt_max: int,
    it_damp_max: int,
    tol: float,
    r_min: float,
    K_elast,
    Q: np.ndarray,
    f: np.ndarray,
    constitutive_matrix_builder,
    linear_system_solver,
    *,
    progress_callback: Callable[[dict], None] | None = None,
):
    """Nested Newton for ``F_lambda(U)=f`` with additional condition ``f^T U = omega``.

    Returns ``(U_it, lambda_it, flag_N, it, history)``.
    """

    U_it = _to_float_matrix(U_ini)
    shape = U_it.shape
    Q = np.asarray(Q, dtype=bool)

    free_idx = q_to_free_indices(Q)
    if free_idx.size == 0:
        history = {
            "criterion": np.array([0.0], dtype=np.float64),
            "residual": np.array([0.0], dtype=np.float64),
            "r": np.array([r_min], dtype=np.float64),
            "alpha": np.array([1.0], dtype=np.float64),
            "lambda": np.array([float(lambda_it)], dtype=np.float64),
            "delta_lambda": np.array([0.0], dtype=np.float64),
            "accepted_delta_lambda": np.array([0.0], dtype=np.float64),
            "linear_iterations": np.array([0], dtype=np.int64),
            "linear_solve_time": np.array([0.0], dtype=np.float64),
            "linear_preconditioner_time": np.array([0.0], dtype=np.float64),
            "linear_orthogonalization_time": np.array([0.0], dtype=np.float64),
            "iteration_wall_time": np.array([0.0], dtype=np.float64),
        }
        return U_it, float(lambda_it), 0, 0, history
    f_free = _to_free_vector(f, Q)

    norm_f = _free_norm(f, Q)
    if norm_f == 0.0:
        norm_f = 1.0

    eps = tol / 1000.0
    it = 0
    flag_N = 0
    r = float(r_min)
    compute_diffs = True
    rel_resid = np.nan

    criterion_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    residual_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    r_hist = np.zeros(int(it_newt_max), dtype=np.float64)
    alpha_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    lambda_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    delta_lambda_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    accepted_delta_lambda_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    linear_iterations_hist = np.zeros(int(it_newt_max), dtype=np.int64)
    linear_solve_hist = np.zeros(int(it_newt_max), dtype=np.float64)
    linear_preconditioner_hist = np.zeros(int(it_newt_max), dtype=np.float64)
    linear_orthogonalization_hist = np.zeros(int(it_newt_max), dtype=np.float64)
    iteration_wall_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)

    while True:
        it += 1
        iter_t0 = perf_counter()
        snap_before_iter = _collector_snapshot(linear_system_solver)

        use_full_operator = _prefers_full_system_operator(linear_system_solver, K_elast)
        use_free_build = _supports_free_builder(constitutive_matrix_builder, "build_F_all_free")
        use_local_build = use_full_operator and _supports_local_builder(constitutive_matrix_builder, "build_F_all_local")
        comm = _local_comm_from_operator(K_elast) if use_local_build else None
        K_tangent = None
        K_r = None
        F = None
        F_local = None
        F_free_local = None
        f_free_local = None

        if compute_diffs:
            if use_local_build:
                constitutive_matrix_builder.reduction(lambda_it)
                constitutive_matrix_builder.constitutive_problem_stress_tangent(U_it)
                F_local = np.asarray(constitutive_matrix_builder.build_F_local(), dtype=np.float64).reshape(-1)
                F_free_local = np.asarray(constitutive_matrix_builder.build_F_free_local(), dtype=np.float64).reshape(-1)
                F_free = F_free_local
                K_r = constitutive_matrix_builder.build_K_regularized(r)
            elif use_full_operator and _supports_free_builder(constitutive_matrix_builder, "build_F_K_regularized_all_free"):
                F_free, K_r = constitutive_matrix_builder.build_F_K_regularized_all_free(lambda_it, U_it, r)
                F_free = np.asarray(F_free, dtype=np.float64).reshape(-1)
            elif use_full_operator:
                regularized_pair = _build_regularized_if_available(constitutive_matrix_builder, lam=lambda_it, U=U_it, r=r)
                if regularized_pair is not None:
                    F, K_r = regularized_pair
                    F_free = _to_free_vector(F, Q)
                else:
                    F, K_tangent = constitutive_matrix_builder.build_F_K_tangent_all(lambda_it, U_it)
                    F_free = _to_free_vector(F, Q)
            elif _supports_free_builder(constitutive_matrix_builder, "build_F_K_tangent_all_free"):
                F_free, K_tangent = constitutive_matrix_builder.build_F_K_tangent_all_free(lambda_it, U_it)
                F_free = np.asarray(F_free, dtype=np.float64).reshape(-1)
            else:
                F, K_tangent = constitutive_matrix_builder.build_F_K_tangent_all(lambda_it, U_it)
                F_free = _to_free_vector(F, Q)
        else:
            if use_local_build:
                F_local = np.asarray(constitutive_matrix_builder.build_F_all_local(lambda_it, U_it), dtype=np.float64).reshape(-1)
                F_free_local = np.asarray(constitutive_matrix_builder.build_F_all_free_local(lambda_it, U_it), dtype=np.float64).reshape(-1)
                F_free = F_free_local
            elif use_free_build:
                F_free = np.asarray(constitutive_matrix_builder.build_F_all_free(lambda_it, U_it), dtype=np.float64).reshape(-1)
            else:
                F = constitutive_matrix_builder.build_F_all(lambda_it, U_it)
                F_free = _to_free_vector(F, Q)

        if use_local_build:
            f_free_local = _local_owned_free_rows_from_field(f, constitutive_matrix_builder.owned_tangent_pattern)
            criterion = _dist_norm_local(F_free_local - f_free_local, comm)
        else:
            criterion = float(np.linalg.norm(F_free - f_free))
        rel_resid = criterion / norm_f
        criterion_hist[it - 1] = criterion
        residual_hist[it - 1] = rel_resid
        lambda_hist[it - 1] = float(lambda_it)
        if compute_diffs and rel_resid < tol and it > 1:
            iteration_wall_hist[it - 1] = float(perf_counter() - iter_t0)
            _emit_progress(
                progress_callback,
                event="newton_iteration",
                solver="newton_ind_ssr",
                iteration=int(it),
                criterion=float(criterion),
                rel_residual=float(rel_resid),
                alpha=None,
                r=float(r),
                lambda_value=float(lambda_it),
                delta_lambda=None,
                accepted_delta_lambda=None,
                omega_value=float(omega),
                linear_iterations=0,
                linear_solve_time=0.0,
                linear_preconditioner_time=0.0,
                linear_orthogonalization_time=0.0,
                iteration_wall_time=float(iteration_wall_hist[it - 1]),
                tolerance=float(tol),
                status="converged",
            )
            _cleanup_pre_solve_iteration_mats(
                K_tangent=K_tangent,
                K_r=K_r,
                use_full_operator=use_full_operator,
                constitutive_matrix_builder=constitutive_matrix_builder,
            )
            break

        r_hist[it - 1] = r
        if K_r is None:
            cached_regularized = _build_regularized_from_cached_if_available(constitutive_matrix_builder, r) if use_full_operator else None
            if cached_regularized is not None:
                K_r = cached_regularized
            else:
                K_tangent = _ensure_tangent_matrix_for_regularization(
                    constitutive_matrix_builder,
                    U_it,
                    K_tangent=K_tangent,
                    use_free_build=use_free_build,
                )
                K_r = _combine_matrices(r, K_elast, 1.0 - r, K_tangent)
        preconditioning_matrix = None
        if use_full_operator and _requires_explicit_preconditioning_matrix(linear_system_solver):
            if _needs_preconditioning_matrix_refresh(linear_system_solver):
                preconditioning_matrix = _explicit_preconditioning_matrix(
                    constitutive_matrix_builder,
                    linear_system_solver,
                    regularization_r=r,
                    K_elast=K_elast,
                )
        K_free = None

        # G = dF/dlambda approximated numerically.
        if use_local_build:
            F_eps_local = np.asarray(constitutive_matrix_builder.build_F_all_local(lambda_it + eps, U_it), dtype=np.float64).reshape(-1)
            F_eps_free_local = np.asarray(constitutive_matrix_builder.build_F_all_free_local(lambda_it + eps, U_it), dtype=np.float64).reshape(-1)
            G_local = (F_eps_local - F_local) / eps
            G_free_local = (F_eps_free_local - F_free_local) / eps
            G_free = G_free_local
            G = None
        elif use_free_build:
            F_eps_free = np.asarray(constitutive_matrix_builder.build_F_all_free(lambda_it + eps, U_it), dtype=np.float64).reshape(-1)
            G_free = (F_eps_free - F_free) / eps
            G = None
        else:
            F_eps = constitutive_matrix_builder.build_F_all(lambda_it + eps, U_it)
            G = (F_eps - F) / eps
            G_free = _to_free_vector(G, Q)

        try:
            if use_full_operator:
                _setup_linear_system(
                    linear_system_solver,
                    K_r,
                    A_full=K_r,
                    free_idx=free_idx,
                    preconditioning_matrix=preconditioning_matrix,
                )
                if getattr(linear_system_solver, "supports_a_orthogonalization", lambda: True)():
                    linear_system_solver.A_orthogonalize(K_r)
                if use_local_build:
                    f_local = _local_owned_rows_from_field(f, constitutive_matrix_builder.owned_tangent_pattern)
                    f_free_local = _local_owned_free_rows_from_field(f, constitutive_matrix_builder.owned_tangent_pattern)
                    rhs_v_local = f_local - F_local
                    rhs_v = f_free_local - F_free_local
                    dW_free = _solve_linear_system_local(
                        linear_system_solver,
                        K_r,
                        -G_free_local,
                        b_full=-G_local,
                        local_rhs=-G_local,
                        free_idx=free_idx,
                    )
                    dV_free = _solve_linear_system_local(
                        linear_system_solver,
                        K_r,
                        rhs_v,
                        b_full=rhs_v_local,
                        local_rhs=rhs_v_local,
                        free_idx=free_idx,
                    )
                else:
                    dW_free = _solve_linear_system(linear_system_solver, K_r, -G_free, free_idx=free_idx)
                    dV_free = _solve_linear_system(
                        linear_system_solver,
                        K_r,
                        f_free - F_free,
                        free_idx=free_idx,
                    )
            else:
                K_free = extract_submatrix_free(K_r, free_idx)
                _setup_linear_system(linear_system_solver, K_free, A_full=K_r, free_idx=free_idx)
                if getattr(linear_system_solver, "supports_a_orthogonalization", lambda: True)():
                    linear_system_solver.A_orthogonalize(K_free)
                dW_free = _solve_linear_system(linear_system_solver, K_free, -G_free, free_idx=free_idx)
                dV_free = _solve_linear_system(
                    linear_system_solver,
                    K_free,
                    f_free - F_free,
                    free_idx=free_idx,
                )
        finally:
            _release_iteration_resources(linear_system_solver)
            _destroy_petsc_mat(K_free)
            if not _is_builder_cached_matrix(K_tangent, constitutive_matrix_builder):
                _destroy_petsc_mat(K_tangent)
            if not use_full_operator and not _is_builder_cached_matrix(K_r, constitutive_matrix_builder):
                _destroy_petsc_mat(K_r)
        iter_delta = _collector_delta(snap_before_iter, _collector_snapshot(linear_system_solver))

        W = np.zeros(U_it.size, dtype=np.float64)
        V = np.zeros(U_it.size, dtype=np.float64)
        W[free_idx] = np.asarray(dW_free, dtype=np.float64)
        V[free_idx] = np.asarray(dV_free, dtype=np.float64)
        W = W.reshape(shape, order="F")
        V = V.reshape(shape, order="F")

        fQ = _to_free_vector(f, Q)
        WQ = _to_free_vector(W, Q)
        VQ = _to_free_vector(V, Q)
        denom = float(np.dot(fQ, WQ))
        d_l = 0.0 if abs(denom) < 1e-30 else -float(np.dot(fQ, VQ)) / denom

        d_U = V + d_l * W
        alpha = damping_alg5(
            it_damp_max,
            U_it,
            lambda_it,
            d_U,
            d_l,
            f,
            criterion,
            Q,
            constitutive_matrix_builder,
            f_free=f_free,
            f_local_free=f_free_local if use_local_build else None,
            comm=comm,
        )
        alpha_hist[it - 1] = alpha
        delta_lambda_hist[it - 1] = float(d_l)
        accepted_delta_lambda_hist[it - 1] = float(alpha * d_l)
        linear_iterations_hist[it - 1] = int(iter_delta["iterations"])
        linear_solve_hist[it - 1] = float(iter_delta["solve_time"])
        linear_preconditioner_hist[it - 1] = float(iter_delta["preconditioner_time"])
        linear_orthogonalization_hist[it - 1] = float(iter_delta["orthogonalization_time"])
        iteration_wall_hist[it - 1] = float(perf_counter() - iter_t0)

        _emit_progress(
            progress_callback,
            event="newton_iteration",
            solver="newton_ind_ssr",
            iteration=int(it),
            criterion=float(criterion),
            rel_residual=float(rel_resid),
            alpha=float(alpha),
            r=float(r),
            lambda_value=float(lambda_it),
            delta_lambda=float(d_l),
            accepted_delta_lambda=float(alpha * d_l),
            omega_value=float(omega),
            linear_iterations=int(iter_delta["iterations"]),
            linear_solve_time=float(iter_delta["solve_time"]),
            linear_preconditioner_time=float(iter_delta["preconditioner_time"]),
            linear_orthogonalization_time=float(iter_delta["orthogonalization_time"]),
            iteration_wall_time=float(iteration_wall_hist[it - 1]),
            tolerance=float(tol),
            status="iterate",
        )

        compute_diffs = True
        if alpha < 1e-1:
            if alpha == 0.0:
                compute_diffs = False
                r *= 2.0
            else:
                r *= 2.0 ** 0.25
        else:
            if getattr(linear_system_solver, "supports_dynamic_deflation_basis", lambda: True)():
                linear_system_solver.expand_deflation_basis(_to_free_vector(W, Q))
                linear_system_solver.expand_deflation_basis(_to_free_vector(V, Q))
            if alpha > 0.5:
                r = max(r / np.sqrt(2.0), r_min)

        if alpha == 0.0 and r > 1.0:
            if rel_resid > 10.0 * tol:
                flag_N = 1
            break

        U_it = U_it + alpha * d_U
        denom = _free_dot(f, U_it, Q)
        if denom != 0.0:
            U_it = U_it * (omega / denom)

        lambda_it = lambda_it + alpha * d_l

        if np.isnan(rel_resid) or it == it_newt_max:
            if rel_resid > 10.0 * tol:
                flag_N = 1
            break

    history = {
        "criterion": criterion_hist[:it],
        "residual": residual_hist[:it],
        "r": r_hist[:it],
        "alpha": alpha_hist[:it],
        "lambda": lambda_hist[:it],
        "delta_lambda": delta_lambda_hist[:it],
        "accepted_delta_lambda": accepted_delta_lambda_hist[:it],
        "linear_iterations": linear_iterations_hist[:it],
        "linear_solve_time": linear_solve_hist[:it],
        "linear_preconditioner_time": linear_preconditioner_hist[:it],
        "linear_orthogonalization_time": linear_orthogonalization_hist[:it],
        "iteration_wall_time": iteration_wall_hist[:it],
    }
    return U_it, float(lambda_it), flag_N, it, history


def newton_ind_ll(
    U_ini: np.ndarray,
    t_ini: float,
    omega: float,
    it_newt_max: int,
    it_damp_max: int,
    tol: float,
    r_min: float,
    K_elast,
    Q: np.ndarray,
    f: np.ndarray,
    constitutive_matrix_builder,
    linear_system_solver,
    *,
    progress_callback: Callable[[dict], None] | None = None,
):
    """Nested Newton for indirect limit-load continuation.

    Returns ``(U_it, t_it, flag_N, it, history)``.
    """

    U_it = _to_float_matrix(U_ini)
    shape = U_it.shape
    Q = np.asarray(Q, dtype=bool)

    free_idx = q_to_free_indices(Q)
    if free_idx.size == 0:
        history = {
            "criterion": np.array([0.0], dtype=np.float64),
            "residual": np.array([0.0], dtype=np.float64),
            "r": np.array([r_min], dtype=np.float64),
            "alpha": np.array([1.0], dtype=np.float64),
            "lambda": np.array([float(t_ini)], dtype=np.float64),
            "delta_lambda": np.array([0.0], dtype=np.float64),
            "accepted_delta_lambda": np.array([0.0], dtype=np.float64),
            "linear_iterations": np.array([0], dtype=np.int64),
            "linear_solve_time": np.array([0.0], dtype=np.float64),
            "linear_preconditioner_time": np.array([0.0], dtype=np.float64),
            "linear_orthogonalization_time": np.array([0.0], dtype=np.float64),
            "iteration_wall_time": np.array([0.0], dtype=np.float64),
        }
        return U_it, float(t_ini), 0, 0, history

    norm_f = _free_norm(f, Q)
    if norm_f == 0.0:
        norm_f = 1.0

    t_it = float(t_ini)
    it = 0
    flag_N = 0
    r = float(r_min)
    compute_diffs = True
    rel_resid = np.nan

    criterion_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    residual_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    r_hist = np.zeros(int(it_newt_max), dtype=np.float64)
    alpha_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    lambda_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    delta_lambda_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    accepted_delta_lambda_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    linear_iterations_hist = np.zeros(int(it_newt_max), dtype=np.int64)
    linear_solve_hist = np.zeros(int(it_newt_max), dtype=np.float64)
    linear_preconditioner_hist = np.zeros(int(it_newt_max), dtype=np.float64)
    linear_orthogonalization_hist = np.zeros(int(it_newt_max), dtype=np.float64)
    iteration_wall_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)

    while True:
        it += 1
        iter_t0 = perf_counter()
        snap_before_iter = _collector_snapshot(linear_system_solver)

        use_full_operator = _prefers_full_system_operator(linear_system_solver, K_elast)
        use_free_build = _supports_free_builder(constitutive_matrix_builder, "build_F_reduced_free")
        use_local_build = use_full_operator and _supports_local_builder(constitutive_matrix_builder, "build_F_reduced_local")
        comm = _local_comm_from_operator(K_elast) if use_local_build else None
        K_tangent = None
        K_r = None
        F_int = None
        F_int_local = None
        F_int_free_local = None
        f_free_local = None

        if compute_diffs:
            if use_local_build:
                constitutive_matrix_builder.constitutive_problem_stress_tangent(U_it)
                F_int_local = np.asarray(constitutive_matrix_builder.build_F_local(), dtype=np.float64).reshape(-1)
                F_int_free_local = np.asarray(constitutive_matrix_builder.build_F_free_local(), dtype=np.float64).reshape(-1)
                F_int_free = F_int_free_local
                K_r = constitutive_matrix_builder.build_K_regularized(r)
            elif use_full_operator and _supports_free_builder(constitutive_matrix_builder, "build_F_K_regularized_reduced_free"):
                F_int_free, K_r = constitutive_matrix_builder.build_F_K_regularized_reduced_free(U_it, r)
                F_int_free = np.asarray(F_int_free, dtype=np.float64).reshape(-1)
            elif use_full_operator:
                regularized_pair = _build_regularized_if_available(constitutive_matrix_builder, U=U_it, r=r)
                if regularized_pair is not None:
                    F_int, K_r = regularized_pair
                    F_int_free = _to_free_vector(F_int, Q)
                else:
                    F_int, K_tangent = constitutive_matrix_builder.build_F_K_tangent_reduced(U_it)
                    F_int_free = _to_free_vector(F_int, Q)
            elif _supports_free_builder(constitutive_matrix_builder, "build_F_K_tangent_reduced_free"):
                F_int_free, K_tangent = constitutive_matrix_builder.build_F_K_tangent_reduced_free(U_it)
                F_int_free = np.asarray(F_int_free, dtype=np.float64).reshape(-1)
            else:
                F_int, K_tangent = constitutive_matrix_builder.build_F_K_tangent_reduced(U_it)
                F_int_free = _to_free_vector(F_int, Q)
        else:
            if use_local_build:
                F_int_local = np.asarray(constitutive_matrix_builder.build_F_reduced_local(U_it), dtype=np.float64).reshape(-1)
                F_int_free_local = np.asarray(constitutive_matrix_builder.build_F_reduced_free_local(U_it), dtype=np.float64).reshape(-1)
                F_int_free = F_int_free_local
            elif use_free_build:
                F_int_free = np.asarray(constitutive_matrix_builder.build_F_reduced_free(U_it), dtype=np.float64).reshape(-1)
            else:
                F_int = constitutive_matrix_builder.build_F_reduced(U_it)
                F_int_free = _to_free_vector(F_int, Q)

        if use_local_build:
            f_free_local = _local_owned_free_rows_from_field(f, constitutive_matrix_builder.owned_tangent_pattern)
            criterion = _dist_norm_local(t_it * f_free_local - F_int_free_local, comm)
        else:
            criterion = float(np.linalg.norm(t_it * _to_free_vector(f, Q) - F_int_free))
        rel_resid = criterion / norm_f
        criterion_hist[it - 1] = criterion
        residual_hist[it - 1] = rel_resid
        lambda_hist[it - 1] = float(t_it)
        if compute_diffs and rel_resid < tol and it > 1:
            iteration_wall_hist[it - 1] = float(perf_counter() - iter_t0)
            _emit_progress(
                progress_callback,
                event="newton_iteration",
                solver="newton_ind_ll",
                iteration=int(it),
                criterion=float(criterion),
                rel_residual=float(rel_resid),
                alpha=None,
                r=float(r),
                lambda_value=float(t_it),
                delta_lambda=None,
                accepted_delta_lambda=None,
                omega_value=float(omega),
                linear_iterations=0,
                linear_solve_time=0.0,
                linear_preconditioner_time=0.0,
                linear_orthogonalization_time=0.0,
                iteration_wall_time=float(iteration_wall_hist[it - 1]),
                tolerance=float(tol),
                status="converged",
            )
            _cleanup_pre_solve_iteration_mats(
                K_tangent=K_tangent,
                K_r=K_r,
                use_full_operator=use_full_operator,
                constitutive_matrix_builder=constitutive_matrix_builder,
            )
            break

        r_hist[it - 1] = r
        if K_r is None:
            cached_regularized = _build_regularized_from_cached_if_available(constitutive_matrix_builder, r) if use_full_operator else None
            if cached_regularized is not None:
                K_r = cached_regularized
            else:
                K_tangent = _ensure_tangent_matrix_for_regularization(
                    constitutive_matrix_builder,
                    U_it,
                    K_tangent=K_tangent,
                    use_free_build=use_free_build,
                )
                K_r = _combine_matrices(r, K_elast, 1.0 - r, K_tangent)
        preconditioning_matrix = None
        if use_full_operator and _requires_explicit_preconditioning_matrix(linear_system_solver):
            if _needs_preconditioning_matrix_refresh(linear_system_solver):
                preconditioning_matrix = _explicit_preconditioning_matrix(
                    constitutive_matrix_builder,
                    linear_system_solver,
                    regularization_r=r,
                    K_elast=K_elast,
                )
        K_free = None

        try:
            if use_full_operator:
                _setup_linear_system(
                    linear_system_solver,
                    K_r,
                    A_full=K_r,
                    free_idx=free_idx,
                    preconditioning_matrix=preconditioning_matrix,
                )
                if getattr(linear_system_solver, "supports_a_orthogonalization", lambda: True)():
                    linear_system_solver.A_orthogonalize(K_r)
                if use_local_build:
                    f_local = _local_owned_rows_from_field(f, constitutive_matrix_builder.owned_tangent_pattern)
                    f_free_local = _local_owned_free_rows_from_field(f, constitutive_matrix_builder.owned_tangent_pattern)
                    rhs_v_local = t_it * f_local - F_int_local
                    rhs_v = t_it * f_free_local - F_int_free_local
                    dW_free = _solve_linear_system_local(
                        linear_system_solver,
                        K_r,
                        f_free_local,
                        b_full=f_local,
                        local_rhs=f_local,
                        free_idx=free_idx,
                    )
                    dV_free = _solve_linear_system_local(
                        linear_system_solver,
                        K_r,
                        rhs_v,
                        b_full=rhs_v_local,
                        local_rhs=rhs_v_local,
                        free_idx=free_idx,
                    )
                else:
                    dW_free = _solve_linear_system(
                        linear_system_solver,
                        K_r,
                        _to_free_vector(f, Q),
                        free_idx=free_idx,
                    )
                    dV_free = _solve_linear_system(
                        linear_system_solver,
                        K_r,
                        t_it * _to_free_vector(f, Q) - F_int_free,
                        free_idx=free_idx,
                    )
            else:
                K_free = extract_submatrix_free(K_r, free_idx)
                _setup_linear_system(linear_system_solver, K_free, A_full=K_r, free_idx=free_idx)
                if getattr(linear_system_solver, "supports_a_orthogonalization", lambda: True)():
                    linear_system_solver.A_orthogonalize(K_free)
                dW_free = _solve_linear_system(
                    linear_system_solver,
                    K_free,
                    _to_free_vector(f, Q),
                    free_idx=free_idx,
                )
                dV_free = _solve_linear_system(
                    linear_system_solver,
                    K_free,
                    t_it * _to_free_vector(f, Q) - F_int_free,
                    free_idx=free_idx,
                )
        finally:
            _release_iteration_resources(linear_system_solver)
            _destroy_petsc_mat(K_free)
            if not _is_builder_cached_matrix(K_tangent, constitutive_matrix_builder):
                _destroy_petsc_mat(K_tangent)
            if not use_full_operator and not _is_builder_cached_matrix(K_r, constitutive_matrix_builder):
                _destroy_petsc_mat(K_r)
        iter_delta = _collector_delta(snap_before_iter, _collector_snapshot(linear_system_solver))

        W = np.zeros(U_it.size, dtype=np.float64)
        V = np.zeros(U_it.size, dtype=np.float64)
        W[free_idx] = np.asarray(dW_free, dtype=np.float64)
        V[free_idx] = np.asarray(dV_free, dtype=np.float64)
        W = W.reshape(shape, order="F")
        V = V.reshape(shape, order="F")

        fQ = _to_free_vector(f, Q)
        WQ = _to_free_vector(W, Q)
        VQ = _to_free_vector(V, Q)
        denom = float(np.dot(fQ, WQ))
        d_t = 0.0 if abs(denom) < 1e-30 else -float(np.dot(fQ, VQ)) / denom

        d_U = V + d_t * W
        alpha = damping(
            it_damp_max,
            U_it,
            d_U,
            F_int,
            np.zeros_like(f),
            constitutive_matrix_builder,
            Q,
            F_free=F_int_free,
            f_free=np.zeros_like(F_int_free),
            F_local_free=F_int_free_local,
            f_local_free=np.zeros_like(F_int_free_local) if use_local_build else None,
            dU_local_free=_local_owned_free_rows_from_field(d_U, constitutive_matrix_builder.owned_tangent_pattern) if use_local_build else None,
            comm=comm,
        )
        alpha_hist[it - 1] = alpha
        delta_lambda_hist[it - 1] = float(d_t)
        accepted_delta_lambda_hist[it - 1] = float(alpha * d_t)
        linear_iterations_hist[it - 1] = int(iter_delta["iterations"])
        linear_solve_hist[it - 1] = float(iter_delta["solve_time"])
        linear_preconditioner_hist[it - 1] = float(iter_delta["preconditioner_time"])
        linear_orthogonalization_hist[it - 1] = float(iter_delta["orthogonalization_time"])
        iteration_wall_hist[it - 1] = float(perf_counter() - iter_t0)

        _emit_progress(
            progress_callback,
            event="newton_iteration",
            solver="newton_ind_ll",
            iteration=int(it),
            criterion=float(criterion),
            rel_residual=float(rel_resid),
            alpha=float(alpha),
            r=float(r),
            lambda_value=float(t_it),
            delta_lambda=float(d_t),
            accepted_delta_lambda=float(alpha * d_t),
            omega_value=float(omega),
            linear_iterations=int(iter_delta["iterations"]),
            linear_solve_time=float(iter_delta["solve_time"]),
            linear_preconditioner_time=float(iter_delta["preconditioner_time"]),
            linear_orthogonalization_time=float(iter_delta["orthogonalization_time"]),
            iteration_wall_time=float(iteration_wall_hist[it - 1]),
            tolerance=float(tol),
            status="iterate",
        )

        compute_diffs = True
        if alpha < 1e-1:
            if alpha == 0.0:
                compute_diffs = False
                r *= 2.0
            else:
                r *= 2.0 ** 0.25
        else:
            if getattr(linear_system_solver, "supports_dynamic_deflation_basis", lambda: True)():
                linear_system_solver.expand_deflation_basis(_to_free_vector(W, Q))
                linear_system_solver.expand_deflation_basis(_to_free_vector(V, Q))
            if alpha > 0.5:
                r = max(r / np.sqrt(2.0), r_min)

        if alpha == 0.0 and r > 1.0:
            flag_N = 1
            break

        U_it = U_it + alpha * d_U
        denom = _free_dot(f, U_it, Q)
        if denom != 0.0:
            U_it = U_it * (omega / denom)

        t_it = t_it + d_t

        if np.isnan(rel_resid) or it == it_newt_max:
            flag_N = 1
            break

    history = {
        "criterion": criterion_hist[:it],
        "residual": residual_hist[:it],
        "r": r_hist[:it],
        "alpha": alpha_hist[:it],
        "lambda": lambda_hist[:it],
        "delta_lambda": delta_lambda_hist[:it],
        "accepted_delta_lambda": accepted_delta_lambda_hist[:it],
        "linear_iterations": linear_iterations_hist[:it],
        "linear_solve_time": linear_solve_hist[:it],
        "linear_preconditioner_time": linear_preconditioner_hist[:it],
        "linear_orthogonalization_time": linear_orthogonalization_hist[:it],
        "iteration_wall_time": iteration_wall_hist[:it],
    }
    return U_it, t_it, flag_N, it, history
