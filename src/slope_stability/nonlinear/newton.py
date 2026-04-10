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


def _basis_snapshot(linear_system_solver):
    getter = getattr(linear_system_solver, "get_deflation_basis_snapshot", None)
    if callable(getter):
        return getter()
    basis = getattr(linear_system_solver, "deflation_basis", None)
    if basis is None:
        return None
    return np.array(basis, dtype=np.float64, copy=True)


def _basis_restore(linear_system_solver, snapshot) -> None:
    restore = getattr(linear_system_solver, "restore_deflation_basis", None)
    if callable(restore):
        restore(snapshot)
        return
    if hasattr(linear_system_solver, "deflation_basis"):
        linear_system_solver.deflation_basis = np.array(snapshot, dtype=np.float64, copy=True)


def _basis_cols(linear_system_solver) -> int:
    basis = getattr(linear_system_solver, "deflation_basis", None)
    if basis is None:
        return 0
    arr = np.asarray(basis)
    if arr.size == 0:
        return 0
    if arr.ndim == 1:
        return 1
    return int(arr.shape[1])


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
    if (
        K_r is not None
        and K_r is not K_tangent
        and not _is_builder_cached_matrix(K_r, constitutive_matrix_builder)
    ):
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


def _normalize_stopping_criterion(stopping_criterion: str | None) -> str:
    raw = "relative_residual" if stopping_criterion is None else str(stopping_criterion).strip().lower()
    aliases = {
        "residual": "relative_residual",
        "rel_residual": "relative_residual",
        "relative_residual": "relative_residual",
        "correction": "relative_correction",
        "rel_correction": "relative_correction",
        "relative_correction": "relative_correction",
        "relative_newton_correction": "relative_correction",
        "delta_lambda": "absolute_delta_lambda",
        "abs_delta_lambda": "absolute_delta_lambda",
        "absolute_delta_lambda": "absolute_delta_lambda",
    }
    mode = aliases.get(raw)
    if mode is None:
        raise ValueError(f"Unsupported Newton stopping_criterion {stopping_criterion!r}")
    return mode


def _resolve_stopping_tolerance(tol: float, stopping_tol: float | None) -> float:
    value = float(tol if stopping_tol is None else stopping_tol)
    if value < 0.0:
        raise ValueError("Newton stopping tolerance must be non-negative.")
    return value


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
        try:
            return fn(r)
        except ValueError as exc:
            message = str(exc)
            if "Tangent DS not computed" in message or "DS must have shape" in message:
                return None
            raise
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


def _normalize_nonlinear_policy(nonlinear_policy: str | None) -> str:
    raw = "residual" if nonlinear_policy is None else str(nonlinear_policy).strip().lower()
    aliases = {
        "residual": "residual",
        "directional_residual": "residual",
        "energy": "energy_armijo",
        "armijo": "energy_armijo",
        "energy_armijo": "energy_armijo",
    }
    mode = aliases.get(raw)
    if mode is None:
        raise ValueError(f"Unsupported nonlinear policy {nonlinear_policy!r}")
    return mode


def _resolve_combined_stopping(combined_stopping: dict[str, float] | None) -> dict[str, float]:
    if not combined_stopping:
        return {}
    aliases = {
        "residual": "relative_residual",
        "rel_residual": "relative_residual",
        "relative_residual": "relative_residual",
        "correction": "relative_correction",
        "rel_correction": "relative_correction",
        "relative_correction": "relative_correction",
        "relative_newton_correction": "relative_correction",
        "energy": "energy_change",
        "energy_change": "energy_change",
        "denergy": "energy_change",
    }
    resolved: dict[str, float] = {}
    for raw_key, raw_value in dict(combined_stopping).items():
        key = aliases.get(str(raw_key).strip().lower())
        if key is None:
            raise ValueError(f"Unsupported combined stopping metric {raw_key!r}")
        value = float(raw_value)
        if value < 0.0:
            raise ValueError("Combined stopping tolerances must be non-negative.")
        resolved[key] = value
    return resolved


def _combined_stop_requires_post_step(combined_stopping: dict[str, float]) -> bool:
    return bool({"relative_correction", "energy_change"} & set(combined_stopping.keys()))


def _combined_stop_satisfied(
    combined_stopping: dict[str, float],
    *,
    relative_residual: float | None,
    relative_correction: float | None,
    energy_change: float | None,
) -> bool:
    if not combined_stopping:
        return True
    checks = {
        "relative_residual": relative_residual,
        "relative_correction": relative_correction,
        "energy_change": energy_change,
    }
    for key, tol in combined_stopping.items():
        value = checks.get(key)
        if value is None or not np.isfinite(value) or float(value) >= float(tol):
            return False
    return True


def _newton_history_template(
    *,
    stop_mode: str,
    stop_tol: float,
    tol: float,
    nonlinear_policy: str,
    combined_stopping: dict[str, float],
) -> dict[str, object]:
    return {
        "criterion": [],
        "residual": [],
        "r": [],
        "alpha": [],
        "accepted_correction_norm": [],
        "iterate_free_norm": [],
        "accepted_relative_correction_norm": [],
        "linear_iterations": [],
        "linear_solve_time": [],
        "linear_preconditioner_time": [],
        "linear_orthogonalization_time": [],
        "iteration_wall_time": [],
        "line_search_iterations": [],
        "deflation_basis_dim_solve": [],
        "deflation_basis_dim_end": [],
        "linear_true_residual_final": [],
        "linear_hit_max_iterations": [],
        "linear_converged_reason": [],
        "descent_direction": [],
        "guarded_max_it_accept": [],
        "regularization_retry_count": [],
        "regularization_r_try": [],
        "energy": [],
        "accepted_energy": [],
        "energy_change": [],
        "stop_criterion": str(stop_mode),
        "stop_tolerance": float(stop_tol),
        "residual_tolerance": float(tol),
        "nonlinear_policy": str(nonlinear_policy),
        "combined_stopping": dict(combined_stopping),
    }


def _history_append(
    history: dict[str, object],
    *,
    criterion_abs: float,
    rel_residual: float,
    r_value: float,
    alpha: float | None,
    accepted_correction_norm: float | None,
    iterate_free_norm: float | None,
    accepted_relative_correction_norm: float | None,
    linear_iterations: int,
    linear_solve_time: float,
    linear_preconditioner_time: float,
    linear_orthogonalization_time: float,
    iteration_wall_time: float,
    line_search_iterations: int,
    deflation_basis_dim_solve: int,
    deflation_basis_dim_end: int,
    linear_true_residual_final: float | None,
    linear_hit_max_iterations: bool,
    linear_converged_reason: int | None,
    descent_direction: bool | None,
    guarded_max_it_accept: bool,
    regularization_retry_count: int,
    regularization_r_try: float,
    energy_value: float | None,
    accepted_energy: float | None,
    energy_change: float | None,
) -> None:
    history["criterion"].append(float(criterion_abs))
    history["residual"].append(float(rel_residual))
    history["r"].append(float(r_value))
    history["alpha"].append(np.nan if alpha is None else float(alpha))
    history["accepted_correction_norm"].append(
        np.nan if accepted_correction_norm is None else float(accepted_correction_norm)
    )
    history["iterate_free_norm"].append(np.nan if iterate_free_norm is None else float(iterate_free_norm))
    history["accepted_relative_correction_norm"].append(
        np.nan if accepted_relative_correction_norm is None else float(accepted_relative_correction_norm)
    )
    history["linear_iterations"].append(int(linear_iterations))
    history["linear_solve_time"].append(float(linear_solve_time))
    history["linear_preconditioner_time"].append(float(linear_preconditioner_time))
    history["linear_orthogonalization_time"].append(float(linear_orthogonalization_time))
    history["iteration_wall_time"].append(float(iteration_wall_time))
    history["line_search_iterations"].append(int(line_search_iterations))
    history["deflation_basis_dim_solve"].append(int(deflation_basis_dim_solve))
    history["deflation_basis_dim_end"].append(int(deflation_basis_dim_end))
    history["linear_true_residual_final"].append(
        np.nan if linear_true_residual_final is None else float(linear_true_residual_final)
    )
    history["linear_hit_max_iterations"].append(bool(linear_hit_max_iterations))
    history["linear_converged_reason"].append(np.nan if linear_converged_reason is None else float(linear_converged_reason))
    history["descent_direction"].append(np.nan if descent_direction is None else float(bool(descent_direction)))
    history["guarded_max_it_accept"].append(bool(guarded_max_it_accept))
    history["regularization_retry_count"].append(int(regularization_retry_count))
    history["regularization_r_try"].append(float(regularization_r_try))
    history["energy"].append(np.nan if energy_value is None else float(energy_value))
    history["accepted_energy"].append(np.nan if accepted_energy is None else float(accepted_energy))
    history["energy_change"].append(np.nan if energy_change is None else float(energy_change))


def _finalize_newton_history(history: dict[str, object], *, iterations: int, flag_N: int) -> dict[str, object]:
    finalized = dict(history)
    finalized["iterations"] = int(iterations)
    finalized["flag_N"] = int(flag_N)
    bool_fields = {"linear_hit_max_iterations", "guarded_max_it_accept"}
    int_fields = {
        "linear_iterations",
        "regularization_retry_count",
        "line_search_iterations",
        "deflation_basis_dim_solve",
        "deflation_basis_dim_end",
    }
    for key, value in tuple(finalized.items()):
        if not isinstance(value, list):
            continue
        if key in bool_fields:
            finalized[key] = np.asarray(value, dtype=bool)
        elif key in int_fields:
            finalized[key] = np.asarray(value, dtype=np.int64)
        else:
            finalized[key] = np.asarray(value, dtype=np.float64)
    return finalized


def _maybe_enable_solver_diagnostics(linear_system_solver) -> None:
    backend = str(getattr(linear_system_solver, "_pc_backend", "")).strip().lower()
    if type(linear_system_solver).__name__ == "PetscMatlabExactDFGMRESSolver" and backend == "pmg_shell":
        return
    enable = getattr(linear_system_solver, "enable_diagnostics", None)
    if callable(enable):
        enable(True)


def _last_solve_info(linear_system_solver) -> dict[str, object]:
    getter = getattr(linear_system_solver, "get_last_solve_info", None)
    if callable(getter):
        return dict(getter())
    return {}


def _linear_tolerance(linear_system_solver) -> float:
    try:
        return float(getattr(linear_system_solver, "tolerance"))
    except Exception:
        return np.nan


def _regularization_retry_values(r: float, enabled: bool) -> list[float]:
    if not enabled:
        return [float(r)]
    values = [float(r), float(min(4.0 * r, 1.0)), float(min(16.0 * r, 1.0))]
    deduped: list[float] = []
    for value in values:
        if not deduped or abs(float(value) - float(deduped[-1])) > 1.0e-15:
            deduped.append(float(value))
    return deduped


def _energy_merit(
    constitutive_matrix_builder,
    U: np.ndarray,
    f: np.ndarray,
    Q: np.ndarray,
    *,
    external_load_scale: float,
) -> float:
    return float(
        constitutive_matrix_builder.potential_energy(U)
        - float(external_load_scale) * _free_dot(f, U, Q)
    )


def _armijo_backtracking(
    U_it: np.ndarray,
    dU: np.ndarray,
    *,
    phi_it: float,
    gradient_dot_direction: float,
    constitutive_matrix_builder,
    f: np.ndarray,
    Q: np.ndarray,
    armijo_alpha0: float,
    armijo_c1: float,
    armijo_shrink: float,
    armijo_max_ls: int,
    external_load_scale: float,
) -> dict[str, float | bool | None]:
    if (
        armijo_max_ls <= 0
        or not np.isfinite(phi_it)
        or not np.isfinite(gradient_dot_direction)
        or gradient_dot_direction >= 0.0
    ):
        return {
            "alpha": 0.0,
            "accepted": False,
            "accepted_energy": None,
            "energy_change": None,
            "last_tried_alpha": 0.0,
            "line_search_evaluations": 0,
        }

    def _energy_at_alpha(alpha_value: float) -> float:
        return _energy_merit(
            constitutive_matrix_builder,
            U_it + float(alpha_value) * dU,
            f,
            Q,
            external_load_scale=external_load_scale,
        )

    def _bounded_armijo(alpha_lo: float, alpha_hi: float, directional_derivative: float):
        alpha_lo = float(alpha_lo)
        alpha_hi = float(alpha_hi)
        if (
            not np.isfinite(alpha_lo)
            or not np.isfinite(alpha_hi)
            or not np.isfinite(float(armijo_shrink))
            or float(armijo_shrink) <= 0.0
            or float(armijo_shrink) >= 1.0
            or alpha_hi <= max(alpha_lo, 0.0) + 1.0e-14
        ):
            return 0.0, np.inf, 0, False, 0.0
        alpha_trial = min(
            max(float(armijo_alpha0), max(alpha_lo, 1.0e-12)),
            float(alpha_hi),
        )
        n_eval = 0
        last_tried_alpha = 0.0
        for _ in range(max(1, int(armijo_max_ls))):
            last_tried_alpha = float(alpha_trial)
            trial_value = _energy_at_alpha(alpha_trial)
            n_eval += 1
            if (
                np.isfinite(trial_value)
                and trial_value <= phi_it + float(armijo_c1) * alpha_trial * directional_derivative
            ):
                return float(alpha_trial), float(trial_value), int(n_eval), True, float(last_tried_alpha)
            next_alpha = alpha_trial * float(armijo_shrink)
            floor_alpha = max(alpha_lo, 1.0e-12)
            if next_alpha <= floor_alpha + 1.0e-16:
                alpha_trial = floor_alpha
                if alpha_trial <= floor_alpha + 1.0e-16:
                    break
            else:
                alpha_trial = next_alpha
        return 0.0, np.inf, int(n_eval), False, float(last_tried_alpha)

    alpha, phi_alpha, n_eval, accepted, last_tried_alpha = _bounded_armijo(
        0.0,
        1.0,
        float(gradient_dot_direction),
    )
    if accepted:
        return {
            "alpha": float(alpha),
            "accepted": True,
            "accepted_energy": float(phi_alpha),
            "energy_change": float(phi_alpha - phi_it),
            "last_tried_alpha": float(last_tried_alpha),
            "line_search_evaluations": int(n_eval),
        }

    return {
        "alpha": 0.0,
        "accepted": False,
        "accepted_energy": None,
        "energy_change": None,
        "last_tried_alpha": float(last_tried_alpha),
        "line_search_evaluations": int(n_eval),
    }


def _solve_direction_once(
    linear_system_solver,
    K_r,
    rhs: np.ndarray,
    *,
    use_full_operator: bool,
    free_idx: np.ndarray,
    b_full=None,
    local_rhs=None,
    preconditioning_matrix=None,
):
    snap_before = _collector_snapshot(linear_system_solver)
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
            if local_rhs is not None:
                dU_free = _solve_linear_system_local(
                    linear_system_solver,
                    K_r,
                    rhs,
                    b_full=b_full,
                    local_rhs=local_rhs,
                    free_idx=free_idx,
                )
            else:
                dU_free = _solve_linear_system(linear_system_solver, K_r, rhs, b_full=b_full, free_idx=free_idx)
        else:
            K_free = extract_submatrix_free(K_r, free_idx)
            _setup_linear_system(linear_system_solver, K_free, A_full=K_r, free_idx=free_idx)
            if getattr(linear_system_solver, "supports_a_orthogonalization", lambda: True)():
                linear_system_solver.A_orthogonalize(K_free)
            dU_free = _solve_linear_system(linear_system_solver, K_free, rhs, b_full=b_full, free_idx=free_idx)
    finally:
        _release_iteration_resources(linear_system_solver)
        _destroy_petsc_mat(K_free)
    return (
        np.asarray(dU_free, dtype=np.float64).reshape(-1),
        _collector_delta(snap_before, _collector_snapshot(linear_system_solver)),
        _last_solve_info(linear_system_solver),
    )


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
    stopping_criterion: str = "relative_residual",
    stopping_tol: float | None = None,
    return_history: bool = False,
    nonlinear_policy: str = "residual",
    combined_stopping: dict[str, float] | None = None,
    armijo_alpha0: float = 1.0,
    armijo_c1: float = 1.0e-4,
    armijo_shrink: float = 0.5,
    armijo_max_ls: int | None = None,
    inner_regularization_retry: bool = False,
    guarded_max_it_accept: bool = False,
    energy_force_scale: float = 1.0,
):
    """Plain Newton solver for ``F(U) = f``.

    Returns ``(U_it, flag_N, it)`` unless ``return_history=True``, in which case
    it returns ``(U_it, flag_N, it, history)``.
    """

    U_it = _to_float_matrix(U_ini)
    Q = np.asarray(Q, dtype=bool)
    shape = U_it.shape

    free_idx = q_to_free_indices(Q)
    if free_idx.size == 0:
        if return_history:
            empty_history = _finalize_newton_history(
                _newton_history_template(
                    stop_mode=_normalize_stopping_criterion(stopping_criterion),
                    stop_tol=_resolve_stopping_tolerance(tol, stopping_tol),
                    tol=tol,
                    nonlinear_policy=_normalize_nonlinear_policy(nonlinear_policy),
                    combined_stopping=_resolve_combined_stopping(combined_stopping),
                ),
                iterations=0,
                flag_N=0,
            )
            return U_it, 0, 0, empty_history
        return U_it, 0, 0
    f_free = _to_free_vector(f, Q)

    norm_f = _free_norm(f, Q)
    if norm_f == 0.0:
        norm_f = 1.0
    stop_mode = _normalize_stopping_criterion(stopping_criterion)
    if stop_mode == "absolute_delta_lambda":
        raise ValueError("Newton stopping_criterion='absolute_delta_lambda' is only supported by newton_ind_ssr.")
    stop_tol = _resolve_stopping_tolerance(tol, stopping_tol)
    nonlinear_mode = _normalize_nonlinear_policy(nonlinear_policy)
    combined_stop = _resolve_combined_stopping(combined_stopping)
    post_step_stop_required = _combined_stop_requires_post_step(combined_stop)
    armijo_max_ls_effective = int(it_damp_max if armijo_max_ls is None else armijo_max_ls)
    if return_history or guarded_max_it_accept or inner_regularization_retry or nonlinear_mode == "energy_armijo":
        _maybe_enable_solver_diagnostics(linear_system_solver)

    it = 0
    flag_N = 0
    r = float(r_min)
    compute_diffs = True
    history = _newton_history_template(
        stop_mode=stop_mode,
        stop_tol=stop_tol,
        tol=tol,
        nonlinear_policy=nonlinear_mode,
        combined_stopping=combined_stop,
    )

    while True:
        it += 1
        iter_t0 = perf_counter()

        use_full_operator = _prefers_full_system_operator(linear_system_solver, K_elast)
        use_free_build = _supports_free_builder(constitutive_matrix_builder, "build_F_reduced_free")
        use_local_build = use_full_operator and _supports_local_builder(constitutive_matrix_builder, "build_F_reduced_local")
        comm = _local_comm_from_operator(K_elast) if use_local_build else None
        K_tangent = None
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
        energy_value = None
        if nonlinear_mode == "energy_armijo":
            energy_value = _energy_merit(
                constitutive_matrix_builder,
                U_it,
                f,
                Q,
                external_load_scale=float(energy_force_scale),
            )

        early_residual_stop = (
            stop_mode == "relative_residual"
            and criterion < stop_tol
            and not post_step_stop_required
            and _combined_stop_satisfied(
                combined_stop,
                relative_residual=criterion,
                relative_correction=None,
                energy_change=None,
            )
        )
        if compute_diffs and early_residual_stop:
            basis_dim_current = int(_basis_cols(linear_system_solver))
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
                line_search_iterations=0,
                deflation_basis_dim_solve=int(basis_dim_current),
                deflation_basis_dim_end=int(basis_dim_current),
                tolerance=float(stop_tol),
                residual_tolerance=float(tol),
                stop_criterion=str(stop_mode),
                stop_tolerance=float(stop_tol),
                stopping_value=float(criterion),
                iterate_free_norm=float(_free_norm(U_it, Q)),
                accepted_correction_norm=None,
                accepted_relative_correction_norm=None,
                linear_true_residual_final=None,
                linear_hit_max_iterations=False,
                linear_converged_reason=None,
                descent_direction=None,
                guarded_max_it_accept=False,
                energy_value=(None if energy_value is None else float(energy_value)),
                accepted_energy=(None if energy_value is None else float(energy_value)),
                energy_change=0.0 if energy_value is not None else None,
                status="converged",
            )
            _history_append(
                history,
                criterion_abs=float(criterion_abs),
                rel_residual=float(criterion),
                r_value=float(r),
                alpha=None,
                accepted_correction_norm=None,
                iterate_free_norm=float(_free_norm(U_it, Q)),
                accepted_relative_correction_norm=None,
                linear_iterations=0,
                linear_solve_time=0.0,
                linear_preconditioner_time=0.0,
                linear_orthogonalization_time=0.0,
                iteration_wall_time=float(perf_counter() - iter_t0),
                line_search_iterations=0,
                deflation_basis_dim_solve=int(basis_dim_current),
                deflation_basis_dim_end=int(basis_dim_current),
                linear_true_residual_final=None,
                linear_hit_max_iterations=False,
                linear_converged_reason=None,
                descent_direction=None,
                guarded_max_it_accept=False,
                regularization_retry_count=0,
                regularization_r_try=float(r),
                energy_value=energy_value,
                accepted_energy=energy_value,
                energy_change=(0.0 if energy_value is not None else None),
            )
            _cleanup_pre_solve_iteration_mats(
                K_tangent=K_tangent,
                K_r=None,
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
        retry_values = _regularization_retry_values(r, bool(inner_regularization_retry))
        basis_snapshot = _basis_snapshot(linear_system_solver) if len(retry_values) > 1 else None
        linear_tol = _linear_tolerance(linear_system_solver)
        chosen_attempt = None
        last_attempt = None
        for retry_index, r_try in enumerate(retry_values):
            if retry_index > 0 and basis_snapshot is not None:
                _basis_restore(linear_system_solver, basis_snapshot)
            cached_regularized = _build_regularized_from_cached_if_available(constitutive_matrix_builder, r_try)
            if cached_regularized is not None:
                K_r_try = cached_regularized
            else:
                K_tangent = _ensure_tangent_matrix_for_regularization(
                    constitutive_matrix_builder,
                    U_it,
                    K_tangent=K_tangent,
                    use_free_build=use_free_build,
                )
                K_r_try = _combine_matrices(r_try, K_elast, 1.0 - r_try, K_tangent)
            preconditioning_matrix = None
            if use_full_operator and _requires_explicit_preconditioning_matrix(linear_system_solver):
                if _needs_preconditioning_matrix_refresh(linear_system_solver):
                    preconditioning_matrix = _explicit_preconditioning_matrix(
                        constitutive_matrix_builder,
                        linear_system_solver,
                        regularization_r=r_try,
                        K_elast=K_elast,
                    )
            dU_free, iter_delta, solve_info = _solve_direction_once(
                linear_system_solver,
                K_r_try,
                rhs,
                use_full_operator=use_full_operator,
                free_idx=free_idx,
                b_full=(rhs_local if use_local_build else None),
                local_rhs=(rhs_local if use_local_build else None),
                preconditioning_matrix=preconditioning_matrix,
            )
            dU = np.zeros(U_it.size, dtype=np.float64)
            dU[free_idx] = np.asarray(dU_free, dtype=np.float64)
            dU = dU.reshape(shape, order="F")
            dU_local_free = (
                _local_owned_free_rows_from_field(dU, constitutive_matrix_builder.owned_tangent_pattern)
                if use_local_build
                else None
            )
            if use_local_build:
                rhs_dot_dU = _dist_dot_local(rhs, dU_local_free, comm)
            else:
                rhs_dot_dU = float(np.dot(np.asarray(rhs, dtype=np.float64).reshape(-1), dU_free.reshape(-1)))
            descent_direction = bool(np.isfinite(rhs_dot_dU) and rhs_dot_dU > 0.0)
            if nonlinear_mode == "energy_armijo":
                armijo = _armijo_backtracking(
                    U_it,
                    dU,
                    phi_it=float(
                        energy_value
                        if energy_value is not None
                        else _energy_merit(
                            constitutive_matrix_builder,
                            U_it,
                            f,
                            Q,
                            external_load_scale=float(energy_force_scale),
                        )
                    ),
                    gradient_dot_direction=float(-rhs_dot_dU),
                    constitutive_matrix_builder=constitutive_matrix_builder,
                    f=f,
                    Q=Q,
                    armijo_alpha0=float(armijo_alpha0),
                    armijo_c1=float(armijo_c1),
                    armijo_shrink=float(armijo_shrink),
                    armijo_max_ls=int(armijo_max_ls_effective),
                    external_load_scale=float(energy_force_scale),
                )
                alpha_cap = 1.0
                if bool(armijo["accepted"]) and np.isfinite(float(armijo["alpha"])) and float(armijo["alpha"]) > 0.0:
                    alpha_cap = float(armijo["alpha"])
                damping_info = damping(
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
                    dU_local_free=dU_local_free,
                    comm=comm,
                    alpha_upper=float(alpha_cap),
                    return_info=True,
                )
                alpha = float(damping_info["alpha"])
                line_search_iterations = int(damping_info["line_search_iterations"]) + int(
                    armijo.get("line_search_evaluations", 0)
                )
                if float(alpha) > 0.0:
                    accepted_energy = _energy_merit(
                        constitutive_matrix_builder,
                        U_it + float(alpha) * dU,
                        f,
                        Q,
                        external_load_scale=float(energy_force_scale),
                    )
                    energy_change = None if energy_value is None else abs(float(accepted_energy - energy_value))
                else:
                    accepted_energy = None
                    energy_change = None
                line_search_failed_for_retry = float(alpha) == 0.0
            else:
                damping_info = damping(
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
                    dU_local_free=dU_local_free,
                    comm=comm,
                    return_info=True,
                )
                alpha = float(damping_info["alpha"])
                line_search_iterations = int(damping_info["line_search_iterations"])
                accepted_energy = energy_value
                energy_change = None
                line_search_failed_for_retry = False
            iterate_free_norm = _free_norm(U_it, Q)
            accepted_correction_norm = float(np.linalg.norm(float(alpha) * _to_free_vector(dU, Q)))
            accepted_relative_correction_norm = float(accepted_correction_norm / max(iterate_free_norm, 1.0e-30))
            true_residual_final = solve_info.get("true_residual_final")
            hit_max_iterations = bool(solve_info.get("hit_max_iterations", False))
            guarded_accept = bool(
                guarded_max_it_accept
                and hit_max_iterations
                and descent_direction
                and float(alpha) >= 0.25
                and true_residual_final is not None
                and np.isfinite(float(true_residual_final))
                and np.isfinite(linear_tol)
                and float(true_residual_final) <= 2.0 * float(linear_tol)
            )
            retry_needed = False
            if bool(inner_regularization_retry):
                retry_needed = (not descent_direction) or (hit_max_iterations and not guarded_accept) or line_search_failed_for_retry
            attempt = {
                "alpha": float(alpha),
                "dU": dU,
                "iter_delta": dict(iter_delta),
                "solve_info": dict(solve_info),
                "r_try": float(r_try),
                "retry_count": int(retry_index),
                "K_r": K_r_try,
                "descent_direction": bool(descent_direction),
                "guarded_accept": bool(guarded_accept),
                "iterate_free_norm": float(iterate_free_norm),
                "accepted_correction_norm": float(accepted_correction_norm),
                "accepted_relative_correction_norm": float(accepted_relative_correction_norm),
                "accepted_energy": accepted_energy,
                "energy_change": energy_change,
                "line_search_iterations": int(line_search_iterations),
                "deflation_basis_dim_solve": int(solve_info.get("basis_cols", _basis_cols(linear_system_solver))),
            }
            last_attempt = attempt
            if retry_needed:
                if retry_index + 1 < len(retry_values):
                    if basis_snapshot is not None:
                        _basis_restore(linear_system_solver, basis_snapshot)
                    if not _is_builder_cached_matrix(K_r_try, constitutive_matrix_builder):
                        _destroy_petsc_mat(K_r_try)
                continue
            chosen_attempt = attempt
            break

        if chosen_attempt is None and last_attempt is not None:
            chosen_attempt = dict(last_attempt)
            chosen_attempt["alpha"] = 0.0
            chosen_attempt["accepted_correction_norm"] = 0.0
            chosen_attempt["accepted_relative_correction_norm"] = 0.0
            chosen_attempt["accepted_energy"] = energy_value
            chosen_attempt["energy_change"] = None
        if chosen_attempt is None:
            raise RuntimeError("Newton failed to produce a trial step")

        r = float(chosen_attempt["r_try"])
        dU = np.asarray(chosen_attempt["dU"], dtype=np.float64)
        alpha = float(chosen_attempt["alpha"])
        iter_delta = dict(chosen_attempt["iter_delta"])
        solve_info = dict(chosen_attempt["solve_info"])
        iterate_free_norm = float(chosen_attempt["iterate_free_norm"])
        accepted_correction_norm = float(chosen_attempt["accepted_correction_norm"])
        accepted_relative_correction_norm = float(chosen_attempt["accepted_relative_correction_norm"])
        descent_direction = bool(chosen_attempt["descent_direction"])
        guarded_accept = bool(chosen_attempt["guarded_accept"])
        accepted_energy = chosen_attempt["accepted_energy"]
        energy_change = chosen_attempt["energy_change"]
        retry_count = int(chosen_attempt["retry_count"])
        K_r = chosen_attempt["K_r"]
        line_search_iterations = int(chosen_attempt["line_search_iterations"])
        deflation_basis_dim_solve = int(chosen_attempt["deflation_basis_dim_solve"])
        converged_on_correction = (
            stop_mode == "relative_correction"
            and float(alpha) > 0.0
            and accepted_relative_correction_norm < stop_tol
        )
        if combined_stop:
            stop_satisfied = bool(
                float(alpha) > 0.0
                and _combined_stop_satisfied(
                    combined_stop,
                    relative_residual=float(criterion),
                    relative_correction=float(accepted_relative_correction_norm),
                    energy_change=(None if energy_change is None else float(energy_change)),
                )
            )
        else:
            stop_satisfied = bool(converged_on_correction)

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
            line_search_iterations=int(line_search_iterations),
            deflation_basis_dim_solve=int(deflation_basis_dim_solve),
            deflation_basis_dim_end=int(deflation_basis_dim_solve),
            tolerance=float(stop_tol),
            residual_tolerance=float(tol),
            stop_criterion=str(stop_mode),
            stop_tolerance=float(stop_tol),
            stopping_value=float(criterion if stop_mode == "relative_residual" else accepted_relative_correction_norm),
            iterate_free_norm=float(iterate_free_norm),
            accepted_correction_norm=float(accepted_correction_norm),
            accepted_relative_correction_norm=float(accepted_relative_correction_norm),
            linear_true_residual_final=(
                None if solve_info.get("true_residual_final") is None else float(solve_info.get("true_residual_final"))
            ),
            linear_hit_max_iterations=bool(solve_info.get("hit_max_iterations", False)),
            linear_converged_reason=(
                None if solve_info.get("converged_reason") is None else int(solve_info.get("converged_reason"))
            ),
            descent_direction=bool(descent_direction),
            guarded_max_it_accept=bool(guarded_accept),
            energy_value=(None if energy_value is None else float(energy_value)),
            accepted_energy=(None if accepted_energy is None else float(accepted_energy)),
            energy_change=(None if energy_change is None else float(energy_change)),
            status="converged" if stop_satisfied else "iterate",
        )
        _history_append(
            history,
            criterion_abs=float(criterion_abs),
            rel_residual=float(criterion),
            r_value=float(r),
            alpha=float(alpha),
            accepted_correction_norm=float(accepted_correction_norm),
            iterate_free_norm=float(iterate_free_norm),
            accepted_relative_correction_norm=float(accepted_relative_correction_norm),
            linear_iterations=int(iter_delta["iterations"]),
            linear_solve_time=float(iter_delta["solve_time"]),
            linear_preconditioner_time=float(iter_delta["preconditioner_time"]),
            linear_orthogonalization_time=float(iter_delta["orthogonalization_time"]),
            iteration_wall_time=float(perf_counter() - iter_t0),
            line_search_iterations=int(line_search_iterations),
            deflation_basis_dim_solve=int(deflation_basis_dim_solve),
            deflation_basis_dim_end=int(deflation_basis_dim_solve),
            linear_true_residual_final=(
                None if solve_info.get("true_residual_final") is None else float(solve_info.get("true_residual_final"))
            ),
            linear_hit_max_iterations=bool(solve_info.get("hit_max_iterations", False)),
            linear_converged_reason=(
                None if solve_info.get("converged_reason") is None else int(solve_info.get("converged_reason"))
            ),
            descent_direction=bool(descent_direction),
            guarded_max_it_accept=bool(guarded_accept),
            regularization_retry_count=int(retry_count),
            regularization_r_try=float(r),
            energy_value=energy_value,
            accepted_energy=accepted_energy,
            energy_change=energy_change,
        )
        _cleanup_pre_solve_iteration_mats(
            K_tangent=K_tangent,
            K_r=K_r,
            use_full_operator=use_full_operator,
            constitutive_matrix_builder=constitutive_matrix_builder,
        )

        if stop_satisfied:
            U_it = U_it + alpha * dU
            break

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

    if return_history:
        return U_it, flag_N, it, _finalize_newton_history(history, iterations=it, flag_N=flag_N)
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
    first_iteration_extra_basis_free: list[np.ndarray] | None = None,
    stopping_criterion: str = "relative_residual",
    stopping_tol: float | None = None,
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
            "accepted_correction_norm": np.array([0.0], dtype=np.float64),
            "iterate_free_norm": np.array([0.0], dtype=np.float64),
            "accepted_relative_correction_norm": np.array([0.0], dtype=np.float64),
            "linear_iterations": np.array([0], dtype=np.int64),
            "linear_solve_time": np.array([0.0], dtype=np.float64),
            "linear_preconditioner_time": np.array([0.0], dtype=np.float64),
            "linear_orthogonalization_time": np.array([0.0], dtype=np.float64),
            "iteration_wall_time": np.array([0.0], dtype=np.float64),
            "line_search_iterations": np.array([0], dtype=np.int64),
            "deflation_basis_dim_solve": np.array([0], dtype=np.int64),
            "deflation_basis_dim_end": np.array([0], dtype=np.int64),
            "first_iteration_linear_iterations": 0,
            "first_iteration_linear_solve_time": 0.0,
            "first_iteration_linear_preconditioner_time": 0.0,
            "first_iteration_linear_orthogonalization_time": 0.0,
            "first_iteration_warm_start_active": False,
            "first_iteration_warm_start_basis_dim": 0,
            "first_accepted_correction_iteration": 0,
            "first_accepted_correction_norm": 0.0,
            "first_accepted_correction_free": np.zeros(0, dtype=np.float64),
            "stop_criterion": str(_normalize_stopping_criterion(stopping_criterion)),
            "stop_tolerance": float(_resolve_stopping_tolerance(tol, stopping_tol)),
            "residual_tolerance": float(tol),
        }
        return U_it, float(lambda_it), 0, 0, history
    f_free = _to_free_vector(f, Q)
    first_iteration_extra_basis_free = [
        np.asarray(v, dtype=np.float64).reshape(-1).copy()
        for v in (first_iteration_extra_basis_free or [])
        if np.asarray(v, dtype=np.float64).size
    ]

    norm_f = _free_norm(f, Q)
    if norm_f == 0.0:
        norm_f = 1.0
    stop_mode = _normalize_stopping_criterion(stopping_criterion)
    stop_tol = _resolve_stopping_tolerance(tol, stopping_tol)

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
    accepted_correction_norm_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    iterate_free_norm_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    accepted_relative_correction_norm_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    linear_iterations_hist = np.zeros(int(it_newt_max), dtype=np.int64)
    linear_solve_hist = np.zeros(int(it_newt_max), dtype=np.float64)
    linear_preconditioner_hist = np.zeros(int(it_newt_max), dtype=np.float64)
    linear_orthogonalization_hist = np.zeros(int(it_newt_max), dtype=np.float64)
    iteration_wall_hist = np.full(int(it_newt_max), np.nan, dtype=np.float64)
    line_search_iterations_hist = np.zeros(int(it_newt_max), dtype=np.int64)
    deflation_basis_dim_solve_hist = np.zeros(int(it_newt_max), dtype=np.int64)
    deflation_basis_dim_end_hist = np.zeros(int(it_newt_max), dtype=np.int64)
    first_iteration_linear_iterations = 0
    first_iteration_linear_solve_time = 0.0
    first_iteration_linear_preconditioner_time = 0.0
    first_iteration_linear_orthogonalization_time = 0.0
    first_iteration_warm_start_active = False
    first_iteration_warm_start_basis_dim = 0
    first_accepted_correction_iteration = 0
    first_accepted_correction_norm = np.nan
    first_accepted_correction_free: np.ndarray | None = None

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
        if compute_diffs and stop_mode == "relative_residual" and rel_resid < stop_tol and it > 1:
            basis_dim_current = int(_basis_cols(linear_system_solver))
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
                line_search_iterations=0,
                deflation_basis_dim_solve=int(basis_dim_current),
                deflation_basis_dim_end=int(basis_dim_current),
                tolerance=float(stop_tol),
                residual_tolerance=float(tol),
                stop_criterion=str(stop_mode),
                stop_tolerance=float(stop_tol),
                stopping_value=float(rel_resid),
                iterate_free_norm=float(_free_norm(U_it, Q)),
                accepted_correction_norm=None,
                accepted_relative_correction_norm=None,
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
                temporary_basis_snapshot = None
                if (
                    it == 1
                    and first_iteration_extra_basis_free
                    and getattr(linear_system_solver, "supports_dynamic_deflation_basis", lambda: True)()
                ):
                    temporary_basis_snapshot = _basis_snapshot(linear_system_solver)
                    for vec in first_iteration_extra_basis_free:
                        linear_system_solver.expand_deflation_basis(vec)
                    first_iteration_warm_start_active = True
                    first_iteration_warm_start_basis_dim = int(len(first_iteration_extra_basis_free))
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
                temporary_basis_snapshot = None
                if (
                    it == 1
                    and first_iteration_extra_basis_free
                    and getattr(linear_system_solver, "supports_dynamic_deflation_basis", lambda: True)()
                ):
                    temporary_basis_snapshot = _basis_snapshot(linear_system_solver)
                    for vec in first_iteration_extra_basis_free:
                        linear_system_solver.expand_deflation_basis(vec)
                    first_iteration_warm_start_active = True
                    first_iteration_warm_start_basis_dim = int(len(first_iteration_extra_basis_free))
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
            if temporary_basis_snapshot is not None:
                _basis_restore(linear_system_solver, temporary_basis_snapshot)
            _destroy_petsc_mat(K_free)
            if not _is_builder_cached_matrix(K_tangent, constitutive_matrix_builder):
                _destroy_petsc_mat(K_tangent)
            if not use_full_operator and not _is_builder_cached_matrix(K_r, constitutive_matrix_builder):
                _destroy_petsc_mat(K_r)
        iter_delta = _collector_delta(snap_before_iter, _collector_snapshot(linear_system_solver))
        solve_info = _last_solve_info(linear_system_solver)
        if it == 1:
            first_iteration_linear_iterations = int(iter_delta["iterations"])
            first_iteration_linear_solve_time = float(iter_delta["solve_time"])
            first_iteration_linear_preconditioner_time = float(iter_delta["preconditioner_time"])
            first_iteration_linear_orthogonalization_time = float(iter_delta["orthogonalization_time"])

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
        damping_info = damping_alg5(
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
            return_info=True,
        )
        alpha = float(damping_info["alpha"])
        line_search_iterations = int(damping_info["line_search_iterations"])
        alpha_hist[it - 1] = alpha
        delta_lambda_hist[it - 1] = float(d_l)
        accepted_delta_lambda_hist[it - 1] = float(alpha * d_l)
        abs_delta_lambda = float(abs(d_l))
        iterate_free_norm_hist[it - 1] = _free_norm(U_it, Q)
        correction_free = np.asarray(alpha * _to_free_vector(d_U, Q), dtype=np.float64).reshape(-1)
        accepted_correction_norm_hist[it - 1] = float(np.linalg.norm(correction_free))
        denom_u = max(float(iterate_free_norm_hist[it - 1]), 1.0e-30)
        accepted_relative_correction_norm_hist[it - 1] = float(accepted_correction_norm_hist[it - 1] / denom_u)
        converged_on_correction = (
            stop_mode == "relative_correction"
            and float(alpha) > 0.0
            and accepted_relative_correction_norm_hist[it - 1] < stop_tol
        )
        converged_on_delta_lambda = (
            stop_mode == "absolute_delta_lambda"
            and float(alpha) > 0.0
            and abs_delta_lambda < stop_tol
        )
        linear_iterations_hist[it - 1] = int(iter_delta["iterations"])
        linear_solve_hist[it - 1] = float(iter_delta["solve_time"])
        linear_preconditioner_hist[it - 1] = float(iter_delta["preconditioner_time"])
        linear_orthogonalization_hist[it - 1] = float(iter_delta["orthogonalization_time"])
        iteration_wall_hist[it - 1] = float(perf_counter() - iter_t0)
        line_search_iterations_hist[it - 1] = int(line_search_iterations)
        deflation_basis_dim_solve_hist[it - 1] = int(solve_info.get("basis_cols", _basis_cols(linear_system_solver)))
        deflation_basis_dim_end_hist[it - 1] = int(deflation_basis_dim_solve_hist[it - 1])
        if first_accepted_correction_free is None and float(alpha) > 0.0:
            first_accepted_correction_iteration = int(it)
            first_accepted_correction_norm = float(np.linalg.norm(correction_free))
            first_accepted_correction_free = correction_free

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
            line_search_iterations=int(line_search_iterations_hist[it - 1]),
            deflation_basis_dim_solve=int(deflation_basis_dim_solve_hist[it - 1]),
            deflation_basis_dim_end=int(deflation_basis_dim_end_hist[it - 1]),
            tolerance=float(stop_tol),
            residual_tolerance=float(tol),
            stop_criterion=str(stop_mode),
            stop_tolerance=float(stop_tol),
            stopping_value=float(
                rel_resid
                if stop_mode == "relative_residual"
                else abs_delta_lambda
                if stop_mode == "absolute_delta_lambda"
                else accepted_relative_correction_norm_hist[it - 1]
            ),
            iterate_free_norm=float(iterate_free_norm_hist[it - 1]),
            accepted_correction_norm=float(accepted_correction_norm_hist[it - 1]),
            accepted_relative_correction_norm=float(accepted_relative_correction_norm_hist[it - 1]),
            status="converged" if (converged_on_correction or converged_on_delta_lambda) else "iterate",
        )

        if converged_on_correction or converged_on_delta_lambda:
            U_it = U_it + alpha * d_U
            denom = _free_dot(f, U_it, Q)
            if denom != 0.0:
                U_it = U_it * (omega / denom)
            lambda_it = lambda_it + alpha * d_l
            break

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
        "accepted_correction_norm": accepted_correction_norm_hist[:it],
        "iterate_free_norm": iterate_free_norm_hist[:it],
        "accepted_relative_correction_norm": accepted_relative_correction_norm_hist[:it],
        "linear_iterations": linear_iterations_hist[:it],
        "linear_solve_time": linear_solve_hist[:it],
        "linear_preconditioner_time": linear_preconditioner_hist[:it],
        "linear_orthogonalization_time": linear_orthogonalization_hist[:it],
        "iteration_wall_time": iteration_wall_hist[:it],
        "line_search_iterations": line_search_iterations_hist[:it],
        "deflation_basis_dim_solve": deflation_basis_dim_solve_hist[:it],
        "deflation_basis_dim_end": deflation_basis_dim_end_hist[:it],
        "first_iteration_linear_iterations": int(first_iteration_linear_iterations),
        "first_iteration_linear_solve_time": float(first_iteration_linear_solve_time),
        "first_iteration_linear_preconditioner_time": float(first_iteration_linear_preconditioner_time),
        "first_iteration_linear_orthogonalization_time": float(first_iteration_linear_orthogonalization_time),
        "first_iteration_warm_start_active": bool(first_iteration_warm_start_active),
        "first_iteration_warm_start_basis_dim": int(first_iteration_warm_start_basis_dim),
        "first_accepted_correction_iteration": int(first_accepted_correction_iteration),
        "first_accepted_correction_norm": float(first_accepted_correction_norm)
        if np.isfinite(first_accepted_correction_norm)
        else np.nan,
        "first_accepted_correction_free": np.asarray(first_accepted_correction_free, dtype=np.float64).copy()
        if first_accepted_correction_free is not None
        else np.zeros(0, dtype=np.float64),
        "stop_criterion": str(stop_mode),
        "stop_tolerance": float(stop_tol),
        "residual_tolerance": float(tol),
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
