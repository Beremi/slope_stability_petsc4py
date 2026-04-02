"""Indirect continuation routines for strength reduction."""

from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np

from ..nonlinear.damping import damping_alg5
from ..nonlinear.newton import _combine_matrices, _normalize_stopping_criterion, newton, newton_ind_ssr
from ..utils import petsc_vec_to_global_array, q_to_free_indices, release_petsc_aij_matrix

try:  # pragma: no cover - PETSc optional in tests
    from petsc4py import PETSc
except Exception:  # pragma: no cover
    PETSc = None

try:  # pragma: no cover - optional SciPy path for small dense reduced solves
    from scipy.linalg import lu_factor as _scipy_lu_factor
    from scipy.linalg import lu_solve as _scipy_lu_solve
except Exception:  # pragma: no cover
    _scipy_lu_factor = None
    _scipy_lu_solve = None


def _free_indices(Q: np.ndarray) -> np.ndarray:
    return q_to_free_indices(np.asarray(Q, dtype=bool))


def _free(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(-1, order="F")[_free_indices(Q)]


def _free_dot(a: np.ndarray, b: np.ndarray, Q: np.ndarray) -> float:
    return float(np.dot(_free(a, Q), _free(b, Q)))


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


def _collector_snapshot(solver):
    collector = solver.iteration_collector
    return {
        "iterations": collector.get_total_iterations(),
        "solve_time": collector.get_total_solve_time(),
        "preconditioner_time": collector.get_total_preconditioner_time(),
        "orthogonalization_time": collector.get_total_orthogonalization_time(),
    }


def _collector_delta(before: dict, after: dict) -> dict:
    return {
        "iterations": after["iterations"] - before["iterations"],
        "solve_time": after["solve_time"] - before["solve_time"],
        "preconditioner_time": after["preconditioner_time"] - before["preconditioner_time"],
        "orthogonalization_time": after["orthogonalization_time"] - before["orthogonalization_time"],
    }


def _basis_snapshot(solver):
    getter = getattr(solver, "get_deflation_basis_snapshot", None)
    if callable(getter):
        return getter()
    basis = getattr(solver, "deflation_basis", None)
    if basis is None:
        return None
    return np.array(basis, dtype=np.float64, copy=True)


def _basis_restore(solver, snapshot) -> None:
    restore = getattr(solver, "restore_deflation_basis", None)
    if callable(restore):
        restore(snapshot)
        return
    if hasattr(solver, "deflation_basis"):
        solver.deflation_basis = np.array(snapshot, dtype=np.float64, copy=True)


def _notify_attempt(solver, *, success: bool) -> None:
    notify = getattr(solver, "notify_continuation_attempt", None)
    if callable(notify):
        notify(success=bool(success))


def _positive_median(values: list[float]) -> float | None:
    positive = [float(v) for v in values if np.isfinite(v) and float(v) > 0.0]
    if not positive:
        return None
    return float(np.median(np.asarray(positive, dtype=np.float64)))


def _final_finite(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(finite[-1])


def _rescale_to_target_omega(U: np.ndarray, omega_target: float, f: np.ndarray, Q: np.ndarray) -> np.ndarray:
    U_arr = np.asarray(U, dtype=np.float64)
    denom = _free_dot(f, U_arr, Q)
    if abs(float(denom)) <= 1.0e-30:
        return U_arr.copy()
    return U_arr * (float(omega_target) / float(denom))


def _predictor_info_defaults() -> dict[str, float | bool | str | None]:
    return {
        "basis_dim": np.nan,
        "basis_condition": np.nan,
        "predictor_wall_time": 0.0,
        "predictor_alpha": np.nan,
        "predictor_beta": np.nan,
        "predictor_gamma": np.nan,
        "projected_delta_lambda": np.nan,
        "projected_correction_norm": np.nan,
        "energy_eval_count": np.nan,
        "energy_value": np.nan,
        "coarse_solve_wall_time": np.nan,
        "coarse_newton_iterations": np.nan,
        "coarse_residual_end": np.nan,
        "reduced_newton_iterations": np.nan,
        "reduced_gmres_iterations": np.nan,
        "reduced_projected_residual": np.nan,
        "reduced_omega_residual": np.nan,
        "state_coefficients": None,
        "state_coefficients_ref": None,
        "state_coefficient_sum": np.nan,
        "secant_correction_active": False,
        "secant_correction_basis_dim": np.nan,
        "secant_correction_trust_region_clipped": False,
        "secant_correction_predicted_residual_decrease": np.nan,
        "fallback_used": False,
        "fallback_error": None,
    }


def _trim_basis_snapshot(snapshot, max_cols: int | None):
    if snapshot is None:
        return None
    if max_cols is None:
        return snapshot
    max_cols = max(int(max_cols), 1)
    arr = np.asarray(snapshot, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] <= max_cols:
        return arr.copy()
    return np.asarray(arr[:, -max_cols:], dtype=np.float64)


def _streaming_reference_scales(
    *,
    omega_old: float,
    omega: float,
    lambda_old: float,
    lambda_value: float,
) -> tuple[float, float]:
    omega_scale = max(abs(float(omega) - float(omega_old)), 1.0e-12)
    lambda_scale = max(abs(float(lambda_value) - float(lambda_old)), 1.0e-12)
    return float(omega_scale), float(lambda_scale)


def _streaming_micro_domega(
    *,
    omega_prev: float,
    omega_curr: float,
    lambda_prev: float,
    lambda_curr: float,
    omega_scale: float,
    lambda_scale: float,
    s_micro: float,
    omega_remaining: float,
) -> tuple[float, float, float]:
    domega_prev = float(omega_curr) - float(omega_prev)
    if abs(domega_prev) <= 1.0e-12:
        return 0.0, 0.0, 0.0
    dlambda_prev = float(lambda_curr) - float(lambda_prev)
    slope = float(dlambda_prev / domega_prev)
    denom = float(np.sqrt((1.0 / max(float(omega_scale), 1.0e-12)) ** 2 + (slope / max(float(lambda_scale), 1.0e-12)) ** 2))
    if not np.isfinite(denom) or denom <= 1.0e-30:
        return 0.0, 0.0, 0.0
    domega = min(float(max(s_micro, 0.0)) / denom, max(float(omega_remaining), 0.0))
    if domega <= 0.0:
        return 0.0, 0.0, 0.0
    alpha_sec = float(domega / domega_prev)
    ds = float(denom * domega)
    return float(domega), float(alpha_sec), float(ds)


def _initial_segment_length_cap(
    *,
    domega_candidate: float,
    domega_initial: float,
    dlambda_initial: float,
    omega_anchor_prev: float,
    omega_anchor_curr: float,
    lambda_anchor_prev: float,
    lambda_anchor_curr: float,
    cap_factor: float,
) -> tuple[float, float, float]:
    domega_initial_abs = max(abs(float(domega_initial)), 1.0e-12)
    dlambda_initial_abs = max(abs(float(dlambda_initial)), 1.0e-12)
    length_limit = float(np.sqrt(2.0) * max(float(cap_factor), 0.0))
    domega_raw = float(max(float(domega_candidate), 0.0))
    if not np.isfinite(domega_raw) or domega_raw <= 0.0 or not np.isfinite(length_limit) or length_limit <= 0.0:
        return domega_raw, 0.0, length_limit
    omega_span = float(omega_anchor_curr) - float(omega_anchor_prev)
    if abs(omega_span) <= 1.0e-12:
        return domega_raw, float(domega_raw / domega_initial_abs), length_limit
    slope = (float(lambda_anchor_curr) - float(lambda_anchor_prev)) / omega_span
    raw_length = float(
        np.sqrt(
            (domega_raw / domega_initial_abs) ** 2
            + (abs(float(slope)) * domega_raw / dlambda_initial_abs) ** 2
        )
    )
    if not np.isfinite(raw_length) or raw_length <= length_limit:
        return domega_raw, raw_length, length_limit
    cap_ratio = float(length_limit / max(raw_length, 1.0e-12))
    domega_capped = float(max(domega_raw * cap_ratio, 1.0e-12))
    return domega_capped, raw_length, length_limit


def _history_box_scales(
    *,
    omega_hist: np.ndarray,
    lambda_hist: np.ndarray,
) -> tuple[float, float]:
    omega_arr = np.asarray(omega_hist, dtype=np.float64).reshape(-1)
    lambda_arr = np.asarray(lambda_hist, dtype=np.float64).reshape(-1)
    omega_span = max(float(np.max(omega_arr) - np.min(omega_arr)), 1.0e-12)
    lambda_span = max(float(np.max(lambda_arr) - np.min(lambda_arr)), 1.0e-12)
    return float(omega_span), float(lambda_span)


def _history_box_initial_segment_length(
    *,
    omega_hist: np.ndarray,
    lambda_hist: np.ndarray,
    factor: float = 1.0,
) -> float:
    omega_arr = np.asarray(omega_hist, dtype=np.float64).reshape(-1)
    lambda_arr = np.asarray(lambda_hist, dtype=np.float64).reshape(-1)
    if omega_arr.size < 2 or lambda_arr.size < 2:
        return 0.0
    omega_span, lambda_span = _history_box_scales(omega_hist=omega_arr, lambda_hist=lambda_arr)
    domega_initial = float(omega_arr[1] - omega_arr[0])
    dlambda_initial = float(lambda_arr[1] - lambda_arr[0])
    return float(
        np.sqrt((abs(domega_initial) / omega_span) ** 2 + (abs(dlambda_initial) / lambda_span) ** 2)
        * max(float(factor), 0.0)
    )


def _history_box_projected_length(
    *,
    domega_candidate: float,
    omega_hist: np.ndarray,
    lambda_hist: np.ndarray,
    omega_anchor_prev: float,
    omega_anchor_curr: float,
    lambda_anchor_prev: float,
    lambda_anchor_curr: float,
) -> float:
    domega_raw = float(max(float(domega_candidate), 0.0))
    if not np.isfinite(domega_raw) or domega_raw <= 0.0:
        return 0.0
    omega_span, lambda_span = _history_box_scales(omega_hist=omega_hist, lambda_hist=lambda_hist)
    omega_delta_last = float(omega_anchor_curr) - float(omega_anchor_prev)
    if abs(omega_delta_last) <= 1.0e-12:
        return float(abs(domega_raw) / omega_span)
    slope = (float(lambda_anchor_curr) - float(lambda_anchor_prev)) / omega_delta_last
    return float(
        np.sqrt(
            (domega_raw / omega_span) ** 2
            + (abs(float(slope)) * domega_raw / lambda_span) ** 2
        )
    )


def _history_box_path_length(
    *,
    omega_hist: np.ndarray,
    lambda_hist: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    omega_arr = np.asarray(omega_hist, dtype=np.float64).reshape(-1)
    lambda_arr = np.asarray(lambda_hist, dtype=np.float64).reshape(-1)
    if omega_arr.size < 2 or lambda_arr.size < 2:
        return 0.0
    start = max(int(start_idx), 0)
    end = min(int(end_idx), int(min(omega_arr.size, lambda_arr.size)) - 1)
    if end <= start:
        return 0.0
    omega_span, lambda_span = _history_box_scales(omega_hist=omega_arr, lambda_hist=lambda_arr)
    total = 0.0
    for idx in range(start + 1, end + 1):
        total += float(
            np.sqrt(
                (abs(float(omega_arr[idx] - omega_arr[idx - 1])) / omega_span) ** 2
                + (abs(float(lambda_arr[idx] - lambda_arr[idx - 1])) / lambda_span) ** 2
            )
        )
    return float(total)


def _history_box_step_length_cap(
    *,
    domega_candidate: float,
    omega_hist: np.ndarray,
    lambda_hist: np.ndarray,
    omega_anchor_prev: float,
    omega_anchor_curr: float,
    lambda_anchor_prev: float,
    lambda_anchor_curr: float,
    cap_factor: float,
) -> tuple[float, float, float]:
    omega_arr = np.asarray(omega_hist, dtype=np.float64).reshape(-1)
    lambda_arr = np.asarray(lambda_hist, dtype=np.float64).reshape(-1)
    if omega_arr.size < 2 or lambda_arr.size < 2:
        return float(max(float(domega_candidate), 0.0)), 0.0, 0.0

    domega_raw = float(max(float(domega_candidate), 0.0))
    if not np.isfinite(domega_raw) or domega_raw <= 0.0:
        return domega_raw, 0.0, 0.0

    length_limit = _history_box_initial_segment_length(
        omega_hist=omega_arr,
        lambda_hist=lambda_arr,
        factor=float(cap_factor),
    )
    if not np.isfinite(length_limit) or length_limit <= 0.0:
        return domega_raw, 0.0, length_limit

    raw_length = _history_box_projected_length(
        domega_candidate=domega_raw,
        omega_hist=omega_arr,
        lambda_hist=lambda_arr,
        omega_anchor_prev=float(omega_anchor_prev),
        omega_anchor_curr=float(omega_anchor_curr),
        lambda_anchor_prev=float(lambda_anchor_prev),
        lambda_anchor_curr=float(lambda_anchor_curr),
    )
    if not np.isfinite(raw_length) or raw_length <= length_limit:
        return domega_raw, raw_length, length_limit
    cap_ratio = float(length_limit / max(raw_length, 1.0e-12))
    domega_capped = float(max(domega_raw * cap_ratio, 1.0e-12))
    return domega_capped, raw_length, length_limit


def _orthogonalize_vectors_against_direction(
    vectors: list[np.ndarray],
    direction: np.ndarray,
    *,
    tol: float = 1.0e-10,
) -> list[np.ndarray]:
    if not vectors:
        return []
    dir_vec = np.asarray(direction, dtype=np.float64).reshape(-1)
    dir_norm_sq = float(np.dot(dir_vec, dir_vec))
    if not np.isfinite(dir_norm_sq) or dir_norm_sq <= 1.0e-30:
        return [np.asarray(v, dtype=np.float64).reshape(-1).copy() for v in vectors]
    out: list[np.ndarray] = []
    dir_norm = float(np.sqrt(dir_norm_sq))
    for vec in vectors:
        cand = np.asarray(vec, dtype=np.float64).reshape(-1).copy()
        coeff = float(np.dot(dir_vec, cand) / dir_norm_sq)
        cand -= coeff * dir_vec
        cand_norm = float(np.linalg.norm(cand))
        ref_scale = max(float(np.linalg.norm(vec)), dir_norm, 1.0)
        if np.isfinite(cand_norm) and cand_norm > float(tol) * ref_scale:
            out.append(cand)
    return out


def _build_first_newton_warm_start_basis(
    *,
    first_newton_correction_free_hist: list[np.ndarray],
    continuation_increment_free_hist: list[np.ndarray],
    max_corrections: int = 2,
) -> tuple[list[np.ndarray], int]:
    vectors: list[np.ndarray] = []
    if first_newton_correction_free_hist:
        for vec in first_newton_correction_free_hist[-max(int(max_corrections), 1) :]:
            arr = np.asarray(vec, dtype=np.float64).reshape(-1)
            if arr.size:
                vectors.append(arr.copy())
    if continuation_increment_free_hist:
        anchor = np.asarray(continuation_increment_free_hist[-1], dtype=np.float64).reshape(-1)
        if anchor.size:
            vectors.append(anchor.copy())
    basis = _orthonormalize_free_basis(vectors)
    if basis is None or basis.size == 0:
        return [], 0
    return [np.asarray(basis[:, j], dtype=np.float64).reshape(-1).copy() for j in range(int(basis.shape[1]))], int(basis.shape[1])


def _predictor_free_residual(
    *,
    U: np.ndarray,
    lambda_value: float,
    Q: np.ndarray,
    f: np.ndarray,
    constitutive_matrix_builder,
) -> tuple[np.ndarray, float]:
    f_free = _free(np.asarray(f, dtype=np.float64), Q)
    norm_f = float(np.linalg.norm(f_free))
    if norm_f <= 1.0e-30:
        norm_f = 1.0
    build_F_all_free = getattr(constitutive_matrix_builder, "build_F_all_free", None)
    if callable(build_F_all_free):
        F_free = np.asarray(build_F_all_free(float(lambda_value), np.asarray(U, dtype=np.float64)), dtype=np.float64).reshape(-1)
    else:
        F_all = np.asarray(constitutive_matrix_builder.build_F_all(float(lambda_value), np.asarray(U, dtype=np.float64)), dtype=np.float64)
        F_free = _free(F_all, Q)
    residual = np.asarray(F_free - f_free, dtype=np.float64).reshape(-1)
    return residual, float(norm_f)


def _refine_lambda_for_fixed_u_gauss_newton(
    *,
    U: np.ndarray,
    omega_old: float,
    omega: float,
    omega_target: float,
    lambda_old: float,
    lambda_value: float,
    Q: np.ndarray,
    f: np.ndarray,
    constitutive_matrix_builder,
    extra_lambda_candidates: Sequence[float] | None = None,
) -> tuple[float, float, int]:
    alpha_sec = _secant_alpha(omega_old=float(omega_old), omega=float(omega), omega_target=float(omega_target))
    lambda_center = float(lambda_value + alpha_sec * (float(lambda_value) - float(lambda_old)))
    candidate_values = [float(lambda_old), float(lambda_value), float(lambda_center)]
    if extra_lambda_candidates is not None:
        candidate_values.extend(float(v) for v in extra_lambda_candidates)
    lam_candidates = np.asarray(candidate_values, dtype=np.float64)
    lam_pos = lam_candidates[np.isfinite(lam_candidates) & (lam_candidates > 1.0e-8)]
    if lam_pos.size == 0:
        return max(float(lambda_value), 1.0e-6), np.inf, 0
    lam_lo = max(1.0e-6, 0.5 * float(np.min(lam_pos)))
    lam_hi = max(lam_lo * 1.01, 2.0 * float(np.max(lam_pos)))
    candidate_unique = np.unique(np.clip(lam_pos, lam_lo, lam_hi))
    eval_count = 0
    lam0 = None
    r0 = None
    norm_f = 1.0
    merit0 = np.inf
    for lam in candidate_unique:
        residual, residual_norm_f = _predictor_free_residual(
            U=np.asarray(U, dtype=np.float64),
            lambda_value=float(lam),
            Q=Q,
            f=f,
            constitutive_matrix_builder=constitutive_matrix_builder,
        )
        eval_count += 1
        merit = float(np.dot(residual, residual) / max(residual_norm_f * residual_norm_f, 1.0))
        if np.isfinite(merit) and merit < merit0:
            lam0 = float(lam)
            r0 = np.asarray(residual, dtype=np.float64)
            norm_f = float(residual_norm_f)
            merit0 = float(merit)
    if lam0 is None or r0 is None:
        return max(float(lambda_value), 1.0e-6), np.inf, eval_count
    fd_step = max(1.0e-6, 1.0e-3 * max(abs(lam0), 1.0))
    lam_fd = float(min(lam_hi, lam0 + fd_step))
    if abs(lam_fd - lam0) <= 1.0e-12:
        return float(lam0), float(merit0), eval_count
    r_fd, _ = _predictor_free_residual(
        U=np.asarray(U, dtype=np.float64),
        lambda_value=float(lam_fd),
        Q=Q,
        f=f,
        constitutive_matrix_builder=constitutive_matrix_builder,
    )
    eval_count += 1
    g = (r_fd - r0) / float(lam_fd - lam0)
    denom = float(np.dot(g, g))
    if not np.isfinite(denom) or denom <= 1.0e-30:
        return float(lam0), float(merit0), eval_count
    delta_lambda = -float(np.dot(g, r0)) / denom
    lam1 = float(np.clip(lam0 + delta_lambda, lam_lo, lam_hi))
    if abs(lam1 - lam0) <= 1.0e-12:
        return float(lam0), float(merit0), eval_count
    r1, _ = _predictor_free_residual(
        U=np.asarray(U, dtype=np.float64),
        lambda_value=float(lam1),
        Q=Q,
        f=f,
        constitutive_matrix_builder=constitutive_matrix_builder,
    )
    eval_count += 1
    merit1 = float(np.dot(r1, r1) / max(norm_f * norm_f, 1.0))
    if np.isfinite(merit1) and merit1 < merit0:
        return float(lam1), float(merit1), eval_count
    return float(lam0), float(merit0), eval_count


def _constraint_nullspace_basis(constraints: np.ndarray, *, rtol: float = 1.0e-12) -> np.ndarray:
    C = np.asarray(constraints, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    if C.ndim != 2:
        raise ValueError("Expected a 1D or 2D array of linear constraints.")
    if C.shape[1] == 0:
        raise ValueError("Expected at least one coefficient in the constraint matrix.")
    if not np.all(np.isfinite(C)):
        raise ValueError("Expected finite entries in the constraint matrix.")
    _, s, vh = np.linalg.svd(C, full_matrices=True)
    if s.size == 0:
        rank = 0
    else:
        tol = float(max(C.shape) * max(float(s[0]), 1.0) * float(rtol))
        rank = int(np.sum(s > tol))
    basis = np.asarray(vh[rank:, :].T, dtype=np.float64)
    if basis.shape[0] != C.shape[1]:
        raise ValueError("Unexpected nullspace basis row count.")
    return basis


def _secant_alpha(*, omega_old: float, omega: float, omega_target: float) -> float:
    denom = float(omega - omega_old)
    if abs(denom) <= 1.0e-30:
        return 0.0
    return float((float(omega_target) - float(omega)) / denom)


def _secant_predictor(
    *,
    omega_old: float,
    omega: float,
    omega_target: float,
    U_old: np.ndarray,
    U: np.ndarray,
    lambda_value: float,
) -> tuple[np.ndarray, float, str]:
    alpha = _secant_alpha(omega_old=omega_old, omega=omega, omega_target=omega_target)
    denom = float(omega - omega_old)
    if abs(denom) <= 1.0e-30:
        return np.asarray(U, dtype=np.float64), float(lambda_value), "secant_zero_span"
    U_ini = np.asarray(U, dtype=np.float64) + float(alpha) * (
        np.asarray(U, dtype=np.float64) - np.asarray(U_old, dtype=np.float64)
    )
    return U_ini, float(lambda_value), "secant"


def _increment_power_vectors(
    *,
    U_old: np.ndarray,
    U: np.ndarray,
    Q: np.ndarray,
    power_order: int,
) -> tuple[np.ndarray, float]:
    U_prev = np.asarray(U_old, dtype=np.float64)
    U_curr = np.asarray(U, dtype=np.float64)
    field_shape = U_curr.shape
    delta_flat = (U_curr - U_prev).reshape(-1, order="F")
    free_idx = _free_indices(np.asarray(Q, dtype=bool))
    delta_free = np.asarray(delta_flat[free_idx], dtype=np.float64).reshape(-1)
    scale = float(np.max(np.abs(delta_free))) if delta_free.size else 0.0
    if not np.isfinite(scale) or scale <= 1.0e-30:
        return np.zeros((delta_flat.size, 0), dtype=np.float64), 0.0
    abs_delta = np.abs(delta_flat)
    sign_delta = np.sign(delta_flat)
    cols: list[np.ndarray] = []
    for power in range(1, int(power_order) + 1):
        if power == 1:
            vec_flat = np.asarray(delta_flat, dtype=np.float64)
        else:
            vec_flat = np.asarray(sign_delta * (abs_delta**power) / (scale ** (power - 1)), dtype=np.float64)
        cols.append(vec_flat.reshape(field_shape, order="F").reshape(-1, order="F"))
    return np.column_stack(cols), scale


def _increment_power_window_vectors(
    *,
    continuation_state_hist: list[np.ndarray],
    Q: np.ndarray,
    power_order: int,
    increment_window_size: int,
) -> tuple[np.ndarray, list[float], int]:
    history = [np.asarray(v, dtype=np.float64).copy() for v in continuation_state_hist]
    hist_len = len(history)
    window_n = max(int(increment_window_size), 1)
    if hist_len < window_n + 1:
        return np.zeros((0, 0), dtype=np.float64), [], hist_len
    states = history[-(window_n + 1) :]
    cols: list[np.ndarray] = []
    scales: list[float] = []
    for idx in range(len(states) - 1, 0, -1):
        V_inc, scale = _increment_power_vectors(
            U_old=np.asarray(states[idx - 1], dtype=np.float64),
            U=np.asarray(states[idx], dtype=np.float64),
            Q=np.asarray(Q, dtype=bool),
            power_order=int(power_order),
        )
        if V_inc.shape[1] == 0:
            return np.zeros((0, 0), dtype=np.float64), scales, hist_len
        cols.append(np.asarray(V_inc, dtype=np.float64))
        scales.append(float(scale))
    return np.column_stack(cols), scales, hist_len


def _increment_power_reduced_newton_predictor(
    *,
    predictor_label: str,
    omega_old: float,
    omega: float,
    omega_target: float,
    U_old: np.ndarray,
    U: np.ndarray,
    lambda_value: float,
    Q: np.ndarray,
    f: np.ndarray,
    K_elast,
    constitutive_matrix_builder,
    it_damp_max: int,
    tol: float,
    projected_tolerance: float | None,
    r_min: float,
    power_order: int = 3,
    init_strategy: str = "secant",
    continuation_state_hist: list[np.ndarray] | None = None,
    increment_window_size: int = 1,
    max_projected_newton_iterations: int = 25,
    use_partial_result_on_nonconvergence: bool = False,
) -> tuple[np.ndarray, float, str, dict[str, float | bool | str | None]]:
    U_sec, lambda_ini, _ = _secant_predictor(
        omega_old=omega_old,
        omega=omega,
        omega_target=omega_target,
        U_old=U_old,
        U=U,
        lambda_value=lambda_value,
    )
    info = _predictor_info_defaults()
    t0 = perf_counter()
    alpha_sec = _secant_alpha(omega_old=omega_old, omega=omega, omega_target=omega_target)
    info["predictor_alpha"] = float(alpha_sec)
    info["predictor_beta"] = float(power_order)

    Q = np.asarray(Q, dtype=bool)
    field_shape = np.asarray(U_sec, dtype=np.float64).shape
    full_size = int(np.asarray(U_sec, dtype=np.float64).size)
    free_idx = _free_indices(Q)
    f_free = _free(f, Q)
    norm_f = max(float(np.linalg.norm(f_free)), 1.0)
    projected_tol = float(tol) if projected_tolerance is None else max(float(projected_tolerance), 0.0)
    eps = max(float(projected_tol) / 1000.0, 1.0e-12)

    window_n = max(int(increment_window_size), 1)
    if window_n <= 1:
        V_full, scales = _increment_power_vectors(
            U_old=np.asarray(U_old, dtype=np.float64),
            U=np.asarray(U, dtype=np.float64),
            Q=Q,
            power_order=int(power_order),
        )
        V_full = np.asarray(V_full, dtype=np.float64)
        scales_list = [float(scales)] if np.size(scales) else []
    else:
        if continuation_state_hist is None:
            info["fallback_used"] = True
            info["fallback_error"] = "missing_continuation_state_history_for_power_window"
            info["predictor_wall_time"] = float(perf_counter() - t0)
            return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info
        V_full, scales_list, hist_len = _increment_power_window_vectors(
            continuation_state_hist=continuation_state_hist,
            Q=Q,
            power_order=int(power_order),
            increment_window_size=window_n,
        )
        if V_full.size == 0:
            info["fallback_used"] = True
            info["fallback_error"] = f"insufficient_continuation_state_history:{hist_len}"
            info["predictor_wall_time"] = float(perf_counter() - t0)
            return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info
        V_full = np.asarray(V_full, dtype=np.float64)
    if V_full.shape[1] == 0:
        info["fallback_used"] = True
        info["fallback_error"] = "zero_increment_power_basis"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

    omega_coeffs = np.asarray([_free_dot(f, V_full[:, j].reshape(field_shape, order="F"), Q) for j in range(V_full.shape[1])], dtype=np.float64)
    coeff_tangent = _constraint_nullspace_basis(omega_coeffs.reshape(1, -1))
    if coeff_tangent.shape[1] == 0:
        info["fallback_used"] = True
        info["fallback_error"] = "power_basis_constraint_rank_zero"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

    if str(init_strategy).strip().lower() == "equal_split":
        c_raw = np.full(V_full.shape[1], float(alpha_sec) / max(V_full.shape[1], 1), dtype=np.float64)
    else:
        c_raw = np.zeros(V_full.shape[1], dtype=np.float64)
        c_raw[0] = float(alpha_sec)
    rhs_omega = float(omega_target) - float(omega)
    denom_omega = float(np.dot(omega_coeffs, omega_coeffs))
    if not np.isfinite(denom_omega) or denom_omega <= 1.0e-30:
        info["fallback_used"] = True
        info["fallback_error"] = "power_basis_omega_projection_degenerate"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info
    c_ref = np.asarray(c_raw - omega_coeffs * ((float(np.dot(omega_coeffs, c_raw)) - rhs_omega) / denom_omega), dtype=np.float64)
    info["state_coefficients_ref"] = c_ref.tolist()
    info["state_coefficients"] = c_ref.tolist()
    info["state_coefficient_sum"] = float(np.sum(c_ref))
    info["predictor_gamma"] = float(max(scales_list)) if scales_list else np.nan

    B_full = np.asarray(V_full @ coeff_tangent, dtype=np.float64)
    B_free = np.asarray(B_full[free_idx, :], dtype=np.float64)
    if B_free.size == 0:
        info["fallback_used"] = True
        info["fallback_error"] = "empty_power_basis"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

    basis_rank = int(np.linalg.matrix_rank(B_free))
    if basis_rank < B_free.shape[1]:
        info["fallback_used"] = True
        info["fallback_error"] = f"power_basis_rank_deficient:{basis_rank}/{B_free.shape[1]}"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

    info["basis_dim"] = float(B_free.shape[1])
    info["basis_condition"] = float(np.linalg.cond(B_free))

    U_base = np.asarray(U, dtype=np.float64).reshape(-1, order="F")
    ref_flat = np.asarray(U_base + V_full @ c_ref, dtype=np.float64)
    U_it = ref_flat.reshape(field_shape, order="F")
    lambda_it = float(lambda_ini)
    c_it = np.asarray(c_ref, dtype=np.float64).copy()
    r = float(r_min)
    compute_diffs = True
    projected_rel = np.nan
    omega_rel = np.nan
    dU_free_last = np.zeros_like(f_free)
    converged = False

    for iteration in range(1, int(max_projected_newton_iterations) + 1):
        F_full = None
        K_tangent = None
        K_r = None
        try:
            if compute_diffs:
                F_full, K_tangent = constitutive_matrix_builder.build_F_K_tangent_all(float(lambda_it), np.asarray(U_it, dtype=np.float64))
            else:
                F_full = constitutive_matrix_builder.build_F_all(float(lambda_it), np.asarray(U_it, dtype=np.float64))
            F_free = _free(F_full, Q)
            residual_free = np.asarray(F_free - f_free, dtype=np.float64).reshape(-1)
            criterion = float(np.linalg.norm(residual_free))
            projected_residual = np.asarray(B_free.T @ residual_free, dtype=np.float64)
            projected_rel = float(np.linalg.norm(projected_residual) / norm_f)
            omega_now = _free_dot(f, U_it, Q)
            omega_rel = abs(float(omega_now - float(omega_target))) / max(abs(float(omega_target)), 1.0)

            if compute_diffs and projected_rel <= float(projected_tol) and omega_rel <= float(projected_tol):
                converged = True
                info["reduced_newton_iterations"] = float(max(0, iteration - 1))
                break

            if K_tangent is None:
                _, K_tangent = constitutive_matrix_builder.build_F_K_tangent_all(float(lambda_it), np.asarray(U_it, dtype=np.float64))
            K_r = _combine_matrices(float(r), K_elast, 1.0 - float(r), K_tangent)

            F_eps_full = constitutive_matrix_builder.build_F_all(float(lambda_it) + eps, np.asarray(U_it, dtype=np.float64))
            G_free = (_free(F_eps_full, Q) - F_free) / eps

            KB = np.column_stack([_operator_matvec(K_r, B_full[:, j])[free_idx] for j in range(B_full.shape[1])])
            A_red = np.asarray(B_free.T @ KB, dtype=np.float64)

            coeff_w = _dense_reduced_solve(A_red, -(B_free.T @ np.asarray(G_free, dtype=np.float64).reshape(-1)))
            coeff_v = _dense_reduced_solve(A_red, B_free.T @ np.asarray(f_free - F_free, dtype=np.float64).reshape(-1))
            dW_free = np.asarray(B_free @ coeff_w, dtype=np.float64).reshape(-1)
            dV_free = np.asarray(B_free @ coeff_v, dtype=np.float64).reshape(-1)

            dW_full = np.zeros(full_size, dtype=np.float64)
            dV_full = np.zeros(full_size, dtype=np.float64)
            dW_full[free_idx] = dW_free
            dV_full[free_idx] = dV_free
            W = dW_full.reshape(field_shape, order="F")
            V = dV_full.reshape(field_shape, order="F")

            denom = float(np.dot(f_free, dW_free))
            d_l = 0.0 if abs(denom) < 1.0e-30 else -float(np.dot(f_free, dV_free)) / denom
            delta_red = np.asarray(coeff_v + d_l * coeff_w, dtype=np.float64)
            d_c = np.asarray(coeff_tangent @ delta_red, dtype=np.float64)
            dU_free_last = np.asarray(dV_free + d_l * dW_free, dtype=np.float64)
            d_U = (dV_full + d_l * dW_full).reshape(field_shape, order="F")

            alpha = float(
                damping_alg5(
                    int(it_damp_max),
                    np.asarray(U_it, dtype=np.float64),
                    float(lambda_it),
                    d_U,
                    float(d_l),
                    np.asarray(f, dtype=np.float64),
                    float(criterion),
                    Q,
                    constitutive_matrix_builder,
                    f_free=f_free,
                )
            )
            info["predictor_alpha"] = float(alpha)

            compute_diffs = True
            if alpha < 1.0e-1:
                if alpha == 0.0:
                    compute_diffs = False
                    r *= 2.0
                else:
                    r *= 2.0 ** 0.25
            else:
                if alpha > 0.5:
                    r = max(r / np.sqrt(2.0), float(r_min))

            if alpha == 0.0 and r > 1.0:
                info["state_coefficients"] = c_it.tolist()
                info["state_coefficient_sum"] = float(np.sum(c_it))
                info["predictor_wall_time"] = float(perf_counter() - t0)
                info["reduced_newton_iterations"] = float(iteration)
                info["reduced_gmres_iterations"] = 0.0
                info["reduced_projected_residual"] = float(projected_rel)
                info["reduced_omega_residual"] = float(omega_rel)
                if use_partial_result_on_nonconvergence:
                    info["fallback_used"] = False
                    info["fallback_error"] = "power_projected_newton_stalled_partial"
                    return np.asarray(U_it, dtype=np.float64), float(lambda_it), f"{predictor_label}_partial", info
                info["fallback_used"] = True
                info["fallback_error"] = "power_projected_newton_stalled"
                return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

            c_it = np.asarray(c_it + float(alpha) * d_c, dtype=np.float64)
            info["state_coefficients"] = c_it.tolist()
            info["state_coefficient_sum"] = float(np.sum(c_it))
            U_it = np.asarray((U_base + V_full @ c_it).reshape(field_shape, order="F"), dtype=np.float64)
            lambda_it = float(lambda_it + float(alpha) * float(d_l))
        except Exception as exc:
            info["state_coefficients"] = c_it.tolist()
            info["state_coefficient_sum"] = float(np.sum(c_it))
            info["fallback_used"] = True
            info["fallback_error"] = repr(exc)
            info["predictor_wall_time"] = float(perf_counter() - t0)
            return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info
        finally:
            if K_r is not None and not _is_builder_cached_matrix(K_r, constitutive_matrix_builder):
                _destroy_petsc_mat(K_r)
            if K_tangent is not None and not _is_builder_cached_matrix(K_tangent, constitutive_matrix_builder):
                _destroy_petsc_mat(K_tangent)

        info["reduced_newton_iterations"] = float(iteration)

    if not converged:
        info["state_coefficients"] = c_it.tolist()
        info["state_coefficient_sum"] = float(np.sum(c_it))
        info["predictor_wall_time"] = float(perf_counter() - t0)
        info["reduced_gmres_iterations"] = 0.0
        info["reduced_projected_residual"] = float(projected_rel)
        info["reduced_omega_residual"] = float(omega_rel)
        if use_partial_result_on_nonconvergence:
            info["fallback_used"] = False
            info["fallback_error"] = "power_projected_newton_not_converged_partial"
            return np.asarray(U_it, dtype=np.float64), float(lambda_it), f"{predictor_label}_partial", info
        info["fallback_used"] = True
        info["fallback_error"] = "power_projected_newton_not_converged"
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

    info["predictor_wall_time"] = float(perf_counter() - t0)
    info["projected_delta_lambda"] = float(lambda_it - float(lambda_ini))
    info["projected_correction_norm"] = float(np.linalg.norm(dU_free_last))
    info["reduced_gmres_iterations"] = 0.0
    info["reduced_projected_residual"] = float(projected_rel)
    info["reduced_omega_residual"] = float(omega_rel)
    return np.asarray(U_it, dtype=np.float64), float(lambda_it), predictor_label, info


def _orthonormalize_free_basis(vectors: list[np.ndarray], *, tol: float = 1.0e-10) -> np.ndarray | None:
    if not vectors:
        return None
    cols = [np.asarray(v, dtype=np.float64).reshape(-1) for v in vectors]
    V = np.column_stack(cols)
    if V.size == 0:
        return None
    Qmat, Rmat = np.linalg.qr(V, mode="reduced")
    if Rmat.size == 0:
        return None
    diag = np.abs(np.diag(Rmat))
    scale = float(np.max(diag)) if diag.size else 0.0
    if scale <= 0.0:
        return None
    keep = diag > tol * scale
    if not np.any(keep):
        return None
    return np.asarray(Qmat[:, keep], dtype=np.float64)


def _operator_matvec(A, x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if PETSc is not None and isinstance(A, PETSc.Mat):
        comm = A.getComm()
        row0, row1 = A.getOwnershipRange()
        try:
            col0, col1 = A.getOwnershipRangeColumn()
        except Exception:
            col0, col1 = row0, row1
        x_local = np.array(x_arr[int(col0) : int(col1)], dtype=np.float64, copy=True)
        x_vec = PETSc.Vec().createWithArray(
            x_local,
            size=(int(col1) - int(col0), x_arr.size),
            comm=comm,
        )
        n_rows, _n_cols = A.getSize()
        y_vec = PETSc.Vec().createMPI(
            size=(int(row1) - int(row0), int(n_rows)),
            comm=comm,
        )
        try:
            A.mult(x_vec, y_vec)
            return np.asarray(petsc_vec_to_global_array(y_vec), dtype=np.float64).reshape(-1)
        finally:
            x_vec.destroy()
            y_vec.destroy()
    if hasattr(A, "__matmul__"):
        return np.asarray(A @ x_arr, dtype=np.float64).reshape(-1)
    return np.asarray(A.dot(x_arr), dtype=np.float64).reshape(-1)


def _projected_ssr_reduced_correction(
    *,
    basis_free: np.ndarray,
    K_free,
    residual_free: np.ndarray,
    G_free: np.ndarray,
    f_free: np.ndarray,
) -> tuple[np.ndarray, float]:
    basis = np.asarray(basis_free, dtype=np.float64)
    if basis.ndim == 1:
        basis = basis[:, np.newaxis]
    if basis.size == 0 or basis.shape[1] == 0:
        return np.zeros_like(np.asarray(residual_free, dtype=np.float64).reshape(-1)), 0.0

    KB = np.column_stack([_operator_matvec(K_free, basis[:, j]) for j in range(basis.shape[1])])
    A_red = basis.T @ KB
    g_red = basis.T @ np.asarray(G_free, dtype=np.float64).reshape(-1)
    c_red = basis.T @ np.asarray(f_free, dtype=np.float64).reshape(-1)
    rhs_red = np.concatenate(
        [
            basis.T @ np.asarray(residual_free, dtype=np.float64).reshape(-1),
            np.zeros(1, dtype=np.float64),
        ]
    )
    saddle = np.zeros((basis.shape[1] + 1, basis.shape[1] + 1), dtype=np.float64)
    saddle[: basis.shape[1], : basis.shape[1]] = A_red
    saddle[: basis.shape[1], basis.shape[1]] = g_red
    saddle[basis.shape[1], : basis.shape[1]] = c_red
    sol, *_ = np.linalg.lstsq(saddle, rhs_red, rcond=None)
    coeff = np.asarray(sol[: basis.shape[1]], dtype=np.float64)
    d_l = float(sol[basis.shape[1]])
    return np.asarray(basis @ coeff, dtype=np.float64).reshape(-1), d_l


def _dense_reduced_solve(A_red: np.ndarray, rhs_red: np.ndarray) -> np.ndarray:
    A = np.asarray(A_red, dtype=np.float64)
    b = np.asarray(rhs_red, dtype=np.float64)
    if _scipy_lu_factor is not None and _scipy_lu_solve is not None:
        lu, piv = _scipy_lu_factor(A, check_finite=False)
        return np.asarray(_scipy_lu_solve((lu, piv), b, check_finite=False), dtype=np.float64)
    try:
        return np.asarray(np.linalg.solve(A, b), dtype=np.float64)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return np.asarray(sol, dtype=np.float64)


def _orthonormalize_free_basis_with_condition(
    vectors: list[np.ndarray],
    *,
    tol: float = 1.0e-10,
) -> tuple[np.ndarray | None, float]:
    if not vectors:
        return None, np.nan
    cols = [np.asarray(v, dtype=np.float64).reshape(-1) for v in vectors]
    V = np.column_stack(cols)
    if V.size == 0:
        return None, np.nan
    singular = np.linalg.svd(V, compute_uv=False)
    cond = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0.0 else np.inf
    Qmat, Rmat = np.linalg.qr(V, mode="reduced")
    if Rmat.size == 0:
        return None, cond
    diag = np.abs(np.diag(Rmat))
    scale = float(np.max(diag)) if diag.size else 0.0
    if scale <= 0.0:
        return None, cond
    keep = diag > tol * scale
    if not np.any(keep):
        return None, cond
    return np.asarray(Qmat[:, keep], dtype=np.float64), cond


def _select_continuation_increment_basis(
    continuation_increment_free_hist: list[np.ndarray],
    *,
    window_size: int | None,
    min_history: int,
) -> tuple[list[np.ndarray], int]:
    history = [np.asarray(v, dtype=np.float64).reshape(-1) for v in continuation_increment_free_hist]
    hist_len = len(history)
    if hist_len < int(min_history):
        return [], hist_len
    if window_size is None:
        return history, hist_len
    window_n = max(int(window_size), 1)
    if hist_len < window_n:
        return [], hist_len
    return history[-window_n:], hist_len


def _select_continuation_state_history(
    continuation_state_hist: list[np.ndarray],
    *,
    window_size: int | None,
    min_history: int,
) -> tuple[list[np.ndarray], int]:
    history = [np.asarray(v, dtype=np.float64).copy() for v in continuation_state_hist]
    hist_len = len(history)
    if hist_len < int(min_history):
        return [], hist_len
    if window_size is None:
        return history, hist_len
    window_n = max(int(window_size), 1)
    if hist_len < window_n:
        return [], hist_len
    return history[-window_n:], hist_len


def _secant_orthogonal_increment_ls_predictor(
    *,
    omega_old: float,
    omega: float,
    omega_target: float,
    U_old: np.ndarray,
    U: np.ndarray,
    lambda_value: float,
    Q: np.ndarray,
    f: np.ndarray,
    K_elast,
    constitutive_matrix_builder,
    continuation_increment_free_hist: list[np.ndarray],
    r_min: float,
    trust_region_scale: float = 0.15,
    min_predicted_decrease: float = 1.0e-3,
) -> tuple[np.ndarray, float, str, dict[str, float | bool | str | None]]:
    U_sec, lambda_ini, _ = _secant_predictor(
        omega_old=omega_old,
        omega=omega,
        omega_target=omega_target,
        U_old=U_old,
        U=U,
        lambda_value=lambda_value,
    )
    info = _predictor_info_defaults()
    t0 = perf_counter()
    alpha_sec = _secant_alpha(omega_old=omega_old, omega=omega, omega_target=omega_target)
    info["predictor_alpha"] = float(alpha_sec)

    history = [np.asarray(v, dtype=np.float64).reshape(-1) for v in continuation_increment_free_hist]
    if not history:
        info["fallback_used"] = True
        info["fallback_error"] = "insufficient_continuation_increment_history:0"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), "secant_orthogonal_increment_ls_fallback_secant", info
    candidate_vectors = history[-2:]
    secant_increment_free = _free(np.asarray(U_sec, dtype=np.float64) - np.asarray(U, dtype=np.float64), np.asarray(Q, dtype=bool))
    orthogonal_candidates = _orthogonalize_vectors_against_direction(candidate_vectors, secant_increment_free)
    basis_free, basis_cond = _orthonormalize_free_basis_with_condition(orthogonal_candidates)
    if basis_free is None or basis_free.shape[1] == 0:
        info["fallback_used"] = True
        info["fallback_error"] = "empty_secant_orthogonal_increment_basis"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), "secant_orthogonal_increment_ls_fallback_secant", info

    Q = np.asarray(Q, dtype=bool)
    field_shape = np.asarray(U_sec, dtype=np.float64).shape
    free_idx = _free_indices(Q)
    full_size = int(np.asarray(U_sec, dtype=np.float64).size)
    basis_dim = int(basis_free.shape[1])
    basis_full = np.zeros((full_size, basis_dim), dtype=np.float64)
    basis_full[free_idx, :] = basis_free

    f_free = _free(f, Q)
    norm_f_sq = max(float(np.dot(f_free, f_free)), 1.0)
    K_tangent = None
    K_r = None
    try:
        F_full, K_tangent = constitutive_matrix_builder.build_F_K_tangent_all(float(lambda_ini), np.asarray(U_sec, dtype=np.float64))
        residual_free = np.asarray(_free(F_full, Q) - f_free, dtype=np.float64).reshape(-1)
        base_residual_sq = float(np.dot(residual_free, residual_free))
        K_r = _combine_matrices(float(r_min), K_elast, 1.0 - float(r_min), K_tangent)
        KB = np.column_stack([_operator_matvec(K_r, basis_full[:, j])[free_idx] for j in range(basis_dim)])
        gram = np.asarray(KB.T @ KB, dtype=np.float64)
        gram_scale = max(float(np.linalg.norm(gram, ord=2)) if gram.size else 0.0, 1.0)
        rhs = -np.asarray(KB.T @ residual_free, dtype=np.float64)
        coeff = _dense_reduced_solve(gram + 1.0e-10 * gram_scale * np.eye(basis_dim, dtype=np.float64), rhs)
        correction_free = np.asarray(basis_free @ coeff, dtype=np.float64).reshape(-1)
        correction_norm = float(np.linalg.norm(correction_free))
        secant_norm = float(np.linalg.norm(secant_increment_free))
        trust_radius = float(max(trust_region_scale, 0.0) * secant_norm)
        clipped = False
        if trust_radius > 0.0 and correction_norm > trust_radius:
            scale = trust_radius / max(correction_norm, 1.0e-30)
            coeff = np.asarray(coeff * scale, dtype=np.float64)
            correction_free = np.asarray(correction_free * scale, dtype=np.float64)
            correction_norm = float(np.linalg.norm(correction_free))
            clipped = True
        predicted_residual = np.asarray(residual_free + KB @ coeff, dtype=np.float64).reshape(-1)
        predicted_residual_sq = float(np.dot(predicted_residual, predicted_residual))
        predicted_decrease = float((base_residual_sq - predicted_residual_sq) / max(base_residual_sq, 1.0))
        info["basis_dim"] = float(basis_dim)
        info["basis_condition"] = float(basis_cond)
        info["projected_correction_norm"] = float(correction_norm)
        info["energy_value"] = float(predicted_residual_sq / norm_f_sq)
        info["secant_correction_active"] = True
        info["secant_correction_basis_dim"] = float(basis_dim)
        info["secant_correction_trust_region_clipped"] = bool(clipped)
        info["secant_correction_predicted_residual_decrease"] = float(predicted_decrease)
        if not np.isfinite(predicted_decrease) or predicted_decrease <= float(min_predicted_decrease):
            info["fallback_used"] = True
            info["fallback_error"] = "weak_predicted_residual_decrease"
            info["predictor_wall_time"] = float(perf_counter() - t0)
            return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), "secant_orthogonal_increment_ls_fallback_secant", info
        U_corr_flat = np.asarray(U_sec, dtype=np.float64).reshape(-1, order="F").copy()
        U_corr_flat[free_idx] += correction_free
        U_corr = U_corr_flat.reshape(field_shape, order="F")
        U_corr = _rescale_to_target_omega(U_corr, float(omega_target), np.asarray(f, dtype=np.float64), Q)
    except Exception as exc:
        info["fallback_used"] = True
        info["fallback_error"] = repr(exc)
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), "secant_orthogonal_increment_ls_fallback_secant", info
    finally:
        if K_r is not None and not _is_builder_cached_matrix(K_r, constitutive_matrix_builder):
            _destroy_petsc_mat(K_r)
        if K_tangent is not None and not _is_builder_cached_matrix(K_tangent, constitutive_matrix_builder):
            _destroy_petsc_mat(K_tangent)

    info["predictor_wall_time"] = float(perf_counter() - t0)
    return np.asarray(U_corr, dtype=np.float64), float(lambda_ini), "secant_orthogonal_increment_ls", info


def _projected_reduced_newton_predictor(
    *,
    predictor_label: str,
    omega_old: float,
    omega: float,
    omega_target: float,
    U_old: np.ndarray,
    U: np.ndarray,
    lambda_value: float,
    Q: np.ndarray,
    f: np.ndarray,
    K_elast,
    constitutive_matrix_builder,
    it_damp_max: int,
    tol: float,
    projected_tolerance: float | None,
    r_min: float,
    continuation_increment_free_hist: list[np.ndarray],
    window_size: int | None,
    min_history: int = 3,
    max_projected_newton_iterations: int = 25,
) -> tuple[np.ndarray, float, str, dict[str, float | bool | str | None]]:
    U_sec, lambda_ini, _ = _secant_predictor(
        omega_old=omega_old,
        omega=omega,
        omega_target=omega_target,
        U_old=U_old,
        U=U,
        lambda_value=lambda_value,
    )
    info = _predictor_info_defaults()
    t0 = perf_counter()
    info["predictor_alpha"] = float(_secant_alpha(omega_old=omega_old, omega=omega, omega_target=omega_target))

    selected_vectors, hist_len = _select_continuation_increment_basis(
        continuation_increment_free_hist,
        window_size=window_size,
        min_history=min_history,
    )
    if not selected_vectors:
        info["fallback_used"] = True
        info["fallback_error"] = f"insufficient_continuation_history:{hist_len}"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

    basis_free, basis_cond = _orthonormalize_free_basis_with_condition(selected_vectors)
    if basis_free is None or basis_free.shape[1] == 0:
        info["fallback_used"] = True
        info["fallback_error"] = "basis_rank_zero"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

    Q = np.asarray(Q, dtype=bool)
    free_idx = _free_indices(Q)
    field_shape = np.asarray(U_sec, dtype=np.float64).shape
    full_size = int(np.asarray(U_sec, dtype=np.float64).size)
    basis_dim = int(basis_free.shape[1])
    basis_full = np.zeros((full_size, basis_dim), dtype=np.float64)
    basis_full[free_idx, :] = basis_free
    f_free = _free(f, Q)
    norm_f = max(float(np.linalg.norm(f_free)), 1.0)
    projected_tol = float(tol) if projected_tolerance is None else max(float(projected_tolerance), 0.0)
    eps = max(float(projected_tol) / 1000.0, 1.0e-12)

    info["basis_dim"] = float(basis_dim)
    info["basis_condition"] = float(basis_cond)

    U_it = np.asarray(U_sec, dtype=np.float64).copy()
    lambda_it = float(lambda_ini)
    r = float(r_min)
    compute_diffs = True
    projected_rel = np.nan
    omega_rel = np.nan
    dU_free_last = np.zeros_like(f_free)
    converged = False

    for iteration in range(1, int(max_projected_newton_iterations) + 1):
        F_full = None
        K_tangent = None
        K_r = None
        try:
            if compute_diffs:
                F_full, K_tangent = constitutive_matrix_builder.build_F_K_tangent_all(float(lambda_it), np.asarray(U_it, dtype=np.float64))
            else:
                F_full = constitutive_matrix_builder.build_F_all(float(lambda_it), np.asarray(U_it, dtype=np.float64))
            F_free = _free(F_full, Q)
            residual_free = np.asarray(F_free - f_free, dtype=np.float64).reshape(-1)
            criterion = float(np.linalg.norm(residual_free))
            projected_residual = np.asarray(basis_free.T @ residual_free, dtype=np.float64)
            projected_rel = float(np.linalg.norm(projected_residual) / norm_f)
            omega_now = _free_dot(f, U_it, Q)
            omega_rel = abs(float(omega_now - float(omega_target))) / max(abs(float(omega_target)), 1.0)

            if compute_diffs and projected_rel <= float(projected_tol) and omega_rel <= float(projected_tol):
                converged = True
                info["reduced_newton_iterations"] = float(max(0, iteration - 1))
                break

            if K_tangent is None:
                _, K_tangent = constitutive_matrix_builder.build_F_K_tangent_all(float(lambda_it), np.asarray(U_it, dtype=np.float64))
            K_r = _combine_matrices(float(r), K_elast, 1.0 - float(r), K_tangent)

            F_eps_full = constitutive_matrix_builder.build_F_all(float(lambda_it) + eps, np.asarray(U_it, dtype=np.float64))
            G_free = (_free(F_eps_full, Q) - F_free) / eps

            KB = np.column_stack([_operator_matvec(K_r, basis_full[:, j])[free_idx] for j in range(basis_dim)])
            A_red = np.asarray(basis_free.T @ KB, dtype=np.float64)

            coeff_w = _dense_reduced_solve(A_red, -(basis_free.T @ np.asarray(G_free, dtype=np.float64).reshape(-1)))
            coeff_v = _dense_reduced_solve(A_red, basis_free.T @ np.asarray(f_free - F_free, dtype=np.float64).reshape(-1))
            dW_free = np.asarray(basis_free @ coeff_w, dtype=np.float64).reshape(-1)
            dV_free = np.asarray(basis_free @ coeff_v, dtype=np.float64).reshape(-1)

            dW_full = np.zeros(full_size, dtype=np.float64)
            dV_full = np.zeros(full_size, dtype=np.float64)
            dW_full[free_idx] = dW_free
            dV_full[free_idx] = dV_free
            W = dW_full.reshape(field_shape, order="F")
            V = dV_full.reshape(field_shape, order="F")

            denom = float(np.dot(f_free, dW_free))
            d_l = 0.0 if abs(denom) < 1.0e-30 else -float(np.dot(f_free, dV_free)) / denom
            dU_free_last = np.asarray(dV_free + d_l * dW_free, dtype=np.float64)
            d_U = (dV_full + d_l * dW_full).reshape(field_shape, order="F")

            alpha = float(
                damping_alg5(
                    int(it_damp_max),
                    np.asarray(U_it, dtype=np.float64),
                    float(lambda_it),
                    d_U,
                    float(d_l),
                    np.asarray(f, dtype=np.float64),
                    float(criterion),
                    Q,
                    constitutive_matrix_builder,
                    f_free=f_free,
                )
            )
            info["predictor_alpha"] = float(alpha)

            compute_diffs = True
            if alpha < 1.0e-1:
                if alpha == 0.0:
                    compute_diffs = False
                    r *= 2.0
                else:
                    r *= 2.0 ** 0.25
            else:
                if alpha > 0.5:
                    r = max(r / np.sqrt(2.0), float(r_min))

            if alpha == 0.0 and r > 1.0:
                info["fallback_used"] = True
                info["fallback_error"] = "projected_newton_stalled"
                info["predictor_wall_time"] = float(perf_counter() - t0)
                info["reduced_newton_iterations"] = float(iteration)
                info["reduced_gmres_iterations"] = 0.0
                info["reduced_projected_residual"] = float(projected_rel)
                info["reduced_omega_residual"] = float(omega_rel)
                return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

            U_it = np.asarray(U_it, dtype=np.float64) + float(alpha) * d_U
            denom_omega = _free_dot(f, U_it, Q)
            if abs(float(denom_omega)) > 1.0e-30:
                U_it = U_it * (float(omega_target) / float(denom_omega))
            lambda_it = float(lambda_it + float(alpha) * float(d_l))
        except Exception as exc:
            info["fallback_used"] = True
            info["fallback_error"] = repr(exc)
            info["predictor_wall_time"] = float(perf_counter() - t0)
            return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info
        finally:
            if K_r is not None and not _is_builder_cached_matrix(K_r, constitutive_matrix_builder):
                _destroy_petsc_mat(K_r)
            if K_tangent is not None and not _is_builder_cached_matrix(K_tangent, constitutive_matrix_builder):
                _destroy_petsc_mat(K_tangent)

        info["reduced_newton_iterations"] = float(iteration)

    if not converged:
        info["fallback_used"] = True
        info["fallback_error"] = "projected_newton_not_converged"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        info["reduced_gmres_iterations"] = 0.0
        info["reduced_projected_residual"] = float(projected_rel)
        info["reduced_omega_residual"] = float(omega_rel)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

    info["predictor_wall_time"] = float(perf_counter() - t0)
    info["projected_delta_lambda"] = float(lambda_it - float(lambda_ini))
    info["projected_correction_norm"] = float(np.linalg.norm(dU_free_last))
    info["reduced_gmres_iterations"] = 0.0
    info["reduced_projected_residual"] = float(projected_rel)
    info["reduced_omega_residual"] = float(omega_rel)
    return np.asarray(U_it, dtype=np.float64), float(lambda_it), predictor_label, info


def _affine_state_reduced_newton_predictor(
    *,
    predictor_label: str,
    omega_old: float,
    omega: float,
    omega_target: float,
    U_old: np.ndarray,
    U: np.ndarray,
    lambda_value: float,
    Q: np.ndarray,
    f: np.ndarray,
    K_elast,
    constitutive_matrix_builder,
    it_damp_max: int,
    tol: float,
    projected_tolerance: float | None,
    r_min: float,
    continuation_state_hist: list[np.ndarray],
    window_size: int | None,
    min_history: int = 2,
    max_projected_newton_iterations: int = 25,
    use_partial_result_on_nonconvergence: bool = False,
) -> tuple[np.ndarray, float, str, dict[str, float | bool | str | None]]:
    U_sec, lambda_ini, _ = _secant_predictor(
        omega_old=omega_old,
        omega=omega,
        omega_target=omega_target,
        U_old=U_old,
        U=U,
        lambda_value=lambda_value,
    )
    info = _predictor_info_defaults()
    t0 = perf_counter()
    alpha_sec = _secant_alpha(omega_old=omega_old, omega=omega, omega_target=omega_target)
    info["predictor_alpha"] = float(alpha_sec)

    selected_states, hist_len = _select_continuation_state_history(
        continuation_state_hist,
        window_size=window_size,
        min_history=min_history,
    )
    if not selected_states:
        info["fallback_used"] = True
        info["fallback_error"] = f"insufficient_continuation_state_history:{hist_len}"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

    n_states = len(selected_states)
    field_shape = np.asarray(U_sec, dtype=np.float64).shape
    full_size = int(np.asarray(U_sec, dtype=np.float64).size)
    if n_states < 2:
        info["fallback_used"] = True
        info["fallback_error"] = "insufficient_state_count"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

    c_ref = np.zeros(n_states, dtype=np.float64)
    c_ref[-1] = 1.0 + float(alpha_sec)
    c_ref[-2] = -float(alpha_sec)
    info["state_coefficients_ref"] = c_ref.tolist()
    info["state_coefficients"] = c_ref.tolist()
    info["state_coefficient_sum"] = float(np.sum(c_ref))
    sum_row = np.ones((1, n_states), dtype=np.float64)
    coeff_tangent = _constraint_nullspace_basis(sum_row)
    if coeff_tangent.shape[1] == 0:
        info["fallback_used"] = True
        info["fallback_error"] = "affine_state_basis_rank_zero"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

    S_full = np.column_stack([np.asarray(v, dtype=np.float64).reshape(-1, order="F") for v in selected_states])
    ref_flat = np.asarray(S_full @ c_ref, dtype=np.float64).reshape(-1)
    U_ref = ref_flat.reshape(field_shape, order="F")
    free_idx = _free_indices(Q)
    B_full = np.asarray(S_full @ coeff_tangent, dtype=np.float64)
    B_free = np.asarray(B_full[free_idx, :], dtype=np.float64)
    if B_free.size == 0:
        info["fallback_used"] = True
        info["fallback_error"] = "empty_affine_state_basis"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

    basis_rank = int(np.linalg.matrix_rank(B_free))
    if basis_rank < B_free.shape[1]:
        info["fallback_used"] = True
        info["fallback_error"] = f"affine_state_basis_rank_deficient:{basis_rank}/{B_free.shape[1]}"
        info["predictor_wall_time"] = float(perf_counter() - t0)
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

    info["basis_dim"] = float(B_free.shape[1])
    info["basis_condition"] = float(np.linalg.cond(B_free))
    info["predictor_beta"] = float(n_states)
    info["predictor_gamma"] = float(np.sum(c_ref))

    f_free = _free(f, Q)
    norm_f = max(float(np.linalg.norm(f_free)), 1.0)
    projected_tol = float(tol) if projected_tolerance is None else max(float(projected_tolerance), 0.0)
    eps = max(float(projected_tol) / 1000.0, 1.0e-12)

    c_it = np.asarray(c_ref, dtype=np.float64).copy()
    U_it = np.asarray(U_ref, dtype=np.float64).copy()
    lambda_it = float(lambda_ini)
    r = float(r_min)
    compute_diffs = True
    projected_rel = np.nan
    omega_rel = np.nan
    dU_free_last = np.zeros_like(f_free)
    converged = False

    for iteration in range(1, int(max_projected_newton_iterations) + 1):
        F_full = None
        K_tangent = None
        K_r = None
        try:
            if compute_diffs:
                F_full, K_tangent = constitutive_matrix_builder.build_F_K_tangent_all(float(lambda_it), np.asarray(U_it, dtype=np.float64))
            else:
                F_full = constitutive_matrix_builder.build_F_all(float(lambda_it), np.asarray(U_it, dtype=np.float64))
            F_free = _free(F_full, Q)
            residual_free = np.asarray(F_free - f_free, dtype=np.float64).reshape(-1)
            criterion = float(np.linalg.norm(residual_free))
            projected_residual = np.asarray(B_free.T @ residual_free, dtype=np.float64)
            projected_rel = float(np.linalg.norm(projected_residual) / norm_f)
            omega_now = _free_dot(f, U_it, Q)
            omega_rel = abs(float(omega_now - float(omega_target))) / max(abs(float(omega_target)), 1.0)

            if compute_diffs and projected_rel <= float(projected_tol) and omega_rel <= float(projected_tol):
                converged = True
                info["reduced_newton_iterations"] = float(max(0, iteration - 1))
                break

            if K_tangent is None:
                _, K_tangent = constitutive_matrix_builder.build_F_K_tangent_all(float(lambda_it), np.asarray(U_it, dtype=np.float64))
            K_r = _combine_matrices(float(r), K_elast, 1.0 - float(r), K_tangent)

            F_eps_full = constitutive_matrix_builder.build_F_all(float(lambda_it) + eps, np.asarray(U_it, dtype=np.float64))
            G_free = (_free(F_eps_full, Q) - F_free) / eps

            KB = np.column_stack([_operator_matvec(K_r, B_full[:, j])[free_idx] for j in range(B_full.shape[1])])
            A_red = np.asarray(B_free.T @ KB, dtype=np.float64)

            coeff_w = _dense_reduced_solve(A_red, -(B_free.T @ np.asarray(G_free, dtype=np.float64).reshape(-1)))
            coeff_v = _dense_reduced_solve(A_red, B_free.T @ np.asarray(f_free - F_free, dtype=np.float64).reshape(-1))
            dW_free = np.asarray(B_free @ coeff_w, dtype=np.float64).reshape(-1)
            dV_free = np.asarray(B_free @ coeff_v, dtype=np.float64).reshape(-1)

            dW_full = np.zeros(full_size, dtype=np.float64)
            dV_full = np.zeros(full_size, dtype=np.float64)
            dW_full[free_idx] = dW_free
            dV_full[free_idx] = dV_free
            W = dW_full.reshape(field_shape, order="F")
            V = dV_full.reshape(field_shape, order="F")

            denom = float(np.dot(f_free, dW_free))
            d_l = 0.0 if abs(denom) < 1.0e-30 else -float(np.dot(f_free, dV_free)) / denom
            delta_red = np.asarray(coeff_v + d_l * coeff_w, dtype=np.float64)
            d_c = np.asarray(coeff_tangent @ delta_red, dtype=np.float64)
            dU_free_last = np.asarray(dV_free + d_l * dW_free, dtype=np.float64)
            d_U = (dV_full + d_l * dW_full).reshape(field_shape, order="F")

            alpha = float(
                damping_alg5(
                    int(it_damp_max),
                    np.asarray(U_it, dtype=np.float64),
                    float(lambda_it),
                    d_U,
                    float(d_l),
                    np.asarray(f, dtype=np.float64),
                    float(criterion),
                    Q,
                    constitutive_matrix_builder,
                    f_free=f_free,
                )
            )
            info["predictor_alpha"] = float(alpha)

            compute_diffs = True
            if alpha < 1.0e-1:
                if alpha == 0.0:
                    compute_diffs = False
                    r *= 2.0
                else:
                    r *= 2.0 ** 0.25
            else:
                if alpha > 0.5:
                    r = max(r / np.sqrt(2.0), float(r_min))

            if alpha == 0.0 and r > 1.0:
                info["state_coefficients"] = c_it.tolist()
                info["state_coefficient_sum"] = float(np.sum(c_it))
                info["predictor_wall_time"] = float(perf_counter() - t0)
                info["reduced_newton_iterations"] = float(iteration)
                info["reduced_gmres_iterations"] = 0.0
                info["reduced_projected_residual"] = float(projected_rel)
                info["reduced_omega_residual"] = float(omega_rel)
                if use_partial_result_on_nonconvergence:
                    info["fallback_used"] = False
                    info["fallback_error"] = "affine_state_projected_newton_stalled_partial"
                    return np.asarray(U_it, dtype=np.float64), float(lambda_it), f"{predictor_label}_partial", info
                info["fallback_used"] = True
                info["fallback_error"] = "affine_state_projected_newton_stalled"
                return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

            c_it = np.asarray(c_it + float(alpha) * d_c, dtype=np.float64)
            sum_c = float(np.sum(c_it))
            if np.isfinite(sum_c) and abs(sum_c) > 1.0e-30:
                c_it = c_it / sum_c
            info["state_coefficients"] = c_it.tolist()
            info["state_coefficient_sum"] = float(np.sum(c_it))
            U_it = np.asarray((S_full @ c_it).reshape(field_shape, order="F"), dtype=np.float64)
            lambda_it = float(lambda_it + float(alpha) * float(d_l))
        except Exception as exc:
            info["state_coefficients"] = c_it.tolist()
            info["state_coefficient_sum"] = float(np.sum(c_it))
            info["fallback_used"] = True
            info["fallback_error"] = repr(exc)
            info["predictor_wall_time"] = float(perf_counter() - t0)
            return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info
        finally:
            if K_r is not None and not _is_builder_cached_matrix(K_r, constitutive_matrix_builder):
                _destroy_petsc_mat(K_r)
            if K_tangent is not None and not _is_builder_cached_matrix(K_tangent, constitutive_matrix_builder):
                _destroy_petsc_mat(K_tangent)

        info["reduced_newton_iterations"] = float(iteration)

    if not converged:
        info["state_coefficients"] = c_it.tolist()
        info["state_coefficient_sum"] = float(np.sum(c_it))
        info["predictor_wall_time"] = float(perf_counter() - t0)
        info["reduced_gmres_iterations"] = 0.0
        info["reduced_projected_residual"] = float(projected_rel)
        info["reduced_omega_residual"] = float(omega_rel)
        if use_partial_result_on_nonconvergence:
            info["fallback_used"] = False
            info["fallback_error"] = "affine_state_projected_newton_not_converged_partial"
            return np.asarray(U_it, dtype=np.float64), float(lambda_it), f"{predictor_label}_partial", info
        info["fallback_used"] = True
        info["fallback_error"] = "affine_state_projected_newton_not_converged"
        return np.asarray(U_sec, dtype=np.float64), float(lambda_ini), f"{predictor_label}_fallback_secant", info

    info["predictor_wall_time"] = float(perf_counter() - t0)
    info["projected_delta_lambda"] = float(lambda_it - float(lambda_ini))
    info["projected_correction_norm"] = float(np.linalg.norm(dU_free_last))
    info["reduced_gmres_iterations"] = 0.0
    info["reduced_projected_residual"] = float(projected_rel)
    info["reduced_omega_residual"] = float(omega_rel)
    info["affine_state_count"] = float(n_states)
    info["affine_coefficient_sum"] = float(np.sum(c_it))
    info["state_coefficients"] = c_it.tolist()
    info["state_coefficient_sum"] = float(np.sum(c_it))
    return np.asarray(U_it, dtype=np.float64), float(lambda_it), predictor_label, info


def init_phase_SSR_indirect_continuation(
    lambda_init: float,
    d_lambda_init: float,
    d_lambda_min: float,
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
    newton_stopping_criterion: str = "relative_residual",
    newton_stopping_tol: float | None = None,
    init_newton_stopping_criterion: str | None = None,
    init_newton_stopping_tol: float | None = None,
):
    """Compute first two converged points for indirect continuation."""

    Q = np.asarray(Q, dtype=bool)
    n_nodes = Q.shape[1]
    dim = Q.shape[0]
    U_ini = np.zeros((dim, n_nodes), dtype=np.float64)

    all_newton_its: list[int] = []
    d_lambda = float(d_lambda_init)
    lambda1 = float(lambda_init)
    normalized_stop_mode = _normalize_stopping_criterion(newton_stopping_criterion)
    default_init_stop_criterion = (
        "relative_residual" if normalized_stop_mode == "absolute_delta_lambda" else str(newton_stopping_criterion)
    )
    default_init_stop_tol = None if normalized_stop_mode == "absolute_delta_lambda" else newton_stopping_tol
    init_stop_criterion = (
        default_init_stop_criterion
        if init_newton_stopping_criterion is None
        else str(init_newton_stopping_criterion)
    )
    init_stop_tol = default_init_stop_tol if init_newton_stopping_tol is None else init_newton_stopping_tol

    while True:
        constitutive_matrix_builder.reduction(lambda1)
        U_it, flag_N, it_newt = newton(
            U_ini,
            tol,
            it_newt_max,
            it_damp_max,
            r_min,
            K_elast,
            Q,
            f,
            constitutive_matrix_builder,
            linear_system_solver,
            stopping_criterion=str(init_stop_criterion),
            stopping_tol=init_stop_tol,
        )
        all_newton_its.append(it_newt)
        if flag_N == 0:
            break
        lambda1 *= 0.5
        d_lambda *= 0.5
        if d_lambda < d_lambda_min:
            raise RuntimeError("Initial choice of lambda seems to be too large.")

    U1 = U_it
    omega1 = _free_dot(f, U1, Q)

    if getattr(linear_system_solver, "supports_dynamic_deflation_basis", lambda: True)():
        linear_system_solver.expand_deflation_basis(_free(U_it, Q))

    while True:
        lambda_it = lambda1 + d_lambda
        constitutive_matrix_builder.reduction(lambda_it)

        U_it, flag_N, it_newt = newton(
            U1,
            tol,
            it_newt_max,
            it_damp_max,
            r_min,
            K_elast,
            Q,
            f,
            constitutive_matrix_builder,
            linear_system_solver,
            stopping_criterion=str(init_stop_criterion),
            stopping_tol=init_stop_tol,
        )
        all_newton_its.append(it_newt)

        if flag_N == 1:
            d_lambda /= 2.0
        else:
            U2 = U_it
            omega2 = _free_dot(f, U2, Q)
            lambda2 = lambda_it
            if (omega2 - omega1) / max(1.0, omega1) < 1e-5:
                U1 = U2
                lambda1 = lambda2
                omega1 = omega2
            else:
                break

        if d_lambda < d_lambda_min:
            raise RuntimeError("It seems that the FoS is equal to lambda_init.")
        if lambda1 > 10.0:
            raise RuntimeError("It seems that the FoS is greater than 10.")

    return U1, U2, omega1, omega2, lambda1, lambda2, all_newton_its


def _SSR_indirect_continuation_streaming_microstep(
    lambda_init: float,
    d_lambda_init: float,
    d_lambda_min: float,
    step_max: int,
    omega_max_stop: float,
    it_newt_max: int,
    it_damp_max: int,
    tol: float,
    r_min: float,
    K_elast,
    Q: np.ndarray,
    f: np.ndarray,
    constitutive_matrix_builder,
    linear_system_solver,
    progress_callback: Callable[[dict], None] | None = None,
    *,
    store_step_u: bool = True,
    step_guess_diagnostics: Callable[[np.ndarray, np.ndarray], dict[str, float]] | None = None,
    streaming_micro_target_length: float = 0.15,
    streaming_micro_min_length: float = 0.05,
    streaming_micro_max_length: float = 0.30,
    streaming_move_relres_threshold: float = 5.0e-3,
    streaming_alpha_advance_threshold: float = 0.5,
    streaming_micro_max_corrections: int = 40,
    streaming_basis_max_vectors: int = 8,
    newton_stopping_criterion: str = "relative_residual",
    newton_stopping_tol: float | None = None,
    init_newton_stopping_criterion: str | None = None,
    init_newton_stopping_tol: float | None = None,
):
    Q = np.asarray(Q, dtype=bool)

    def _emit(event: str, **payload) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": event,
                "continuation_kind": "ssr_indirect",
                "continuation_mode": "streaming_microstep",
                **payload,
            }
        )

    def _newton_progress_callback(
        *,
        target_step: int,
        accepted_steps: int,
        attempt_in_step: int,
        omega_target: float,
        lambda_before: float,
    ):
        if progress_callback is None:
            return None

        def _callback(event: dict) -> None:
            progress_callback(
                {
                    "continuation_kind": "ssr_indirect",
                    "continuation_mode": "streaming_microstep",
                    "phase": "continuation",
                    "target_step": int(target_step),
                    "accepted_steps": int(accepted_steps),
                    "attempt_in_step": int(attempt_in_step),
                    "omega_target": float(omega_target),
                    "lambda_before": float(lambda_before),
                    **event,
                }
            )

        return _callback

    stats = {
        "init_newton_iterations": [],
        "init_linear_iterations": 0,
        "init_linear_solve_time": 0.0,
        "init_linear_preconditioner_time": 0.0,
        "init_linear_orthogonalization_time": 0.0,
        "attempt_step": [],
        "attempt_success": [],
        "attempt_wall_time": [],
        "attempt_newton_iterations": [],
        "attempt_newton_flag": [],
        "attempt_newton_relres_end": [],
        "attempt_newton_relcorr_end": [],
        "attempt_linear_iterations": [],
        "attempt_linear_solve_time": [],
        "attempt_linear_preconditioner_time": [],
        "attempt_linear_orthogonalization_time": [],
        "attempt_omega_target": [],
        "attempt_lambda_before": [],
        "attempt_lambda_initial_guess": [],
        "attempt_predictor_kind": [],
        "attempt_lambda_after": [],
        "attempt_lambda_initial_guess_abs_error": [],
        "attempt_initial_guess_displacement_diff_volume_integral": [],
        "attempt_initial_guess_deviatoric_strain_diff_volume_integral": [],
        "attempt_secant_reference_displacement_diff_volume_integral": [],
        "attempt_secant_reference_deviatoric_strain_diff_volume_integral": [],
        "attempt_first_newton_linear_iterations": [],
        "attempt_first_newton_linear_solve_time": [],
        "attempt_first_newton_linear_preconditioner_time": [],
        "attempt_first_newton_linear_orthogonalization_time": [],
        "attempt_first_newton_correction_norm": [],
        "attempt_micro_advanced": [],
        "attempt_micro_arc_length_increment": [],
        "attempt_micro_arc_length_accumulated": [],
        "attempt_micro_target_length": [],
        "step_index": [],
        "step_attempt_count": [],
        "step_wall_time": [],
        "step_newton_iterations": [],
        "step_newton_iterations_total": [],
        "step_newton_relres_end": [],
        "step_newton_relcorr_end": [],
        "step_linear_iterations": [],
        "step_linear_solve_time": [],
        "step_linear_preconditioner_time": [],
        "step_linear_orthogonalization_time": [],
        "step_lambda": [],
        "step_omega": [],
        "step_lambda_initial_guess": [],
        "step_predictor_kind": [],
        "step_initial_guess_displacement_diff_volume_integral": [],
        "step_initial_guess_deviatoric_strain_diff_volume_integral": [],
        "step_secant_reference_displacement_diff_volume_integral": [],
        "step_secant_reference_deviatoric_strain_diff_volume_integral": [],
        "step_lambda_initial_guess_abs_error": [],
        "step_d_omega": [],
        "step_next_d_omega": [],
        "step_d_omega_scale": [],
        "step_micro_attempt_count": [],
        "step_micro_arc_length_total": [],
        "step_micro_target_length_last": [],
        "step_U": [],
        "total_wall_time": 0.0,
        "stop_reason": "completed",
    }

    t_total = perf_counter()
    snap_init_0 = _collector_snapshot(linear_system_solver)
    U_old, U, omega_old, omega, lambda_old, lambda_value, init_newton_its = init_phase_SSR_indirect_continuation(
        lambda_init,
        d_lambda_init,
        d_lambda_min,
        it_newt_max,
        it_damp_max,
        tol,
        r_min,
        K_elast,
        Q,
        f,
        constitutive_matrix_builder,
        linear_system_solver.copy(),
        newton_stopping_criterion=str(newton_stopping_criterion),
        newton_stopping_tol=newton_stopping_tol,
        init_newton_stopping_criterion=init_newton_stopping_criterion,
        init_newton_stopping_tol=init_newton_stopping_tol,
    )
    snap_init_1 = _collector_snapshot(linear_system_solver)
    delta_init = _collector_delta(snap_init_0, snap_init_1)
    stats["init_newton_iterations"] = init_newton_its
    stats["init_linear_iterations"] = delta_init["iterations"]
    stats["init_linear_solve_time"] = delta_init["solve_time"]
    stats["init_linear_preconditioner_time"] = delta_init["preconditioner_time"]
    stats["init_linear_orthogonalization_time"] = delta_init["orthogonalization_time"]

    if getattr(linear_system_solver, "supports_dynamic_deflation_basis", lambda: True)():
        linear_system_solver.expand_deflation_basis(_free(U_old, Q))
        linear_system_solver.expand_deflation_basis(_free(U, Q))
    accepted_basis_snapshot = _trim_basis_snapshot(_basis_snapshot(linear_system_solver), streaming_basis_max_vectors)
    _basis_restore(linear_system_solver, accepted_basis_snapshot)

    lambda_hist = np.zeros(1000, dtype=np.float64)
    omega_hist = np.zeros(1000, dtype=np.float64)
    Umax_hist = np.zeros(1000, dtype=np.float64)
    lambda_hist[0] = lambda_old
    lambda_hist[1] = lambda_value
    omega_hist[0] = omega_old
    omega_hist[1] = omega
    Umax_hist[0] = np.max(np.linalg.norm(U_old, axis=0))
    Umax_hist[1] = np.max(np.linalg.norm(U, axis=0))
    if store_step_u:
        stats["step_U"].append(U_old.copy())
        stats["step_U"].append(U.copy())

    omega_scale, lambda_scale = _streaming_reference_scales(
        omega_old=float(omega_old),
        omega=float(omega),
        lambda_old=float(lambda_old),
        lambda_value=float(lambda_value),
    )
    base_s_micro = float(np.clip(float(streaming_micro_target_length), float(streaming_micro_min_length), float(streaming_micro_max_length)))
    s_micro = float(base_s_micro)

    step = 2
    stop_reason = "completed"

    accepted_prev_U = U_old.copy()
    accepted_prev_lambda = float(lambda_old)
    accepted_prev_omega = float(omega_old)
    accepted_curr_U = U.copy()
    accepted_curr_lambda = float(lambda_value)
    accepted_curr_omega = float(omega)

    U_work = accepted_curr_U.copy()
    lambda_work = float(accepted_curr_lambda)
    omega_work = float(accepted_curr_omega)

    arc_prev_U = accepted_prev_U.copy()
    arc_prev_lambda = float(accepted_prev_lambda)
    arc_prev_omega = float(accepted_prev_omega)
    arc_curr_U = accepted_curr_U.copy()
    arc_curr_lambda = float(accepted_curr_lambda)
    arc_curr_omega = float(accepted_curr_omega)

    step_wall_accum = 0.0
    step_lin_it_accum = 0
    step_lin_solve_accum = 0.0
    step_lin_prec_accum = 0.0
    step_lin_orth_accum = 0.0
    step_newton_it_accum = 0
    step_attempt_count = 0
    micro_attempts_since_accept = 0
    s_acc = 0.0

    while True:
        if step >= step_max:
            stop_reason = "step_max"
            break
        if micro_attempts_since_accept >= int(streaming_micro_max_corrections):
            stop_reason = "streaming_microstep_stall"
            break
        if omega_work >= omega_max_stop - 1.0e-12 and s_acc == 0.0 and micro_attempts_since_accept == 0:
            stop_reason = "omega_max_stop"
            break

        lambda_ini = float(lambda_work)
        U_ini = np.asarray(U_work, dtype=np.float64).copy()
        attempt_in_step = int(step_attempt_count + 1)
        omega_before_correction = float(omega_work)

        attempt_t0 = perf_counter()
        snap_before_attempt = _collector_snapshot(linear_system_solver)
        _basis_restore(linear_system_solver, accepted_basis_snapshot)
        U_corr, lambda_corr, flag, it_newt, history = newton_ind_ssr(
            U_ini,
            float(omega_work),
            float(lambda_ini),
            1,
            int(it_damp_max),
            float(tol),
            float(r_min),
            K_elast,
            Q,
            f,
            constitutive_matrix_builder,
            linear_system_solver,
            progress_callback=_newton_progress_callback(
                target_step=int(step + 1),
                accepted_steps=int(step),
                attempt_in_step=attempt_in_step,
                omega_target=float(omega_work),
                lambda_before=float(lambda_ini),
            ),
            stopping_criterion=str(newton_stopping_criterion),
            stopping_tol=newton_stopping_tol,
        )
        _basis_restore(linear_system_solver, accepted_basis_snapshot)
        snap_after_attempt = _collector_snapshot(linear_system_solver)
        attempt_delta = _collector_delta(snap_before_attempt, snap_after_attempt)
        attempt_wall = float(perf_counter() - attempt_t0)

        residual_hist = np.asarray(history.get("residual", []), dtype=np.float64)
        attempt_relres = float(residual_hist[-1]) if residual_hist.size else np.nan
        attempt_relcorr = _final_finite(history.get("accepted_relative_correction_norm", []))
        alpha_hist = np.asarray(history.get("alpha", []), dtype=np.float64)
        alpha_last = float(alpha_hist[-1]) if alpha_hist.size else 0.0

        U_work = np.asarray(U_corr, dtype=np.float64)
        lambda_work = float(lambda_corr)
        arc_curr_U = U_work.copy()
        arc_curr_lambda = float(lambda_work)
        arc_curr_omega = float(omega_work)

        first_newton_linear_iterations = float(history.get("first_iteration_linear_iterations", np.nan))
        first_newton_linear_solve_time = float(history.get("first_iteration_linear_solve_time", np.nan))
        first_newton_linear_preconditioner_time = float(history.get("first_iteration_linear_preconditioner_time", np.nan))
        first_newton_linear_orthogonalization_time = float(history.get("first_iteration_linear_orthogonalization_time", np.nan))
        first_newton_correction_norm = float(history.get("first_accepted_correction_norm", np.nan))

        if np.isfinite(alpha_last):
            if alpha_last >= 0.9:
                s_micro = min(float(streaming_micro_max_length), float(s_micro) * 1.15)
            elif alpha_last < float(streaming_alpha_advance_threshold):
                s_micro = max(float(streaming_micro_min_length), float(s_micro) * 0.5)

        freeze_for_accept = bool(s_acc >= 1.0 or omega_work >= omega_max_stop - 1.0e-12)
        relres_for_move = np.inf if np.isnan(attempt_relres) else float(attempt_relres)
        advanced = False
        ds_inc = 0.0

        if (
            not freeze_for_accept
            and relres_for_move <= float(streaming_move_relres_threshold)
            and alpha_last >= float(streaming_alpha_advance_threshold)
            and float(omega_work) < float(omega_max_stop) - 1.0e-12
        ):
            domega, alpha_sec, ds_inc = _streaming_micro_domega(
                omega_prev=float(arc_prev_omega),
                omega_curr=float(arc_curr_omega),
                lambda_prev=float(arc_prev_lambda),
                lambda_curr=float(arc_curr_lambda),
                omega_scale=float(omega_scale),
                lambda_scale=float(lambda_scale),
                s_micro=float(min(s_micro, max(1.0 - s_acc, float(streaming_micro_min_length)))),
                omega_remaining=float(omega_max_stop - omega_work),
            )
            if domega > 0.0:
                U_pred = np.asarray(arc_curr_U + float(alpha_sec) * (arc_curr_U - arc_prev_U), dtype=np.float64)
                lambda_pred = float(arc_curr_lambda + float(alpha_sec) * (arc_curr_lambda - arc_prev_lambda))
                omega_pred = float(omega_work + domega)
                U_pred = _rescale_to_target_omega(U_pred, omega_pred, f, Q)
                arc_prev_U = arc_curr_U.copy()
                arc_prev_lambda = float(arc_curr_lambda)
                arc_prev_omega = float(arc_curr_omega)
                U_work = U_pred.copy()
                lambda_work = float(lambda_pred)
                omega_work = float(omega_pred)
                s_acc += float(ds_inc)
                advanced = True

        step_wall_accum += attempt_wall
        step_lin_it_accum += int(attempt_delta["iterations"])
        step_lin_solve_accum += float(attempt_delta["solve_time"])
        step_lin_prec_accum += float(attempt_delta["preconditioner_time"])
        step_lin_orth_accum += float(attempt_delta["orthogonalization_time"])
        step_newton_it_accum += int(it_newt)
        step_attempt_count += 1
        micro_attempts_since_accept += 1

        accepted_now = bool(freeze_for_accept and relres_for_move <= float(tol))

        stats["attempt_step"].append(int(step + 1))
        stats["attempt_success"].append(bool(accepted_now))
        stats["attempt_wall_time"].append(float(attempt_wall))
        stats["attempt_newton_iterations"].append(int(it_newt))
        stats["attempt_newton_flag"].append(int(flag))
        stats["attempt_newton_relres_end"].append(float(attempt_relres))
        stats["attempt_newton_relcorr_end"].append(float(attempt_relcorr))
        stats["attempt_linear_iterations"].append(int(attempt_delta["iterations"]))
        stats["attempt_linear_solve_time"].append(float(attempt_delta["solve_time"]))
        stats["attempt_linear_preconditioner_time"].append(float(attempt_delta["preconditioner_time"]))
        stats["attempt_linear_orthogonalization_time"].append(float(attempt_delta["orthogonalization_time"]))
        stats["attempt_omega_target"].append(float(omega_before_correction))
        stats["attempt_lambda_before"].append(float(lambda_ini))
        stats["attempt_lambda_initial_guess"].append(float(lambda_ini))
        stats["attempt_predictor_kind"].append("streaming_microstep_secant")
        stats["attempt_lambda_after"].append(float(lambda_corr))
        stats["attempt_lambda_initial_guess_abs_error"].append(abs(float(lambda_corr) - float(lambda_ini)))
        stats["attempt_initial_guess_displacement_diff_volume_integral"].append(np.nan)
        stats["attempt_initial_guess_deviatoric_strain_diff_volume_integral"].append(np.nan)
        stats["attempt_secant_reference_displacement_diff_volume_integral"].append(np.nan)
        stats["attempt_secant_reference_deviatoric_strain_diff_volume_integral"].append(np.nan)
        stats["attempt_first_newton_linear_iterations"].append(float(first_newton_linear_iterations))
        stats["attempt_first_newton_linear_solve_time"].append(float(first_newton_linear_solve_time))
        stats["attempt_first_newton_linear_preconditioner_time"].append(float(first_newton_linear_preconditioner_time))
        stats["attempt_first_newton_linear_orthogonalization_time"].append(float(first_newton_linear_orthogonalization_time))
        stats["attempt_first_newton_correction_norm"].append(float(first_newton_correction_norm))
        stats["attempt_micro_advanced"].append(bool(advanced))
        stats["attempt_micro_arc_length_increment"].append(float(ds_inc))
        stats["attempt_micro_arc_length_accumulated"].append(float(s_acc))
        stats["attempt_micro_target_length"].append(float(s_micro))

        _emit(
            "attempt_complete",
            phase="continuation",
            target_step=int(step + 1),
            accepted_steps=int(step),
            attempt_in_step=attempt_in_step,
            success=bool(accepted_now),
            omega_target=float(omega_before_correction),
            omega_after=float(omega_work),
            lambda_before=float(lambda_ini),
            lambda_after=float(lambda_corr),
            newton_iterations=int(it_newt),
            newton_flag=int(flag),
            newton_relres_end=None if np.isnan(attempt_relres) else float(attempt_relres),
            newton_relcorr_end=None if np.isnan(attempt_relcorr) else float(attempt_relcorr),
            linear_iterations=int(attempt_delta["iterations"]),
            linear_solve_time=float(attempt_delta["solve_time"]),
            linear_preconditioner_time=float(attempt_delta["preconditioner_time"]),
            linear_orthogonalization_time=float(attempt_delta["orthogonalization_time"]),
            micro_advanced=bool(advanced),
            micro_arc_length_increment=float(ds_inc),
            micro_arc_length_accumulated=float(s_acc),
            micro_target_length=float(s_micro),
            attempt_wall_time=float(attempt_wall),
            total_wall_time=float(perf_counter() - t_total),
        )

        if accepted_now:
            step += 1
            accepted_prev_U = accepted_curr_U.copy()
            accepted_prev_lambda = float(accepted_curr_lambda)
            accepted_prev_omega = float(accepted_curr_omega)
            accepted_curr_U = np.asarray(U_work, dtype=np.float64).copy()
            accepted_curr_lambda = float(lambda_work)
            accepted_curr_omega = float(omega_before_correction)
            U_old = accepted_prev_U.copy()
            U = accepted_curr_U.copy()
            omega_old = float(accepted_prev_omega)
            omega = float(accepted_curr_omega)
            lambda_old = float(accepted_prev_lambda)
            lambda_value = float(accepted_curr_lambda)
            lambda_hist[step - 1] = float(lambda_value)
            omega_hist[step - 1] = float(omega)
            Umax_hist[step - 1] = np.max(np.linalg.norm(U, axis=0))

            attempt_u_diff = np.nan
            attempt_dev_diff = np.nan
            attempt_secant_u_diff = np.nan
            attempt_secant_dev_diff = np.nan
            if step_guess_diagnostics is not None:
                try:
                    guess_diag = step_guess_diagnostics(np.asarray(U_ini, dtype=np.float64), np.asarray(U, dtype=np.float64))
                    attempt_u_diff = float(guess_diag.get("displacement_diff_volume_integral", np.nan))
                    attempt_dev_diff = float(guess_diag.get("deviatoric_strain_diff_volume_integral", np.nan))
                except Exception:
                    attempt_u_diff = np.nan
                    attempt_dev_diff = np.nan
                try:
                    U_secant_ref, _lambda_secant_ref, _ = _secant_predictor(
                        omega_old=float(accepted_prev_omega),
                        omega=float(accepted_curr_omega),
                        omega_target=float(accepted_curr_omega),
                        U_old=np.asarray(accepted_prev_U, dtype=np.float64),
                        U=np.asarray(accepted_curr_U, dtype=np.float64),
                        lambda_value=float(accepted_curr_lambda),
                    )
                    secant_diag = step_guess_diagnostics(np.asarray(U_secant_ref, dtype=np.float64), np.asarray(U, dtype=np.float64))
                    attempt_secant_u_diff = float(secant_diag.get("displacement_diff_volume_integral", np.nan))
                    attempt_secant_dev_diff = float(secant_diag.get("deviatoric_strain_diff_volume_integral", np.nan))
                except Exception:
                    attempt_secant_u_diff = np.nan
                    attempt_secant_dev_diff = np.nan

            stats["step_index"].append(int(step))
            stats["step_attempt_count"].append(int(step_attempt_count))
            stats["step_wall_time"].append(float(step_wall_accum))
            stats["step_newton_iterations"].append(int(it_newt))
            stats["step_newton_iterations_total"].append(int(step_newton_it_accum))
            stats["step_newton_relres_end"].append(float(attempt_relres))
            stats["step_newton_relcorr_end"].append(float(attempt_relcorr))
            stats["step_linear_iterations"].append(int(step_lin_it_accum))
            stats["step_linear_solve_time"].append(float(step_lin_solve_accum))
            stats["step_linear_preconditioner_time"].append(float(step_lin_prec_accum))
            stats["step_linear_orthogonalization_time"].append(float(step_lin_orth_accum))
            stats["step_lambda"].append(float(lambda_value))
            stats["step_omega"].append(float(omega))
            stats["step_lambda_initial_guess"].append(float(lambda_ini))
            stats["step_predictor_kind"].append("streaming_microstep_secant")
            stats["step_initial_guess_displacement_diff_volume_integral"].append(float(attempt_u_diff))
            stats["step_initial_guess_deviatoric_strain_diff_volume_integral"].append(float(attempt_dev_diff))
            stats["step_secant_reference_displacement_diff_volume_integral"].append(float(attempt_secant_u_diff))
            stats["step_secant_reference_deviatoric_strain_diff_volume_integral"].append(float(attempt_secant_dev_diff))
            stats["step_lambda_initial_guess_abs_error"].append(abs(float(lambda_value) - float(lambda_ini)))
            stats["step_d_omega"].append(float(accepted_curr_omega - accepted_prev_omega))
            stats["step_next_d_omega"].append(np.nan)
            stats["step_d_omega_scale"].append(np.nan)
            stats["step_micro_attempt_count"].append(int(micro_attempts_since_accept))
            stats["step_micro_arc_length_total"].append(float(s_acc))
            stats["step_micro_target_length_last"].append(float(s_micro))
            if store_step_u:
                stats["step_U"].append(U.copy())

            if getattr(linear_system_solver, "supports_dynamic_deflation_basis", lambda: True)():
                linear_system_solver.expand_deflation_basis(_free(U, Q))
            accepted_basis_snapshot = _trim_basis_snapshot(_basis_snapshot(linear_system_solver), streaming_basis_max_vectors)
            _basis_restore(linear_system_solver, accepted_basis_snapshot)

            _emit(
                "step_accepted",
                phase="continuation",
                accepted_step=int(step),
                lambda_value=float(lambda_value),
                omega_value=float(omega),
                d_omega=float(accepted_curr_omega - accepted_prev_omega),
                step_wall_time=float(step_wall_accum),
                step_newton_iterations_total=int(step_newton_it_accum),
                step_linear_iterations=int(step_lin_it_accum),
                step_newton_relres_end=None if np.isnan(attempt_relres) else float(attempt_relres),
                step_newton_relcorr_end=None if np.isnan(attempt_relcorr) else float(attempt_relcorr),
                micro_attempt_count=int(micro_attempts_since_accept),
                micro_arc_length_total=float(s_acc),
                total_wall_time=float(perf_counter() - t_total),
            )

            arc_prev_U = accepted_prev_U.copy()
            arc_prev_lambda = float(accepted_prev_lambda)
            arc_prev_omega = float(accepted_prev_omega)
            arc_curr_U = accepted_curr_U.copy()
            arc_curr_lambda = float(accepted_curr_lambda)
            arc_curr_omega = float(accepted_curr_omega)
            U_work = accepted_curr_U.copy()
            lambda_work = float(accepted_curr_lambda)
            omega_work = float(accepted_curr_omega)
            step_wall_accum = 0.0
            step_lin_it_accum = 0
            step_lin_solve_accum = 0.0
            step_lin_prec_accum = 0.0
            step_lin_orth_accum = 0.0
            step_newton_it_accum = 0
            step_attempt_count = 0
            micro_attempts_since_accept = 0
            s_acc = 0.0
            s_micro = float(base_s_micro)

            if omega >= omega_max_stop - 1.0e-12:
                stop_reason = "omega_max_stop"
                break

    stats["total_wall_time"] = float(perf_counter() - t_total)
    stats["stop_reason"] = str(stop_reason)
    return U, lambda_hist[:step], omega_hist[:step], Umax_hist[:step], stats


def SSR_indirect_continuation(
    lambda_init: float,
    d_lambda_init: float,
    d_lambda_min: float,
    d_lambda_diff_scaled_min: float,
    step_max: int,
    omega_max_stop: float,
    it_newt_max: int,
    it_damp_max: int,
    tol: float,
    r_min: float,
    K_elast,
    Q: np.ndarray,
    f: np.ndarray,
    constitutive_matrix_builder,
    linear_system_solver,
    progress_callback: Callable[[dict], None] | None = None,
    *,
    store_step_u: bool = True,
    continuation_predictor: str = "secant",
    omega_step_controller: str = "legacy",
    step_guess_diagnostics: Callable[[np.ndarray, np.ndarray], dict[str, float]] | None = None,
    omega_no_increase_newton_threshold: int | None = None,
    omega_half_newton_threshold: int | None = None,
    omega_target_newton_iterations: float | None = None,
    omega_adapt_min_scale: float | None = None,
    omega_adapt_max_scale: float | None = None,
    omega_hard_newton_threshold: int | None = None,
    omega_hard_linear_threshold: int | None = None,
    omega_efficiency_floor: float | None = None,
    omega_efficiency_drop_ratio: float | None = None,
    omega_efficiency_window: int = 3,
    omega_hard_shrink_scale: float | None = None,
    step_length_cap_mode: str = "none",
    step_length_cap_factor: float = 1.0,
    newton_stopping_criterion: str = "relative_residual",
    newton_stopping_tol: float | None = None,
    init_newton_stopping_criterion: str | None = None,
    init_newton_stopping_tol: float | None = None,
    fine_newton_stopping_criterion: str | None = None,
    fine_newton_stopping_tol: float | None = None,
    fine_switch_mode: str = "none",
    fine_switch_distance_factor: float = 2.0,
    continuation_predictor_use_projected_lambda: bool = True,
    continuation_predictor_refine_lambda_for_fixed_u: bool = False,
    continuation_predictor_switch_ordinal: int | None = None,
    continuation_predictor_switch_to: str | None = None,
    continuation_predictor_window_size: int | None = None,
    continuation_predictor_reduced_max_iterations: int | None = None,
    continuation_predictor_reduced_use_partial_result: bool = False,
    continuation_predictor_reduced_tolerance: float | None = None,
    continuation_predictor_power_order: int | None = None,
    continuation_predictor_power_init: str | None = None,
    continuation_secant_correction_mode: str = "none",
    continuation_first_newton_warm_start_mode: str = "none",
    continuation_mode: str = "classic",
    streaming_micro_target_length: float = 0.15,
    streaming_micro_min_length: float = 0.05,
    streaming_micro_max_length: float = 0.30,
    streaming_move_relres_threshold: float = 5.0e-3,
    streaming_alpha_advance_threshold: float = 0.5,
    streaming_micro_max_corrections: int = 40,
    streaming_basis_max_vectors: int = 8,
):
    """Indirect continuation in ``omega`` with nested-Newton solves."""

    Q = np.asarray(Q, dtype=bool)
    continuation_mode_name = str(continuation_mode).strip().lower()
    if continuation_mode_name not in {"classic", "streaming_microstep"}:
        raise ValueError(f"Unsupported continuation_mode {continuation_mode!r}.")
    if continuation_mode_name == "streaming_microstep":
        return _SSR_indirect_continuation_streaming_microstep(
            lambda_init=lambda_init,
            d_lambda_init=d_lambda_init,
            d_lambda_min=d_lambda_min,
            step_max=step_max,
            omega_max_stop=omega_max_stop,
            it_newt_max=it_newt_max,
            it_damp_max=it_damp_max,
            tol=tol,
            r_min=r_min,
            K_elast=K_elast,
            Q=Q,
            f=f,
            constitutive_matrix_builder=constitutive_matrix_builder,
            linear_system_solver=linear_system_solver,
            progress_callback=progress_callback,
            store_step_u=store_step_u,
            step_guess_diagnostics=step_guess_diagnostics,
            newton_stopping_criterion=str(newton_stopping_criterion),
            newton_stopping_tol=newton_stopping_tol,
            init_newton_stopping_criterion=init_newton_stopping_criterion,
            init_newton_stopping_tol=init_newton_stopping_tol,
            streaming_micro_target_length=streaming_micro_target_length,
            streaming_micro_min_length=streaming_micro_min_length,
            streaming_micro_max_length=streaming_micro_max_length,
            streaming_move_relres_threshold=streaming_move_relres_threshold,
            streaming_alpha_advance_threshold=streaming_alpha_advance_threshold,
            streaming_micro_max_corrections=streaming_micro_max_corrections,
            streaming_basis_max_vectors=streaming_basis_max_vectors,
        )

    predictor_mode = str(continuation_predictor).strip().lower()
    allowed_predictor_modes = {
        "secant",
        "reduced_newton_all_prev",
        "reduced_newton_affine_all_prev",
        "reduced_newton_window",
        "reduced_newton_increment_power",
    }
    if predictor_mode not in allowed_predictor_modes:
        raise ValueError(f"Unsupported continuation_predictor {continuation_predictor!r}.")
    omega_step_controller_mode = str(omega_step_controller).strip().lower()
    if omega_step_controller_mode not in {"legacy", "adaptive"}:
        raise ValueError(f"Unsupported omega_step_controller {omega_step_controller!r}.")
    predictor_switch_to_mode = None if continuation_predictor_switch_to is None else str(continuation_predictor_switch_to).strip().lower()
    if predictor_switch_to_mode is not None and predictor_switch_to_mode not in allowed_predictor_modes:
        raise ValueError(f"Unsupported continuation_predictor_switch_to {continuation_predictor_switch_to!r}.")
    predictor_switch_ordinal = None if continuation_predictor_switch_ordinal is None else int(continuation_predictor_switch_ordinal)
    if predictor_switch_ordinal is not None and predictor_switch_ordinal <= 0:
        predictor_switch_ordinal = None
    predictor_window_size = None if continuation_predictor_window_size is None else max(int(continuation_predictor_window_size), 1)
    predictor_reduced_max_iterations = (
        25 if continuation_predictor_reduced_max_iterations is None else max(int(continuation_predictor_reduced_max_iterations), 1)
    )
    predictor_reduced_tolerance = (
        None if continuation_predictor_reduced_tolerance is None else max(float(continuation_predictor_reduced_tolerance), 0.0)
    )
    predictor_power_order = 3 if continuation_predictor_power_order is None else max(int(continuation_predictor_power_order), 1)
    predictor_power_init = "secant" if continuation_predictor_power_init is None else str(continuation_predictor_power_init).strip().lower()
    if predictor_power_init not in {"secant", "equal_split"}:
        raise ValueError(f"Unsupported continuation_predictor_power_init {continuation_predictor_power_init!r}.")
    secant_correction_mode = str(continuation_secant_correction_mode).strip().lower()
    if secant_correction_mode not in {"none", "orthogonal_increment_ls"}:
        raise ValueError(
            f"Unsupported continuation_secant_correction_mode {continuation_secant_correction_mode!r}."
        )
    first_newton_warm_start_mode = str(continuation_first_newton_warm_start_mode).strip().lower()
    if first_newton_warm_start_mode not in {"none", "history_deflation"}:
        raise ValueError(
            f"Unsupported continuation_first_newton_warm_start_mode {continuation_first_newton_warm_start_mode!r}."
        )
    step_length_cap_mode_name = str(step_length_cap_mode).strip().lower()
    if step_length_cap_mode_name not in {"none", "initial_segment", "history_box"}:
        raise ValueError(f"Unsupported step_length_cap_mode {step_length_cap_mode!r}.")
    fine_switch_mode_name = str(fine_switch_mode).strip().lower()
    if fine_switch_mode_name not in {"none", "history_box_cumulative_distance"}:
        raise ValueError(f"Unsupported fine_switch_mode {fine_switch_mode!r}.")
    if fine_switch_mode_name != "none" and step_length_cap_mode_name != "history_box":
        raise ValueError("fine_switch_mode='history_box_cumulative_distance' requires step_length_cap_mode='history_box'.")
    rough_stop_criterion = str(newton_stopping_criterion)
    rough_stop_tol = newton_stopping_tol
    fine_stop_criterion = rough_stop_criterion if fine_newton_stopping_criterion is None else str(fine_newton_stopping_criterion)
    fine_stop_tol = rough_stop_tol if fine_newton_stopping_tol is None else fine_newton_stopping_tol
    fine_switch_distance_factor_value = max(float(fine_switch_distance_factor), 0.0)

    stats = {
        "init_newton_iterations": [],
        "init_linear_iterations": 0,
        "init_linear_solve_time": 0.0,
        "init_linear_preconditioner_time": 0.0,
        "init_linear_orthogonalization_time": 0.0,
        "attempt_step": [],
        "attempt_success": [],
        "attempt_wall_time": [],
        "attempt_newton_iterations": [],
        "attempt_newton_flag": [],
        "attempt_newton_relres_end": [],
        "attempt_newton_relcorr_end": [],
        "attempt_linear_iterations": [],
        "attempt_linear_solve_time": [],
        "attempt_linear_preconditioner_time": [],
        "attempt_linear_orthogonalization_time": [],
        "attempt_omega_target": [],
        "attempt_lambda_before": [],
        "attempt_lambda_initial_guess": [],
        "attempt_predictor_kind": [],
        "attempt_predictor_basis_dim": [],
        "attempt_predictor_wall_time": [],
        "attempt_predictor_alpha": [],
        "attempt_predictor_beta": [],
        "attempt_predictor_gamma": [],
        "attempt_predictor_projected_delta_lambda": [],
        "attempt_predictor_projected_correction_norm": [],
        "attempt_predictor_energy_eval_count": [],
        "attempt_predictor_energy_value": [],
        "attempt_predictor_coarse_solve_wall_time": [],
        "attempt_predictor_coarse_newton_iterations": [],
        "attempt_predictor_coarse_residual_end": [],
        "attempt_predictor_basis_condition": [],
        "attempt_predictor_reduced_newton_iterations": [],
        "attempt_predictor_reduced_gmres_iterations": [],
        "attempt_predictor_reduced_projected_residual": [],
        "attempt_predictor_reduced_omega_residual": [],
        "attempt_predictor_state_coefficients": [],
        "attempt_predictor_state_coefficients_ref": [],
        "attempt_predictor_state_coefficient_sum": [],
        "attempt_secant_correction_active": [],
        "attempt_secant_correction_basis_dim": [],
        "attempt_secant_correction_trust_region_clipped": [],
        "attempt_secant_correction_predicted_residual_decrease": [],
        "attempt_first_newton_warm_start_active": [],
        "attempt_first_newton_warm_start_basis_dim": [],
        "attempt_first_newton_linear_iterations": [],
        "attempt_first_newton_linear_solve_time": [],
        "attempt_first_newton_linear_preconditioner_time": [],
        "attempt_first_newton_linear_orthogonalization_time": [],
        "attempt_first_newton_correction_norm": [],
        "attempt_predictor_fallback_used": [],
        "attempt_predictor_error": [],
        "attempt_lambda_after": [],
        "attempt_lambda_initial_guess_abs_error": [],
        "attempt_initial_guess_displacement_diff_volume_integral": [],
        "attempt_initial_guess_deviatoric_strain_diff_volume_integral": [],
        "attempt_secant_reference_displacement_diff_volume_integral": [],
        "attempt_secant_reference_deviatoric_strain_diff_volume_integral": [],
        "attempt_precision_mode": [],
        "attempt_stopping_criterion": [],
        "attempt_stopping_tol": [],
        "attempt_fine_switch_triggered": [],
        "attempt_fine_switch_cumulative_distance": [],
        "attempt_fine_switch_current_length": [],
        "attempt_fine_switch_distance_with_current": [],
        "attempt_fine_switch_threshold": [],
        "attempt_fine_reference_step": [],
        "attempt_fine_reference_lambda": [],
        "attempt_fine_reference_omega": [],
        "step_index": [],
        "step_attempt_count": [],
        "step_wall_time": [],
        "step_newton_iterations": [],
        "step_newton_iterations_total": [],
        "step_newton_relres_end": [],
        "step_newton_relcorr_end": [],
        "step_linear_iterations": [],
        "step_linear_solve_time": [],
        "step_linear_preconditioner_time": [],
        "step_linear_orthogonalization_time": [],
        "step_lambda": [],
        "step_omega": [],
        "step_lambda_initial_guess": [],
        "step_predictor_kind": [],
        "step_predictor_basis_dim": [],
        "step_predictor_wall_time": [],
        "step_predictor_alpha": [],
        "step_predictor_beta": [],
        "step_predictor_gamma": [],
        "step_predictor_projected_delta_lambda": [],
        "step_predictor_projected_correction_norm": [],
        "step_predictor_energy_eval_count": [],
        "step_predictor_energy_value": [],
        "step_predictor_coarse_solve_wall_time": [],
        "step_predictor_coarse_newton_iterations": [],
        "step_predictor_coarse_residual_end": [],
        "step_predictor_basis_condition": [],
        "step_predictor_reduced_newton_iterations": [],
        "step_predictor_reduced_gmres_iterations": [],
        "step_predictor_reduced_projected_residual": [],
        "step_predictor_reduced_omega_residual": [],
        "step_predictor_state_coefficients": [],
        "step_predictor_state_coefficients_ref": [],
        "step_predictor_state_coefficient_sum": [],
        "step_secant_correction_active": [],
        "step_secant_correction_basis_dim": [],
        "step_secant_correction_trust_region_clipped": [],
        "step_secant_correction_predicted_residual_decrease": [],
        "step_first_newton_warm_start_active": [],
        "step_first_newton_warm_start_basis_dim": [],
        "step_first_newton_linear_iterations": [],
        "step_first_newton_linear_solve_time": [],
        "step_first_newton_linear_preconditioner_time": [],
        "step_first_newton_linear_orthogonalization_time": [],
        "step_first_newton_correction_norm": [],
        "step_predictor_fallback_used": [],
        "step_predictor_error": [],
        "step_initial_guess_displacement_diff_volume_integral": [],
        "step_initial_guess_deviatoric_strain_diff_volume_integral": [],
        "step_secant_reference_displacement_diff_volume_integral": [],
        "step_secant_reference_deviatoric_strain_diff_volume_integral": [],
        "step_lambda_initial_guess_abs_error": [],
        "step_d_omega": [],
        "step_next_d_omega": [],
        "step_d_omega_scale": [],
        "step_branch_efficiency": [],
        "step_branch_efficiency_ref": [],
        "step_growth_blocked": [],
        "step_hard_mode": [],
        "step_length_cap_applied": [],
        "step_length_cap_raw_length": [],
        "step_length_cap_limit": [],
        "step_precision_mode": [],
        "step_stopping_criterion": [],
        "step_stopping_tol": [],
        "step_fine_switch_triggered": [],
        "step_fine_switch_cumulative_distance": [],
        "step_fine_switch_current_length": [],
        "step_fine_switch_distance_with_current": [],
        "step_fine_switch_threshold": [],
        "step_fine_reference_step": [],
        "step_fine_reference_lambda": [],
        "step_fine_reference_omega": [],
        "step_U": [],
        "total_wall_time": 0.0,
    }

    def _emit(event: str, **payload) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": event,
                "continuation_kind": "ssr_indirect",
                **payload,
            }
        )

    def _newton_progress_callback(
        *,
        target_step: int,
        accepted_steps: int,
        attempt_in_step: int,
        omega_target: float,
        lambda_before: float,
        attempt_context: dict[str, object] | None = None,
    ):
        if progress_callback is None:
            return None

        def _callback(event: dict) -> None:
            progress_callback(
                {
                    "continuation_kind": "ssr_indirect",
                    "phase": "continuation",
                    "target_step": int(target_step),
                    "accepted_steps": int(accepted_steps),
                    "attempt_in_step": int(attempt_in_step),
                    "omega_target": float(omega_target),
                    "lambda_before": float(lambda_before),
                    **({} if attempt_context is None else dict(attempt_context)),
                    **event,
                }
            )

        return _callback

    lambda_hist = np.zeros(1000, dtype=np.float64)
    omega_hist = np.zeros(1000, dtype=np.float64)
    Umax_hist = np.zeros(1000, dtype=np.float64)

    t_total = perf_counter()

    snap_init_0 = _collector_snapshot(linear_system_solver)
    U_old, U, omega_old, omega, lambda_old, lambda_value, init_newton_its = init_phase_SSR_indirect_continuation(
        lambda_init,
        d_lambda_init,
        d_lambda_min,
        it_newt_max,
        it_damp_max,
        tol,
        r_min,
        K_elast,
        Q,
        f,
        constitutive_matrix_builder,
        linear_system_solver.copy(),
        newton_stopping_criterion=str(newton_stopping_criterion),
        newton_stopping_tol=newton_stopping_tol,
        init_newton_stopping_criterion=init_newton_stopping_criterion,
        init_newton_stopping_tol=init_newton_stopping_tol,
    )
    snap_init_1 = _collector_snapshot(linear_system_solver)
    delta_init = _collector_delta(snap_init_0, snap_init_1)

    stats["init_newton_iterations"] = init_newton_its
    stats["init_linear_iterations"] = delta_init["iterations"]
    stats["init_linear_solve_time"] = delta_init["solve_time"]
    stats["init_linear_preconditioner_time"] = delta_init["preconditioner_time"]
    stats["init_linear_orthogonalization_time"] = delta_init["orthogonalization_time"]

    _emit(
        "init_complete",
        phase="init",
        accepted_steps=2,
        lambda_hist=[float(lambda_old), float(lambda_value)],
        omega_hist=[float(omega_old), float(omega)],
        init_newton_iterations=[int(v) for v in init_newton_its],
        init_linear_iterations=int(delta_init["iterations"]),
        init_linear_solve_time=float(delta_init["solve_time"]),
        init_linear_preconditioner_time=float(delta_init["preconditioner_time"]),
        init_linear_orthogonalization_time=float(delta_init["orthogonalization_time"]),
        total_wall_time=float(perf_counter() - t_total),
    )

    if getattr(linear_system_solver, "supports_dynamic_deflation_basis", lambda: True)():
        linear_system_solver.expand_deflation_basis(_free(U_old, Q))

    omega_hist[0] = omega_old
    lambda_hist[0] = lambda_old
    omega_hist[1] = omega
    lambda_hist[1] = lambda_value
    Umax_hist[0] = np.max(np.linalg.norm(U_old, axis=0))
    Umax_hist[1] = np.max(np.linalg.norm(U, axis=0))
    if store_step_u:
        stats["step_U"].append(U_old.copy())
        stats["step_U"].append(U.copy())

    predictor_omega_hist: list[float] = [float(omega_old), float(omega)]
    predictor_lambda_hist: list[float] = [float(lambda_old), float(lambda_value)]
    predictor_u_hist: list[np.ndarray] = [U_old.copy(), U.copy()]
    continuation_increment_free_hist: list[np.ndarray] = []
    continuation_first_newton_correction_free_hist: list[np.ndarray] = []
    continuation_state_hist_only: list[np.ndarray] = []
    continuation_u_hist_full: list[np.ndarray] = [U_old.copy(), U.copy()]
    continuation_omega_hist_full: list[float] = [float(omega_old), float(omega)]
    continuation_lambda_hist_full: list[float] = [float(lambda_old), float(lambda_value)]
    continuation_increment_full_hist: list[np.ndarray] = []
    last_fine_hist_idx = 1

    d_omega = omega - omega_old
    if omega_max_stop < omega + d_omega:
        raise ValueError("Too small value of omega_max_stop. Increase it and rerun.")

    step = 2
    n_omega = 0
    n_omega_max = 5
    stop_reason = "completed"

    step_wall_accum = 0.0
    step_lin_it_accum = 0
    step_lin_solve_accum = 0.0
    step_lin_prec_accum = 0.0
    step_lin_orth_accum = 0.0
    step_newton_it_accum = 0
    step_attempt_count = 0
    step_branch_efficiency_hist: list[float] = []

    while True:
        omega_it = min(omega + d_omega, omega_max_stop)
        d_omega = omega_it - omega

        predictor_info = _predictor_info_defaults()
        U_secant_ref, _lambda_secant_ref, _ = _secant_predictor(
            omega_old=float(omega_old),
            omega=float(omega),
            omega_target=float(omega_it),
            U_old=U_old,
            U=U,
            lambda_value=float(lambda_value),
        )
        active_predictor_mode = predictor_mode
        current_continuation_ordinal = int(step - 1)
        if (
            predictor_switch_ordinal is not None
            and predictor_switch_to_mode is not None
            and current_continuation_ordinal >= predictor_switch_ordinal
        ):
            active_predictor_mode = predictor_switch_to_mode
        if active_predictor_mode == "reduced_newton_all_prev":
            U_ini, lambda_ini, predictor_kind, predictor_info = _projected_reduced_newton_predictor(
                predictor_label="reduced_newton_all_prev_projected",
                omega_old=float(omega_old),
                omega=float(omega),
                omega_target=float(omega_it),
                U_old=U_old,
                U=U,
                lambda_value=float(lambda_value),
                Q=Q,
                f=f,
                K_elast=K_elast,
                constitutive_matrix_builder=constitutive_matrix_builder,
                it_damp_max=int(it_damp_max),
                tol=float(tol),
                projected_tolerance=predictor_reduced_tolerance,
                r_min=float(r_min),
                continuation_increment_free_hist=continuation_increment_free_hist,
                window_size=None,
            )
        elif active_predictor_mode == "reduced_newton_affine_all_prev":
            U_ini, lambda_ini, predictor_kind, predictor_info = _affine_state_reduced_newton_predictor(
                predictor_label="reduced_newton_affine_all_prev_projected",
                omega_old=float(omega_old),
                omega=float(omega),
                omega_target=float(omega_it),
                U_old=U_old,
                U=U,
                lambda_value=float(lambda_value),
                Q=Q,
                f=f,
                K_elast=K_elast,
                constitutive_matrix_builder=constitutive_matrix_builder,
                it_damp_max=int(it_damp_max),
                tol=float(tol),
                projected_tolerance=predictor_reduced_tolerance,
                r_min=float(r_min),
                continuation_state_hist=continuation_state_hist_only,
                window_size=predictor_window_size,
                min_history=2 if predictor_window_size is None else max(int(predictor_window_size), 2),
                max_projected_newton_iterations=int(predictor_reduced_max_iterations),
                use_partial_result_on_nonconvergence=bool(continuation_predictor_reduced_use_partial_result),
            )
        elif active_predictor_mode == "reduced_newton_increment_power":
            U_ini, lambda_ini, predictor_kind, predictor_info = _increment_power_reduced_newton_predictor(
                predictor_label="reduced_newton_increment_power_projected",
                omega_old=float(omega_old),
                omega=float(omega),
                omega_target=float(omega_it),
                U_old=U_old,
                U=U,
                lambda_value=float(lambda_value),
                Q=Q,
                f=f,
                K_elast=K_elast,
                constitutive_matrix_builder=constitutive_matrix_builder,
                it_damp_max=int(it_damp_max),
                tol=float(tol),
                projected_tolerance=predictor_reduced_tolerance,
                r_min=float(r_min),
                power_order=int(predictor_power_order),
                init_strategy=str(predictor_power_init),
                continuation_state_hist=continuation_u_hist_full,
                increment_window_size=(1 if predictor_window_size is None else int(predictor_window_size)),
                max_projected_newton_iterations=int(predictor_reduced_max_iterations),
                use_partial_result_on_nonconvergence=bool(continuation_predictor_reduced_use_partial_result),
            )
        elif active_predictor_mode == "reduced_newton_window":
            U_ini, lambda_ini, predictor_kind, predictor_info = _projected_reduced_newton_predictor(
                predictor_label="reduced_newton_window_projected",
                omega_old=float(omega_old),
                omega=float(omega),
                omega_target=float(omega_it),
                U_old=U_old,
                U=U,
                lambda_value=float(lambda_value),
                Q=Q,
                f=f,
                K_elast=K_elast,
                constitutive_matrix_builder=constitutive_matrix_builder,
                it_damp_max=int(it_damp_max),
                tol=float(tol),
                projected_tolerance=predictor_reduced_tolerance,
                r_min=float(r_min),
                continuation_increment_free_hist=continuation_increment_free_hist,
                window_size=(3 if predictor_window_size is None else predictor_window_size),
            )
        else:
            U_ini, lambda_ini, predictor_kind = _secant_predictor(
                omega_old=float(omega_old),
                omega=float(omega),
                omega_target=float(omega_it),
                U_old=U_old,
                U=U,
                lambda_value=float(lambda_value),
            )
            predictor_info["predictor_alpha"] = _secant_alpha(
                omega_old=float(omega_old),
                omega=float(omega),
                omega_target=float(omega_it),
            )
            if secant_correction_mode == "orthogonal_increment_ls":
                U_ini, lambda_ini, predictor_kind, predictor_info = _secant_orthogonal_increment_ls_predictor(
                    omega_old=float(omega_old),
                    omega=float(omega),
                    omega_target=float(omega_it),
                    U_old=U_old,
                    U=U,
                    lambda_value=float(lambda_value),
                    Q=Q,
                    f=f,
                    K_elast=K_elast,
                    constitutive_matrix_builder=constitutive_matrix_builder,
                    continuation_increment_free_hist=continuation_increment_free_hist,
                    r_min=float(r_min),
                )

        if (
            active_predictor_mode in {"reduced_newton_all_prev", "reduced_newton_affine_all_prev", "reduced_newton_window", "reduced_newton_increment_power"}
            and not bool(continuation_predictor_use_projected_lambda)
            and "fallback_secant" not in str(predictor_kind)
        ):
            predictor_info["projected_lambda_candidate"] = float(lambda_ini)
            predictor_info["projected_delta_lambda"] = float(lambda_ini - float(lambda_value))
            predictor_info["use_projected_lambda"] = False
            lambda_ini = float(lambda_value)
        else:
            predictor_info["projected_lambda_candidate"] = float(lambda_ini)
            predictor_info["use_projected_lambda"] = True

        if (
            active_predictor_mode in {"reduced_newton_all_prev", "reduced_newton_affine_all_prev", "reduced_newton_window", "reduced_newton_increment_power"}
            and bool(continuation_predictor_refine_lambda_for_fixed_u)
            and "fallback_secant" not in str(predictor_kind)
            and len(predictor_lambda_hist) >= 2
        ):
            extra_candidates = [float(predictor_info.get("projected_lambda_candidate", np.nan))]
            lambda_refined, lambda_merit, lambda_eval_count = _refine_lambda_for_fixed_u_gauss_newton(
                U=np.asarray(U_ini, dtype=np.float64),
                omega_old=float(omega_old),
                omega=float(omega),
                omega_target=float(omega_it),
                lambda_old=float(predictor_lambda_hist[-2]),
                lambda_value=float(lambda_ini),
                Q=Q,
                f=f,
                constitutive_matrix_builder=constitutive_matrix_builder,
                extra_lambda_candidates=extra_candidates,
            )
            predictor_info["energy_eval_count"] = float(lambda_eval_count)
            predictor_info["energy_value"] = float(lambda_merit)
            predictor_info["projected_delta_lambda"] = float(lambda_refined - float(lambda_ini))
            predictor_info["lambda_refined_for_fixed_u"] = True
            lambda_ini = float(lambda_refined)
        else:
            predictor_info["lambda_refined_for_fixed_u"] = False

        warm_start_basis_free: list[np.ndarray] | None = None
        warm_start_basis_dim = 0
        if first_newton_warm_start_mode == "history_deflation":
            warm_start_basis_free, warm_start_basis_dim = _build_first_newton_warm_start_basis(
                first_newton_correction_free_hist=continuation_first_newton_correction_free_hist,
                continuation_increment_free_hist=continuation_increment_free_hist,
                max_corrections=2,
            )
            if not warm_start_basis_free:
                warm_start_basis_free = None

        current_precision_mode = "base"
        current_stop_criterion = rough_stop_criterion
        current_stop_tol = rough_stop_tol
        fine_switch_triggered = False
        fine_switch_cumulative_distance = np.nan
        fine_switch_current_length = np.nan
        fine_switch_distance_with_current = np.nan
        fine_switch_threshold = np.nan
        fine_reference_step = int(last_fine_hist_idx + 1)
        fine_reference_lambda = float(lambda_hist[last_fine_hist_idx])
        fine_reference_omega = float(omega_hist[last_fine_hist_idx])
        if fine_switch_mode_name == "history_box_cumulative_distance":
            current_precision_mode = "rough"
            hist_omega = np.asarray(omega_hist[:step], dtype=np.float64)
            hist_lambda = np.asarray(lambda_hist[:step], dtype=np.float64)
            fine_switch_threshold = float(
                _history_box_initial_segment_length(
                    omega_hist=hist_omega,
                    lambda_hist=hist_lambda,
                    factor=1.0,
                )
                * max(float(fine_switch_distance_factor_value), 0.0)
            )
            fine_switch_cumulative_distance = _history_box_path_length(
                omega_hist=hist_omega,
                lambda_hist=hist_lambda,
                start_idx=int(last_fine_hist_idx),
                end_idx=int(step - 1),
            )
            fine_switch_current_length = _history_box_projected_length(
                domega_candidate=float(d_omega),
                omega_hist=hist_omega,
                lambda_hist=hist_lambda,
                omega_anchor_prev=float(omega_old),
                omega_anchor_curr=float(omega),
                lambda_anchor_prev=float(lambda_hist[step - 2]),
                lambda_anchor_curr=float(lambda_hist[step - 1]),
            )
            fine_switch_distance_with_current = float(fine_switch_cumulative_distance + fine_switch_current_length)
            if (
                np.isfinite(fine_switch_threshold)
                and np.isfinite(fine_switch_distance_with_current)
                and fine_switch_distance_with_current > fine_switch_threshold + 1.0e-12
            ):
                current_precision_mode = "fine"
                current_stop_criterion = fine_stop_criterion
                current_stop_tol = fine_stop_tol
                fine_switch_triggered = True

        attempt_in_step = step_attempt_count + 1
        t_attempt = perf_counter()
        basis_before_attempt = _basis_snapshot(linear_system_solver)
        snap_before = _collector_snapshot(linear_system_solver)
        attempt_context = {
            "precision_mode": str(current_precision_mode),
            "stopping_criterion": str(current_stop_criterion),
            "stopping_tol": None if current_stop_tol is None else float(current_stop_tol),
            "fine_switch_mode": str(fine_switch_mode_name),
            "fine_switch_triggered": bool(fine_switch_triggered),
            "fine_switch_cumulative_distance": None
            if np.isnan(fine_switch_cumulative_distance)
            else float(fine_switch_cumulative_distance),
            "fine_switch_current_length": None if np.isnan(fine_switch_current_length) else float(fine_switch_current_length),
            "fine_switch_distance_with_current": None
            if np.isnan(fine_switch_distance_with_current)
            else float(fine_switch_distance_with_current),
            "fine_switch_threshold": None if np.isnan(fine_switch_threshold) else float(fine_switch_threshold),
            "fine_reference_step": int(fine_reference_step),
            "fine_reference_lambda": float(fine_reference_lambda),
            "fine_reference_omega": float(fine_reference_omega),
        }
        newton_kwargs = {
            "progress_callback": _newton_progress_callback(
                target_step=step + 1,
                accepted_steps=step,
                attempt_in_step=attempt_in_step,
                omega_target=float(omega_it),
                lambda_before=float(lambda_value),
                attempt_context=attempt_context,
            ),
        }
        if warm_start_basis_free is not None:
            newton_kwargs["first_iteration_extra_basis_free"] = warm_start_basis_free
        U_it, lambda_candidate, flag, it_newt, history = newton_ind_ssr(
            U_ini,
            omega_it,
            lambda_ini,
            it_newt_max,
            it_damp_max,
            tol,
            r_min,
            K_elast,
            Q,
            f,
            constitutive_matrix_builder,
            linear_system_solver,
            **newton_kwargs,
            stopping_criterion=str(current_stop_criterion),
            stopping_tol=current_stop_tol,
        )
        _basis_restore(linear_system_solver, basis_before_attempt)
        _notify_attempt(linear_system_solver, success=(flag == 0))
        snap_after = _collector_snapshot(linear_system_solver)
        attempt_delta = _collector_delta(snap_before, snap_after)
        attempt_wall = perf_counter() - t_attempt
        guess_diag = step_guess_diagnostics(U_ini, U_it) if callable(step_guess_diagnostics) else {}
        secant_guess_diag = step_guess_diagnostics(U_secant_ref, U_it) if callable(step_guess_diagnostics) else {}
        attempt_u_diff = float(guess_diag.get("displacement_diff_volume_integral", np.nan))
        attempt_dev_diff = float(guess_diag.get("deviatoric_strain_diff_volume_integral", np.nan))
        attempt_secant_u_diff = float(secant_guess_diag.get("displacement_diff_volume_integral", np.nan))
        attempt_secant_dev_diff = float(secant_guess_diag.get("deviatoric_strain_diff_volume_integral", np.nan))
        attempt_first_newton_warm_start_active = bool(history.get("first_iteration_warm_start_active", False))
        attempt_first_newton_warm_start_basis_dim = int(history.get("first_iteration_warm_start_basis_dim", 0))
        attempt_first_newton_linear_iterations = float(history.get("first_iteration_linear_iterations", np.nan))
        attempt_first_newton_linear_solve_time = float(history.get("first_iteration_linear_solve_time", np.nan))
        attempt_first_newton_linear_preconditioner_time = float(
            history.get("first_iteration_linear_preconditioner_time", np.nan)
        )
        attempt_first_newton_linear_orthogonalization_time = float(
            history.get("first_iteration_linear_orthogonalization_time", np.nan)
        )
        attempt_first_newton_correction_norm = float(history.get("first_accepted_correction_norm", np.nan))

        attempt_relres = np.nan
        if history["residual"].size:
            attempt_relres = float(history["residual"][-1])
        attempt_relcorr = _final_finite(history.get("accepted_relative_correction_norm", []))

        stats["attempt_step"].append(step + 1)
        stats["attempt_success"].append(flag == 0)
        stats["attempt_wall_time"].append(attempt_wall)
        stats["attempt_newton_iterations"].append(it_newt)
        stats["attempt_newton_flag"].append(flag)
        stats["attempt_newton_relres_end"].append(attempt_relres)
        stats["attempt_newton_relcorr_end"].append(attempt_relcorr)
        stats["attempt_linear_iterations"].append(attempt_delta["iterations"])
        stats["attempt_linear_solve_time"].append(attempt_delta["solve_time"])
        stats["attempt_linear_preconditioner_time"].append(attempt_delta["preconditioner_time"])
        stats["attempt_linear_orthogonalization_time"].append(attempt_delta["orthogonalization_time"])
        stats["attempt_omega_target"].append(omega_it)
        stats["attempt_lambda_before"].append(lambda_value)
        stats["attempt_lambda_initial_guess"].append(lambda_ini)
        stats["attempt_predictor_kind"].append(predictor_kind)
        stats["attempt_predictor_basis_dim"].append(float(predictor_info.get("basis_dim", np.nan)))
        stats["attempt_predictor_wall_time"].append(float(predictor_info.get("predictor_wall_time", np.nan)))
        stats["attempt_predictor_alpha"].append(float(predictor_info.get("predictor_alpha", np.nan)))
        stats["attempt_predictor_beta"].append(float(predictor_info.get("predictor_beta", np.nan)))
        stats["attempt_predictor_gamma"].append(float(predictor_info.get("predictor_gamma", np.nan)))
        stats["attempt_predictor_projected_delta_lambda"].append(float(predictor_info.get("projected_delta_lambda", np.nan)))
        stats["attempt_predictor_projected_correction_norm"].append(float(predictor_info.get("projected_correction_norm", np.nan)))
        stats["attempt_predictor_energy_eval_count"].append(float(predictor_info.get("energy_eval_count", np.nan)))
        stats["attempt_predictor_energy_value"].append(float(predictor_info.get("energy_value", np.nan)))
        stats["attempt_predictor_coarse_solve_wall_time"].append(float(predictor_info.get("coarse_solve_wall_time", np.nan)))
        stats["attempt_predictor_coarse_newton_iterations"].append(float(predictor_info.get("coarse_newton_iterations", np.nan)))
        stats["attempt_predictor_coarse_residual_end"].append(float(predictor_info.get("coarse_residual_end", np.nan)))
        stats["attempt_predictor_basis_condition"].append(float(predictor_info.get("basis_condition", np.nan)))
        stats["attempt_predictor_reduced_newton_iterations"].append(float(predictor_info.get("reduced_newton_iterations", np.nan)))
        stats["attempt_predictor_reduced_gmres_iterations"].append(float(predictor_info.get("reduced_gmres_iterations", np.nan)))
        stats["attempt_predictor_reduced_projected_residual"].append(float(predictor_info.get("reduced_projected_residual", np.nan)))
        stats["attempt_predictor_reduced_omega_residual"].append(float(predictor_info.get("reduced_omega_residual", np.nan)))
        stats["attempt_predictor_state_coefficients"].append(predictor_info.get("state_coefficients"))
        stats["attempt_predictor_state_coefficients_ref"].append(predictor_info.get("state_coefficients_ref"))
        stats["attempt_predictor_state_coefficient_sum"].append(float(predictor_info.get("state_coefficient_sum", np.nan)))
        stats["attempt_secant_correction_active"].append(bool(predictor_info.get("secant_correction_active", False)))
        stats["attempt_secant_correction_basis_dim"].append(float(predictor_info.get("secant_correction_basis_dim", np.nan)))
        stats["attempt_secant_correction_trust_region_clipped"].append(
            bool(predictor_info.get("secant_correction_trust_region_clipped", False))
        )
        stats["attempt_secant_correction_predicted_residual_decrease"].append(
            float(predictor_info.get("secant_correction_predicted_residual_decrease", np.nan))
        )
        stats["attempt_first_newton_warm_start_active"].append(bool(attempt_first_newton_warm_start_active))
        stats["attempt_first_newton_warm_start_basis_dim"].append(float(attempt_first_newton_warm_start_basis_dim))
        stats["attempt_first_newton_linear_iterations"].append(float(attempt_first_newton_linear_iterations))
        stats["attempt_first_newton_linear_solve_time"].append(float(attempt_first_newton_linear_solve_time))
        stats["attempt_first_newton_linear_preconditioner_time"].append(
            float(attempt_first_newton_linear_preconditioner_time)
        )
        stats["attempt_first_newton_linear_orthogonalization_time"].append(
            float(attempt_first_newton_linear_orthogonalization_time)
        )
        stats["attempt_first_newton_correction_norm"].append(float(attempt_first_newton_correction_norm))
        stats["attempt_predictor_fallback_used"].append(bool(predictor_info.get("fallback_used", False)))
        stats["attempt_predictor_error"].append(
            "" if predictor_info.get("fallback_error") is None else str(predictor_info.get("fallback_error"))
        )
        stats["attempt_lambda_after"].append(lambda_candidate if flag == 0 else np.nan)
        stats["attempt_lambda_initial_guess_abs_error"].append(
            abs(float(lambda_candidate) - float(lambda_ini)) if flag == 0 else np.nan
        )
        stats["attempt_initial_guess_displacement_diff_volume_integral"].append(attempt_u_diff)
        stats["attempt_initial_guess_deviatoric_strain_diff_volume_integral"].append(attempt_dev_diff)
        stats["attempt_secant_reference_displacement_diff_volume_integral"].append(attempt_secant_u_diff)
        stats["attempt_secant_reference_deviatoric_strain_diff_volume_integral"].append(attempt_secant_dev_diff)
        stats["attempt_precision_mode"].append(str(current_precision_mode))
        stats["attempt_stopping_criterion"].append(str(current_stop_criterion))
        stats["attempt_stopping_tol"].append(np.nan if current_stop_tol is None else float(current_stop_tol))
        stats["attempt_fine_switch_triggered"].append(bool(fine_switch_triggered))
        stats["attempt_fine_switch_cumulative_distance"].append(float(fine_switch_cumulative_distance))
        stats["attempt_fine_switch_current_length"].append(float(fine_switch_current_length))
        stats["attempt_fine_switch_distance_with_current"].append(float(fine_switch_distance_with_current))
        stats["attempt_fine_switch_threshold"].append(float(fine_switch_threshold))
        stats["attempt_fine_reference_step"].append(int(fine_reference_step))
        stats["attempt_fine_reference_lambda"].append(float(fine_reference_lambda))
        stats["attempt_fine_reference_omega"].append(float(fine_reference_omega))

        _emit(
            "attempt_complete",
            phase="continuation",
            target_step=int(step + 1),
            accepted_steps=int(step),
            attempt_in_step=int(attempt_in_step),
            success=bool(flag == 0),
            omega_target=float(omega_it),
            lambda_before=float(lambda_value),
            lambda_initial_guess=float(lambda_ini),
            predictor_kind=str(predictor_kind),
            predictor_error=None if predictor_info.get("fallback_error") is None else str(predictor_info.get("fallback_error")),
            predictor_wall_time=None
            if np.isnan(float(predictor_info.get("predictor_wall_time", np.nan)))
            else float(predictor_info.get("predictor_wall_time", np.nan)),
            predictor_basis_dim=None
            if np.isnan(float(predictor_info.get("basis_dim", np.nan)))
            else float(predictor_info.get("basis_dim", np.nan)),
            predictor_basis_condition=None
            if np.isnan(float(predictor_info.get("basis_condition", np.nan)))
            else float(predictor_info.get("basis_condition", np.nan)),
            predictor_fallback_used=bool(predictor_info.get("fallback_used", False)),
            predictor_reduced_newton_iterations=None
            if np.isnan(float(predictor_info.get("reduced_newton_iterations", np.nan)))
            else float(predictor_info.get("reduced_newton_iterations", np.nan)),
            predictor_reduced_projected_residual=None
            if np.isnan(float(predictor_info.get("reduced_projected_residual", np.nan)))
            else float(predictor_info.get("reduced_projected_residual", np.nan)),
            predictor_state_coefficients=predictor_info.get("state_coefficients"),
            predictor_state_coefficients_ref=predictor_info.get("state_coefficients_ref"),
            predictor_state_coefficient_sum=None
            if np.isnan(float(predictor_info.get("state_coefficient_sum", np.nan)))
            else float(predictor_info.get("state_coefficient_sum", np.nan)),
            secant_correction_active=bool(predictor_info.get("secant_correction_active", False)),
            secant_correction_basis_dim=None
            if np.isnan(float(predictor_info.get("secant_correction_basis_dim", np.nan)))
            else float(predictor_info.get("secant_correction_basis_dim", np.nan)),
            secant_correction_trust_region_clipped=bool(
                predictor_info.get("secant_correction_trust_region_clipped", False)
            ),
            secant_correction_predicted_residual_decrease=None
            if np.isnan(float(predictor_info.get("secant_correction_predicted_residual_decrease", np.nan)))
            else float(predictor_info.get("secant_correction_predicted_residual_decrease", np.nan)),
            first_newton_warm_start_active=bool(attempt_first_newton_warm_start_active),
            first_newton_warm_start_basis_dim=int(attempt_first_newton_warm_start_basis_dim),
            first_newton_linear_iterations=None
            if np.isnan(float(attempt_first_newton_linear_iterations))
            else float(attempt_first_newton_linear_iterations),
            first_newton_linear_solve_time=None
            if np.isnan(float(attempt_first_newton_linear_solve_time))
            else float(attempt_first_newton_linear_solve_time),
            first_newton_linear_preconditioner_time=None
            if np.isnan(float(attempt_first_newton_linear_preconditioner_time))
            else float(attempt_first_newton_linear_preconditioner_time),
            first_newton_linear_orthogonalization_time=None
            if np.isnan(float(attempt_first_newton_linear_orthogonalization_time))
            else float(attempt_first_newton_linear_orthogonalization_time),
            first_newton_correction_norm=None
            if np.isnan(float(attempt_first_newton_correction_norm))
            else float(attempt_first_newton_correction_norm),
            precision_mode=str(current_precision_mode),
            stopping_criterion=str(current_stop_criterion),
            stopping_tol=None if current_stop_tol is None else float(current_stop_tol),
            fine_switch_mode=str(fine_switch_mode_name),
            fine_switch_triggered=bool(fine_switch_triggered),
            fine_switch_cumulative_distance=None
            if np.isnan(fine_switch_cumulative_distance)
            else float(fine_switch_cumulative_distance),
            fine_switch_current_length=None if np.isnan(fine_switch_current_length) else float(fine_switch_current_length),
            fine_switch_distance_with_current=None
            if np.isnan(fine_switch_distance_with_current)
            else float(fine_switch_distance_with_current),
            fine_switch_threshold=None if np.isnan(fine_switch_threshold) else float(fine_switch_threshold),
            fine_reference_step=int(fine_reference_step),
            fine_reference_lambda=float(fine_reference_lambda),
            fine_reference_omega=float(fine_reference_omega),
            lambda_after=float(lambda_candidate) if flag == 0 else None,
            newton_iterations=int(it_newt),
            newton_flag=int(flag),
            newton_relres_end=None if np.isnan(attempt_relres) else float(attempt_relres),
            newton_relcorr_end=None if np.isnan(attempt_relcorr) else float(attempt_relcorr),
            linear_iterations=int(attempt_delta["iterations"]),
            linear_solve_time=float(attempt_delta["solve_time"]),
            linear_preconditioner_time=float(attempt_delta["preconditioner_time"]),
            linear_orthogonalization_time=float(attempt_delta["orthogonalization_time"]),
            initial_guess_displacement_diff_volume_integral=None if np.isnan(attempt_u_diff) else float(attempt_u_diff),
            initial_guess_deviatoric_strain_diff_volume_integral=None if np.isnan(attempt_dev_diff) else float(attempt_dev_diff),
            secant_reference_displacement_diff_volume_integral=None if np.isnan(attempt_secant_u_diff) else float(attempt_secant_u_diff),
            secant_reference_deviatoric_strain_diff_volume_integral=None if np.isnan(attempt_secant_dev_diff) else float(attempt_secant_dev_diff),
            attempt_wall_time=float(attempt_wall),
            total_wall_time=float(perf_counter() - t_total),
        )

        step_wall_accum += attempt_wall
        step_lin_it_accum += attempt_delta["iterations"]
        step_lin_solve_accum += attempt_delta["solve_time"]
        step_lin_prec_accum += attempt_delta["preconditioner_time"]
        step_lin_orth_accum += attempt_delta["orthogonalization_time"]
        step_newton_it_accum += it_newt
        step_attempt_count += 1

        if flag == 1:
            d_omega /= 2.0
            n_omega += 1
        else:
            step += 1
            U_old = U
            U = U_it
            if getattr(linear_system_solver, "supports_dynamic_deflation_basis", lambda: True)():
                linear_system_solver.expand_deflation_basis(_free(U, Q))
            omega_old = omega
            omega = omega_it
            d_lambda = lambda_candidate - lambda_value
            lambda_value = lambda_candidate
            d_lambda_diff_scaled = np.nan if d_omega == 0.0 else (d_lambda / d_omega) * (omega - omega_hist[0])
            n_omega = 0

            lambda_hist[step - 1] = lambda_value
            omega_hist[step - 1] = omega
            Umax_hist[step - 1] = np.max(np.linalg.norm(U, axis=0))

            stats["step_index"].append(step)
            stats["step_attempt_count"].append(step_attempt_count)
            stats["step_wall_time"].append(step_wall_accum)
            stats["step_newton_iterations"].append(it_newt)
            stats["step_newton_iterations_total"].append(step_newton_it_accum)
            stats["step_newton_relres_end"].append(attempt_relres)
            stats["step_newton_relcorr_end"].append(attempt_relcorr)
            stats["step_linear_iterations"].append(step_lin_it_accum)
            stats["step_linear_solve_time"].append(step_lin_solve_accum)
            stats["step_linear_preconditioner_time"].append(step_lin_prec_accum)
            stats["step_linear_orthogonalization_time"].append(step_lin_orth_accum)
            stats["step_lambda"].append(lambda_value)
            stats["step_omega"].append(omega)
            stats["step_lambda_initial_guess"].append(lambda_ini)
            stats["step_predictor_kind"].append(predictor_kind)
            stats["step_predictor_basis_dim"].append(float(predictor_info.get("basis_dim", np.nan)))
            stats["step_predictor_wall_time"].append(float(predictor_info.get("predictor_wall_time", np.nan)))
            stats["step_predictor_alpha"].append(float(predictor_info.get("predictor_alpha", np.nan)))
            stats["step_predictor_beta"].append(float(predictor_info.get("predictor_beta", np.nan)))
            stats["step_predictor_gamma"].append(float(predictor_info.get("predictor_gamma", np.nan)))
            stats["step_predictor_projected_delta_lambda"].append(float(predictor_info.get("projected_delta_lambda", np.nan)))
            stats["step_predictor_projected_correction_norm"].append(float(predictor_info.get("projected_correction_norm", np.nan)))
            stats["step_predictor_energy_eval_count"].append(float(predictor_info.get("energy_eval_count", np.nan)))
            stats["step_predictor_energy_value"].append(float(predictor_info.get("energy_value", np.nan)))
            stats["step_predictor_coarse_solve_wall_time"].append(float(predictor_info.get("coarse_solve_wall_time", np.nan)))
            stats["step_predictor_coarse_newton_iterations"].append(float(predictor_info.get("coarse_newton_iterations", np.nan)))
            stats["step_predictor_coarse_residual_end"].append(float(predictor_info.get("coarse_residual_end", np.nan)))
            stats["step_predictor_basis_condition"].append(float(predictor_info.get("basis_condition", np.nan)))
            stats["step_predictor_reduced_newton_iterations"].append(float(predictor_info.get("reduced_newton_iterations", np.nan)))
            stats["step_predictor_reduced_gmres_iterations"].append(float(predictor_info.get("reduced_gmres_iterations", np.nan)))
            stats["step_predictor_reduced_projected_residual"].append(float(predictor_info.get("reduced_projected_residual", np.nan)))
            stats["step_predictor_reduced_omega_residual"].append(float(predictor_info.get("reduced_omega_residual", np.nan)))
            stats["step_predictor_state_coefficients"].append(predictor_info.get("state_coefficients"))
            stats["step_predictor_state_coefficients_ref"].append(predictor_info.get("state_coefficients_ref"))
            stats["step_predictor_state_coefficient_sum"].append(float(predictor_info.get("state_coefficient_sum", np.nan)))
            stats["step_secant_correction_active"].append(bool(predictor_info.get("secant_correction_active", False)))
            stats["step_secant_correction_basis_dim"].append(float(predictor_info.get("secant_correction_basis_dim", np.nan)))
            stats["step_secant_correction_trust_region_clipped"].append(
                bool(predictor_info.get("secant_correction_trust_region_clipped", False))
            )
            stats["step_secant_correction_predicted_residual_decrease"].append(
                float(predictor_info.get("secant_correction_predicted_residual_decrease", np.nan))
            )
            stats["step_first_newton_warm_start_active"].append(bool(attempt_first_newton_warm_start_active))
            stats["step_first_newton_warm_start_basis_dim"].append(float(attempt_first_newton_warm_start_basis_dim))
            stats["step_first_newton_linear_iterations"].append(float(attempt_first_newton_linear_iterations))
            stats["step_first_newton_linear_solve_time"].append(float(attempt_first_newton_linear_solve_time))
            stats["step_first_newton_linear_preconditioner_time"].append(
                float(attempt_first_newton_linear_preconditioner_time)
            )
            stats["step_first_newton_linear_orthogonalization_time"].append(
                float(attempt_first_newton_linear_orthogonalization_time)
            )
            stats["step_first_newton_correction_norm"].append(float(attempt_first_newton_correction_norm))
            stats["step_predictor_fallback_used"].append(bool(predictor_info.get("fallback_used", False)))
            stats["step_predictor_error"].append(
                "" if predictor_info.get("fallback_error") is None else str(predictor_info.get("fallback_error"))
            )
            stats["step_initial_guess_displacement_diff_volume_integral"].append(attempt_u_diff)
            stats["step_initial_guess_deviatoric_strain_diff_volume_integral"].append(attempt_dev_diff)
            stats["step_secant_reference_displacement_diff_volume_integral"].append(attempt_secant_u_diff)
            stats["step_secant_reference_deviatoric_strain_diff_volume_integral"].append(attempt_secant_dev_diff)
            stats["step_lambda_initial_guess_abs_error"].append(abs(float(lambda_value) - float(lambda_ini)))
            stats["step_precision_mode"].append(str(current_precision_mode))
            stats["step_stopping_criterion"].append(str(current_stop_criterion))
            stats["step_stopping_tol"].append(np.nan if current_stop_tol is None else float(current_stop_tol))
            stats["step_fine_switch_triggered"].append(bool(fine_switch_triggered))
            stats["step_fine_switch_cumulative_distance"].append(float(fine_switch_cumulative_distance))
            stats["step_fine_switch_current_length"].append(float(fine_switch_current_length))
            stats["step_fine_switch_distance_with_current"].append(float(fine_switch_distance_with_current))
            stats["step_fine_switch_threshold"].append(float(fine_switch_threshold))
            stats["step_fine_reference_step"].append(int(fine_reference_step))
            stats["step_fine_reference_lambda"].append(float(fine_reference_lambda))
            stats["step_fine_reference_omega"].append(float(fine_reference_omega))
            if store_step_u:
                stats["step_U"].append(U.copy())

            _emit(
                "step_accepted",
                phase="continuation",
                accepted_step=int(step),
                step_attempt_count=int(step_attempt_count),
                lambda_value=float(lambda_value),
                lambda_initial_guess=float(lambda_ini),
                predictor_kind=str(predictor_kind),
                predictor_error=None if predictor_info.get("fallback_error") is None else str(predictor_info.get("fallback_error")),
                predictor_wall_time=None
                if np.isnan(float(predictor_info.get("predictor_wall_time", np.nan)))
                else float(predictor_info.get("predictor_wall_time", np.nan)),
                predictor_basis_dim=None
                if np.isnan(float(predictor_info.get("basis_dim", np.nan)))
                else float(predictor_info.get("basis_dim", np.nan)),
                predictor_basis_condition=None
                if np.isnan(float(predictor_info.get("basis_condition", np.nan)))
                else float(predictor_info.get("basis_condition", np.nan)),
                predictor_fallback_used=bool(predictor_info.get("fallback_used", False)),
                predictor_reduced_newton_iterations=None
                if np.isnan(float(predictor_info.get("reduced_newton_iterations", np.nan)))
                else float(predictor_info.get("reduced_newton_iterations", np.nan)),
                predictor_reduced_projected_residual=None
                if np.isnan(float(predictor_info.get("reduced_projected_residual", np.nan)))
                else float(predictor_info.get("reduced_projected_residual", np.nan)),
                predictor_state_coefficients=predictor_info.get("state_coefficients"),
                predictor_state_coefficients_ref=predictor_info.get("state_coefficients_ref"),
                predictor_state_coefficient_sum=None
                if np.isnan(float(predictor_info.get("state_coefficient_sum", np.nan)))
                else float(predictor_info.get("state_coefficient_sum", np.nan)),
                secant_correction_active=bool(predictor_info.get("secant_correction_active", False)),
                secant_correction_basis_dim=None
                if np.isnan(float(predictor_info.get("secant_correction_basis_dim", np.nan)))
                else float(predictor_info.get("secant_correction_basis_dim", np.nan)),
                secant_correction_trust_region_clipped=bool(
                    predictor_info.get("secant_correction_trust_region_clipped", False)
                ),
                secant_correction_predicted_residual_decrease=None
                if np.isnan(float(predictor_info.get("secant_correction_predicted_residual_decrease", np.nan)))
                else float(predictor_info.get("secant_correction_predicted_residual_decrease", np.nan)),
                first_newton_warm_start_active=bool(attempt_first_newton_warm_start_active),
                first_newton_warm_start_basis_dim=int(attempt_first_newton_warm_start_basis_dim),
                first_newton_linear_iterations=None
                if np.isnan(float(attempt_first_newton_linear_iterations))
                else float(attempt_first_newton_linear_iterations),
                first_newton_linear_solve_time=None
                if np.isnan(float(attempt_first_newton_linear_solve_time))
                else float(attempt_first_newton_linear_solve_time),
                first_newton_linear_preconditioner_time=None
                if np.isnan(float(attempt_first_newton_linear_preconditioner_time))
                else float(attempt_first_newton_linear_preconditioner_time),
                first_newton_linear_orthogonalization_time=None
                if np.isnan(float(attempt_first_newton_linear_orthogonalization_time))
                else float(attempt_first_newton_linear_orthogonalization_time),
                first_newton_correction_norm=None
                if np.isnan(float(attempt_first_newton_correction_norm))
                else float(attempt_first_newton_correction_norm),
                precision_mode=str(current_precision_mode),
                stopping_criterion=str(current_stop_criterion),
                stopping_tol=None if current_stop_tol is None else float(current_stop_tol),
                fine_switch_mode=str(fine_switch_mode_name),
                fine_switch_triggered=bool(fine_switch_triggered),
                fine_switch_cumulative_distance=None
                if np.isnan(fine_switch_cumulative_distance)
                else float(fine_switch_cumulative_distance),
                fine_switch_current_length=None if np.isnan(fine_switch_current_length) else float(fine_switch_current_length),
                fine_switch_distance_with_current=None
                if np.isnan(fine_switch_distance_with_current)
                else float(fine_switch_distance_with_current),
                fine_switch_threshold=None if np.isnan(fine_switch_threshold) else float(fine_switch_threshold),
                fine_reference_step=int(fine_reference_step),
                fine_reference_lambda=float(fine_reference_lambda),
                fine_reference_omega=float(fine_reference_omega),
                d_lambda=float(d_lambda),
                d_lambda_diff_scaled=None if np.isnan(d_lambda_diff_scaled) else float(d_lambda_diff_scaled),
                omega_value=float(omega),
                d_omega=float(d_omega),
                u_max=float(Umax_hist[step - 1]),
                step_wall_time=float(step_wall_accum),
                step_newton_iterations=int(it_newt),
                step_newton_iterations_total=int(step_newton_it_accum),
                step_newton_relres_end=None if np.isnan(attempt_relres) else float(attempt_relres),
                step_newton_relcorr_end=None if np.isnan(attempt_relcorr) else float(attempt_relcorr),
                step_linear_iterations=int(step_lin_it_accum),
                step_linear_solve_time=float(step_lin_solve_accum),
                step_linear_preconditioner_time=float(step_lin_prec_accum),
                step_linear_orthogonalization_time=float(step_lin_orth_accum),
                initial_guess_displacement_diff_volume_integral=None if np.isnan(attempt_u_diff) else float(attempt_u_diff),
                initial_guess_deviatoric_strain_diff_volume_integral=None if np.isnan(attempt_dev_diff) else float(attempt_dev_diff),
                secant_reference_displacement_diff_volume_integral=None if np.isnan(attempt_secant_u_diff) else float(attempt_secant_u_diff),
                secant_reference_deviatoric_strain_diff_volume_integral=None if np.isnan(attempt_secant_dev_diff) else float(attempt_secant_dev_diff),
                total_wall_time=float(perf_counter() - t_total),
            )
            if current_precision_mode == "fine":
                last_fine_hist_idx = int(step - 1)

            predictor_omega_hist.append(float(omega))
            predictor_lambda_hist.append(float(lambda_value))
            predictor_u_hist.append(U.copy())
            continuation_increment_free_hist.append(_free(U - U_old, Q))
            first_newton_correction_free = np.asarray(history.get("first_accepted_correction_free", np.zeros(0, dtype=np.float64)), dtype=np.float64)
            if first_newton_correction_free.size:
                continuation_first_newton_correction_free_hist.append(first_newton_correction_free.reshape(-1).copy())
                if len(continuation_first_newton_correction_free_hist) > 2:
                    continuation_first_newton_correction_free_hist.pop(0)
            continuation_state_hist_only.append(U.copy())
            continuation_u_hist_full.append(U.copy())
            continuation_omega_hist_full.append(float(omega))
            continuation_lambda_hist_full.append(float(lambda_value))
            continuation_increment_full_hist.append(np.asarray(U - U_old, dtype=np.float64))
            if len(predictor_omega_hist) > 3:
                predictor_omega_hist.pop(0)
                predictor_lambda_hist.pop(0)
                predictor_u_hist.pop(0)

            accepted_step_newton_total = int(step_newton_it_accum)
            accepted_step_linear_total = int(step_lin_it_accum)
            accepted_step_d_omega = float(d_omega)
            branch_efficiency = np.nan
            if accepted_step_d_omega != 0.0:
                branch_efficiency = float(abs(d_lambda / accepted_step_d_omega))
            branch_efficiency_ref = np.nan
            if omega_efficiency_drop_ratio is not None:
                branch_efficiency_ref_value = _positive_median(step_branch_efficiency_hist[-max(int(omega_efficiency_window), 1) :])
                if branch_efficiency_ref_value is not None:
                    branch_efficiency_ref = float(branch_efficiency_ref_value)

            step_wall_accum = 0.0
            step_lin_it_accum = 0
            step_lin_solve_accum = 0.0
            step_lin_prec_accum = 0.0
            step_lin_orth_accum = 0.0
            step_newton_it_accum = 0
            step_attempt_count = 0

            branch_shape_requests_increase = (lambda_hist[step - 1] - lambda_hist[step - 2]) < 0.9 * (
                lambda_hist[step - 2] - lambda_hist[step - 3]
            )
            controller_enabled = omega_step_controller_mode == "adaptive"
            growth_blocked = False
            hard_mode = False
            if controller_enabled:
                adapt_min_scale = 0.7 if omega_adapt_min_scale is None else float(omega_adapt_min_scale)
                adapt_max_scale = 1.25 if omega_adapt_max_scale is None else float(omega_adapt_max_scale)
                hard_shrink_scale = 0.7 if omega_hard_shrink_scale is None else float(omega_hard_shrink_scale)
                target_newton = 12.0 if omega_target_newton_iterations is None else float(omega_target_newton_iterations)
                proposed_scale = 1.0
                if accepted_step_newton_total > 0:
                    proposed_scale = float(np.sqrt(target_newton / float(accepted_step_newton_total)))
                    proposed_scale = float(np.clip(proposed_scale, adapt_min_scale, adapt_max_scale))

                if (
                    omega_no_increase_newton_threshold is not None
                    and accepted_step_newton_total > int(omega_no_increase_newton_threshold)
                ):
                    growth_blocked = True
                if (
                    omega_efficiency_drop_ratio is not None
                    and np.isfinite(branch_efficiency)
                    and np.isfinite(branch_efficiency_ref)
                    and branch_efficiency < float(omega_efficiency_drop_ratio) * branch_efficiency_ref
                ):
                    growth_blocked = True

                if omega_half_newton_threshold is not None and accepted_step_newton_total > int(omega_half_newton_threshold):
                    hard_mode = True
                if omega_hard_newton_threshold is not None and accepted_step_newton_total > int(omega_hard_newton_threshold):
                    hard_mode = True
                if omega_hard_linear_threshold is not None and accepted_step_linear_total > int(omega_hard_linear_threshold):
                    hard_mode = True
                if (
                    omega_efficiency_floor is not None
                    and np.isfinite(branch_efficiency)
                    and branch_efficiency < float(omega_efficiency_floor)
                ):
                    hard_mode = True

                if hard_mode:
                    proposed_scale = min(proposed_scale, hard_shrink_scale)
                elif not branch_shape_requests_increase or growth_blocked:
                    proposed_scale = min(proposed_scale, 1.0)

                d_omega *= proposed_scale
                step_d_omega_scale = float(proposed_scale)
            else:
                if branch_shape_requests_increase:
                    d_omega *= 2.0
                step_d_omega_scale = float(d_omega / accepted_step_d_omega) if accepted_step_d_omega != 0.0 else 1.0

            step_length_cap_applied = False
            step_length_cap_raw_length = np.nan
            step_length_cap_limit = np.nan
            if step_length_cap_mode_name in {"initial_segment", "history_box"} and step >= 3:
                if step_length_cap_mode_name == "initial_segment":
                    d_omega_capped, step_length_cap_raw_length, step_length_cap_limit = _initial_segment_length_cap(
                        domega_candidate=float(d_omega),
                        domega_initial=float(omega_hist[1] - omega_hist[0]),
                        dlambda_initial=float(lambda_hist[1] - lambda_hist[0]),
                        omega_anchor_prev=float(omega_old),
                        omega_anchor_curr=float(omega),
                        lambda_anchor_prev=float(lambda_hist[step - 2]),
                        lambda_anchor_curr=float(lambda_hist[step - 1]),
                        cap_factor=float(step_length_cap_factor),
                    )
                else:
                    d_omega_capped, step_length_cap_raw_length, step_length_cap_limit = _history_box_step_length_cap(
                        domega_candidate=float(d_omega),
                        omega_hist=np.asarray(omega_hist[:step], dtype=np.float64),
                        lambda_hist=np.asarray(lambda_hist[:step], dtype=np.float64),
                        omega_anchor_prev=float(omega_old),
                        omega_anchor_curr=float(omega),
                        lambda_anchor_prev=float(lambda_hist[step - 2]),
                        lambda_anchor_curr=float(lambda_hist[step - 1]),
                        cap_factor=float(step_length_cap_factor),
                    )
                step_length_cap_applied = bool(d_omega_capped + 1.0e-12 < float(d_omega))
                d_omega = float(d_omega_capped)

            stats["step_d_omega"].append(accepted_step_d_omega)
            stats["step_next_d_omega"].append(float(d_omega))
            stats["step_d_omega_scale"].append(float(step_d_omega_scale))
            stats["step_branch_efficiency"].append(float(branch_efficiency))
            stats["step_branch_efficiency_ref"].append(float(branch_efficiency_ref))
            stats["step_growth_blocked"].append(bool(growth_blocked))
            stats["step_hard_mode"].append(bool(hard_mode))
            stats["step_length_cap_applied"].append(bool(step_length_cap_applied))
            stats["step_length_cap_raw_length"].append(float(step_length_cap_raw_length))
            stats["step_length_cap_limit"].append(float(step_length_cap_limit))

            if np.isfinite(branch_efficiency) and branch_efficiency > 0.0:
                step_branch_efficiency_hist.append(float(branch_efficiency))

            if d_lambda_diff_scaled_min > 0.0 and d_lambda_diff_scaled < d_lambda_diff_scaled_min:
                stop_reason = "d_lambda_diff_scaled_min"
                break
            if omega >= omega_max_stop:
                stop_reason = "omega_max_stop"
                break

        if n_omega >= n_omega_max:
            stop_reason = "omega_reduction_limit"
            break
        if step >= step_max:
            stop_reason = "step_max"
            break

    lambda_hist = lambda_hist[:step]
    omega_hist = omega_hist[:step]
    Umax_hist = Umax_hist[:step]
    stats["total_wall_time"] = perf_counter() - t_total

    _emit(
        "finished",
        phase="continuation",
        accepted_steps=int(step),
        lambda_last=float(lambda_hist[-1]),
        omega_last=float(omega_hist[-1]),
        stop_reason=str(stop_reason),
        total_wall_time=float(stats["total_wall_time"]),
    )

    return U, lambda_hist, omega_hist, Umax_hist, stats
