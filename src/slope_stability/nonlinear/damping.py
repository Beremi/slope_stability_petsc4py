"""Damping and line-search helpers for nonlinear solvers."""

from __future__ import annotations

import numpy as np
from mpi4py import MPI

from ..utils import q_to_free_indices


def _is_invalid_lambda_trial(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "Reduction parameter lambda must be positive" in str(exc)


def _flat(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1, order="F")


def _extract(value: np.ndarray, q_mask: np.ndarray | None) -> np.ndarray:
    arr = _flat(value)
    if q_mask is None:
        return arr
    return arr[q_to_free_indices(q_mask)]


def _dot(x: np.ndarray, y: np.ndarray, q_mask: np.ndarray | None = None) -> float:
    return float(np.dot(_extract(x, q_mask), _extract(y, q_mask)))


def _norm(x: np.ndarray, q_mask: np.ndarray | None = None) -> float:
    return float(np.linalg.norm(_extract(x, q_mask)))


def _dist_dot_local(x_local: np.ndarray, y_local: np.ndarray, comm) -> float:
    value = float(np.dot(np.asarray(x_local, dtype=np.float64).reshape(-1), np.asarray(y_local, dtype=np.float64).reshape(-1)))
    if comm is None:
        return value
    return float(comm.allreduce(value, op=MPI.SUM))


def _dist_norm_local(x_local: np.ndarray, comm) -> float:
    return float(np.sqrt(max(_dist_dot_local(x_local, x_local, comm), 0.0)))


def _is_local_rhs(value: object) -> bool:
    return hasattr(value, "owned_free_rows") and hasattr(value, "dot_field")


def _rhs_array(value) -> np.ndarray:
    if _is_local_rhs(value):
        return np.asarray(value.materialize_full(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _rhs_dot_field(value, field: np.ndarray, q_mask: np.ndarray | None, constitutive_matrix_builder=None) -> float:
    if _is_local_rhs(value):
        pattern = getattr(constitutive_matrix_builder, "owned_tangent_pattern", None)
        return float(value.dot_field(field, pattern=pattern))
    return _dot(np.asarray(value, dtype=np.float64), field, q_mask=q_mask)


def _line_search_result(alpha: float, iterations: int) -> dict[str, float | int]:
    return {
        "alpha": float(alpha),
        "line_search_iterations": int(iterations),
    }


def _indirect_trial_residual_norm(
    *,
    U_it: np.ndarray,
    d_U: np.ndarray,
    lambda_it: float,
    d_l: float,
    alpha: float,
    f: np.ndarray,
    q_mask: np.ndarray,
    constitutive_matrix_builder,
    f_free: np.ndarray | None = None,
    f_local_free: np.ndarray | None = None,
    comm=None,
    omega_target: float | None = None,
    rescale_trial_to_omega: bool = False,
) -> float:
    alpha = float(alpha)
    U_alpha = np.asarray(U_it, dtype=np.float64) + alpha * np.asarray(d_U, dtype=np.float64)
    lambda_alpha = float(lambda_it) + alpha * float(d_l)
    if lambda_alpha <= 0.0:
        return np.inf

    if bool(rescale_trial_to_omega) and omega_target is not None:
        denom = _rhs_dot_field(f, U_alpha, q_mask=q_mask, constitutive_matrix_builder=constitutive_matrix_builder)
        if not np.isfinite(denom) or abs(float(denom)) <= 1.0e-30:
            return np.inf
        U_alpha = U_alpha * (float(omega_target) / float(denom))

    build_F_all_free_local = getattr(constitutive_matrix_builder, "build_F_all_free_local", None)
    build_F_all_free = getattr(constitutive_matrix_builder, "build_F_all_free", None)
    try:
        if f_local_free is not None and callable(build_F_all_free_local):
            F_alpha_local_free = np.asarray(build_F_all_free_local(lambda_alpha, U_alpha), dtype=np.float64).reshape(-1)
            return _dist_norm_local(
                F_alpha_local_free - np.asarray(f_local_free, dtype=np.float64).reshape(-1),
                comm,
            )
        if f_free is not None and callable(build_F_all_free):
            F_alpha_free = np.asarray(build_F_all_free(lambda_alpha, U_alpha), dtype=np.float64).reshape(-1)
            return float(np.linalg.norm(F_alpha_free - np.asarray(f_free, dtype=np.float64).reshape(-1)))
        F_alpha = constitutive_matrix_builder.build_F_all(lambda_alpha, U_alpha)
        return _norm(F_alpha - _rhs_array(f), q_mask=q_mask)
    except Exception as exc:  # pragma: no cover - defensive for constitutive backends
        if _is_invalid_lambda_trial(exc):
            return np.inf
        raise


def _damping_alg5_monotone(
    it_damp_max: int,
    U_it: np.ndarray,
    lambda_it: float,
    d_U: np.ndarray,
    d_l: float,
    f: np.ndarray,
    criterion: float,
    q_mask: np.ndarray,
    constitutive_matrix_builder,
    *,
    f_free: np.ndarray | None = None,
    f_local_free: np.ndarray | None = None,
    comm=None,
) -> dict[str, float | int]:
    if np.isnan(d_l) or np.isinf(d_l):
        return _line_search_result(0.0, 0)
    if not np.isfinite(criterion):
        return _line_search_result(0.0, 0)
    if it_damp_max <= 0:
        return _line_search_result(0.0, 0)

    alpha = 1.0
    last_evaluated_alpha: float | None = None
    line_search_iterations = 0

    for _ in range(int(it_damp_max)):
        line_search_iterations += 1
        crit_alpha = _indirect_trial_residual_norm(
            U_it=U_it,
            d_U=d_U,
            lambda_it=lambda_it,
            d_l=d_l,
            alpha=alpha,
            f=f,
            q_mask=q_mask,
            constitutive_matrix_builder=constitutive_matrix_builder,
            f_free=f_free,
            f_local_free=f_local_free,
            comm=comm,
            omega_target=None,
            rescale_trial_to_omega=False,
        )
        if not np.isfinite(crit_alpha):
            alpha *= 0.5
            if alpha <= 0.0:
                return _line_search_result(0.0, int(line_search_iterations))
            continue
        last_evaluated_alpha = float(alpha)

        if crit_alpha < criterion:
            break

        alpha *= 0.5
        if alpha <= 0.0:
            return _line_search_result(0.0, int(line_search_iterations))

    if last_evaluated_alpha is None:
        return _line_search_result(0.0, int(line_search_iterations))
    return _line_search_result(
        float(alpha if alpha == last_evaluated_alpha else last_evaluated_alpha),
        int(line_search_iterations),
    )


def damping(
    it_damp_max: int,
    U_it: np.ndarray,
    dU: np.ndarray,
    F: np.ndarray | None,
    f: np.ndarray,
    constitutive_matrix_builder,
    q_mask: np.ndarray | None = None,
    *,
    F_free: np.ndarray | None = None,
    f_free: np.ndarray | None = None,
    F_local_free: np.ndarray | None = None,
    f_local_free: np.ndarray | None = None,
    dU_local_free: np.ndarray | None = None,
    comm=None,
    alpha_upper: float = 1.0,
    return_info: bool = False,
) -> float | dict[str, float | int]:
    """Line-search damping for plain Newton updates.

    The free-mask is optional. If provided, all checks are evaluated over active
    degrees of freedom in MATLAB's column-major order.
    """

    if it_damp_max < 0:
        result = _line_search_result(0.0, 0)
        return result if return_info else result["alpha"]

    U_it = np.asarray(U_it, dtype=np.float64)
    dU = np.asarray(dU, dtype=np.float64)
    F = None if F is None else np.asarray(F, dtype=np.float64)
    f_array = None if _is_local_rhs(f) else np.asarray(f, dtype=np.float64)
    if q_mask is not None:
        q_mask = np.asarray(q_mask, dtype=bool)

    if F_local_free is not None and f_local_free is not None and dU_local_free is not None:
        dU_eval = np.asarray(dU_local_free, dtype=np.float64).reshape(-1)
        residual_eval = np.asarray(F_local_free, dtype=np.float64).reshape(-1) - np.asarray(f_local_free, dtype=np.float64).reshape(-1)
        initial_decrease = _dist_dot_local(residual_eval, dU_eval, comm)
        dU_norm = _dist_norm_local(dU_eval, comm)
    else:
        dU_eval = _extract(dU, q_mask)
        dU_norm = float(np.linalg.norm(dU_eval))
        if F_free is not None and f_free is not None:
            residual_eval = np.asarray(F_free, dtype=np.float64).reshape(-1) - np.asarray(f_free, dtype=np.float64).reshape(-1)
        else:
            if F is None:
                raise ValueError("F must be provided when reduced residuals are unavailable")
            residual_eval = _extract(F - (f_array if f_array is not None else _rhs_array(f)), q_mask)
        initial_decrease = float(np.dot(residual_eval, dU_eval))
    if (
        np.isnan(initial_decrease)
        or not np.isfinite(dU_norm)
        or initial_decrease >= 0.0
    ):
        result = _line_search_result(0.0, 0)
        return result if return_info else result["alpha"]

    alpha_upper = float(alpha_upper)
    if not np.isfinite(alpha_upper) or alpha_upper <= 0.0:
        result = _line_search_result(0.0, 0)
        return result if return_info else result["alpha"]
    alpha = min(alpha_upper, 1.0)
    alpha_min = 0.0
    alpha_max = float(alpha)
    line_search_iterations = 0

    for _ in range(int(it_damp_max)):
        line_search_iterations += 1
        U_alpha = U_it + alpha * dU
        build_F_reduced_free_local = getattr(constitutive_matrix_builder, "build_F_reduced_free_local", None)
        build_F_reduced_free = getattr(constitutive_matrix_builder, "build_F_reduced_free", None)
        if (
            dU_local_free is not None
            and f_local_free is not None
            and callable(build_F_reduced_free_local)
        ):
            F_alpha_local_free = np.asarray(build_F_reduced_free_local(U_alpha), dtype=np.float64).reshape(-1)
            decrease = _dist_dot_local(
                F_alpha_local_free - np.asarray(f_local_free, dtype=np.float64).reshape(-1),
                dU_eval,
                comm,
            )
        elif q_mask is not None and f_free is not None and callable(build_F_reduced_free):
            F_alpha_free = np.asarray(build_F_reduced_free(U_alpha), dtype=np.float64).reshape(-1)
            decrease = float(np.dot(F_alpha_free - np.asarray(f_free, dtype=np.float64).reshape(-1), dU_eval))
        else:
            F_alpha = constitutive_matrix_builder.build_F_reduced(U_alpha)
            decrease = _dot(F_alpha - (f_array if f_array is not None else _rhs_array(f)), dU, q_mask=q_mask)

        if decrease < 0.0:
            if alpha == 1.0:
                break
            alpha_min = alpha
        else:
            alpha_max = alpha

        alpha = 0.5 * (alpha_min + alpha_max)

    result = _line_search_result(float(alpha), int(line_search_iterations))
    return result if return_info else result["alpha"]


def damping_alg5(
    it_damp_max: int,
    U_it: np.ndarray,
    lambda_it: float,
    d_U: np.ndarray,
    d_l: float,
    f: np.ndarray,
    criterion: float,
    q_mask: np.ndarray,
    constitutive_matrix_builder,
    *,
    f_free: np.ndarray | None = None,
    f_local_free: np.ndarray | None = None,
    comm=None,
    mode: str = "alg5",
    omega_target: float | None = None,
    armijo_alpha0: float = 1.0,
    armijo_c1: float = 1.0e-4,
    armijo_shrink: float = 0.5,
    armijo_max_ls: int | None = None,
    armijo_rescale_trial_to_omega: bool = True,
    armijo_fallback_to_alg5: bool = True,
    return_info: bool = False,
) -> float | dict[str, float | int]:
    """Line-search damping for nested-Newton (`ALG5`) continuation updates.

    `criterion` is the current residual in the constrained norm and must be
    computed on free degrees of freedom.
    """

    U_it = np.asarray(U_it, dtype=np.float64)
    d_U = np.asarray(d_U, dtype=np.float64)
    q_mask = np.asarray(q_mask, dtype=bool)
    mode_name = str(mode).strip().lower()
    if mode_name == "alg5":
        result = _damping_alg5_monotone(
            it_damp_max,
            U_it,
            lambda_it,
            d_U,
            d_l,
            f,
            criterion,
            q_mask,
            constitutive_matrix_builder,
            f_free=f_free,
            f_local_free=f_local_free,
            comm=comm,
        )
        result["line_search_mode"] = "alg5"
        result["armijo_accepted"] = False
        result["fallback_used"] = False
        return result if return_info else result["alpha"]

    if mode_name != "armijo_residual":
        raise ValueError(f"Unsupported indirect line-search mode {mode!r}")

    if np.isnan(d_l) or np.isinf(d_l) or not np.isfinite(criterion):
        result = _line_search_result(0.0, 0)
        result["line_search_mode"] = "armijo_residual"
        result["armijo_accepted"] = False
        result["fallback_used"] = False
        return result if return_info else result["alpha"]

    phi_it = 0.5 * float(criterion) * float(criterion)
    directional_derivative = -float(criterion) * float(criterion)
    max_ls = int(it_damp_max if armijo_max_ls is None else armijo_max_ls)
    if (
        max_ls <= 0
        or not np.isfinite(phi_it)
        or not np.isfinite(directional_derivative)
        or directional_derivative >= 0.0
        or not np.isfinite(float(armijo_shrink))
        or float(armijo_shrink) <= 0.0
        or float(armijo_shrink) >= 1.0
    ):
        result = _line_search_result(0.0, 0)
        result["line_search_mode"] = "armijo_residual"
        result["armijo_accepted"] = False
        result["fallback_used"] = False
        return result if return_info else result["alpha"]

    alpha_trial = min(max(float(armijo_alpha0), 1.0e-12), 1.0)
    line_search_iterations = 0
    accepted = False
    accepted_alpha = 0.0
    last_tried_alpha = 0.0

    for _ in range(max_ls):
        line_search_iterations += 1
        last_tried_alpha = float(alpha_trial)
        crit_alpha = _indirect_trial_residual_norm(
            U_it=U_it,
            d_U=d_U,
            lambda_it=lambda_it,
            d_l=d_l,
            alpha=alpha_trial,
            f=f,
            q_mask=q_mask,
            constitutive_matrix_builder=constitutive_matrix_builder,
            f_free=f_free,
            f_local_free=f_local_free,
            comm=comm,
            omega_target=omega_target,
            rescale_trial_to_omega=bool(armijo_rescale_trial_to_omega),
        )
        phi_alpha = 0.5 * float(crit_alpha) * float(crit_alpha)
        if np.isfinite(phi_alpha) and phi_alpha <= phi_it + float(armijo_c1) * float(alpha_trial) * directional_derivative:
            accepted = True
            accepted_alpha = float(alpha_trial)
            break
        next_alpha = float(alpha_trial) * float(armijo_shrink)
        if next_alpha <= 1.0e-12:
            alpha_trial = 1.0e-12
            break
        alpha_trial = next_alpha

    if accepted:
        result = _line_search_result(float(accepted_alpha), int(line_search_iterations))
        result["line_search_mode"] = "armijo_residual"
        result["armijo_accepted"] = True
        result["fallback_used"] = False
        result["last_tried_alpha"] = float(last_tried_alpha)
        return result if return_info else result["alpha"]

    if bool(armijo_fallback_to_alg5):
        fallback = _damping_alg5_monotone(
            it_damp_max,
            U_it,
            lambda_it,
            d_U,
            d_l,
            f,
            criterion,
            q_mask,
            constitutive_matrix_builder,
            f_free=f_free,
            f_local_free=f_local_free,
            comm=comm,
        )
        fallback["line_search_iterations"] = int(line_search_iterations) + int(fallback["line_search_iterations"])
        fallback["line_search_mode"] = "armijo_residual"
        fallback["armijo_accepted"] = False
        fallback["fallback_used"] = True
        fallback["last_tried_alpha"] = float(last_tried_alpha)
        result = fallback
        return result if return_info else result["alpha"]

    result = _line_search_result(0.0, int(line_search_iterations))
    result["line_search_mode"] = "armijo_residual"
    result["armijo_accepted"] = False
    result["fallback_used"] = False
    result["last_tried_alpha"] = float(last_tried_alpha)
    return result if return_info else result["alpha"]
