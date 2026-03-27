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
) -> float:
    """Line-search damping for plain Newton updates.

    The free-mask is optional. If provided, all checks are evaluated over active
    degrees of freedom in MATLAB's column-major order.
    """

    if it_damp_max < 0:
        return 0.0

    U_it = np.asarray(U_it, dtype=np.float64)
    dU = np.asarray(dU, dtype=np.float64)
    F = None if F is None else np.asarray(F, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
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
            residual_eval = _extract(F - f, q_mask)
        initial_decrease = float(np.dot(residual_eval, dU_eval))
    if (
        np.isnan(initial_decrease)
        or not np.isfinite(dU_norm)
        or initial_decrease >= 0.0
    ):
        return 0.0

    alpha = 1.0
    alpha_min = 0.0
    alpha_max = 1.0

    for _ in range(int(it_damp_max)):
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
            decrease = _dot(F_alpha - f, dU, q_mask=q_mask)

        if decrease < 0.0:
            if alpha == 1.0:
                break
            alpha_min = alpha
        else:
            alpha_max = alpha

        alpha = 0.5 * (alpha_min + alpha_max)

    return float(alpha)


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
) -> float:
    """Line-search damping for nested-Newton (`ALG5`) continuation updates.

    `criterion` is the current residual in the constrained norm and must be
    computed on free degrees of freedom.
    """

    if np.isnan(d_l) or np.isinf(d_l):
        return 0.0
    if not np.isfinite(criterion):
        return 0.0
    if it_damp_max <= 0:
        return 0.0

    U_it = np.asarray(U_it, dtype=np.float64)
    d_U = np.asarray(d_U, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    q_mask = np.asarray(q_mask, dtype=bool)

    alpha = 1.0
    last_evaluated_alpha: float | None = None

    for _ in range(int(it_damp_max)):
        U_alpha = U_it + alpha * d_U
        lambda_alpha = lambda_it + alpha * d_l
        if lambda_alpha <= 0.0:
            alpha *= 0.5
            if alpha <= 0.0:
                return 0.0
            continue
        build_F_all_free_local = getattr(constitutive_matrix_builder, "build_F_all_free_local", None)
        build_F_all_free = getattr(constitutive_matrix_builder, "build_F_all_free", None)
        try:
            if f_local_free is not None and callable(build_F_all_free_local):
                F_alpha_local_free = np.asarray(build_F_all_free_local(lambda_alpha, U_alpha), dtype=np.float64).reshape(-1)
                crit_alpha = _dist_norm_local(
                    F_alpha_local_free - np.asarray(f_local_free, dtype=np.float64).reshape(-1),
                    comm,
                )
            elif f_free is not None and callable(build_F_all_free):
                F_alpha_free = np.asarray(build_F_all_free(lambda_alpha, U_alpha), dtype=np.float64).reshape(-1)
                crit_alpha = float(np.linalg.norm(F_alpha_free - np.asarray(f_free, dtype=np.float64).reshape(-1)))
            else:
                F_alpha = constitutive_matrix_builder.build_F_all(lambda_alpha, U_alpha)
                crit_alpha = _norm(F_alpha - f, q_mask=q_mask)
        except Exception as exc:  # pragma: no cover - defensive for constitutive backends
            if not _is_invalid_lambda_trial(exc):
                raise
            alpha *= 0.5
            if alpha <= 0.0:
                return 0.0
            continue
        last_evaluated_alpha = float(alpha)

        if crit_alpha < criterion:
            break

        alpha *= 0.5
        if alpha <= 0.0:
            return 0.0

    if last_evaluated_alpha is None:
        return 0.0
    return float(alpha if alpha == last_evaluated_alpha else last_evaluated_alpha)
