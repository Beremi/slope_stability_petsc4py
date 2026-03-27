"""Direct continuation helpers for the SSR direct path."""

from __future__ import annotations

import numpy as np

from ..utils import q_to_free_indices
from ..nonlinear.newton import newton


def _flat(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(-1, order="F")


def _free(v: np.ndarray, q_mask: np.ndarray) -> np.ndarray:
    return _flat(v)[q_to_free_indices(q_mask)]


def _free_dot(a: np.ndarray, b: np.ndarray, q_mask: np.ndarray) -> float:
    return float(np.dot(_free(a, q_mask), _free(b, q_mask)))


def omega_SSR_direct_continuation(
    lambda_ini: float,
    U_ini: np.ndarray,
    eps: float,
    d_lambda: float,
    it_newt_max: int,
    it_damp_max: int,
    tol: float,
    r_min: float,
    K_elast,
    Q: np.ndarray,
    f: np.ndarray,
    constitutive_matrix_builder,
    linear_system_solver,
):
    """Compute displacement for fixed ``lambda`` and return derivative-like omega."""

    f = np.asarray(f, dtype=np.float64)
    Q = np.asarray(Q, dtype=bool)

    omega = 0.0
    constitutive_matrix_builder.reduction(lambda_ini)

    U, flag, _ = newton(
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
    )
    if flag == 1:
        return U, omega, flag

    Psi_integrated = constitutive_matrix_builder.potential_energy(U)
    J = Psi_integrated - _free_dot(f, U, Q)

    beta = min(1.0, eps / d_lambda)
    U_beta = beta * np.asarray(U_ini, dtype=np.float64) + (1.0 - beta) * np.asarray(U, dtype=np.float64)

    constitutive_matrix_builder.reduction(lambda_ini - eps)
    U_eps, flag, _ = newton(
        U_beta,
        tol,
        it_newt_max,
        it_damp_max,
        r_min,
        K_elast,
        Q,
        f,
        constitutive_matrix_builder,
        linear_system_solver,
    )
    if flag == 1:
        return U, omega, flag

    Psi_integrated_eps = constitutive_matrix_builder.potential_energy(U_eps)
    J_eps = Psi_integrated_eps - _free_dot(f, U_eps, Q)
    omega = (J_eps - J) / eps

    return U, float(omega), 0
