"""Direct continuation workflows for SSR."""

from __future__ import annotations

import numpy as np

from .omega import omega_SSR_direct_continuation
from ..nonlinear.newton import newton
from ..utils import q_to_free_indices


def init_phase_SSR_direct_continuation(
    lambda_init: float,
    d_lambda_init: float,
    d_lambda_min: float,
    it_newt_max: int,
    it_damp_max: int,
    tol: float,
    eps: float,
    r_min: float,
    K_elast,
    Q: np.ndarray,
    f: np.ndarray,
    constitutive_matrix_builder,
    linear_system_solver,
):
    """Initial two-step phase for the direct continuation algorithm."""

    Q = np.asarray(Q, dtype=bool)
    dim = Q.shape[0]
    n_nodes = Q.shape[1]
    U_ini = np.zeros((dim, n_nodes), dtype=np.float64)

    U1, omega1, flag = omega_SSR_direct_continuation(
        lambda_init,
        U_ini,
        eps,
        1000.0,
        it_newt_max,
        it_damp_max,
        tol,
        r_min,
        K_elast,
        Q,
        f,
        constitutive_matrix_builder,
        linear_system_solver.copy(),
    )
    if flag == 1:
        raise RuntimeError("Initial choice of lambda seems to be too large.")

    d_lambda = float(d_lambda_init)
    lambda1 = float(lambda_init)

    while True:
        lambda_it = lambda1 + d_lambda

        linear_system_solver.expand_deflation_basis((U1.reshape(-1, order="F")[q_to_free_indices(Q)]))
        U2, omega2, flag = omega_SSR_direct_continuation(
            lambda_it,
            U1,
            eps,
            d_lambda,
            it_newt_max,
            it_damp_max,
            tol,
            r_min,
            K_elast,
            Q,
            f,
            constitutive_matrix_builder,
            linear_system_solver.copy(),
        )
        if flag == 1:
            d_lambda /= 2.0
        else:
            if (omega2 - omega1) / max(1.0, omega1) < 1e-5:
                U1 = U2
                lambda1 = lambda_it
                omega1 = omega2
            else:
                lambda2 = lambda_it
                break

        if d_lambda < d_lambda_min:
            raise RuntimeError("It seems that the FoS is equal to lambda_init.")
        if lambda1 > 10.0:
            raise RuntimeError("It seems that the FoS is greater than 10.")

    return U1, U2, omega1, omega2, lambda1, lambda2


def SSR_direct_continuation(
    lambda_init: float,
    d_lambda_init: float,
    d_lambda_min: float,
    d_lambda_diff_scaled_min: float,
    step_max: int,
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
    """Direct continuation loop over the reduction factor ``lambda``."""

    f = np.asarray(f, dtype=np.float64)

    lambda_hist = np.empty(1000, dtype=np.float64)
    omega_hist = np.empty(1000, dtype=np.float64)
    Umax_hist = np.empty(1000, dtype=np.float64)
    work_hist = np.empty(1000, dtype=np.float64)

    eps = tol * 100.0
    U_old, U, omega_old, omega, lambda_old, lambda_it = init_phase_SSR_direct_continuation(
        lambda_init,
        d_lambda_init,
        d_lambda_min,
        it_newt_max,
        it_damp_max,
        tol,
        eps,
        r_min,
        K_elast,
        Q,
        f,
        constitutive_matrix_builder,
        linear_system_solver,
    )

    linear_system_solver.expand_deflation_basis(U_old.reshape(-1, order="F")[q_to_free_indices(Q)])

    lambda_hist[0] = lambda_old
    omega_hist[0] = omega_old
    Umax_hist[0] = np.max(np.linalg.norm(U_old, axis=0))
    work_hist[0] = float(np.dot(U_old.ravel(order="F"), f.ravel(order="F")))

    lambda_hist[1] = lambda_it
    omega_hist[1] = omega
    Umax_hist[1] = np.max(np.linalg.norm(U, axis=0))
    work_hist[1] = float(np.dot(U.ravel(order="F"), f.ravel(order="F")))

    d_omega = omega - omega_old
    d_lambda = lambda_it - lambda_init
    step = 2

    while True:
        lambda_candidate = lambda_it + d_lambda
        U_it, omega_it, flag = omega_SSR_direct_continuation(
            lambda_candidate,
            U,
            eps,
            d_lambda,
            it_newt_max,
            it_damp_max,
            tol,
            r_min,
            K_elast,
            Q,
            f,
            constitutive_matrix_builder,
            linear_system_solver.copy(),
        )

        d_omega_test = omega_it - omega_hist[step - 1]

        if (flag == 1) or (d_omega_test < 0.0):
            d_lambda *= 0.5
        else:
            U = U_it
            omega = omega_it
            lambda_it = lambda_candidate
            step += 1
            lambda_hist[step - 1] = lambda_it
            omega_hist[step - 1] = omega
            Umax_hist[step - 1] = np.max(np.linalg.norm(U, axis=0))
            work_hist[step - 1] = float(np.dot(U.ravel(order="F"), f.ravel(order="F")))
            linear_system_solver.expand_deflation_basis(U.reshape(-1, order="F")[q_to_free_indices(Q)])

            if (d_lambda / d_omega_test) * (omega_hist[step - 1] - omega_hist[0]) < d_lambda_diff_scaled_min:
                break

            if d_omega_test > 1.5 * d_omega:
                d_lambda *= 0.5
            d_omega = d_omega_test

        if d_lambda < d_lambda_min:
            break
        if step >= step_max:
            break

    lambda_hist = lambda_hist[:step]
    omega_hist = omega_hist[:step]
    Umax_hist = Umax_hist[:step]
    work_hist = work_hist[:step]

    return U, lambda_hist, omega_hist, Umax_hist, work_hist
