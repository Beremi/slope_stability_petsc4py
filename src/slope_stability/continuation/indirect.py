"""Indirect continuation routines for strength reduction."""

from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np

from ..nonlinear.newton import newton, newton_ind_ssr
from ..utils import q_to_free_indices


def _free_indices(Q: np.ndarray) -> np.ndarray:
    return q_to_free_indices(np.asarray(Q, dtype=bool))


def _free(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(-1, order="F")[_free_indices(Q)]


def _free_dot(a: np.ndarray, b: np.ndarray, Q: np.ndarray) -> float:
    return float(np.dot(_free(a, Q), _free(b, Q)))


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
):
    """Compute first two converged points for indirect continuation."""

    Q = np.asarray(Q, dtype=bool)
    n_nodes = Q.shape[1]
    dim = Q.shape[0]
    U_ini = np.zeros((dim, n_nodes), dtype=np.float64)

    constitutive_matrix_builder.reduction(lambda_init)

    all_newton_its: list[int] = []
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
    )
    all_newton_its.append(it_newt)
    if flag_N == 1:
        raise RuntimeError("Initial choice of lambda seems to be too large.")

    U1 = U_it
    omega1 = _free_dot(f, U1, Q)

    d_lambda = float(d_lambda_init)
    lambda1 = float(lambda_init)
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
):
    """Indirect continuation in ``omega`` with nested-Newton solves."""

    Q = np.asarray(Q, dtype=bool)

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
        "attempt_linear_iterations": [],
        "attempt_linear_solve_time": [],
        "attempt_linear_preconditioner_time": [],
        "attempt_linear_orthogonalization_time": [],
        "attempt_omega_target": [],
        "attempt_lambda_before": [],
        "attempt_lambda_after": [],
        "step_index": [],
        "step_attempt_count": [],
        "step_wall_time": [],
        "step_newton_iterations": [],
        "step_newton_iterations_total": [],
        "step_newton_relres_end": [],
        "step_linear_iterations": [],
        "step_linear_solve_time": [],
        "step_linear_preconditioner_time": [],
        "step_linear_orthogonalization_time": [],
        "step_lambda": [],
        "step_omega": [],
        "step_U": [],
        "total_wall_time": 0.0,
    }

    def _emit(event: str, **payload) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": event,
                **payload,
            }
        )

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

    d_omega = omega - omega_old
    if omega_max_stop < omega + d_omega:
        raise ValueError("Too small value of omega_max_stop. Increase it and rerun.")

    step = 2
    n_omega = 0
    n_omega_max = 5

    step_wall_accum = 0.0
    step_lin_it_accum = 0
    step_lin_solve_accum = 0.0
    step_lin_prec_accum = 0.0
    step_lin_orth_accum = 0.0
    step_newton_it_accum = 0
    step_attempt_count = 0

    while True:
        omega_it = min(omega + d_omega, omega_max_stop)
        d_omega = omega_it - omega

        denom = omega - omega_old
        if denom == 0.0:
            U_ini = U
        else:
            U_ini = d_omega * (U - U_old) / denom + U

        t_attempt = perf_counter()
        snap_before = _collector_snapshot(linear_system_solver)
        U_it, lambda_candidate, flag, it_newt, history = newton_ind_ssr(
            U_ini,
            omega_it,
            lambda_value,
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
        snap_after = _collector_snapshot(linear_system_solver)
        attempt_delta = _collector_delta(snap_before, snap_after)
        attempt_wall = perf_counter() - t_attempt

        attempt_relres = np.nan
        if history["residual"].size:
            attempt_relres = float(history["residual"][-1])

        stats["attempt_step"].append(step + 1)
        stats["attempt_success"].append(flag == 0)
        stats["attempt_wall_time"].append(attempt_wall)
        stats["attempt_newton_iterations"].append(it_newt)
        stats["attempt_newton_flag"].append(flag)
        stats["attempt_newton_relres_end"].append(attempt_relres)
        stats["attempt_linear_iterations"].append(attempt_delta["iterations"])
        stats["attempt_linear_solve_time"].append(attempt_delta["solve_time"])
        stats["attempt_linear_preconditioner_time"].append(attempt_delta["preconditioner_time"])
        stats["attempt_linear_orthogonalization_time"].append(attempt_delta["orthogonalization_time"])
        stats["attempt_omega_target"].append(omega_it)
        stats["attempt_lambda_before"].append(lambda_value)
        stats["attempt_lambda_after"].append(lambda_candidate if flag == 0 else np.nan)

        _emit(
            "attempt_complete",
            target_step=int(step + 1),
            accepted_steps=int(step),
            attempt_in_step=int(step_attempt_count + 1),
            success=bool(flag == 0),
            omega_target=float(omega_it),
            lambda_before=float(lambda_value),
            lambda_after=float(lambda_candidate) if flag == 0 else None,
            newton_iterations=int(it_newt),
            newton_flag=int(flag),
            newton_relres_end=None if np.isnan(attempt_relres) else float(attempt_relres),
            linear_iterations=int(attempt_delta["iterations"]),
            linear_solve_time=float(attempt_delta["solve_time"]),
            linear_preconditioner_time=float(attempt_delta["preconditioner_time"]),
            linear_orthogonalization_time=float(attempt_delta["orthogonalization_time"]),
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
            linear_system_solver.expand_deflation_basis(_free(U, Q))
            omega_old = omega
            omega = omega_it
            d_lambda = lambda_candidate - lambda_value
            lambda_value = lambda_candidate
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
            stats["step_linear_iterations"].append(step_lin_it_accum)
            stats["step_linear_solve_time"].append(step_lin_solve_accum)
            stats["step_linear_preconditioner_time"].append(step_lin_prec_accum)
            stats["step_linear_orthogonalization_time"].append(step_lin_orth_accum)
            stats["step_lambda"].append(lambda_value)
            stats["step_omega"].append(omega)
            if store_step_u:
                stats["step_U"].append(U.copy())

            _emit(
                "step_accepted",
                accepted_step=int(step),
                step_attempt_count=int(step_attempt_count),
                lambda_value=float(lambda_value),
                omega_value=float(omega),
                u_max=float(Umax_hist[step - 1]),
                step_wall_time=float(step_wall_accum),
                step_newton_iterations=int(it_newt),
                step_newton_iterations_total=int(step_newton_it_accum),
                step_newton_relres_end=None if np.isnan(attempt_relres) else float(attempt_relres),
                step_linear_iterations=int(step_lin_it_accum),
                step_linear_solve_time=float(step_lin_solve_accum),
                step_linear_preconditioner_time=float(step_lin_prec_accum),
                step_linear_orthogonalization_time=float(step_lin_orth_accum),
                total_wall_time=float(perf_counter() - t_total),
            )

            step_wall_accum = 0.0
            step_lin_it_accum = 0
            step_lin_solve_accum = 0.0
            step_lin_prec_accum = 0.0
            step_lin_orth_accum = 0.0
            step_newton_it_accum = 0
            step_attempt_count = 0

            if (d_lambda / d_omega) * (omega_hist[step - 1] - omega_hist[0]) < d_lambda_diff_scaled_min:
                break
            if omega >= omega_max_stop:
                break
            if (lambda_hist[step - 1] - lambda_hist[step - 2]) < 0.9 * (lambda_hist[step - 2] - lambda_hist[step - 3]):
                d_omega *= 2.0

        if n_omega >= n_omega_max:
            break
        if step >= step_max:
            break

    lambda_hist = lambda_hist[:step]
    omega_hist = omega_hist[:step]
    Umax_hist = Umax_hist[:step]
    stats["total_wall_time"] = perf_counter() - t_total

    _emit(
        "finished",
        accepted_steps=int(step),
        lambda_last=float(lambda_hist[-1]),
        omega_last=float(omega_hist[-1]),
        total_wall_time=float(stats["total_wall_time"]),
    )

    return U, lambda_hist, omega_hist, Umax_hist, stats
