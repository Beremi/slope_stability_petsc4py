"""Limit-load continuation for the load multiplier ``t``."""

from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np

from ..nonlinear.newton import newton_ind_ll
from ..utils import q_to_free_indices


def _free_indices(Q: np.ndarray) -> np.ndarray:
    return q_to_free_indices(np.asarray(Q, dtype=bool))


def _free(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(-1, order="F")[_free_indices(Q)]


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


def LL_indirect_continuation(
    d_omega_ini: float,
    d_t_min: float,
    step_max: int,
    omega_max: float,
    it_newt_max: int,
    it_damp_max: int,
    tol: float,
    r_min: float,
    K_elast,
    U_elast: np.ndarray,
    Q: np.ndarray,
    f: np.ndarray,
    constitutive_matrix_builder,
    linear_system_solver,
    progress_callback: Callable[[dict], None] | None = None,
):
    """Limit-load continuation (indirect) for ``F(U) = t f`` and ``f'U = omega``."""

    U = np.asarray(U_elast, dtype=np.float64)
    Q = np.asarray(Q, dtype=bool)

    history_size = max(2, int(step_max) + 1)
    omega_hist = np.zeros(history_size, dtype=np.float64)
    t_hist = np.zeros(history_size, dtype=np.float64)
    U_max_hist = np.zeros(history_size, dtype=np.float64)

    omega = 0.0
    omega_old = 0.0
    d_omega = float(d_omega_ini)
    t = 0.0
    U_old = np.array(U, copy=True)
    U_ini = np.array(U, copy=True)

    stats = {
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
        "step_U": [U.copy()],
        "total_wall_time": 0.0,
    }

    def _emit(event: str, **payload) -> None:
        if progress_callback is None:
            return
        progress_callback({"event": event, "continuation_kind": "ll_indirect", **payload})

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
                    "continuation_kind": "ll_indirect",
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

    _emit(
        "init_complete",
        phase="init",
        accepted_steps=1,
        lambda_hist=[float(t)],
        omega_hist=[float(omega)],
        init_newton_iterations=[],
        init_linear_iterations=0,
        init_linear_solve_time=0.0,
        init_linear_preconditioner_time=0.0,
        init_linear_orthogonalization_time=0.0,
        total_wall_time=0.0,
    )

    n_omega = 0
    n_omega_max = 5
    step = 1
    t_total = perf_counter()
    stop_reason = "completed"

    attempt_counter_for_step = 0
    step_wall_accum = 0.0
    step_newton_accum = 0
    step_linear_accum = 0
    step_solve_accum = 0.0
    step_pc_accum = 0.0
    step_orth_accum = 0.0

    while True:
        omega_it = omega + d_omega
        if omega_it > omega_max:
            omega_it = omega_max
            d_omega = omega_it - omega

        if step > 1:
            denom = omega - omega_old
            if denom == 0.0:
                U_ini = U
            else:
                U_ini = d_omega * (U - U_old) / denom + U

        snap_before = _collector_snapshot(linear_system_solver)
        t_attempt = perf_counter()
        attempt_in_step = attempt_counter_for_step + 1
        basis_before_attempt = _basis_snapshot(linear_system_solver)
        U_it, t_it, flag, it_newt, history = newton_ind_ll(
            U_ini,
            t,
            omega_it,
            it_newt_max,
            it_damp_max,
            tol,
            r_min,
            K_elast,
            Q,
            f,
            constitutive_matrix_builder,
            linear_system_solver,
            progress_callback=_newton_progress_callback(
                target_step=step + 1,
                accepted_steps=step,
                attempt_in_step=attempt_in_step,
                omega_target=float(omega_it),
                lambda_before=float(t),
            ),
        )
        _basis_restore(linear_system_solver, basis_before_attempt)
        _notify_attempt(linear_system_solver, success=(flag == 0))
        attempt_wall = perf_counter() - t_attempt
        snap_after = _collector_snapshot(linear_system_solver)
        attempt_delta = _collector_delta(snap_before, snap_after)
        relres_end = float(history["residual"][-1]) if history["residual"].size else np.nan

        attempt_counter_for_step += 1
        step_wall_accum += attempt_wall
        step_newton_accum += int(it_newt)
        step_linear_accum += int(attempt_delta["iterations"])
        step_solve_accum += float(attempt_delta["solve_time"])
        step_pc_accum += float(attempt_delta["preconditioner_time"])
        step_orth_accum += float(attempt_delta["orthogonalization_time"])

        stats["attempt_step"].append(int(step + 1))
        stats["attempt_success"].append(int(flag == 0))
        stats["attempt_wall_time"].append(float(attempt_wall))
        stats["attempt_newton_iterations"].append(int(it_newt))
        stats["attempt_newton_flag"].append(int(flag))
        stats["attempt_newton_relres_end"].append(float(relres_end))
        stats["attempt_linear_iterations"].append(int(attempt_delta["iterations"]))
        stats["attempt_linear_solve_time"].append(float(attempt_delta["solve_time"]))
        stats["attempt_linear_preconditioner_time"].append(float(attempt_delta["preconditioner_time"]))
        stats["attempt_linear_orthogonalization_time"].append(float(attempt_delta["orthogonalization_time"]))
        stats["attempt_omega_target"].append(float(omega_it))
        stats["attempt_lambda_before"].append(float(t))
        stats["attempt_lambda_after"].append(float(t_it))

        _emit(
            "attempt_complete",
            phase="continuation",
            target_step=int(step + 1),
            accepted_steps=int(step),
            attempt_in_step=int(attempt_in_step),
            success=bool(flag == 0),
            omega_target=float(omega_it),
            lambda_before=float(t),
            lambda_after=float(t_it),
            newton_iterations=int(it_newt),
            newton_relres_end=float(relres_end),
            linear_iterations=int(attempt_delta["iterations"]),
            linear_solve_time=float(attempt_delta["solve_time"]),
            linear_preconditioner_time=float(attempt_delta["preconditioner_time"]),
            linear_orthogonalization_time=float(attempt_delta["orthogonalization_time"]),
            attempt_wall_time=float(attempt_wall),
            total_wall_time=float(perf_counter() - t_total),
        )

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

            d_t = t_it - t
            t = t_it
            n_omega = 0

            omega_hist[step - 1] = omega
            t_hist[step - 1] = t
            U_max_hist[step - 1] = np.max(np.linalg.norm(U, axis=0))

            stats["step_index"].append(int(step))
            stats["step_attempt_count"].append(int(attempt_counter_for_step))
            stats["step_wall_time"].append(float(step_wall_accum))
            stats["step_newton_iterations"].append(int(it_newt))
            stats["step_newton_iterations_total"].append(int(step_newton_accum))
            stats["step_newton_relres_end"].append(float(relres_end))
            stats["step_linear_iterations"].append(int(step_linear_accum))
            stats["step_linear_solve_time"].append(float(step_solve_accum))
            stats["step_linear_preconditioner_time"].append(float(step_pc_accum))
            stats["step_linear_orthogonalization_time"].append(float(step_orth_accum))
            stats["step_lambda"].append(float(t))
            stats["step_omega"].append(float(omega))
            stats["step_U"].append(U.copy())

            _emit(
                "step_accepted",
                phase="continuation",
                accepted_steps=int(step),
                lambda_value=float(t),
                omega_value=float(omega),
                d_lambda=float(d_t),
                d_omega=float(d_omega),
                u_max=float(U_max_hist[step - 1]),
                attempt_count=int(attempt_counter_for_step),
                newton_iterations=int(it_newt),
                newton_iterations_total=int(step_newton_accum),
                newton_relres_end=float(relres_end),
                linear_iterations=int(step_linear_accum),
                linear_solve_time=float(step_solve_accum),
                linear_preconditioner_time=float(step_pc_accum),
                linear_orthogonalization_time=float(step_orth_accum),
                step_wall_time=float(step_wall_accum),
                total_wall_time=float(perf_counter() - t_total),
            )

            attempt_counter_for_step = 0
            step_wall_accum = 0.0
            step_newton_accum = 0
            step_linear_accum = 0
            step_solve_accum = 0.0
            step_pc_accum = 0.0
            step_orth_accum = 0.0

            if d_t < d_t_min:
                stop_reason = "d_t_min"
                break

            if (it_newt < 20) and (step > 2) and (
                t_hist[step - 1] - t_hist[step - 2] < 0.9 * (t_hist[step - 2] - t_hist[step - 3])
            ):
                d_omega *= 2.0

        if n_omega >= n_omega_max:
            stop_reason = "omega_reduction_limit"
            break
        if omega >= omega_max:
            stop_reason = "omega_max"
            break
        if step >= step_max:
            stop_reason = "step_max"
            break

    t_hist = t_hist[:step]
    omega_hist = omega_hist[:step]
    U_max_hist = U_max_hist[:step]
    stats["total_wall_time"] = float(perf_counter() - t_total)

    _emit(
        "finished",
        phase="continuation",
        accepted_steps=int(step),
        lambda_last=float(t_hist[-1]),
        omega_last=float(omega_hist[-1]),
        stop_reason=str(stop_reason),
        total_wall_time=float(stats["total_wall_time"]),
    )

    return U, t_hist, omega_hist, U_max_hist, stats
