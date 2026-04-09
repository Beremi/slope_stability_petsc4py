from __future__ import annotations

import numpy as np

from slope_stability.linear import SolverFactory
from slope_stability.nonlinear.newton import (
    _destroy_petsc_mat,
    _prefers_full_system_operator,
    _setup_linear_system,
    _solve_linear_system,
)
from slope_stability.utils import extract_submatrix_free, full_field_from_free_values, q_to_free_indices


def _collector_snapshot(solver) -> dict[str, float | int]:
    collector = solver.iteration_collector
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


def solve_elastic_initial_guess(
    *,
    solver_type: str,
    linear_tolerance: float,
    linear_max_iter: int,
    linear_system_solver,
    preconditioner_options: dict[str, object],
    effective_pc_backend: str | None,
    q_mask: np.ndarray,
    coord: np.ndarray,
    K_elast,
    f_V: np.ndarray,
) -> dict[str, object]:
    free_idx = q_to_free_indices(np.asarray(q_mask, dtype=bool))
    f_full = np.asarray(f_V, dtype=np.float64).reshape(-1, order="F")
    f_free = np.asarray(f_full[free_idx], dtype=np.float64)

    init_linear_solver = linear_system_solver
    if str(effective_pc_backend).strip().lower() in {"pmg", "pmg_shell"}:
        init_preconditioner_options = dict(preconditioner_options)
        init_preconditioner_options["pc_backend"] = "hypre"
        init_preconditioner_options.pop("pmg_hierarchy", None)
        for key in tuple(init_preconditioner_options.keys()):
            if key.startswith("mg_") or key.startswith("pc_mg_"):
                init_preconditioner_options.pop(key, None)
        init_linear_solver = SolverFactory.create(
            solver_type,
            tolerance=float(linear_tolerance),
            max_iterations=int(linear_max_iter),
            deflation_basis_tolerance=1e-3,
            verbose=False,
            q_mask=np.asarray(q_mask, dtype=bool),
            coord=np.asarray(coord, dtype=np.float64),
            preconditioner_options=init_preconditioner_options,
        )

    snap_init_0 = _collector_snapshot(init_linear_solver)
    U_elast_free = None
    K_free = None
    try:
        if _prefers_full_system_operator(init_linear_solver, K_elast):
            _setup_linear_system(init_linear_solver, K_elast, A_full=K_elast, free_idx=free_idx)
            U_elast_free = _solve_linear_system(
                init_linear_solver,
                K_elast,
                f_free,
                b_full=f_full,
                free_idx=free_idx,
            )
        else:
            K_free = extract_submatrix_free(K_elast, free_idx)
            _setup_linear_system(init_linear_solver, K_free, A_full=K_elast, free_idx=free_idx)
            U_elast_free = _solve_linear_system(
                init_linear_solver,
                K_free,
                f_free,
                b_full=f_full,
                free_idx=free_idx,
            )
    finally:
        release = getattr(init_linear_solver, "release_iteration_resources", None)
        if callable(release):
            release()
        _destroy_petsc_mat(K_free)

    snap_init_1 = _collector_snapshot(init_linear_solver)
    init_delta = _collector_delta(snap_init_0, snap_init_1)
    U_elast_free = np.asarray(U_elast_free, dtype=np.float64).reshape(-1)
    U_elast = full_field_from_free_values(U_elast_free, free_idx, f_V.shape)
    omega_el = float(np.dot(f_free, U_elast_free))
    return {
        "U_elast": U_elast,
        "U_elast_free": U_elast_free,
        "free_idx": free_idx,
        "omega_el": omega_el,
        "init_linear": {
            "init_linear_iterations": int(init_delta["iterations"]),
            "init_linear_solve_time": float(init_delta["solve_time"]),
            "init_linear_preconditioner_time": float(init_delta["preconditioner_time"]),
            "init_linear_orthogonalization_time": float(init_delta["orthogonalization_time"]),
        },
    }
