#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from petsc4py import PETSc

from slope_stability.cli.assembly_policy import use_lightweight_mpi_elastic_path, use_owned_tangent_path
from slope_stability.cli.run_3D_hetero_SSR_capture import _parse_petsc_opt_entries
from slope_stability.constitutive import ConstitutiveOperator
from slope_stability.continuation.indirect import (
    _free,
    _free_dot,
    _predictor_free_residual,
    _rescale_to_target_omega,
    _secant_predictor,
    init_phase_SSR_indirect_continuation,
)
from slope_stability.fem import (
    assemble_owned_elastic_rows_for_comm,
    assemble_strain_operator,
    prepare_owned_tangent_pattern,
    quadrature_volume_3d,
    vector_volume,
)
from slope_stability.linear import SolverFactory
from slope_stability.linear.pmg import build_3d_pmg_hierarchy
from slope_stability.mesh import MaterialSpec, heterogenous_materials, load_mesh_from_file
from slope_stability.nonlinear.newton import (
    _build_regularized_from_cached_if_available,
    _build_regularized_if_available,
    _cleanup_pre_solve_iteration_mats,
    _collector_delta,
    _collector_snapshot,
    _combine_matrices,
    _destroy_petsc_mat,
    _ensure_tangent_matrix_for_regularization,
    _explicit_preconditioning_matrix,
    _free_norm,
    _is_builder_cached_matrix,
    _local_comm_from_operator,
    _local_owned_free_rows_from_field,
    _local_owned_rows_from_field,
    _needs_preconditioning_matrix_refresh,
    _prefers_full_system_operator,
    _release_iteration_resources,
    _requires_explicit_preconditioning_matrix,
    _setup_linear_system,
    _solve_linear_system,
    _solve_linear_system_local,
    _supports_free_builder,
    _supports_local_builder,
    _to_free_vector,
)
from slope_stability.nonlinear.damping import damping_alg5
from slope_stability.utils import extract_submatrix_free, local_csr_to_petsc_aij_matrix


ROOT = Path(__file__).resolve().parents[3]
BASELINE_RUN_INFO = ROOT / "artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_rank8_step12/data/run_info.json"


def _write_jsonl(path: Path, payload: dict[str, object]) -> None:
    if PETSc.COMM_WORLD.getRank() != 0:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def _material_specs(rows: list[list[float]]) -> list[MaterialSpec]:
    return [
        MaterialSpec(
            c0=float(row[0]),
            phi=float(row[1]),
            psi=float(row[2]),
            young=float(row[3]),
            poisson=float(row[4]),
            gamma_sat=float(row[5]),
            gamma_unsat=float(row[6]),
        )
        for row in rows
    ]


def _restore_rolling_solution_basis(
    solver,
    solution_hist_free: list[np.ndarray],
    *,
    max_vectors: int,
    compression_mode: str = "none",
    svd_basis_size: int = 10,
) -> int:
    if not getattr(solver, "supports_dynamic_deflation_basis", lambda: True)():
        return 0
    restore = getattr(solver, "restore_deflation_basis", None)
    if callable(restore):
        restore(None)
    recent = solution_hist_free[-max(int(max_vectors), 1) :]
    basis_vectors: list[np.ndarray]
    if str(compression_mode) == "svd" and recent:
        mat = np.column_stack([np.asarray(vec, dtype=np.float64).reshape(-1) for vec in recent])
        try:
            u, s, _vh = np.linalg.svd(mat, full_matrices=False)
        except np.linalg.LinAlgError:
            u = np.asarray(mat, dtype=np.float64)
            s = np.ones(u.shape[1], dtype=np.float64)
        rank = int(min(max(int(svd_basis_size), 1), u.shape[1]))
        if s.size:
            finite = np.isfinite(s)
            if np.any(finite):
                keep = np.where(finite & (s > max(float(np.max(s[finite])) * 1.0e-12, 1.0e-14)))[0]
                if keep.size:
                    rank = int(min(rank, keep.size))
        basis_vectors = [np.asarray(u[:, j], dtype=np.float64).reshape(-1) for j in range(rank)]
    else:
        basis_vectors = [np.asarray(vec, dtype=np.float64).reshape(-1) for vec in recent]
    for vec in basis_vectors:
        solver.expand_deflation_basis(np.asarray(vec, dtype=np.float64).reshape(-1))
    snapshot = getattr(solver, "get_deflation_basis_snapshot", lambda: None)()
    if snapshot is None:
        return 0
    arr = np.asarray(snapshot, dtype=np.float64)
    return int(arr.shape[1]) if arr.ndim == 2 else 0


def _disable_solver_deflation_features(solver) -> None:
    def _false() -> bool:
        return False

    def _noop(*_args, **_kwargs):
        return None

    setattr(solver, "supports_dynamic_deflation_basis", _false)
    setattr(solver, "supports_a_orthogonalization", _false)
    setattr(solver, "expand_deflation_basis", _noop)
    setattr(solver, "restore_deflation_basis", _noop)
    setattr(solver, "get_deflation_basis_snapshot", lambda: None)
    setattr(solver, "A_orthogonalize", _noop)
    if hasattr(solver, "deflation_basis"):
        solver.deflation_basis = np.empty((0, 0), dtype=np.float64)


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
    initial_length = float(np.sqrt(2.0) * max(float(cap_factor), 0.0))
    domega_raw = float(max(float(domega_candidate), 0.0))
    if not np.isfinite(domega_raw) or domega_raw <= 0.0 or not np.isfinite(initial_length) or initial_length <= 0.0:
        return domega_raw, 0.0, initial_length
    omega_span = float(omega_anchor_curr) - float(omega_anchor_prev)
    if abs(omega_span) <= 1.0e-12:
        return domega_raw, float(domega_raw / domega_initial_abs), initial_length
    slope = (float(lambda_anchor_curr) - float(lambda_anchor_prev)) / omega_span
    normalized_length_raw = float(
        np.sqrt(
            (domega_raw / domega_initial_abs) ** 2
            + (abs(float(slope)) * domega_raw / dlambda_initial_abs) ** 2
        )
    )
    if not np.isfinite(normalized_length_raw) or normalized_length_raw <= initial_length:
        return domega_raw, normalized_length_raw, initial_length
    cap_ratio = float(initial_length / max(normalized_length_raw, 1.0e-12))
    domega_capped = float(max(domega_raw * cap_ratio, 1.0e-12))
    normalized_length_capped = float(
        np.sqrt(
            (domega_capped / domega_initial_abs) ** 2
            + (abs(float(slope)) * domega_capped / dlambda_initial_abs) ** 2
        )
    )
    return domega_capped, normalized_length_capped, initial_length


def _run_single_newton_micro_step(
    *,
    U_ini: np.ndarray,
    lambda_ini: float,
    omega_target: float,
    params: dict,
    K_elast,
    q_mask: np.ndarray,
    f_V: np.ndarray,
    const_builder,
    solver,
    same_matrix_halfstep_threshold: float = np.inf,
) -> dict[str, object]:
    U_it = np.asarray(U_ini, dtype=np.float64).copy()
    t_it = float(lambda_ini)
    shape = U_it.shape
    free_idx = np.flatnonzero(np.asarray(q_mask, dtype=bool).reshape(-1, order="F"))
    norm_f = _free_norm(f_V, q_mask)
    if norm_f == 0.0:
        norm_f = 1.0
    r = float(params["r_min"])
    it_damp_max = int(params["it_damp_max"])
    step_t0 = perf_counter()
    snap_before = _collector_snapshot(solver)

    use_full_operator = _prefers_full_system_operator(solver, K_elast)
    use_free_build = _supports_free_builder(const_builder, "build_F_all_free")
    use_local_build = use_full_operator and _supports_local_builder(const_builder, "build_F_all_local")
    comm = _local_comm_from_operator(K_elast) if use_local_build else None
    f_free = _to_free_vector(f_V, q_mask)
    eps = 1.0e-5

    def _evaluate_state(U_eval: np.ndarray, lambda_eval: float, *, build_matrix: bool):
        K_tangent = None
        K_r = None
        F_all = None
        F_all_local = None
        F_all_free_local = None
        F_all_free = None
        if build_matrix:
            if use_local_build:
                const_builder.reduction(lambda_eval)
                const_builder.constitutive_problem_stress_tangent(U_eval)
                F_all_local = np.asarray(const_builder.build_F_local(), dtype=np.float64).reshape(-1)
                F_all_free_local = np.asarray(const_builder.build_F_free_local(), dtype=np.float64).reshape(-1)
                F_all_free = F_all_free_local
                K_r = const_builder.build_K_regularized(r)
            elif use_full_operator and _supports_free_builder(const_builder, "build_F_K_regularized_all_free"):
                F_all_free, K_r = const_builder.build_F_K_regularized_all_free(lambda_eval, U_eval, r)
                F_all_free = np.asarray(F_all_free, dtype=np.float64).reshape(-1)
            elif use_full_operator:
                regularized_pair = _build_regularized_if_available(const_builder, lam=lambda_eval, U=U_eval, r=r)
                if regularized_pair is not None:
                    F_all, K_r = regularized_pair
                    F_all_free = _to_free_vector(F_all, q_mask)
                else:
                    F_all, K_tangent = const_builder.build_F_K_tangent_all(lambda_eval, U_eval)
                    F_all_free = _to_free_vector(F_all, q_mask)
                    K_r = _combine_matrices(r, K_elast, 1.0 - r, K_tangent)
            elif _supports_free_builder(const_builder, "build_F_K_tangent_all_free"):
                F_all_free, K_tangent = const_builder.build_F_K_tangent_all_free(lambda_eval, U_eval)
                F_all_free = np.asarray(F_all_free, dtype=np.float64).reshape(-1)
                K_r = _combine_matrices(r, K_elast, 1.0 - r, K_tangent)
            else:
                F_all, K_tangent = const_builder.build_F_K_tangent_all(lambda_eval, U_eval)
                F_all_free = _to_free_vector(F_all, q_mask)
                K_r = _combine_matrices(r, K_elast, 1.0 - r, K_tangent)
        else:
            if use_local_build:
                F_all_local = np.asarray(const_builder.build_F_all_local(lambda_eval, U_eval), dtype=np.float64).reshape(-1)
                F_all_free_local = np.asarray(const_builder.build_F_all_free_local(lambda_eval, U_eval), dtype=np.float64).reshape(-1)
                F_all_free = F_all_free_local
            elif use_free_build:
                F_all_free = np.asarray(const_builder.build_F_all_free(lambda_eval, U_eval), dtype=np.float64).reshape(-1)
            else:
                F_all = const_builder.build_F_all(lambda_eval, U_eval)
                F_all_free = _to_free_vector(F_all, q_mask)
        return {
            "F_all": F_all,
            "F_all_local": F_all_local,
            "F_all_free_local": F_all_free_local,
            "F_all_free": np.asarray(F_all_free, dtype=np.float64).reshape(-1),
            "K_tangent": K_tangent,
            "K_r": K_r,
        }

    def _evaluate_lambda_derivative(U_eval: np.ndarray, lambda_eval: float, state: dict[str, object]):
        F_all = state["F_all"]
        F_all_local = state["F_all_local"]
        F_all_free_local = state["F_all_free_local"]
        F_all_free = state["F_all_free"]
        if use_local_build:
            F_eps_local = np.asarray(const_builder.build_F_all_local(lambda_eval + eps, U_eval), dtype=np.float64).reshape(-1)
            F_eps_free_local = np.asarray(const_builder.build_F_all_free_local(lambda_eval + eps, U_eval), dtype=np.float64).reshape(-1)
            return {
                "G": None,
                "G_local": (F_eps_local - F_all_local) / eps,
                "G_free": (F_eps_free_local - F_all_free_local) / eps,
            }
        if use_free_build:
            F_eps_free = np.asarray(const_builder.build_F_all_free(lambda_eval + eps, U_eval), dtype=np.float64).reshape(-1)
            return {
                "G": None,
                "G_local": None,
                "G_free": (F_eps_free - F_all_free) / eps,
            }
        F_eps = const_builder.build_F_all(lambda_eval + eps, U_eval)
        G = (F_eps - F_all) / eps
        return {
            "G": G,
            "G_local": None,
            "G_free": _to_free_vector(G, q_mask),
        }

    first_eval = _evaluate_state(U_it, t_it, build_matrix=True)
    F_all = first_eval["F_all"]
    F_all_local = first_eval["F_all_local"]
    F_all_free_local = first_eval["F_all_free_local"]
    F_all_free = first_eval["F_all_free"]
    K_tangent = first_eval["K_tangent"]
    K_r = first_eval["K_r"]
    if use_local_build:
        f_free_local = _local_owned_free_rows_from_field(f_V, const_builder.owned_tangent_pattern)
        criterion_before = float(np.linalg.norm(F_all_free_local - f_free_local))
    else:
        f_free_local = None
        criterion_before = float(np.linalg.norm(F_all_free - f_free))
    rel_before = float(criterion_before / norm_f)

    derivative_state = _evaluate_lambda_derivative(U_it, t_it, first_eval)
    G_local = derivative_state["G_local"]
    G_free = np.asarray(derivative_state["G_free"], dtype=np.float64).reshape(-1)

    preconditioning_matrix = None
    if use_full_operator and _requires_explicit_preconditioning_matrix(solver):
        if _needs_preconditioning_matrix_refresh(solver):
            preconditioning_matrix = _explicit_preconditioning_matrix(
                const_builder,
                solver,
                regularization_r=r,
                K_elast=K_elast,
            )

    K_free = None
    W = None
    V = None
    half_step_triggered = False
    half_step_linear_iterations = 0
    half_step_wall_time = 0.0
    newton_substeps = 1
    half_step_rel_after_first = np.nan
    linear_solve_infos: list[dict[str, object]] = []

    def _record_last_linear_solve(label: str, phase: str) -> None:
        getter = getattr(solver, "get_last_solve_info", None)
        info = getter() if callable(getter) else {}
        if not isinstance(info, dict):
            info = {}
        linear_solve_infos.append(
            {
                "label": str(label),
                "phase": str(phase),
                "iterations": int(info.get("iterations", -1)) if info.get("iterations", None) is not None else None,
                "converged": None if info.get("converged", None) is None else bool(info.get("converged")),
                "hit_max_iterations": None
                if info.get("hit_max_iterations", None) is None
                else bool(info.get("hit_max_iterations")),
                "converged_reason": info.get("converged_reason"),
                "true_residual_final": None
                if info.get("true_residual_final", None) is None
                else float(info.get("true_residual_final")),
                "reported_residual_final": None
                if info.get("reported_residual_final", None) is None
                else float(info.get("reported_residual_final")),
                "basis_cols": int(info.get("basis_cols", 0)) if info.get("basis_cols", None) is not None else None,
                "time_s": None if info.get("time_s", None) is None else float(info.get("time_s")),
            }
        )
    try:
        if use_full_operator:
            _setup_linear_system(
                solver,
                K_r,
                A_full=K_r,
                free_idx=free_idx,
                preconditioning_matrix=preconditioning_matrix,
            )
            if getattr(solver, "supports_a_orthogonalization", lambda: True)():
                solver.A_orthogonalize(K_r)
            if use_local_build:
                f_local = _local_owned_rows_from_field(f_V, const_builder.owned_tangent_pattern)
                rhs_v_local = f_local - F_all_local
                rhs_v = f_free_local - F_all_free_local
                dW_free = _solve_linear_system_local(
                    solver,
                    K_r,
                    -G_free,
                    b_full=-G_local,
                    local_rhs=-G_local,
                    free_idx=free_idx,
                )
                _record_last_linear_solve("dW", "main")
                dV_free = _solve_linear_system_local(
                    solver,
                    K_r,
                    rhs_v,
                    b_full=rhs_v_local,
                    local_rhs=rhs_v_local,
                    free_idx=free_idx,
                )
                _record_last_linear_solve("dV", "main")
            else:
                dW_free = _solve_linear_system(solver, K_r, -G_free, free_idx=free_idx)
                _record_last_linear_solve("dW", "main")
                dV_free = _solve_linear_system(solver, K_r, f_free - F_all_free, free_idx=free_idx)
                _record_last_linear_solve("dV", "main")
        else:
            K_free = extract_submatrix_free(K_r, free_idx)
            _setup_linear_system(solver, K_free, A_full=K_r, free_idx=free_idx)
            if getattr(solver, "supports_a_orthogonalization", lambda: True)():
                solver.A_orthogonalize(K_free)
            dW_free = _solve_linear_system(solver, K_free, -G_free, free_idx=free_idx)
            _record_last_linear_solve("dW", "main")
            dV_free = _solve_linear_system(solver, K_free, f_free - F_all_free, free_idx=free_idx)
            _record_last_linear_solve("dV", "main")

        iter_delta_1 = _collector_delta(snap_before, _collector_snapshot(solver))
        W = np.zeros(U_it.size, dtype=np.float64)
        V = np.zeros(U_it.size, dtype=np.float64)
        W[free_idx] = np.asarray(dW_free, dtype=np.float64)
        V[free_idx] = np.asarray(dV_free, dtype=np.float64)
        W = W.reshape(shape, order="F")
        V = V.reshape(shape, order="F")
        fQ = _to_free_vector(f_V, q_mask)
        WQ = _to_free_vector(W, q_mask)
        VQ = _to_free_vector(V, q_mask)
        denom = float(np.dot(fQ, WQ))
        d_t = 0.0 if abs(denom) < 1.0e-30 else -float(np.dot(fQ, VQ)) / denom
        d_U = V + d_t * W
        alpha = damping_alg5(
            it_damp_max,
            U_it,
            t_it,
            d_U,
            d_t,
            f_V,
            criterion_before,
            q_mask,
            const_builder,
            f_free=f_free,
            f_local_free=f_free_local if use_local_build else None,
            comm=comm,
        )
        t_next = float(t_it + alpha * d_t)
        U_next = np.asarray(U_it + alpha * d_U, dtype=np.float64)
        denom_omega = _free_dot(f_V, U_next, q_mask)
        if denom_omega != 0.0:
            U_next = np.asarray(U_next * (omega_target / denom_omega), dtype=np.float64)

        residual_after, norm_f_after = _predictor_free_residual(
            U=np.asarray(U_next, dtype=np.float64),
            lambda_value=float(t_next),
            Q=q_mask,
            f=f_V,
            constitutive_matrix_builder=const_builder,
        )
        criterion_after = float(np.linalg.norm(residual_after))
        rel_after = float(criterion_after / max(float(norm_f_after), 1.0))
        half_step_rel_after_first = float(rel_after)
        omega_after = float(_free_dot(f_V, np.asarray(U_next, dtype=np.float64), q_mask))
        correction_norm = float(np.linalg.norm(_free(np.asarray(U_next - U_ini, dtype=np.float64), q_mask)))
        iter_delta_total = dict(iter_delta_1)
        step_wall_total = float(perf_counter() - step_t0)
        linear_iterations_total = int(iter_delta_1["iterations"])
        delta_lambda = float(d_t)
        accepted_delta_lambda = float(alpha * d_t)
        flag_N = 0
        controller_U_next = np.asarray(U_next, dtype=np.float64).copy()
        controller_lambda_next = float(t_next)
        controller_omega_after = float(omega_after)
        controller_criterion_after = float(criterion_after)
        controller_rel_after = float(rel_after)
        controller_correction_norm = float(correction_norm)
        controller_alpha = float(alpha)
        controller_delta_lambda = float(delta_lambda)
        controller_accepted_delta_lambda = float(accepted_delta_lambda)

        if float(rel_after) > float(same_matrix_halfstep_threshold):
            half_step_triggered = True
            newton_substeps = 2
            snap_before_half = _collector_snapshot(solver)
            second_eval = _evaluate_state(U_next, t_next, build_matrix=False)
            F_all2 = second_eval["F_all"]
            F_all_local2 = second_eval["F_all_local"]
            F_all_free_local2 = second_eval["F_all_free_local"]
            F_all_free2 = second_eval["F_all_free"]
            derivative_state_2 = _evaluate_lambda_derivative(U_next, t_next, second_eval)
            G_local2 = derivative_state_2["G_local"]
            G_free2 = np.asarray(derivative_state_2["G_free"], dtype=np.float64).reshape(-1)

            if use_full_operator:
                if use_local_build:
                    f_local = _local_owned_rows_from_field(f_V, const_builder.owned_tangent_pattern)
                    rhs_v_local = f_local - F_all_local2
                    rhs_v = f_free_local - F_all_free_local2
                    dW_free_2 = _solve_linear_system_local(
                        solver,
                        K_r,
                        -G_free2,
                        b_full=-G_local2,
                        local_rhs=-G_local2,
                        free_idx=free_idx,
                    )
                    _record_last_linear_solve("dW", "halfstep")
                    dV_free_2 = _solve_linear_system_local(
                        solver,
                        K_r,
                        rhs_v,
                        b_full=rhs_v_local,
                        local_rhs=rhs_v_local,
                        free_idx=free_idx,
                    )
                    _record_last_linear_solve("dV", "halfstep")
                else:
                    dW_free_2 = _solve_linear_system(solver, K_r, -G_free2, free_idx=free_idx)
                    _record_last_linear_solve("dW", "halfstep")
                    dV_free_2 = _solve_linear_system(solver, K_r, f_free - F_all_free2, free_idx=free_idx)
                    _record_last_linear_solve("dV", "halfstep")
            else:
                dW_free_2 = _solve_linear_system(solver, K_free, -G_free2, free_idx=free_idx)
                _record_last_linear_solve("dW", "halfstep")
                dV_free_2 = _solve_linear_system(solver, K_free, f_free - F_all_free2, free_idx=free_idx)
                _record_last_linear_solve("dV", "halfstep")

            iter_delta_2 = _collector_delta(snap_before_half, _collector_snapshot(solver))
            W2 = np.zeros(U_next.size, dtype=np.float64)
            V2 = np.zeros(U_next.size, dtype=np.float64)
            W2[free_idx] = np.asarray(dW_free_2, dtype=np.float64)
            V2[free_idx] = np.asarray(dV_free_2, dtype=np.float64)
            W2 = W2.reshape(shape, order="F")
            V2 = V2.reshape(shape, order="F")
            WQ2 = _to_free_vector(W2, q_mask)
            VQ2 = _to_free_vector(V2, q_mask)
            denom_2 = float(np.dot(fQ, WQ2))
            d_t_2 = 0.0 if abs(denom_2) < 1.0e-30 else -float(np.dot(fQ, VQ2)) / denom_2
            d_U_2 = V2 + d_t_2 * W2
            alpha_2 = damping_alg5(
                it_damp_max,
                U_next,
                t_next,
                d_U_2,
                d_t_2,
                f_V,
                criterion_after,
                q_mask,
                const_builder,
                f_free=f_free,
                f_local_free=f_free_local if use_local_build else None,
                comm=comm,
            )
            t_next_2 = float(t_next + alpha_2 * d_t_2)
            U_next_2 = np.asarray(U_next + alpha_2 * d_U_2, dtype=np.float64)
            denom_omega_2 = _free_dot(f_V, U_next_2, q_mask)
            if denom_omega_2 != 0.0:
                U_next_2 = np.asarray(U_next_2 * (omega_target / denom_omega_2), dtype=np.float64)

            residual_after_2, norm_f_after_2 = _predictor_free_residual(
                U=np.asarray(U_next_2, dtype=np.float64),
                lambda_value=float(t_next_2),
                Q=q_mask,
                f=f_V,
                constitutive_matrix_builder=const_builder,
            )
            criterion_after = float(np.linalg.norm(residual_after_2))
            rel_after = float(criterion_after / max(float(norm_f_after_2), 1.0))
            omega_after = float(_free_dot(f_V, np.asarray(U_next_2, dtype=np.float64), q_mask))
            correction_norm = float(np.linalg.norm(_free(np.asarray(U_next_2 - U_ini, dtype=np.float64), q_mask)))
            U_next = np.asarray(U_next_2, dtype=np.float64)
            t_next = float(t_next_2)
            W = W2
            V = V2
            d_U = d_U_2
            alpha = float(alpha_2)
            delta_lambda = float(d_t_2)
            accepted_delta_lambda = float(alpha_2 * d_t_2)
            half_step_linear_iterations = int(iter_delta_2["iterations"])
            linear_iterations_total += int(iter_delta_2["iterations"])
            iter_delta_total = {
                "iterations": int(iter_delta_1["iterations"]) + int(iter_delta_2["iterations"]),
                "solve_time": float(iter_delta_1["solve_time"]) + float(iter_delta_2["solve_time"]),
                "preconditioner_time": float(iter_delta_1["preconditioner_time"]) + float(iter_delta_2["preconditioner_time"]),
                "orthogonalization_time": float(iter_delta_1["orthogonalization_time"]) + float(iter_delta_2["orthogonalization_time"]),
            }
            step_wall_total = float(perf_counter() - step_t0)
            half_step_wall_time = float(iter_delta_2["solve_time"])

        if float(alpha) >= 1.0e-1 and getattr(solver, "supports_dynamic_deflation_basis", lambda: True)():
            solver.expand_deflation_basis(_to_free_vector(W, q_mask))
            solver.expand_deflation_basis(_to_free_vector(V, q_mask))
    finally:
        _release_iteration_resources(solver)
        _destroy_petsc_mat(K_free)
        if not _is_builder_cached_matrix(K_tangent, const_builder):
            _destroy_petsc_mat(K_tangent)
        if not use_full_operator and not _is_builder_cached_matrix(K_r, const_builder):
            _destroy_petsc_mat(K_r)

    return {
        "U_next": np.asarray(U_next, dtype=np.float64),
        "lambda_next": float(t_next),
        "omega_after": float(omega_after),
        "controller_U_next": np.asarray(controller_U_next, dtype=np.float64),
        "controller_lambda_next": float(controller_lambda_next),
        "controller_omega_after": float(controller_omega_after),
        "controller_criterion_after": float(controller_criterion_after),
        "controller_rel_after": float(controller_rel_after),
        "controller_correction_norm": float(controller_correction_norm),
        "controller_alpha": float(controller_alpha),
        "controller_delta_lambda": float(controller_delta_lambda),
        "controller_accepted_delta_lambda": float(controller_accepted_delta_lambda),
        "flag_N": int(flag_N),
        "it_newt": int(newton_substeps),
        "history": {},
        "step_wall": float(step_wall_total),
        "delta": iter_delta_total,
        "criterion_after": float(criterion_after),
        "rel_after": float(rel_after),
        "correction_norm": float(correction_norm),
        "alpha": float(alpha),
        "criterion_before": float(criterion_before),
        "rel_before": float(rel_before),
        "delta_lambda": float(delta_lambda),
        "accepted_delta_lambda": float(accepted_delta_lambda),
        "linear_iterations": int(linear_iterations_total),
        "newton_substeps": int(newton_substeps),
        "half_step_triggered": bool(half_step_triggered),
        "half_step_linear_iterations": int(half_step_linear_iterations),
        "half_step_wall_time": float(half_step_wall_time),
        "half_step_rel_after_first": float(half_step_rel_after_first),
        "linear_solve_infos": linear_solve_infos,
        "committed_lambda_after": float(t_next),
        "committed_omega_after": float(omega_after),
        "committed_criterion_after": float(criterion_after),
        "committed_rel_after": float(rel_after),
        "committed_correction_norm": float(correction_norm),
    }


def _build_case(run_info: dict, *, max_deflation_basis_vectors: int) -> dict[str, object]:
    params = run_info["params"]
    mesh_info = run_info["mesh"]
    mesh_path = Path(mesh_info["mesh_file"])
    node_ordering = str(params["node_ordering"])
    elem_type = str(params["elem_type"]).upper()
    mesh_boundary_type = int(params["mesh_boundary_type"])
    material_rows = params["material_rows"]
    materials = _material_specs(material_rows)
    partition_count = int(PETSc.COMM_WORLD.getSize()) if node_ordering.lower() == "block_metis" else None
    pmg_hierarchy = build_3d_pmg_hierarchy(
        mesh_path,
        boundary_type=mesh_boundary_type,
        node_ordering=node_ordering,
        reorder_parts=partition_count,
        material_rows=np.asarray(material_rows, dtype=np.float64).tolist(),
        comm=PETSc.COMM_WORLD,
    )
    coord = pmg_hierarchy.fine_level.coord.astype(np.float64)
    elem = pmg_hierarchy.fine_level.elem.astype(np.int64)
    q_mask = pmg_hierarchy.fine_level.q_mask.astype(bool)
    mesh = load_mesh_from_file(mesh_path, boundary_type=mesh_boundary_type, elem_type=elem_type)
    material_identifier = np.asarray(mesh.material, dtype=np.int64).ravel()

    n_q = int(quadrature_volume_3d(elem_type)[0].shape[1])
    n_int = int(elem.shape[1] * n_q)
    c0, phi, psi, shear, bulk, lame, gamma = heterogenous_materials(
        material_identifier,
        np.ones(n_int, dtype=bool),
        n_q,
        materials,
    )

    solver_type = str(run_info["run_info"]["solver_type"])
    mpi_distribute_by_nodes = bool(params["mpi_distribute_by_nodes"])
    use_owned_mpi_tangent_path = use_owned_tangent_path(
        solver_type=solver_type,
        mpi_distribute_by_nodes=mpi_distribute_by_nodes,
    )
    use_lightweight_mpi_path = use_lightweight_mpi_elastic_path(
        solver_type=solver_type,
        mpi_distribute_by_nodes=mpi_distribute_by_nodes,
        constitutive_mode=str(params["constitutive_mode"]),
    )

    B = None
    weight = np.zeros(n_int, dtype=np.float64)
    elastic_rows = None
    tangent_pattern = None
    if use_lightweight_mpi_path:
        elastic_rows = assemble_owned_elastic_rows_for_comm(
            coord,
            elem,
            q_mask,
            material_identifier,
            materials,
            PETSc.COMM_WORLD,
            elem_type=elem_type,
        )
        global_size = int(coord.shape[0] * coord.shape[1])
        K_elast = local_csr_to_petsc_aij_matrix(
            elastic_rows.local_matrix,
            global_shape=(global_size, global_size),
            comm=PETSc.COMM_WORLD,
            block_size=coord.shape[0],
        )
        rhs_parts = PETSc.COMM_WORLD.tompi4py().allgather(np.asarray(elastic_rows.local_rhs, dtype=np.float64))
        f_V = np.concatenate(rhs_parts).reshape(coord.shape[0], coord.shape[1], order="F")
    else:
        assembly = assemble_strain_operator(coord, elem, elem_type, dim=3)
        from slope_stability.fem.assembly import build_elastic_stiffness_matrix

        K_elast, weight, B = build_elastic_stiffness_matrix(assembly, shear, lame, bulk)
        f_v_int = np.vstack(
            (
                np.zeros(assembly.n_int, dtype=np.float64),
                -gamma.astype(np.float64),
                np.zeros(assembly.n_int, dtype=np.float64),
            )
        )
        f_V = vector_volume(assembly, f_v_int, weight)

    const_builder = ConstitutiveOperator(
        B=B,
        c0=c0,
        phi=phi,
        psi=psi,
        Davis_type=str(params["davis_type"]),
        shear=shear,
        bulk=bulk,
        lame=lame,
        WEIGHT=weight,
        n_strain=6,
        n_int=n_int,
        dim=3,
        q_mask=q_mask,
    )

    if use_owned_mpi_tangent_path:
        from slope_stability.utils import owned_block_range

        row0, row1 = owned_block_range(coord.shape[1], coord.shape[0], PETSc.COMM_WORLD)
        tangent_pattern = prepare_owned_tangent_pattern(
            coord,
            elem,
            q_mask,
            material_identifier,
            materials,
            (row0 // coord.shape[0], row1 // coord.shape[0]),
            elem_type=elem_type,
            include_unique=(str(params["constitutive_mode"]).lower() != "overlap"),
            include_legacy_scatter=(str(params["tangent_kernel"]).lower() == "legacy"),
            include_overlap_B=(str(params["tangent_kernel"]).lower() == "legacy"),
            elastic_rows=elastic_rows if use_lightweight_mpi_path else None,
        )
        const_builder.set_owned_tangent_pattern(
            tangent_pattern,
            use_compiled=True,
            tangent_kernel=str(params["tangent_kernel"]),
            constitutive_mode=str(params["constitutive_mode"]),
            use_compiled_constitutive=True,
        )

    preconditioner_options = {
        "threads": 16,
        "print_level": 0,
        "use_as_preconditioner": True,
        "factor_solver_type": params["factor_solver_type"],
        "pc_backend": params["pc_backend"],
        "pmg_coarse_mesh_path": None,
        "preconditioner_matrix_source": str(params["preconditioner_matrix_source"]),
        "preconditioner_matrix_policy": str(params["preconditioner_matrix_policy"]),
        "preconditioner_rebuild_policy": str(params["preconditioner_rebuild_policy"]),
        "preconditioner_rebuild_interval": int(params["preconditioner_rebuild_interval"]),
        "mpi_distribute_by_nodes": bool(params["mpi_distribute_by_nodes"]),
        "use_coordinates": True,
        "max_deflation_basis_vectors": int(max_deflation_basis_vectors),
        "full_system_preconditioner": False,
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_max_it": 3,
        "mg_levels_pc_type": "sor",
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "hypre",
        "mg_coarse_pc_hypre_type": "boomeramg",
        "pmg_hierarchy": pmg_hierarchy,
    }
    preconditioner_options.update(_parse_petsc_opt_entries(params["petsc_opt"]))
    linear_system_solver = SolverFactory.create(
        solver_type,
        tolerance=1.0e-1,
        max_iterations=100,
        deflation_basis_tolerance=1.0e-3,
        verbose=False,
        q_mask=q_mask,
        coord=coord,
        preconditioner_options=preconditioner_options,
    )
    enable_diagnostics = getattr(linear_system_solver, "enable_diagnostics", None)
    if callable(enable_diagnostics):
        enable_diagnostics(True)
    return {
        "coord": coord,
        "q_mask": q_mask,
        "f_V": f_V,
        "K_elast": K_elast,
        "const_builder": const_builder,
        "solver": linear_system_solver,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed d_omega/5 micro-walk with one Newton iteration per advance.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/rank8_micro100",
    )
    parser.add_argument("--micro-divisor", type=float, default=5.0)
    parser.add_argument("--micro-steps", type=int, default=100)
    parser.add_argument("--mode", type=str, default="fixed", choices=("fixed", "adaptive"))
    parser.add_argument("--rolling-basis-size", type=int, default=20)
    parser.add_argument("--basis-compression", type=str, default="none", choices=("none", "svd"))
    parser.add_argument("--svd-basis-size", type=int, default=10)
    parser.add_argument("--omega-max-stop", type=float, default=6.52e6)
    parser.add_argument("--disable-deflation-basis", action="store_true")
    parser.add_argument("--stall-recovery", action="store_true")
    parser.add_argument("--stall-trigger", type=int, default=3)
    parser.add_argument("--stall-extra-newton-steps", type=int, default=2)
    parser.add_argument("--stall-rel-res-threshold", type=float, default=5.0e-3)
    parser.add_argument("--stall-retry-factor", type=float, default=0.25)
    parser.add_argument("--omega-freeze-alpha-threshold", type=float, default=0.5)
    parser.add_argument("--omega-growth-factor", type=float, default=1.5)
    parser.add_argument("--no-progress-shrink-factor", type=float, default=0.8)
    parser.add_argument("--correction-gate-mode", type=str, default="relative", choices=("relative", "absolute"))
    parser.add_argument("--correction-rel-threshold", type=float, default=0.7)
    parser.add_argument("--correction-abs-threshold", type=float, default=1.0)
    parser.add_argument("--step-length-cap-mode", type=str, default="none", choices=("none", "initial_segment"))
    parser.add_argument("--step-length-cap-factor", type=float, default=1.0)
    parser.add_argument("--same-matrix-halfstep-threshold", type=float, default=2.0e-2)
    parser.add_argument("--min-domega-initial", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    data_dir = out_dir / "data"
    if PETSc.COMM_WORLD.getRank() == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        progress_path = data_dir / "progress.jsonl"
        if progress_path.exists():
            progress_path.unlink()
    PETSc.COMM_WORLD.Barrier()
    progress_path = data_dir / "progress.jsonl"

    run_info = json.loads(BASELINE_RUN_INFO.read_text(encoding="utf-8"))
    built = _build_case(run_info, max_deflation_basis_vectors=int(args.rolling_basis_size))
    q_mask = np.asarray(built["q_mask"], dtype=bool)
    f_V = np.asarray(built["f_V"], dtype=np.float64)
    K_elast = built["K_elast"]
    const_builder = built["const_builder"]
    solver = built["solver"]
    if bool(args.disable_deflation_basis):
        _disable_solver_deflation_features(solver)
    params = run_info["params"]

    init_t0 = perf_counter()
    init_snap_before = _collector_snapshot(solver)
    U1, U2, omega1, omega2, lambda1, lambda2, init_newton_its = init_phase_SSR_indirect_continuation(
        lambda_init=float(params["lambda_init"]),
        d_lambda_init=float(params["d_lambda_init"]),
        d_lambda_min=float(params["d_lambda_min"]),
        it_newt_max=int(params["it_newt_max"]),
        it_damp_max=int(params["it_damp_max"]),
        tol=float(params["tol"]),
        r_min=float(params["r_min"]),
        K_elast=K_elast,
        Q=q_mask,
        f=f_V,
        constitutive_matrix_builder=const_builder,
        linear_system_solver=solver,
    )
    init_wall = float(perf_counter() - init_t0)
    init_delta = _collector_delta(init_snap_before, _collector_snapshot(solver))
    domega_base = float(omega2 - omega1)
    domega_micro = float(domega_base / float(args.micro_divisor))
    dlambda_base = float(lambda2 - lambda1)
    omega_progress_tol = float(max(1.0e-9, 1.0e-6 * max(abs(domega_base), 1.0)))
    domega_floor = float(domega_micro if bool(args.min_domega_initial) else 1.0e-12)

    def _clamp_domega(value: float) -> float:
        return float(max(float(value), domega_floor))

    init_payload = {
        "event": "init_complete",
        "lambda_hist": [float(lambda1), float(lambda2)],
        "omega_hist": [float(omega1), float(omega2)],
        "init_newton_iterations": [int(v) for v in init_newton_its],
        "init_wall_time": float(init_wall),
        "init_linear_iterations": int(init_delta["iterations"]),
        "init_linear_solve_time": float(init_delta["solve_time"]),
        "init_linear_preconditioner_time": float(init_delta["preconditioner_time"]),
        "init_linear_orthogonalization_time": float(init_delta["orthogonalization_time"]),
        "domega_base": float(domega_base),
        "domega_micro": float(domega_micro),
        "omega_progress_tolerance": float(omega_progress_tol),
        "micro_divisor": float(args.micro_divisor),
        "micro_steps_requested": int(args.micro_steps),
        "mode": str(args.mode),
        "rolling_basis_size": int(args.rolling_basis_size),
        "basis_compression": str(args.basis_compression),
        "svd_basis_size": int(args.svd_basis_size),
        "disable_deflation_basis": bool(args.disable_deflation_basis),
        "stall_recovery": bool(args.stall_recovery),
        "stall_trigger": int(args.stall_trigger),
        "stall_extra_newton_steps": int(args.stall_extra_newton_steps),
        "stall_rel_res_threshold": float(args.stall_rel_res_threshold),
        "stall_retry_factor": float(args.stall_retry_factor),
        "omega_freeze_alpha_threshold": float(args.omega_freeze_alpha_threshold),
        "omega_growth_factor": float(args.omega_growth_factor),
        "no_progress_shrink_factor": float(args.no_progress_shrink_factor),
        "correction_gate_mode": str(args.correction_gate_mode),
        "correction_rel_threshold": float(args.correction_rel_threshold),
        "correction_abs_threshold": float(args.correction_abs_threshold),
        "step_length_cap_mode": str(args.step_length_cap_mode),
        "step_length_cap_factor": float(args.step_length_cap_factor),
        "same_matrix_halfstep_threshold": float(args.same_matrix_halfstep_threshold),
        "min_domega_initial": bool(args.min_domega_initial),
        "domega_floor": float(domega_floor),
    }
    _write_jsonl(progress_path, init_payload)

    U_anchor_prev = np.asarray(U1, dtype=np.float64).copy()
    U_anchor_curr = np.asarray(U2, dtype=np.float64).copy()
    omega_anchor_prev = float(omega1)
    omega_anchor_curr = float(omega2)
    lambda_anchor_prev = float(lambda1)
    lambda_anchor_curr = float(lambda2)
    U_curr = np.asarray(U2, dtype=np.float64).copy()
    omega_curr = float(omega2)
    lambda_curr = float(lambda2)
    solution_hist_free: list[np.ndarray] = [
        np.asarray(_free(np.asarray(U1, dtype=np.float64), q_mask), dtype=np.float64).reshape(-1).copy(),
        np.asarray(_free(np.asarray(U2, dtype=np.float64), q_mask), dtype=np.float64).reshape(-1).copy(),
    ]
    domega_current = float(domega_micro)
    omega_frozen_due_alpha = False
    omega_frozen_due_residual = False
    omega_frozen_due_correction = False
    omega_increase_frozen = False
    lambda_freeze_reference = float("nan")
    shrink_factor = 0.66
    last_successful_domega = float(domega_micro)
    consecutive_no_progress = 0

    records: list[dict[str, float | int | bool | None]] = []
    total_t0 = perf_counter()
    stop_reason = "completed"

    for micro_index in range(1, int(args.micro_steps) + 1):
        if np.isfinite(float(args.omega_max_stop)) and float(omega_curr) >= float(args.omega_max_stop) - 1.0e-9:
            stop_reason = "omega_max_stop"
            break
        if str(args.mode) == "adaptive":
            domega_used = 0.0 if (omega_frozen_due_alpha or omega_frozen_due_residual or omega_frozen_due_correction or omega_increase_frozen) else float(domega_current)
        else:
            domega_used = float(domega_micro)
        step_length_cap_limit = float("nan")
        step_length_cap_raw = float("nan")
        step_length_cap_applied = False
        if (
            str(args.step_length_cap_mode) == "initial_segment"
            and float(domega_used) > 0.0
        ):
            domega_capped, step_length_cap_raw, step_length_cap_limit = _initial_segment_length_cap(
                domega_candidate=float(domega_used),
                domega_initial=float(domega_base),
                dlambda_initial=float(dlambda_base),
                omega_anchor_prev=float(omega_anchor_prev),
                omega_anchor_curr=float(omega_anchor_curr),
                lambda_anchor_prev=float(lambda_anchor_prev),
                lambda_anchor_curr=float(lambda_anchor_curr),
                cap_factor=float(args.step_length_cap_factor),
            )
            step_length_cap_applied = bool(domega_capped + 1.0e-12 < float(domega_used))
            domega_used = float(domega_capped)
        basis_dim = _restore_rolling_solution_basis(
            solver,
            solution_hist_free,
            max_vectors=int(args.rolling_basis_size),
            compression_mode=str(args.basis_compression),
            svd_basis_size=int(args.svd_basis_size),
        )
        omega_target = float(omega_curr + domega_used)
        if np.isfinite(float(args.omega_max_stop)):
            omega_target = float(min(omega_target, float(args.omega_max_stop)))
        if abs(float(omega_target) - float(omega_curr)) <= 1.0e-12:
            U_ini = np.asarray(U_curr, dtype=np.float64).copy()
            lambda_ini = float(lambda_curr)
            predictor_kind = "current_state_fixed_omega"
        else:
            U_ini, lambda_ini, predictor_kind = _secant_predictor(
                omega_old=float(omega_anchor_prev),
                omega=float(omega_anchor_curr),
                omega_target=float(omega_target),
                U_old=np.asarray(U_anchor_prev, dtype=np.float64),
                U=np.asarray(U_anchor_curr, dtype=np.float64),
                lambda_value=float(lambda_curr),
            )
            U_ini = _rescale_to_target_omega(np.asarray(U_ini, dtype=np.float64), float(omega_target), f_V, q_mask)
        step_result = _run_single_newton_micro_step(
            U_ini=np.asarray(U_ini, dtype=np.float64),
            lambda_ini=float(lambda_ini),
            omega_target=float(omega_target),
            params=params,
            K_elast=K_elast,
            q_mask=q_mask,
            f_V=f_V,
            const_builder=const_builder,
            solver=solver,
            same_matrix_halfstep_threshold=float(args.same_matrix_halfstep_threshold),
        )
        committed_U_next = np.asarray(step_result["U_next"], dtype=np.float64)
        committed_lambda_next = float(step_result["lambda_next"])
        committed_omega_after = float(step_result["omega_after"])
        committed_criterion_after = float(step_result.get("committed_criterion_after", step_result["criterion_after"]))
        committed_rel_after = float(step_result.get("committed_rel_after", step_result["rel_after"]))
        committed_correction_norm = float(step_result.get("committed_correction_norm", step_result["correction_norm"]))
        controller_U_next = np.asarray(step_result.get("controller_U_next", committed_U_next), dtype=np.float64)
        controller_lambda_next = float(step_result.get("controller_lambda_next", committed_lambda_next))
        controller_omega_after = float(step_result.get("controller_omega_after", committed_omega_after))
        controller_criterion_after = float(step_result.get("controller_criterion_after", step_result["criterion_after"]))
        controller_rel_after = float(step_result.get("controller_rel_after", step_result["rel_after"]))
        controller_correction_norm = float(step_result.get("controller_correction_norm", step_result["correction_norm"]))
        controller_alpha = float(step_result.get("controller_alpha", step_result["alpha"]))
        controller_delta_lambda = float(step_result.get("controller_delta_lambda", step_result["delta_lambda"]))
        controller_accepted_delta_lambda = float(step_result.get("controller_accepted_delta_lambda", step_result["accepted_delta_lambda"]))
        flag_N = int(step_result["flag_N"])
        it_newt = int(step_result["it_newt"])
        history = step_result["history"]
        step_wall = float(step_result["step_wall"])
        delta = step_result["delta"]
        criterion_after = float(controller_criterion_after)
        rel_after = float(controller_rel_after)
        omega_after = float(controller_omega_after)
        correction_norm = float(controller_correction_norm)
        alpha = float(controller_alpha)
        criterion_before = float(step_result["criterion_before"])
        rel_before = float(step_result["rel_before"])
        delta_lambda = float(controller_delta_lambda)
        accepted_delta_lambda = float(controller_accepted_delta_lambda)
        linear_iterations = int(step_result["linear_iterations"])
        anchor_increment_norm = float(
            np.linalg.norm(
                _free(np.asarray(U_anchor_curr, dtype=np.float64) - np.asarray(U_anchor_prev, dtype=np.float64), q_mask)
            )
        )
        correction_rel = float(correction_norm / max(anchor_increment_norm, 1.0e-12))
        domega_next = float(domega_current)
        rolled_back_due_residual = False
        state_U_next = np.asarray(committed_U_next, dtype=np.float64).copy()
        state_lambda_next = float(committed_lambda_next)
        state_omega_after = float(committed_omega_after)
        stall_repair_steps = 0
        stall_repair_rejected_steps = 0
        fixed_omega_outer_rejected = False
        same_matrix_halfstep_triggered = bool(step_result.get("half_step_triggered", False))
        half_step_linear_iterations = int(step_result.get("half_step_linear_iterations", 0))
        half_step_wall_time = float(step_result.get("half_step_wall_time", 0.0))
        half_step_rel_after_first = float(step_result.get("half_step_rel_after_first", np.nan))
        newton_substeps = int(step_result.get("newton_substeps", 1))
        linear_solve_infos: list[dict[str, object]] = list(step_result.get("linear_solve_infos", []))
        if str(args.mode) == "adaptive":
            if float(rel_after) > 1.0e1:
                rolled_back_due_residual = True
                domega_next = _clamp_domega(shrink_factor * domega_current)
                omega_frozen_due_alpha = False
                omega_frozen_due_residual = False
                omega_frozen_due_correction = False
                state_U_next = np.asarray(U_curr, dtype=np.float64).copy()
                state_lambda_next = float(lambda_curr)
                state_omega_after = float(omega_curr)
            else:
                if float(accepted_delta_lambda) < 0.0:
                    omega_increase_frozen = True
                    lambda_freeze_reference = float(lambda_curr)
                    domega_next = _clamp_domega(shrink_factor * domega_current)
                if omega_increase_frozen and (
                    float(controller_lambda_next) > float(lambda_freeze_reference)
                    or float(rel_after) < 1.0e-3
                    or float(rel_after) < 0.8 * max(float(rel_before), 1.0e-12)
                ):
                    omega_increase_frozen = False
                    lambda_freeze_reference = float("nan")
                omega_frozen_due_alpha = bool(float(alpha) < float(args.omega_freeze_alpha_threshold))
                omega_frozen_due_residual = bool(float(rel_after) > 1.0)
                if str(args.correction_gate_mode) == "absolute":
                    omega_frozen_due_correction = bool(float(correction_norm) > float(args.correction_abs_threshold))
                else:
                    omega_frozen_due_correction = bool(float(correction_rel) > float(args.correction_rel_threshold))
                if float(rel_after) < 1.0e-1 and (not omega_increase_frozen) and float(alpha) >= 0.5 and abs(float(domega_used)) > float(omega_progress_tol):
                    domega_next = _clamp_domega(domega_current * float(args.omega_growth_factor))
            if abs(float(state_omega_after) - float(omega_curr)) <= float(omega_progress_tol):
                domega_next = _clamp_domega(float(args.no_progress_shrink_factor) * domega_next)
        omega_progressed = bool(abs(float(state_omega_after) - float(omega_curr)) > float(omega_progress_tol))
        if omega_progressed:
            consecutive_no_progress = 0
            last_successful_domega = float(max(abs(domega_used), 1.0e-12))
        else:
            consecutive_no_progress += 1
        if (
            bool(args.stall_recovery)
            and str(args.mode) == "adaptive"
            and (not rolled_back_due_residual)
            and (not omega_progressed)
            and consecutive_no_progress >= int(args.stall_trigger)
        ):
            repair_U = np.asarray(state_U_next, dtype=np.float64).copy()
            repair_lambda = float(state_lambda_next)
            for _ in range(int(args.stall_extra_newton_steps)):
                if float(rel_after) <= float(args.stall_rel_res_threshold):
                    break
                prev_repair_rel = float(rel_after)
                repair_result = _run_single_newton_micro_step(
                    U_ini=repair_U,
                    lambda_ini=repair_lambda,
                    omega_target=float(omega_curr),
                    params=params,
                    K_elast=K_elast,
                    q_mask=q_mask,
                    f_V=f_V,
                    const_builder=const_builder,
                    solver=solver,
                )
                if float(repair_result["rel_after"]) > float(prev_repair_rel):
                    domega_next = _clamp_domega(shrink_factor * domega_next)
                    stall_repair_rejected_steps += 1
                    break
                repair_U = np.asarray(repair_result["U_next"], dtype=np.float64)
                repair_lambda = float(repair_result["lambda_next"])
                state_U_next = repair_U.copy()
                state_lambda_next = float(repair_lambda)
                state_omega_after = float(omega_curr)
                criterion_after = float(repair_result["criterion_after"])
                rel_after = float(repair_result["rel_after"])
                correction_norm = float(repair_result["correction_norm"])
                correction_rel = float(correction_norm / max(anchor_increment_norm, 1.0e-12))
                alpha = float(repair_result["alpha"])
                criterion_before = float(repair_result["criterion_before"])
                rel_before = float(repair_result["rel_before"])
                delta_lambda = float(repair_result["delta_lambda"])
                accepted_delta_lambda = float(repair_result["accepted_delta_lambda"])
                linear_iterations += int(repair_result["linear_iterations"])
                step_wall += float(repair_result["step_wall"])
                delta["iterations"] = int(delta.get("iterations", 0)) + int(repair_result["delta"].get("iterations", 0))
                delta["solve_time"] = float(delta.get("solve_time", 0.0)) + float(repair_result["delta"].get("solve_time", 0.0))
                delta["preconditioner_time"] = float(delta.get("preconditioner_time", 0.0)) + float(repair_result["delta"].get("preconditioner_time", 0.0))
                delta["orthogonalization_time"] = float(delta.get("orthogonalization_time", 0.0)) + float(repair_result["delta"].get("orthogonalization_time", 0.0))
                flag_N = int(max(flag_N, int(repair_result["flag_N"])))
                it_newt += int(repair_result["it_newt"])
                stall_repair_steps += 1
                linear_solve_infos.extend(list(repair_result.get("linear_solve_infos", [])))
            if float(rel_after) < float(args.stall_rel_res_threshold):
                domega_next = _clamp_domega(float(args.stall_retry_factor) * last_successful_domega)
                omega_frozen_due_alpha = False
                omega_frozen_due_residual = False
                omega_frozen_due_correction = False
        if (
            str(args.mode) == "adaptive"
            and (not rolled_back_due_residual)
            and abs(float(domega_used)) <= 1.0e-12
            and float(rel_after) > float(step_result["rel_before"])
        ):
            fixed_omega_outer_rejected = True
            domega_next = _clamp_domega(shrink_factor * domega_next)
            state_U_next = np.asarray(U_curr, dtype=np.float64).copy()
            state_lambda_next = float(lambda_curr)
            state_omega_after = float(omega_curr)
            criterion_after = float(step_result["criterion_before"])
            rel_after = float(step_result["rel_before"])
            correction_norm = 0.0
            correction_rel = 0.0
        linear_any_not_converged = any(info.get("converged") is False for info in linear_solve_infos)
        linear_any_hit_max_iterations = any(bool(info.get("hit_max_iterations")) for info in linear_solve_infos)
        linear_last_info = linear_solve_infos[-1] if linear_solve_infos else {}
        record = {
            "event": "micro_step_complete",
            "micro_index": int(micro_index),
            "mode": str(args.mode),
            "predictor_kind": str(predictor_kind),
            "omega_before": float(omega_curr),
            "omega_target": float(omega_target),
            "omega_after": float(omega_after),
            "domega_used": float(domega_used),
            "domega_next": float(domega_next),
            "omega_progress_tolerance": float(omega_progress_tol),
            "step_length_cap_mode": str(args.step_length_cap_mode),
            "step_length_cap_factor": float(args.step_length_cap_factor),
            "step_length_cap_applied": bool(step_length_cap_applied),
            "step_length_cap_raw_length": float(step_length_cap_raw),
            "step_length_cap_limit": float(step_length_cap_limit),
            "lambda_before": float(lambda_curr),
            "lambda_ini": float(lambda_ini),
            "lambda_after": float(controller_lambda_next),
            "committed_omega_after": float(state_omega_after),
            "committed_lambda_after": float(state_lambda_next),
            "criterion_before": float(criterion_before),
            "criterion_after": float(criterion_after),
            "committed_criterion_after": float(criterion_after),
            "rel_residual_before": float(rel_before),
            "rel_residual_after": float(rel_after),
            "committed_rel_residual_after": float(rel_after),
            "alpha": float(alpha),
            "delta_lambda": float(delta_lambda),
            "accepted_delta_lambda": float(accepted_delta_lambda),
            "linear_iterations": int(linear_iterations),
            "linear_solve_time": float(delta["solve_time"]),
            "linear_preconditioner_time": float(delta["preconditioner_time"]),
            "linear_orthogonalization_time": float(delta["orthogonalization_time"]),
            "iteration_wall_time": float(step_wall),
            "correction_free_norm": float(correction_norm),
            "committed_correction_free_norm": float(correction_norm),
            "rolling_basis_dim": int(basis_dim),
            "omega_frozen_due_alpha_next": bool(omega_frozen_due_alpha),
            "omega_frozen_due_residual_next": bool(omega_frozen_due_residual),
            "omega_frozen_due_correction_next": bool(omega_frozen_due_correction),
            "omega_increase_frozen": bool(omega_increase_frozen),
            "rolled_back_due_residual": bool(rolled_back_due_residual),
            "basis_compression": str(args.basis_compression),
            "correction_gate_mode": str(args.correction_gate_mode),
            "correction_abs_threshold": float(args.correction_abs_threshold),
            "correction_rel_threshold": float(args.correction_rel_threshold),
            "omega_progressed": bool(omega_progressed),
            "lambda_freeze_reference": float(lambda_freeze_reference) if np.isfinite(lambda_freeze_reference) else None,
            "flag_N": int(flag_N),
            "newton_iterations": int(it_newt),
            "newton_substeps": int(newton_substeps),
            "same_matrix_halfstep_triggered": bool(same_matrix_halfstep_triggered),
            "same_matrix_halfstep_threshold": float(args.same_matrix_halfstep_threshold),
            "half_step_linear_iterations": int(half_step_linear_iterations),
            "half_step_wall_time": float(half_step_wall_time),
            "half_step_rel_after_first": float(half_step_rel_after_first),
            "correction_rel": float(correction_rel),
            "anchor_increment_norm": float(anchor_increment_norm),
            "consecutive_no_progress": int(consecutive_no_progress),
            "stall_repair_steps": int(stall_repair_steps),
            "stall_repair_rejected_steps": int(stall_repair_rejected_steps),
            "fixed_omega_outer_rejected": bool(fixed_omega_outer_rejected),
            "linear_solve_infos": linear_solve_infos,
            "linear_last_converged": linear_last_info.get("converged"),
            "linear_last_hit_max_iterations": linear_last_info.get("hit_max_iterations"),
            "linear_last_converged_reason": linear_last_info.get("converged_reason"),
            "linear_last_true_residual_final": linear_last_info.get("true_residual_final"),
            "linear_last_reported_residual_final": linear_last_info.get("reported_residual_final"),
            "linear_any_not_converged": bool(linear_any_not_converged),
            "linear_any_hit_max_iterations": bool(linear_any_hit_max_iterations),
            "total_wall_time": float(init_wall + (perf_counter() - total_t0)),
        }
        records.append(record)
        _write_jsonl(progress_path, record)

        if not rolled_back_due_residual:
            U_curr = np.asarray(state_U_next, dtype=np.float64).copy()
            omega_curr = float(state_omega_after)
            lambda_curr = float(state_lambda_next)
            if omega_progressed:
                U_anchor_prev = np.asarray(U_anchor_curr, dtype=np.float64).copy()
                U_anchor_curr = np.asarray(U_curr, dtype=np.float64).copy()
                omega_anchor_prev = float(omega_anchor_curr)
                omega_anchor_curr = float(omega_curr)
                lambda_anchor_prev = float(lambda_anchor_curr)
                lambda_anchor_curr = float(lambda_curr)
                solution_hist_free.append(np.asarray(_free(np.asarray(U_curr, dtype=np.float64), q_mask), dtype=np.float64).reshape(-1).copy())
                if len(solution_hist_free) > int(args.rolling_basis_size):
                    solution_hist_free = solution_hist_free[-int(args.rolling_basis_size) :]
        domega_current = float(domega_next)

        if not (
            np.isfinite(float(lambda_curr))
            and float(lambda_curr) > 0.0
            and np.isfinite(float(omega_curr))
            and np.isfinite(float(criterion_after))
            and np.isfinite(float(rel_after))
        ):
            stop_reason = "nonfinite_or_nonpositive_state"
            break

    final_payload = {
        "event": "run_complete",
        "stop_reason": str(stop_reason),
        "micro_steps_completed": int(len(records)),
        "micro_steps_requested": int(args.micro_steps),
        "mode": str(args.mode),
        "final_lambda": float(lambda_curr),
        "final_omega": float(omega_curr),
        "domega_base": float(domega_base),
        "domega_micro": float(domega_micro),
        "domega_last": float(domega_current),
        "omega_max_stop": float(args.omega_max_stop),
        "runtime_seconds": float(init_wall + (perf_counter() - total_t0)),
    }
    _write_jsonl(progress_path, final_payload)

    if PETSc.COMM_WORLD.getRank() == 0:
        arrays: dict[str, np.ndarray] = {}
        scalar_fields = [
            "micro_index",
            "omega_before",
            "omega_target",
            "omega_after",
            "lambda_before",
            "lambda_ini",
            "lambda_after",
            "criterion_before",
            "criterion_after",
            "rel_residual_before",
            "rel_residual_after",
            "alpha",
            "delta_lambda",
            "accepted_delta_lambda",
            "linear_iterations",
            "linear_solve_time",
            "linear_preconditioner_time",
            "linear_orthogonalization_time",
            "iteration_wall_time",
            "correction_free_norm",
            "flag_N",
            "newton_iterations",
            "total_wall_time",
            "domega_used",
            "domega_next",
            "rolling_basis_dim",
        ]
        for field in scalar_fields:
            arrays[field] = np.asarray([rec[field] for rec in records], dtype=np.float64)
        np.savez(data_dir / "walk_history.npz", **arrays)
        (data_dir / "summary.json").write_text(
            json.dumps(
                {
                    "init": init_payload,
                    "final": final_payload,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    PETSc.COMM_WORLD.Barrier()


if __name__ == "__main__":
    main()
