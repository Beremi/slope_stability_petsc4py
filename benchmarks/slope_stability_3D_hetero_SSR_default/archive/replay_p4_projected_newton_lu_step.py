#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from slope_stability.nonlinear.damping import damping_alg5
from slope_stability.nonlinear.newton import _combine_matrices, _is_builder_cached_matrix
from slope_stability.utils import (
    flatten_field,
    full_field_from_free_values,
    matvec_to_numpy,
    q_to_free_indices,
    release_petsc_aij_matrix,
)


DEFAULT_CASE_DIR = Path("artifacts/p4_pmg_shell_best_rank8_full/p4_rank8_step100")
DEFAULT_OUT_DIR = Path("artifacts/p4_step7_projected_newton_lu")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relpath(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, start=from_path.parent)


def _load_case(case_dir: Path) -> dict[str, object]:
    data_dir = case_dir / "data"
    npz = np.load(data_dir / "petsc_run.npz", allow_pickle=True)
    run_info = json.loads((data_dir / "run_info.json").read_text(encoding="utf-8"))
    return {"case_dir": case_dir, "npz": npz, "run_info": run_info}


def _safe_destroy_petsc_mat(A, *, constitutive_builder=None) -> None:
    if A is None:
        return
    if _is_builder_cached_matrix(A, constitutive_builder):
        return
    try:
        release_petsc_aij_matrix(A)
    except Exception:
        pass
    destroy = getattr(A, "destroy", None)
    if callable(destroy):
        try:
            destroy()
        except Exception:
            pass


def _secant_predict(
    *,
    omega_hist: np.ndarray,
    u_full_hist: np.ndarray,
    lambda_hist: np.ndarray,
    idx: int,
) -> tuple[np.ndarray, float, float]:
    omega_prev = float(omega_hist[idx - 1])
    omega_prevprev = float(omega_hist[idx - 2])
    omega_target = float(omega_hist[idx])
    alpha_sec = (omega_target - omega_prev) / (omega_prev - omega_prevprev)
    u_prev = np.asarray(u_full_hist[idx - 1], dtype=np.float64)
    u_prevprev = np.asarray(u_full_hist[idx - 2], dtype=np.float64)
    pred_u = u_prev + alpha_sec * (u_prev - u_prevprev)
    pred_lambda = float(lambda_hist[idx - 1] + alpha_sec * (lambda_hist[idx - 1] - lambda_hist[idx - 2]))
    return pred_u, pred_lambda, float(alpha_sec)


def _increment_basis_full(u_full_hist: np.ndarray, idx: int) -> np.ndarray:
    increments = [u_full_hist[j] - u_full_hist[j - 1] for j in range(2, idx)]
    if not increments:
        raise ValueError(f"No previous continuation increments are available for idx={idx}.")
    return np.column_stack(increments).astype(np.float64, copy=False)


def _orthonormalize_basis(raw_free: np.ndarray, *, tol: float = 1.0e-12) -> tuple[np.ndarray, float]:
    singular = np.linalg.svd(raw_free, compute_uv=False)
    cond = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else math.inf
    q, r = np.linalg.qr(raw_free, mode="reduced")
    diag = np.abs(np.diag(r))
    keep = diag > tol * max(float(diag.max(initial=0.0)), 1.0)
    q = q[:, keep]
    if q.size == 0:
        raise ValueError("Basis collapsed under QR rank filtering.")
    return np.asarray(q, dtype=np.float64), cond


def _field_from_flat(flat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(shape, order="F")


def _build_problem(case: dict[str, object]) -> dict[str, object]:
    probe_path = Path(__file__).with_name("probe_hypre_frozen.py")
    spec = importlib.util.spec_from_file_location("_probe_hypre_frozen_local", probe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load probe helper from {probe_path}")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    run_info = case["run_info"]
    params = run_info["params"]
    mesh_file = Path(run_info["mesh"]["mesh_file"])
    problem = probe._build_problem(
        mesh_path=mesh_file,
        elem_type=str(params["elem_type"]),
        node_ordering=str(params["node_ordering"]),
        reorder_parts=int(run_info["run_info"]["mpi_size"]) if str(params["node_ordering"]).lower() == "block_metis" else None,
        material_rows=params.get("material_rows"),
        davis_type=str(params.get("davis_type", "B")),
        constitutive_mode=str(params.get("constitutive_mode", "overlap")),
        tangent_kernel=str(params.get("tangent_kernel", "rows")),
        pc_backend="hypre",
        pmg_coarse_mesh_paths=(),
    )
    q_mask = np.asarray(problem["q_mask"], dtype=bool)
    free_idx = q_to_free_indices(q_mask)
    f_full = flatten_field(problem["f_V"])
    problem["q_mask"] = q_mask
    problem["free_idx"] = free_idx
    problem["field_shape"] = tuple(int(v) for v in np.asarray(problem["coord"]).shape)
    problem["f_full"] = f_full
    problem["f_free"] = np.asarray(f_full[free_idx], dtype=np.float64)
    problem["norm_f_free"] = max(float(np.linalg.norm(problem["f_free"])), 1.0)
    return problem


def _basis_coeff(
    u_full_flat: np.ndarray,
    *,
    ref_u_full: np.ndarray,
    basis_free: np.ndarray,
    free_idx: np.ndarray,
) -> tuple[np.ndarray, float]:
    delta_free = np.asarray(u_full_flat[free_idx] - ref_u_full[free_idx], dtype=np.float64)
    coeff = np.asarray(basis_free.T @ delta_free, dtype=np.float64)
    recon = basis_free @ coeff
    denom = max(float(np.linalg.norm(delta_free)), 1.0e-30)
    affine_resid_rel = float(np.linalg.norm(delta_free - recon) / denom)
    return coeff, affine_resid_rel


def _projected_residual_metrics(
    *,
    residual_free: np.ndarray,
    basis_free: np.ndarray,
    norm_f_free: float,
) -> tuple[np.ndarray, float, float]:
    proj_residual = np.asarray(basis_free.T @ residual_free, dtype=np.float64)
    proj_rel = float(np.linalg.norm(proj_residual) / norm_f_free)
    full_rel = float(np.linalg.norm(residual_free) / norm_f_free)
    return proj_residual, proj_rel, full_rel


def _reduced_lu_solve(
    *,
    K_r,
    basis_full: np.ndarray,
    basis_free: np.ndarray,
    rhs_free: np.ndarray,
    free_idx: np.ndarray,
    constitutive_builder,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    t0 = perf_counter()
    k_basis_cols = []
    for j in range(basis_full.shape[1]):
        col = np.asarray(matvec_to_numpy(K_r, basis_full[:, j]), dtype=np.float64)[free_idx]
        k_basis_cols.append(col)
    KB = np.column_stack(k_basis_cols)
    mat_build_wall_s = float(perf_counter() - t0)

    A_red = np.asarray(basis_free.T @ KB, dtype=np.float64)
    rhs_red = np.asarray(basis_free.T @ rhs_free, dtype=np.float64)

    t1 = perf_counter()
    lu, piv = lu_factor(A_red, check_finite=False)
    coeff = np.asarray(lu_solve((lu, piv), rhs_red, check_finite=False), dtype=np.float64)
    solve_wall_s = float(perf_counter() - t1)
    x_free = np.asarray(basis_free @ coeff, dtype=np.float64)
    stats = {
        "mat_build_wall_s": mat_build_wall_s,
        "solve_wall_s": solve_wall_s,
        "matrix_cond": float(np.linalg.cond(A_red)),
    }
    return x_free, coeff, stats


def _run_projected_newton_replay(
    *,
    problem: dict[str, object],
    case: dict[str, object],
    target_idx: int,
    max_iterations: int | None = None,
    progress_path: Path | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    params = case["run_info"]["params"]
    tol = float(params["tol"])
    eps = tol / 1000.0
    r_min = float(params["r_min"])
    it_damp_max = int(params["it_damp_max"])
    it_newt_max = int(params["it_newt_max"] if max_iterations is None else max_iterations)

    npz = case["npz"]
    step_u = np.asarray(npz["step_U"], dtype=np.float64)
    u_full_hist = np.asarray([flatten_field(step_u[i]) for i in range(step_u.shape[0])], dtype=np.float64)
    omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
    lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)

    free_idx = problem["free_idx"]
    q_mask = problem["q_mask"]
    field_shape = problem["field_shape"]
    f_full = problem["f_full"]
    f_free = problem["f_free"]
    norm_f_free = float(problem["norm_f_free"])
    const_builder = problem["const_builder"]
    K_elast = problem["K_elast"]

    ref_u_full = np.asarray(u_full_hist[target_idx - 1], dtype=np.float64)
    target_u_full = np.asarray(u_full_hist[target_idx], dtype=np.float64)
    omega_target = float(omega_hist[target_idx])
    lambda_target = float(lambda_hist[target_idx])

    raw_basis_full = _increment_basis_full(u_full_hist, target_idx)
    raw_basis_free = np.asarray(raw_basis_full[free_idx, :], dtype=np.float64)
    basis_free, raw_cond = _orthonormalize_basis(raw_basis_free)
    basis_full = np.zeros((u_full_hist.shape[1], basis_free.shape[1]), dtype=np.float64)
    basis_full[free_idx, :] = basis_free
    target_coeff, target_affine_resid_rel = _basis_coeff(
        target_u_full,
        ref_u_full=ref_u_full,
        basis_free=basis_free,
        free_idx=free_idx,
    )

    u_ini_full, lambda_ini, alpha_sec = _secant_predict(
        omega_hist=omega_hist,
        u_full_hist=u_full_hist,
        lambda_hist=lambda_hist,
        idx=target_idx,
    )
    u_ini_field = _field_from_flat(u_ini_full, field_shape)
    initial_coeff, initial_affine_resid_rel = _basis_coeff(
        u_ini_full,
        ref_u_full=ref_u_full,
        basis_free=basis_free,
        free_idx=free_idx,
    )

    U_it = np.asarray(u_ini_field, dtype=np.float64)
    lambda_it = float(lambda_ini)
    r = float(r_min)
    compute_diffs = True
    flag = 0
    rel_resid = math.nan
    total_mat_build_wall_s = 0.0
    total_lu_solve_wall_s = 0.0
    t_total = perf_counter()
    iter_rows: list[dict[str, object]] = []

    def _emit(row: dict[str, object]) -> None:
        if progress_path is not None:
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        if verbose:
            parts = [
                f"it={row.get('iteration')}",
                f"status={row.get('status')}",
            ]
            for key in ("alpha", "lambda", "projected_residual_rel", "full_residual_rel", "u_l2_rel", "lambda_abs_err"):
                if key in row:
                    value = row[key]
                    if isinstance(value, float):
                        parts.append(f"{key}={value:.6e}")
                    else:
                        parts.append(f"{key}={value}")
            print(" ".join(parts), flush=True)

    for it in range(1, it_newt_max + 1):
        iter_t0 = perf_counter()
        F = None
        K_tangent = None

        if compute_diffs:
            F, K_tangent = const_builder.build_F_K_tangent_all(lambda_it, U_it)
            F_free = flatten_field(F)[free_idx]
        else:
            F = const_builder.build_F_all(lambda_it, U_it)
            F_free = flatten_field(F)[free_idx]

        criterion = float(np.linalg.norm(F_free - f_free))
        rel_resid = criterion / norm_f_free

        if compute_diffs and rel_resid < tol and it > 1:
            iter_rows.append(
                {
                    "iteration": int(it),
                    "status": "converged",
                    "lambda": float(lambda_it),
                    "r": float(r),
                    "criterion": float(criterion),
                    "full_residual_rel": float(rel_resid),
                    "iteration_wall_s": float(perf_counter() - iter_t0),
                }
            )
            _emit(iter_rows[-1])
            _safe_destroy_petsc_mat(K_tangent, constitutive_builder=const_builder)
            break

        if K_tangent is None:
            _, K_tangent = const_builder.build_F_K_tangent_all(lambda_it, U_it)
        K_r = _combine_matrices(r, K_elast, 1.0 - r, K_tangent)

        try:
            F_eps = const_builder.build_F_all(lambda_it + eps, U_it)
            G_free = (flatten_field(F_eps)[free_idx] - F_free) / eps

            dW_free, coeff_w, stats_w = _reduced_lu_solve(
                K_r=K_r,
                basis_full=basis_full,
                basis_free=basis_free,
                rhs_free=-G_free,
                free_idx=free_idx,
                constitutive_builder=const_builder,
            )
            dV_free, coeff_v, stats_v = _reduced_lu_solve(
                K_r=K_r,
                basis_full=basis_full,
                basis_free=basis_free,
                rhs_free=f_free - F_free,
                free_idx=free_idx,
                constitutive_builder=const_builder,
            )
        finally:
            _safe_destroy_petsc_mat(K_tangent, constitutive_builder=const_builder)
            if not _is_builder_cached_matrix(K_r, const_builder):
                _safe_destroy_petsc_mat(K_r, constitutive_builder=const_builder)

        total_mat_build_wall_s += float(stats_w["mat_build_wall_s"] + stats_v["mat_build_wall_s"])
        total_lu_solve_wall_s += float(stats_w["solve_wall_s"] + stats_v["solve_wall_s"])

        W_full = np.zeros_like(ref_u_full)
        V_full = np.zeros_like(ref_u_full)
        W_full[free_idx] = dW_free
        V_full[free_idx] = dV_free
        W = _field_from_flat(W_full, field_shape)
        V = _field_from_flat(V_full, field_shape)

        fQ = f_free
        WQ = dW_free
        VQ = dV_free
        denom = float(np.dot(fQ, WQ))
        d_lambda = 0.0 if abs(denom) < 1.0e-30 else -float(np.dot(fQ, VQ)) / denom
        d_u_coeff = np.asarray(coeff_v + d_lambda * coeff_w, dtype=np.float64)
        d_U = V + d_lambda * W

        proj_residual, proj_rel, full_rel = _projected_residual_metrics(
            residual_free=np.asarray(F_free - f_free, dtype=np.float64),
            basis_free=basis_free,
            norm_f_free=norm_f_free,
        )
        merit = float(math.sqrt(proj_rel**2))
        alpha = float(
            damping_alg5(
                it_damp_max,
                U_it,
                lambda_it,
                d_U,
                d_lambda,
                problem["f_V"],
                criterion,
                q_mask,
                const_builder,
                f_free=f_free,
            )
        )

        compute_diffs = True
        if alpha < 1.0e-1:
            if alpha == 0.0:
                compute_diffs = False
                r *= 2.0
            else:
                r *= 2.0 ** 0.25
        else:
            if alpha > 0.5:
                r = max(r / math.sqrt(2.0), r_min)

        if alpha == 0.0 and r > 1.0:
            if rel_resid > 10.0 * tol:
                flag = 1
            iter_rows.append(
                {
                    "iteration": int(it),
                    "status": "stalled",
                    "lambda": float(lambda_it),
                    "r": float(r),
                    "criterion": float(criterion),
                    "projected_residual_rel": float(proj_rel),
                    "full_residual_rel": float(full_rel),
                    "d_lambda": float(d_lambda),
                    "alpha": float(alpha),
                    "coeff_before": [float(v) for v in _basis_coeff(flatten_field(U_it), ref_u_full=ref_u_full, basis_free=basis_free, free_idx=free_idx)[0]],
                    "delta_coeff": [float(v) for v in d_u_coeff],
                    "iteration_wall_s": float(perf_counter() - iter_t0),
                }
            )
            _emit(iter_rows[-1])
            break

        U_it = U_it + alpha * d_U
        denom_omega = float(np.dot(f_free, flatten_field(U_it)[free_idx]))
        if denom_omega != 0.0:
            U_it = U_it * (omega_target / denom_omega)
        lambda_it = float(lambda_it + alpha * d_lambda)

        current_u_full = flatten_field(U_it)
        current_coeff, affine_resid_rel = _basis_coeff(
            current_u_full,
            ref_u_full=ref_u_full,
            basis_free=basis_free,
            free_idx=free_idx,
        )
        u_l2_abs = float(np.linalg.norm(current_u_full - target_u_full))
        true_norm = max(float(np.linalg.norm(target_u_full)), 1.0e-30)
        inc_norm = max(float(np.linalg.norm(target_u_full - ref_u_full)), 1.0e-30)

        iter_rows.append(
            {
                "iteration": int(it),
                "status": "iterate",
                "lambda": float(lambda_it),
                "r": float(r),
                "criterion": float(criterion),
                "projected_residual_rel": float(proj_rel),
                "full_residual_rel": float(full_rel),
                "merit": merit,
                "d_lambda": float(d_lambda),
                "accepted_delta_lambda": float(alpha * d_lambda),
                "alpha": float(alpha),
                "coeff_w": [float(v) for v in coeff_w],
                "coeff_v": [float(v) for v in coeff_v],
                "delta_coeff": [float(v) for v in d_u_coeff],
                "coeff_after": [float(v) for v in current_coeff],
                "target_coeff_diff": [float(v) for v in (current_coeff - target_coeff)],
                "affine_projection_residual_rel": float(affine_resid_rel),
                "u_l2_rel": float(u_l2_abs / true_norm),
                "u_increment_rel": float(u_l2_abs / inc_norm),
                "lambda_abs_err": abs(float(lambda_it) - lambda_target),
                "lu_matrix_cond_w": float(stats_w["matrix_cond"]),
                "lu_matrix_cond_v": float(stats_v["matrix_cond"]),
                "iteration_wall_s": float(perf_counter() - iter_t0),
            }
        )
        _emit(iter_rows[-1])

        if np.isnan(rel_resid) or it == it_newt_max:
            if rel_resid > 10.0 * tol:
                flag = 1
            break

    total_wall_s = float(perf_counter() - t_total)
    final_u_full = flatten_field(U_it)
    final_coeff, final_affine_resid_rel = _basis_coeff(
        final_u_full,
        ref_u_full=ref_u_full,
        basis_free=basis_free,
        free_idx=free_idx,
    )
    u_l2_abs = float(np.linalg.norm(final_u_full - target_u_full))
    true_norm = max(float(np.linalg.norm(target_u_full)), 1.0e-30)
    inc_norm = max(float(np.linalg.norm(target_u_full - ref_u_full)), 1.0e-30)
    return {
        "accepted_step": int(target_idx + 1),
        "continuation_ordinal": int(target_idx - 1),
        "omega_target": float(omega_target),
        "lambda_target": float(lambda_target),
        "basis_dim": int(basis_free.shape[1]),
        "raw_basis_cond": float(raw_cond),
        "secant_alpha": float(alpha_sec),
        "lambda_ini_secant": float(lambda_ini),
        "initial_coeff": [float(v) for v in initial_coeff],
        "initial_affine_projection_residual_rel": float(initial_affine_resid_rel),
        "target_coeff": [float(v) for v in target_coeff],
        "target_affine_projection_residual_rel": float(target_affine_resid_rel),
        "projected_newton_iterations": int(sum(1 for row in iter_rows if row.get("status") == "iterate")),
        "flag": int(flag),
        "lambda_pred": float(lambda_it),
        "lambda_abs_err": abs(float(lambda_it) - lambda_target),
        "u_l2_abs": float(u_l2_abs),
        "u_l2_rel": float(u_l2_abs / true_norm),
        "u_increment_rel": float(u_l2_abs / inc_norm),
        "final_coeff": [float(v) for v in final_coeff],
        "final_affine_projection_residual_rel": float(final_affine_resid_rel),
        "total_mat_build_wall_s": float(total_mat_build_wall_s),
        "total_lu_solve_wall_s": float(total_lu_solve_wall_s),
        "total_wall_s": float(total_wall_s),
        "iter_rows": iter_rows,
    }


def _plot_curves(out_dir: Path, replay: dict[str, object]) -> dict[str, Path]:
    rows = [row for row in replay["iter_rows"] if row.get("status") == "iterate"]
    paths: dict[str, Path] = {}
    if not rows:
        return paths
    xs = [int(row["iteration"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(xs, [float(row["projected_residual_rel"]) for row in rows], marker="o", label="Projected residual")
    ax.plot(xs, [float(row["full_residual_rel"]) for row in rows], marker="s", label="Full residual")
    ax.set_yscale("log")
    ax.set_xlabel("Projected Newton iteration")
    ax.set_ylabel("Relative residual")
    ax.set_title("Projected vs Full Residual")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    paths["residuals"] = out_dir / "projected_residuals.png"
    fig.savefig(paths["residuals"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(xs, [float(row["lambda"]) for row in rows], marker="o", label="Projected Newton lambda")
    ax.axhline(float(replay["lambda_target"]), color="k", linestyle="--", linewidth=1.2, label="Target lambda")
    ax.set_xlabel("Projected Newton iteration")
    ax.set_ylabel("lambda")
    ax.set_title("Lambda Convergence")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    paths["lambda"] = out_dir / "lambda_convergence.png"
    fig.savefig(paths["lambda"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    basis_dim = len(rows[0]["coeff_after"])
    for j in range(basis_dim):
        ax.plot(xs, [float(row["coeff_after"][j]) for row in rows], marker="o", linewidth=1.4, label=f"c{j+1}")
    for j in range(basis_dim):
        ax.axhline(float(replay["target_coeff"][j]), linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Projected Newton iteration")
    ax.set_ylabel("Basis coefficient")
    ax.set_title("Reduced Coefficients")
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=2)
    fig.tight_layout()
    paths["coefficients"] = out_dir / "coefficients.png"
    fig.savefig(paths["coefficients"])
    plt.close(fig)
    return paths


def _write_report(
    *,
    out_dir: Path,
    case: dict[str, object],
    replay: dict[str, object],
    plot_paths: dict[str, Path],
) -> Path:
    report_path = out_dir / "README.md"
    npz = case["npz"]
    stats_step_index = np.asarray(npz["stats_step_index"], dtype=np.int64)
    mask = stats_step_index == int(replay["accepted_step"])
    if np.any(mask):
        idx = int(np.nonzero(mask)[0][0])
        full_newton_iterations = int(np.asarray(npz["stats_step_newton_iterations"], dtype=np.int64)[idx])
        full_linear_iterations = int(np.asarray(npz["stats_step_linear_iterations"], dtype=np.int64)[idx])
        full_step_wall_s = float(np.asarray(npz["stats_step_wall_time"], dtype=np.float64)[idx])
    else:
        full_newton_iterations = -1
        full_linear_iterations = -1
        full_step_wall_s = math.nan

    rows = [row for row in replay["iter_rows"] if row.get("status") == "iterate"]
    lines = [
        "# Projected Newton Replay on Saved `P4(L1)` Step",
        "",
        "This replay uses the saved rank-8 `P4(L1)` branch from "
        f"[p4_rank8_step100]({_relpath(report_path, case['case_dir'] / 'data' / 'run_info.json')}). "
        f"This report targets overall accepted step `{replay['accepted_step']}` from that saved branch.",
        "",
        "The replay follows `newton_ind_ssr` exactly in the nonlinear logic:",
        "- same secant predictor for `U_ini` and `lambda_ini`",
        "- same finite-difference `G = dF/dlambda`",
        "- same regularized tangent update `K_r = r K_elast + (1-r) K_tangent`",
        "- same `d_lambda = -(f^T V)/(f^T W)` formula",
        "- same `damping_alg5` and same post-update `omega` rescaling",
        "- only the two sparse solves are replaced by dense LU solves on the projected system in the basis of all previous continuation increments",
        "",
        "## Summary",
        "",
        f"- Accepted step replayed: `{replay['accepted_step']}`",
        f"- Continuation ordinal after init: `{replay['continuation_ordinal']}`",
        f"- Basis dimension: `{replay['basis_dim']}`",
        f"- Raw basis condition number: `{replay['raw_basis_cond']:.3e}`",
        f"- Secant predictor alpha: `{replay['secant_alpha']:.12g}`",
        f"- Secant predictor lambda: `{replay['lambda_ini_secant']:.12g}`",
        f"- Projected Newton iterations: `{replay['projected_newton_iterations']}`",
        f"- Final projected lambda: `{replay['lambda_pred']:.12g}`",
        f"- Target lambda from saved branch: `{replay['lambda_target']:.12g}`",
        f"- Final lambda abs. error: `{replay['lambda_abs_err']:.6e}`",
        f"- Final relative displacement error to saved branch: `{replay['u_l2_rel']:.6e}`",
        f"- Final increment-relative displacement error: `{replay['u_increment_rel']:.6e}`",
        f"- Total reduced matrix-build wall time: `{replay['total_mat_build_wall_s']:.3f} s`",
        f"- Total dense LU solve wall time: `{replay['total_lu_solve_wall_s']:.3f} s`",
        f"- Total replay wall time: `{replay['total_wall_s']:.3f} s`",
        "",
        "Reference from the original saved full step:",
        f"- Newton iterations: `{full_newton_iterations}`",
        f"- Linear iterations: `{full_linear_iterations}`",
        f"- Step wall time: `{full_step_wall_s:.3f} s`",
        "",
        "## Coefficients",
        "",
        f"- Initial secant coefficients in the reduced basis: `{np.array(replay['initial_coeff']).round(6).tolist()}`",
        f"- Final projected-Newton coefficients: `{np.array(replay['final_coeff']).round(6).tolist()}`",
        f"- Target coefficients from the saved converged step projected into the same basis: `{np.array(replay['target_coeff']).round(6).tolist()}`",
        f"- Initial affine projection residual: `{replay['initial_affine_projection_residual_rel']:.6e}`",
        f"- Final affine projection residual: `{replay['final_affine_projection_residual_rel']:.6e}`",
        f"- Target affine projection residual: `{replay['target_affine_projection_residual_rel']:.6e}`",
        "",
        "## Iteration Table",
        "",
        "| Iter | alpha | lambda | proj res | full res | d_lambda | coeff_after | u_l2_rel | lambda abs err |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        coeff_text = "[" + ", ".join(f"{float(v):.4f}" for v in row["coeff_after"]) + "]"
        lines.append(
            f"| {int(row['iteration'])} | {float(row['alpha']):.4f} | {float(row['lambda']):.9f} | "
            f"{float(row['projected_residual_rel']):.3e} | {float(row['full_residual_rel']):.3e} | "
            f"{float(row['d_lambda']):.3e} | `{coeff_text}` | {float(row['u_l2_rel']):.3e} | "
            f"{float(row['lambda_abs_err']):.3e} |"
        )
    if plot_paths:
        lines.extend(
            [
                "",
                "## Plots",
                "",
                f"![Projected Residuals]({_relpath(report_path, plot_paths['residuals'])})",
                "",
                f"![Lambda Convergence]({_relpath(report_path, plot_paths['lambda'])})",
                "",
                f"![Reduced Coefficients]({_relpath(report_path, plot_paths['coefficients'])})",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a saved P4(L1) continuation step with exact projected Newton and dense LU.")
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--accepted-step",
        type=int,
        default=7,
        help="Overall accepted step index from the saved branch. Default 7 = fifth continuation advance after the two init states.",
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = _ensure_dir(Path(args.out_dir))
    progress_path = out_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    case = _load_case(Path(args.case_dir))
    problem = _build_problem(case)
    target_idx = int(args.accepted_step) - 1
    replay = _run_projected_newton_replay(
        problem=problem,
        case=case,
        target_idx=target_idx,
        max_iterations=args.max_iterations,
        progress_path=progress_path,
        verbose=bool(args.verbose),
    )
    plot_paths = _plot_curves(out_dir, replay)
    report_path = _write_report(out_dir=out_dir, case=case, replay=replay, plot_paths=plot_paths)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(replay, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
