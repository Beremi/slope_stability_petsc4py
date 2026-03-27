#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from slope_stability.nonlinear.newton import _is_builder_cached_matrix
from slope_stability.utils import flatten_field, full_field_from_free_values, matvec_to_numpy, q_to_free_indices


DEFAULT_CASE_DIR = Path("artifacts/p4_pmg_shell_best_rank8_full/p4_rank8_step100")
DEFAULT_OUT_DIR = Path("artifacts/p4_saved_reduced_newton_analysis")


@dataclass(frozen=True)
class ReducedMethodSpec:
    name: str
    label: str
    min_state_index: int


REDUCED_METHODS = (
    ReducedMethodSpec("reduced_newton_span2", "Projected Newton Span-2", 3),
    ReducedMethodSpec("reduced_newton_span3", "Projected Newton Span-3", 4),
    ReducedMethodSpec("reduced_newton_all_prev_cont", "Projected Newton All Previous Continuation Increments", 3),
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relpath(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, start=from_path.parent)


def _load_case(case_dir: Path) -> dict[str, object]:
    data_dir = case_dir / "data"
    npz = np.load(data_dir / "petsc_run.npz", allow_pickle=True)
    run_info = json.loads((data_dir / "run_info.json").read_text(encoding="utf-8"))
    return {
        "case_dir": case_dir,
        "npz": npz,
        "run_info": run_info,
    }


def _lagrange_coeffs(xs: np.ndarray, xt: float) -> np.ndarray:
    coeff = np.ones(xs.size, dtype=float)
    for j in range(xs.size):
        for k in range(xs.size):
            if j != k:
                coeff[j] *= (xt - xs[k]) / (xs[j] - xs[k])
    return coeff


def _secant_predict(
    *,
    omega_hist: np.ndarray,
    U_flat: np.ndarray,
    lambda_hist: np.ndarray,
    idx: int,
) -> tuple[np.ndarray, float]:
    o0, o1 = float(omega_hist[idx - 2]), float(omega_hist[idx - 1])
    scale = (float(omega_hist[idx]) - o1) / (o1 - o0)
    u0 = np.asarray(U_flat[idx - 2], dtype=np.float64)
    u1 = np.asarray(U_flat[idx - 1], dtype=np.float64)
    pred_u = u1 + scale * (u1 - u0)
    pred_lambda = float(lambda_hist[idx - 1] + scale * (lambda_hist[idx - 1] - lambda_hist[idx - 2]))
    return pred_u, pred_lambda


def _quadratic_predict(
    *,
    omega_hist: np.ndarray,
    U_flat: np.ndarray,
    lambda_hist: np.ndarray,
    idx: int,
) -> tuple[np.ndarray, float]:
    xs = np.asarray(omega_hist[idx - 3 : idx], dtype=np.float64)
    coeff = _lagrange_coeffs(xs, float(omega_hist[idx]))
    pred_u = coeff @ np.asarray(U_flat[idx - 3 : idx], dtype=np.float64)
    pred_lambda = float(np.dot(coeff, np.asarray(lambda_hist[idx - 3 : idx], dtype=np.float64)))
    return pred_u, pred_lambda


def _blend03_predict(
    *,
    omega_hist: np.ndarray,
    U_flat: np.ndarray,
    lambda_hist: np.ndarray,
    idx: int,
) -> tuple[np.ndarray, float]:
    sec_u, sec_lambda = _secant_predict(omega_hist=omega_hist, U_flat=U_flat, lambda_hist=lambda_hist, idx=idx)
    quad_u, quad_lambda = _quadratic_predict(omega_hist=omega_hist, U_flat=U_flat, lambda_hist=lambda_hist, idx=idx)
    return 0.7 * sec_u + 0.3 * quad_u, float(0.7 * sec_lambda + 0.3 * quad_lambda)


def _increment_basis_full(U_flat: np.ndarray, idx: int, method_name: str) -> np.ndarray:
    if method_name == "reduced_newton_span2":
        increments = [U_flat[idx - j] - U_flat[idx - j - 1] for j in range(1, 3)]
    elif method_name == "reduced_newton_span3":
        increments = [U_flat[idx - j] - U_flat[idx - j - 1] for j in range(1, 4)]
    elif method_name == "reduced_newton_all_prev_cont":
        increments = [U_flat[j] - U_flat[j - 1] for j in range(2, idx)]
    else:
        raise ValueError(f"Unsupported reduced method {method_name!r}")
    if not increments:
        raise ValueError(f"No increments available for {method_name!r} at idx={idx}.")
    return np.column_stack(increments).astype(np.float64, copy=False)


def _orthonormalize_basis(raw_free: np.ndarray, *, tol: float = 1.0e-12) -> tuple[np.ndarray, float]:
    if raw_free.ndim != 2 or raw_free.shape[1] == 0:
        raise ValueError("Expected a non-empty 2D basis array.")
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


def _safe_destroy_petsc_mat(A, *, constitutive_builder=None) -> None:
    if A is None:
        return
    if _is_builder_cached_matrix(A, constitutive_builder):
        return
    release = getattr(importlib.import_module("slope_stability.utils"), "release_petsc_aij_matrix", None)
    if callable(release):
        try:
            release(A)
        except Exception:
            pass
    destroy = getattr(A, "destroy", None)
    if callable(destroy):
        try:
            destroy()
        except Exception:
            pass


def _build_problem_from_case(case: dict[str, object]) -> dict[str, object]:
    probe = importlib.import_module("benchmarks.3d_hetero_ssr_default.archive.probe_hypre_frozen")
    build_problem = probe._build_problem
    run_info = case["run_info"]
    params = run_info["params"]
    mesh_file = Path(run_info["mesh"]["mesh_file"])
    t0 = perf_counter()
    problem = build_problem(
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
    problem["problem_build_wall_s"] = float(perf_counter() - t0)
    q_mask = np.asarray(problem["q_mask"], dtype=bool)
    free_idx = q_to_free_indices(q_mask)
    f_full = flatten_field(problem["f_V"])
    problem["free_idx"] = free_idx
    problem["f_full"] = f_full
    problem["f_free"] = np.asarray(f_full[free_idx], dtype=np.float64)
    problem["norm_f_free"] = max(float(np.linalg.norm(problem["f_free"])), 1.0)
    problem["field_shape"] = tuple(int(v) for v in np.asarray(problem["coord"]).shape)
    return problem


def _evaluate_projected_system(
    *,
    problem: dict[str, object],
    basis_free: np.ndarray,
    basis_full: np.ndarray,
    u_full_flat: np.ndarray,
    lambda_value: float,
    omega_target: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    const_builder = problem["const_builder"]
    free_idx = problem["free_idx"]
    f_free = problem["f_free"]
    norm_f = float(problem["norm_f_free"])
    field_shape = problem["field_shape"]

    reduction_t0 = perf_counter()
    const_builder.reduction(float(lambda_value))
    reduction_wall_s = float(perf_counter() - reduction_t0)

    build_t0 = perf_counter()
    F_full, K_full = const_builder.build_F_K_tangent_reduced(_field_from_flat(u_full_flat, field_shape))
    build_wall_s = float(perf_counter() - build_t0)

    f_int_full = flatten_field(np.asarray(F_full, dtype=np.float64))
    residual_free = float(lambda_value) * f_free - np.asarray(f_int_full[free_idx], dtype=np.float64)
    omega_value = float(np.dot(f_free, u_full_flat[free_idx]))
    omega_err = float(omega_value - omega_target)

    proj_residual = np.asarray(basis_free.T @ residual_free, dtype=np.float64)
    projected_rel = float(np.linalg.norm(proj_residual) / norm_f)
    full_rel = float(np.linalg.norm(residual_free) / norm_f)
    omega_rel = abs(omega_err) / max(abs(float(omega_target)), 1.0)

    kv_cols = []
    for j in range(basis_full.shape[1]):
        kv_cols.append(np.asarray(matvec_to_numpy(K_full, basis_full[:, j]), dtype=np.float64)[free_idx])
    k_basis_free = np.column_stack(kv_cols) if kv_cols else np.zeros((basis_free.shape[0], 0), dtype=np.float64)
    j11 = -(basis_free.T @ k_basis_free)
    j12 = basis_free.T @ f_free
    j21 = f_free @ basis_free
    jacobian = np.zeros((basis_free.shape[1] + 1, basis_free.shape[1] + 1), dtype=np.float64)
    jacobian[:-1, :-1] = j11
    jacobian[:-1, -1] = j12
    jacobian[-1, :-1] = j21

    _safe_destroy_petsc_mat(K_full, constitutive_builder=const_builder)
    return proj_residual, jacobian, omega_err, projected_rel, full_rel, omega_rel, reduction_wall_s + build_wall_s


def _reduced_newton_solve(
    *,
    problem: dict[str, object],
    method_name: str,
    target_idx: int,
    omega_hist: np.ndarray,
    lambda_hist: np.ndarray,
    u_free_hist: np.ndarray,
    u_full_hist: np.ndarray,
    max_iterations: int,
    tol_projected: float,
    tol_omega: float,
) -> dict[str, object]:
    raw_basis_full = _increment_basis_full(u_full_hist, target_idx, method_name)
    free_idx = problem["free_idx"]
    raw_basis_free = np.asarray(raw_basis_full[free_idx, :], dtype=np.float64)
    basis_free, raw_cond = _orthonormalize_basis(raw_basis_free)
    basis_full = np.zeros((u_full_hist.shape[1], basis_free.shape[1]), dtype=np.float64)
    basis_full[free_idx, :] = basis_free

    ref_u_full = np.asarray(u_full_hist[target_idx - 1], dtype=np.float64)
    ref_u_free = np.asarray(u_free_hist[target_idx - 1], dtype=np.float64)
    sec_u_full, sec_lambda = _secant_predict(omega_hist=omega_hist, U_flat=u_full_hist, lambda_hist=lambda_hist, idx=target_idx)
    delta_sec_free = np.asarray(sec_u_full[free_idx] - ref_u_free, dtype=np.float64)
    coeff = np.asarray(basis_free.T @ delta_sec_free, dtype=np.float64)
    lambda_value = float(sec_lambda)
    omega_target = float(omega_hist[target_idx])
    target_u_full = np.asarray(u_full_hist[target_idx], dtype=np.float64)
    target_lambda = float(lambda_hist[target_idx])

    iter_rows: list[dict[str, object]] = []
    total_eval_wall = 0.0
    line_search_rejects = 0
    t_total = perf_counter()

    def _state_from_coeff(current_coeff: np.ndarray) -> np.ndarray:
        delta_free = basis_free @ current_coeff
        return ref_u_full + full_field_from_free_values(delta_free, free_idx, problem["field_shape"]).reshape(-1, order="F")

    current_u_full = _state_from_coeff(coeff)
    proj_residual, jacobian, omega_err, proj_rel, full_rel, omega_rel, eval_wall = _evaluate_projected_system(
        problem=problem,
        basis_free=basis_free,
        basis_full=basis_full,
        u_full_flat=current_u_full,
        lambda_value=lambda_value,
        omega_target=omega_target,
    )
    total_eval_wall += eval_wall
    merit = math.sqrt(proj_rel**2 + omega_rel**2)

    converged = proj_rel <= tol_projected and omega_rel <= tol_omega
    iteration = 0
    while not converged and iteration < max_iterations:
        iteration += 1
        rhs = -np.concatenate([proj_residual, np.array([omega_err], dtype=np.float64)])
        try:
            delta = np.linalg.solve(jacobian, rhs)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(jacobian, rhs, rcond=None)
        delta_coeff = np.asarray(delta[:-1], dtype=np.float64)
        delta_lambda = float(delta[-1])

        accepted_alpha = 0.0
        accepted_state = None
        accepted_payload = None
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial_coeff = coeff + alpha * delta_coeff
            trial_lambda = float(lambda_value + alpha * delta_lambda)
            trial_u_full = _state_from_coeff(trial_coeff)
            trial_proj_residual, trial_jacobian, trial_omega_err, trial_proj_rel, trial_full_rel, trial_omega_rel, trial_eval_wall = _evaluate_projected_system(
                problem=problem,
                basis_free=basis_free,
                basis_full=basis_full,
                u_full_flat=trial_u_full,
                lambda_value=trial_lambda,
                omega_target=omega_target,
            )
            total_eval_wall += trial_eval_wall
            trial_merit = math.sqrt(trial_proj_rel**2 + trial_omega_rel**2)
            if trial_merit <= merit or alpha <= 0.0625:
                accepted_alpha = float(alpha)
                accepted_state = (trial_coeff, trial_lambda, trial_u_full)
                accepted_payload = (
                    trial_proj_residual,
                    trial_jacobian,
                    trial_omega_err,
                    trial_proj_rel,
                    trial_full_rel,
                    trial_omega_rel,
                    trial_merit,
                )
                break
            line_search_rejects += 1

        assert accepted_state is not None and accepted_payload is not None
        coeff, lambda_value, current_u_full = accepted_state
        (
            proj_residual,
            jacobian,
            omega_err,
            proj_rel,
            full_rel,
            omega_rel,
            merit,
        ) = accepted_payload
        converged = proj_rel <= tol_projected and omega_rel <= tol_omega
        u_l2_abs = float(np.linalg.norm(current_u_full - target_u_full))
        u_l2_rel = u_l2_abs / max(float(np.linalg.norm(target_u_full)), 1.0e-30)
        inc_norm = max(float(np.linalg.norm(target_u_full - ref_u_full)), 1.0e-30)
        iter_rows.append(
            {
                "iteration": iteration,
                "alpha": accepted_alpha,
                "projected_residual_rel": proj_rel,
                "full_residual_rel": full_rel,
                "omega_rel": omega_rel,
                "u_l2_rel": u_l2_rel,
                "u_increment_rel": u_l2_abs / inc_norm,
                "lambda_abs_err": abs(float(lambda_value) - target_lambda),
            }
        )

    total_wall = float(perf_counter() - t_total)
    u_l2_abs = float(np.linalg.norm(current_u_full - target_u_full))
    true_norm = max(float(np.linalg.norm(target_u_full)), 1.0e-30)
    inc_norm = max(float(np.linalg.norm(target_u_full - ref_u_full)), 1.0e-30)
    diff_field = (current_u_full - target_u_full).reshape(problem["field_shape"], order="F")
    return {
        "accepted_step": int(target_idx + 1),
        "state_idx": int(target_idx),
        "omega": float(omega_target),
        "lambda": target_lambda,
        "basis_dim": int(basis_free.shape[1]),
        "raw_basis_cond": float(raw_cond),
        "reduced_newton_iterations": int(iteration),
        "converged_projected": bool(converged),
        "projected_residual_rel": float(proj_rel),
        "full_residual_rel": float(full_rel),
        "omega_rel": float(omega_rel),
        "line_search_rejects": int(line_search_rejects),
        "eval_wall_s_total": float(total_eval_wall),
        "wall_s_total": float(total_wall),
        "u_l2_abs": float(u_l2_abs),
        "u_l2_rel": float(u_l2_abs / true_norm),
        "u_increment_rel": float(u_l2_abs / inc_norm),
        "u_max_node_abs": float(np.linalg.norm(diff_field, axis=0).max()),
        "lambda_pred": float(lambda_value),
        "lambda_abs_err": abs(float(lambda_value) - target_lambda),
        "iter_rows": iter_rows,
    }


def _kinematic_rows(
    *,
    omega_hist: np.ndarray,
    lambda_hist: np.ndarray,
    u_full_hist: np.ndarray,
) -> dict[str, list[dict[str, object]]]:
    by_method = {"secant": [], "blend03": []}
    for idx in range(2, omega_hist.size):
        ref = u_full_hist[idx - 1]
        target = u_full_hist[idx]
        true_norm = max(float(np.linalg.norm(target)), 1.0e-30)
        inc_norm = max(float(np.linalg.norm(target - ref)), 1.0e-30)
        for name, fn, min_idx in (
            ("secant", _secant_predict, 2),
            ("blend03", _blend03_predict, 3),
        ):
            if idx < min_idx:
                continue
            pred_u, pred_lambda = fn(omega_hist=omega_hist, U_flat=u_full_hist, lambda_hist=lambda_hist, idx=idx)
            diff = pred_u - target
            diff_field = diff.reshape((3, -1), order="F")
            l2_abs = float(np.linalg.norm(diff))
            by_method[name].append(
                {
                    "accepted_step": int(idx + 1),
                    "state_idx": int(idx),
                    "omega": float(omega_hist[idx]),
                    "lambda": float(lambda_hist[idx]),
                    "u_l2_abs": l2_abs,
                    "u_l2_rel": float(l2_abs / true_norm),
                    "u_increment_rel": float(l2_abs / inc_norm),
                    "u_max_node_abs": float(np.linalg.norm(diff_field, axis=0).max()),
                    "lambda_pred": float(pred_lambda),
                    "lambda_abs_err": abs(float(pred_lambda) - float(lambda_hist[idx])),
                }
            )
    return by_method


def _method_summary(rows: list[dict[str, object]], *, include_newton: bool) -> dict[str, object]:
    arr = lambda key: np.asarray([row[key] for row in rows], dtype=np.float64) if rows else np.asarray([], dtype=np.float64)
    out = {
        "targets": int(len(rows)),
        "mean_u_l2_rel": float(np.mean(arr("u_l2_rel"))) if rows else np.nan,
        "max_u_l2_rel": float(np.max(arr("u_l2_rel"))) if rows else np.nan,
        "mean_u_increment_rel": float(np.mean(arr("u_increment_rel"))) if rows else np.nan,
        "max_u_increment_rel": float(np.max(arr("u_increment_rel"))) if rows else np.nan,
        "mean_lambda_abs_err": float(np.mean(arr("lambda_abs_err"))) if rows else np.nan,
        "max_lambda_abs_err": float(np.max(arr("lambda_abs_err"))) if rows else np.nan,
    }
    if include_newton:
        out.update(
            {
                "mean_projected_residual_rel": float(np.mean(arr("projected_residual_rel"))) if rows else np.nan,
                "max_projected_residual_rel": float(np.max(arr("projected_residual_rel"))) if rows else np.nan,
                "mean_full_residual_rel": float(np.mean(arr("full_residual_rel"))) if rows else np.nan,
                "max_full_residual_rel": float(np.max(arr("full_residual_rel"))) if rows else np.nan,
                "mean_reduced_newton_iterations": float(np.mean(arr("reduced_newton_iterations"))) if rows else np.nan,
                "max_reduced_newton_iterations": float(np.max(arr("reduced_newton_iterations"))) if rows else np.nan,
                "mean_basis_dim": float(np.mean(arr("basis_dim"))) if rows else np.nan,
                "max_basis_dim": float(np.max(arr("basis_dim"))) if rows else np.nan,
                "mean_raw_basis_cond": float(np.mean(arr("raw_basis_cond"))) if rows else np.nan,
                "max_raw_basis_cond": float(np.max(arr("raw_basis_cond"))) if rows else np.nan,
                "mean_eval_wall_s_total": float(np.mean(arr("eval_wall_s_total"))) if rows else np.nan,
                "max_eval_wall_s_total": float(np.max(arr("eval_wall_s_total"))) if rows else np.nan,
                "mean_wall_s_total": float(np.mean(arr("wall_s_total"))) if rows else np.nan,
                "max_wall_s_total": float(np.max(arr("wall_s_total"))) if rows else np.nan,
                "total_wall_s_total": float(np.sum(arr("wall_s_total"))) if rows else np.nan,
                "total_eval_wall_s_total": float(np.sum(arr("eval_wall_s_total"))) if rows else np.nan,
                "total_line_search_rejects": int(np.sum(arr("line_search_rejects"))) if rows else 0,
                "projected_converged_count": int(sum(bool(row["converged_projected"]) for row in rows)),
            }
        )
    return out


def _plot_metric(
    *,
    path: Path,
    title: str,
    ylabel: str,
    rows_by_method: dict[str, list[dict[str, object]]],
    labels: dict[str, str],
    key: str,
    logy: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for name, rows in rows_by_method.items():
        if not rows:
            continue
        xs = [int(row["accepted_step"]) for row in rows]
        ys = [float(row[key]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.6, label=labels[name])
    ax.set_xlabel("Accepted continuation step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_report(
    *,
    out_dir: Path,
    case: dict[str, object],
    problem: dict[str, object],
    kinematic_rows: dict[str, list[dict[str, object]]],
    reduced_rows: dict[str, list[dict[str, object]]],
    summary: dict[str, object],
    plot_paths: dict[str, Path],
) -> Path:
    report_path = out_dir / "README.md"
    run_info = case["run_info"]
    params = run_info["params"]

    labels = {
        "secant": "Current Secant",
        "blend03": "Blend 0.7 Secant + 0.3 Quadratic",
        "reduced_newton_span2": "Projected Newton Span-2",
        "reduced_newton_span3": "Projected Newton Span-3",
        "reduced_newton_all_prev_cont": "Projected Newton All Previous Continuation Increments",
    }

    lines = [
        "# Saved P4 Reduced-Newton Analysis",
        "",
        "This report replays a finished `P4(L1)` continuation branch offline and solves the next accepted continuation point in a reduced displacement space using the real nonlinear residual and tangent.",
        "",
        "Important scope note:",
        "- This is a real projected Newton on the constitutive residual, not an oracle least-squares fit.",
        "- It is still an offline replay on the saved branch, on a single process, with the true target `omega` from the accepted run.",
        "- The unknowns are the reduced-space displacement coefficients plus `lambda`.",
        "- Because the solve is restricted to a small basis, the projected equations can converge while the full residual remains nonzero.",
        "",
        "## Source Run",
        "",
        f"- Saved run: `{case['case_dir']}`",
        f"- Mesh: `{run_info['mesh']['mesh_file']}`",
        f"- Fine space: `{params['elem_type']}`",
        f"- Saved accepted states: `{int(case['npz']['step_U'].shape[0])}`",
        f"- Final saved omega: `{float(case['npz']['omega_hist'][-1]):.6e}`",
        f"- Final saved lambda: `{float(case['npz']['lambda_hist'][-1]):.9f}`",
        "",
        "## Replay Cost",
        "",
        f"- One-time offline problem build wall time: `{float(problem['problem_build_wall_s']):.3f} s`",
        "- This build cost is paid once, then all reduced-step solves reuse the same constitutive operator object.",
        "",
        "## Methods",
        "",
        "- `Current Secant`: existing kinematic predictor from the saved branch.",
        "- `Blend 0.7 Secant + 0.3 Quadratic`: best cheap kinematic variant from the earlier retrospective study.",
        "- `Projected Newton Span-2`: full projected Newton in the span of the last 2 accepted displacement increments.",
        "- `Projected Newton Span-3`: full projected Newton in the span of the last 3 accepted displacement increments.",
        "- `Projected Newton All Previous Continuation Increments`: full projected Newton in the span of all previous continuation increments only.",
        "",
        "## Summary Table",
        "",
        "| Method | Targets | Mean rel U error | Max rel U error | Mean increment-rel error | Mean abs lambda error | Mean full residual rel | Mean projected Newton its | Mean wall/target [s] | Total wall [s] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    ordered_methods = ["secant", "blend03", *[spec.name for spec in REDUCED_METHODS if spec.name in reduced_rows]]
    for name in ordered_methods:
        row = summary[name]
        if name in {"secant", "blend03"}:
            lines.append(
                "| {label} | {targets:d} | {mean_u_l2_rel:.6e} | {max_u_l2_rel:.6e} | {mean_u_increment_rel:.6e} | {mean_lambda_abs_err:.6e} | - | - | - | - |".format(
                    label=labels[name],
                    **row,
                )
            )
        else:
            lines.append(
                "| {label} | {targets:d} | {mean_u_l2_rel:.6e} | {max_u_l2_rel:.6e} | {mean_u_increment_rel:.6e} | {mean_lambda_abs_err:.6e} | {mean_full_residual_rel:.6e} | {mean_reduced_newton_iterations:.3f} | {mean_wall_s_total:.3f} | {total_wall_s_total:.3f} |".format(
                    label=labels[name],
                    **row,
                )
            )

    lines.extend(
        [
            "",
            "## Hard Tail Summary",
            "",
            "Hard tail here means accepted continuation steps `8-13` from the saved branch, where the original continuation became expensive.",
            "",
            "| Method | Mean rel U error | Mean increment-rel error | Mean abs lambda error | Mean full residual rel | Mean projected Newton its | Mean wall/target [s] |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ordered_methods:
        rows = (
            kinematic_rows[name] if name in {"secant", "blend03"} else reduced_rows[name]
        )
        rows = [row for row in rows if int(row["accepted_step"]) >= 8]
        row = _method_summary(rows, include_newton=name not in {"secant", "blend03"})
        if name in {"secant", "blend03"}:
            lines.append(
                "| {label} | {mean_u_l2_rel:.6e} | {mean_u_increment_rel:.6e} | {mean_lambda_abs_err:.6e} | - | - | - |".format(
                    label=labels[name],
                    **row,
                )
            )
        else:
            lines.append(
                "| {label} | {mean_u_l2_rel:.6e} | {mean_u_increment_rel:.6e} | {mean_lambda_abs_err:.6e} | {mean_full_residual_rel:.6e} | {mean_reduced_newton_iterations:.3f} | {mean_wall_s_total:.3f} |".format(
                    label=labels[name],
                    **row,
                )
            )

    lines.extend(
        [
            "",
            "## Per-Step Reduced-Newton Detail",
            "",
            "| Step | Omega | Lambda | Method | Basis dim | Raw basis cond | Red. Newton its | Full residual rel | Rel U error | Lambda abs err | Wall [s] | Eval wall [s] | Line-search rejects |",
            "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in [spec.name for spec in REDUCED_METHODS if spec.name in reduced_rows]:
        for row in reduced_rows[name]:
            lines.append(
                "| {accepted_step:d} | {omega:.6e} | {lambda_value:.9f} | {label} | {basis_dim:d} | {raw_basis_cond:.6e} | {reduced_newton_iterations:d} | {full_residual_rel:.6e} | {u_l2_rel:.6e} | {lambda_abs_err:.6e} | {wall_s_total:.3f} | {eval_wall_s_total:.3f} | {line_search_rejects:d} |".format(
                    lambda_value=float(row["lambda"]),
                    label=labels[name],
                    **row,
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If the reduced Newton methods materially lower `rel U error` versus secant/blend while keeping `full residual rel` small, then the previous-step increment space is not just expressive; it is also dynamically useful under the true nonlinear residual.",
            "- The gap between `projected residual rel` and `full residual rel` shows how much of the remaining error is outside the chosen reduced space.",
            "- `Eval wall [s]` is the main cost signal for a future online version, because it is dominated by constitutive reduction and full residual/tangent builds rather than the tiny reduced linear solve itself.",
            "",
            "## Plots",
            "",
            f"![Relative U Error]({_relpath(report_path, plot_paths['u_rel'])})",
            "",
            f"![Full Residual Relative]({_relpath(report_path, plot_paths['full_res'])})",
            "",
            f"![Reduced Newton Iterations]({_relpath(report_path, plot_paths['iterations'])})",
            "",
            f"![Reduced Newton Wall Time]({_relpath(report_path, plot_paths['wall'])})",
            "",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline reduced-Newton analysis on a saved P4 continuation branch.")
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--tol-projected", type=float, default=1.0e-8)
    parser.add_argument("--tol-omega", type=float, default=1.0e-10)
    parser.add_argument("--only-method", type=str, default=None)
    parser.add_argument("--min-accepted-step", type=int, default=None)
    parser.add_argument("--max-accepted-step", type=int, default=None)
    args = parser.parse_args()

    case = _load_case(args.case_dir)
    out_dir = _ensure_dir(args.out_dir)
    plot_dir = _ensure_dir(out_dir / "plots")

    npz = case["npz"]
    step_u = np.asarray(npz["step_U"], dtype=np.float64)
    if step_u.ndim != 3 or step_u.shape[0] < 4:
        raise ValueError("Saved case must contain at least four accepted full displacement states in step_U.")

    omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
    lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
    u_full_hist = np.stack([flatten_field(step_u[i]) for i in range(step_u.shape[0])], axis=0)

    problem = _build_problem_from_case(case)
    free_idx = np.asarray(problem["free_idx"], dtype=np.int64)
    u_free_hist = np.asarray(u_full_hist[:, free_idx], dtype=np.float64)

    kinematic_rows = _kinematic_rows(omega_hist=omega_hist, lambda_hist=lambda_hist, u_full_hist=u_full_hist)

    selected_specs = [
        spec for spec in REDUCED_METHODS if args.only_method is None or spec.name == str(args.only_method)
    ]
    if not selected_specs:
        raise ValueError(f"No reduced method matched --only-method={args.only_method!r}")

    reduced_rows: dict[str, list[dict[str, object]]] = {spec.name: [] for spec in selected_specs}
    for spec in selected_specs:
        for idx in range(spec.min_state_index, omega_hist.size):
            accepted_step = int(idx + 1)
            if args.min_accepted_step is not None and accepted_step < int(args.min_accepted_step):
                continue
            if args.max_accepted_step is not None and accepted_step > int(args.max_accepted_step):
                continue
            row = _reduced_newton_solve(
                problem=problem,
                method_name=spec.name,
                target_idx=idx,
                omega_hist=omega_hist,
                lambda_hist=lambda_hist,
                u_free_hist=u_free_hist,
                u_full_hist=u_full_hist,
                max_iterations=int(args.max_iterations),
                tol_projected=float(args.tol_projected),
                tol_omega=float(args.tol_omega),
            )
            reduced_rows[spec.name].append(row)

    labels = {
        "secant": "Current Secant",
        "blend03": "Blend 0.7 Secant + 0.3 Quadratic",
        "reduced_newton_span2": "Projected Newton Span-2",
        "reduced_newton_span3": "Projected Newton Span-3",
        "reduced_newton_all_prev_cont": "Projected Newton All Previous Continuation Increments",
    }
    plot_rows = {
        "secant": kinematic_rows["secant"],
        "blend03": kinematic_rows["blend03"],
        **reduced_rows,
    }

    plot_paths = {
        "u_rel": _plot_metric(
            path=plot_dir / "predictor_rel_u_error.png",
            title="Offline Saved-Branch Predictor Quality: Relative U Error",
            ylabel="relative U error",
            rows_by_method=plot_rows,
            labels=labels,
            key="u_l2_rel",
            logy=True,
        ),
        "full_res": _plot_metric(
            path=plot_dir / "reduced_newton_full_residual_rel.png",
            title="Projected Newton: Final Full Residual Relative To ||f||",
            ylabel="full residual relative",
            rows_by_method=reduced_rows,
            labels=labels,
            key="full_residual_rel",
            logy=True,
        ),
        "iterations": _plot_metric(
            path=plot_dir / "reduced_newton_iterations.png",
            title="Projected Newton Iterations By Accepted Step",
            ylabel="iterations",
            rows_by_method=reduced_rows,
            labels=labels,
            key="reduced_newton_iterations",
            logy=False,
        ),
        "wall": _plot_metric(
            path=plot_dir / "reduced_newton_wall_time.png",
            title="Projected Newton Offline Wall Time By Accepted Step",
            ylabel="wall time [s]",
            rows_by_method=reduced_rows,
            labels=labels,
            key="wall_s_total",
            logy=True,
        ),
    }

    summary = {
        "problem_build_wall_s": float(problem["problem_build_wall_s"]),
        "case_dir": str(args.case_dir),
        "methods": {},
    }
    for name, rows in kinematic_rows.items():
        summary[name] = _method_summary(rows, include_newton=False)
        summary["methods"][name] = summary[name]
    for name, rows in reduced_rows.items():
        summary[name] = _method_summary(rows, include_newton=True)
        summary["methods"][name] = summary[name]

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path = _write_report(
        out_dir=out_dir,
        case=case,
        problem=problem,
        kinematic_rows=kinematic_rows,
        reduced_rows=reduced_rows,
        summary=summary,
        plot_paths=plot_paths,
    )
    print(json.dumps({"report": str(report_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
