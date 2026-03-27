#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CASE_DIR = Path(
    "artifacts/p4_pmg_shell_best_rank8_full/p4_rank8_step100"
)
DEFAULT_OUT_DIR = Path("artifacts/p4_saved_predictor_analysis")


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    min_state_index: int
    lambda_predictable: bool = True


METHODS = [
    MethodSpec("secant", "Current Secant", 2),
    MethodSpec("quadratic", "Two-Step Quadratic", 3),
    MethodSpec("cubic", "Three-Step Cubic", 4),
    MethodSpec("blend03", "Blend 0.7 Secant + 0.3 Quadratic", 3),
    MethodSpec("oracle_secant_line_alpha", "Oracle Relaxed Alpha On Secant Line", 2, False),
    MethodSpec("oracle_span1", "Oracle Reduced Span-1", 2, False),
    MethodSpec("oracle_span2", "Oracle Reduced Span-2", 3, False),
    MethodSpec("oracle_span3", "Oracle Reduced Span-3", 4, False),
    MethodSpec("oracle_all_prev_cont", "Oracle Reduced All Previous Continuation Increments", 3, False),
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relpath(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, start=from_path.parent)


def _load_case(case_dir: Path) -> dict[str, object]:
    data_dir = case_dir / "data"
    npz = np.load(data_dir / "petsc_run.npz", allow_pickle=True)
    run_info = json.loads((data_dir / "run_info.json").read_text(encoding="utf-8"))
    progress_lines = (data_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    return {
        "case_dir": case_dir,
        "npz": npz,
        "run_info": run_info,
        "progress_lines": progress_lines,
    }


def _lagrange_coeffs(xs: np.ndarray, xt: float) -> np.ndarray:
    coeff = np.ones(xs.size, dtype=float)
    for j in range(xs.size):
        for k in range(xs.size):
            if j != k:
                coeff[j] *= (xt - xs[k]) / (xs[j] - xs[k])
    return coeff


def _predict_vector(method: str, omega_hist: np.ndarray, U_flat: np.ndarray, idx: int) -> np.ndarray:
    if method == "secant":
        o0, o1 = omega_hist[idx - 2], omega_hist[idx - 1]
        u0, u1 = U_flat[idx - 2], U_flat[idx - 1]
        scale = (omega_hist[idx] - o1) / (o1 - o0)
        return u1 + scale * (u1 - u0)
    if method == "quadratic":
        xs = omega_hist[idx - 3 : idx]
        coeff = _lagrange_coeffs(xs, float(omega_hist[idx]))
        return coeff @ U_flat[idx - 3 : idx]
    if method == "cubic":
        xs = omega_hist[idx - 4 : idx]
        coeff = _lagrange_coeffs(xs, float(omega_hist[idx]))
        return coeff @ U_flat[idx - 4 : idx]
    if method == "blend03":
        sec = _predict_vector("secant", omega_hist, U_flat, idx)
        quad = _predict_vector("quadratic", omega_hist, U_flat, idx)
        return 0.7 * sec + 0.3 * quad
    raise ValueError(f"Unknown method {method}")


def _oracle_span_predict(U_flat: np.ndarray, idx: int, span: int) -> np.ndarray:
    increments = [U_flat[idx - j] - U_flat[idx - j - 1] for j in range(1, span + 1)]
    V = np.stack(increments, axis=1)
    rhs = U_flat[idx] - U_flat[idx - 1]
    coeff, *_ = np.linalg.lstsq(V, rhs, rcond=None)
    return U_flat[idx - 1] + V @ coeff


def _oracle_secant_line_predict(U_flat: np.ndarray, idx: int) -> tuple[np.ndarray, float]:
    direction = U_flat[idx - 1] - U_flat[idx - 2]
    rhs = U_flat[idx] - U_flat[idx - 1]
    denom = float(np.dot(direction, direction))
    alpha = 0.0 if abs(denom) < 1.0e-30 else float(np.dot(rhs, direction) / denom)
    return U_flat[idx - 1] + alpha * direction, alpha


def _oracle_all_prev_cont_predict(U_flat: np.ndarray, idx: int) -> tuple[np.ndarray, int, float]:
    increments = [U_flat[j] - U_flat[j - 1] for j in range(2, idx)]
    V = np.stack(increments, axis=1)
    rhs = U_flat[idx] - U_flat[idx - 1]
    coeff, *_ = np.linalg.lstsq(V, rhs, rcond=None)
    s = np.linalg.svd(V, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else math.inf
    return U_flat[idx - 1] + V @ coeff, V.shape[1], cond


def _predict_scalar(method: str, omega_hist: np.ndarray, y_hist: np.ndarray, idx: int) -> float:
    if method == "secant":
        return float(
            y_hist[idx - 1]
            + (omega_hist[idx] - omega_hist[idx - 1])
            / (omega_hist[idx - 1] - omega_hist[idx - 2])
            * (y_hist[idx - 1] - y_hist[idx - 2])
        )
    if method == "quadratic":
        xs = omega_hist[idx - 3 : idx]
        coeff = _lagrange_coeffs(xs, float(omega_hist[idx]))
        return float(np.dot(coeff, y_hist[idx - 3 : idx]))
    if method == "cubic":
        xs = omega_hist[idx - 4 : idx]
        coeff = _lagrange_coeffs(xs, float(omega_hist[idx]))
        return float(np.dot(coeff, y_hist[idx - 4 : idx]))
    if method == "blend03":
        sec = _predict_scalar("secant", omega_hist, y_hist, idx)
        quad = _predict_scalar("quadratic", omega_hist, y_hist, idx)
        return float(0.7 * sec + 0.3 * quad)
    raise ValueError(f"Unknown method {method}")


def _compute_alpha_sweep(omega_hist: np.ndarray, U_flat: np.ndarray) -> dict[str, object]:
    alphas = np.linspace(0.0, 1.0, 101)
    state_indices = np.arange(4, omega_hist.size)
    mean_rel = []
    max_rel = []
    for alpha in alphas:
        rels = []
        for idx in state_indices:
            sec = _predict_vector("secant", omega_hist, U_flat, idx)
            quad = _predict_vector("quadratic", omega_hist, U_flat, idx)
            pred = (1.0 - alpha) * sec + alpha * quad
            true = U_flat[idx]
            err = np.linalg.norm(pred - true)
            rels.append(err / max(np.linalg.norm(true), 1.0e-30))
        mean_rel.append(float(np.mean(rels)))
        max_rel.append(float(np.max(rels)))
    best_idx = int(np.argmin(mean_rel))
    return {
        "alphas": alphas,
        "mean_rel": np.asarray(mean_rel),
        "max_rel": np.asarray(max_rel),
        "best_alpha": float(alphas[best_idx]),
        "best_mean_rel": float(mean_rel[best_idx]),
        "best_max_rel": float(max_rel[best_idx]),
    }


def _collect_rows(case: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]], dict[str, object]]:
    npz = case["npz"]
    U = npz["step_U"]
    U_flat = U.reshape(U.shape[0], -1)
    omega_hist = npz["omega_hist"]
    lambda_hist = npz["lambda_hist"]
    step_index = npz["stats_step_index"]
    step_omega = npz["stats_step_omega"]
    step_lambda = npz["stats_step_lambda"]
    step_newton = npz["stats_step_newton_iterations_total"]
    step_linear = npz["stats_step_linear_iterations"]
    step_wall = npz["stats_step_wall_time"]

    rows: list[dict[str, object]] = []
    by_method: dict[str, list[dict[str, object]]] = {spec.name: [] for spec in METHODS}
    for local_idx, accepted_step in enumerate(step_index):
        state_idx = local_idx + 2
        row = {
            "accepted_step": int(accepted_step),
            "state_idx": int(state_idx),
            "omega": float(step_omega[local_idx]),
            "lambda": float(step_lambda[local_idx]),
            "newton_total": int(step_newton[local_idx]),
            "linear_total": int(step_linear[local_idx]),
            "step_wall_s": float(step_wall[local_idx]),
        }
        rows.append(row)
        true = U_flat[state_idx]
        prev = U_flat[state_idx - 1]
        true_norm = max(float(np.linalg.norm(true)), 1.0e-30)
        true_inc_norm = max(float(np.linalg.norm(true - prev)), 1.0e-30)
        for spec in METHODS:
            if state_idx < spec.min_state_index:
                continue
            basis_dim = np.nan
            basis_cond = np.nan
            alpha_value = np.nan
            alpha_secant = np.nan
            if spec.name == "oracle_all_prev_cont":
                pred, basis_dim, basis_cond = _oracle_all_prev_cont_predict(U_flat, state_idx)
            elif spec.name == "oracle_secant_line_alpha":
                pred, alpha_value = _oracle_secant_line_predict(U_flat, state_idx)
                denom = float(omega_hist[state_idx - 1] - omega_hist[state_idx - 2])
                alpha_secant = np.nan if abs(denom) < 1.0e-30 else float((omega_hist[state_idx] - omega_hist[state_idx - 1]) / denom)
            elif spec.name.startswith("oracle_span"):
                pred = _oracle_span_predict(U_flat, state_idx, int(spec.name.removeprefix("oracle_span")))
            else:
                pred = _predict_vector(spec.name, omega_hist, U_flat, state_idx)
            diff = pred - true
            diff_vec = diff.reshape(3, -1)
            l2_abs = float(np.linalg.norm(diff))
            l2_rel = l2_abs / true_norm
            inc_rel = l2_abs / true_inc_norm
            max_node = float(np.linalg.norm(diff_vec, axis=0).max())
            if spec.lambda_predictable:
                lam_pred = _predict_scalar(spec.name, omega_hist, lambda_hist, state_idx)
                lam_abs = abs(lam_pred - float(lambda_hist[state_idx]))
            else:
                lam_pred = np.nan
                lam_abs = np.nan
            by_method[spec.name].append(
                {
                    **row,
                    "u_l2_abs": l2_abs,
                    "u_l2_rel": l2_rel,
                    "u_increment_rel": inc_rel,
                    "u_max_node_abs": max_node,
                    "lambda_pred": lam_pred,
                    "lambda_abs_err": lam_abs,
                    "basis_dim": basis_dim,
                    "basis_cond": basis_cond,
                    "alpha_value": alpha_value,
                    "alpha_secant": alpha_secant,
                }
            )
    alpha_sweep = _compute_alpha_sweep(omega_hist, U_flat)
    return rows, by_method, alpha_sweep


def _summary(by_method: dict[str, list[dict[str, object]]], *, common_only: bool) -> list[dict[str, object]]:
    common_steps = None
    if common_only:
        key_sets = [set(row["accepted_step"] for row in rows) for rows in by_method.values() if rows]
        common_steps = set.intersection(*key_sets)
    out = []
    for spec in METHODS:
        rows = by_method[spec.name]
        if common_steps is not None:
            rows = [row for row in rows if row["accepted_step"] in common_steps]
        arr_u_rel = np.asarray([row["u_l2_rel"] for row in rows], dtype=float)
        arr_u_inc = np.asarray([row["u_increment_rel"] for row in rows], dtype=float)
        arr_u_abs = np.asarray([row["u_l2_abs"] for row in rows], dtype=float)
        arr_u_max = np.asarray([row["u_max_node_abs"] for row in rows], dtype=float)
        arr_lam = np.asarray([row["lambda_abs_err"] for row in rows], dtype=float)
        arr_basis_dim = np.asarray([row.get("basis_dim", np.nan) for row in rows], dtype=float)
        arr_basis_cond = np.asarray([row.get("basis_cond", np.nan) for row in rows], dtype=float)
        if arr_u_rel.size == 0:
            out.append(
                {
                    "method": spec.name,
                    "label": spec.label,
                    "targets": 0,
                    "mean_u_l2_rel": np.nan,
                    "max_u_l2_rel": np.nan,
                    "mean_u_increment_rel": np.nan,
                    "max_u_increment_rel": np.nan,
                    "mean_u_l2_abs": np.nan,
                    "max_u_l2_abs": np.nan,
                    "mean_u_max_node_abs": np.nan,
                    "max_u_max_node_abs": np.nan,
                    "mean_lambda_abs_err": np.nan,
                    "max_lambda_abs_err": np.nan,
                }
            )
            continue
        out.append(
            {
                "method": spec.name,
                "label": spec.label,
                "targets": len(rows),
                "mean_u_l2_rel": float(np.mean(arr_u_rel)),
                "max_u_l2_rel": float(np.max(arr_u_rel)),
                "mean_u_increment_rel": float(np.mean(arr_u_inc)),
                "max_u_increment_rel": float(np.max(arr_u_inc)),
                "mean_u_l2_abs": float(np.mean(arr_u_abs)),
                "max_u_l2_abs": float(np.max(arr_u_abs)),
                "mean_u_max_node_abs": float(np.mean(arr_u_max)),
                "max_u_max_node_abs": float(np.max(arr_u_max)),
                "mean_lambda_abs_err": float(np.nanmean(arr_lam)) if np.isfinite(arr_lam).any() else np.nan,
                "max_lambda_abs_err": float(np.nanmax(arr_lam)) if np.isfinite(arr_lam).any() else np.nan,
                "mean_basis_dim": float(np.nanmean(arr_basis_dim)) if np.isfinite(arr_basis_dim).any() else np.nan,
                "max_basis_dim": float(np.nanmax(arr_basis_dim)) if np.isfinite(arr_basis_dim).any() else np.nan,
                "mean_basis_cond": float(np.nanmean(arr_basis_cond)) if np.isfinite(arr_basis_cond).any() else np.nan,
                "max_basis_cond": float(np.nanmax(arr_basis_cond)) if np.isfinite(arr_basis_cond).any() else np.nan,
            }
        )
    return out


def _subset_summary(
    by_method: dict[str, list[dict[str, object]]],
    *,
    min_step: int | None = None,
    max_step: int | None = None,
) -> list[dict[str, object]]:
    out = []
    for spec in METHODS:
        rows = by_method[spec.name]
        if min_step is not None:
            rows = [row for row in rows if row["accepted_step"] >= min_step]
        if max_step is not None:
            rows = [row for row in rows if row["accepted_step"] <= max_step]
        arr_u_rel = np.asarray([row["u_l2_rel"] for row in rows], dtype=float)
        arr_u_inc = np.asarray([row["u_increment_rel"] for row in rows], dtype=float)
        if arr_u_rel.size == 0:
            continue
        basis_dim = np.asarray([row.get("basis_dim", np.nan) for row in rows], dtype=float)
        basis_cond = np.asarray([row.get("basis_cond", np.nan) for row in rows], dtype=float)
        finite_dim = np.isfinite(basis_dim)
        finite_cond = np.isfinite(basis_cond)
        out.append(
            {
                "method": spec.name,
                "label": spec.label,
                "targets": len(rows),
                "mean_u_l2_rel": float(np.mean(arr_u_rel)),
                "max_u_l2_rel": float(np.max(arr_u_rel)),
                "mean_u_increment_rel": float(np.mean(arr_u_inc)),
                "max_u_increment_rel": float(np.max(arr_u_inc)),
                "mean_basis_dim": float(np.mean(basis_dim[finite_dim])) if finite_dim.any() else np.nan,
                "max_basis_dim": float(np.max(basis_dim[finite_dim])) if finite_dim.any() else np.nan,
                "mean_basis_cond": float(np.mean(basis_cond[finite_cond])) if finite_cond.any() else np.nan,
                "max_basis_cond": float(np.max(basis_cond[finite_cond])) if finite_cond.any() else np.nan,
            }
        )
    return out


def _plot_branch(rows: list[dict[str, object]], plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    omega = [row["omega"] for row in rows]
    lam = [row["lambda"] for row in rows]
    ax.plot(omega, lam, marker="o", linewidth=1.8)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(r"Saved P4(L1) Branch Used For Offline Predictor Scoring")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    path = plot_dir / "lambda_omega_branch.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_predictor_metric(
    rows: list[dict[str, object]],
    by_method: dict[str, list[dict[str, object]]],
    *,
    key: str,
    ylabel: str,
    title: str,
    path: Path,
    logy: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    step_x = [row["accepted_step"] for row in rows]
    for spec in METHODS:
        xs = [row["accepted_step"] for row in by_method[spec.name]]
        ys = [row[key] for row in by_method[spec.name]]
        highlight = spec.name == "oracle_secant_line_alpha"
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0 if highlight else 1.6,
            markersize=10.0 if highlight else 5.0,
            markeredgewidth=1.2 if highlight else 0.8,
            zorder=4 if highlight else 2,
            label=spec.label,
        )
    ax.set_xlabel("Accepted continuation step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.35)
    ax.set_xticks(step_x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_step_difficulty(rows: list[dict[str, object]], plot_dir: Path) -> Path:
    x = [row["accepted_step"] for row in rows]
    newton = [row["newton_total"] for row in rows]
    linear = [row["linear_total"] for row in rows]
    wall = [row["step_wall_s"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.2), sharex=True)
    axes[0].plot(x, wall, marker="o")
    axes[0].set_ylabel("Step wall [s]")
    axes[1].plot(x, newton, marker="o", color="tab:orange")
    axes[1].set_ylabel("Newton total")
    axes[2].plot(x, linear, marker="o", color="tab:green")
    axes[2].set_ylabel("Linear total")
    axes[2].set_xlabel("Accepted continuation step")
    for ax in axes:
        ax.grid(True, alpha=0.35)
    axes[0].set_title("Observed Step Difficulty On The Saved Branch")
    fig.tight_layout()
    path = plot_dir / "step_difficulty.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_alpha_sweep(alpha_sweep: dict[str, object], plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(alpha_sweep["alphas"], alpha_sweep["mean_rel"], label="mean relative U error")
    ax.plot(alpha_sweep["alphas"], alpha_sweep["max_rel"], label="max relative U error")
    ax.axvline(alpha_sweep["best_alpha"], color="k", linestyle="--", linewidth=1.0, label=f"best mean alpha={alpha_sweep['best_alpha']:.2f}")
    ax.set_xlabel(r"blend weight $\alpha$ in $(1-\alpha)U_{\mathrm{secant}} + \alpha U_{\mathrm{quadratic}}$")
    ax.set_ylabel("relative U error")
    ax.set_title("Retrospective Secant/Quadratic Blend Sweep On Common Subset")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "blend_alpha_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _summary_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Predictor | Targets | Mean rel U error | Max rel U error | Mean increment-rel error | Max increment-rel error | Mean abs lambda error | Max abs lambda error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {targets:d} | {mean_u_l2_rel:.6e} | {max_u_l2_rel:.6e} | {mean_u_increment_rel:.6e} | {max_u_increment_rel:.6e} | {mean_lambda_abs_err:.6e} | {max_lambda_abs_err:.6e} |".format(
                **row
            )
        )
    return lines


def _subset_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Predictor | Targets | Mean rel U error | Max rel U error | Mean increment-rel error | Max increment-rel error | Mean basis dim | Max basis dim | Mean basis cond | Max basis cond |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {targets:d} | {mean_u_l2_rel:.6e} | {max_u_l2_rel:.6e} | {mean_u_increment_rel:.6e} | {max_u_increment_rel:.6e} | {mean_basis_dim:.3f} | {max_basis_dim:.3f} | {mean_basis_cond:.6e} | {max_basis_cond:.6e} |".format(
                **row
            )
        )
    return lines


def _difficulty_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Accepted step | Omega | Lambda | Step wall [s] | Newton total | Linear total |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['accepted_step']:d} | {row['omega']:.6e} | {row['lambda']:.9f} | {row['step_wall_s']:.3f} | {row['newton_total']:d} | {row['linear_total']:d} |"
        )
    return lines


def _common_predictor_table(by_method: dict[str, list[dict[str, object]]]) -> list[str]:
    common_steps = set.intersection(
        *(set(row["accepted_step"] for row in rows) for rows in by_method.values() if rows)
    )
    spec_by_name = {spec.name: spec.label for spec in METHODS}
    common_rows = {
        spec.name: {row["accepted_step"]: row for row in rows if row["accepted_step"] in common_steps}
        for spec, rows in ((spec, by_method[spec.name]) for spec in METHODS)
    }
    lines = [
        "| Step | Omega | Lambda | Newton | Linear | Secant rel U | Quadratic rel U | Cubic rel U | Blend0.3 rel U | Secant inc-rel | Quadratic inc-rel | Cubic inc-rel | Blend0.3 inc-rel |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step in sorted(common_steps):
        base = common_rows["secant"][step]
        lines.append(
            "| {step:d} | {omega:.6e} | {lam:.9f} | {newton:d} | {linear:d} | {sec_rel:.6e} | {quad_rel:.6e} | {cubic_rel:.6e} | {blend_rel:.6e} | {sec_inc:.6e} | {quad_inc:.6e} | {cubic_inc:.6e} | {blend_inc:.6e} |".format(
                step=step,
                omega=base["omega"],
                lam=base["lambda"],
                newton=base["newton_total"],
                linear=base["linear_total"],
                sec_rel=common_rows["secant"][step]["u_l2_rel"],
                quad_rel=common_rows["quadratic"][step]["u_l2_rel"],
                cubic_rel=common_rows["cubic"][step]["u_l2_rel"],
                blend_rel=common_rows["blend03"][step]["u_l2_rel"],
                sec_inc=common_rows["secant"][step]["u_increment_rel"],
                quad_inc=common_rows["quadratic"][step]["u_increment_rel"],
                cubic_inc=common_rows["cubic"][step]["u_increment_rel"],
                blend_inc=common_rows["blend03"][step]["u_increment_rel"],
            )
        )
    return lines


def _oracle_predictor_table(by_method: dict[str, list[dict[str, object]]]) -> list[str]:
    table_methods = [
        "secant",
        "blend03",
        "oracle_secant_line_alpha",
        "oracle_span1",
        "oracle_span2",
        "oracle_span3",
        "oracle_all_prev_cont",
    ]
    common_steps = set.intersection(
        *(set(row["accepted_step"] for row in by_method[name]) for name in table_methods)
    )
    rows_by_method = {
        name: {row["accepted_step"]: row for row in by_method[name] if row["accepted_step"] in common_steps}
        for name in table_methods
    }
    lines = [
        "| Step | Omega | Lambda | Newton | Linear | Secant rel U | Blend0.3 rel U | Oracle line-alpha rel U | Oracle line alpha | Secant alpha | Oracle span1 rel U | Oracle span2 rel U | Oracle span3 rel U | Oracle all-prev rel U | All-prev basis dim | All-prev cond |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step in sorted(common_steps):
        base = rows_by_method["secant"][step]
        lines.append(
            "| {step:d} | {omega:.6e} | {lam:.9f} | {newton:d} | {linear:d} | {sec_rel:.6e} | {blend_rel:.6e} | {oline_rel:.6e} | {oline_alpha:.6f} | {sec_alpha:.6f} | {o1_rel:.6e} | {o2_rel:.6e} | {o3_rel:.6e} | {oall_rel:.6e} | {oall_dim:.0f} | {oall_cond:.6e} |".format(
                step=step,
                omega=base["omega"],
                lam=base["lambda"],
                newton=base["newton_total"],
                linear=base["linear_total"],
                sec_rel=rows_by_method["secant"][step]["u_l2_rel"],
                blend_rel=rows_by_method["blend03"][step]["u_l2_rel"],
                oline_rel=rows_by_method["oracle_secant_line_alpha"][step]["u_l2_rel"],
                oline_alpha=rows_by_method["oracle_secant_line_alpha"][step]["alpha_value"],
                sec_alpha=rows_by_method["oracle_secant_line_alpha"][step]["alpha_secant"],
                o1_rel=rows_by_method["oracle_span1"][step]["u_l2_rel"],
                o2_rel=rows_by_method["oracle_span2"][step]["u_l2_rel"],
                o3_rel=rows_by_method["oracle_span3"][step]["u_l2_rel"],
                oall_rel=rows_by_method["oracle_all_prev_cont"][step]["u_l2_rel"],
                oall_dim=rows_by_method["oracle_all_prev_cont"][step]["basis_dim"],
                oall_cond=rows_by_method["oracle_all_prev_cont"][step]["basis_cond"],
            )
        )
    return lines


def _line_alpha_table(by_method: dict[str, list[dict[str, object]]]) -> list[str]:
    sec_rows = {row["accepted_step"]: row for row in by_method["secant"]}
    alpha_rows = {row["accepted_step"]: row for row in by_method["oracle_secant_line_alpha"]}
    common_steps = sorted(set(sec_rows).intersection(alpha_rows))
    lines = [
        "| Step | Omega | Lambda | Newton | Linear | Secant alpha | Oracle line alpha | Secant rel U | Oracle line-alpha rel U | Improvement ratio |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step in common_steps:
        sec = sec_rows[step]
        alpha = alpha_rows[step]
        ratio = float(alpha["u_l2_rel"] / max(sec["u_l2_rel"], 1.0e-30))
        lines.append(
            "| {step:d} | {omega:.6e} | {lam:.9f} | {newton:d} | {linear:d} | {alpha_sec:.6f} | {alpha_or:.6f} | {sec_rel:.6e} | {alpha_rel:.6e} | {ratio:.3f} |".format(
                step=step,
                omega=sec["omega"],
                lam=sec["lambda"],
                newton=sec["newton_total"],
                linear=sec["linear_total"],
                alpha_sec=alpha["alpha_secant"],
                alpha_or=alpha["alpha_value"],
                sec_rel=sec["u_l2_rel"],
                alpha_rel=alpha["u_l2_rel"],
                ratio=ratio,
            )
        )
    return lines


def _plot_line_alpha(by_method: dict[str, list[dict[str, object]]], plot_dir: Path) -> Path:
    alpha_rows = by_method["oracle_secant_line_alpha"]
    xs = [row["accepted_step"] for row in alpha_rows]
    sec_alpha = [row["alpha_secant"] for row in alpha_rows]
    oracle_alpha = [row["alpha_value"] for row in alpha_rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(
        xs,
        sec_alpha,
        marker="o",
        linewidth=1.6,
        markersize=5.0,
        markeredgewidth=0.8,
        zorder=2,
        label="Secant alpha from target omega",
    )
    ax.plot(
        xs,
        oracle_alpha,
        marker="o",
        linewidth=2.0,
        markersize=10.0,
        markeredgewidth=1.2,
        zorder=4,
        label="Oracle best alpha on same line",
    )
    ax.set_xlabel("Accepted continuation step")
    ax.set_ylabel("alpha")
    ax.set_title("Scalar Freedom On The Current Secant Line")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "secant_line_alpha.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_report(*, case_dir: Path, out_dir: Path) -> dict[str, object]:
    out_dir = _ensure_dir(out_dir)
    plot_dir = _ensure_dir(out_dir / "plots")
    case = _load_case(case_dir)
    rows, by_method, alpha_sweep = _collect_rows(case)
    all_summary = _summary(by_method, common_only=False)
    common_summary = _summary(by_method, common_only=True)
    early_summary = _subset_summary(by_method, max_step=7)
    hard_summary = _subset_summary(by_method, min_step=8)

    branch_plot = _plot_branch(rows, plot_dir)
    rel_u_plot = _plot_predictor_metric(
        rows,
        by_method,
        key="u_l2_rel",
        ylabel="relative U error",
        title="Predictor Relative Displacement Error",
        path=plot_dir / "predictor_rel_u_error.png",
    )
    inc_u_plot = _plot_predictor_metric(
        rows,
        by_method,
        key="u_increment_rel",
        ylabel="error / ||actual increment||",
        title="Predictor Error Relative To Actual Step Increment",
        path=plot_dir / "predictor_increment_rel_error.png",
    )
    lam_plot = _plot_predictor_metric(
        rows,
        by_method,
        key="lambda_abs_err",
        ylabel="absolute lambda prediction error",
        title="Predictor Lambda Error",
        path=plot_dir / "predictor_lambda_error.png",
    )
    difficulty_plot = _plot_step_difficulty(rows, plot_dir)
    alpha_plot = _plot_alpha_sweep(alpha_sweep, plot_dir)
    line_alpha_plot = _plot_line_alpha(by_method, plot_dir)

    summary = {
        "case_dir": str(case_dir),
        "all_summary": all_summary,
        "common_summary": common_summary,
        "early_summary": early_summary,
        "hard_summary": hard_summary,
        "alpha_sweep": {
            "best_alpha": alpha_sweep["best_alpha"],
            "best_mean_rel": alpha_sweep["best_mean_rel"],
            "best_max_rel": alpha_sweep["best_max_rel"],
        },
        "per_method_rows": by_method,
        "difficulty_rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run_info = case["run_info"]["run_info"]
    report_path = out_dir / "README.md"
    lines = [
        "# Offline Predictor Analysis On Saved P4 Branch",
        "",
        "This report scores offline continuation predictors against a finished `P4(L1)` run that actually saved every accepted displacement field.",
        "",
        f"Source artifact: [{case_dir}]({_relpath(report_path, case_dir)})",
        "",
        "Why this run: among the available `P4` artifacts with saved `step_U`, it is the longest continuation run that also reaches its final `omega` stop.",
        "",
        "Important limitation: this is retrospective scoring against accepted states only. It does not replay rejected attempts or Newton residuals. The question answered here is narrower: how close would each `u_init` guess have been to the accepted next state that was actually reached?",
        "",
        "## Run",
        "",
        f"- MPI ranks: `{run_info['mpi_size']}`",
        f"- Unknowns: `{run_info['unknowns']}`",
        f"- Accepted states saved in `step_U`: `{case['npz']['step_U'].shape[0]}`",
        f"- Final omega: `{case['npz']['omega_hist'][-1]:.6e}`",
        f"- Final lambda: `{case['npz']['lambda_hist'][-1]:.9f}`",
        "",
        "## Predictors",
        "",
        "- `Current Secant`: the existing one-step secant extrapolation in `omega`.",
        "- `Two-Step Quadratic`: quadratic Lagrange extrapolation through the last 3 accepted states.",
        "- `Three-Step Cubic`: cubic Lagrange extrapolation through the last 4 accepted states.",
        "- `Blend 0.7 Secant + 0.3 Quadratic`: a mild curvature correction added to secant.",
        "- `Oracle Relaxed Alpha On Secant Line`: best possible predictor of the form `U_i + alpha (U_i - U_{i-1})`, computed retrospectively against the known next state. This is the right upper-bound test for a future 1D alpha search along the current secant direction. It is not causal.",
        "- `Oracle Reduced Span-k`: best possible displacement predictor inside the span of the last `k` accepted displacement increments, computed retrospectively by least squares against the known next state. This is not causal; it is an upper bound on what a residual-minimizing reduced predictor might achieve in that subspace.",
        "",
        "## Summary Over All Available Targets",
        "",
        *_summary_table(all_summary),
        "",
        "## Fair Summary On Common Comparable Subset",
        "",
        "The common subset starts at accepted continuation step `5`, because cubic needs four prior accepted states.",
        "",
        *_summary_table(common_summary),
        "",
        "## Early Branch Summary",
        "",
        "Accepted continuation steps `3-7`.",
        "",
        *_subset_table(early_summary),
        "",
        "## Hard Tail Summary",
        "",
        "Accepted continuation steps `8-13`.",
        "",
        *_subset_table(hard_summary),
        "",
        "## Blend Sweep",
        "",
        f"A coarse retrospective sweep over `alpha` in `(1-alpha) U_secant + alpha U_quadratic` on the common subset favored `alpha ~= {alpha_sweep['best_alpha']:.2f}` by mean relative displacement error. The report keeps `alpha = 0.30` because it is close to that optimum and easy to interpret.",
        "",
        f"![Blend Alpha Sweep]({_relpath(report_path, alpha_plot)})",
        "",
        "## Scalar Alpha Freedom On The Current Secant Line",
        "",
        "If the next predictor is restricted to the current secant direction, then the only remaining degree of freedom is the scalar amplitude `alpha`. This table answers the narrow question: how much headroom is there on that line if the initial guess is allowed to miss the exact target `omega` and Newton later corrects it?",
        "",
        *_line_alpha_table(by_method),
        "",
        f"![Secant Line Alpha]({_relpath(report_path, line_alpha_plot)})",
        "",
        "## Observed Branch Difficulty",
        "",
        *_difficulty_table(rows),
        "",
        f"![Lambda Omega Branch]({_relpath(report_path, branch_plot)})",
        "",
        f"![Step Difficulty]({_relpath(report_path, difficulty_plot)})",
        "",
        "## Predictor Error Plots",
        "",
        f"![Predictor Relative U Error]({_relpath(report_path, rel_u_plot)})",
        "",
        f"![Predictor Increment Relative Error]({_relpath(report_path, inc_u_plot)})",
        "",
        f"![Predictor Lambda Error]({_relpath(report_path, lam_plot)})",
        "",
        "## Common-Subset Per-Step Comparison",
        "",
        *_common_predictor_table(by_method),
        "",
        "## Oracle Reduced-Subspace Comparison",
        "",
        "This table is the most relevant one for deciding whether a cheap reduced predictor is worth implementing online. If the oracle span predictors are much better than secant, there is genuine headroom in the previous-step increment space.",
        "",
        *_oracle_predictor_table(by_method),
        "",
        "## Takeaways",
        "",
        "- On this saved branch, plain secant is already strong. Full quadratic and especially cubic extrapolation make the displacement guess worse on average.",
        "- There is real scalar freedom on the current secant line in the hard tail. The oracle best line-alpha is noticeably different from the exact-omega secant alpha on steps `8-9`, and that alone lowers displacement error before any higher-dimensional reduced-space idea is used.",
        "- The hard plastic-flow tail is exactly where polynomial extrapolation becomes unstable. Cubic is clearly not useful here.",
        "- A mild blend `0.7 * secant + 0.3 * quadratic` is the only extra variant that looks promising on this branch. On the common subset it lowers both mean relative displacement error and mean increment-relative error versus plain secant.",
        "- The oracle reduced-span results are much stronger than the purely kinematic predictors in the hard tail. That means a residual-minimizing reduced predictor in the span of the last `2-3` increments has real headroom and is worth an online experiment.",
        "- The main gain appears already at span-2. Span-3 is only slightly better, so the cheapest credible reduced-space experiment is a 2-vector basis.",
        "- These oracle span results are still retrospective upper bounds. They do not prove that a one-shot reduced residual solve will match them, but they do show the subspace itself is expressive enough to matter.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "report": str(report_path),
        "summary": str(out_dir / "summary.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline predictor analysis on a saved P4 continuation branch.")
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    result = generate_report(case_dir=Path(args.case_dir), out_dir=Path(args.out_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
