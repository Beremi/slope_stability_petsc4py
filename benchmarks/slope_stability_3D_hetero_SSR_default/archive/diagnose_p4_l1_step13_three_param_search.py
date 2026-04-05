#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from petsc4py import PETSc

from slope_stability.continuation.indirect import _free, _free_dot, _secant_alpha


ROOT = Path(__file__).resolve().parents[3]
REPLAY_SCRIPT = ROOT / "benchmarks/slope_stability_3D_hetero_SSR_default/archive/replay_p4_l1_step13_predictor_compare.py"
OUT_DIR = ROOT / "artifacts/p4_l1_step13_three_param_replay/search_diagnostics"


def _load_replay_module():
    spec = importlib.util.spec_from_file_location("p4_step13_replay", REPLAY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _evaluate_merit(*, U, lambda_value, omega_target, Q, f, constitutive_matrix_builder, penalty_weight):
    f_free = _free(np.asarray(f, dtype=np.float64), Q)
    norm_f = float(np.linalg.norm(f_free))
    if norm_f <= 1.0e-30:
        norm_f = 1.0
    build_F_all_free = getattr(constitutive_matrix_builder, "build_F_all_free", None)
    if callable(build_F_all_free):
        F_free = np.asarray(
            build_F_all_free(float(lambda_value), np.asarray(U, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
    else:
        F_all = np.asarray(
            constitutive_matrix_builder.build_F_all(float(lambda_value), np.asarray(U, dtype=np.float64)),
            dtype=np.float64,
        )
        F_free = _free(F_all, Q)
    res_rel = float(np.linalg.norm(F_free - f_free) / norm_f)
    omega_now = _free_dot(f, U, Q)
    omega_scale = max(abs(float(omega_target)), 1.0)
    omega_rel = float((omega_now - float(omega_target)) / omega_scale)
    total = float(res_rel * res_rel + float(penalty_weight) * omega_rel * omega_rel)
    return {
        "res_rel": float(res_rel),
        "res_rel_sq": float(res_rel * res_rel),
        "omega_now": float(omega_now),
        "omega_rel": float(omega_rel),
        "omega_penalty": float(float(penalty_weight) * omega_rel * omega_rel),
        "total": total,
    }


def _field(coeffs, *, U_i, U_im1, U_im2):
    a, b, c = (float(v) for v in coeffs)
    return a * U_i + b * U_im1 + c * U_im2


def main() -> None:
    replay = _load_replay_module()
    run_info, _target_info, state_npz, target_npz = replay._load_params()
    built = replay._build_case(run_info)

    q_mask = built["q_mask"]
    f_V = built["f_V"]
    const_builder = built["const_builder"]
    solver = built["solver"]

    step_u = np.asarray(state_npz["step_U"], dtype=np.float64)
    omega_hist = np.asarray(state_npz["omega_hist"], dtype=np.float64)
    lambda_hist = np.asarray(state_npz["lambda_hist"], dtype=np.float64)
    U_im2 = np.asarray(step_u[9], dtype=np.float64)
    U_im1 = np.asarray(step_u[10], dtype=np.float64)
    U_i = np.asarray(step_u[11], dtype=np.float64)
    omega_im2 = float(omega_hist[9])
    omega_im1 = float(omega_hist[10])
    omega_i = float(omega_hist[11])
    lambda_i = float(lambda_hist[11])
    omega_target = float(np.asarray(target_npz["omega_hist"], dtype=np.float64)[12])

    penalty_weight = 10.0
    alpha_sec = _secant_alpha(omega_old=omega_im1, omega=omega_i, omega_target=omega_target)
    center = np.asarray([1.0 + alpha_sec, -alpha_sec, 0.0], dtype=np.float64)
    steps = np.asarray([0.25, 0.25, 0.125], dtype=np.float64)
    lower = np.asarray([max(0.0, center[0] - 0.75), max(-2.0, center[1] - 0.75), -0.5], dtype=np.float64)
    upper = np.asarray([min(3.0, center[0] + 0.75), min(1.0, center[1] + 0.75), 0.5], dtype=np.float64)

    axis_points = [("center", center.copy())]
    for axis, delta in enumerate(steps):
        for sign in (-1.0, 1.0):
            coeffs = center.copy()
            coeffs[axis] = np.clip(coeffs[axis] + sign * delta, lower[axis], upper[axis])
            axis_points.append((f"axis{axis}_{'minus' if sign < 0 else 'plus'}", coeffs))

    axis_results = []
    for name, coeffs in axis_points:
        merit = _evaluate_merit(
            U=_field(coeffs, U_i=U_i, U_im1=U_im1, U_im2=U_im2),
            lambda_value=lambda_i,
            omega_target=omega_target,
            Q=q_mask,
            f=f_V,
            constitutive_matrix_builder=const_builder,
            penalty_weight=penalty_weight,
        )
        axis_results.append(
            {
                "label": name,
                "alpha": float(coeffs[0]),
                "beta": float(coeffs[1]),
                "gamma": float(coeffs[2]),
                **merit,
            }
        )

    # Exact-omega plane around the secant center.
    omega_vec = np.asarray([omega_i, omega_im1, omega_im2], dtype=np.float64)
    t1 = np.asarray([1.0, -omega_i / omega_im1, 0.0], dtype=np.float64)
    t2 = np.asarray([1.0, 0.0, -omega_i / omega_im2], dtype=np.float64)
    # orthonormalize tangent directions for stable plotting/steps
    basis = np.column_stack([t1, t2])
    qmat, _ = np.linalg.qr(basis)
    t1n = np.asarray(qmat[:, 0], dtype=np.float64)
    t2n = np.asarray(qmat[:, 1], dtype=np.float64)
    # keep tangent orientation exact numerically
    assert abs(float(np.dot(omega_vec, t1n))) < 1.0e-8 * np.linalg.norm(omega_vec)
    assert abs(float(np.dot(omega_vec, t2n))) < 1.0e-8 * np.linalg.norm(omega_vec)

    grid_vals = np.linspace(-0.3, 0.3, 7)
    plane_records = []
    best_plane = None
    for s in grid_vals:
        for t in grid_vals:
            coeffs = center + float(s) * t1n + float(t) * t2n
            merit = _evaluate_merit(
                U=_field(coeffs, U_i=U_i, U_im1=U_im1, U_im2=U_im2),
                lambda_value=lambda_i,
                omega_target=omega_target,
                Q=q_mask,
                f=f_V,
                constitutive_matrix_builder=const_builder,
                penalty_weight=penalty_weight,
            )
            rec = {
                "s": float(s),
                "t": float(t),
                "alpha": float(coeffs[0]),
                "beta": float(coeffs[1]),
                "gamma": float(coeffs[2]),
                **merit,
            }
            plane_records.append(rec)
            if best_plane is None or merit["res_rel_sq"] < best_plane["res_rel_sq"]:
                best_plane = rec

    close_solver = getattr(solver, "close", None)
    if callable(close_solver):
        close_solver()

    if PETSc.COMM_WORLD.getRank() != 0:
        PETSc.COMM_WORLD.Barrier()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    axis_sorted = axis_results
    labels = [r["label"] for r in axis_sorted]
    totals = [r["total"] for r in axis_sorted]
    res_terms = [r["res_rel_sq"] for r in axis_sorted]
    pen_terms = [r["omega_penalty"] for r in axis_sorted]

    plt.figure(figsize=(9, 5))
    x = np.arange(len(labels))
    plt.bar(x, res_terms, label="residual term")
    plt.bar(x, pen_terms, bottom=res_terms, label="omega penalty")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Merit components")
    plt.title("Axis-neighbor merit around secant center")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "axis_merit_components.png", dpi=180)
    plt.close()

    grid_total = np.asarray([rec["total"] for rec in plane_records], dtype=np.float64).reshape(len(grid_vals), len(grid_vals))
    grid_res = np.asarray([rec["res_rel_sq"] for rec in plane_records], dtype=np.float64).reshape(len(grid_vals), len(grid_vals))
    plt.figure(figsize=(6, 5))
    plt.imshow(grid_res, origin="lower", extent=[grid_vals[0], grid_vals[-1], grid_vals[0], grid_vals[-1]], aspect="auto")
    plt.colorbar(label="Residual term")
    plt.scatter([0.0], [0.0], c="red", s=80, marker="x", label="secant center")
    plt.scatter([best_plane["s"]], [best_plane["t"]], c="white", s=40, marker="o", label="best tested")
    plt.xlabel("s")
    plt.ylabel("t")
    plt.title("Exact-omega tangent-plane residual landscape")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "exact_omega_plane_residual.png", dpi=180)
    plt.close()

    summary = {
        "alpha_sec": float(alpha_sec),
        "center": center.tolist(),
        "axis_results": axis_results,
        "best_plane": best_plane,
        "plane_records": plane_records,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    axis_rows = "\n".join(
        f"| `{r['label']}` | `{r['alpha']:.6f}` | `{r['beta']:.6f}` | `{r['gamma']:.6f}` | `{r['res_rel_sq']:.6e}` | `{r['omega_penalty']:.6e}` | `{r['total']:.6e}` | `{r['omega_rel']:.6e}` |"
        for r in axis_sorted
    )
    report = f"""# Step-13 Three-Parameter Search Diagnostics

This inspects the saved step-13 objective only. It does **not** rerun the continuation.

Source state:

- saved step-12 secant artifact: [run_info.json](../../p4_l1_alpha_refine_compare/rank8_secant_step12/data/run_info.json)
- target accepted step-13 omega: [run_info.json](../../p4_l1_step13_three_param_compare/rank8_secant_step13/data/run_info.json)

The current 3-parameter predictor uses

\\[
u = \\alpha u_i + \\beta u_{{i-1}} + \\gamma u_{{i-2}}
\\]

with merit

\\[
\\|R(u,\\lambda_i)\\|_{{rel}}^2 + 10\\,(f^T u - \\omega_{{target}})^2 / \\omega_{{target}}^2.
\\]

The search center is the secant predictor written in this basis:

\\[
(\\alpha,\\beta,\\gamma) = (1+\\alpha_{{sec}}, -\\alpha_{{sec}}, 0)
\\]

with

\\[
\\alpha_{{sec}} = {alpha_sec:.12f}.
\\]

So the search center is:

- `alpha = {center[0]:.12f}`
- `beta = {center[1]:.12f}`
- `gamma = {center[2]:.12f}`

## Axis Neighbors

| Point | alpha | beta | gamma | residual term | omega penalty | total merit | omega_rel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{axis_rows}

## Best Tested Point On Exact-omega Tangent Plane

The exact-omega plane was parameterized locally by two tangent directions and sampled on a `7 x 7` grid around the secant center.

- best tested `s, t`: `({best_plane['s']:.3f}, {best_plane['t']:.3f})`
- coefficients:
  - `alpha = {best_plane['alpha']:.6f}`
  - `beta = {best_plane['beta']:.6f}`
  - `gamma = {best_plane['gamma']:.6f}`
- residual term: `{best_plane['res_rel_sq']:.6e}`
- total merit: `{best_plane['total']:.6e}`
- omega_rel: `{best_plane['omega_rel']:.6e}`

## Plots

![Axis merit components](plots/axis_merit_components.png)

![Exact-omega tangent plane residual](plots/exact_omega_plane_residual.png)
"""
    (OUT_DIR / "README.md").write_text(report)
    PETSc.COMM_WORLD.Barrier()


if __name__ == "__main__":
    main()
