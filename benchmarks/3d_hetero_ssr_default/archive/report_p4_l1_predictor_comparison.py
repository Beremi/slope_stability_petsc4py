#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("artifacts/p4_l1_smart_predictor_compare_omega7e6/report")
FULL_CASES = (
    {
        "name": "secant",
        "label": "Secant",
        "run_dir": Path("artifacts/p4_l1_smart_predictor_compare_omega7e6/full/secant/data"),
    },
    {
        "name": "secant_energy_alpha",
        "label": "Secant Energy Alpha",
        "run_dir": Path("artifacts/p4_l1_smart_predictor_compare_omega7e6/full/secant_energy_alpha/data"),
    },
)
SMOKE_PREFIX_CASES = (
    {
        "name": "secant",
        "label": "Secant",
        "run_dir": Path("artifacts/p4_l1_smart_predictor_compare_omega7e6/smoke/secant/data"),
    },
    {
        "name": "secant_energy_alpha",
        "label": "Secant Energy Alpha",
        "run_dir": Path("artifacts/p4_l1_smart_predictor_compare_omega7e6/smoke_v4/secant_energy_alpha/data"),
    },
    {
        "name": "coarse_p1_solution",
        "label": "Coarse P1 Solution",
        "run_dir": Path("artifacts/p4_l1_smart_predictor_compare_omega7e6/smoke_v10/coarse_p1_solution/data"),
    },
)
SMOKE_CASES = (
    {
        "name": "coarse_p1_solution",
        "label": "Coarse P1 Solution",
        "run_dir": Path("artifacts/p4_l1_smart_predictor_compare_omega7e6/smoke_v10/coarse_p1_solution/data"),
        "status": "fallback",
        "note": "Smoke only. Coarse nonlinear predictor still falls back to secant after 10 coarse Newton iterations.",
    },
    {
        "name": "coarse_p1_reduced_newton",
        "label": "Coarse P1 Reduced Newton",
        "run_dir": Path("artifacts/p4_l1_smart_predictor_compare_omega7e6/smoke_v1/coarse_p1_reduced_newton/data"),
        "status": "failed",
        "note": "Smoke only. Predictor-stage MPI synchronization bug: rank 0 reached predictor bcast while another rank was still inside allgather.",
    },
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relpath(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, start=from_path.parent)


def _safe_ratio(num: float, denom: float) -> float:
    if not np.isfinite(float(num)) or not np.isfinite(float(denom)) or abs(float(denom)) <= 1.0e-30:
        return math.nan
    return float(num) / float(denom)


def _load_run(run_dir: Path) -> dict[str, object] | None:
    run_info_path = run_dir / "run_info.json"
    npz_path = run_dir / "petsc_run.npz"
    if not run_info_path.exists() or not npz_path.exists():
        return None
    return {
        "run_info": json.loads(run_info_path.read_text(encoding="utf-8")),
        "npz": np.load(npz_path, allow_pickle=True),
        "run_dir": run_dir,
    }


def _continuation_step_axis(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    if "stats_step_index" not in npz.files:
        return np.zeros(0, dtype=np.int64)
    step_index = np.asarray(npz["stats_step_index"], dtype=np.int64)
    if step_index.size == 0:
        return np.zeros(0, dtype=np.int64)
    return step_index - 2


def _npz_array(npz: np.lib.npyio.NpzFile, key: str, *, dtype=np.float64) -> np.ndarray:
    if key not in npz.files:
        return np.zeros(0, dtype=dtype)
    return np.asarray(npz[key], dtype=dtype)


def _extract_completed_summary(spec: dict[str, object], loaded: dict[str, object], baseline_runtime: float | None) -> dict[str, object]:
    run_info = loaded["run_info"]
    npz = loaded["npz"]
    timings = run_info["timings"]["linear"]
    predictor_diag = run_info.get("predictor_diagnostics", {})
    step_axis = _continuation_step_axis(npz)
    step_newton_total = _npz_array(npz, "stats_step_newton_iterations_total")
    step_linear_total = _npz_array(npz, "stats_step_linear_iterations")
    omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
    lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
    runtime = float(run_info["run_info"]["runtime_seconds"])
    final_omega = float(omega_hist[-1]) if omega_hist.size else math.nan
    final_lambda = float(lambda_hist[-1]) if lambda_hist.size else math.nan
    continuation_newton_total = float(np.nansum(step_newton_total))
    continuation_linear_total = float(np.nansum(step_linear_total))
    return {
        "name": str(spec["name"]),
        "label": str(spec["label"]),
        "status": "completed",
        "runtime": runtime,
        "speedup_vs_secant": _safe_ratio(float(baseline_runtime), runtime) if baseline_runtime is not None else 1.0,
        "final_lambda": final_lambda,
        "final_omega": final_omega,
        "accepted_continuation_steps": int(max(0, omega_hist.size - 2)),
        "continuation_newton_total": continuation_newton_total,
        "continuation_linear_total": continuation_linear_total,
        "continuation_linear_per_newton": _safe_ratio(continuation_linear_total, continuation_newton_total),
        "preconditioner_setup_total": float(timings.get("preconditioner_setup_time_total", 0.0)),
        "preconditioner_apply_total": float(timings.get("preconditioner_apply_time_total", 0.0)),
        "predictor_wall_time_total": float(predictor_diag.get("step_predictor_wall_time_total", 0.0)),
        "predictor_fallback_count": int(predictor_diag.get("step_predictor_fallback_count", 0)),
        "run_dir": str(loaded["run_dir"]),
        "step_axis": step_axis,
        "omega_hist": omega_hist,
        "lambda_hist": lambda_hist,
        "step_wall_time": _npz_array(npz, "stats_step_wall_time"),
        "step_newton_total": step_newton_total,
        "step_linear_total": step_linear_total,
        "step_linear_per_newton": np.divide(step_linear_total, np.maximum(step_newton_total, 1.0)),
        "step_u_diff": _npz_array(npz, "stats_step_initial_guess_displacement_diff_volume_integral"),
        "step_dev_diff": _npz_array(npz, "stats_step_initial_guess_deviatoric_strain_diff_volume_integral"),
        "step_predictor_wall": _npz_array(npz, "stats_step_predictor_wall_time"),
    }


def _extract_smoke_summary(spec: dict[str, object], loaded: dict[str, object] | None) -> dict[str, object]:
    summary: dict[str, object] = {
        "name": str(spec["name"]),
        "label": str(spec["label"]),
        "status": str(spec["status"]),
        "note": str(spec["note"]),
        "run_dir": str(spec["run_dir"]),
    }
    if loaded is None:
        summary["available"] = False
        return summary
    run_info = loaded["run_info"]
    npz = loaded["npz"]
    predictor_diag = run_info.get("predictor_diagnostics", {})
    summary["available"] = True
    summary["runtime"] = float(run_info["run_info"]["runtime_seconds"])
    summary["final_lambda"] = float(np.asarray(npz["lambda_hist"], dtype=np.float64)[-1])
    summary["final_omega"] = float(np.asarray(npz["omega_hist"], dtype=np.float64)[-1])
    summary["step_predictor_kind"] = _npz_array(npz, "stats_step_predictor_kind", dtype=object).tolist()
    summary["step_predictor_error"] = _npz_array(npz, "stats_step_predictor_error", dtype=object).tolist()
    summary["predictor_wall_time_total"] = float(predictor_diag.get("step_predictor_wall_time_total", 0.0))
    summary["predictor_fallback_count"] = int(predictor_diag.get("step_predictor_fallback_count", 0))
    summary["attempted_coarse_newton_iterations_total"] = float(
        predictor_diag.get("step_predictor_coarse_newton_iterations_total", 0.0)
    )
    summary["coarse_residual_last"] = predictor_diag.get("step_predictor_coarse_residual_last")
    return summary


def _extract_smoke_completed_summary(spec: dict[str, object], loaded: dict[str, object], baseline_runtime: float | None) -> dict[str, object]:
    run_info = loaded["run_info"]
    npz = loaded["npz"]
    timings = run_info["timings"]["linear"]
    predictor_diag = run_info.get("predictor_diagnostics", {})
    step_axis = _continuation_step_axis(npz)
    step_newton_total = _npz_array(npz, "stats_step_newton_iterations_total")
    step_linear_total = _npz_array(npz, "stats_step_linear_iterations")
    omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
    lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
    runtime = float(run_info["run_info"]["runtime_seconds"])
    final_omega = float(omega_hist[-1]) if omega_hist.size else math.nan
    final_lambda = float(lambda_hist[-1]) if lambda_hist.size else math.nan
    continuation_newton_total = float(np.nansum(step_newton_total))
    continuation_linear_total = float(np.nansum(step_linear_total))
    return {
        "name": str(spec["name"]),
        "label": str(spec["label"]),
        "status": "smoke_completed",
        "runtime": runtime,
        "speedup_vs_secant": _safe_ratio(float(baseline_runtime), runtime) if baseline_runtime is not None else 1.0,
        "final_lambda": final_lambda,
        "final_omega": final_omega,
        "accepted_continuation_steps": int(max(0, omega_hist.size - 2)),
        "continuation_newton_total": continuation_newton_total,
        "continuation_linear_total": continuation_linear_total,
        "continuation_linear_per_newton": _safe_ratio(continuation_linear_total, continuation_newton_total),
        "preconditioner_setup_total": float(timings.get("preconditioner_setup_time_total", 0.0)),
        "preconditioner_apply_total": float(timings.get("preconditioner_apply_time_total", 0.0)),
        "predictor_wall_time_total": float(predictor_diag.get("step_predictor_wall_time_total", 0.0)),
        "predictor_fallback_count": int(predictor_diag.get("step_predictor_fallback_count", 0)),
        "run_dir": str(loaded["run_dir"]),
        "step_axis": step_axis,
        "omega_hist": omega_hist,
        "lambda_hist": lambda_hist,
        "step_wall_time": _npz_array(npz, "stats_step_wall_time"),
        "step_newton_total": step_newton_total,
        "step_linear_total": step_linear_total,
        "step_linear_per_newton": np.divide(step_linear_total, np.maximum(step_newton_total, 1.0)),
        "step_u_diff": _npz_array(npz, "stats_step_initial_guess_displacement_diff_volume_integral"),
        "step_dev_diff": _npz_array(npz, "stats_step_initial_guess_deviatoric_strain_diff_volume_integral"),
        "step_predictor_wall": _npz_array(npz, "stats_step_predictor_wall_time"),
    }


def _plot_metric(*, completed: list[dict[str, object]], out_path: Path, y_key: str, ylabel: str, title: str) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=180)
    for summary in completed:
        x = np.asarray(summary["step_axis"], dtype=np.int64)
        y = np.asarray(summary[y_key], dtype=np.float64)
        if x.size == 0 or y.size == 0:
            continue
        plt.plot(x, y, marker="o", linewidth=1.6, label=str(summary["label"]))
    plt.xlabel("Accepted Continuation Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_lambda_omega(completed: list[dict[str, object]], out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=180)
    for summary in completed:
        plt.plot(summary["omega_hist"], summary["lambda_hist"], marker="o", linewidth=1.6, label=str(summary["label"]))
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\lambda$")
    plt.title(r"P4(L1) Predictor Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_final_lambda_runtime(completed: list[dict[str, object]], out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=180)
    for summary in completed:
        plt.scatter(summary["runtime"], summary["final_lambda"], s=90, label=str(summary["label"]))
        plt.annotate(str(summary["label"]), (summary["runtime"], summary["final_lambda"]), textcoords="offset points", xytext=(4, 4))
    plt.xscale("log")
    plt.xlabel("Runtime [s] (log scale)")
    plt.ylabel(r"Final $\lambda$")
    plt.title(r"Final $\lambda$ vs Runtime")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    out_dir = _ensure_dir(OUT_DIR)
    completed_specs = []
    completed_loaded = []
    for spec in FULL_CASES:
        loaded = _load_run(Path(spec["run_dir"]))
        if loaded is not None:
            completed_specs.append(spec)
            completed_loaded.append(loaded)

    baseline_runtime = None
    for spec, loaded in zip(completed_specs, completed_loaded, strict=False):
        if spec["name"] == "secant":
            baseline_runtime = float(loaded["run_info"]["run_info"]["runtime_seconds"])
            break

    completed = [
        _extract_completed_summary(spec, loaded, baseline_runtime=baseline_runtime)
        for spec, loaded in zip(completed_specs, completed_loaded, strict=False)
    ]
    smoke_prefix_specs = []
    smoke_prefix_loaded = []
    for spec in SMOKE_PREFIX_CASES:
        loaded = _load_run(Path(spec["run_dir"]))
        if loaded is not None:
            smoke_prefix_specs.append(spec)
            smoke_prefix_loaded.append(loaded)
    smoke_prefix_baseline_runtime = None
    for spec, loaded in zip(smoke_prefix_specs, smoke_prefix_loaded, strict=False):
        if spec["name"] == "secant":
            smoke_prefix_baseline_runtime = float(loaded["run_info"]["run_info"]["runtime_seconds"])
            break
    smoke_prefix = [
        _extract_smoke_completed_summary(spec, loaded, baseline_runtime=smoke_prefix_baseline_runtime)
        for spec, loaded in zip(smoke_prefix_specs, smoke_prefix_loaded, strict=False)
    ]
    smoke = [_extract_smoke_summary(spec, _load_run(Path(spec["run_dir"]))) for spec in SMOKE_CASES]

    plots_dir = _ensure_dir(out_dir / "plots")
    plots: list[tuple[Path, str]] = []
    plot_cases = completed if completed else smoke_prefix
    if plot_cases:
        lambda_omega_plot = plots_dir / "lambda_omega.png"
        step_wall_plot = plots_dir / "step_wall_time.png"
        step_newton_plot = plots_dir / "step_newton_iterations.png"
        step_linear_plot = plots_dir / "step_linear_iterations.png"
        step_linear_per_newton_plot = plots_dir / "step_linear_per_newton.png"
        step_u_diff_plot = plots_dir / "step_u_diff.png"
        step_dev_diff_plot = plots_dir / "step_dev_diff.png"
        step_predictor_wall_plot = plots_dir / "step_predictor_wall.png"
        final_lambda_runtime_plot = plots_dir / "final_lambda_vs_runtime.png"
        _plot_lambda_omega(plot_cases, lambda_omega_plot)
        _plot_metric(completed=plot_cases, out_path=step_wall_plot, y_key="step_wall_time", ylabel="Wall Time [s]", title="Accepted-Step Wall Time")
        _plot_metric(completed=plot_cases, out_path=step_newton_plot, y_key="step_newton_total", ylabel="Newton Iterations", title="Accepted-Step Newton Iterations")
        _plot_metric(completed=plot_cases, out_path=step_linear_plot, y_key="step_linear_total", ylabel="Linear Iterations", title="Accepted-Step Linear Iterations")
        _plot_metric(
            completed=plot_cases,
            out_path=step_linear_per_newton_plot,
            y_key="step_linear_per_newton",
            ylabel="Linear / Newton",
            title="Accepted-Step Linear per Newton",
        )
        _plot_metric(
            completed=plot_cases,
            out_path=step_u_diff_plot,
            y_key="step_u_diff",
            ylabel=r"$\int \|u_{\mathrm{newton}}-u_{\mathrm{init}}\|\,dV$",
            title="Accepted-Step Displacement Predictor Mismatch",
        )
        _plot_metric(
            completed=plot_cases,
            out_path=step_dev_diff_plot,
            y_key="step_dev_diff",
            ylabel=r"$\int \|dev(\varepsilon_{\mathrm{newton}}-\varepsilon_{\mathrm{init}})\|\,dV$",
            title="Accepted-Step Deviatoric-Strain Predictor Mismatch",
        )
        _plot_metric(
            completed=plot_cases,
            out_path=step_predictor_wall_plot,
            y_key="step_predictor_wall",
            ylabel="Predictor Wall Time [s]",
            title="Accepted-Step Predictor Wall Time",
        )
        _plot_final_lambda_runtime(plot_cases, final_lambda_runtime_plot)
        plots = [
            (lambda_omega_plot, "Lambda-Omega"),
            (step_wall_plot, "Accepted-Step Wall Time"),
            (step_newton_plot, "Accepted-Step Newton Iterations"),
            (step_linear_plot, "Accepted-Step Linear Iterations"),
            (step_linear_per_newton_plot, "Accepted-Step Linear per Newton"),
            (step_u_diff_plot, "Displacement Predictor Mismatch"),
            (step_dev_diff_plot, "Deviatoric Predictor Mismatch"),
            (step_predictor_wall_plot, "Predictor Wall Time"),
            (final_lambda_runtime_plot, "Final Lambda vs Runtime"),
        ]

    readme = out_dir / "README.md"
    lines: list[str] = []
    lines.append("# P4(L1) Smart-`d_omega` Predictor Comparison")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("- `secant`: full rank-8 rerun with saved `step_U`.")
    lines.append("- `secant_energy_alpha`: full rank-8 rerun with online energy-based alpha search.")
    lines.append("- `coarse_p1_solution`: smoke-only; coarse nonlinear predictor still falls back to secant.")
    lines.append("- `coarse_p1_reduced_newton`: smoke-only; predictor-stage MPI synchronization bug.")
    lines.append("")
    lines.append("## Completed Full Runs")
    lines.append("")
    if completed:
        lines.append("| Predictor | Runtime [s] | Speedup vs Secant | Final lambda | Final omega | Accepted continuation steps | Continuation Newton | Continuation Linear | Linear / Newton | PC Apply [s] | PC Setup [s] | Predictor Wall [s] | Fallbacks | Artifact |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for summary in completed:
            lines.append(
                f"| {summary['label']} | {summary['runtime']:.3f} | {summary['speedup_vs_secant']:.3f} | "
                f"{summary['final_lambda']:.9f} | {summary['final_omega']:.6e} | {summary['accepted_continuation_steps']} | "
                f"{summary['continuation_newton_total']:.0f} | {summary['continuation_linear_total']:.0f} | "
                f"{summary['continuation_linear_per_newton']:.3f} | {summary['preconditioner_apply_total']:.3f} | "
                f"{summary['preconditioner_setup_total']:.3f} | {summary['predictor_wall_time_total']:.3f} | "
                f"{summary['predictor_fallback_count']} | `{summary['run_dir']}` |"
            )
    else:
        lines.append("No completed full-run artifacts were available when this report was generated.")
    lines.append("")
    if smoke_prefix:
        lines.append("## Smoke Prefix Comparison")
        lines.append("")
        lines.append("| Predictor | Runtime [s] | Speedup vs Secant | Final lambda | Final omega | Accepted continuation steps | Continuation Newton | Continuation Linear | Linear / Newton | Predictor Wall [s] | Fallbacks | Artifact |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for summary in smoke_prefix:
            lines.append(
                f"| {summary['label']} | {summary['runtime']:.3f} | {summary['speedup_vs_secant']:.3f} | "
                f"{summary['final_lambda']:.9f} | {summary['final_omega']:.6e} | {summary['accepted_continuation_steps']} | "
                f"{summary['continuation_newton_total']:.0f} | {summary['continuation_linear_total']:.0f} | "
                f"{summary['continuation_linear_per_newton']:.3f} | {summary['predictor_wall_time_total']:.3f} | "
                f"{summary['predictor_fallback_count']} | `{summary['run_dir']}` |"
            )
        lines.append("")
    lines.append("## Smoke-Only Failed Variants")
    lines.append("")
    lines.append("| Predictor | Status | Runtime [s] | Final lambda | Final omega | Predictor kind | Predictor error | Predictor wall [s] | Fallbacks | Note |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |")
    for summary in smoke:
        if not summary.get("available", False):
            lines.append(
                f"| {summary['label']} | missing | - | - | - | - | - | - | - | {summary['note']} |"
            )
            continue
        kind = ", ".join(summary.get("step_predictor_kind", [])) or "-"
        error = ", ".join(summary.get("step_predictor_error", [])) or "-"
        lines.append(
            f"| {summary['label']} | {summary['status']} | {float(summary['runtime']):.3f} | "
            f"{float(summary['final_lambda']):.9f} | {float(summary['final_omega']):.6e} | {kind} | "
            f"{error} | {float(summary['predictor_wall_time_total']):.3f} | "
            f"{int(summary.get('predictor_fallback_count', 0))} | {summary['note']} |"
        )
    lines.append("")
    if plots:
        lines.append("## Plots")
        lines.append("")
        for plot_path, alt in plots:
            lines.append(f"![{alt}]({_relpath(readme, plot_path)})")
            lines.append("")
    readme.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "completed": completed,
        "smoke": smoke,
        "report": str(readme),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
