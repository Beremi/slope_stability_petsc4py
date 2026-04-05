#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from petsc4py import PETSc

from slope_stability.cli.run_3D_hetero_SSR_capture import run_capture


BASELINE_RUN_INFO = Path(
    "artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_smart_controller_v2_rank8_step100/data/run_info.json"
)
DEFAULT_OUT_DIR = Path("artifacts/p4_l1_smart_predictor_compare_omega7e6")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relpath(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, start=from_path.parent)


def _load_baseline_params() -> dict[str, object]:
    payload = json.loads(BASELINE_RUN_INFO.read_text(encoding="utf-8"))
    return dict(payload["params"])


def _case_specs() -> tuple[dict[str, str], ...]:
    return (
        {"name": "secant", "label": "Secant", "predictor": "secant"},
        {"name": "secant_energy_alpha", "label": "Secant Energy Alpha", "predictor": "secant_energy_alpha"},
        {"name": "coarse_p1_solution", "label": "Coarse P1 Solution", "predictor": "coarse_p1_solution"},
        {
            "name": "coarse_p1_reduced_newton",
            "label": "Coarse P1 Reduced Newton",
            "predictor": "coarse_p1_reduced_newton",
        },
    )


def _load_case(case_dir: Path) -> dict[str, object]:
    data_dir = case_dir / "data"
    run_info = json.loads((data_dir / "run_info.json").read_text(encoding="utf-8"))
    npz = np.load(data_dir / "petsc_run.npz", allow_pickle=True)
    return {"case_dir": case_dir, "run_info": run_info, "npz": npz}


def _npz_array(npz: np.lib.npyio.NpzFile, key: str, *, dtype=np.float64) -> np.ndarray:
    if key not in npz.files:
        return np.zeros(0, dtype=dtype)
    return np.asarray(npz[key], dtype=dtype)


def _continuation_step_axis(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    step_index = np.asarray(npz["stats_step_index"], dtype=np.int64)
    if step_index.size == 0:
        return np.zeros(0, dtype=np.int64)
    return step_index - 2


def _safe_ratio(num: float, denom: float) -> float:
    if abs(float(denom)) <= 1.0e-30:
        return math.nan
    return float(num) / float(denom)


def _extract_case_summary(case: dict[str, object], *, baseline_runtime: float | None) -> dict[str, object]:
    run_info = case["run_info"]
    npz = case["npz"]
    timings = run_info["timings"]["linear"]
    predictor_diag = run_info.get("predictor_diagnostics", {})
    omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
    lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
    step_axis = _continuation_step_axis(npz)
    step_newton_total = np.asarray(npz["stats_step_newton_iterations_total"], dtype=np.float64)
    step_linear_total = np.asarray(npz["stats_step_linear_iterations"], dtype=np.float64)
    continuation_newton_total = float(np.nansum(step_newton_total))
    continuation_linear_total = float(np.nansum(step_linear_total))
    runtime = float(run_info["run_info"]["runtime_seconds"])
    final_omega = float(omega_hist[-1]) if omega_hist.size else math.nan
    final_lambda = float(lambda_hist[-1]) if lambda_hist.size else math.nan
    return {
        "label": case["label"],
        "name": case["name"],
        "runtime": runtime,
        "speedup_vs_secant": _safe_ratio(float(baseline_runtime), runtime) if baseline_runtime is not None else 1.0,
        "final_lambda": final_lambda,
        "final_omega": final_omega,
        "accepted_continuation_steps": int(max(0, omega_hist.size - 2)),
        "continuation_newton_total": continuation_newton_total,
        "continuation_linear_total": continuation_linear_total,
        "continuation_linear_per_newton": _safe_ratio(continuation_linear_total, continuation_newton_total),
        "linear_solve_total": float(timings.get("attempt_linear_solve_time_total", 0.0)),
        "linear_preconditioner_total": float(timings.get("attempt_linear_preconditioner_time_total", 0.0)),
        "linear_orthogonalization_total": float(timings.get("attempt_linear_orthogonalization_time_total", 0.0)),
        "preconditioner_setup_total": float(timings.get("preconditioner_setup_time_total", 0.0)),
        "preconditioner_apply_total": float(timings.get("preconditioner_apply_time_total", 0.0)),
        "predictor_wall_time_total": float(predictor_diag.get("step_predictor_wall_time_total", 0.0)),
        "predictor_fallback_count": int(predictor_diag.get("step_predictor_fallback_count", 0)),
        "step_axis": step_axis,
        "omega_hist": omega_hist,
        "lambda_hist": lambda_hist,
        "step_wall_time": np.asarray(npz["stats_step_wall_time"], dtype=np.float64),
        "step_newton_total": step_newton_total,
        "step_linear_total": step_linear_total,
        "step_linear_per_newton": np.divide(
            step_linear_total,
            np.maximum(step_newton_total, 1.0),
        ),
        "step_u_diff": np.asarray(
            npz["stats_step_initial_guess_displacement_diff_volume_integral"],
            dtype=np.float64,
        ),
        "step_dev_diff": np.asarray(
            npz["stats_step_initial_guess_deviatoric_strain_diff_volume_integral"],
            dtype=np.float64,
        ),
        "step_predictor_wall": np.asarray(npz["stats_step_predictor_wall_time"], dtype=np.float64),
        "step_predictor_energy": _npz_array(npz, "stats_step_predictor_energy_value", dtype=np.float64),
        "step_lambda_guess_abs_error": _npz_array(npz, "stats_step_lambda_initial_guess_abs_error", dtype=np.float64),
        "step_predictor_kind": _npz_array(npz, "stats_step_predictor_kind", dtype=object),
        "stop_reason": str(run_info.get("run_info", {}).get("stop_reason", "")),
    }


def _plot_metric(
    *,
    summaries: list[dict[str, object]],
    out_path: Path,
    y_key: str,
    ylabel: str,
    title: str,
) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=180)
    for summary in summaries:
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


def _plot_lambda_omega(summaries: list[dict[str, object]], out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=180)
    for summary in summaries:
        plt.plot(summary["omega_hist"], summary["lambda_hist"], marker="o", linewidth=1.6, label=str(summary["label"]))
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\lambda$")
    plt.title(r"P4(L1) Smart-$d_\omega$ Predictor Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_final_lambda_runtime(summaries: list[dict[str, object]], out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=180)
    for summary in summaries:
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


def _write_report(
    *,
    out_dir: Path,
    cases: list[dict[str, object]],
    summaries: list[dict[str, object]],
) -> Path:
    plots_dir = _ensure_dir(out_dir / "plots")
    lambda_omega_plot = plots_dir / "lambda_omega.png"
    step_wall_plot = plots_dir / "step_wall_time.png"
    step_newton_plot = plots_dir / "step_newton_iterations.png"
    step_linear_plot = plots_dir / "step_linear_iterations.png"
    step_linear_per_newton_plot = plots_dir / "step_linear_per_newton.png"
    step_u_diff_plot = plots_dir / "step_u_diff.png"
    step_dev_diff_plot = plots_dir / "step_dev_diff.png"
    step_predictor_wall_plot = plots_dir / "step_predictor_wall.png"
    final_lambda_runtime_plot = plots_dir / "final_lambda_vs_runtime.png"

    _plot_lambda_omega(summaries, lambda_omega_plot)
    _plot_metric(summaries=summaries, out_path=step_wall_plot, y_key="step_wall_time", ylabel="Wall Time [s]", title="Accepted-Step Wall Time")
    _plot_metric(summaries=summaries, out_path=step_newton_plot, y_key="step_newton_total", ylabel="Newton Iterations", title="Accepted-Step Newton Iterations")
    _plot_metric(summaries=summaries, out_path=step_linear_plot, y_key="step_linear_total", ylabel="Linear Iterations", title="Accepted-Step Linear Iterations")
    _plot_metric(summaries=summaries, out_path=step_linear_per_newton_plot, y_key="step_linear_per_newton", ylabel="Linear / Newton", title="Accepted-Step Linear per Newton")
    _plot_metric(
        summaries=summaries,
        out_path=step_u_diff_plot,
        y_key="step_u_diff",
        ylabel=r"$\int \|u_{\mathrm{newton}}-u_{\mathrm{init}}\|\,dV$",
        title="Accepted-Step Displacement Predictor Mismatch",
    )
    _plot_metric(
        summaries=summaries,
        out_path=step_dev_diff_plot,
        y_key="step_dev_diff",
        ylabel=r"$\int \|dev(\varepsilon_{\mathrm{newton}}-\varepsilon_{\mathrm{init}})\|\,dV$",
        title="Accepted-Step Deviatoric-Strain Predictor Mismatch",
    )
    _plot_metric(
        summaries=summaries,
        out_path=step_predictor_wall_plot,
        y_key="step_predictor_wall",
        ylabel="Predictor Wall Time [s]",
        title="Accepted-Step Predictor Wall Time",
    )
    _plot_final_lambda_runtime(summaries, final_lambda_runtime_plot)

    report_path = out_dir / "README.md"
    lines: list[str] = []
    lines.append("# P4(L1) Smart-`d_omega` Predictor Comparison")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    baseline_runtime = float(summaries[0]["runtime"]) if summaries else math.nan
    for summary in summaries:
        lines.append(
            f"- `{summary['label']}`: runtime `{summary['runtime']:.3f} s`, "
            f"final `lambda={summary['final_lambda']:.9f}`, final `omega={summary['final_omega']:.6e}`, "
            f"accepted continuation steps `{summary['accepted_continuation_steps']}`, "
            f"continuation linear/Newton `{summary['continuation_linear_per_newton']:.3f}`"
        )
    lines.append("")
    lines.append("## End-to-End Comparison")
    lines.append("")
    lines.append("| Predictor | Runtime [s] | Speedup vs Secant | Final lambda | Final omega | Accepted continuation steps | Continuation Newton | Continuation Linear | Linear / Newton | PC Apply [s] | PC Setup [s] | Predictor Wall [s] | Fallbacks |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in summaries:
        lines.append(
            f"| {summary['label']} | {summary['runtime']:.3f} | {summary['speedup_vs_secant']:.3f} | "
            f"{summary['final_lambda']:.9f} | {summary['final_omega']:.6e} | {summary['accepted_continuation_steps']} | "
            f"{summary['continuation_newton_total']:.0f} | {summary['continuation_linear_total']:.0f} | "
            f"{summary['continuation_linear_per_newton']:.3f} | {summary['preconditioner_apply_total']:.3f} | "
            f"{summary['preconditioner_setup_total']:.3f} | "
            f"{summary['predictor_wall_time_total']:.3f} | {summary['predictor_fallback_count']} |"
        )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for plot_path, alt in (
        (lambda_omega_plot, "Lambda-Omega"),
        (step_wall_plot, "Accepted-Step Wall Time"),
        (step_newton_plot, "Accepted-Step Newton Iterations"),
        (step_linear_plot, "Accepted-Step Linear Iterations"),
        (step_linear_per_newton_plot, "Accepted-Step Linear per Newton"),
        (step_u_diff_plot, "Displacement Predictor Mismatch"),
        (step_dev_diff_plot, "Deviatoric Predictor Mismatch"),
        (step_predictor_wall_plot, "Predictor Wall Time"),
        (final_lambda_runtime_plot, "Final Lambda vs Runtime"),
    ):
        lines.append(f"![{alt}]({_relpath(report_path, plot_path)})")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare P4(L1) smart-controller predictor variants.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--step-max", type=int, default=100)
    parser.add_argument("--omega-max-stop", type=float, default=7.0e6)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(spec["name"] for spec in _case_specs()),
        help="Optional subset of predictor cases to run/report.",
    )
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    out_dir = _ensure_dir(Path(args.out_dir))
    cases = list(_case_specs())
    if args.cases:
        selected = set(args.cases)
        cases = [spec for spec in cases if spec["name"] in selected]
    base_params = _load_baseline_params()
    case_outputs: list[dict[str, object]] = []

    if not args.skip_run:
        for spec in cases:
            case_dir = out_dir / spec["name"]
            run_result = run_capture(
                case_dir,
                analysis="ssr",
                mesh_path=None,
                mesh_boundary_type=int(base_params["mesh_boundary_type"]),
                elem_type=str(base_params["elem_type"]),
                davis_type=str(base_params["davis_type"]),
                material_rows=base_params["material_rows"],
                node_ordering=str(base_params["node_ordering"]),
                lambda_init=float(base_params["lambda_init"]),
                d_lambda_init=float(base_params["d_lambda_init"]),
                d_lambda_min=float(base_params["d_lambda_min"]),
                d_lambda_diff_scaled_min=float(base_params["d_lambda_diff_scaled_min"]),
                omega_max_stop=float(args.omega_max_stop),
                continuation_predictor=str(spec["predictor"]),
                omega_no_increase_newton_threshold=base_params["omega_no_increase_newton_threshold"],
                omega_half_newton_threshold=base_params["omega_half_newton_threshold"],
                omega_target_newton_iterations=base_params["omega_target_newton_iterations"],
                omega_adapt_min_scale=base_params["omega_adapt_min_scale"],
                omega_adapt_max_scale=base_params["omega_adapt_max_scale"],
                omega_hard_newton_threshold=base_params["omega_hard_newton_threshold"],
                omega_hard_linear_threshold=base_params["omega_hard_linear_threshold"],
                omega_efficiency_floor=base_params["omega_efficiency_floor"],
                omega_efficiency_drop_ratio=base_params["omega_efficiency_drop_ratio"],
                omega_efficiency_window=int(base_params["omega_efficiency_window"]),
                omega_hard_shrink_scale=base_params["omega_hard_shrink_scale"],
                step_max=int(args.step_max),
                it_newt_max=int(base_params["it_newt_max"]),
                it_damp_max=int(base_params["it_damp_max"]),
                tol=float(base_params["tol"]),
                r_min=float(base_params["r_min"]),
                linear_tolerance=1.0e-1,
                linear_max_iter=100,
                solver_type="PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
                pc_backend="pmg_shell",
                preconditioner_matrix_source="tangent",
                preconditioner_matrix_policy=str(base_params["preconditioner_matrix_policy"]),
                preconditioner_rebuild_policy=str(base_params["preconditioner_rebuild_policy"]),
                preconditioner_rebuild_interval=int(base_params["preconditioner_rebuild_interval"]),
                mpi_distribute_by_nodes=bool(base_params["mpi_distribute_by_nodes"]),
                pc_hypre_coarsen_type=str(base_params["pc_hypre_coarsen_type"]),
                pc_hypre_interp_type=str(base_params["pc_hypre_interp_type"]),
                pc_hypre_strong_threshold=base_params["pc_hypre_strong_threshold"],
                pc_hypre_boomeramg_max_iter=base_params["pc_hypre_boomeramg_max_iter"],
                pc_hypre_P_max=base_params["pc_hypre_P_max"],
                pc_hypre_agg_nl=base_params["pc_hypre_agg_nl"],
                pc_hypre_nongalerkin_tol=base_params["pc_hypre_nongalerkin_tol"],
                petsc_opt=list(base_params["petsc_opt"]),
                compiled_outer=bool(base_params["compiled_outer"]),
                recycle_preconditioner=bool(base_params["recycle_preconditioner"]),
                constitutive_mode=str(base_params["constitutive_mode"]),
                tangent_kernel=str(base_params["tangent_kernel"]),
                store_step_u=True,
            )
            PETSc.COMM_WORLD.Barrier()
            if int(PETSc.COMM_WORLD.getRank()) == 0:
                case_outputs.append({"name": spec["name"], "label": spec["label"], "result": run_result})
    else:
        if int(PETSc.COMM_WORLD.getRank()) == 0:
            for spec in cases:
                case_outputs.append({"name": spec["name"], "label": spec["label"], "result": {"output": str(out_dir / spec["name"])}})

    if int(PETSc.COMM_WORLD.getRank()) != 0:
        return

    loaded_cases: list[dict[str, object]] = []
    for spec in cases:
        case_dir = out_dir / spec["name"]
        loaded = _load_case(case_dir)
        loaded["name"] = spec["name"]
        loaded["label"] = spec["label"]
        loaded_cases.append(loaded)
    baseline_runtime = float(loaded_cases[0]["run_info"]["run_info"]["runtime_seconds"]) if loaded_cases else None
    summaries = [_extract_case_summary(case, baseline_runtime=baseline_runtime) for case in loaded_cases]
    report_path = _write_report(out_dir=out_dir, cases=loaded_cases, summaries=summaries)
    summary_payload = {
        "cases": {
            summary["name"]: {
                "label": summary["label"],
                "runtime": summary["runtime"],
                "speedup_vs_secant": summary["speedup_vs_secant"],
                "final_lambda": summary["final_lambda"],
                "final_omega": summary["final_omega"],
                "accepted_continuation_steps": summary["accepted_continuation_steps"],
                "continuation_newton_total": summary["continuation_newton_total"],
                "continuation_linear_total": summary["continuation_linear_total"],
                "continuation_linear_per_newton": summary["continuation_linear_per_newton"],
                "preconditioner_setup_total": summary["preconditioner_setup_total"],
                "preconditioner_apply_total": summary["preconditioner_apply_total"],
                "predictor_wall_time_total": summary["predictor_wall_time_total"],
                "predictor_fallback_count": summary["predictor_fallback_count"],
            }
            for summary in summaries
        },
        "report": str(report_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
