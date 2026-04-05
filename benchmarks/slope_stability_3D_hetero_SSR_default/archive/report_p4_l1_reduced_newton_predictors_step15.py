#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARTIFACT_ROOT = Path("artifacts/p4_l1_reduced_newton_predictors_step15")
BASELINE_RUN_DIR = Path("artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_smart_controller_v2_rank8_step100/data")
MAX_ACCEPTED_STEP = 15
CASES = (
    {
        "name": "secant",
        "label": "Secant",
        "run_dir": BASELINE_RUN_DIR,
        "progress_path": BASELINE_RUN_DIR / "progress.jsonl",
    },
    {
        "name": "reduced_newton_all_prev",
        "label": "Reduced Newton All Previous",
        "run_dir": ARTIFACT_ROOT / "reduced_newton_all_prev" / "data",
    },
    {
        "name": "reduced_newton_window3",
        "label": "Reduced Newton Window-3",
        "run_dir": ARTIFACT_ROOT / "reduced_newton_window3" / "data",
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


def _load_run(spec: dict[str, object]) -> dict[str, object]:
    run_dir = Path(spec["run_dir"])
    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    npz = np.load(run_dir / "petsc_run.npz", allow_pickle=True)
    progress_records: list[dict[str, object]] = []
    progress_path = spec.get("progress_path")
    if progress_path is not None:
        with Path(progress_path).open(encoding="utf-8") as handle:
            progress_records = [json.loads(line) for line in handle if line.strip()]
    return {"spec": spec, "run_dir": run_dir, "run_info": run_info, "npz": npz, "progress_records": progress_records}


def _npz_array(npz: np.lib.npyio.NpzFile, key: str, *, dtype=np.float64) -> np.ndarray:
    if key not in npz.files:
        return np.zeros(0, dtype=dtype)
    return np.asarray(npz[key], dtype=dtype)


def _step_axis(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    step_index = _npz_array(npz, "stats_step_index", dtype=np.int64)
    return step_index - 2 if step_index.size else np.zeros(0, dtype=np.int64)


def _step_mask(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    step_index = _npz_array(npz, "stats_step_index", dtype=np.int64)
    if step_index.size == 0:
        return np.zeros(0, dtype=bool)
    return step_index <= MAX_ACCEPTED_STEP


def _masked_array(npz: np.lib.npyio.NpzFile, key: str, *, mask: np.ndarray, dtype=np.float64) -> np.ndarray:
    arr = _npz_array(npz, key, dtype=dtype)
    if arr.size == 0 or mask.size == 0:
        return arr
    if arr.shape[0] != mask.shape[0]:
        return arr
    return arr[mask]


def _runtime_prefix_from_progress(progress_records: list[dict[str, object]]) -> float | None:
    if not progress_records:
        return None
    for record in progress_records:
        if record.get("event") == "step_accepted" and int(record.get("accepted_step", -1)) == MAX_ACCEPTED_STEP:
            return float(record.get("total_wall_time", math.nan))
    return None


def _extract_summary(run: dict[str, object], *, baseline_runtime: float | None = None) -> dict[str, object]:
    spec = run["spec"]
    run_info = run["run_info"]
    npz = run["npz"]
    progress_records = run.get("progress_records", [])
    timings = run_info["timings"]["linear"]
    predictor_diag = run_info.get("predictor_diagnostics", {})
    accepted_count = max(0, MAX_ACCEPTED_STEP - 2)
    omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)[: MAX_ACCEPTED_STEP + 1]
    lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)[: MAX_ACCEPTED_STEP + 1]
    mask = _step_mask(npz)
    step_axis = _step_axis(npz)
    if step_axis.size and mask.size == step_axis.size:
        step_axis = step_axis[mask]
    step_newton_total = _masked_array(npz, "stats_step_newton_iterations_total", mask=mask)
    step_linear_total = _masked_array(npz, "stats_step_linear_iterations", mask=mask)
    continuation_newton_total = float(np.nansum(step_newton_total))
    continuation_linear_total = float(np.nansum(step_linear_total))
    runtime = _runtime_prefix_from_progress(progress_records)
    if runtime is None:
        runtime = float(run_info["run_info"]["runtime_seconds"])
    final_omega = float(omega_hist[-1]) if omega_hist.size else math.nan
    final_lambda = float(lambda_hist[-1]) if lambda_hist.size else math.nan
    return {
        "name": str(spec["name"]),
        "label": str(spec["label"]),
        "runtime": runtime,
        "speedup_vs_secant": _safe_ratio(float(baseline_runtime), runtime) if baseline_runtime is not None else 1.0,
        "final_lambda": final_lambda,
        "final_omega": final_omega,
        "accepted_continuation_steps": min(accepted_count, int(max(0, omega_hist.size - 2))),
        "continuation_newton_total": continuation_newton_total,
        "continuation_linear_total": continuation_linear_total,
        "continuation_linear_per_newton": _safe_ratio(continuation_linear_total, continuation_newton_total),
        "preconditioner_setup_total": float(timings.get("preconditioner_setup_time_total", 0.0)),
        "preconditioner_apply_total": float(timings.get("preconditioner_apply_time_total", 0.0)),
        "linear_solve_total": float(timings.get("attempt_linear_solve_time_total", 0.0)),
        "linear_preconditioner_total": float(timings.get("attempt_linear_preconditioner_time_total", 0.0)),
        "linear_orthogonalization_total": float(timings.get("attempt_linear_orthogonalization_time_total", 0.0)),
        "predictor_wall_time_total": float(predictor_diag.get("step_predictor_wall_time_total", 0.0)),
        "predictor_fallback_count": int(predictor_diag.get("step_predictor_fallback_count", 0)),
        "run_dir": run["run_dir"],
        "step_axis": step_axis,
        "omega_hist": omega_hist,
        "lambda_hist": lambda_hist,
        "step_wall_time": _masked_array(npz, "stats_step_wall_time", mask=mask),
        "step_newton_total": step_newton_total,
        "step_linear_total": step_linear_total,
        "step_linear_per_newton": np.divide(step_linear_total, np.maximum(step_newton_total, 1.0)),
        "step_u_diff": _masked_array(npz, "stats_step_initial_guess_displacement_diff_volume_integral", mask=mask),
        "step_dev_diff": _masked_array(npz, "stats_step_initial_guess_deviatoric_strain_diff_volume_integral", mask=mask),
        "step_secant_u_diff": _masked_array(npz, "stats_step_secant_reference_displacement_diff_volume_integral", mask=mask),
        "step_secant_dev_diff": _masked_array(npz, "stats_step_secant_reference_deviatoric_strain_diff_volume_integral", mask=mask),
        "step_predictor_wall": _masked_array(npz, "stats_step_predictor_wall_time", mask=mask),
        "step_predictor_kind": _masked_array(npz, "stats_step_predictor_kind", mask=mask, dtype=object),
        "step_predictor_error": _masked_array(npz, "stats_step_predictor_error", mask=mask, dtype=object),
        "step_predictor_reduced_iterations": _masked_array(npz, "stats_step_predictor_reduced_newton_iterations", mask=mask),
        "step_predictor_projected_residual": _masked_array(npz, "stats_step_predictor_reduced_projected_residual", mask=mask),
        "step_lambda_guess_abs_error": _masked_array(npz, "stats_step_lambda_initial_guess_abs_error", mask=mask),
    }


def _plot_metric(*, summaries: list[dict[str, object]], out_path: Path, y_key: str, ylabel: str, title: str) -> None:
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
    plt.title(r"P4(L1) Reduced-Newton Predictor Comparison")
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


def _plot_predictor_vs_secant(
    *,
    summaries: list[dict[str, object]],
    out_path: Path,
    predictor_key: str,
    secant_key: str,
    ylabel: str,
    title: str,
) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=180)
    for summary in summaries:
        x = np.asarray(summary["step_axis"], dtype=np.int64)
        y = np.asarray(summary[predictor_key], dtype=np.float64)
        if x.size == 0 or y.size == 0:
            continue
        plt.plot(x, y, marker="o", linewidth=1.6, label=str(summary["label"]))
        if summary["name"] != "secant":
            y_sec = np.asarray(summary[secant_key], dtype=np.float64)
            if y_sec.size == x.size:
                plt.plot(x, y_sec, linestyle="--", linewidth=1.2, label=f"{summary['label']} secant ref")
    plt.xlabel("Accepted Continuation Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_report(*, out_dir: Path, summaries: list[dict[str, object]]) -> Path:
    plots_dir = _ensure_dir(out_dir / "plots")
    lambda_omega_plot = plots_dir / "lambda_omega.png"
    step_wall_plot = plots_dir / "step_wall_time.png"
    step_newton_plot = plots_dir / "step_newton_iterations.png"
    step_linear_plot = plots_dir / "step_linear_iterations.png"
    step_linear_per_newton_plot = plots_dir / "step_linear_per_newton.png"
    step_predictor_wall_plot = plots_dir / "step_predictor_wall.png"
    step_u_diff_plot = plots_dir / "step_u_diff_vs_secant.png"
    step_dev_diff_plot = plots_dir / "step_dev_diff_vs_secant.png"
    final_lambda_runtime_plot = plots_dir / "final_lambda_vs_runtime.png"

    _plot_lambda_omega(summaries, lambda_omega_plot)
    _plot_metric(summaries=summaries, out_path=step_wall_plot, y_key="step_wall_time", ylabel="Wall Time [s]", title="Accepted-Step Wall Time")
    _plot_metric(summaries=summaries, out_path=step_newton_plot, y_key="step_newton_total", ylabel="Newton Iterations", title="Accepted-Step Newton Iterations")
    _plot_metric(summaries=summaries, out_path=step_linear_plot, y_key="step_linear_total", ylabel="Linear Iterations", title="Accepted-Step Linear Iterations")
    _plot_metric(summaries=summaries, out_path=step_linear_per_newton_plot, y_key="step_linear_per_newton", ylabel="Linear / Newton", title="Accepted-Step Linear per Newton")
    _plot_metric(summaries=summaries, out_path=step_predictor_wall_plot, y_key="step_predictor_wall", ylabel="Predictor Wall Time [s]", title="Accepted-Step Predictor Wall Time")
    _plot_predictor_vs_secant(
        summaries=summaries,
        out_path=step_u_diff_plot,
        predictor_key="step_u_diff",
        secant_key="step_secant_u_diff",
        ylabel=r"$\int \|u_{\mathrm{newton}}-u_{\mathrm{init}}\|\,dV$",
        title="Displacement Predictor Mismatch vs Secant Reference",
    )
    _plot_predictor_vs_secant(
        summaries=summaries,
        out_path=step_dev_diff_plot,
        predictor_key="step_dev_diff",
        secant_key="step_secant_dev_diff",
        ylabel=r"$\int \|dev(\varepsilon_{\mathrm{newton}}-\varepsilon_{\mathrm{init}})\|\,dV$",
        title="Deviatoric-Strain Predictor Mismatch vs Secant Reference",
    )
    _plot_final_lambda_runtime(summaries, final_lambda_runtime_plot)

    report_path = out_dir / "README.md"
    lines: list[str] = []
    lines.append("# P4(L1) Reduced-Newton Predictor Comparison")
    lines.append("")
    lines.append("These runs compare the first 13 accepted continuation advances on rank 8 with the smart `d_omega` controller.")
    lines.append("The three cases are:")
    lines.append("- standard secant predictor")
    lines.append("- projected reduced Newton in the span of all previous continuation increments")
    lines.append("- projected reduced Newton in a sliding window of the last 3 continuation increments")
    lines.append("")
    lines.append("## End-to-End")
    lines.append("")
    lines.append("| Predictor | Runtime [s] | Speedup vs Secant | Final lambda | Final omega | Continuation steps | Continuation Newton | Continuation Linear | Linear / Newton | PC Apply [s] | PC Setup [s] | Predictor Wall [s] | Fallbacks |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in summaries:
        lines.append(
            f"| {summary['label']} | {summary['runtime']:.3f} | {summary['speedup_vs_secant']:.3f} | "
            f"{summary['final_lambda']:.9f} | {summary['final_omega']:.6e} | {summary['accepted_continuation_steps']} | "
            f"{summary['continuation_newton_total']:.0f} | {summary['continuation_linear_total']:.0f} | "
            f"{summary['continuation_linear_per_newton']:.3f} | {summary['preconditioner_apply_total']:.3f} | "
            f"{summary['preconditioner_setup_total']:.3f} | {summary['predictor_wall_time_total']:.3f} | "
            f"{summary['predictor_fallback_count']} |"
        )
    lines.append("")
    lines.append("## Predictor Quality")
    lines.append("")
    lines.append("| Predictor | Mean disp mismatch | Mean secant-ref disp mismatch | Mean dev mismatch | Mean secant-ref dev mismatch | Mean lambda guess abs err | Mean predictor wall [s] |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in summaries:
        step_u = np.asarray(summary["step_u_diff"], dtype=np.float64)
        step_sec_u = np.asarray(summary["step_secant_u_diff"], dtype=np.float64)
        step_dev = np.asarray(summary["step_dev_diff"], dtype=np.float64)
        step_sec_dev = np.asarray(summary["step_secant_dev_diff"], dtype=np.float64)
        step_lambda_err = np.asarray(summary["step_lambda_guess_abs_error"], dtype=np.float64)
        step_wall = np.asarray(summary["step_predictor_wall"], dtype=np.float64)
        lines.append(
            f"| {summary['label']} | {np.nanmean(step_u):.3f} | {np.nanmean(step_sec_u):.3f} | "
            f"{np.nanmean(step_dev):.3f} | {np.nanmean(step_sec_dev):.3f} | "
            f"{np.nanmean(step_lambda_err):.6f} | {np.nanmean(step_wall):.3f} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for summary in summaries:
        lines.append(f"- `{summary['label']}`: [{summary['run_dir']}](/home/beremi/repos/slope_stability-1/{summary['run_dir']})")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append(f"![Lambda Omega]({_relpath(report_path, lambda_omega_plot)})")
    lines.append("")
    lines.append(f"![Step Wall Time]({_relpath(report_path, step_wall_plot)})")
    lines.append("")
    lines.append(f"![Step Newton]({_relpath(report_path, step_newton_plot)})")
    lines.append("")
    lines.append(f"![Step Linear]({_relpath(report_path, step_linear_plot)})")
    lines.append("")
    lines.append(f"![Step Linear Per Newton]({_relpath(report_path, step_linear_per_newton_plot)})")
    lines.append("")
    lines.append(f"![Predictor Wall]({_relpath(report_path, step_predictor_wall_plot)})")
    lines.append("")
    lines.append(f"![Displacement Mismatch]({_relpath(report_path, step_u_diff_plot)})")
    lines.append("")
    lines.append(f"![Deviatoric Mismatch]({_relpath(report_path, step_dev_diff_plot)})")
    lines.append("")
    lines.append(f"![Final Lambda vs Runtime]({_relpath(report_path, final_lambda_runtime_plot)})")
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    out_dir = _ensure_dir(ARTIFACT_ROOT / "report")
    runs = [_load_run(spec) for spec in CASES]
    summaries: list[dict[str, object]] = []
    for idx, run in enumerate(runs):
        baseline_runtime = None if idx == 0 else float(summaries[0]["runtime"])
        summaries.append(_extract_summary(run, baseline_runtime=baseline_runtime))

    summary_payload = {
        "cases": {
            summary["name"]: {
                key: value
                for key, value in summary.items()
                if key
                not in {
                    "step_axis",
                    "omega_hist",
                    "lambda_hist",
                    "step_wall_time",
                    "step_newton_total",
                    "step_linear_total",
                    "step_linear_per_newton",
                    "step_u_diff",
                    "step_dev_diff",
                    "step_secant_u_diff",
                    "step_secant_dev_diff",
                    "step_predictor_wall",
                    "step_predictor_kind",
                    "step_predictor_error",
                    "step_predictor_reduced_iterations",
                    "step_predictor_projected_residual",
                    "step_lambda_guess_abs_error",
                }
            }
            for summary in summaries
        }
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    report_path = _write_report(out_dir=out_dir, summaries=summaries)
    print(report_path)


if __name__ == "__main__":
    main()
