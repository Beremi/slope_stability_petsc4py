#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ARTIFACT_ROOT = Path("artifacts/p4_l1_step5_alpha_compare")
CASE_PATHS = {
    "secant": ARTIFACT_ROOT / "rank8_secant_step7" / "data",
    "secant_then_alpha_on_step5": ARTIFACT_ROOT / "rank8_secant_switch5_alpha_step7" / "data",
}
TARGET_STEP = 7


def _load_case(path: Path) -> dict[str, object]:
    info = json.loads((path / "run_info.json").read_text(encoding="utf-8"))
    npz = np.load(path / "petsc_run.npz", allow_pickle=True)
    step_index = np.asarray(npz["stats_step_index"], dtype=np.int64)
    step_pos = list(map(int, step_index)).index(TARGET_STEP)
    progress_rows: list[dict[str, object]] = []
    with (path / "progress.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "newton_iteration" and int(row.get("target_step", -1)) == TARGET_STEP:
                progress_rows.append(row)
    return {
        "path": path,
        "run_info": info,
        "npz": npz,
        "step_pos": step_pos,
        "trace": progress_rows,
    }


def _step_metric(case: dict[str, object], key: str) -> float | str:
    npz = case["npz"]
    step_pos = int(case["step_pos"])
    value = npz[key][step_pos]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_summary(cases: dict[str, dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"target_step": TARGET_STEP, "cases": {}}
    for label, case in cases.items():
        run_info = case["run_info"]
        summary["cases"][label] = {
            "artifact_dir": str(case["path"]),
            "runtime_seconds": float(run_info["run_info"]["runtime_seconds"]),
            "step7_predictor_kind": str(_step_metric(case, "stats_step_predictor_kind")),
            "step7_predictor_alpha": float(_step_metric(case, "stats_step_predictor_alpha")),
            "step7_predictor_wall_time": float(_step_metric(case, "stats_step_predictor_wall_time")),
            "step7_predictor_energy_eval_count": float(_step_metric(case, "stats_step_predictor_energy_eval_count")),
            "step7_newton_iterations": int(_step_metric(case, "stats_step_newton_iterations")),
            "step7_linear_iterations": int(_step_metric(case, "stats_step_linear_iterations")),
            "step7_wall_time": float(_step_metric(case, "stats_step_wall_time")),
            "step7_linear_solve_time": float(_step_metric(case, "stats_step_linear_solve_time")),
            "step7_linear_preconditioner_time": float(_step_metric(case, "stats_step_linear_preconditioner_time")),
            "step7_linear_orthogonalization_time": float(_step_metric(case, "stats_step_linear_orthogonalization_time")),
            "step7_relres_end": float(_step_metric(case, "stats_step_newton_relres_end")),
            "step7_u_diff_volume_integral": float(_step_metric(case, "stats_step_initial_guess_displacement_diff_volume_integral")),
            "step7_dev_diff_volume_integral": float(_step_metric(case, "stats_step_initial_guess_deviatoric_strain_diff_volume_integral")),
            "step7_lambda_initial_guess_abs_error": float(_step_metric(case, "stats_step_lambda_initial_guess_abs_error")),
            "step7_lambda": float(_step_metric(case, "stats_step_lambda")),
            "step7_lambda_initial_guess": float(_step_metric(case, "stats_step_lambda_initial_guess")),
            "step7_omega": float(_step_metric(case, "stats_step_omega")),
        }
    sec = summary["cases"]["secant"]
    alpha = summary["cases"]["secant_then_alpha_on_step5"]
    summary["comparison"] = {
        "runtime_ratio_alpha_over_secant": float(alpha["runtime_seconds"] / sec["runtime_seconds"]),
        "step7_wall_ratio_alpha_over_secant": float(alpha["step7_wall_time"] / sec["step7_wall_time"]),
        "step7_newton_delta": int(alpha["step7_newton_iterations"] - sec["step7_newton_iterations"]),
        "step7_linear_delta": int(alpha["step7_linear_iterations"] - sec["step7_linear_iterations"]),
        "step7_u_diff_ratio_alpha_over_secant": float(alpha["step7_u_diff_volume_integral"] / sec["step7_u_diff_volume_integral"]),
        "step7_dev_diff_ratio_alpha_over_secant": float(alpha["step7_dev_diff_volume_integral"] / sec["step7_dev_diff_volume_integral"]),
    }
    return summary


def _plot_residual_trace(cases: dict[str, dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for label, case in cases.items():
        trace = case["trace"]
        x = [int(row["iteration"]) for row in trace]
        y = [float(row["rel_residual"]) for row in trace]
        ax.semilogy(x, y, marker="o", linewidth=2, label=label)
    ax.set_xlabel("Newton Iteration")
    ax.set_ylabel("Relative Residual")
    ax.set_title("Step 7 Newton Convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_cumulative_linear(cases: dict[str, dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for label, case in cases.items():
        cum = 0
        xs: list[int] = []
        ys: list[int] = []
        for row in case["trace"]:
            cum += int(row["linear_iterations"])
            xs.append(int(row["iteration"]))
            ys.append(cum)
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)
    ax.set_xlabel("Newton Iteration")
    ax.set_ylabel("Cumulative Linear Iterations")
    ax.set_title("Step 7 Linear Work Accumulation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_step7_bars(summary: dict[str, object], out_path: Path) -> None:
    labels = ["secant", "secant->alpha@5"]
    sec = summary["cases"]["secant"]
    alpha = summary["cases"]["secant_then_alpha_on_step5"]
    data = {
        "step_wall": [sec["step7_wall_time"], alpha["step7_wall_time"]],
        "solve": [sec["step7_linear_solve_time"], alpha["step7_linear_solve_time"]],
        "pc": [sec["step7_linear_preconditioner_time"], alpha["step7_linear_preconditioner_time"]],
        "orth": [sec["step7_linear_orthogonalization_time"], alpha["step7_linear_orthogonalization_time"]],
        "predictor": [sec["step7_predictor_wall_time"], alpha["step7_predictor_wall_time"]],
    }
    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for idx, (name, values) in enumerate(data.items()):
        ax.bar(x + (idx - 2) * width, values, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall Time [s]")
    ax.set_title("Step 7 Time Breakdown")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_report(summary: dict[str, object], out_path: Path) -> None:
    sec = summary["cases"]["secant"]
    alpha = summary["cases"]["secant_then_alpha_on_step5"]
    cmp = summary["comparison"]
    lines = [
        "# P4(L1) Step-7 Predictor Comparison",
        "",
        "This compares the 5th continuation step on rank 8 for the smart-`d_omega` `P4(L1)` PMG-shell run.",
        "Both runs use plain secant through the first four continuation steps; the second run switches to `secant_energy_alpha` only for step 7.",
        "",
        "## Outcome",
        "",
        f"- Target step: `{TARGET_STEP}`",
        f"- Target omega: `{sec['step7_omega']:.6f}`",
        f"- Plain secant: `{sec['step7_newton_iterations']}` Newton iterations, `{sec['step7_linear_iterations']}` linear iterations, `{sec['step7_wall_time']:.3f} s` step wall time",
        f"- Switched alpha: `{alpha['step7_newton_iterations']}` Newton iterations, `{alpha['step7_linear_iterations']}` linear iterations, `{alpha['step7_wall_time']:.3f} s` step wall time",
        f"- Chosen predictor alpha on step 7: `{alpha['step7_predictor_alpha']:.6f}` with `{int(alpha['step7_predictor_energy_eval_count'])}` energy evaluations",
        "",
        "The online energy search did not help here. It chose `alpha = 0.0`, produced a much worse initial guess mismatch, and needed one extra Newton iteration on the target step.",
        "",
        "## Step-7 Table",
        "",
        "| Metric | Plain secant | Secant then alpha on step 5 |",
        "| --- | ---: | ---: |",
        f"| Predictor kind | `{sec['step7_predictor_kind']}` | `{alpha['step7_predictor_kind']}` |",
        f"| Predictor alpha | `-` | `{alpha['step7_predictor_alpha']:.6f}` |",
        f"| Predictor wall time [s] | `{sec['step7_predictor_wall_time']:.3f}` | `{alpha['step7_predictor_wall_time']:.3f}` |",
        f"| Predictor energy evals | `-` | `{int(alpha['step7_predictor_energy_eval_count'])}` |",
        f"| Newton iterations | `{sec['step7_newton_iterations']}` | `{alpha['step7_newton_iterations']}` |",
        f"| Linear iterations | `{sec['step7_linear_iterations']}` | `{alpha['step7_linear_iterations']}` |",
        f"| Step wall time [s] | `{sec['step7_wall_time']:.3f}` | `{alpha['step7_wall_time']:.3f}` |",
        f"| Linear solve time [s] | `{sec['step7_linear_solve_time']:.3f}` | `{alpha['step7_linear_solve_time']:.3f}` |",
        f"| Preconditioner time [s] | `{sec['step7_linear_preconditioner_time']:.3f}` | `{alpha['step7_linear_preconditioner_time']:.3f}` |",
        f"| Orthogonalization time [s] | `{sec['step7_linear_orthogonalization_time']:.3f}` | `{alpha['step7_linear_orthogonalization_time']:.3f}` |",
        f"| Final Newton relative residual | `{sec['step7_relres_end']:.3e}` | `{alpha['step7_relres_end']:.3e}` |",
        f"| `∫ ||u_newton - u_init|| dV` | `{sec['step7_u_diff_volume_integral']:.3f}` | `{alpha['step7_u_diff_volume_integral']:.3f}` |",
        f"| `∫ ||dev(eps_newton - eps_init)|| dV` | `{sec['step7_dev_diff_volume_integral']:.3f}` | `{alpha['step7_dev_diff_volume_integral']:.3f}` |",
        f"| `|lambda_newton - lambda_init|` | `{sec['step7_lambda_initial_guess_abs_error']:.6f}` | `{alpha['step7_lambda_initial_guess_abs_error']:.6f}` |",
        "",
        "## Whole Run To Step 7",
        "",
        "| Metric | Plain secant | Secant then alpha on step 5 |",
        "| --- | ---: | ---: |",
        f"| Runtime [s] | `{sec['runtime_seconds']:.3f}` | `{alpha['runtime_seconds']:.3f}` |",
        f"| Runtime ratio alpha/secant | `1.000` | `{cmp['runtime_ratio_alpha_over_secant']:.3f}` |",
        "",
        "## Interpretation",
        "",
        f"- Step-7 wall time changed from `{sec['step7_wall_time']:.3f} s` to `{alpha['step7_wall_time']:.3f} s`, a ratio of `{cmp['step7_wall_ratio_alpha_over_secant']:.3f}`.",
        f"- Newton iterations changed by `{cmp['step7_newton_delta']:+d}` and linear iterations by `{cmp['step7_linear_delta']:+d}`.",
        f"- The displacement mismatch got worse by `{cmp['step7_u_diff_ratio_alpha_over_secant']:.2f}x` and the deviatoric mismatch by `{cmp['step7_dev_diff_ratio_alpha_over_secant']:.2f}x`.",
        "",
        "The failure mode is visible in the chosen alpha itself: the energy search collapsed to `alpha = 0.0`, so the predictor effectively abandoned the secant increment before the target-omega rescale. That made the initial guess much farther from the converged step-7 state.",
        "",
        "## Plots",
        "",
        "![Step 7 Newton Convergence](plots/step7_residual_trace.png)",
        "",
        "![Step 7 Cumulative Linear Iterations](plots/step7_cumulative_linear.png)",
        "",
        "![Step 7 Time Breakdown](plots/step7_time_breakdown.png)",
        "",
        "## Artifacts",
        "",
        f"- Plain secant: [{CASE_PATHS['secant']}](/home/beremi/repos/slope_stability-1/{CASE_PATHS['secant']})",
        f"- Switched alpha: [{CASE_PATHS['secant_then_alpha_on_step5']}](/home/beremi/repos/slope_stability-1/{CASE_PATHS['secant_then_alpha_on_step5']})",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cases = {label: _load_case(path) for label, path in CASE_PATHS.items()}
    summary = _build_summary(cases)
    report_dir = ARTIFACT_ROOT / "report"
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_residual_trace(cases, plots_dir / "step7_residual_trace.png")
    _plot_cumulative_linear(cases, plots_dir / "step7_cumulative_linear.png")
    _plot_step7_bars(summary, plots_dir / "step7_time_breakdown.png")
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(summary, report_dir / "README.md")


if __name__ == "__main__":
    main()
