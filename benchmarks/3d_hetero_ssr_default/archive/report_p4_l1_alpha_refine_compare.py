#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ARTIFACT_ROOT = Path("artifacts/p4_l1_alpha_refine_compare")
CASE_PATHS = {
    "secant": ARTIFACT_ROOT / "rank8_secant_step12" / "data",
    "secant_then_alpha_refine": ARTIFACT_ROOT / "rank8_secant_switch6_alpha_step12" / "data",
}
COMPARE_STEPS = [8, 9, 10, 11, 12]


def _load_case(path: Path) -> dict[str, object]:
    info = json.loads((path / "run_info.json").read_text(encoding="utf-8"))
    npz = np.load(path / "petsc_run.npz", allow_pickle=True)
    return {"path": path, "run_info": info, "npz": npz}


def _step_map(npz) -> dict[int, int]:
    return {int(step): idx for idx, step in enumerate(np.asarray(npz["stats_step_index"], dtype=np.int64))}


def _trace_rows(path: Path, target_step: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (path / "progress.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "newton_iteration" and int(row.get("target_step", -1)) == int(target_step):
                rows.append(row)
    return rows


def _case_step_metrics(case: dict[str, object], step: int) -> dict[str, float | str]:
    npz = case["npz"]
    idx = _step_map(npz)[int(step)]
    out: dict[str, float | str] = {
        "predictor_kind": str(npz["stats_step_predictor_kind"][idx]),
        "predictor_alpha": float(npz["stats_step_predictor_alpha"][idx]),
        "predictor_wall_time": float(npz["stats_step_predictor_wall_time"][idx]),
        "predictor_energy_eval_count": float(npz["stats_step_predictor_energy_eval_count"][idx]),
        "step_wall_time": float(npz["stats_step_wall_time"][idx]),
        "newton_iterations": int(npz["stats_step_newton_iterations"][idx]),
        "linear_iterations": int(npz["stats_step_linear_iterations"][idx]),
        "linear_solve_time": float(npz["stats_step_linear_solve_time"][idx]),
        "linear_preconditioner_time": float(npz["stats_step_linear_preconditioner_time"][idx]),
        "linear_orthogonalization_time": float(npz["stats_step_linear_orthogonalization_time"][idx]),
        "relres_end": float(npz["stats_step_newton_relres_end"][idx]),
        "u_diff": float(npz["stats_step_initial_guess_displacement_diff_volume_integral"][idx]),
        "dev_diff": float(npz["stats_step_initial_guess_deviatoric_strain_diff_volume_integral"][idx]),
        "lambda_init_abs_error": float(npz["stats_step_lambda_initial_guess_abs_error"][idx]),
        "lambda_value": float(npz["stats_step_lambda"][idx]),
        "omega_value": float(npz["stats_step_omega"][idx]),
    }
    return out


def _build_summary(cases: dict[str, dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "compare_steps": COMPARE_STEPS,
        "cases": {},
        "step_comparison": {},
    }
    for label, case in cases.items():
        info = case["run_info"]
        summary["cases"][label] = {
            "artifact_dir": str(case["path"]),
            "runtime_seconds": float(info["run_info"]["runtime_seconds"]),
            "step_count": int(info["run_info"]["step_count"]),
            "final_lambda": float(case["npz"]["lambda_hist"][-1]),
            "final_omega": float(case["npz"]["omega_hist"][-1]),
        }
    for step in COMPARE_STEPS:
        step_key = str(step)
        summary["step_comparison"][step_key] = {
            label: _case_step_metrics(case, step) for label, case in cases.items()
        }
    return summary


def _plot_lambda_omega(cases: dict[str, dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for label, case in cases.items():
        npz = case["npz"]
        ax.plot(np.asarray(npz["omega_hist"], dtype=np.float64), np.asarray(npz["lambda_hist"], dtype=np.float64), marker="o", linewidth=2, label=label)
    ax.set_xlabel("omega")
    ax.set_ylabel("lambda")
    ax.set_title("Lambda-Omega Path To Step 12")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_metric(cases: dict[str, dict[str, object]], metric: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for label, case in cases.items():
        npz = case["npz"]
        step_idx = np.asarray(npz["stats_step_index"], dtype=np.int64)
        values = np.asarray(npz[metric], dtype=np.float64)
        mask = step_idx >= COMPARE_STEPS[0]
        ax.plot(step_idx[mask], values[mask], marker="o", linewidth=2, label=label)
    ax.set_xlabel("Accepted Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_step12_convergence(cases: dict[str, dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for label, case in cases.items():
        trace = _trace_rows(case["path"], 12)
        ax.semilogy([int(r["iteration"]) for r in trace], [float(r["rel_residual"]) for r in trace], marker="o", linewidth=2, label=label)
    ax.set_xlabel("Newton Iteration")
    ax.set_ylabel("Relative Residual")
    ax.set_title("Step 12 Newton Convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_report(summary: dict[str, object], out_path: Path) -> None:
    sec = summary["cases"]["secant"]
    alt = summary["cases"]["secant_then_alpha_refine"]
    lines = [
        "# P4(L1) Alpha-Refine Comparison After Five Secant Steps",
        "",
        "Both runs use standard secant for the first five continuation steps.",
        "After that, one run stays on secant and the other switches to `secant_energy_alpha` with local refinement around the secant alpha.",
        "",
        "## Whole Run",
        "",
        "| Metric | Secant | Secant then alpha-refine |",
        "| --- | ---: | ---: |",
        f"| Runtime [s] | `{sec['runtime_seconds']:.3f}` | `{alt['runtime_seconds']:.3f}` |",
        f"| Final lambda | `{sec['final_lambda']:.9f}` | `{alt['final_lambda']:.9f}` |",
        f"| Final omega | `{sec['final_omega']:.3f}` | `{alt['final_omega']:.3f}` |",
        f"| Accepted steps | `{sec['step_count']}` | `{alt['step_count']}` |",
        "",
        "## Steps 8-12",
        "",
    ]
    for step in COMPARE_STEPS:
        sec_step = summary["step_comparison"][str(step)]["secant"]
        alt_step = summary["step_comparison"][str(step)]["secant_then_alpha_refine"]
        lines.extend(
            [
                f"### Step {step}",
                "",
                "| Metric | Secant | Secant then alpha-refine |",
                "| --- | ---: | ---: |",
                f"| Predictor kind | `{sec_step['predictor_kind']}` | `{alt_step['predictor_kind']}` |",
                f"| Predictor alpha | `{sec_step['predictor_alpha']:.6f}` | `{alt_step['predictor_alpha']:.6f}` |",
                f"| Predictor wall [s] | `{sec_step['predictor_wall_time']:.3f}` | `{alt_step['predictor_wall_time']:.3f}` |",
                f"| Energy evals | `-` | `{int(alt_step['predictor_energy_eval_count'])}` |",
                f"| Newton iterations | `{sec_step['newton_iterations']}` | `{alt_step['newton_iterations']}` |",
                f"| Linear iterations | `{sec_step['linear_iterations']}` | `{alt_step['linear_iterations']}` |",
                f"| Step wall [s] | `{sec_step['step_wall_time']:.3f}` | `{alt_step['step_wall_time']:.3f}` |",
                f"| Linear solve [s] | `{sec_step['linear_solve_time']:.3f}` | `{alt_step['linear_solve_time']:.3f}` |",
                f"| PC [s] | `{sec_step['linear_preconditioner_time']:.3f}` | `{alt_step['linear_preconditioner_time']:.3f}` |",
                f"| Orth [s] | `{sec_step['linear_orthogonalization_time']:.3f}` | `{alt_step['linear_orthogonalization_time']:.3f}` |",
                f"| Final relres | `{sec_step['relres_end']:.3e}` | `{alt_step['relres_end']:.3e}` |",
                f"| `∫ ||u_newton - u_init|| dV` | `{sec_step['u_diff']:.3f}` | `{alt_step['u_diff']:.3f}` |",
                f"| `∫ ||dev(eps_newton - eps_init)|| dV` | `{sec_step['dev_diff']:.3f}` | `{alt_step['dev_diff']:.3f}` |",
                f"| `|lambda_newton - lambda_init|` | `{sec_step['lambda_init_abs_error']:.6f}` | `{alt_step['lambda_init_abs_error']:.6f}` |",
                "",
            ]
        )
    lines.extend(
        [
            "## Plots",
            "",
            "![Lambda Omega](plots/lambda_omega.png)",
            "",
            "![Step Wall Time](plots/step_wall_time.png)",
            "",
            "![Newton Iterations](plots/newton_iterations.png)",
            "",
            "![Linear Iterations](plots/linear_iterations.png)",
            "",
            "![Predictor Alpha](plots/predictor_alpha.png)",
            "",
            "![Predictor Displacement Mismatch](plots/u_diff.png)",
            "",
            "![Step 12 Convergence](plots/step12_convergence.png)",
            "",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cases = {label: _load_case(path) for label, path in CASE_PATHS.items()}
    report_dir = ARTIFACT_ROOT / "report"
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_summary(cases)
    _plot_lambda_omega(cases, plots_dir / "lambda_omega.png")
    _plot_metric(cases, "stats_step_wall_time", "Wall Time [s]", "Accepted-Step Wall Time", plots_dir / "step_wall_time.png")
    _plot_metric(cases, "stats_step_newton_iterations", "Newton Iterations", "Accepted-Step Newton Iterations", plots_dir / "newton_iterations.png")
    _plot_metric(cases, "stats_step_linear_iterations", "Linear Iterations", "Accepted-Step Linear Iterations", plots_dir / "linear_iterations.png")
    _plot_metric(cases, "stats_step_predictor_alpha", "Predictor Alpha", "Recorded Predictor Alpha", plots_dir / "predictor_alpha.png")
    _plot_metric(cases, "stats_step_initial_guess_displacement_diff_volume_integral", "Volume Integral", "Predictor Displacement Mismatch", plots_dir / "u_diff.png")
    _plot_step12_convergence(cases, plots_dir / "step12_convergence.png")
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(summary, report_dir / "README.md")


if __name__ == "__main__":
    main()
