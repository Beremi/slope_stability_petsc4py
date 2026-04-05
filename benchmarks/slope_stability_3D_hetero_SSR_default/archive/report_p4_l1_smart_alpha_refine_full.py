#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ARTIFACT_ROOT = Path("artifacts/p4_l1_smart_alpha_refine_full")
CASE_PATHS = {
    "smart_secant_baseline": Path("artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_smart_controller_v2_rank8_step100/data"),
    "smart_alpha_refine": ARTIFACT_ROOT / "rank8_step100" / "data",
}


def _load_case(path: Path) -> dict[str, object]:
    info = json.loads((path / "run_info.json").read_text(encoding="utf-8"))
    npz = np.load(path / "petsc_run.npz", allow_pickle=True)
    return {"path": path, "run_info": info, "npz": npz}


def _plot_series(cases: dict[str, dict[str, object]], metric: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for label, case in cases.items():
        npz = case["npz"]
        steps = np.asarray(npz["stats_step_index"], dtype=np.int64)
        vals = np.asarray(npz[metric], dtype=np.float64)
        ax.plot(steps, vals, marker="o", linewidth=2, label=label)
    ax.set_xlabel("Accepted Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_lambda_omega(cases: dict[str, dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for label, case in cases.items():
        npz = case["npz"]
        ax.plot(np.asarray(npz["omega_hist"], dtype=np.float64), np.asarray(npz["lambda_hist"], dtype=np.float64), marker="o", linewidth=2, label=label)
    ax.set_xlabel("omega")
    ax.set_ylabel("lambda")
    ax.set_title("Smart d_omega Full Run: Lambda-Omega")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_summary(cases: dict[str, dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"cases": {}}
    for label, case in cases.items():
        info = case["run_info"]
        npz = case["npz"]
        summary["cases"][label] = {
            "artifact_dir": str(case["path"]),
            "runtime_seconds": float(info["run_info"]["runtime_seconds"]),
            "step_count": int(info["run_info"]["step_count"]),
            "final_lambda": float(np.asarray(npz["lambda_hist"], dtype=np.float64)[-1]),
            "final_omega": float(np.asarray(npz["omega_hist"], dtype=np.float64)[-1]),
            "continuation_newton_total": int(np.nansum(np.asarray(npz["stats_step_newton_iterations_total"], dtype=np.float64))),
            "continuation_linear_total": int(np.nansum(np.asarray(npz["stats_step_linear_iterations"], dtype=np.float64))),
            "predictor_wall_total": float(np.nansum(np.asarray(npz["stats_step_predictor_wall_time"], dtype=np.float64))),
            "u_diff_total": float(np.nansum(np.asarray(npz["stats_step_initial_guess_displacement_diff_volume_integral"], dtype=np.float64))),
            "dev_diff_total": float(np.nansum(np.asarray(npz["stats_step_initial_guess_deviatoric_strain_diff_volume_integral"], dtype=np.float64))),
            "last_predictor_alpha": float(np.asarray(npz["stats_step_predictor_alpha"], dtype=np.float64)[-1]),
            "last_step_wall_time": float(np.asarray(npz["stats_step_wall_time"], dtype=np.float64)[-1]),
            "last_step_newton": int(np.asarray(npz["stats_step_newton_iterations"], dtype=np.int64)[-1]),
            "last_step_linear": int(np.asarray(npz["stats_step_linear_iterations"], dtype=np.int64)[-1]),
        }
    base = summary["cases"]["smart_secant_baseline"]
    alt = summary["cases"]["smart_alpha_refine"]
    summary["comparison"] = {
        "runtime_ratio_alpha_over_baseline": float(alt["runtime_seconds"] / base["runtime_seconds"]),
        "final_lambda_delta": float(alt["final_lambda"] - base["final_lambda"]),
        "continuation_newton_delta": int(alt["continuation_newton_total"] - base["continuation_newton_total"]),
        "continuation_linear_delta": int(alt["continuation_linear_total"] - base["continuation_linear_total"]),
    }
    return summary


def _write_report(summary: dict[str, object], out_path: Path) -> None:
    base = summary["cases"]["smart_secant_baseline"]
    alt = summary["cases"]["smart_alpha_refine"]
    cmp = summary["comparison"]
    lines = [
        "# P4(L1) Smart d_omega Full Run With Alpha Refine",
        "",
        "This compares the existing smart-secant full run against a full rank-8 run that uses the same smart `d_omega` controller and switches to the refined `secant_energy_alpha` predictor after the first five continuation steps.",
        "",
        "## Summary",
        "",
        "| Metric | Smart secant baseline | Smart alpha-refine |",
        "| --- | ---: | ---: |",
        f"| Runtime [s] | `{base['runtime_seconds']:.3f}` | `{alt['runtime_seconds']:.3f}` |",
        f"| Accepted steps | `{base['step_count']}` | `{alt['step_count']}` |",
        f"| Final lambda | `{base['final_lambda']:.9f}` | `{alt['final_lambda']:.9f}` |",
        f"| Final omega | `{base['final_omega']:.3f}` | `{alt['final_omega']:.3f}` |",
        f"| Continuation Newton total | `{base['continuation_newton_total']}` | `{alt['continuation_newton_total']}` |",
        f"| Continuation linear total | `{base['continuation_linear_total']}` | `{alt['continuation_linear_total']}` |",
        f"| Predictor wall total [s] | `{base['predictor_wall_total']:.3f}` | `{alt['predictor_wall_total']:.3f}` |",
        f"| Total `∫ ||u_newton - u_init|| dV` | `{base['u_diff_total']:.3f}` | `{alt['u_diff_total']:.3f}` |",
        f"| Total `∫ ||dev(eps_newton - eps_init)|| dV` | `{base['dev_diff_total']:.3f}` | `{alt['dev_diff_total']:.3f}` |",
        "",
        "## Comparison",
        "",
        f"- Runtime ratio alpha/baseline: `{cmp['runtime_ratio_alpha_over_baseline']:.3f}`",
        f"- Final lambda delta: `{cmp['final_lambda_delta']:+.6e}`",
        f"- Continuation Newton delta: `{cmp['continuation_newton_delta']:+d}`",
        f"- Continuation linear delta: `{cmp['continuation_linear_delta']:+d}`",
        "",
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
        "## Artifacts",
        "",
        f"- Baseline: [{CASE_PATHS['smart_secant_baseline']}](/home/beremi/repos/slope_stability-1/{CASE_PATHS['smart_secant_baseline']})",
        f"- Alpha refine: [{CASE_PATHS['smart_alpha_refine']}](/home/beremi/repos/slope_stability-1/{CASE_PATHS['smart_alpha_refine']})",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cases = {label: _load_case(path) for label, path in CASE_PATHS.items()}
    report_dir = ARTIFACT_ROOT / "report"
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_summary(cases)
    _plot_lambda_omega(cases, plots_dir / "lambda_omega.png")
    _plot_series(cases, "stats_step_wall_time", "Wall Time [s]", "Accepted-Step Wall Time", plots_dir / "step_wall_time.png")
    _plot_series(cases, "stats_step_newton_iterations", "Newton Iterations", "Accepted-Step Newton Iterations", plots_dir / "newton_iterations.png")
    _plot_series(cases, "stats_step_linear_iterations", "Linear Iterations", "Accepted-Step Linear Iterations", plots_dir / "linear_iterations.png")
    _plot_series(cases, "stats_step_predictor_alpha", "Predictor Alpha", "Recorded Predictor Alpha", plots_dir / "predictor_alpha.png")
    _plot_series(cases, "stats_step_initial_guess_displacement_diff_volume_integral", "Volume Integral", "Predictor Displacement Mismatch", plots_dir / "u_diff.png")
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(summary, report_dir / "README.md")


if __name__ == "__main__":
    main()
