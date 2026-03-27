#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MAX_ACCEPTED_STEP = 30
OUT_DIR = Path("artifacts/p4_l1_halfstep_step30_compare/report")
CASE_SPECS = (
    {
        "name": "secant_halfstep_step30",
        "label": "Secant, Half Step",
        "progress": Path("artifacts/p4_l1_halfstep_step30_compare/secant_halfstep_step30/data/progress.jsonl"),
        "baseline": True,
    },
    {
        "name": "affine_halfstep_step30",
        "label": "Affine States, Half Step",
        "progress": Path("artifacts/p4_l1_halfstep_step30_compare/affine_halfstep_step30/data/progress.jsonl"),
        "baseline": False,
    },
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relpath(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, start=from_path.parent)


def _safe_ratio(num: float, denom: float) -> float:
    if not np.isfinite(num) or not np.isfinite(denom) or abs(denom) <= 1.0e-30:
        return math.nan
    return num / denom


def _load_progress(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _extract_case(spec: dict[str, object]) -> dict[str, object]:
    records = _load_progress(Path(spec["progress"]))
    init_record = next(obj for obj in records if obj.get("event") == "init_complete")
    step_records = [
        obj
        for obj in records
        if obj.get("event") == "step_accepted" and 3 <= int(obj["accepted_step"]) <= MAX_ACCEPTED_STEP
    ]
    step_axis = np.asarray([int(obj["accepted_step"]) for obj in step_records], dtype=np.int64)
    lambda_hist = np.asarray(init_record["lambda_hist"] + [obj["lambda_value"] for obj in step_records], dtype=np.float64)
    omega_hist = np.asarray(init_record["omega_hist"] + [obj["omega_value"] for obj in step_records], dtype=np.float64)
    continuation_wall = float(sum(float(obj["step_wall_time"]) for obj in step_records))
    continuation_newton_total = float(sum(float(obj["step_newton_iterations_total"]) for obj in step_records))
    continuation_linear_total = float(sum(float(obj["step_linear_iterations"]) for obj in step_records))
    prefix_runtime = float(step_records[-1]["total_wall_time"]) if step_records else float(init_record["total_wall_time"])
    init_runtime = float(init_record["total_wall_time"])
    extra_overhead = prefix_runtime - init_runtime - continuation_wall
    return {
        "name": str(spec["name"]),
        "label": str(spec["label"]),
        "progress_path": Path(spec["progress"]),
        "baseline": bool(spec["baseline"]),
        "init": init_record,
        "steps": step_records,
        "step_axis": step_axis,
        "lambda_hist": lambda_hist,
        "omega_hist": omega_hist,
        "step_wall_time": np.asarray([float(obj["step_wall_time"]) for obj in step_records], dtype=np.float64),
        "step_newton_total": np.asarray([float(obj["step_newton_iterations_total"]) for obj in step_records], dtype=np.float64),
        "step_linear_total": np.asarray([float(obj["step_linear_iterations"]) for obj in step_records], dtype=np.float64),
        "step_linear_per_newton": np.asarray(
            [
                float(obj["step_linear_iterations"]) / max(float(obj["step_newton_iterations_total"]), 1.0)
                for obj in step_records
            ],
            dtype=np.float64,
        ),
        "cumulative_total_wall": np.asarray([float(obj["total_wall_time"]) for obj in step_records], dtype=np.float64),
        "prefix_runtime": prefix_runtime,
        "init_runtime": init_runtime,
        "continuation_wall": continuation_wall,
        "extra_overhead": extra_overhead,
        "continuation_newton_total": continuation_newton_total,
        "continuation_linear_total": continuation_linear_total,
        "linear_per_newton": _safe_ratio(continuation_linear_total, continuation_newton_total),
        "final_lambda": float(step_records[-1]["lambda_value"]) if step_records else float(init_record["lambda_hist"][-1]),
        "final_omega": float(step_records[-1]["omega_value"]) if step_records else float(init_record["omega_hist"][-1]),
        "accepted_steps": int(step_records[-1]["accepted_step"]) if step_records else 2,
    }


def _plot_metric(*, cases: list[dict[str, object]], out_path: Path, key: str, ylabel: str, title: str) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=180)
    for case in cases:
        if case["step_axis"].size == 0:
            continue
        plt.plot(case["step_axis"], case[key], marker="o", linewidth=1.6, label=str(case["label"]))
    plt.xlabel("Accepted Continuation Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_lambda_omega(*, cases: list[dict[str, object]], out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=180)
    for case in cases:
        plt.plot(case["omega_hist"], case["lambda_hist"], marker="o", linewidth=1.6, label=str(case["label"]))
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\lambda$")
    plt.title(r"P4(L1) Half-Step Prefix to Accepted Step 30")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_predictor_quality(*, cases: list[dict[str, object]], out_path: Path, key: str, secant_key: str, ylabel: str, title: str) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=180)
    for case in cases:
        if case["baseline"] or case["step_axis"].size == 0:
            continue
        x = np.asarray(case["step_axis"], dtype=np.int64)
        y = np.asarray([float(obj.get(key, math.nan)) for obj in case["steps"]], dtype=np.float64)
        y_sec = np.asarray([float(obj.get(secant_key, math.nan)) for obj in case["steps"]], dtype=np.float64)
        plt.plot(x, y, marker="o", linewidth=1.6, label=str(case["label"]))
        plt.plot(x, y_sec, marker="o", linewidth=1.2, linestyle="--", label=f"{case['label']} secant ref")
    plt.xlabel("Accepted Continuation Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_final_lambda_runtime(*, cases: list[dict[str, object]], out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=180)
    for case in cases:
        plt.scatter(case["prefix_runtime"], case["final_lambda"], s=90, label=str(case["label"]))
        plt.annotate(str(case["label"]), (case["prefix_runtime"], case["final_lambda"]), textcoords="offset points", xytext=(4, 4))
    plt.xscale("log")
    plt.xlabel("Runtime to last accepted step [s] (log scale)")
    plt.ylabel(r"$\lambda$")
    plt.title(r"Final Prefix $\lambda$ vs Runtime")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    out_dir = _ensure_dir(OUT_DIR)
    plots_dir = _ensure_dir(out_dir / "plots")

    cases = [_extract_case(spec) for spec in CASE_SPECS if Path(spec["progress"]).exists()]
    baseline = next(case for case in cases if case["baseline"])
    predictor_cases = [case for case in cases if not case["baseline"]]

    lambda_omega_plot = plots_dir / "lambda_omega_prefix.png"
    step_wall_plot = plots_dir / "step_wall_time.png"
    step_newton_plot = plots_dir / "step_newton_iterations.png"
    step_linear_plot = plots_dir / "step_linear_iterations.png"
    step_linear_per_newton_plot = plots_dir / "step_linear_per_newton.png"
    cumulative_wall_plot = plots_dir / "cumulative_total_wall.png"
    disp_mismatch_plot = plots_dir / "predictor_disp_vs_secant.png"
    dev_mismatch_plot = plots_dir / "predictor_dev_vs_secant.png"
    final_lambda_runtime_plot = plots_dir / "final_lambda_vs_runtime.png"

    _plot_lambda_omega(cases=cases, out_path=lambda_omega_plot)
    _plot_metric(cases=cases, out_path=step_wall_plot, key="step_wall_time", ylabel="Wall Time [s]", title="Accepted-Step Wall Time")
    _plot_metric(cases=cases, out_path=step_newton_plot, key="step_newton_total", ylabel="Newton Iterations", title="Accepted-Step Newton Iterations")
    _plot_metric(cases=cases, out_path=step_linear_plot, key="step_linear_total", ylabel="Linear Iterations", title="Accepted-Step Linear Iterations")
    _plot_metric(cases=cases, out_path=step_linear_per_newton_plot, key="step_linear_per_newton", ylabel="Linear / Newton", title="Accepted-Step Linear per Newton")
    _plot_metric(cases=cases, out_path=cumulative_wall_plot, key="cumulative_total_wall", ylabel="Cumulative Total Wall Time [s]", title="Cumulative Runtime Prefix")
    _plot_predictor_quality(
        cases=cases,
        out_path=disp_mismatch_plot,
        key="initial_guess_displacement_diff_volume_integral",
        secant_key="secant_reference_displacement_diff_volume_integral",
        ylabel=r"$\int \|u_{\mathrm{newton}}-u_{\mathrm{init}}\|\,dV$",
        title="Predictor Displacement Mismatch vs Secant Reference",
    )
    _plot_predictor_quality(
        cases=cases,
        out_path=dev_mismatch_plot,
        key="initial_guess_deviatoric_strain_diff_volume_integral",
        secant_key="secant_reference_deviatoric_strain_diff_volume_integral",
        ylabel=r"$\int \|dev(\varepsilon_{\mathrm{newton}}-\varepsilon_{\mathrm{init}})\|\,dV$",
        title="Predictor Deviatoric Mismatch vs Secant Reference",
    )
    _plot_final_lambda_runtime(cases=cases, out_path=final_lambda_runtime_plot)

    summary_payload = {
        case["name"]: {
            "accepted_steps": case["accepted_steps"],
            "prefix_runtime": case["prefix_runtime"],
            "init_runtime": case["init_runtime"],
            "continuation_wall": case["continuation_wall"],
            "extra_overhead": case["extra_overhead"],
            "continuation_newton_total": case["continuation_newton_total"],
            "continuation_linear_total": case["continuation_linear_total"],
            "linear_per_newton": case["linear_per_newton"],
            "final_lambda": case["final_lambda"],
            "final_omega": case["final_omega"],
            "speedup_vs_baseline": _safe_ratio(baseline["prefix_runtime"], case["prefix_runtime"]) if case is not baseline else 1.0,
        }
        for case in cases
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    report_path = out_dir / "README.md"
    lines: list[str] = []
    lines.append("# P4(L1) Half-Step Step-30 Comparison")
    lines.append("")
    lines.append("This report compares two fresh reruns with half initial continuation steps:")
    lines.append("- secant predictor")
    lines.append("- affine reduced Newton on previous continuation states with `sum c_j = 1`, using current fine `lambda_i`")
    lines.append("")
    lines.append(f"Each case is run to accepted continuation step cap `{MAX_ACCEPTED_STEP}`.")
    lines.append("")
    lines.append("## Prefix Summary")
    lines.append("")
    lines.append("| Case | Accepted Steps | Runtime [s] | Init [s] | Accepted-Step Wall [s] | Extra Overhead [s] | Newton Total | Linear Total | Linear / Newton | Final lambda | Final omega | Speedup vs Secant |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in cases:
        lines.append(
            f"| {case['label']} | {case['accepted_steps']} | {case['prefix_runtime']:.3f} | {case['init_runtime']:.3f} | {case['continuation_wall']:.3f} | {case['extra_overhead']:.3f} | "
            f"{case['continuation_newton_total']:.0f} | {case['continuation_linear_total']:.0f} | {case['linear_per_newton']:.3f} | "
            f"{case['final_lambda']:.9f} | {case['final_omega']:.6e} | "
            f"{(_safe_ratio(baseline['prefix_runtime'], case['prefix_runtime']) if case is not baseline else 1.0):.3f} |"
        )
    lines.append("")
    lines.append("`Extra Overhead` is `Runtime - Init - sum(accepted-step wall times)`. For the affine run this is mostly projected-predictor work and fallback work that happens before the fine Newton step timers start.")
    lines.append("")
    lines.append("## Step-Wise Comparison")
    lines.append("")
    lines.append("| Case | Step | Secant Wall [s] | Secant Newton | Secant Linear | Variant Wall [s] | Variant Newton | Variant Linear | Wall Ratio | Newton Ratio | Linear Ratio | Predictor Kind |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    baseline_by_step = {int(obj["accepted_step"]): obj for obj in baseline["steps"]}
    for case in predictor_cases:
        for obj in case["steps"]:
            step = int(obj["accepted_step"])
            if step not in baseline_by_step:
                continue
            base = baseline_by_step[step]
            lines.append(
                f"| {case['label']} | {step} | {float(base['step_wall_time']):.3f} | {int(base['step_newton_iterations_total'])} | {int(base['step_linear_iterations'])} | "
                f"{float(obj['step_wall_time']):.3f} | {int(obj['step_newton_iterations_total'])} | {int(obj['step_linear_iterations'])} | "
                f"{_safe_ratio(float(obj['step_wall_time']), float(base['step_wall_time'])):.3f} | "
                f"{_safe_ratio(float(obj['step_newton_iterations_total']), float(base['step_newton_iterations_total'])):.3f} | "
                f"{_safe_ratio(float(obj['step_linear_iterations']), float(base['step_linear_iterations'])):.3f} | {obj.get('predictor_kind', '')} |"
            )
    lines.append("")
    lines.append("## Predictor Quality vs Secant Reference")
    lines.append("")
    lines.append("| Case | Step | Predictor Kind | Predictor Error | `u_init -> u_final` | Secant Ref | Ratio | Dev Mismatch | Secant Ref Dev | Dev Ratio | Lambda Ini | Lambda Final |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in predictor_cases:
        for obj in case["steps"]:
            disp = float(obj.get("initial_guess_displacement_diff_volume_integral", math.nan))
            sec_disp = float(obj.get("secant_reference_displacement_diff_volume_integral", math.nan))
            dev = float(obj.get("initial_guess_deviatoric_strain_diff_volume_integral", math.nan))
            sec_dev = float(obj.get("secant_reference_deviatoric_strain_diff_volume_integral", math.nan))
            lines.append(
                f"| {case['label']} | {int(obj['accepted_step'])} | {obj.get('predictor_kind', '')} | "
                f"{obj.get('predictor_error', '')} | {disp:.3f} | {sec_disp:.3f} | {_safe_ratio(disp, sec_disp):.3f} | "
                f"{dev:.3f} | {sec_dev:.3f} | {_safe_ratio(dev, sec_dev):.3f} | {float(obj['lambda_initial_guess']):.9f} | {float(obj['lambda_value']):.9f} |"
            )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for case in cases:
        lines.append(f"- `{case['label']}` progress: [{case['progress_path']}]({case['progress_path'].resolve()})")
    lines.append(f"- Summary JSON: [{out_dir / 'summary.json'}]({(out_dir / 'summary.json').resolve()})")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for plot in (
        lambda_omega_plot,
        step_wall_plot,
        step_newton_plot,
        step_linear_plot,
        step_linear_per_newton_plot,
        cumulative_wall_plot,
        disp_mismatch_plot,
        dev_mismatch_plot,
        final_lambda_runtime_plot,
    ):
        lines.append(f"![{plot.stem}]({_relpath(report_path, plot)})")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
