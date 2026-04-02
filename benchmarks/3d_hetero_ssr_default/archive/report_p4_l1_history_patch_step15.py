#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MAX_ACCEPTED_STEP = 15
OUT_DIR = Path("artifacts/p4_l1_history_patch_step15/report")
CASE_SPECS = (
    {
        "name": "secant",
        "label": "Secant Baseline",
        "progress": Path("artifacts/p4_l1_halfstep_step30_compare/secant_halfstep_step30/data/progress.jsonl"),
        "baseline": True,
    },
    {
        "name": "secant_correction",
        "label": "Secant + Mini Correction",
        "progress": Path("artifacts/p4_l1_history_patch_step15/secant_correction/data/progress.jsonl"),
        "baseline": False,
    },
    {
        "name": "first_newton_warm_start",
        "label": "Secant + First-Newton Warm Start",
        "progress": Path("artifacts/p4_l1_history_patch_step15/first_newton_warm_start/data/progress.jsonl"),
        "baseline": False,
    },
    {
        "name": "both",
        "label": "Secant + Both",
        "progress": Path("artifacts/p4_l1_history_patch_step15/both/data/progress.jsonl"),
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
    first_newton_linear_total = float(sum(float(obj.get("first_newton_linear_iterations") or 0.0) for obj in step_records))
    first_newton_linear_solve_total = float(sum(float(obj.get("first_newton_linear_solve_time") or 0.0) for obj in step_records))
    predictor_wall_total = float(sum(float(obj.get("predictor_wall_time") or 0.0) for obj in step_records))
    return {
        "name": str(spec["name"]),
        "label": str(spec["label"]),
        "progress_path": Path(spec["progress"]),
        "baseline": bool(spec["baseline"]),
        "steps": step_records,
        "step_axis": step_axis,
        "lambda_hist": lambda_hist,
        "omega_hist": omega_hist,
        "step_wall_time": np.asarray([float(obj["step_wall_time"]) for obj in step_records], dtype=np.float64),
        "step_newton_total": np.asarray([float(obj["step_newton_iterations_total"]) for obj in step_records], dtype=np.float64),
        "step_linear_total": np.asarray([float(obj["step_linear_iterations"]) for obj in step_records], dtype=np.float64),
        "step_predictor_wall_time": np.asarray(
            [float(obj.get("predictor_wall_time", math.nan)) if obj.get("predictor_wall_time") is not None else math.nan for obj in step_records],
            dtype=np.float64,
        ),
        "step_first_newton_linear_iterations": np.asarray(
            [
                float(obj.get("first_newton_linear_iterations", math.nan))
                if obj.get("first_newton_linear_iterations") is not None
                else math.nan
                for obj in step_records
            ],
            dtype=np.float64,
        ),
        "step_first_newton_linear_solve_time": np.asarray(
            [
                float(obj.get("first_newton_linear_solve_time", math.nan))
                if obj.get("first_newton_linear_solve_time") is not None
                else math.nan
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
        "first_newton_linear_total": first_newton_linear_total,
        "first_newton_linear_solve_total": first_newton_linear_solve_total,
        "predictor_wall_total": predictor_wall_total,
        "final_lambda": float(step_records[-1]["lambda_value"]) if step_records else float(init_record["lambda_hist"][-1]),
        "final_omega": float(step_records[-1]["omega_value"]) if step_records else float(init_record["omega_hist"][-1]),
        "accepted_steps": int(step_records[-1]["accepted_step"]) if step_records else 2,
    }


def _value_at_step(case: dict[str, object], field: str, step: int) -> float:
    for obj in case["steps"]:
        if int(obj["accepted_step"]) == int(step):
            return float(obj[field])
    return math.nan


def _prefix_totals_to_step(case: dict[str, object], step: int) -> tuple[float, float, float]:
    runtime = math.nan
    newton = 0.0
    linear = 0.0
    for obj in case["steps"]:
        step_id = int(obj["accepted_step"])
        if step_id > int(step):
            break
        runtime = float(obj["total_wall_time"])
        newton += float(obj["step_newton_iterations_total"])
        linear += float(obj["step_linear_iterations"])
    return runtime, newton, linear


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
    plt.title(r"P4(L1) Half-Step Prefix to Accepted Step 15")
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
    baseline_by_step = {int(obj["accepted_step"]): obj for obj in baseline["steps"]}

    lambda_omega_plot = plots_dir / "lambda_omega_prefix.png"
    step_wall_plot = plots_dir / "step_wall_time.png"
    step_newton_plot = plots_dir / "step_newton_iterations.png"
    step_linear_plot = plots_dir / "step_linear_iterations.png"
    predictor_wall_plot = plots_dir / "predictor_wall_time.png"
    first_newton_linear_plot = plots_dir / "first_newton_linear_iterations.png"
    first_newton_solve_plot = plots_dir / "first_newton_linear_solve_time.png"
    cumulative_wall_plot = plots_dir / "cumulative_total_wall.png"

    _plot_lambda_omega(cases=cases, out_path=lambda_omega_plot)
    _plot_metric(cases=cases, out_path=step_wall_plot, key="step_wall_time", ylabel="Wall Time [s]", title="Accepted-Step Wall Time")
    _plot_metric(cases=cases, out_path=step_newton_plot, key="step_newton_total", ylabel="Newton Iterations", title="Accepted-Step Newton Iterations")
    _plot_metric(cases=cases, out_path=step_linear_plot, key="step_linear_total", ylabel="Linear Iterations", title="Accepted-Step Linear Iterations")
    _plot_metric(cases=cases, out_path=predictor_wall_plot, key="step_predictor_wall_time", ylabel="Predictor Wall [s]", title="Accepted-Step Predictor Wall Time")
    _plot_metric(
        cases=cases,
        out_path=first_newton_linear_plot,
        key="step_first_newton_linear_iterations",
        ylabel="Linear Iterations",
        title="First-Newton-Only Linear Iterations",
    )
    _plot_metric(
        cases=cases,
        out_path=first_newton_solve_plot,
        key="step_first_newton_linear_solve_time",
        ylabel="Linear Solve Time [s]",
        title="First-Newton-Only Linear Solve Time",
    )
    _plot_metric(cases=cases, out_path=cumulative_wall_plot, key="cumulative_total_wall", ylabel="Cumulative Total Wall Time [s]", title="Cumulative Runtime Prefix")

    summary_payload = {
        case["name"]: {
            "accepted_steps": case["accepted_steps"],
            "prefix_runtime": case["prefix_runtime"],
            "init_runtime": case["init_runtime"],
            "continuation_wall": case["continuation_wall"],
            "extra_overhead": case["extra_overhead"],
            "continuation_newton_total": case["continuation_newton_total"],
            "continuation_linear_total": case["continuation_linear_total"],
            "first_newton_linear_total": case["first_newton_linear_total"],
            "first_newton_linear_solve_total": case["first_newton_linear_solve_total"],
            "predictor_wall_total": case["predictor_wall_total"],
            "final_lambda": case["final_lambda"],
            "final_omega": case["final_omega"],
            "speedup_vs_baseline": _safe_ratio(baseline["prefix_runtime"], case["prefix_runtime"]) if case is not baseline else 1.0,
        }
        for case in cases
    }
    viable_candidates: list[dict[str, object]] = []
    for case in cases:
        compare_step = min(int(case["accepted_steps"]), int(baseline["accepted_steps"]))
        baseline_runtime_same_step, baseline_newton_same_step, baseline_linear_same_step = _prefix_totals_to_step(baseline, compare_step)
        case_runtime_same_step, case_newton_same_step, case_linear_same_step = _prefix_totals_to_step(case, compare_step)
        summary_payload[case["name"]]["compare_step"] = int(compare_step)
        summary_payload[case["name"]]["baseline_runtime_same_step"] = float(baseline_runtime_same_step)
        summary_payload[case["name"]]["runtime_same_step"] = float(case_runtime_same_step)
        summary_payload[case["name"]]["speedup_vs_baseline_same_step"] = _safe_ratio(
            float(baseline_runtime_same_step), float(case_runtime_same_step)
        )
        summary_payload[case["name"]]["baseline_newton_same_step"] = float(baseline_newton_same_step)
        summary_payload[case["name"]]["baseline_linear_same_step"] = float(baseline_linear_same_step)
        summary_payload[case["name"]]["newton_same_step"] = float(case_newton_same_step)
        summary_payload[case["name"]]["linear_same_step"] = float(case_linear_same_step)
        if case is not baseline and case_runtime_same_step < baseline_runtime_same_step:
            viable_candidates.append(case)
    summary_payload["winner"] = None if not viable_candidates else {
        "name": str(min(viable_candidates, key=lambda case: _prefix_totals_to_step(case, min(int(case["accepted_steps"]), int(baseline["accepted_steps"])))[0])["name"])
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    report_path = out_dir / "README.md"
    lines: list[str] = []
    lines.append("# P4(L1) Secant-History Patch Prefix Comparison")
    lines.append("")
    lines.append("This report compares the half-step prefix to accepted step 15 for the four secant-anchored history variants:")
    lines.append("- plain secant baseline")
    lines.append("- secant + mini orthogonal increment least-squares correction")
    lines.append("- secant + first-Newton history warm-start")
    lines.append("- secant + both")
    lines.append("")
    lines.append("## Prefix Summary")
    lines.append("")
    lines.append("| Case | Accepted Steps | Runtime [s] | Init [s] | Accepted-Step Wall [s] | Extra Overhead [s] | Newton Total | Linear Total | First-Newton Linear Total | First-Newton Solve [s] | Predictor Wall [s] | Final lambda | Final omega | Speedup vs Secant |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in cases:
        lines.append(
            f"| {case['label']} | {case['accepted_steps']} | {case['prefix_runtime']:.3f} | {case['init_runtime']:.3f} | "
            f"{case['continuation_wall']:.3f} | {case['extra_overhead']:.3f} | {case['continuation_newton_total']:.0f} | "
            f"{case['continuation_linear_total']:.0f} | {case['first_newton_linear_total']:.0f} | "
            f"{case['first_newton_linear_solve_total']:.3f} | {case['predictor_wall_total']:.3f} | "
            f"{case['final_lambda']:.9f} | {case['final_omega']:.6e} | "
            f"{(_safe_ratio(baseline['prefix_runtime'], case['prefix_runtime']) if case is not baseline else 1.0):.3f} |"
        )
    lines.append("")
    lines.append("## Matched Prefix Comparison")
    lines.append("")
    lines.append("| Case | Compare Step | Baseline Runtime [s] | Case Runtime [s] | Speedup vs Baseline | Baseline Newton | Case Newton | Baseline Linear | Case Linear |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in cases:
        compare_step = min(int(case["accepted_steps"]), int(baseline["accepted_steps"]))
        baseline_runtime_same_step, baseline_newton_same_step, baseline_linear_same_step = _prefix_totals_to_step(baseline, compare_step)
        case_runtime_same_step, case_newton_same_step, case_linear_same_step = _prefix_totals_to_step(case, compare_step)
        lines.append(
            f"| {case['label']} | {compare_step} | {baseline_runtime_same_step:.3f} | {case_runtime_same_step:.3f} | "
            f"{_safe_ratio(baseline_runtime_same_step, case_runtime_same_step):.3f} | {baseline_newton_same_step:.0f} | "
            f"{case_newton_same_step:.0f} | {baseline_linear_same_step:.0f} | {case_linear_same_step:.0f} |"
        )
    lines.append("")
    if viable_candidates:
        winner = min(
            viable_candidates,
            key=lambda case: _prefix_totals_to_step(case, min(int(case["accepted_steps"]), int(baseline["accepted_steps"])))[0],
        )
        lines.append(f"Best candidate on a matched prefix: `{winner['label']}`.")
    else:
        lines.append("No candidate beat the secant baseline on its matched accepted-step prefix.")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for plot in (
        lambda_omega_plot,
        step_wall_plot,
        step_newton_plot,
        step_linear_plot,
        predictor_wall_plot,
        first_newton_linear_plot,
        first_newton_solve_plot,
        cumulative_wall_plot,
    ):
        lines.append(f"![{plot.stem}]({_relpath(report_path, plot)})")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
