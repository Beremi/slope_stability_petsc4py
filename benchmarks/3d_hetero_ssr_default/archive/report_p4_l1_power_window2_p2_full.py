#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("artifacts/p4_l1_power_window2_p2_full/report")
CASE_SPECS = (
    {
        "name": "secant_baseline",
        "label": "Smart Secant Baseline",
        "progress": Path("artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_smart_controller_v2_rank8_step100/data/progress.jsonl"),
        "baseline": True,
    },
    {
        "name": "power_window2_p2",
        "label": "Power+Window Basis W2 P2",
        "progress": Path("artifacts/p4_l1_power_window2_p2_full/rank8_step100/data/progress.jsonl"),
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


def _nullable_float(value: object) -> float:
    if value is None:
        return math.nan
    return float(value)


def _load_progress(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _extract_case(spec: dict[str, object]) -> dict[str, object]:
    records = _load_progress(Path(spec["progress"]))
    init_record = next(obj for obj in records if obj.get("event") == "init_complete")
    step_records = [obj for obj in records if obj.get("event") == "step_accepted" and int(obj["accepted_step"]) >= 3]
    latest_record = records[-1]
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
        "step_predictor_wall_time": np.asarray(
            [float(obj.get("predictor_wall_time", math.nan)) if obj.get("predictor_wall_time") is not None else math.nan for obj in step_records],
            dtype=np.float64,
        ),
        "step_predictor_reduced_newton_iterations": np.asarray(
            [
                float(obj.get("predictor_reduced_newton_iterations", math.nan))
                if obj.get("predictor_reduced_newton_iterations") is not None
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
        "linear_per_newton": _safe_ratio(continuation_linear_total, continuation_newton_total),
        "predictor_wall_total": float(np.nansum([float(obj.get("predictor_wall_time", math.nan)) for obj in step_records])),
        "predictor_reduced_newton_total": float(
            np.nansum([_nullable_float(obj.get("predictor_reduced_newton_iterations", math.nan)) for obj in step_records])
        ),
        "final_lambda": float(step_records[-1]["lambda_value"]) if step_records else float(init_record["lambda_hist"][-1]),
        "final_omega": float(step_records[-1]["omega_value"]) if step_records else float(init_record["omega_hist"][-1]),
        "accepted_steps": int(step_records[-1]["accepted_step"]) if step_records else 2,
        "stop_reason": str(latest_record.get("stop_reason", "")),
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
    plt.title(r"P4(L1) Full Standard-Step Comparison to $\omega=7\times10^6$")
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
    plt.xlabel("Runtime [s] (log scale)")
    plt.ylabel(r"$\lambda$")
    plt.title(r"Final $\lambda$ vs Runtime")
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
    candidate = next(case for case in cases if not case["baseline"])

    lambda_omega_plot = plots_dir / "lambda_omega_full.png"
    step_wall_plot = plots_dir / "step_wall_time.png"
    step_newton_plot = plots_dir / "step_newton_iterations.png"
    step_linear_plot = plots_dir / "step_linear_iterations.png"
    step_linear_per_newton_plot = plots_dir / "step_linear_per_newton.png"
    predictor_wall_plot = plots_dir / "predictor_wall_time.png"
    predictor_reduced_newton_plot = plots_dir / "predictor_reduced_newton_iterations.png"
    cumulative_wall_plot = plots_dir / "cumulative_total_wall.png"
    final_lambda_runtime_plot = plots_dir / "final_lambda_vs_runtime.png"

    _plot_lambda_omega(cases=cases, out_path=lambda_omega_plot)
    _plot_metric(cases=cases, out_path=step_wall_plot, key="step_wall_time", ylabel="Wall Time [s]", title="Accepted-Step Wall Time")
    _plot_metric(cases=cases, out_path=step_newton_plot, key="step_newton_total", ylabel="Newton Iterations", title="Accepted-Step Newton Iterations")
    _plot_metric(cases=cases, out_path=step_linear_plot, key="step_linear_total", ylabel="Linear Iterations", title="Accepted-Step Linear Iterations")
    _plot_metric(cases=cases, out_path=step_linear_per_newton_plot, key="step_linear_per_newton", ylabel="Linear / Newton", title="Accepted-Step Linear per Newton")
    _plot_metric(cases=cases, out_path=predictor_wall_plot, key="step_predictor_wall_time", ylabel="Predictor Wall [s]", title="Accepted-Step Predictor Wall Time")
    _plot_metric(cases=cases, out_path=predictor_reduced_newton_plot, key="step_predictor_reduced_newton_iterations", ylabel="Reduced Newton Iterations", title="Accepted-Step Reduced Predictor Iterations")
    _plot_metric(cases=cases, out_path=cumulative_wall_plot, key="cumulative_total_wall", ylabel="Cumulative Total Wall Time [s]", title="Cumulative Runtime")
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
            "predictor_wall_total": case["predictor_wall_total"],
            "predictor_reduced_newton_total": case["predictor_reduced_newton_total"],
            "final_lambda": case["final_lambda"],
            "final_omega": case["final_omega"],
            "speedup_vs_baseline": _safe_ratio(baseline["prefix_runtime"], case["prefix_runtime"]) if case is not baseline else 1.0,
            "stop_reason": case["stop_reason"],
        }
        for case in cases
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    report_path = out_dir / "README.md"
    lines: list[str] = []
    lines.append("# P4(L1) Full Standard-Step Comparison")
    lines.append("")
    lines.append("This compares the archived smart-secant full run against the new mixed power+window predictor full run:")
    lines.append("- `Smart Secant Baseline` from the archived `p4_l1_smart_controller_v2_rank8_step100` artifact")
    lines.append("- `Power+Window Basis W2 P2` using the last two increments with `p=2`, current `lambda_i`, fixed-`U` lambda refinement, max 5 reduced iterations, partial use on nonconvergence")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Case | Accepted Steps | Runtime [s] | Init [s] | Accepted-Step Wall [s] | Extra Overhead [s] | Newton Total | Linear Total | Linear / Newton | Predictor Wall [s] | Reduced Newton Total | Final lambda | Final omega | Speedup vs Baseline | Stop Reason |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for case in cases:
        lines.append(
            f"| {case['label']} | {case['accepted_steps']} | {case['prefix_runtime']:.3f} | {case['init_runtime']:.3f} | {case['continuation_wall']:.3f} | {case['extra_overhead']:.3f} | "
            f"{case['continuation_newton_total']:.0f} | {case['continuation_linear_total']:.0f} | {case['linear_per_newton']:.3f} | "
            f"{case['predictor_wall_total']:.3f} | {case['predictor_reduced_newton_total']:.0f} | {case['final_lambda']:.9f} | {case['final_omega']:.6e} | "
            f"{(_safe_ratio(baseline['prefix_runtime'], case['prefix_runtime']) if case is not baseline else 1.0):.3f} | {case['stop_reason']} |"
        )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for label, path in [
        ("Lambda-Omega", lambda_omega_plot),
        ("Accepted-Step Wall Time", step_wall_plot),
        ("Accepted-Step Newton Iterations", step_newton_plot),
        ("Accepted-Step Linear Iterations", step_linear_plot),
        ("Accepted-Step Linear per Newton", step_linear_per_newton_plot),
        ("Accepted-Step Predictor Wall Time", predictor_wall_plot),
        ("Accepted-Step Reduced Predictor Iterations", predictor_reduced_newton_plot),
        ("Cumulative Runtime", cumulative_wall_plot),
        ("Final Lambda vs Runtime", final_lambda_runtime_plot),
    ]:
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"![{label}]({_relpath(report_path, path)})")
        lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for case in cases:
        lines.append(
            f"- {case['label']}: "
            f"[progress.jsonl]({_relpath(report_path, case['progress_path'])})"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
