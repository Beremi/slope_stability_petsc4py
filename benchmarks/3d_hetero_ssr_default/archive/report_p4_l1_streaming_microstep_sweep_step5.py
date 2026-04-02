#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("artifacts/p4_l1_streaming_microstep_sweep_step5_report")
PLOTS_DIR = OUT_DIR / "plots"
TARGET_OMEGA = 6243502.256218291

BASELINE_PROGRESS = Path("artifacts/p4_l1_halfstep_step30_compare/secant_halfstep_step30/data/progress.jsonl")
COMPLETED_STREAMING_CASES = (
    {
        "name": "streaming_default",
        "label": "Streaming Default",
        "progress": Path("artifacts/p4_l1_streaming_microstep_sweep_step5/default/data/progress.jsonl"),
    },
    {
        "name": "streaming_fixed060_thr5e3",
        "label": "Streaming Fixed 0.60, thr=5e-3",
        "progress": Path("artifacts/p4_l1_streaming_microstep_sweep_step5_targeted/fixed060_thr5e3/data/progress.jsonl"),
    },
    {
        "name": "streaming_fixed100_thr1e2",
        "label": "Streaming Fixed 1.00, thr=1e-2",
        "progress": Path("artifacts/p4_l1_streaming_microstep_sweep_step5_targeted/fixed100_thr1e2/data/progress.jsonl"),
    },
)
PRUNED_CASES = (
    {
        "label": "Streaming Fixed 0.30, thr=5e-3",
        "progress": Path("artifacts/p4_l1_streaming_microstep_sweep_step5/fixed030/data/progress.jsonl"),
        "reason": "pruned after early prefix looked dominated",
    },
    {
        "label": "Streaming Fixed 1.00, thr=5e-3",
        "progress": Path("artifacts/p4_l1_streaming_microstep_sweep_step5_targeted/fixed100_thr5e3/data/progress.jsonl"),
        "reason": "pruned after early prefix; looser 1e-2 variant was the better candidate",
    },
)


def _load_progress(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _find_baseline_steps_to_target(records: list[dict], target_omega: float) -> list[dict]:
    steps = [obj for obj in records if obj.get("event") == "step_accepted"]
    eligible = [obj for obj in steps if float(obj["omega_value"]) <= float(target_omega) + 1.0e-9]
    return eligible


def _extract_baseline_case(path: Path, target_omega: float) -> dict:
    records = _load_progress(path)
    steps = _find_baseline_steps_to_target(records, target_omega)
    final = steps[-1]
    return {
        "name": "secant_baseline",
        "label": "Secant Baseline",
        "records": records,
        "steps": steps,
        "attempts": [],
        "newton": [],
        "final_wall": float(final["total_wall_time"]),
        "final_omega": float(final["omega_value"]),
        "continuation_newton_total": float(sum(float(obj["step_newton_iterations_total"]) for obj in steps)),
        "continuation_linear_total": float(sum(float(obj["step_linear_iterations"]) for obj in steps)),
    }


def _extract_streaming_case(spec: dict) -> dict:
    records = _load_progress(spec["progress"])
    steps = [obj for obj in records if obj.get("event") == "step_accepted"]
    attempts = [obj for obj in records if obj.get("event") == "attempt_complete"]
    newton = [obj for obj in records if obj.get("event") == "newton_iteration"]
    final = steps[-1]
    return {
        "name": str(spec["name"]),
        "label": str(spec["label"]),
        "records": records,
        "steps": steps,
        "attempts": attempts,
        "newton": newton,
        "final_wall": float(final["total_wall_time"]),
        "final_omega": float(final["omega_value"]),
        "continuation_newton_total": float(sum(float(obj["step_newton_iterations_total"]) for obj in steps)),
        "continuation_linear_total": float(sum(float(obj["step_linear_iterations"]) for obj in steps)),
    }


def _extract_pruned_case(spec: dict) -> dict:
    records = _load_progress(spec["progress"])
    steps = [obj for obj in records if obj.get("event") == "step_accepted"]
    attempts = [obj for obj in records if obj.get("event") == "attempt_complete"]
    newton = [obj for obj in records if obj.get("event") == "newton_iteration"]
    payload = {
        "label": str(spec["label"]),
        "reason": str(spec["reason"]),
        "attempts": len(attempts),
        "accepted_points": len(steps),
    }
    if steps:
        final = steps[-1]
        payload.update(
            {
                "final_omega": float(final["omega_value"]),
                "total_wall": float(final["total_wall_time"]),
                "continuation_newton_total": float(sum(float(obj["step_newton_iterations_total"]) for obj in steps)),
                "continuation_linear_total": float(sum(float(obj["step_linear_iterations"]) for obj in steps)),
            }
        )
    if newton:
        payload["last_alpha"] = float(newton[-1].get("alpha", math.nan))
        payload["last_criterion"] = float(newton[-1].get("criterion", math.nan))
        payload["last_rel_residual"] = float(newton[-1].get("rel_residual", math.nan))
    return payload


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _plot_lambda_omega(cases: list[dict], baseline: dict) -> str:
    out_path = PLOTS_DIR / "accepted_lambda_vs_omega.png"
    fig = plt.figure(figsize=(8.2, 6.0), dpi=180)
    plt.plot(
        np.asarray([float(obj["omega_value"]) for obj in baseline["steps"]]) / 1.0e6,
        np.asarray([float(obj["lambda_value"]) for obj in baseline["steps"]]),
        marker="o",
        linewidth=2.0,
        label=baseline["label"],
    )
    for case in cases:
        plt.plot(
            np.asarray([float(obj["omega_value"]) for obj in case["steps"]]) / 1.0e6,
            np.asarray([float(obj["lambda_value"]) for obj in case["steps"]]),
            marker="o",
            linewidth=1.8,
            label=case["label"],
        )
    plt.xlabel(r"Accepted $\omega$ [$10^6$]")
    plt.ylabel(r"Accepted $\lambda$")
    plt.title(r"Accepted Continuation Points in the $(\omega,\lambda)$ Plane")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path.name


def _plot_cumulative_wall_vs_omega(cases: list[dict], baseline: dict) -> str:
    out_path = PLOTS_DIR / "accepted_cumulative_wall_vs_omega.png"
    fig = plt.figure(figsize=(8.2, 6.0), dpi=180)
    plt.plot(
        np.asarray([float(obj["omega_value"]) for obj in baseline["steps"]]) / 1.0e6,
        np.asarray([float(obj["total_wall_time"]) for obj in baseline["steps"]]),
        marker="o",
        linewidth=2.0,
        label=baseline["label"],
    )
    for case in cases:
        plt.plot(
            np.asarray([float(obj["omega_value"]) for obj in case["steps"]]) / 1.0e6,
            np.asarray([float(obj["total_wall_time"]) for obj in case["steps"]]),
            marker="o",
            linewidth=1.8,
            label=case["label"],
        )
    plt.xlabel(r"Accepted $\omega$ [$10^6$]")
    plt.ylabel("Cumulative Total Wall Time [s]")
    plt.title("Cumulative Runtime Along the Accepted Continuation Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path.name


def _plot_step_metrics(cases: list[dict], baseline: dict, field: str, ylabel: str, title: str, filename: str) -> str:
    out_path = PLOTS_DIR / filename
    fig = plt.figure(figsize=(8.2, 6.0), dpi=180)
    plt.plot(
        np.asarray([int(obj["accepted_step"]) for obj in baseline["steps"]]),
        np.asarray([float(obj[field]) for obj in baseline["steps"]]),
        marker="o",
        linewidth=2.0,
        label=baseline["label"],
    )
    for case in cases:
        plt.plot(
            np.asarray([int(obj["accepted_step"]) for obj in case["steps"]]),
            np.asarray([float(obj[field]) for obj in case["steps"]]),
            marker="o",
            linewidth=1.8,
            label=case["label"],
        )
    plt.xlabel("Accepted Continuation Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path.name


def _plot_attempt_metric(cases: list[dict], field: str, ylabel: str, title: str, filename: str, *, logy: bool = False) -> str:
    out_path = PLOTS_DIR / filename
    fig = plt.figure(figsize=(9.0, 6.0), dpi=180)
    for case in cases:
        vals = np.asarray([float(obj.get(field, math.nan)) for obj in case["attempts"]], dtype=np.float64)
        x = np.arange(1, vals.size + 1, dtype=np.int64)
        plt.plot(x, vals, marker="o", markersize=3.0, linewidth=1.4, label=case["label"])
    if logy:
        plt.yscale("log")
    plt.xlabel("Micro-Attempt Index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path.name


def _plot_newton_metric(cases: list[dict], field: str, ylabel: str, title: str, filename: str, *, logy: bool = False) -> str:
    out_path = PLOTS_DIR / filename
    fig = plt.figure(figsize=(9.0, 6.0), dpi=180)
    for case in cases:
        vals = np.asarray([float(obj.get(field, math.nan)) for obj in case["newton"]], dtype=np.float64)
        x = np.arange(1, vals.size + 1, dtype=np.int64)
        plt.plot(x, vals, marker="o", markersize=3.0, linewidth=1.4, label=case["label"])
    if logy:
        plt.yscale("log")
    plt.xlabel("Newton Micro-Iteration Index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path.name


def _plot_attempt_metric_vs_omega(cases: list[dict], field: str, ylabel: str, title: str, filename: str, *, logy: bool = False) -> str:
    out_path = PLOTS_DIR / filename
    fig = plt.figure(figsize=(9.0, 6.0), dpi=180)
    for case in cases:
        x = np.asarray([float(obj.get("omega_target", math.nan)) for obj in case["attempts"]], dtype=np.float64) / 1.0e6
        y = np.asarray([float(obj.get(field, math.nan)) for obj in case["attempts"]], dtype=np.float64)
        plt.plot(x, y, marker="o", markersize=3.0, linewidth=1.4, label=case["label"])
    if logy:
        plt.yscale("log")
    plt.xlabel(r"Attempt Target $\omega$ [$10^6$]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path.name


def _plot_micro_attempt_count(cases: list[dict]) -> str:
    out_path = PLOTS_DIR / "accepted_micro_attempt_count.png"
    fig = plt.figure(figsize=(9.0, 6.0), dpi=180)
    for case in cases:
        x = np.asarray([int(obj["accepted_step"]) for obj in case["steps"]], dtype=np.int64)
        y = np.asarray([float(obj["micro_attempt_count"]) for obj in case["steps"]], dtype=np.float64)
        plt.plot(x, y, marker="o", linewidth=1.8, label=case["label"])
    plt.xlabel("Accepted Continuation Step")
    plt.ylabel("Micro-Attempts per Accepted Step")
    plt.title("How Many Micro-Corrections Were Needed Before Acceptance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path.name


def _summary_table_row(case: dict, baseline: dict) -> str:
    return "| {} | {:.3f} | {:.0f} | {:.0f} | {:.6e} | {:.3f}x | {:.3f}x | {:.3f}x |".format(
        case["label"],
        case["final_wall"],
        case["continuation_newton_total"],
        case["continuation_linear_total"],
        case["final_omega"],
        case["final_wall"] / baseline["final_wall"],
        case["continuation_newton_total"] / baseline["continuation_newton_total"],
        case["continuation_linear_total"] / baseline["continuation_linear_total"],
    )


def main() -> None:
    _ensure_dirs()
    baseline = _extract_baseline_case(BASELINE_PROGRESS, TARGET_OMEGA)
    completed = [_extract_streaming_case(spec) for spec in COMPLETED_STREAMING_CASES]
    pruned = [_extract_pruned_case(spec) for spec in PRUNED_CASES]

    plots = {
        "lambda_omega": _plot_lambda_omega(completed, baseline),
        "cum_wall_omega": _plot_cumulative_wall_vs_omega(completed, baseline),
        "step_newton": _plot_step_metrics(
            completed,
            baseline,
            field="step_newton_iterations_total",
            ylabel="Newton Iterations",
            title="Accepted-Step Newton Iterations",
            filename="accepted_step_newton_iterations.png",
        ),
        "step_linear": _plot_step_metrics(
            completed,
            baseline,
            field="step_linear_iterations",
            ylabel="Linear Iterations",
            title="Accepted-Step Linear Iterations",
            filename="accepted_step_linear_iterations.png",
        ),
        "step_wall": _plot_step_metrics(
            completed,
            baseline,
            field="step_wall_time",
            ylabel="Wall Time [s]",
            title="Accepted-Step Wall Time",
            filename="accepted_step_wall_time.png",
        ),
        "attempt_relres_idx": _plot_attempt_metric(
            completed,
            field="newton_relres_end",
            ylabel="Residual Criterion",
            title="Residual at the End of Each Micro-Attempt",
            filename="attempt_relres_vs_attempt.png",
            logy=True,
        ),
        "attempt_relres_omega": _plot_attempt_metric_vs_omega(
            completed,
            field="newton_relres_end",
            ylabel="Residual Criterion",
            title=r"Residual at Each Micro-Attempt vs Target $\omega$",
            filename="attempt_relres_vs_omega.png",
            logy=True,
        ),
        "attempt_linear_idx": _plot_attempt_metric(
            completed,
            field="linear_iterations",
            ylabel="Linear Iterations",
            title="Linear Iterations per Micro-Attempt",
            filename="attempt_linear_iterations_vs_attempt.png",
        ),
        "attempt_wall_idx": _plot_attempt_metric(
            completed,
            field="attempt_wall_time",
            ylabel="Attempt Wall Time [s]",
            title="Wall Time per Micro-Attempt",
            filename="attempt_wall_time_vs_attempt.png",
        ),
        "attempt_arc_idx": _plot_attempt_metric(
            completed,
            field="micro_arc_length_accumulated",
            ylabel="Accumulated Normalized Arc Length",
            title="Accumulated Normalized Arc Length Inside the Current Accepted Step",
            filename="attempt_micro_arc_accumulated_vs_attempt.png",
        ),
        "attempt_target_idx": _plot_attempt_metric(
            completed,
            field="micro_target_length",
            ylabel="Target Normalized Arc Length",
            title="Adaptive Micro-Step Length Target",
            filename="attempt_micro_target_length_vs_attempt.png",
        ),
        "newton_alpha_idx": _plot_newton_metric(
            completed,
            field="alpha",
            ylabel=r"Damping $\alpha$",
            title="Damping Alpha for Each One-Step Newton Correction",
            filename="newton_alpha_vs_attempt.png",
        ),
        "newton_criterion_idx": _plot_newton_metric(
            completed,
            field="criterion",
            ylabel="Criterion",
            title="Newton Criterion During Streaming Micro-Corrections",
            filename="newton_criterion_vs_attempt.png",
            logy=True,
        ),
        "micro_attempt_count": _plot_micro_attempt_count(completed),
    }

    lines: list[str] = []
    lines.append("# Streaming Microstep Sweep to Fixed Omega 6.243502e6")
    lines.append("")
    lines.append("Target endpoint was matched to the archived secant half-step run at accepted step 5.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Case | Total wall [s] | Continuation Newton | Continuation linear | Final omega | Wall vs secant | Newton vs secant | Linear vs secant |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(_summary_table_row(baseline, baseline))
    for case in completed:
        lines.append(_summary_table_row(case, baseline))
    lines.append("")
    lines.append("Baseline secant is still the best on this fixed-endpoint test. The best completed streaming setting was `fixed 0.60`, but it still needed more continuation Newton work, more linear work, and more total wall time than secant.")
    lines.append("")
    lines.append("## Pruned Cases")
    lines.append("")
    lines.append("| Case | Accepted points reached | Attempts written | Final omega | Total wall [s] | Newton total | Linear total | Why pruned |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for case in pruned:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                case["label"],
                case.get("accepted_points", 0),
                case.get("attempts", 0),
                f"{case.get('final_omega', math.nan):.6e}" if "final_omega" in case else "n/a",
                f"{case.get('total_wall', math.nan):.3f}" if "total_wall" in case else "n/a",
                f"{case.get('continuation_newton_total', math.nan):.0f}" if "continuation_newton_total" in case else "n/a",
                f"{case.get('continuation_linear_total', math.nan):.0f}" if "continuation_linear_total" in case else "n/a",
                case["reason"],
            )
        )
    lines.append("")
    lines.append("## Figure Gallery")
    lines.append("")
    lines.append("### Accepted-Curve Plots")
    lines.append("")
    lines.append(f"![Accepted lambda vs omega](plots/{plots['lambda_omega']})")
    lines.append("")
    lines.append("Accepted continuation points in the `(omega, lambda)` plane. This is the cleanest high-level curve comparison.")
    lines.append("")
    lines.append(f"![Accepted cumulative wall vs omega](plots/{plots['cum_wall_omega']})")
    lines.append("")
    lines.append("Cumulative runtime as the accepted points move along the branch. This is the easiest plot for judging whether a variant is actually winning.")
    lines.append("")
    lines.append(f"![Accepted step Newton iterations](plots/{plots['step_newton']})")
    lines.append("")
    lines.append("Newton iterations per accepted continuation step.")
    lines.append("")
    lines.append(f"![Accepted step linear iterations](plots/{plots['step_linear']})")
    lines.append("")
    lines.append("Linear iterations per accepted continuation step.")
    lines.append("")
    lines.append(f"![Accepted step wall time](plots/{plots['step_wall']})")
    lines.append("")
    lines.append("Accepted-step wall time. This shows whether the smaller streaming moves are buying anything on the difficult accepted segments.")
    lines.append("")
    lines.append(f"![Accepted micro attempt count](plots/{plots['micro_attempt_count']})")
    lines.append("")
    lines.append("How many one-step micro-corrections were needed before the algorithm emitted the next accepted continuation point.")
    lines.append("")
    lines.append("### Micro-Attempt Convergence Plots")
    lines.append("")
    lines.append(f"![Attempt residual vs attempt index](plots/{plots['attempt_relres_idx']})")
    lines.append("")
    lines.append("Residual criterion at the end of each micro-attempt, on a log scale.")
    lines.append("")
    lines.append(f"![Attempt residual vs omega](plots/{plots['attempt_relres_omega']})")
    lines.append("")
    lines.append("Same residual criterion, but plotted against the target `omega` of each micro-attempt.")
    lines.append("")
    lines.append(f"![Attempt linear iterations vs attempt index](plots/{plots['attempt_linear_idx']})")
    lines.append("")
    lines.append("Linear iterations spent in each micro-attempt.")
    lines.append("")
    lines.append(f"![Attempt wall time vs attempt index](plots/{plots['attempt_wall_idx']})")
    lines.append("")
    lines.append("Wall time per micro-attempt.")
    lines.append("")
    lines.append(f"![Accumulated micro arc length vs attempt index](plots/{plots['attempt_arc_idx']})")
    lines.append("")
    lines.append("Accumulated normalized arc length inside the current accepted step. The sawtooth resets show where a continuation point was finally accepted.")
    lines.append("")
    lines.append(f"![Micro target length vs attempt index](plots/{plots['attempt_target_idx']})")
    lines.append("")
    lines.append("Adaptive micro-step target length. This reveals whether the algorithm is actually trusting larger continuation moves or shrinking them back down.")
    lines.append("")
    lines.append("### One-Step Newton Diagnostics")
    lines.append("")
    lines.append(f"![Newton alpha vs attempt index](plots/{plots['newton_alpha_idx']})")
    lines.append("")
    lines.append("Damping alpha for the single Newton correction used in each micro-attempt.")
    lines.append("")
    lines.append(f"![Newton criterion vs attempt index](plots/{plots['newton_criterion_idx']})")
    lines.append("")
    lines.append("Newton criterion for the one-step correction, on a log scale.")
    lines.append("")

    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
