#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/full_step_cap_variant_report"
PLOTS_DIR = OUT_DIR / "plots"

CASES = [
    {
        "key": "baseline_no_cap",
        "label": "Baseline Relative Gate, No Cap",
        "path": ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/full_rel_fixedomega_currentstate_omega7e6_corrected/data/progress.jsonl",
        "color": "#1f77b4",
    },
    {
        "key": "cap_sqrt2",
        "label": "Cap at Initial Segment Length",
        "path": ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/full_rel_fixedomega_currentstate_omega7e6_cap1p0/data/progress.jsonl",
        "color": "#d62728",
    },
    {
        "key": "cap_unit",
        "label": "Cap at Unit Length",
        "path": ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/full_rel_fixedomega_currentstate_omega7e6_capunit/data/progress.jsonl",
        "color": "#2ca02c",
    },
    {
        "key": "growth_1p25",
        "label": "No Cap, Gentler Growth x1.25",
        "path": ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/full_rel_fixedomega_currentstate_omega7e6_growth1p25/data/progress.jsonl",
        "color": "#9467bd",
    },
]


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_case(spec: dict[str, object]) -> dict[str, object]:
    path = Path(spec["path"])
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    init = next(r for r in records if r.get("event") == "init_complete")
    steps = [r for r in records if r.get("event") == "micro_step_complete"]
    final = next((r for r in reversed(records) if r.get("event") == "run_complete"), None)
    if final is None:
        last = steps[-1]
        final = {
            "event": "run_complete_partial",
            "stop_reason": "partial_log_no_run_complete",
            "final_lambda": float(last.get("committed_lambda_after", last.get("lambda_after", math.nan))),
            "final_omega": float(last.get("committed_omega_after", last.get("omega_after", math.nan))),
            "runtime_seconds": float(last.get("total_wall_time", math.nan)),
            "micro_steps_completed": int(len(steps)),
        }

    def _arr(field: str, default: float = math.nan) -> np.ndarray:
        values: list[float] = []
        for step in steps:
            value = step.get(field, default)
            if value is None:
                value = default
            values.append(float(value))
        return np.asarray(values, dtype=np.float64)

    omega = _arr("committed_omega_after")
    lam = _arr("committed_lambda_after")
    target_omega = float(init.get("omega_max_stop", 7.0e6))
    omega_start = float(init["omega_hist"][1])
    coverage = float((np.nanmax(omega) - omega_start) / max(target_omega - omega_start, 1.0e-12)) if omega.size else math.nan
    diagnostics_logged = any("linear_any_not_converged" in step for step in steps)
    return {
        "key": str(spec["key"]),
        "label": str(spec["label"]),
        "color": str(spec["color"]),
        "path": str(path),
        "init": init,
        "steps": steps,
        "final": final,
        "micro_idx": _arr("micro_index"),
        "omega": omega,
        "lambda": lam,
        "rel_after": _arr("rel_residual_after"),
        "alpha": _arr("alpha"),
        "linear_iterations": _arr("linear_iterations"),
        "total_wall": _arr("total_wall_time"),
        "domega_used": _arr("domega_used"),
        "correction_rel": _arr("correction_rel"),
        "correction_norm": _arr("correction_free_norm"),
        "step_length_cap_applied": _arr("step_length_cap_applied", default=0.0),
        "step_length_cap_length": _arr("step_length_cap_raw_length"),
        "omega_progressed": _arr("omega_progressed", default=0.0),
        "linear_any_not_converged": _arr("linear_any_not_converged", default=0.0),
        "linear_any_hit_max_iterations": _arr("linear_any_hit_max_iterations", default=0.0),
        "coverage": coverage,
        "max_lambda": float(np.nanmax(lam)) if lam.size else math.nan,
        "max_omega": float(np.nanmax(omega)) if omega.size else math.nan,
        "linear_total": int(np.nansum(_arr("linear_iterations", default=0.0))),
        "zero_domega_steps": int(np.sum(np.abs(_arr("domega_used", default=0.0)) <= 1.0e-12)),
        "cap_applied_count": int(np.sum(_arr("step_length_cap_applied", default=0.0) > 0.5)),
        "not_converged_count": int(np.sum(_arr("linear_any_not_converged", default=0.0) > 0.5)),
        "hit_max_count": int(np.sum(_arr("linear_any_hit_max_iterations", default=0.0) > 0.5)),
        "diagnostics_logged": bool(diagnostics_logged),
    }


def _save_plot(fig: plt.Figure, filename: str) -> str:
    out = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return filename


def _plot_series(
    cases: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    logy: bool = False,
    hlines: list[tuple[float, str, str]] | None = None,
) -> str:
    fig = plt.figure(figsize=(8.6, 5.7), dpi=180)
    for case in cases:
        x = np.asarray(case[x_key], dtype=np.float64)
        y = np.asarray(case[y_key], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        plt.plot(
            x[finite],
            y[finite],
            marker="o",
            markersize=3.0,
            linewidth=1.5,
            color=str(case["color"]),
            label=str(case["label"]),
        )
    if hlines:
        used: set[str] = set()
        for y, label, style in hlines:
            lab = label if label not in used else "_nolegend_"
            plt.axhline(float(y), color="0.25", linestyle=style, linewidth=1.0, alpha=0.8, label=lab)
            used.add(label)
    if logy:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    return _save_plot(fig, filename)


def _plot_lambda_vs_omega(cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(8.6, 5.7), dpi=180)
    for case in cases:
        plt.plot(
            np.asarray(case["omega"], dtype=np.float64) / 1.0e6,
            np.asarray(case["lambda"], dtype=np.float64),
            marker="o",
            markersize=3.0,
            linewidth=1.5,
            color=str(case["color"]),
            label=str(case["label"]),
        )
    plt.xlabel(r"Committed $\omega$ [$10^6$]")
    plt.ylabel(r"Committed $\lambda$")
    plt.title(r"Full Micro-Walk Trajectories in the $(\omega,\lambda)$ Plane")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return _save_plot(fig, "lambda_vs_omega.png")


def _plot_cap_lengths(cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(8.6, 5.7), dpi=180)
    for case in cases:
        if "Cap" not in str(case["label"]):
            continue
        x = np.asarray(case["micro_idx"], dtype=np.float64)
        y = np.asarray(case["step_length_cap_length"], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        plt.plot(
            x[finite],
            y[finite],
            marker="o",
            markersize=3.0,
            linewidth=1.5,
            color=str(case["color"]),
            label=str(case["label"]),
        )
    plt.axhline(math.sqrt(2.0), color="0.25", linestyle="--", linewidth=1.0, alpha=0.8, label=r"$\sqrt{2}$ limit")
    plt.axhline(1.0, color="0.25", linestyle=":", linewidth=1.0, alpha=0.8, label="1.0 limit")
    plt.xlabel("Micro-Step Index")
    plt.ylabel("Normalized Proposed Step Length")
    plt.title("Normalized Step Length Under Cap Variants")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return _save_plot(fig, "cap_length_vs_step.png")


def _build_summary(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cases": [
            {
                "key": case["key"],
                "label": case["label"],
                "path": case["path"],
                "max_omega": case["max_omega"],
                "max_lambda": case["max_lambda"],
                "coverage": case["coverage"],
                "linear_total": case["linear_total"],
                "zero_domega_steps": case["zero_domega_steps"],
                "cap_applied_count": case["cap_applied_count"],
                "not_converged_count": case["not_converged_count"],
                "hit_max_count": case["hit_max_count"],
                "diagnostics_logged": case["diagnostics_logged"],
                "final": case["final"],
            }
            for case in cases
        ]
    }


def main() -> None:
    _ensure_dirs()
    cases = [_load_case(spec) for spec in CASES]
    plots = {
        "lambda_vs_omega": _plot_lambda_vs_omega(cases),
        "omega_vs_step": _plot_series(
            cases,
            x_key="micro_idx",
            y_key="omega",
            title="Committed Omega by Micro-Step",
            xlabel="Micro-Step Index",
            ylabel=r"Committed $\omega$",
            filename="omega_vs_step.png",
        ),
        "lambda_vs_step": _plot_series(
            cases,
            x_key="micro_idx",
            y_key="lambda",
            title="Committed Lambda by Micro-Step",
            xlabel="Micro-Step Index",
            ylabel=r"Committed $\lambda$",
            filename="lambda_vs_step.png",
        ),
        "rel_after": _plot_series(
            cases,
            x_key="micro_idx",
            y_key="rel_after",
            title="Outer-Step Relative Residual After Each Micro-Step",
            xlabel="Micro-Step Index",
            ylabel="Relative Residual After",
            filename="rel_after_vs_step.png",
            logy=True,
            hlines=[(1.0e-1, "1e-1", ":"), (1.0, "1", "--"), (1.0e1, "1e1", "-.")],
        ),
        "linear_iterations": _plot_series(
            cases,
            x_key="micro_idx",
            y_key="linear_iterations",
            title="Linear Iterations per Micro-Step",
            xlabel="Micro-Step Index",
            ylabel="Linear Iterations",
            filename="linear_iterations_vs_step.png",
        ),
        "domega_used": _plot_series(
            cases,
            x_key="micro_idx",
            y_key="domega_used",
            title="Used Omega Increment by Micro-Step",
            xlabel="Micro-Step Index",
            ylabel=r"$d\omega$ used",
            filename="domega_used_vs_step.png",
        ),
        "correction_rel": _plot_series(
            cases,
            x_key="micro_idx",
            y_key="correction_rel",
            title="Relative Free-DOF Correction Norm",
            xlabel="Micro-Step Index",
            ylabel="Correction Relative Norm",
            filename="correction_rel_vs_step.png",
            hlines=[(0.7, "relative gate 0.7", "--")],
        ),
        "cumulative_wall_vs_omega": _plot_series(
            cases,
            x_key="omega",
            y_key="total_wall",
            title="Cumulative Wall Time Along the Trajectory",
            xlabel=r"Committed $\omega$",
            ylabel="Cumulative Wall Time [s]",
            filename="cumulative_wall_vs_omega.png",
        ),
        "cap_lengths": _plot_cap_lengths(cases),
    }

    summary = _build_summary(cases)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Full Micro-Walk Controller Variants\n")
    lines.append("This report compares the existing no-cap controller against the requested omega-step limiter and one nearby variant that reduces growth instead of capping the step.\n")
    lines.append("Main question: whether the lambda overshoot can be reduced without losing too much omega coverage toward the `7e6` horizon.\n")
    lines.append("## Cases\n")
    lines.append("| Case | Max omega | Max lambda | Coverage to 7e6 | Linear total | Zero-domega steps | Cap applied | Linear nonconverged | Diagnostics logged |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
    for case in cases:
        lines.append(
            "| "
            + f"{case['label']} | {case['max_omega'] / 1.0e6:.6f}e6 | {case['max_lambda']:.6f} | {100.0 * case['coverage']:.1f}% | "
            + f"{case['linear_total']} | {case['zero_domega_steps']} | {case['cap_applied_count']} | {case['not_converged_count']} | {case['diagnostics_logged']} |\n"
        )
    lines.append("\n")
    lines.append("## Interpretation\n")
    lines.append("- The no-cap baseline still reaches the farthest point on the branch: about `6.760e6`.\n")
    lines.append("- The requested cap variants do limit the proposed step size, but both stall earlier than the baseline.\n")
    lines.append("- The gentler-growth variant is the best of the new experiments. It reaches about `6.541e6` with much milder residual growth than the cap variants, but it still does not beat the existing no-cap controller on coverage.\n")
    lines.append("- None of the compared runs logged a linear-solver nonconvergence event or a hit-max-iteration event. The breakdown is therefore still controller-side, not a hidden Krylov failure.\n")
    lines.append("\n")
    lines.append("## Plot Gallery\n")
    for key in [
        "lambda_vs_omega",
        "omega_vs_step",
        "lambda_vs_step",
        "rel_after",
        "linear_iterations",
        "domega_used",
        "correction_rel",
        "cumulative_wall_vs_omega",
        "cap_lengths",
    ]:
        lines.append(f"![{key}](plots/{plots[key]})\n")

    (OUT_DIR / "README.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
