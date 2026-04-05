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
OUT_DIR = ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/controller_tuning_summary"
PLOTS_DIR = OUT_DIR / "plots"

CASES = [
    {
        "key": "diag_652",
        "label": "Diagnostics Run to 6.52e6",
        "path": ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/diag_rel_fixedomega_currentstate_diag/data/progress.jsonl",
        "residual_field": "rel_residual_after",
    },
    {
        "key": "prev_tuned",
        "label": "Previous Tuned 7e6 Partial",
        "path": ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/full_rel_fixedomega_currentstate_omega7e6_tuned/data/progress.jsonl",
        "residual_field": "rel_residual_after",
    },
    {
        "key": "corrected",
        "label": "Corrected 7e6 Partial",
        "path": ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/full_rel_fixedomega_currentstate_omega7e6_corrected/data/progress.jsonl",
        "residual_field": "rel_residual_after",
    },
]


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_case(spec: dict[str, object]) -> dict[str, object]:
    path = Path(spec["path"])
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    init = next(obj for obj in records if obj.get("event") == "init_complete")
    steps = [obj for obj in records if obj.get("event") == "micro_step_complete"]
    final = next((obj for obj in reversed(records) if obj.get("event") == "run_complete"), None)
    if final is None:
        last = steps[-1]
        final = {
            "event": "run_complete_partial",
            "stop_reason": "partial_log_no_run_complete",
            "micro_steps_completed": int(len(steps)),
            "micro_steps_requested": int(init.get("micro_steps_requested", len(steps))),
            "mode": init.get("mode"),
            "final_lambda": float(last.get("committed_lambda_after", last["lambda_after"])),
            "final_omega": float(last.get("committed_omega_after", last["omega_after"])),
            "runtime_seconds": float(last.get("total_wall_time", init.get("init_wall_time", math.nan))),
        }

    def _arr(field: str, default: float = math.nan) -> np.ndarray:
        vals = []
        for step in steps:
            val = step.get(field, default)
            if val is None:
                val = default
            vals.append(val)
        return np.asarray(vals, dtype=np.float64)

    return {
        "key": str(spec["key"]),
        "label": str(spec["label"]),
        "path": str(path),
        "init": init,
        "final": final,
        "steps": steps,
        "micro_idx": np.asarray([int(s["micro_index"]) for s in steps], dtype=np.float64),
        "omega": _arr("committed_omega_after"),
        "lambda": _arr("committed_lambda_after"),
        "rel_before": _arr("rel_residual_before"),
        "rel_after": _arr(str(spec["residual_field"])),
        "alpha": _arr("alpha"),
        "linear_iterations": _arr("linear_iterations"),
        "step_wall": _arr("iteration_wall_time"),
        "total_wall": _arr("total_wall_time"),
        "domega_used": _arr("domega_used"),
        "domega_next": _arr("domega_next"),
        "correction_norm": _arr("correction_free_norm"),
        "correction_rel": _arr("correction_rel"),
        "stall_repair_steps": _arr("stall_repair_steps", default=0.0),
        "stall_repair_rejected_steps": _arr("stall_repair_rejected_steps", default=0.0),
        "consecutive_no_progress": _arr("consecutive_no_progress", default=0.0),
        "fixed_omega_outer_rejected": _arr("fixed_omega_outer_rejected", default=0.0),
        "zero_domega_steps": int(sum(abs(float(s.get("domega_used", 0.0))) <= 1.0e-12 for s in steps)),
        "linear_total": int(sum(int(s.get("linear_iterations", 0)) for s in steps)),
        "stall_repair_total": int(sum(int(s.get("stall_repair_steps", 0)) for s in steps)),
        "stall_repair_rejected_total": int(sum(int(s.get("stall_repair_rejected_steps", 0)) for s in steps)),
    }


def _save_plot(fig: plt.Figure, filename: str) -> str:
    out = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return filename


def _plot_lambda_vs_omega(cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(8.4, 5.6), dpi=180)
    for case in cases:
        plt.plot(
            np.asarray(case["omega"], dtype=np.float64) / 1.0e6,
            np.asarray(case["lambda"], dtype=np.float64),
            marker="o",
            markersize=3.0,
            linewidth=1.5,
            label=str(case["label"]),
        )
    plt.xlabel(r"$\omega$ [$10^6$]")
    plt.ylabel(r"$\lambda$")
    plt.title(r"Micro-Walk Trajectories in the $(\omega,\lambda)$ Plane")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return _save_plot(fig, "lambda_vs_omega.png")


def _plot_series(
    cases: list[dict[str, object]],
    *,
    y_key: str,
    title: str,
    ylabel: str,
    filename: str,
    logy: bool = False,
    hlines: list[tuple[float, str, str]] | None = None,
) -> str:
    fig = plt.figure(figsize=(8.4, 5.6), dpi=180)
    for case in cases:
        x = np.asarray(case["micro_idx"], dtype=np.float64)
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
            label=str(case["label"]),
        )
    if hlines:
        seen: set[str] = set()
        for y, label, style in hlines:
            lab = label if label not in seen else "_nolegend_"
            plt.axhline(float(y), color="0.25", linestyle=style, linewidth=1.0, alpha=0.8, label=lab)
            seen.add(label)
    if logy:
        plt.yscale("log")
    plt.xlabel("Micro-Step Index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    return _save_plot(fig, filename)


def _plot_cumulative_wall_vs_omega(cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(8.4, 5.6), dpi=180)
    for case in cases:
        x = np.asarray(case["omega"], dtype=np.float64) / 1.0e6
        y = np.asarray(case["total_wall"], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        plt.plot(
            x[finite],
            y[finite],
            marker="o",
            markersize=3.0,
            linewidth=1.5,
            label=str(case["label"]),
        )
    plt.xlabel(r"Committed $\omega$ [$10^6$]")
    plt.ylabel("Cumulative Wall Time [s]")
    plt.title("Cumulative Wall Time Along the Micro-Walk")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return _save_plot(fig, "cumulative_wall_vs_omega.png")


def _plot_tail_residual_compare(cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(8.4, 5.6), dpi=180)
    for case in cases:
        if case["key"] == "diag_652":
            continue
        x = np.asarray(case["micro_idx"], dtype=np.float64)
        y = np.asarray(case["rel_after"], dtype=np.float64)
        mask = x >= 12.0
        finite = mask & np.isfinite(y)
        if not np.any(finite):
            continue
        plt.plot(
            x[finite],
            y[finite],
            marker="o",
            markersize=3.2,
            linewidth=1.6,
            label=str(case["label"]),
        )
    plt.xlabel("Micro-Step Index")
    plt.ylabel("Outer-Step Relative Residual After")
    plt.title("Late-Tail Residual Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return _save_plot(fig, "tail_rel_after_compare.png")


def _build_summary(cases: list[dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {"cases": []}
    for case in cases:
        payload["cases"].append(
            {
                "key": case["key"],
                "label": case["label"],
                "path": case["path"],
                "micro_steps_saved": int(len(case["steps"])),
                "final": case["final"],
                "linear_total": int(case["linear_total"]),
                "zero_domega_steps": int(case["zero_domega_steps"]),
                "stall_repair_total": int(case["stall_repair_total"]),
                "stall_repair_rejected_total": int(case["stall_repair_rejected_total"]),
                "max_omega": float(np.max(case["omega"])) if len(case["omega"]) else math.nan,
            }
        )
    return payload


def main() -> None:
    _ensure_dirs()
    cases = [_load_case(spec) for spec in CASES]

    plots = {
        "lambda_vs_omega": _plot_lambda_vs_omega(cases),
        "rel_after": _plot_series(
            cases,
            y_key="rel_after",
            title="Outer-Step Relative Residual After Each Micro-Step",
            ylabel="Relative Residual After",
            filename="rel_after_vs_step.png",
            logy=True,
            hlines=[(1.0e-1, "1e-1", ":"), (1.0, "1", "--")],
        ),
        "alpha": _plot_series(
            cases,
            y_key="alpha",
            title="Damping Alpha per Micro-Step",
            ylabel=r"$\alpha$",
            filename="alpha_vs_step.png",
            hlines=[(0.25, "alpha freeze = 0.25", "--"), (0.5, "alpha = 0.5", ":")],
        ),
        "linear_iterations": _plot_series(
            cases,
            y_key="linear_iterations",
            title="Linear Iterations per Micro-Step",
            ylabel="Linear Iterations",
            filename="linear_iterations_vs_step.png",
        ),
        "domega_used": _plot_series(
            cases,
            y_key="domega_used",
            title=r"Used $d\omega$ per Micro-Step",
            ylabel=r"$d\omega$",
            filename="domega_used_vs_step.png",
        ),
        "correction_rel": _plot_series(
            cases,
            y_key="correction_rel",
            title="Relative Free-DOF Correction Norm",
            ylabel="correction_rel",
            filename="correction_rel_vs_step.png",
            hlines=[(0.7, "relative gate = 0.7", "--")],
        ),
        "correction_norm": _plot_series(
            cases,
            y_key="correction_norm",
            title="Absolute Free-DOF Correction Norm",
            ylabel="correction_free_norm",
            filename="correction_norm_vs_step.png",
            hlines=[(1.0, "absolute gate = 1.0", "--")],
        ),
        "cumulative_wall": _plot_cumulative_wall_vs_omega(cases),
        "tail_rel_compare": _plot_tail_residual_compare(cases),
    }

    summary = _build_summary(cases)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    diag_case = next(case for case in cases if case["key"] == "diag_652")
    prev_case = next(case for case in cases if case["key"] == "prev_tuned")
    corr_case = next(case for case in cases if case["key"] == "corrected")

    common_steps = min(len(prev_case["steps"]), len(corr_case["steps"]))
    matched_rows = []
    for idx in range(max(0, common_steps - 4), common_steps):
        prev = prev_case["steps"][idx]
        corr = corr_case["steps"][idx]
        matched_rows.append(
            (
                idx + 1,
                float(prev.get("committed_omega_after", prev["omega_after"])),
                float(corr.get("committed_omega_after", corr["omega_after"])),
                float(prev.get("committed_rel_residual_after", prev.get("rel_residual_after", math.nan))),
                float(corr.get("committed_rel_residual_after", corr.get("rel_residual_after", math.nan))),
                int(prev.get("linear_iterations", 0)),
            )
        )

    readme = f"""# Micro-Walk Controller Tuning Summary

## Summary

This report compares three saved traces:

- `Diagnostics Run to 6.52e6`
- `Previous Tuned 7e6 Partial`
- `Corrected 7e6 Partial`

The key conclusion is unchanged:

- the late-tail issue is **not** a linear-solver convergence failure,
- the real improvement came from fixing fixed-`omega` steps to continue from the current state,
- and that fix is enough to make the shorter `omega = 6.52e6` run work cleanly,
- but the later `omega > 6.7e6` tail is still unresolved.

The diagnostics-enabled short run reached:

- final `omega = {diag_case["final"]["final_omega"]:.9e}`
- final `lambda = {diag_case["final"]["final_lambda"]:.12f}`
- `10` micro-steps
- `{diag_case["linear_total"]}` total linear iterations
- `{diag_case["final"]["runtime_seconds"]:.2f} s` wall time

For the two partial `7e6` traces, the currently saved data are still effectively the same trajectory up to the saved cutoff:

| Case | steps saved | max `omega` reached | linear iterations | zero-`domega` steps | stall-repair steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous tuned | {len(prev_case["steps"])} | {np.max(prev_case["omega"]):.9e} | {prev_case["linear_total"]} | {prev_case["zero_domega_steps"]} | {prev_case["stall_repair_total"]} |
| corrected | {len(corr_case["steps"])} | {np.max(corr_case["omega"]):.9e} | {corr_case["linear_total"]} | {corr_case["zero_domega_steps"]} | {corr_case["stall_repair_total"]} |

The apparent change at step 17 is mostly a logging correction: the old tuned artifact had a stale committed residual field in that regime, while the corrected run writes the actual committed value.

## Matched Tail Steps

| Step | previous tuned `omega` | corrected `omega` | previous tuned committed relres | corrected committed relres | linear its |
| --- | ---: | ---: | ---: | ---: | ---: |
"""
    for step, prev_omega, corr_omega, prev_rel, corr_rel, lin in matched_rows:
        readme += f"| {step} | `{prev_omega:.9e}` | `{corr_omega:.9e}` | `{prev_rel:.9e}` | `{corr_rel:.9e}` | {lin} |\n"

    readme += f"""

## Plots

The residual plots use the recorded outer-step `rel_residual_after` field for comparison. This avoids the stale committed-residual field in the older tuned artifact.

### Branch Path

![lambda_vs_omega](plots/{plots["lambda_vs_omega"]})

### Relative Residual After

![rel_after](plots/{plots["rel_after"]})

### Late-Tail Relative Residual Compare

![tail_rel_compare](plots/{plots["tail_rel_compare"]})

### Damping Alpha

![alpha](plots/{plots["alpha"]})

### Linear Iterations

![linear_iterations](plots/{plots["linear_iterations"]})

### Used dOmega

![domega_used](plots/{plots["domega_used"]})

### Relative Correction Norm

![correction_rel](plots/{plots["correction_rel"]})

### Absolute Correction Norm

![correction_norm](plots/{plots["correction_norm"]})

### Cumulative Wall Time vs Omega

![cumulative_wall](plots/{plots["cumulative_wall"]})

## Interpretation

- The diagnostics-enabled short run confirms the Krylov solves converge cleanly.
- The fixed-`omega current-state` patch is a real controller improvement.
- The latest outer-step rejection patch has not yet materially changed the saved `7e6` partial trajectory.
- The remaining issue is still a genuine hard-tail controller problem, not a hidden linear-solver cap problem.
"""

    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
