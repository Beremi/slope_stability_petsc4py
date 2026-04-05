#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DIR = ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/rank8_micro100/data"
DEFAULT_ADAPTIVE_RUN_DIR = ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/rank8_micro100_adaptive/data"
DEFAULT_OUT_DIR = ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/report"
DEFAULT_SECANT_PROGRESS = (
    ROOT / "artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_smart_controller_v2_rank8_step100/data/progress.jsonl"
)

def _ensure_dirs(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def _load_progress(run_dir: Path) -> tuple[dict, list[dict], dict]:
    records = [json.loads(line) for line in (run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    init_record = next(obj for obj in records if obj.get("event") == "init_complete")
    steps = [obj for obj in records if obj.get("event") == "micro_step_complete"]
    final_record = next((obj for obj in reversed(records) if obj.get("event") == "run_complete"), None)
    if final_record is None:
        if not steps:
            raise RuntimeError(f"No micro-step records found in {run_dir}")
        last = steps[-1]
        final_record = {
            "event": "run_complete_partial",
            "mode": init_record.get("mode"),
            "micro_steps_completed": len(steps),
            "micro_steps_requested": int(init_record.get("micro_steps_requested", len(steps))),
            "stop_reason": "partial_log_no_run_complete",
            "runtime_seconds": float(last.get("total_wall_time", init_record.get("init_wall_time", math.nan))),
            "final_omega": float(last.get("committed_omega_after", last["omega_after"])),
            "final_lambda": float(last.get("committed_lambda_after", last["lambda_after"])),
        }
    return init_record, steps, final_record


def _load_case(run_dir: Path, *, label: str, fallback_mode: str) -> dict[str, object]:
    init_record, steps, final_record = _load_progress(run_dir)
    if not steps:
        raise RuntimeError(f"No micro-step records found in {run_dir}")
    micro_idx = np.asarray([int(obj["micro_index"]) for obj in steps], dtype=np.int64)
    omega_after = np.asarray([float(obj["omega_after"]) for obj in steps], dtype=np.float64)
    lambda_after = np.asarray([float(obj["lambda_after"]) for obj in steps], dtype=np.float64)
    committed_omega_after = np.asarray([float(obj.get("committed_omega_after", obj["omega_after"])) for obj in steps], dtype=np.float64)
    committed_lambda_after = np.asarray([float(obj.get("committed_lambda_after", obj["lambda_after"])) for obj in steps], dtype=np.float64)
    omega_hist = np.asarray(init_record["omega_hist"] + omega_after.tolist(), dtype=np.float64)
    lambda_hist = np.asarray(init_record["lambda_hist"] + lambda_after.tolist(), dtype=np.float64)
    derived_domega = np.asarray(
        [float(obj["omega_target"]) - float(obj["omega_before"]) for obj in steps],
        dtype=np.float64,
    )

    def _array(field: str, *, default: float = math.nan) -> np.ndarray:
        values = []
        for obj in steps:
            val = obj.get(field, default)
            if val is None:
                val = default
            values.append(val)
        return np.asarray(values, dtype=np.float64)

    mode = str(final_record.get("mode", init_record.get("mode", fallback_mode)))
    return {
        "label": str(label),
        "mode": mode,
        "run_dir": run_dir,
        "init": init_record,
        "final": final_record,
        "steps": steps,
        "micro_idx": micro_idx,
        "omega_hist": omega_hist,
        "lambda_hist": lambda_hist,
        "omega_after": omega_after,
        "lambda_after": lambda_after,
        "committed_omega_after": committed_omega_after,
        "committed_lambda_after": committed_lambda_after,
        "criterion_before": _array("criterion_before"),
        "criterion_after": _array("criterion_after"),
        "committed_criterion_after": _array("committed_criterion_after"),
        "rel_before": _array("rel_residual_before"),
        "rel_after": _array("rel_residual_after"),
        "committed_rel_after": _array("committed_rel_residual_after"),
        "linear_iterations": _array("linear_iterations"),
        "linear_solve_time": _array("linear_solve_time"),
        "linear_preconditioner_time": _array("linear_preconditioner_time"),
        "linear_orthogonalization_time": _array("linear_orthogonalization_time"),
        "step_wall": _array("iteration_wall_time"),
        "alpha": _array("alpha"),
        "accepted_delta_lambda": _array("accepted_delta_lambda"),
        "total_wall": _array("total_wall_time"),
        "correction_norm": _array("correction_free_norm"),
        "committed_correction_norm": _array("committed_correction_free_norm"),
        "correction_rel": _array("correction_rel"),
        "anchor_increment_norm": _array("anchor_increment_norm"),
        "newton_substeps": _array("newton_substeps", default=1.0),
        "same_matrix_halfstep_triggered": _array("same_matrix_halfstep_triggered", default=0.0),
        "half_step_linear_iterations": _array("half_step_linear_iterations", default=0.0),
        "half_step_wall_time": _array("half_step_wall_time", default=0.0),
        "half_step_rel_after_first": _array("half_step_rel_after_first", default=math.nan),
        "flag_N": _array("flag_N"),
        "domega_used": _array("domega_used", default=math.nan),
        "domega_next": _array("domega_next", default=math.nan),
        "rolling_basis_dim": _array("rolling_basis_dim", default=math.nan),
        "domega_used_derived": derived_domega,
        "effective_idx": micro_idx + 0.5 * np.cumsum(_array("same_matrix_halfstep_triggered", default=0.0)),
        "half_idx": micro_idx.astype(np.float64) + 0.5,
    }


def _load_trimmed_secant_branch(progress_path: Path, *, omega_min: float, omega_max: float) -> tuple[np.ndarray, np.ndarray]:
    records = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    init_record = next(obj for obj in records if obj.get("event") == "init_complete")
    step_records = [obj for obj in records if obj.get("event") == "step_accepted"]
    omega_vals = [float(v) for v in init_record["omega_hist"]]
    lambda_vals = [float(v) for v in init_record["lambda_hist"]]
    for obj in step_records:
        omega_val = float(obj["omega_value"])
        if float(omega_min) - 1.0e-9 <= omega_val <= float(omega_max) + 1.0e-9:
            omega_vals.append(omega_val)
            lambda_vals.append(float(obj["lambda_value"]))
    omega_arr = np.asarray(omega_vals, dtype=np.float64)
    lambda_arr = np.asarray(lambda_vals, dtype=np.float64)
    mask = (omega_arr >= float(omega_min) - 1.0e-9) & (omega_arr <= float(omega_max) + 1.0e-9)
    return omega_arr[mask], lambda_arr[mask]


def _plot_xy_cases(
    cases: list[dict[str, object]],
    *,
    x_getter,
    y_getter,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    plots_dir: Path,
    logy: bool = False,
    hlines: list[tuple[float, str, str]] | None = None,
    halfstep_y_getter=None,
) -> str:
    out_path = plots_dir / filename
    fig = plt.figure(figsize=(8.4, 5.6), dpi=180)
    for case in cases:
        x = np.asarray(x_getter(case), dtype=np.float64)
        y = np.asarray(y_getter(case), dtype=np.float64)
        if x.size == 0 or y.size == 0:
            continue
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        line = plt.plot(
            x[finite],
            y[finite],
            marker="o",
            markersize=3.0,
            linewidth=1.5,
            label=str(case["label"]),
        )[0]
        if halfstep_y_getter is not None:
            half_y = np.asarray(halfstep_y_getter(case), dtype=np.float64)
            half_x = np.asarray(case.get("half_idx", []), dtype=np.float64)
            half_mask = (
                np.asarray(case.get("same_matrix_halfstep_triggered", []), dtype=np.float64) > 0.5
            ) & np.isfinite(half_x) & np.isfinite(half_y)
            if np.any(half_mask):
                plt.scatter(
                    half_x[half_mask],
                    half_y[half_mask],
                    marker="x",
                    s=26.0,
                    linewidths=1.2,
                    color=line.get_color(),
                    alpha=0.95,
                    label="_nolegend_",
                )
    if hlines:
        seen_labels: set[str] = set()
        for y, label, style in hlines:
            if not np.isfinite(float(y)):
                continue
            plot_label = label if label not in seen_labels else "_nolegend_"
            plt.axhline(float(y), color="0.25", linestyle=style, linewidth=1.0, alpha=0.8, label=plot_label)
            seen_labels.add(label)
    if logy:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path.name


def _plot_lambda_omega(
    cases: list[dict[str, object]],
    *,
    secant_omega_hist: np.ndarray | None,
    secant_lambda_hist: np.ndarray | None,
    plots_dir: Path,
) -> str:
    out_path = plots_dir / "lambda_vs_omega.png"
    fig = plt.figure(figsize=(8.4, 5.6), dpi=180)
    if secant_omega_hist is not None and secant_lambda_hist is not None and secant_omega_hist.size:
        plt.plot(
            secant_omega_hist / 1.0e6,
            secant_lambda_hist,
            marker="s",
            markersize=3.2,
            linewidth=1.5,
            label="Converged secant branch",
        )
    for case in cases:
        plt.plot(
            np.asarray(case["omega_hist"], dtype=np.float64) / 1.0e6,
            np.asarray(case["lambda_hist"], dtype=np.float64),
            marker="o",
            markersize=3.0,
            linewidth=1.6,
            label=str(case["label"]),
        )
    plt.xlabel(r"$\omega$ [$10^6$]")
    plt.ylabel(r"$\lambda$")
    plt.title(r"Single-Newton-Step Walks in the $(\omega,\lambda)$ Plane")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path.name


def _plot_dual_criterion(cases: list[dict[str, object]], *, plots_dir: Path) -> str:
    out_path = plots_dir / "criterion_before_after.png"
    fig = plt.figure(figsize=(8.4, 5.6), dpi=180)
    for case in cases:
        x = np.asarray(case["micro_idx"], dtype=np.float64)
        plt.plot(
            x,
            np.asarray(case["criterion_before"], dtype=np.float64),
            marker="o",
            markersize=2.6,
            linewidth=1.2,
            linestyle="--",
            label=f"{case['label']} before",
        )
        line_after = plt.plot(
            x,
            np.asarray(case["criterion_after"], dtype=np.float64),
            marker="s",
            markersize=2.4,
            linewidth=1.4,
            label=f"{case['label']} after",
        )[0]
        half_x = np.asarray(case.get("half_idx", []), dtype=np.float64)
        half_y = np.asarray(case.get("committed_criterion_after", []), dtype=np.float64)
        half_mask = (
            np.asarray(case.get("same_matrix_halfstep_triggered", []), dtype=np.float64) > 0.5
        ) & np.isfinite(half_x) & np.isfinite(half_y)
        if np.any(half_mask):
            plt.scatter(
                half_x[half_mask],
                half_y[half_mask],
                marker="x",
                s=26.0,
                linewidths=1.2,
                color=line_after.get_color(),
                alpha=0.95,
                label="_nolegend_",
            )
    plt.yscale("log")
    plt.xlabel("Micro-Step Index")
    plt.ylabel("Residual Criterion")
    plt.title("Residual Criterion Before and After Each Single Newton Step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path.name


def _plot_dual_relres(cases: list[dict[str, object]], *, plots_dir: Path) -> str:
    out_path = plots_dir / "rel_residual_before_after.png"
    fig = plt.figure(figsize=(8.4, 5.6), dpi=180)
    for case in cases:
        x = np.asarray(case["micro_idx"], dtype=np.float64)
        plt.plot(
            x,
            np.asarray(case["rel_before"], dtype=np.float64),
            marker="o",
            markersize=2.6,
            linewidth=1.2,
            linestyle="--",
            label=f"{case['label']} before",
        )
        line_after = plt.plot(
            x,
            np.asarray(case["rel_after"], dtype=np.float64),
            marker="s",
            markersize=2.4,
            linewidth=1.4,
            label=f"{case['label']} after",
        )[0]
        half_x = np.asarray(case.get("half_idx", []), dtype=np.float64)
        half_y = np.asarray(case.get("committed_rel_after", []), dtype=np.float64)
        half_mask = (
            np.asarray(case.get("same_matrix_halfstep_triggered", []), dtype=np.float64) > 0.5
        ) & np.isfinite(half_x) & np.isfinite(half_y)
        if np.any(half_mask):
            plt.scatter(
                half_x[half_mask],
                half_y[half_mask],
                marker="x",
                s=26.0,
                linewidths=1.2,
                color=line_after.get_color(),
                alpha=0.95,
                label="_nolegend_",
            )
    for y, label, style in (
        (1.0e-3, r"unfreeze $\Delta\lambda$ at $10^{-3}$", ":"),
        (5.0e-3, r"stall-repair target $5\times10^{-3}$", "-."),
        (1.0e-1, r"grow $\Delta\omega$ below $10^{-1}$", "--"),
        (1.0, r"freeze $\omega$ above $1$", "--"),
        (1.0e1, r"rollback above $10$", ":"),
    ):
        plt.axhline(float(y), color="0.25", linestyle=style, linewidth=1.0, alpha=0.8, label=label)
    plt.yscale("log")
    plt.xlabel("Micro-Step Index")
    plt.ylabel("Relative Residual")
    plt.title("Relative Residual Before and After Each Single Newton Step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Report fixed/adaptive single-Newton-step walk comparison.")
    parser.add_argument("--fixed-run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--adaptive-run-dir", type=Path, default=DEFAULT_ADAPTIVE_RUN_DIR)
    parser.add_argument(
        "--compare-run",
        action="append",
        default=[],
        help="Additional labeled run in the form 'Label:path/to/data'. If provided, these define the compared cases.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--secant-progress", type=Path, default=DEFAULT_SECANT_PROGRESS)
    args = parser.parse_args()

    fixed_run_dir = args.fixed_run_dir.resolve()
    adaptive_run_dir = args.adaptive_run_dir.resolve()
    out_dir = args.out_dir.resolve()
    secant_progress = args.secant_progress.resolve()
    plots_dir = _ensure_dirs(out_dir)
    cases: list[dict[str, object]] = []
    if args.compare_run:
        for spec in args.compare_run:
            if ":" not in spec:
                raise ValueError(f"Invalid --compare-run value: {spec!r}")
            label, raw_path = spec.split(":", 1)
            run_dir = Path(raw_path).resolve()
            fallback_mode = "adaptive" if "adaptive" in label.lower() else "fixed"
            cases.append(_load_case(run_dir, label=label.strip(), fallback_mode=fallback_mode))
    else:
        cases = [
            _load_case(fixed_run_dir, label="Fixed d_omega/5", fallback_mode="fixed"),
        ]
        if adaptive_run_dir.exists():
            cases.append(_load_case(adaptive_run_dir, label="Adaptive d_omega + rolling basis 20", fallback_mode="adaptive"))

    omega_min = min(float(np.min(np.asarray(case["omega_hist"], dtype=np.float64))) for case in cases)
    omega_max = max(float(np.max(np.asarray(case["omega_hist"], dtype=np.float64))) for case in cases)
    secant_omega_hist, secant_lambda_hist = _load_trimmed_secant_branch(
        secant_progress,
        omega_min=omega_min,
        omega_max=omega_max,
    )

    plots = {
        "lambda_vs_omega": _plot_lambda_omega(
            cases,
            secant_omega_hist=secant_omega_hist,
            secant_lambda_hist=secant_lambda_hist,
            plots_dir=plots_dir,
        ),
        "lambda_vs_step": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["lambda_after"],
            halfstep_y_getter=lambda case: case["committed_lambda_after"],
            xlabel="Micro-Step Index",
            ylabel=r"$\lambda$ after one Newton step",
            title=r"Updated $\lambda$ Along the One-Step Walks",
            filename="lambda_vs_step.png",
            plots_dir=plots_dir,
        ),
        "omega_vs_step": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: np.asarray(case["omega_after"], dtype=np.float64) / 1.0e6,
            halfstep_y_getter=lambda case: np.asarray(case["committed_omega_after"], dtype=np.float64) / 1.0e6,
            xlabel="Micro-Step Index",
            ylabel=r"$\omega$ after one Newton step [$10^6$]",
            title=r"Updated $\omega$ Along the One-Step Walks",
            filename="omega_vs_step.png",
            plots_dir=plots_dir,
        ),
        "criterion_before_after": _plot_dual_criterion(cases, plots_dir=plots_dir),
        "relres_before_after": _plot_dual_relres(cases, plots_dir=plots_dir),
        "half_step_rel_after_first": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["half_step_rel_after_first"],
            xlabel="Micro-Step Index",
            ylabel="Relative Residual After First Newton Iteration",
            title="First-Iteration Relative Residual Before Same-Matrix Half-Step Decision",
            filename="half_step_rel_after_first_vs_step.png",
            plots_dir=plots_dir,
            logy=True,
            hlines=[(2.0e-2, r"same-matrix half-step trigger $2\times10^{-2}$", "--")],
        ),
        "linear_iterations": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["linear_iterations"],
            xlabel="Micro-Step Index",
            ylabel="Linear Iterations",
            title="Linear Iterations per Single Newton Step",
            filename="linear_iterations_vs_step.png",
            plots_dir=plots_dir,
        ),
        "linear_solve_time": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["linear_solve_time"],
            xlabel="Micro-Step Index",
            ylabel="Linear Solve Time [s]",
            title="Linear Solve Time per Single Newton Step",
            filename="linear_solve_time_vs_step.png",
            plots_dir=plots_dir,
        ),
        "linear_preconditioner_time": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["linear_preconditioner_time"],
            xlabel="Micro-Step Index",
            ylabel="Preconditioner Time [s]",
            title="Preconditioner Time per Single Newton Step",
            filename="linear_preconditioner_time_vs_step.png",
            plots_dir=plots_dir,
        ),
        "linear_orthogonalization_time": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["linear_orthogonalization_time"],
            xlabel="Micro-Step Index",
            ylabel="Orthogonalization Time [s]",
            title="Orthogonalization Time per Single Newton Step",
            filename="linear_orthogonalization_time_vs_step.png",
            plots_dir=plots_dir,
        ),
        "step_wall": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["step_wall"],
            xlabel="Micro-Step Index",
            ylabel="Single-Step Wall Time [s]",
            title="Wall Time per Single Newton Step",
            filename="step_wall_vs_step.png",
            plots_dir=plots_dir,
        ),
        "alpha": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["alpha"],
            xlabel="Micro-Step Index",
            ylabel=r"Damping $\alpha$",
            title=r"Damping Factor for Each Single Newton Step",
            filename="alpha_vs_step.png",
            plots_dir=plots_dir,
            hlines=[(0.5, r"freeze $\omega$ below $\alpha=0.5$", "--")],
        ),
        "accepted_delta_lambda": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["accepted_delta_lambda"],
            xlabel="Micro-Step Index",
            ylabel=r"Accepted $\Delta\lambda$",
            title=r"Accepted $\Delta\lambda$ per Single Newton Step",
            filename="accepted_delta_lambda_vs_step.png",
            plots_dir=plots_dir,
            hlines=[(0.0, r"negative $\Delta\lambda$ freezes growth", "--")],
        ),
        "cumulative_wall": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["total_wall"],
            xlabel="Micro-Step Index",
            ylabel="Cumulative Wall Time [s]",
            title="Cumulative Runtime of the One-Step Walks",
            filename="cumulative_wall_vs_step.png",
            plots_dir=plots_dir,
        ),
        "newton_substeps": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["newton_substeps"],
            xlabel="Micro-Step Index",
            ylabel="Newton Substeps in Micro-Step",
            title="Number of Newton Substeps per Micro-Step",
            filename="newton_substeps_vs_step.png",
            plots_dir=plots_dir,
            hlines=[(1.0, "base one-step update", ":"), (2.0, "same-matrix half-step used", "--")],
        ),
        "effective_idx_vs_omega": _plot_xy_cases(
            cases,
            x_getter=lambda case: np.asarray(case["omega_after"], dtype=np.float64) / 1.0e6,
            y_getter=lambda case: case["effective_idx"],
            xlabel=r"$\omega$ after one Newton step [$10^6$]",
            ylabel="Effective Step Index",
            title="Effective Step Index With Half-Step Accounting",
            filename="effective_idx_vs_omega.png",
            plots_dir=plots_dir,
        ),
        "correction_norm": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["correction_norm"],
            halfstep_y_getter=lambda case: case["committed_correction_norm"] if "committed_correction_norm" in case else case["correction_norm"],
            xlabel="Micro-Step Index",
            ylabel="Free-DOF Correction Norm",
            title="Free-DOF Norm of the Accepted Single-Step Newton Correction",
            filename="correction_norm_vs_step.png",
            plots_dir=plots_dir,
            logy=True,
            hlines=[(1.0, "historical absolute correction gate = 1 (inactive here)", ":")],
        ),
        "correction_rel": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["correction_rel"],
            xlabel="Micro-Step Index",
            ylabel="Relative Free-DOF Correction Norm",
            title="Relative Free-DOF Correction Norm per Single Newton Step",
            filename="correction_rel_vs_step.png",
            plots_dir=plots_dir,
            logy=True,
            hlines=[(0.7, r"freeze $\omega$ above relative correction 0.7", "--")],
        ),
        "domega_used": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: np.where(
                np.isfinite(np.asarray(case["domega_used"], dtype=np.float64)),
                np.asarray(case["domega_used"], dtype=np.float64),
                np.asarray(case["domega_used_derived"], dtype=np.float64),
            ),
            xlabel="Micro-Step Index",
            ylabel=r"Used $\Delta\omega$",
            title=r"Used $\Delta\omega$ per One-Step Walk Iteration",
            filename="domega_used_vs_step.png",
            plots_dir=plots_dir,
        ),
        "domega_used_vs_omega": _plot_xy_cases(
            cases,
            x_getter=lambda case: np.asarray(case["omega_after"], dtype=np.float64) / 1.0e6,
            y_getter=lambda case: np.where(
                np.isfinite(np.asarray(case["domega_used"], dtype=np.float64)),
                np.asarray(case["domega_used"], dtype=np.float64),
                np.asarray(case["domega_used_derived"], dtype=np.float64),
            ),
            xlabel=r"$\omega$ after one Newton step [$10^6$]",
            ylabel=r"Used $\Delta\omega$",
            title=r"Used $\Delta\omega$ as a Function of $\omega$",
            filename="domega_used_vs_omega.png",
            plots_dir=plots_dir,
        ),
        "domega_next": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["domega_next"],
            xlabel="Micro-Step Index",
            ylabel=r"Next $\Delta\omega$",
            title=r"Scheduled Next $\Delta\omega$ per One-Step Walk Iteration",
            filename="domega_next_vs_step.png",
            plots_dir=plots_dir,
        ),
        "rolling_basis_dim": _plot_xy_cases(
            cases,
            x_getter=lambda case: case["micro_idx"],
            y_getter=lambda case: case["rolling_basis_dim"],
            xlabel="Micro-Step Index",
            ylabel="Rolling Deflation Basis Dimension",
            title="Rolling Solution-Basis Size per One-Step Walk Iteration",
            filename="rolling_basis_dim_vs_step.png",
            plots_dir=plots_dir,
        ),
    }

    case_summaries = []
    for case in cases:
        rel_after = np.asarray(case["rel_after"], dtype=np.float64)
        alpha = np.asarray(case["alpha"], dtype=np.float64)
        linear_iterations = np.asarray(case["linear_iterations"], dtype=np.float64)
        case_summaries.append(
            {
                "label": str(case["label"]),
                "mode": str(case["mode"]),
                "micro_steps_completed": int(len(case["steps"])),
                "micro_steps_requested": int(case["final"]["micro_steps_requested"]),
                "stop_reason": str(case["final"]["stop_reason"]),
                "final_lambda": float(case["final"]["final_lambda"]),
                "final_omega": float(case["final"]["final_omega"]),
                "runtime_seconds": float(case["final"]["runtime_seconds"]),
                "linear_iterations_total": int(np.sum(linear_iterations)),
                "linear_iterations_mean": float(np.mean(linear_iterations)),
                "linear_iterations_max": int(np.max(linear_iterations)),
                "criterion_after_min": float(np.min(np.asarray(case["criterion_after"], dtype=np.float64))),
                "criterion_after_max": float(np.max(np.asarray(case["criterion_after"], dtype=np.float64))),
                "rel_residual_after_min": float(np.min(rel_after)),
                "rel_residual_after_max": float(np.max(rel_after)),
                "alpha_min": float(np.min(alpha)),
                "alpha_max": float(np.max(alpha)),
                "flagged_steps": int(np.sum(np.asarray(case["flag_N"], dtype=np.float64) != 0)),
            }
        )
    summary = {
        "cases": case_summaries,
        "secant_overlay_points": int(secant_omega_hist.size),
        "plot_files": plots,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Single-Newton-Step Walk Comparison")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        "These experiments use the rank-8 `P4(L1)` `pmg_shell` SSR setup from the archived baseline. "
        "Both start from the same two converged initialization points at `lambda=1.0` and `lambda=1.1`, "
        "then perform one damped Newton iteration after each continuation advance."
    )
    lines.append("")
    lines.append(
        "Each micro-step deliberately calls the SSR Newton solver with `it_newt_max = 1`, so the internal Newton flag "
        "does not indicate full-step convergence here; the meaningful trace is the recorded one-step update history."
    )
    lines.append("")
    lines.append(f"- Initial converged lambdas: `{cases[0]['init']['lambda_hist'][0]:.12f}`, `{cases[0]['init']['lambda_hist'][1]:.12f}`")
    lines.append(f"- Initial converged omegas: `{cases[0]['init']['omega_hist'][0]:.6f}`, `{cases[0]['init']['omega_hist'][1]:.6f}`")
    lines.append(f"- `d_omega_base = {cases[0]['init']['domega_base']:.6f}`")
    lines.append(f"- Initial `d_omega_micro = {cases[0]['init']['domega_micro']:.6f}`")
    lines.append(f"- Converged secant overlay points in the same omega range: `{int(secant_omega_hist.size)}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    for case in case_summaries:
        lines.append(f"- `{case['label']}`:")
        lines.append(
            f"  mode `{case['mode']}`, runtime `{case['runtime_seconds']:.3f} s`, "
            f"final `(omega, lambda)=({case['final_omega']:.6f}, {case['final_lambda']:.12f})`, "
            f"steps `{case['micro_steps_completed']}/{case['micro_steps_requested']}`, stop `{case['stop_reason']}`, "
            f"linear total `{case['linear_iterations_total']}`, "
            f"rel-residual-after range `{case['rel_residual_after_min']:.3e} .. {case['rel_residual_after_max']:.3e}`, "
            f"alpha range `{case['alpha_min']:.6f} .. {case['alpha_max']:.6f}`"
        )
    lines.append(f"- Converged secant overlay points in the same omega range: `{int(secant_omega_hist.size)}`")
    lines.append("")
    lines.append("## Controller Criteria")
    lines.append("")
    lines.append("The walk makes its main controller decision from the first damped Newton substep of each micro-step, then updates the next `d_omega` from simple gates.")
    lines.append("")
    lines.append("Shared rules:")
    lines.append("- If `rel_residual_after < 1e-1` and `alpha >= 0.5`, the next `d_omega` is grown by `x1.5`.")
    lines.append("- If `rel_residual_after > 1`, the next `omega` advance is frozen.")
    lines.append("- If `alpha < 0.5`, the next `omega` advance is frozen.")
    lines.append("- If the first-iteration `rel_residual_after > 2e-2`, one detached same-matrix refinement is attempted after the guard-rail decision.")
    lines.append("- That detached refinement does not change rollback/freeze counters for the main micro-step; it is plotted separately at `k + 0.5` and also reflected in the `effective step index` / `Newton substeps` traces.")
    lines.append("- If `accepted_delta_lambda < 0`, further `omega` growth is frozen and `d_omega` is reduced by `x0.66` until recovery.")
    lines.append("- If a step produces no `omega` progress, the scheduled next `d_omega` is additionally reduced by `x0.8`.")
    lines.append("- If `rel_residual_after > 1e1`, the attempted step is rolled back and `d_omega` is reduced by `x0.66`.")
    lines.append("")
    lines.append("New controller only:")
    lines.append("- The correction gate is based on the relative free-DOF correction")
    lines.append("  `correction_rel = ||ΔU||_free / ||U_anchor,k - U_anchor,k-1||_free`.")
    lines.append("- If `correction_rel > 0.7`, the next `omega` advance is frozen.")
    lines.append("- The historical absolute rule `correction_free_norm > 1` is shown in the plots as a reference, but it is not active in the new controller.")
    lines.append("- After `3` consecutive zero-progress steps, stall recovery performs `2` extra fixed-omega Newton repairs.")
    lines.append("- If stall repair reduces `rel_residual_after < 5e-3`, the retry step size is reset to `0.25 * d_omega_last_successful`.")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for key in (
        "lambda_vs_omega",
        "lambda_vs_step",
        "omega_vs_step",
        "criterion_before_after",
        "relres_before_after",
        "half_step_rel_after_first",
        "linear_iterations",
        "linear_solve_time",
        "linear_preconditioner_time",
        "linear_orthogonalization_time",
        "step_wall",
        "alpha",
        "accepted_delta_lambda",
        "cumulative_wall",
        "newton_substeps",
        "effective_idx_vs_omega",
        "correction_norm",
        "correction_rel",
        "domega_used",
        "domega_used_vs_omega",
        "domega_next",
        "rolling_basis_dim",
    ):
        filename = plots[key]
        title = filename.replace(".png", "").replace("_", " ")
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}](plots/{filename})")
        lines.append("")

    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
