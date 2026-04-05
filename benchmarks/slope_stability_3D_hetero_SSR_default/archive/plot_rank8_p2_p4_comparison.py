from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "artifacts" / "p4_pmg_shell_best_rank8_full"
PLOTS_DIR = OUT_ROOT / "plots_vs_p2"
REPORT_PATH = OUT_ROOT / "report_with_p2_and_old_p4.md"


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    color: str
    run_info_path: Path
    progress_path: Path
    npz_path: Path


RUNS = (
    RunSpec(
        key="p2_ref",
        label="P2 Hypre Ref",
        color="#1f77b4",
        run_info_path=ROOT
        / "artifacts"
        / "p2_p4_compare_rank8_final_guarded80_v2"
        / "p2_rank8_step100"
        / "data"
        / "run_info.json",
        progress_path=ROOT
        / "artifacts"
        / "p2_p4_compare_rank8_final_guarded80_v2"
        / "p2_rank8_step100"
        / "data"
        / "progress_latest.json",
        npz_path=ROOT
        / "artifacts"
        / "p2_p4_compare_rank8_final_guarded80_v2"
        / "p2_rank8_step100"
        / "data"
        / "petsc_run.npz",
    ),
    RunSpec(
        key="p4_hypre_old",
        label="P4 Hypre Old",
        color="#d62728",
        run_info_path=ROOT
        / "artifacts"
        / "p2_p4_compare_rank8_final_guarded80_v2"
        / "p4_rank8_step100"
        / "data"
        / "run_info.json",
        progress_path=ROOT
        / "artifacts"
        / "p2_p4_compare_rank8_final_guarded80_v2"
        / "p4_rank8_step100"
        / "data"
        / "progress_latest.json",
        npz_path=ROOT
        / "artifacts"
        / "p2_p4_compare_rank8_final_guarded80_v2"
        / "p4_rank8_step100"
        / "data"
        / "petsc_run.npz",
    ),
    RunSpec(
        key="p4_pmg_shell_new",
        label="P4 PMG-Shell New",
        color="#2ca02c",
        run_info_path=ROOT
        / "artifacts"
        / "p4_pmg_shell_best_rank8_full"
        / "p4_rank8_step100"
        / "data"
        / "run_info.json",
        progress_path=ROOT
        / "artifacts"
        / "p4_pmg_shell_best_rank8_full"
        / "p4_rank8_step100"
        / "data"
        / "progress_latest.json",
        npz_path=ROOT
        / "artifacts"
        / "p4_pmg_shell_best_rank8_full"
        / "p4_rank8_step100"
        / "data"
        / "petsc_run.npz",
    ),
)


def _load_run(spec: RunSpec) -> dict[str, object]:
    run_info = json.loads(spec.run_info_path.read_text(encoding="utf-8"))
    progress = json.loads(spec.progress_path.read_text(encoding="utf-8"))
    with np.load(spec.npz_path, allow_pickle=True) as npz:
        lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
        omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
        step_index = np.asarray(npz["stats_step_index"], dtype=np.int64)
        step_wall_time = np.asarray(npz["stats_step_wall_time"], dtype=np.float64)
        step_linear_iterations = np.asarray(npz["stats_step_linear_iterations"], dtype=np.int64)
        step_newton_iterations = np.asarray(npz["stats_step_newton_iterations"], dtype=np.int64)
        step_lambda = np.asarray(npz["stats_step_lambda"], dtype=np.float64)
        step_omega = np.asarray(npz["stats_step_omega"], dtype=np.float64)

    cumulative_linear_iterations = np.cumsum(step_linear_iterations, dtype=np.int64)
    return {
        "spec": spec,
        "run_info": run_info,
        "progress": progress,
        "lambda_hist": lambda_hist,
        "omega_hist": omega_hist,
        "step_index": step_index,
        "step_wall_time": step_wall_time,
        "step_linear_iterations": step_linear_iterations,
        "step_newton_iterations": step_newton_iterations,
        "step_lambda": step_lambda,
        "step_omega": step_omega,
        "cumulative_linear_iterations": cumulative_linear_iterations,
    }


def _make_plot(
    *,
    runs: list[dict[str, object]],
    y_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=150)
    all_steps: set[int] = set()
    for run in runs:
        spec = run["spec"]
        x = np.asarray(run["step_index"], dtype=np.int64)
        y = np.asarray(run[y_key])
        all_steps.update(int(v) for v in x.tolist())
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=spec.label,
            color=spec.color,
        )
    ax.set_title(title)
    ax.set_xlabel("Accepted Continuation Step")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(all_steps))
    ax.grid(True, alpha=0.3)
    if logy:
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _make_continuation_plot(runs: list[dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=150)
    for run in runs:
        spec = run["spec"]
        omega_hist = np.asarray(run["omega_hist"], dtype=np.float64)
        lambda_hist = np.asarray(run["lambda_hist"], dtype=np.float64)
        ax.plot(
            omega_hist,
            lambda_hist,
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            label=spec.label,
            color=spec.color,
        )
    ax.set_title("Continuation Curve")
    ax.set_xlabel("Omega")
    ax.set_ylabel("Lambda")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _make_combined_plot(runs: list[dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 13.0), dpi=150, sharex=True)
    configs = (
        ("step_wall_time", "Step Wall Time [s]", "Accepted Step Wall Time", True),
        ("step_newton_iterations", "Newton Iterations", "Newton Iterations per Accepted Step", False),
        ("cumulative_linear_iterations", "Cumulative Linear Iterations", "Cumulative Linear Iterations", False),
    )

    all_steps: set[int] = set()
    for ax, (key, ylabel, title, logy) in zip(axes, configs, strict=True):
        for run in runs:
            spec = run["spec"]
            x = np.asarray(run["step_index"], dtype=np.int64)
            y = np.asarray(run[key])
            all_steps.update(int(v) for v in x.tolist())
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                label=spec.label,
                color=spec.color,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if logy:
            ax.set_yscale("log")

    axes[-1].set_xlabel("Accepted Continuation Step")
    axes[-1].set_xticks(sorted(all_steps))
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _fmt_float(value: float) -> str:
    return f"{float(value):.3f}"


def _step_value(run: dict[str, object], step: int, key: str) -> str:
    step_index = np.asarray(run["step_index"], dtype=np.int64)
    values = np.asarray(run[key])
    matches = np.where(step_index == int(step))[0]
    if matches.size == 0:
        return "-"
    value = values[int(matches[0])]
    if np.issubdtype(values.dtype, np.integer):
        return str(int(value))
    return _fmt_float(float(value))


def _step_numeric_value(run: dict[str, object], step: int, key: str) -> float | int | None:
    step_index = np.asarray(run["step_index"], dtype=np.int64)
    values = np.asarray(run[key])
    matches = np.where(step_index == int(step))[0]
    if matches.size == 0:
        return None
    value = values[int(matches[0])]
    if np.issubdtype(values.dtype, np.integer):
        return int(value)
    return float(value)


def _write_report(runs: list[dict[str, object]]) -> None:
    by_key = {run["spec"].key: run for run in runs}
    p2 = by_key["p2_ref"]
    p4_old = by_key["p4_hypre_old"]
    p4_new = by_key["p4_pmg_shell_new"]

    steps = sorted(
        {
            *(int(v) for v in np.asarray(p2["step_index"], dtype=np.int64).tolist()),
            *(int(v) for v in np.asarray(p4_old["step_index"], dtype=np.int64).tolist()),
            *(int(v) for v in np.asarray(p4_new["step_index"], dtype=np.int64).tolist()),
        }
    )

    lines: list[str] = []
    lines.append("# Rank-8 P2 / P4 Comparison With Step And Continuation Graphs")
    lines.append("")
    lines.append("- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`")
    lines.append("- MPI ranks: `8`")
    lines.append("- Runs compared:")
    lines.append("  - archived `P2` reference")
    lines.append("  - archived `P4` Hypre full run")
    lines.append("  - new `P4 pmg_shell` full run with coarse `BoomerAMG max_iter=4`, `tol=0.0`")
    lines.append("")
    lines.append("## Final Summary")
    lines.append("")
    lines.append("| Run | Accepted states | Final lambda | Final omega | Runtime [s] | Init lin it | Continuation lin it |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run in runs:
        spec = run["spec"]
        top = run["run_info"]["run_info"]
        prog = run["progress"]
        linear = run["run_info"]["timings"]["linear"]
        lines.append(
            f"| {spec.label} | {int(prog['accepted_steps'])} | {float(prog['lambda_last']):.9f} | "
            f"{float(prog['omega_last']):.1f} | {float(top['runtime_seconds']):.3f} | "
            f"{int(linear['init_linear_iterations'])} | {int(linear['attempt_linear_iterations_total'])} |"
        )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append("![Continuation curve](plots_vs_p2/continuation_curve.png)")
    lines.append("")
    lines.append("![Combined comparison](plots_vs_p2/combined_step_metrics.png)")
    lines.append("")
    lines.append("![Step wall time](plots_vs_p2/step_wall_time.png)")
    lines.append("")
    lines.append("![Newton iterations](plots_vs_p2/step_newton_iterations.png)")
    lines.append("")
    lines.append("![Cumulative linear iterations](plots_vs_p2/cumulative_linear_iterations.png)")
    lines.append("")
    lines.append("## Accepted-Step Wall Time")
    lines.append("")
    lines.append("| Step | P2 [s] | P4 Hypre [s] | P4 PMG-Shell [s] |")
    lines.append("| --- | ---: | ---: | ---: |")
    for step in steps:
        lines.append(
            f"| {step} | {_step_value(p2, step, 'step_wall_time')} | "
            f"{_step_value(p4_old, step, 'step_wall_time')} | {_step_value(p4_new, step, 'step_wall_time')} |"
        )
    lines.append("")
    lines.append("## Step-Wall-Time Slowdown vs P2")
    lines.append("")
    lines.append("| Step | P2 [s] | P4 Hypre / P2 | P4 PMG-Shell / P2 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for step in steps:
        p2_wall = _step_numeric_value(p2, step, "step_wall_time")
        p4_old_wall = _step_numeric_value(p4_old, step, "step_wall_time")
        p4_new_wall = _step_numeric_value(p4_new, step, "step_wall_time")
        old_ratio = "-" if p2_wall in {None, 0} or p4_old_wall is None else f"{float(p4_old_wall) / float(p2_wall):.3f}x"
        new_ratio = "-" if p2_wall in {None, 0} or p4_new_wall is None else f"{float(p4_new_wall) / float(p2_wall):.3f}x"
        p2_wall_str = "-" if p2_wall is None else _fmt_float(float(p2_wall))
        lines.append(f"| {step} | {p2_wall_str} | {old_ratio} | {new_ratio} |")
    lines.append("")
    lines.append("## Newton Iterations Per Accepted Step")
    lines.append("")
    lines.append("| Step | P2 | P4 Hypre | P4 PMG-Shell |")
    lines.append("| --- | ---: | ---: | ---: |")
    for step in steps:
        lines.append(
            f"| {step} | {_step_value(p2, step, 'step_newton_iterations')} | "
            f"{_step_value(p4_old, step, 'step_newton_iterations')} | {_step_value(p4_new, step, 'step_newton_iterations')} |"
        )
    lines.append("")
    lines.append("## Cumulative Linear Iterations")
    lines.append("")
    lines.append("| Step | P2 | P4 Hypre | P4 PMG-Shell |")
    lines.append("| --- | ---: | ---: | ---: |")
    for step in steps:
        lines.append(
            f"| {step} | {_step_value(p2, step, 'cumulative_linear_iterations')} | "
            f"{_step_value(p4_old, step, 'cumulative_linear_iterations')} | {_step_value(p4_new, step, 'cumulative_linear_iterations')} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- The step-wall-time graph uses a log y-axis so the `P2` curve remains visible next to the `P4` curves.")
    lines.append("- The new `P4 pmg_shell` run is much faster than old `P4 Hypre` on early accepted steps, then similar or slower on the late hard steps.")
    lines.append("- The new `P4 pmg_shell` run reaches one accepted step farther than the archived `P4` Hypre run and reaches the same `omega_max = 1.2e7` cap as the archived `P2` reference.")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = [_load_run(spec) for spec in RUNS]

    _make_plot(
        runs=runs,
        y_key="step_wall_time",
        ylabel="Accepted Step Wall Time [s]",
        title="Accepted Continuation Step Wall Time",
        out_path=PLOTS_DIR / "step_wall_time.png",
        logy=True,
    )
    _make_plot(
        runs=runs,
        y_key="step_newton_iterations",
        ylabel="Newton Iterations",
        title="Newton Iterations per Accepted Continuation Step",
        out_path=PLOTS_DIR / "step_newton_iterations.png",
        logy=False,
    )
    _make_plot(
        runs=runs,
        y_key="cumulative_linear_iterations",
        ylabel="Cumulative Linear Iterations",
        title="Cumulative Linear Iterations by Accepted Continuation Step",
        out_path=PLOTS_DIR / "cumulative_linear_iterations.png",
        logy=False,
    )
    _make_continuation_plot(runs, PLOTS_DIR / "continuation_curve.png")
    _make_combined_plot(runs, PLOTS_DIR / "combined_step_metrics.png")
    _write_report(runs)


if __name__ == "__main__":
    main()
