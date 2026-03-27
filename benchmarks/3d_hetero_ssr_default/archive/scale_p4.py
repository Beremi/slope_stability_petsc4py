from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "archive" else SCRIPT_DIR
ROOT = BENCHMARK_DIR.parents[1]
DEFAULT_MESH = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
DEFAULT_RANKS = (1, 2, 4, 8)
DEFAULT_KERNELS = ("legacy", "rows")
DEFAULT_CONSTITUTIVE_MODES = ("overlap",)
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "p4_scaling_step2"
DEFAULT_REPORT = SCRIPT_DIR / "report_p4_scaling_step2.md"
DEFAULT_CONSTITUTIVE_REPORT = SCRIPT_DIR / "report_p4_constitutive_modes.md"


def _load_progress_summary(out_dir: Path) -> dict[str, int]:
    progress_path = out_dir / "data" / "progress.jsonl"
    init_accepted_states = 0
    final_accepted_states = 0
    attempt_count = 0
    successful_attempt_count = 0
    if not progress_path.exists():
        return {
            "init_accepted_states": init_accepted_states,
            "final_accepted_states": final_accepted_states,
            "attempt_count": attempt_count,
            "successful_attempt_count": successful_attempt_count,
        }

    for raw_line in progress_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        event = json.loads(line)
        kind = str(event.get("event", ""))
        if kind == "init_complete":
            init_accepted_states = int(event.get("accepted_steps", init_accepted_states))
        elif kind == "attempt_complete":
            attempt_count += 1
            if bool(event.get("success", False)):
                successful_attempt_count += 1
        elif kind == "finished":
            final_accepted_states = int(event.get("accepted_steps", final_accepted_states))

    return {
        "init_accepted_states": init_accepted_states,
        "final_accepted_states": final_accepted_states,
        "attempt_count": attempt_count,
        "successful_attempt_count": successful_attempt_count,
    }


def _run_case(
    *,
    ranks: int,
    mesh_path: Path,
    step_max: int,
    tangent_kernel: str,
    constitutive_mode: str,
    out_dir: Path,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        "mpiexec",
        "-n",
        str(int(ranks)),
        sys.executable,
        "-m",
        "slope_stability.cli.run_3D_hetero_SSR_capture",
        "--mesh_path",
        str(mesh_path),
        "--elem_type",
        "P4",
        "--step_max",
        str(int(step_max)),
        "--tangent_kernel",
        str(tangent_kernel),
        "--constitutive_mode",
        str(constitutive_mode),
        "--out_dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _load_case_metrics(out_dir: Path) -> dict[str, object]:
    run_info = json.loads((out_dir / "data" / "run_info.json").read_text(encoding="utf-8"))
    progress = _load_progress_summary(out_dir)
    with np.load(out_dir / "data" / "petsc_run.npz", allow_pickle=True) as npz:
        lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
        omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
        umax_hist = np.asarray(npz["Umax_hist"], dtype=np.float64)
        step_idx = np.asarray(npz.get("stats_step_index", np.empty(0)), dtype=np.int64)
        step_linear_iterations = np.asarray(npz.get("stats_step_linear_iterations", np.empty(0)), dtype=np.int64)
        step_newton_iterations = np.asarray(npz.get("stats_step_newton_iterations", np.empty(0)), dtype=np.int64)
        step_wall_time = np.asarray(npz.get("stats_step_wall_time", np.empty(0)), dtype=np.float64)
        step_lambda = np.asarray(npz.get("stats_step_lambda", np.empty(0)), dtype=np.float64)
        step_omega = np.asarray(npz.get("stats_step_omega", np.empty(0)), dtype=np.float64)

    info = run_info["run_info"]
    timings = run_info["timings"]
    linear = timings["linear"]
    constitutive = timings["constitutive"]
    final_accepted_states = int(progress["final_accepted_states"] or info["step_count"])
    init_accepted_states = int(progress["init_accepted_states"])
    return {
        "runtime_seconds": float(info["runtime_seconds"]),
        "mpi_size": int(info["mpi_size"]),
        "mesh_nodes": int(info["mesh_nodes"]),
        "mesh_elements": int(info["mesh_elements"]),
        "unknowns": int(info["unknowns"]),
        "step_count": int(info["step_count"]),
        "init_accepted_states": init_accepted_states,
        "final_accepted_states": final_accepted_states,
        "accepted_continuation_advances": int(max(0, final_accepted_states - init_accepted_states)),
        "attempt_count": int(progress["attempt_count"]),
        "successful_attempt_count": int(progress["successful_attempt_count"]),
        "recorded_step_stats": int(step_idx.size),
        "lambda_last": float(lambda_hist[-1]),
        "omega_last": float(omega_hist[-1]),
        "umax_last": float(umax_hist[-1]),
        "step_lambda": step_lambda.tolist(),
        "step_omega": step_omega.tolist(),
        "step_wall_time_total": float(step_wall_time.sum()) if step_wall_time.size else 0.0,
        "step_linear_iterations_total": int(step_linear_iterations.sum()) if step_linear_iterations.size else 0,
        "step_newton_iterations_total": int(step_newton_iterations.sum()) if step_newton_iterations.size else 0,
        "init_linear_iterations": int(linear["init_linear_iterations"]),
        "init_linear_solve_time": float(linear["init_linear_solve_time"]),
        "init_linear_preconditioner_time": float(linear["init_linear_preconditioner_time"]),
        "init_linear_orthogonalization_time": float(linear["init_linear_orthogonalization_time"]),
        "attempt_linear_iterations_total": int(linear["attempt_linear_iterations_total"]),
        "attempt_linear_solve_time_total": float(linear["attempt_linear_solve_time_total"]),
        "attempt_linear_preconditioner_time_total": float(linear["attempt_linear_preconditioner_time_total"]),
        "attempt_linear_orthogonalization_time_total": float(linear["attempt_linear_orthogonalization_time_total"]),
        "build_tangent_local": float(constitutive["build_tangent_local"]),
        "build_F": float(constitutive["build_F"]),
        "local_strain": float(constitutive["local_strain"]),
        "local_constitutive": float(constitutive["local_constitutive"]),
        "local_constitutive_comm": float(constitutive.get("local_constitutive_comm", 0.0)),
        "continuation_total_wall_time": float(timings["continuation_total_wall_time"]),
    }


def _fmt_ratio(x: float) -> str:
    return f"{float(x):.3f}x"


def _mesh_label(mesh_path: Path) -> str:
    mesh_path = Path(mesh_path)
    if not mesh_path.is_absolute():
        return str(mesh_path)
    try:
        return str(mesh_path.relative_to(ROOT))
    except ValueError:
        return str(mesh_path)


def _write_csv(out_root: Path, *, results: dict[int, dict[str, object]]) -> Path:
    csv_path = out_root / "summary.csv"
    fieldnames = [
        "ranks",
        "runtime_seconds",
        "speedup_vs_rank1",
        "efficiency",
        "mpi_size",
        "mesh_nodes",
        "mesh_elements",
        "unknowns",
        "step_count",
        "init_accepted_states",
        "final_accepted_states",
        "accepted_continuation_advances",
        "attempt_count",
        "successful_attempt_count",
        "recorded_step_stats",
        "init_linear_iterations",
        "step_linear_iterations_total",
        "step_newton_iterations_total",
        "lambda_last",
        "omega_last",
        "umax_last",
        "delta_lambda_vs_rank1",
        "delta_omega_vs_rank1",
        "delta_umax_vs_rank1",
        "init_linear_solve_time",
        "attempt_linear_solve_time_total",
        "attempt_linear_preconditioner_time_total",
        "build_tangent_local",
        "build_F",
        "local_strain",
        "local_constitutive",
        "local_constitutive_comm",
        "continuation_total_wall_time",
        "step_wall_time_total",
    ]
    rank1 = results[min(results)]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for ranks in sorted(results):
            row = dict(results[ranks])
            runtime = float(row["runtime_seconds"])
            speedup = float(rank1["runtime_seconds"]) / runtime
            efficiency = speedup / float(ranks)
            writer.writerow(
                {
                    "ranks": ranks,
                    "runtime_seconds": runtime,
                    "speedup_vs_rank1": speedup,
                    "efficiency": efficiency,
                    "mpi_size": int(row["mpi_size"]),
                    "mesh_nodes": int(row["mesh_nodes"]),
                    "mesh_elements": int(row["mesh_elements"]),
                    "unknowns": int(row["unknowns"]),
                    "step_count": int(row["step_count"]),
                    "init_accepted_states": int(row["init_accepted_states"]),
                    "final_accepted_states": int(row["final_accepted_states"]),
                    "accepted_continuation_advances": int(row["accepted_continuation_advances"]),
                    "attempt_count": int(row["attempt_count"]),
                    "successful_attempt_count": int(row["successful_attempt_count"]),
                    "recorded_step_stats": int(row["recorded_step_stats"]),
                    "init_linear_iterations": int(row["init_linear_iterations"]),
                    "step_linear_iterations_total": int(row["step_linear_iterations_total"]),
                    "step_newton_iterations_total": int(row["step_newton_iterations_total"]),
                    "lambda_last": float(row["lambda_last"]),
                    "omega_last": float(row["omega_last"]),
                    "umax_last": float(row["umax_last"]),
                    "delta_lambda_vs_rank1": abs(float(row["lambda_last"]) - float(rank1["lambda_last"])),
                    "delta_omega_vs_rank1": abs(float(row["omega_last"]) - float(rank1["omega_last"])),
                    "delta_umax_vs_rank1": abs(float(row["umax_last"]) - float(rank1["umax_last"])),
                    "init_linear_solve_time": float(row["init_linear_solve_time"]),
                    "attempt_linear_solve_time_total": float(row["attempt_linear_solve_time_total"]),
                    "attempt_linear_preconditioner_time_total": float(row["attempt_linear_preconditioner_time_total"]),
                    "build_tangent_local": float(row["build_tangent_local"]),
                    "build_F": float(row["build_F"]),
                    "local_strain": float(row["local_strain"]),
                    "local_constitutive": float(row["local_constitutive"]),
                    "local_constitutive_comm": float(row["local_constitutive_comm"]),
                    "continuation_total_wall_time": float(row["continuation_total_wall_time"]),
                    "step_wall_time_total": float(row["step_wall_time_total"]),
                }
            )
    return csv_path


def _write_json(
    out_root: Path,
    *,
    mesh_path: Path,
    step_max: int,
    tangent_kernel: str,
    constitutive_mode: str,
    results: dict[int, dict[str, object]],
) -> Path:
    json_path = out_root / "summary.json"
    rank1 = results[min(results)]
    payload = {
        "mesh_path": _mesh_label(mesh_path),
        "element_order": "P4",
        "tangent_kernel": str(tangent_kernel),
        "constitutive_mode": str(constitutive_mode),
        "step_max": int(step_max),
        "ranks": sorted(results),
        "rank1_reference": min(results),
        "cases": {},
    }
    for ranks in sorted(results):
        row = dict(results[ranks])
        runtime = float(row["runtime_seconds"])
        payload["cases"][str(ranks)] = {
            **row,
            "speedup_vs_rank1": float(rank1["runtime_seconds"]) / runtime,
            "efficiency": (float(rank1["runtime_seconds"]) / runtime) / float(ranks),
            "delta_lambda_vs_rank1": abs(float(row["lambda_last"]) - float(rank1["lambda_last"])),
            "delta_omega_vs_rank1": abs(float(row["omega_last"]) - float(rank1["omega_last"])),
            "delta_umax_vs_rank1": abs(float(row["umax_last"]) - float(rank1["umax_last"])),
        }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json_path


def _write_plots(out_root: Path, *, results: dict[int, dict[str, object]]) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    ranks = np.asarray(sorted(results), dtype=np.int64)
    rank1 = results[int(ranks.min())]
    runtime = np.asarray([float(results[int(r)]["runtime_seconds"]) for r in ranks], dtype=np.float64)
    speedup = float(rank1["runtime_seconds"]) / runtime
    efficiency = speedup / ranks.astype(np.float64)

    runtime_speedup_path = plot_dir / "runtime_speedup.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(ranks, runtime, marker="o", linewidth=2)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(ranks)
    axes[0].set_xticklabels([str(v) for v in ranks])
    axes[0].set_xlabel("MPI ranks")
    axes[0].set_ylabel("Runtime [s]")
    axes[0].set_title("P4 Runtime")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ranks, speedup, marker="o", linewidth=2, label="Measured speedup")
    axes[1].plot(ranks, ranks / ranks.min(), linestyle="--", linewidth=1.5, label="Ideal speedup")
    axes[1].plot(ranks, efficiency, marker="s", linewidth=2, label="Efficiency")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(ranks)
    axes[1].set_xticklabels([str(v) for v in ranks])
    axes[1].set_xlabel("MPI ranks")
    axes[1].set_ylabel("Relative factor")
    axes[1].set_title("Strong Scaling")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.savefig(runtime_speedup_path, dpi=180)
    plt.close(fig)

    solution_reach_path = plot_dir / "solution_reach.png"
    lambda_vals = np.asarray([float(results[int(r)]["lambda_last"]) for r in ranks], dtype=np.float64)
    omega_vals = np.asarray([float(results[int(r)]["omega_last"]) for r in ranks], dtype=np.float64)
    umax_vals = np.asarray([float(results[int(r)]["umax_last"]) for r in ranks], dtype=np.float64)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    series = [
        ("Final lambda", lambda_vals, abs(lambda_vals - float(rank1["lambda_last"]))),
        ("Final omega", omega_vals, abs(omega_vals - float(rank1["omega_last"]))),
        ("Final Umax", umax_vals, abs(umax_vals - float(rank1["umax_last"]))),
    ]
    for col, (title, values, deltas) in enumerate(series):
        axes[0, col].plot(ranks, values, marker="o", linewidth=2)
        axes[0, col].set_xscale("log", base=2)
        axes[0, col].set_xticks(ranks)
        axes[0, col].set_xticklabels([str(v) for v in ranks])
        axes[0, col].set_title(title)
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].set_xlabel("MPI ranks")

        axes[1, col].plot(ranks, deltas, marker="o", linewidth=2)
        axes[1, col].set_yscale("log")
        axes[1, col].set_xscale("log", base=2)
        axes[1, col].set_xticks(ranks)
        axes[1, col].set_xticklabels([str(v) for v in ranks])
        axes[1, col].set_title("|delta vs rank1|")
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].set_xlabel("MPI ranks")
    fig.savefig(solution_reach_path, dpi=180)
    plt.close(fig)

    timing_breakdown_path = plot_dir / "timing_breakdown.png"
    timing_series = {
        "init solve": np.asarray([float(results[int(r)]["init_linear_solve_time"]) for r in ranks], dtype=np.float64),
        "attempt solves": np.asarray([float(results[int(r)]["attempt_linear_solve_time_total"]) for r in ranks], dtype=np.float64),
        "attempt prec": np.asarray([float(results[int(r)]["attempt_linear_preconditioner_time_total"]) for r in ranks], dtype=np.float64),
        "tangent local": np.asarray([float(results[int(r)]["build_tangent_local"]) for r in ranks], dtype=np.float64),
        "constitutive": np.asarray([float(results[int(r)]["local_constitutive"]) for r in ranks], dtype=np.float64),
        "constitutive comm": np.asarray([float(results[int(r)]["local_constitutive_comm"]) for r in ranks], dtype=np.float64),
    }
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bottom = np.zeros_like(ranks, dtype=np.float64)
    for label, values in timing_series.items():
        ax.bar([str(v) for v in ranks], values, bottom=bottom, label=label)
        bottom += values
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Time [s]")
    ax.set_title("Dominant Timing Components")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(timing_breakdown_path, dpi=180)
    plt.close(fig)

    return {
        "runtime_speedup": runtime_speedup_path,
        "solution_reach": solution_reach_path,
        "timing_breakdown": timing_breakdown_path,
    }


def _write_report(
    report_path: Path,
    *,
    mesh_path: Path,
    step_max: int,
    tangent_kernel: str,
    constitutive_mode: str,
    results: dict[int, dict[str, object]],
) -> None:
    rank1 = results[min(results)]
    best_rank = min(results, key=lambda ranks: float(results[ranks]["runtime_seconds"]))
    max_delta_lambda = max(abs(float(row["lambda_last"]) - float(rank1["lambda_last"])) for row in results.values())
    max_delta_omega = max(abs(float(row["omega_last"]) - float(rank1["omega_last"])) for row in results.values())
    max_delta_umax = max(abs(float(row["umax_last"]) - float(rank1["umax_last"])) for row in results.values())
    lines: list[str] = [
        "# 3D Hetero SSR P4 Scaling",
        "",
        f"- Mesh: `{_mesh_label(mesh_path)}`",
        f"- Element order: `P4`",
        f"- Tangent kernel: `{tangent_kernel}`",
        f"- Constitutive mode: `{constitutive_mode}`",
        f"- `step_max`: `{int(step_max)}`",
        f"- Ranks tested: `{', '.join(str(r) for r in sorted(results))}`",
        "- Runner: `slope_stability.cli.run_3D_hetero_SSR_capture`",
        f"- Mesh nodes: `{int(rank1['mesh_nodes'])}`",
        f"- Mesh elements: `{int(rank1['mesh_elements'])}`",
        f"- Unknowns: `{int(rank1['unknowns'])}`",
        "",
        "## Strong Scaling",
        "",
        "| Ranks | Runtime [s] | Speedup vs 1 | Efficiency | Final accepted states | Continuation advances | Init lin iters | Step lin iters | Step Newton iters |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ranks in sorted(results):
        row = results[ranks]
        speedup = float(rank1["runtime_seconds"]) / float(row["runtime_seconds"])
        efficiency = speedup / float(ranks)
        lines.append(
            f"| {ranks} | {float(row['runtime_seconds']):.3f} | {_fmt_ratio(speedup)} | {efficiency:.3f} | "
            f"{int(row['final_accepted_states'])} | {int(row['accepted_continuation_advances'])} | "
            f"{int(row['init_linear_iterations'])} | {int(row['step_linear_iterations_total'])} | "
            f"{int(row['step_newton_iterations_total'])} |"
        )

    lines.extend(
        [
            "",
            "## Continuation Reach",
            "",
            "| Ranks | Init accepted states | Final accepted states | Successful continuation attempts |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for ranks in sorted(results):
        row = results[ranks]
        lines.append(
            f"| {ranks} | {int(row['init_accepted_states'])} | {int(row['final_accepted_states'])} | "
            f"{int(row['successful_attempt_count'])} / {int(row['attempt_count'])} |"
        )

    lines.extend(
        [
            "",
            "## Solution Reach",
            "",
            "| Ranks | Final lambda | Final omega | Final Umax | |lambda-rank1| | |omega-rank1| | |Umax-rank1| |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for ranks in sorted(results):
        row = results[ranks]
        lines.append(
            f"| {ranks} | {float(row['lambda_last']):.9f} | {float(row['omega_last']):.9f} | {float(row['umax_last']):.9f} | "
            f"{abs(float(row['lambda_last']) - float(rank1['lambda_last'])):.3e} | "
            f"{abs(float(row['omega_last']) - float(rank1['omega_last'])):.3e} | "
            f"{abs(float(row['umax_last']) - float(rank1['umax_last'])):.3e} |"
        )

    lines.extend(
        [
            "",
            "## Timing Breakdown",
            "",
            "| Ranks | Init solve [s] | Attempt solves [s] | Attempt prec [s] | Tangent local [s] | Build F [s] | Local strain [s] | Local constitutive [s] | Constitutive comm [s] |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for ranks in sorted(results):
        row = results[ranks]
        lines.append(
            f"| {ranks} | {float(row['init_linear_solve_time']):.3f} | {float(row['attempt_linear_solve_time_total']):.3f} | "
            f"{float(row['attempt_linear_preconditioner_time_total']):.3f} | {float(row['build_tangent_local']):.3f} | "
            f"{float(row['build_F']):.3f} | {float(row['local_strain']):.3f} | {float(row['local_constitutive']):.3f} | "
            f"{float(row['local_constitutive_comm']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Strong-scaling speedup and efficiency are computed against the `1`-rank runtime.",
            f"- Best runtime in this sweep was at `{best_rank}` ranks.",
            "- Each run started from `2` accepted initialization states (`lambda = 1.0` and `lambda = 1.1`) and then accepted `1` continuation advance; allowing `2` continuation steps did not produce a second accepted advance on this configuration.",
            f"- Maximum solution drift across ranks was `|delta lambda| = {max_delta_lambda:.3e}`, `|delta omega| = {max_delta_omega:.3e}`, and `|delta Umax| = {max_delta_umax:.3e}` relative to rank `1`.",
            "- The elevated mesh and unknown count are constant across ranks; only the parallel decomposition changes.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_kernel_comparison_report(
    report_path: Path,
    *,
    mesh_path: Path,
    step_max: int,
    constitutive_mode: str,
    results_by_kernel: dict[str, dict[int, dict[str, object]]],
) -> None:
    kernels = list(results_by_kernel)
    lines = [
        "# 3D Hetero SSR P4 Kernel Comparison",
        "",
        f"- Mesh: `{_mesh_label(mesh_path)}`",
        "- Element order: `P4`",
        f"- `step_max`: `{int(step_max)}`",
        f"- Constitutive mode: `{constitutive_mode}`",
        f"- Kernels: `{', '.join(kernels)}`",
        "",
    ]

    for kernel in kernels:
        results = results_by_kernel[kernel]
        rank1 = results[min(results)]
        best_rank = min(results, key=lambda ranks: float(results[ranks]["runtime_seconds"]))
        lines.extend(
            [
                f"## `{kernel}`",
                "",
                f"- Rank-1 runtime: `{float(rank1['runtime_seconds']):.3f}` s",
                f"- Rank-1 `build_tangent_local`: `{float(rank1['build_tangent_local']):.3f}` s",
                f"- Rank-1 `local_constitutive_comm`: `{float(rank1['local_constitutive_comm']):.3f}` s",
                f"- Best runtime rank: `{best_rank}`",
                "",
            ]
        )

    baseline = "legacy" if "legacy" in results_by_kernel else kernels[0]
    if "rows" in results_by_kernel and baseline in results_by_kernel:
        common_ranks = sorted(set(results_by_kernel[baseline]) & set(results_by_kernel["rows"]))
        lines.extend(
            [
                f"## `rows` vs `{baseline}`",
                "",
                "| Ranks | Runtime baseline [s] | Runtime rows [s] | Runtime speedup | Tangent baseline [s] | Tangent rows [s] | Tangent speedup | |delta lambda| | |delta omega| | |delta Umax| |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for ranks in common_ranks:
            base = results_by_kernel[baseline][ranks]
            rows = results_by_kernel["rows"][ranks]
            runtime_speedup = float(base["runtime_seconds"]) / max(float(rows["runtime_seconds"]), 1.0e-30)
            tangent_speedup = float(base["build_tangent_local"]) / max(float(rows["build_tangent_local"]), 1.0e-30)
            lines.append(
                f"| {ranks} | {float(base['runtime_seconds']):.3f} | {float(rows['runtime_seconds']):.3f} | "
                f"{_fmt_ratio(runtime_speedup)} | {float(base['build_tangent_local']):.3f} | {float(rows['build_tangent_local']):.3f} | "
                f"{_fmt_ratio(tangent_speedup)} | {abs(float(rows['lambda_last']) - float(base['lambda_last'])):.3e} | "
                f"{abs(float(rows['omega_last']) - float(base['omega_last'])):.3e} | {abs(float(rows['umax_last']) - float(base['umax_last'])):.3e} |"
            )

    lines.extend(
        [
            "",
            "## Artifact Layout",
            "",
            "- Per-kernel summaries, plots, and rank subdirectories are written under `kernel_<name>/` inside the output root.",
            "- This root report compares the kernels directly while preserving the original per-kernel scaling artifacts.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_constitutive_comparison_report(
    report_path: Path,
    *,
    mesh_path: Path,
    step_max: int,
    tangent_kernel: str,
    results_by_mode: dict[str, dict[int, dict[str, object]]],
) -> None:
    constitutive_modes = list(results_by_mode)
    lines = [
        "# 3D Hetero SSR P4 Constitutive Comparison",
        "",
        f"- Mesh: `{_mesh_label(mesh_path)}`",
        "- Element order: `P4`",
        f"- Tangent kernel: `{tangent_kernel}`",
        f"- `step_max`: `{int(step_max)}`",
        f"- Constitutive modes: `{', '.join(constitutive_modes)}`",
        "",
    ]

    for constitutive_mode in constitutive_modes:
        results = results_by_mode[constitutive_mode]
        rank1 = results[min(results)]
        best_rank = min(results, key=lambda ranks: float(results[ranks]["runtime_seconds"]))
        lines.extend(
            [
                f"## `{constitutive_mode}`",
                "",
                f"- Rank-1 runtime: `{float(rank1['runtime_seconds']):.3f}` s",
                f"- Rank-1 `local_constitutive`: `{float(rank1['local_constitutive']):.3f}` s",
                f"- Rank-1 `local_constitutive_comm`: `{float(rank1['local_constitutive_comm']):.3f}` s",
                f"- Best runtime rank: `{best_rank}`",
                "",
            ]
        )

    baseline = "overlap" if "overlap" in results_by_mode else constitutive_modes[0]
    if "unique_exchange" in results_by_mode and baseline in results_by_mode:
        common_ranks = sorted(set(results_by_mode[baseline]) & set(results_by_mode["unique_exchange"]))
        lines.extend(
            [
                f"## `unique_exchange` vs `{baseline}`",
                "",
                "| Ranks | Runtime baseline [s] | Runtime unique_exchange [s] | Runtime speedup | Local constitutive baseline [s] | Local constitutive unique_exchange [s] | Local comm baseline [s] | Local comm unique_exchange [s] | |delta lambda| | |delta omega| | |delta Umax| |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for ranks in common_ranks:
            base = results_by_mode[baseline][ranks]
            exchange = results_by_mode["unique_exchange"][ranks]
            runtime_speedup = float(base["runtime_seconds"]) / max(float(exchange["runtime_seconds"]), 1.0e-30)
            lines.append(
                f"| {ranks} | {float(base['runtime_seconds']):.3f} | {float(exchange['runtime_seconds']):.3f} | "
                f"{_fmt_ratio(runtime_speedup)} | {float(base['local_constitutive']):.3f} | {float(exchange['local_constitutive']):.3f} | "
                f"{float(base['local_constitutive_comm']):.3f} | {float(exchange['local_constitutive_comm']):.3f} | "
                f"{abs(float(exchange['lambda_last']) - float(base['lambda_last'])):.3e} | "
                f"{abs(float(exchange['omega_last']) - float(base['omega_last'])):.3e} | "
                f"{abs(float(exchange['umax_last']) - float(base['umax_last'])):.3e} |"
            )

    lines.extend(
        [
            "",
            "## Artifact Layout",
            "",
            "- Per-mode results are written under `mode_<name>/kernel_<name>/` when multiple constitutive modes are requested.",
            "- This report compares constitutive modes with the tangent kernel held fixed.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a P4 strong-scaling study for 3D hetero SSR.")
    parser.add_argument("--mesh-path", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--step-max", type=int, default=2)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--constitutive-report-path", type=Path, default=DEFAULT_CONSTITUTIVE_REPORT)
    parser.add_argument("--reuse-existing", action="store_true", default=False)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS))
    parser.add_argument("--kernels", type=str, nargs="+", default=list(DEFAULT_KERNELS))
    parser.add_argument("--constitutive-modes", type=str, nargs="+", default=list(DEFAULT_CONSTITUTIVE_MODES))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    kernels = [str(v).lower() for v in args.kernels]
    constitutive_modes = [str(v).lower() for v in args.constitutive_modes]
    results_by_mode: dict[str, dict[str, dict[int, dict[str, object]]]] = {}

    for constitutive_mode in constitutive_modes:
        mode_root = out_root if len(constitutive_modes) == 1 else out_root / f"mode_{constitutive_mode}"
        mode_root.mkdir(parents=True, exist_ok=True)
        results_by_kernel: dict[str, dict[int, dict[str, object]]] = {}
        for kernel in kernels:
            kernel_root = mode_root / f"kernel_{kernel}"
            kernel_root.mkdir(parents=True, exist_ok=True)
            results: dict[int, dict[str, object]] = {}
            for ranks in [int(v) for v in args.ranks]:
                out_dir = kernel_root / f"rank{ranks}"
                info_path = out_dir / "data" / "run_info.json"
                if args.reuse_existing and info_path.exists():
                    results[ranks] = _load_case_metrics(out_dir)
                    continue
                _run_case(
                    ranks=ranks,
                    mesh_path=Path(args.mesh_path),
                    step_max=int(args.step_max),
                    tangent_kernel=kernel,
                    constitutive_mode=constitutive_mode,
                    out_dir=out_dir,
                )
                results[ranks] = _load_case_metrics(out_dir)

            results_by_kernel[kernel] = results
            _write_csv(kernel_root, results=results)
            _write_json(
                kernel_root,
                mesh_path=Path(args.mesh_path),
                step_max=int(args.step_max),
                tangent_kernel=kernel,
                constitutive_mode=constitutive_mode,
                results=results,
            )
            _write_plots(kernel_root, results=results)
            _write_report(
                kernel_root / "report.md",
                mesh_path=Path(args.mesh_path),
                step_max=int(args.step_max),
                tangent_kernel=kernel,
                constitutive_mode=constitutive_mode,
                results=results,
            )

        results_by_mode[constitutive_mode] = results_by_kernel
        if len(kernels) > 1:
            kernel_report_path = (
                Path(args.report_path) if len(constitutive_modes) == 1 else mode_root / "kernel_comparison.md"
            )
            _write_kernel_comparison_report(
                kernel_report_path,
                mesh_path=Path(args.mesh_path),
                step_max=int(args.step_max),
                constitutive_mode=constitutive_mode,
                results_by_kernel=results_by_kernel,
            )

    if len(constitutive_modes) > 1:
        for kernel in kernels:
            results_for_kernel = {
                constitutive_mode: results_by_mode[constitutive_mode][kernel]
                for constitutive_mode in constitutive_modes
            }
            report_path = (
                Path(args.constitutive_report_path)
                if len(kernels) == 1
                else out_root / f"constitutive_comparison_kernel_{kernel}.md"
            )
            _write_constitutive_comparison_report(
                report_path,
                mesh_path=Path(args.mesh_path),
                step_max=int(args.step_max),
                tangent_kernel=kernel,
                results_by_mode=results_for_kernel,
            )


if __name__ == "__main__":
    main()
