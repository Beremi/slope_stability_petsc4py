from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MESH = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
DEFAULT_RANKS = (1, 2, 4, 8, 16)
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "p4_scaling_step2"
DEFAULT_REPORT = Path(__file__).resolve().parent / "report_p4_scaling_step2.md"


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


def _run_case(*, ranks: int, mesh_path: Path, step_max: int, out_dir: Path) -> None:
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
        "continuation_total_wall_time": float(timings["continuation_total_wall_time"]),
    }


def _fmt_ratio(x: float) -> str:
    return f"{float(x):.3f}x"


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
                    "continuation_total_wall_time": float(row["continuation_total_wall_time"]),
                    "step_wall_time_total": float(row["step_wall_time_total"]),
                }
            )
    return csv_path


def _write_json(out_root: Path, *, mesh_path: Path, step_max: int, results: dict[int, dict[str, object]]) -> Path:
    json_path = out_root / "summary.json"
    rank1 = results[min(results)]
    payload = {
        "mesh_path": str(mesh_path.relative_to(ROOT)),
        "element_order": "P4",
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


def _write_report(report_path: Path, *, mesh_path: Path, step_max: int, results: dict[int, dict[str, object]]) -> None:
    rank1 = results[min(results)]
    best_rank = min(results, key=lambda ranks: float(results[ranks]["runtime_seconds"]))
    max_delta_lambda = max(abs(float(row["lambda_last"]) - float(rank1["lambda_last"])) for row in results.values())
    max_delta_omega = max(abs(float(row["omega_last"]) - float(rank1["omega_last"])) for row in results.values())
    max_delta_umax = max(abs(float(row["umax_last"]) - float(rank1["umax_last"])) for row in results.values())
    lines: list[str] = [
        "# 3D Hetero SSR P4 Scaling",
        "",
        f"- Mesh: `{mesh_path.relative_to(ROOT)}`",
        f"- Element order: `P4`",
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
            "| Ranks | Init solve [s] | Attempt solves [s] | Attempt prec [s] | Tangent local [s] | Build F [s] | Local strain [s] | Local constitutive [s] |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for ranks in sorted(results):
        row = results[ranks]
        lines.append(
            f"| {ranks} | {float(row['init_linear_solve_time']):.3f} | {float(row['attempt_linear_solve_time_total']):.3f} | "
            f"{float(row['attempt_linear_preconditioner_time_total']):.3f} | {float(row['build_tangent_local']):.3f} | "
            f"{float(row['build_F']):.3f} | {float(row['local_strain']):.3f} | {float(row['local_constitutive']):.3f} |"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a P4 strong-scaling study for 3D hetero SSR.")
    parser.add_argument("--mesh-path", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--step-max", type=int, default=2)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reuse-existing", action="store_true", default=False)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(DEFAULT_RANKS))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, object]] = {}
    for ranks in [int(v) for v in args.ranks]:
        out_dir = out_root / f"rank{ranks}"
        info_path = out_dir / "data" / "run_info.json"
        if args.reuse_existing and info_path.exists():
            results[ranks] = _load_case_metrics(out_dir)
            continue
        _run_case(ranks=ranks, mesh_path=Path(args.mesh_path), step_max=int(args.step_max), out_dir=out_dir)
        results[ranks] = _load_case_metrics(out_dir)

    _write_csv(out_root, results=results)
    _write_json(out_root, mesh_path=Path(args.mesh_path), step_max=int(args.step_max), results=results)
    _write_plots(out_root, results=results)
    _write_report(Path(args.report_path), mesh_path=Path(args.mesh_path), step_max=int(args.step_max), results=results)


if __name__ == "__main__":
    main()
