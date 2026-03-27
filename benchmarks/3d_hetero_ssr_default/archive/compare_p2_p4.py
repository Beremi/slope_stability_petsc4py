from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "archive" else SCRIPT_DIR
ROOT = BENCHMARK_DIR.parents[1]
DEFAULT_MESH = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "p2_p4_compare_rank8_final"
DEFAULT_REPORT = SCRIPT_DIR / "report_p2_vs_p4_rank8_final.md"


def _load_memory_guard_summary(path: Path | None) -> dict[str, float | int | bool | str] | None:
    if path is None or not path.exists():
        return None
    path = path.resolve()

    peak_rss_gib = 0.0
    min_available_gib = np.inf
    sample_count = 0
    guard_triggered = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        event = json.loads(line)
        sample_count += 1
        if str(event.get("event", "")) == "guard_triggered":
            guard_triggered = True
        if "rss_gib" in event:
            peak_rss_gib = max(peak_rss_gib, float(event["rss_gib"]))
        if "mem_available_gib" in event:
            min_available_gib = min(min_available_gib, float(event["mem_available_gib"]))

    if not np.isfinite(min_available_gib):
        min_available_gib = 0.0

    return {
        "path": str(path.relative_to(ROOT)),
        "peak_rss_gib": float(peak_rss_gib),
        "min_available_gib": float(min_available_gib),
        "sample_count": int(sample_count),
        "guard_triggered": bool(guard_triggered),
    }


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


def _run_order(
    order: str,
    *,
    mpi_ranks: int,
    mesh_path: Path,
    step_max: int,
    out_dir: Path,
    max_deflation_basis_vectors: int,
    store_step_u: bool,
    tangent_kernel: str,
    constitutive_mode: str,
) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        "mpiexec",
        "-n",
        str(int(mpi_ranks)),
        sys.executable,
        "-m",
        "slope_stability.cli.run_3D_hetero_SSR_capture",
        "--mesh_path",
        str(mesh_path),
        "--elem_type",
        str(order).upper(),
        "--tangent_kernel",
        str(tangent_kernel),
        "--constitutive_mode",
        str(constitutive_mode),
        "--step_max",
        str(int(step_max)),
        "--max_deflation_basis_vectors",
        str(int(max_deflation_basis_vectors)),
        "--store_step_u" if bool(store_step_u) else "--no-store_step_u",
        "--out_dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    return _load_metrics(out_dir)


def _load_metrics(out_dir: Path) -> dict[str, object]:
    run_info = json.loads((out_dir / "data" / "run_info.json").read_text(encoding="utf-8"))
    progress = _load_progress_summary(out_dir)
    with np.load(out_dir / "data" / "petsc_run.npz", allow_pickle=True) as npz:
        lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
        omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
        umax_hist = np.asarray(npz["Umax_hist"], dtype=np.float64)
        step_index = np.asarray(npz.get("stats_step_index", np.empty(0)), dtype=np.int64)
        step_lambda = np.asarray(npz.get("stats_step_lambda", np.empty(0)), dtype=np.float64)
        step_omega = np.asarray(npz.get("stats_step_omega", np.empty(0)), dtype=np.float64)
        step_linear_iterations = np.asarray(npz.get("stats_step_linear_iterations", np.empty(0)), dtype=np.int64)
        step_newton_iterations = np.asarray(npz.get("stats_step_newton_iterations", np.empty(0)), dtype=np.int64)
        step_wall_time = np.asarray(npz.get("stats_step_wall_time", np.empty(0)), dtype=np.float64)
        step_attempt_count = np.asarray(npz.get("stats_step_attempt_count", np.empty(0)), dtype=np.int64)

        attempt_step = np.asarray(npz.get("stats_attempt_step", np.empty(0)), dtype=np.int64)
        attempt_success = np.asarray(npz.get("stats_attempt_success", np.empty(0)), dtype=bool)
        attempt_omega_target = np.asarray(npz.get("stats_attempt_omega_target", np.empty(0)), dtype=np.float64)
        attempt_lambda_before = np.asarray(npz.get("stats_attempt_lambda_before", np.empty(0)), dtype=np.float64)
        attempt_lambda_after = np.asarray(npz.get("stats_attempt_lambda_after", np.empty(0)), dtype=np.float64)
        attempt_linear_iterations = np.asarray(npz.get("stats_attempt_linear_iterations", np.empty(0)), dtype=np.int64)
        attempt_newton_iterations = np.asarray(npz.get("stats_attempt_newton_iterations", np.empty(0)), dtype=np.int64)
        attempt_wall_time = np.asarray(npz.get("stats_attempt_wall_time", np.empty(0)), dtype=np.float64)
        attempt_relres_end = np.asarray(npz.get("stats_attempt_newton_relres_end", np.empty(0)), dtype=np.float64)

    info = run_info["run_info"]
    timings = run_info["timings"]
    linear = timings["linear"]
    constitutive = timings["constitutive"]
    owned_pattern = run_info.get("owned_tangent_pattern", {})
    owned_stats_max = owned_pattern.get("stats_max", {})
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
        "lambda_last": float(lambda_hist[-1]),
        "omega_last": float(omega_hist[-1]),
        "umax_last": float(umax_hist[-1]),
        "lambda_hist": lambda_hist,
        "omega_hist": omega_hist,
        "umax_hist": umax_hist,
        "step_index": step_index,
        "step_lambda": step_lambda,
        "step_omega": step_omega,
        "step_linear_iterations": step_linear_iterations,
        "step_newton_iterations": step_newton_iterations,
        "step_wall_time": step_wall_time,
        "step_attempt_count": step_attempt_count,
        "attempt_step": attempt_step,
        "attempt_success": attempt_success,
        "attempt_omega_target": attempt_omega_target,
        "attempt_lambda_before": attempt_lambda_before,
        "attempt_lambda_after": attempt_lambda_after,
        "attempt_linear_iterations": attempt_linear_iterations,
        "attempt_newton_iterations": attempt_newton_iterations,
        "attempt_wall_time": attempt_wall_time,
        "attempt_relres_end": attempt_relres_end,
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
        "continuation_wall_time": float(timings["continuation_total_wall_time"]),
        "owned_tangent_pattern_stats_max": dict(owned_stats_max),
    }


def _format_ratio(a: float, b: float) -> str:
    if float(b) == 0.0:
        return "-"
    return f"{float(a) / float(b):.3f}x"


def _format_delta(a: float, b: float) -> str:
    return f"{abs(float(a) - float(b)):.3e}"


def _plot_continuation_curves(plot_dir: Path, *, p2: dict[str, object], p4: dict[str, object]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    axes[0].plot(np.asarray(p2["omega_hist"]), np.asarray(p2["lambda_hist"]), marker="o", linewidth=2, label="P2")
    axes[0].plot(np.asarray(p4["omega_hist"]), np.asarray(p4["lambda_hist"]), marker="s", linewidth=2, label="P4")
    axes[0].set_xlabel(r"$\omega$")
    axes[0].set_ylabel(r"$\lambda$")
    axes[0].set_title("Continuation Curve")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(np.arange(len(np.asarray(p2["umax_hist"]))), np.asarray(p2["umax_hist"]), marker="o", linewidth=2, label="P2")
    axes[1].plot(np.arange(len(np.asarray(p4["umax_hist"]))), np.asarray(p4["umax_hist"]), marker="s", linewidth=2, label="P4")
    axes[1].set_xlabel("Accepted state index")
    axes[1].set_ylabel("Umax")
    axes[1].set_title("Displacement Growth")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(np.arange(len(np.asarray(p2["omega_hist"]))), np.asarray(p2["omega_hist"]), marker="o", linewidth=2, label="P2")
    axes[2].plot(np.arange(len(np.asarray(p4["omega_hist"]))), np.asarray(p4["omega_hist"]), marker="s", linewidth=2, label="P4")
    axes[2].set_xlabel("Accepted state index")
    axes[2].set_ylabel(r"$\omega$")
    axes[2].set_title("Accepted-State Omega")
    axes[2].grid(True, alpha=0.3)

    path = plot_dir / "continuation_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_iterations(plot_dir: Path, *, p2: dict[str, object], p4: dict[str, object]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    p2_step_idx = np.asarray(p2["step_index"], dtype=np.int64)
    p4_step_idx = np.asarray(p4["step_index"], dtype=np.int64)
    if p2_step_idx.size:
        axes[0, 0].plot(p2_step_idx, np.asarray(p2["step_linear_iterations"]), marker="o", linewidth=2, label="P2")
        axes[0, 1].plot(p2_step_idx, np.asarray(p2["step_newton_iterations"]), marker="o", linewidth=2, label="P2")
    if p4_step_idx.size:
        axes[0, 0].plot(p4_step_idx, np.asarray(p4["step_linear_iterations"]), marker="s", linewidth=2, label="P4")
        axes[0, 1].plot(p4_step_idx, np.asarray(p4["step_newton_iterations"]), marker="s", linewidth=2, label="P4")

    axes[0, 0].set_title("Accepted-Step Linear Iterations")
    axes[0, 0].set_xlabel("Accepted continuation step")
    axes[0, 0].set_ylabel("Linear iterations")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best")

    axes[0, 1].set_title("Accepted-Step Newton Iterations")
    axes[0, 1].set_xlabel("Accepted continuation step")
    axes[0, 1].set_ylabel("Newton iterations")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc="best")

    p2_attempt = np.arange(1, len(np.asarray(p2["attempt_linear_iterations"])) + 1)
    p4_attempt = np.arange(1, len(np.asarray(p4["attempt_linear_iterations"])) + 1)
    if p2_attempt.size:
        axes[1, 0].plot(p2_attempt, np.asarray(p2["attempt_linear_iterations"]), marker="o", linewidth=2, label="P2")
        axes[1, 1].plot(p2_attempt, np.asarray(p2["attempt_wall_time"]), marker="o", linewidth=2, label="P2")
    if p4_attempt.size:
        axes[1, 0].plot(p4_attempt, np.asarray(p4["attempt_linear_iterations"]), marker="s", linewidth=2, label="P4")
        axes[1, 1].plot(p4_attempt, np.asarray(p4["attempt_wall_time"]), marker="s", linewidth=2, label="P4")

    axes[1, 0].set_title("Attempt Linear Iterations")
    axes[1, 0].set_xlabel("Attempt index")
    axes[1, 0].set_ylabel("Linear iterations")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="best")

    axes[1, 1].set_title("Attempt Wall Time")
    axes[1, 1].set_xlabel("Attempt index")
    axes[1, 1].set_ylabel("Wall time [s]")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="best")

    path = plot_dir / "iterations.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_timing_breakdown(plot_dir: Path, *, p2: dict[str, object], p4: dict[str, object]) -> Path:
    labels = ["P2", "P4"]
    series = {
        "init solve": np.asarray([p2["init_linear_solve_time"], p4["init_linear_solve_time"]], dtype=np.float64),
        "attempt solves": np.asarray([p2["attempt_linear_solve_time_total"], p4["attempt_linear_solve_time_total"]], dtype=np.float64),
        "attempt prec": np.asarray([p2["attempt_linear_preconditioner_time_total"], p4["attempt_linear_preconditioner_time_total"]], dtype=np.float64),
        "tangent local": np.asarray([p2["build_tangent_local"], p4["build_tangent_local"]], dtype=np.float64),
        "build F": np.asarray([p2["build_F"], p4["build_F"]], dtype=np.float64),
    }

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bottom = np.zeros(2, dtype=np.float64)
    for label, values in series.items():
        ax.bar(labels, values, bottom=bottom, label=label)
        bottom += values
    ax.set_ylabel("Time [s]")
    ax.set_title("Timing Breakdown")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")

    path = plot_dir / "timing_breakdown.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_summary_json(
    out_root: Path,
    *,
    mesh_path: Path,
    mpi_ranks: int,
    step_max: int,
    p2: dict[str, object],
    p4: dict[str, object],
    p4_memory_guard: dict[str, float | int | bool | str] | None,
) -> Path:
    out_root = out_root.resolve()
    mesh_path = mesh_path.resolve()
    path = out_root / "summary.json"

    def _to_serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        if isinstance(value, dict):
            return {key: _to_serializable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_serializable(v) for v in value]
        return value

    payload = {
        "mesh_path": str(mesh_path.relative_to(ROOT)),
        "mpi_ranks": int(mpi_ranks),
        "step_max": int(step_max),
        "P2": _to_serializable(p2),
        "P4": _to_serializable(p4),
        "P4_memory_guard": _to_serializable(p4_memory_guard),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_report(
    report_path: Path,
    *,
    mesh_path: Path,
    mpi_ranks: int,
    step_max: int,
    out_root: Path,
    plot_paths: dict[str, Path],
    p2: dict[str, object],
    p4: dict[str, object],
    p4_memory_guard: dict[str, float | int | bool | str] | None,
) -> None:
    report_path = report_path.resolve()
    out_root = out_root.resolve()
    mesh_path = mesh_path.resolve()

    def rel(path: Path) -> str:
        return os.path.relpath(path.resolve(), start=report_path.parent)

    lines = [
        "# 3D Hetero SSR: P2 vs P4 Final-State Comparison",
        "",
        f"- Mesh: `{mesh_path.relative_to(ROOT)}`",
        f"- MPI ranks: `{int(mpi_ranks)}`",
        f"- `step_max`: `{int(step_max)}`",
        "- Runner: `slope_stability.cli.run_3D_hetero_SSR_capture`",
        f"- Raw artifacts: `{out_root.relative_to(ROOT)}`",
        f"- Tangent kernel: `{p2.get('tangent_kernel', 'unknown')}`",
        f"- Constitutive mode: `{p2.get('constitutive_mode', 'unknown')}`",
        "",
        "## Headline Metrics",
        "",
        "| Metric | P2 | P4 | P4 / P2 |",
        "| --- | ---: | ---: | ---: |",
        f"| Mesh nodes | {p2['mesh_nodes']} | {p4['mesh_nodes']} | {_format_ratio(p4['mesh_nodes'], p2['mesh_nodes'])} |",
        f"| Unknowns | {p2['unknowns']} | {p4['unknowns']} | {_format_ratio(p4['unknowns'], p2['unknowns'])} |",
        f"| Runtime [s] | {p2['runtime_seconds']:.3f} | {p4['runtime_seconds']:.3f} | {_format_ratio(p4['runtime_seconds'], p2['runtime_seconds'])} |",
        f"| Final accepted states | {p2['final_accepted_states']} | {p4['final_accepted_states']} | {_format_ratio(p4['final_accepted_states'], p2['final_accepted_states'])} |",
        f"| Continuation advances after init | {p2['accepted_continuation_advances']} | {p4['accepted_continuation_advances']} | {_format_ratio(p4['accepted_continuation_advances'], p2['accepted_continuation_advances'])} |",
        f"| Init linear iterations | {p2['init_linear_iterations']} | {p4['init_linear_iterations']} | {_format_ratio(p4['init_linear_iterations'], p2['init_linear_iterations'])} |",
        f"| Attempt linear iterations total | {p2['attempt_linear_iterations_total']} | {p4['attempt_linear_iterations_total']} | {_format_ratio(p4['attempt_linear_iterations_total'], p2['attempt_linear_iterations_total'])} |",
        f"| Accepted-step Newton iterations total | {int(np.asarray(p2['step_newton_iterations']).sum())} | {int(np.asarray(p4['step_newton_iterations']).sum())} | {_format_ratio(np.asarray(p4['step_newton_iterations']).sum(), np.asarray(p2['step_newton_iterations']).sum())} |",
        "",
        "## Final State",
        "",
        "| Metric | P2 | P4 | Absolute difference |",
        "| --- | ---: | ---: | ---: |",
        f"| Final lambda | {p2['lambda_last']:.9f} | {p4['lambda_last']:.9f} | {_format_delta(p2['lambda_last'], p4['lambda_last'])} |",
        f"| Final omega | {p2['omega_last']:.9f} | {p4['omega_last']:.9f} | {_format_delta(p2['omega_last'], p4['omega_last'])} |",
        f"| Final Umax | {p2['umax_last']:.9f} | {p4['umax_last']:.9f} | {_format_delta(p2['umax_last'], p4['umax_last'])} |",
        "",
        "## Continuation Reach",
        "",
        "| Order | Init accepted states | Final accepted states | Successful attempts |",
        "| --- | ---: | ---: | ---: |",
        f"| P2 | {p2['init_accepted_states']} | {p2['final_accepted_states']} | {p2['successful_attempt_count']} / {p2['attempt_count']} |",
        f"| P4 | {p4['init_accepted_states']} | {p4['final_accepted_states']} | {p4['successful_attempt_count']} / {p4['attempt_count']} |",
        "",
        "## Timing Breakdown",
        "",
        "| Metric | P2 [s] | P4 [s] | P4 / P2 |",
        "| --- | ---: | ---: | ---: |",
        f"| Init solve | {p2['init_linear_solve_time']:.3f} | {p4['init_linear_solve_time']:.3f} | {_format_ratio(p4['init_linear_solve_time'], p2['init_linear_solve_time'])} |",
        f"| Attempt solves total | {p2['attempt_linear_solve_time_total']:.3f} | {p4['attempt_linear_solve_time_total']:.3f} | {_format_ratio(p4['attempt_linear_solve_time_total'], p2['attempt_linear_solve_time_total'])} |",
        f"| Attempt preconditioner total | {p2['attempt_linear_preconditioner_time_total']:.3f} | {p4['attempt_linear_preconditioner_time_total']:.3f} | {_format_ratio(p4['attempt_linear_preconditioner_time_total'], p2['attempt_linear_preconditioner_time_total'])} |",
        f"| Tangent local | {p2['build_tangent_local']:.3f} | {p4['build_tangent_local']:.3f} | {_format_ratio(p4['build_tangent_local'], p2['build_tangent_local'])} |",
        f"| Build F | {p2['build_F']:.3f} | {p4['build_F']:.3f} | {_format_ratio(p4['build_F'], p2['build_F'])} |",
        f"| Local strain | {p2['local_strain']:.3f} | {p4['local_strain']:.3f} | {_format_ratio(p4['local_strain'], p2['local_strain'])} |",
        f"| Local constitutive | {p2['local_constitutive']:.3f} | {p4['local_constitutive']:.3f} | {_format_ratio(p4['local_constitutive'], p2['local_constitutive'])} |",
        "",
        "## Owned Pattern Bytes",
        "",
        "| Metric | P2 | P4 | P4 / P2 |",
        "| --- | ---: | ---: | ---: |",
        f"| `scatter_bytes` | {float(p2['owned_tangent_pattern_stats_max'].get('scatter_bytes', 0.0)):.0f} | {float(p4['owned_tangent_pattern_stats_max'].get('scatter_bytes', 0.0)):.0f} | {_format_ratio(float(p4['owned_tangent_pattern_stats_max'].get('scatter_bytes', 0.0)), float(p2['owned_tangent_pattern_stats_max'].get('scatter_bytes', 0.0)))} |",
        f"| `row_slot_bytes` | {float(p2['owned_tangent_pattern_stats_max'].get('row_slot_bytes', 0.0)):.0f} | {float(p4['owned_tangent_pattern_stats_max'].get('row_slot_bytes', 0.0)):.0f} | {_format_ratio(float(p4['owned_tangent_pattern_stats_max'].get('row_slot_bytes', 0.0)), float(p2['owned_tangent_pattern_stats_max'].get('row_slot_bytes', 0.0)))} |",
        f"| `overlap_B_bytes` | {float(p2['owned_tangent_pattern_stats_max'].get('overlap_B_bytes', 0.0)):.0f} | {float(p4['owned_tangent_pattern_stats_max'].get('overlap_B_bytes', 0.0)):.0f} | {_format_ratio(float(p4['owned_tangent_pattern_stats_max'].get('overlap_B_bytes', 0.0)), float(p2['owned_tangent_pattern_stats_max'].get('overlap_B_bytes', 0.0)))} |",
        f"| `unique_B_bytes` | {float(p2['owned_tangent_pattern_stats_max'].get('unique_B_bytes', 0.0)):.0f} | {float(p4['owned_tangent_pattern_stats_max'].get('unique_B_bytes', 0.0)):.0f} | {_format_ratio(float(p4['owned_tangent_pattern_stats_max'].get('unique_B_bytes', 0.0)), float(p2['owned_tangent_pattern_stats_max'].get('unique_B_bytes', 0.0)))} |",
        f"| `dphi_bytes` | {float(p2['owned_tangent_pattern_stats_max'].get('dphi_bytes', 0.0)):.0f} | {float(p4['owned_tangent_pattern_stats_max'].get('dphi_bytes', 0.0)):.0f} | {_format_ratio(float(p4['owned_tangent_pattern_stats_max'].get('dphi_bytes', 0.0)), float(p2['owned_tangent_pattern_stats_max'].get('dphi_bytes', 0.0)))} |",
        "",
        "## Memory Guard",
        "",
    ]
    if p4_memory_guard is None:
        lines.extend(
            [
                "- No `P4` memory guard log was provided for this report.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "| Metric | P4 guarded run |",
                "| --- | ---: |",
                f"| Peak RSS [GiB] | {float(p4_memory_guard['peak_rss_gib']):.3f} |",
                f"| Minimum MemAvailable [GiB] | {float(p4_memory_guard['min_available_gib']):.3f} |",
                f"| Samples | {int(p4_memory_guard['sample_count'])} |",
                f"| Guard triggered | {'yes' if bool(p4_memory_guard['guard_triggered']) else 'no'} |",
                f"| Guard log | `{p4_memory_guard['path']}` |",
                "",
            ]
        )

    lines.extend(
        [
        "## Plots",
        "",
        f"![Continuation curves]({rel(plot_paths['continuation'])})",
        "",
        f"![Iteration comparison]({rel(plot_paths['iterations'])})",
        "",
        f"![Timing breakdown]({rel(plot_paths['timing'])})",
        "",
        "## Notes",
        "",
        "- This comparison uses the same tet4 `.msh` source mesh and elevates it in-memory to `tet10`/`tri6` for `P2` and `tet35`/`tri15` for `P4` after loading.",
        "- The current VTU/export path linearizes higher-order simplex cells for visualization. Solver-side assembly still uses the full elevated connectivity.",
        "- On this mesh, both orders terminated naturally after the same continuation reach. The large difference is computational cost, not a different final branch point.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and compare 3D hetero SSR P2 vs P4.")
    parser.add_argument("--mesh-path", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--step-max", type=int, default=100)
    parser.add_argument("--max-deflation-basis-vectors", type=int, default=16)
    parser.add_argument("--store-step-u", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--p4-memory-log", type=Path, default=None)
    parser.add_argument("--reuse-existing", action="store_true", default=False)
    parser.add_argument("--tangent-kernel", type=str, default="rows")
    parser.add_argument("--constitutive-mode", type=str, default="overlap")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tag = f"rank{int(args.mpi_ranks)}_step{int(args.step_max)}"
    p2_dir = out_root / f"p2_{tag}"
    p4_dir = out_root / f"p4_{tag}"

    if args.reuse_existing and (p2_dir / "data" / "run_info.json").exists():
        p2 = _load_metrics(p2_dir)
    else:
        p2 = _run_order(
            "P2",
            mpi_ranks=args.mpi_ranks,
            mesh_path=Path(args.mesh_path),
            step_max=args.step_max,
            out_dir=p2_dir,
            max_deflation_basis_vectors=args.max_deflation_basis_vectors,
            store_step_u=args.store_step_u,
            tangent_kernel=args.tangent_kernel,
            constitutive_mode=args.constitutive_mode,
        )
    p2["tangent_kernel"] = str(args.tangent_kernel)
    p2["constitutive_mode"] = str(args.constitutive_mode)

    if args.reuse_existing and (p4_dir / "data" / "run_info.json").exists():
        p4 = _load_metrics(p4_dir)
    else:
        p4 = _run_order(
            "P4",
            mpi_ranks=args.mpi_ranks,
            mesh_path=Path(args.mesh_path),
            step_max=args.step_max,
            out_dir=p4_dir,
            max_deflation_basis_vectors=args.max_deflation_basis_vectors,
            store_step_u=args.store_step_u,
            tangent_kernel=args.tangent_kernel,
            constitutive_mode=args.constitutive_mode,
        )
    p4["tangent_kernel"] = str(args.tangent_kernel)
    p4["constitutive_mode"] = str(args.constitutive_mode)

    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {
        "continuation": _plot_continuation_curves(plot_dir, p2=p2, p4=p4),
        "iterations": _plot_iterations(plot_dir, p2=p2, p4=p4),
        "timing": _plot_timing_breakdown(plot_dir, p2=p2, p4=p4),
    }
    p4_memory_guard = _load_memory_guard_summary(Path(args.p4_memory_log)) if args.p4_memory_log is not None else None
    _write_summary_json(
        out_root,
        mesh_path=Path(args.mesh_path),
        mpi_ranks=int(args.mpi_ranks),
        step_max=int(args.step_max),
        p2=p2,
        p4=p4,
        p4_memory_guard=p4_memory_guard,
    )
    _write_report(
        Path(args.report_path),
        mesh_path=Path(args.mesh_path),
        mpi_ranks=int(args.mpi_ranks),
        step_max=int(args.step_max),
        out_root=out_root,
        plot_paths=plot_paths,
        p2=p2,
        p4=p4,
        p4_memory_guard=p4_memory_guard,
    )


if __name__ == "__main__":
    main()
