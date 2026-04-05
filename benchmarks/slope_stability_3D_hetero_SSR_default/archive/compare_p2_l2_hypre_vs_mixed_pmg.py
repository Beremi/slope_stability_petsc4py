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

FINE_MESH = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L2.msh"
COARSE_MESH = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "p2_l2_rank8_hypre_vs_mixed_pmg_step10"
DEFAULT_REPORT = SCRIPT_DIR / "report_p2_l2_rank8_hypre_vs_mixed_pmg_step10.md"

PMG_PETSC_OPTIONS = (
    "manualmg_coarse_operator_source=direct_elastic_full_system",
    "mg_levels_ksp_type=chebyshev",
    "mg_levels_ksp_max_it=3",
    "mg_levels_pc_type=jacobi",
    "mg_coarse_ksp_type=cg",
    "mg_coarse_max_it=4",
    "mg_coarse_rtol=0.0",
    "pc_hypre_boomeramg_numfunctions=3",
    "pc_hypre_boomeramg_nodal_coarsen=6",
    "pc_hypre_boomeramg_nodal_coarsen_diag=1",
    "pc_hypre_boomeramg_vec_interp_variant=3",
    "pc_hypre_boomeramg_vec_interp_qmax=4",
    "pc_hypre_boomeramg_vec_interp_smooth=true",
    "pc_hypre_boomeramg_coarsen_type=HMIS",
    "pc_hypre_boomeramg_interp_type=ext+i",
    "pc_hypre_boomeramg_P_max=4",
    "pc_hypre_boomeramg_strong_threshold=0.5",
    "pc_hypre_boomeramg_max_iter=4",
    "pc_hypre_boomeramg_tol=0.0",
    "pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi",
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _report_relpath(report_path: Path, target: Path | str) -> str:
    return os.path.relpath(Path(target).resolve(), report_path.parent.resolve())


def _parse_progress(out_dir: Path) -> dict[str, int]:
    progress_path = out_dir / "data" / "progress.jsonl"
    summary = {
        "init_accepted_states": 0,
        "final_accepted_states": 0,
        "attempt_count": 0,
        "successful_attempt_count": 0,
    }
    if not progress_path.exists():
        return summary
    for raw_line in progress_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        event = json.loads(line)
        kind = str(event.get("event", ""))
        if kind == "init_complete":
            summary["init_accepted_states"] = int(event.get("accepted_steps", summary["init_accepted_states"]))
        elif kind == "attempt_complete":
            summary["attempt_count"] += 1
            if bool(event.get("success", False)):
                summary["successful_attempt_count"] += 1
        elif kind == "finished":
            summary["final_accepted_states"] = int(event.get("accepted_steps", summary["final_accepted_states"]))
    return summary


def _run_case(
    *,
    label: str,
    out_dir: Path,
    mpi_ranks: int,
    step_max: int,
    fine_mesh: Path,
    coarse_mesh: Path,
    linear_tolerance: float,
    linear_max_iter: int,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    cmd = [
        "mpirun",
        "-n",
        str(int(mpi_ranks)),
        sys.executable,
        "-m",
        "slope_stability.cli.run_3D_hetero_SSR_capture",
        "--out_dir",
        str(out_dir),
        "--mesh_path",
        str(fine_mesh),
        "--elem_type",
        "P2",
        "--node_ordering",
        "original",
        "--step_max",
        str(int(step_max)),
        "--solver_type",
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        "--linear_tolerance",
        str(float(linear_tolerance)),
        "--linear_max_iter",
        str(int(linear_max_iter)),
        "--no-store_step_u",
        "--pc_backend",
        "hypre" if label == "hypre_default" else "pmg_shell",
        "--preconditioner_matrix_source",
        "tangent",
    ]
    if label == "pmg_shell_mixed":
        cmd.extend(["--pmg_coarse_mesh_path", str(coarse_mesh)])
        for opt in PMG_PETSC_OPTIONS:
            cmd.extend(["--petsc-opt", opt])
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _load_case(out_dir: Path, *, label: str) -> dict[str, object]:
    run_info = json.loads((out_dir / "data" / "run_info.json").read_text(encoding="utf-8"))
    progress = _parse_progress(out_dir)
    with np.load(out_dir / "data" / "petsc_run.npz", allow_pickle=True) as npz:
        lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
        omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
        umax_hist = np.asarray(npz["Umax_hist"], dtype=np.float64)
        step_index = np.asarray(npz.get("stats_step_index", np.empty(0)), dtype=np.int64)
        step_lambda = np.asarray(npz.get("stats_step_lambda", np.empty(0)), dtype=np.float64)
        step_omega = np.asarray(npz.get("stats_step_omega", np.empty(0)), dtype=np.float64)
        step_wall_time = np.asarray(npz.get("stats_step_wall_time", np.empty(0)), dtype=np.float64)
        step_attempt_count = np.asarray(npz.get("stats_step_attempt_count", np.empty(0)), dtype=np.int64)
        step_newton_iterations = np.asarray(npz.get("stats_step_newton_iterations", np.empty(0)), dtype=np.int64)
        step_newton_iterations_total = np.asarray(
            npz.get("stats_step_newton_iterations_total", np.empty(0)),
            dtype=np.int64,
        )
        step_linear_iterations = np.asarray(npz.get("stats_step_linear_iterations", np.empty(0)), dtype=np.int64)
        step_linear_solve_time = np.asarray(npz.get("stats_step_linear_solve_time", np.empty(0)), dtype=np.float64)
        step_linear_preconditioner_time = np.asarray(
            npz.get("stats_step_linear_preconditioner_time", np.empty(0)),
            dtype=np.float64,
        )
        step_linear_orthogonalization_time = np.asarray(
            npz.get("stats_step_linear_orthogonalization_time", np.empty(0)),
            dtype=np.float64,
        )
        attempt_step = np.asarray(npz.get("stats_attempt_step", np.empty(0)), dtype=np.int64)
        attempt_success = np.asarray(npz.get("stats_attempt_success", np.empty(0)), dtype=bool)
        attempt_linear_iterations = np.asarray(npz.get("stats_attempt_linear_iterations", np.empty(0)), dtype=np.int64)
        attempt_newton_iterations = np.asarray(npz.get("stats_attempt_newton_iterations", np.empty(0)), dtype=np.int64)
        attempt_wall_time = np.asarray(npz.get("stats_attempt_wall_time", np.empty(0)), dtype=np.float64)
        attempt_lambda_before = np.asarray(npz.get("stats_attempt_lambda_before", np.empty(0)), dtype=np.float64)
        attempt_lambda_after = np.asarray(npz.get("stats_attempt_lambda_after", np.empty(0)), dtype=np.float64)
        attempt_omega_target = np.asarray(npz.get("stats_attempt_omega_target", np.empty(0)), dtype=np.float64)

    info = run_info["run_info"]
    params = run_info["params"]
    timings = run_info["timings"]
    linear = timings["linear"]
    constitutive = timings["constitutive"]

    init_accepted = int(progress["init_accepted_states"] or 2)
    final_accepted = int(progress["final_accepted_states"] or info["step_count"])
    continuation_index = np.asarray(step_index - init_accepted, dtype=np.int64)
    step_linear_per_newton = np.divide(
        step_linear_iterations.astype(np.float64),
        np.maximum(step_newton_iterations_total.astype(np.float64), 1.0),
    )

    return {
        "label": label,
        "out_dir": str(out_dir),
        "run_info_path": str(out_dir / "data" / "run_info.json"),
        "npz_path": str(out_dir / "data" / "petsc_run.npz"),
        "runtime_seconds": float(info["runtime_seconds"]),
        "continuation_total_wall_time": float(timings["continuation_total_wall_time"]),
        "mpi_size": int(info["mpi_size"]),
        "mesh_nodes": int(info["mesh_nodes"]),
        "mesh_elements": int(info["mesh_elements"]),
        "unknowns": int(info["unknowns"]),
        "step_count": int(info["step_count"]),
        "init_accepted_states": init_accepted,
        "final_accepted_states": final_accepted,
        "accepted_continuation_advances": int(max(0, final_accepted - init_accepted)),
        "attempt_count": int(progress["attempt_count"]),
        "successful_attempt_count": int(progress["successful_attempt_count"]),
        "lambda_hist": lambda_hist,
        "omega_hist": omega_hist,
        "umax_hist": umax_hist,
        "lambda_last": float(lambda_hist[-1]),
        "omega_last": float(omega_hist[-1]),
        "umax_last": float(umax_hist[-1]),
        "step_index": step_index,
        "continuation_index": continuation_index,
        "step_lambda": step_lambda,
        "step_omega": step_omega,
        "step_wall_time": step_wall_time,
        "step_attempt_count": step_attempt_count,
        "step_newton_iterations": step_newton_iterations,
        "step_newton_iterations_total": step_newton_iterations_total,
        "step_linear_iterations": step_linear_iterations,
        "step_linear_solve_time": step_linear_solve_time,
        "step_linear_preconditioner_time": step_linear_preconditioner_time,
        "step_linear_orthogonalization_time": step_linear_orthogonalization_time,
        "step_linear_per_newton": step_linear_per_newton,
        "attempt_step": attempt_step,
        "attempt_success": attempt_success,
        "attempt_linear_iterations": attempt_linear_iterations,
        "attempt_newton_iterations": attempt_newton_iterations,
        "attempt_wall_time": attempt_wall_time,
        "attempt_lambda_before": attempt_lambda_before,
        "attempt_lambda_after": attempt_lambda_after,
        "attempt_omega_target": attempt_omega_target,
        "init_linear_iterations": int(linear["init_linear_iterations"]),
        "init_linear_solve_time": float(linear["init_linear_solve_time"]),
        "init_linear_preconditioner_time": float(linear["init_linear_preconditioner_time"]),
        "init_linear_orthogonalization_time": float(linear["init_linear_orthogonalization_time"]),
        "attempt_linear_iterations_total": int(linear["attempt_linear_iterations_total"]),
        "attempt_linear_solve_time_total": float(linear["attempt_linear_solve_time_total"]),
        "attempt_linear_preconditioner_time_total": float(linear["attempt_linear_preconditioner_time_total"]),
        "attempt_linear_orthogonalization_time_total": float(linear["attempt_linear_orthogonalization_time_total"]),
        "preconditioner_setup_time_total": float(linear.get("preconditioner_setup_time_total", 0.0)),
        "preconditioner_apply_time_total": float(linear.get("preconditioner_apply_time_total", 0.0)),
        "preconditioner_rebuild_count": int(linear.get("preconditioner_rebuild_count", 0)),
        "build_tangent_local": float(constitutive["build_tangent_local"]),
        "build_F": float(constitutive["build_F"]),
        "local_strain": float(constitutive["local_strain"]),
        "local_constitutive": float(constitutive["local_constitutive"]),
        "stress": float(constitutive["stress"]),
        "stress_tangent": float(constitutive["stress_tangent"]),
        "params": params,
    }


def _plot_continuation(out_path: Path, *, hypre: dict[str, object], pmg: dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for case, marker in ((hypre, "o"), (pmg, "s")):
        axes[0].plot(case["omega_hist"], case["lambda_hist"], marker=marker, linewidth=2, label=case["label"])
        axes[1].plot(np.arange(len(case["lambda_hist"])), case["lambda_hist"], marker=marker, linewidth=2, label=case["label"])
        axes[2].plot(np.arange(len(case["omega_hist"])), case["omega_hist"], marker=marker, linewidth=2, label=case["label"])
    axes[0].set_xlabel(r"$\omega$")
    axes[0].set_ylabel(r"$\lambda$")
    axes[0].set_title(r"$\lambda(\omega)$")
    axes[1].set_xlabel("Accepted state index")
    axes[1].set_ylabel(r"$\lambda$")
    axes[1].set_title("Lambda evolution")
    axes[2].set_xlabel("Accepted state index")
    axes[2].set_ylabel(r"$\omega$")
    axes[2].set_title("Omega evolution")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_step_metrics(out_path: Path, *, hypre: dict[str, object], pmg: dict[str, object]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    series = (
        ("step_wall_time", "Step wall time [s]"),
        ("step_newton_iterations_total", "Newton iterations / accepted step"),
        ("step_linear_iterations", "Linear iterations / accepted step"),
        ("step_linear_per_newton", "Linear iterations / Newton"),
    )
    for ax, (key, ylabel) in zip(axes.ravel(), series):
        for case, marker in ((hypre, "o"), (pmg, "s")):
            x = np.asarray(case["continuation_index"], dtype=np.int64)
            y = np.asarray(case[key], dtype=np.float64)
            ax.plot(x, y, marker=marker, linewidth=2, label=case["label"])
        ax.set_xlabel("Accepted continuation step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_linear_timing(out_path: Path, *, hypre: dict[str, object], pmg: dict[str, object]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    series = (
        ("step_linear_solve_time", "Linear solve wall time [s]"),
        ("step_linear_preconditioner_time", "Linear preconditioner wall time [s]"),
        ("step_linear_orthogonalization_time", "Orthogonalization wall time [s]"),
        ("step_attempt_count", "Attempts per accepted step"),
    )
    for ax, (key, ylabel) in zip(axes.ravel(), series):
        for case, marker in ((hypre, "o"), (pmg, "s")):
            x = np.asarray(case["continuation_index"], dtype=np.int64)
            y = np.asarray(case[key], dtype=np.float64)
            ax.plot(x, y, marker=marker, linewidth=2, label=case["label"])
        ax.set_xlabel("Accepted continuation step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_total_timing(out_path: Path, *, hypre: dict[str, object], pmg: dict[str, object]) -> None:
    labels = ["Hypre default", "Mixed PMG-shell"]
    components = {
        "init solve": np.asarray([hypre["init_linear_solve_time"], pmg["init_linear_solve_time"]], dtype=np.float64),
        "init prec": np.asarray([hypre["init_linear_preconditioner_time"], pmg["init_linear_preconditioner_time"]], dtype=np.float64),
        "attempt solves": np.asarray([hypre["attempt_linear_solve_time_total"], pmg["attempt_linear_solve_time_total"]], dtype=np.float64),
        "attempt prec": np.asarray([hypre["attempt_linear_preconditioner_time_total"], pmg["attempt_linear_preconditioner_time_total"]], dtype=np.float64),
        "build tangent": np.asarray([hypre["build_tangent_local"], pmg["build_tangent_local"]], dtype=np.float64),
        "build F": np.asarray([hypre["build_F"], pmg["build_F"]], dtype=np.float64),
    }
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    bottom = np.zeros(2, dtype=np.float64)
    for label, values in components.items():
        ax.bar(labels, values, bottom=bottom, label=label)
        bottom += values
    ax.set_ylabel("Seconds")
    ax.set_title("Selected total timing components")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _continuation_table_rows(hypre: dict[str, object], pmg: dict[str, object], *, limit: int) -> list[str]:
    rows = [
        "| Cont. step | Hypre lambda | PMG lambda | Hypre omega | PMG omega | Hypre Umax | PMG Umax |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx in range(1, int(limit) + 1):
        h_mask = np.asarray(hypre["continuation_index"], dtype=np.int64) == idx
        p_mask = np.asarray(pmg["continuation_index"], dtype=np.int64) == idx
        if not bool(np.any(h_mask)) and not bool(np.any(p_mask)):
            continue
        h_lambda = np.asarray(hypre["step_lambda"], dtype=np.float64)[h_mask]
        p_lambda = np.asarray(pmg["step_lambda"], dtype=np.float64)[p_mask]
        h_omega = np.asarray(hypre["step_omega"], dtype=np.float64)[h_mask]
        p_omega = np.asarray(pmg["step_omega"], dtype=np.float64)[p_mask]
        h_umax = np.asarray(hypre["umax_hist"], dtype=np.float64)[idx + int(hypre["init_accepted_states"]) - 1] if np.any(h_mask) else np.array([])
        p_umax = np.asarray(pmg["umax_hist"], dtype=np.float64)[idx + int(pmg["init_accepted_states"]) - 1] if np.any(p_mask) else np.array([])
        rows.append(
            f"| {idx} | "
            f"{(f'{float(h_lambda[0]):.9f}' if h_lambda.size else '-')} | "
            f"{(f'{float(p_lambda[0]):.9f}' if p_lambda.size else '-')} | "
            f"{(f'{float(h_omega[0]):.3f}' if h_omega.size else '-')} | "
            f"{(f'{float(p_omega[0]):.3f}' if p_omega.size else '-')} | "
            f"{(f'{float(h_umax):.6f}' if h_umax.size else '-')} | "
            f"{(f'{float(p_umax):.6f}' if p_umax.size else '-')} |"
        )
    return rows


def _iteration_table_rows(hypre: dict[str, object], pmg: dict[str, object], *, limit: int) -> list[str]:
    rows = [
        "| Cont. step | Hypre wall [s] | PMG wall [s] | Hypre attempts | PMG attempts | Hypre Newton | PMG Newton | Hypre linear | PMG linear | Hypre lin/Newton | PMG lin/Newton |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx in range(1, int(limit) + 1):
        h_mask = np.asarray(hypre["continuation_index"], dtype=np.int64) == idx
        p_mask = np.asarray(pmg["continuation_index"], dtype=np.int64) == idx
        if not bool(np.any(h_mask)) and not bool(np.any(p_mask)):
            continue
        def _pick(case: dict[str, object], key: str, mask: np.ndarray) -> str:
            arr = np.asarray(case[key])
            if not np.any(mask):
                return "-"
            val = arr[mask][0]
            if np.issubdtype(arr.dtype, np.integer):
                return str(int(val))
            return f"{float(val):.3f}"
        rows.append(
            f"| {idx} | "
            f"{_pick(hypre, 'step_wall_time', h_mask)} | {_pick(pmg, 'step_wall_time', p_mask)} | "
            f"{_pick(hypre, 'step_attempt_count', h_mask)} | {_pick(pmg, 'step_attempt_count', p_mask)} | "
            f"{_pick(hypre, 'step_newton_iterations_total', h_mask)} | {_pick(pmg, 'step_newton_iterations_total', p_mask)} | "
            f"{_pick(hypre, 'step_linear_iterations', h_mask)} | {_pick(pmg, 'step_linear_iterations', p_mask)} | "
            f"{_pick(hypre, 'step_linear_per_newton', h_mask)} | {_pick(pmg, 'step_linear_per_newton', p_mask)} |"
        )
    return rows


def _timing_rows(hypre: dict[str, object], pmg: dict[str, object]) -> list[str]:
    rows = [
        "| Metric | Hypre default | Mixed PMG-shell | PMG / Hypre |",
        "| --- | ---: | ---: | ---: |",
    ]

    def add(label: str, h: float | int, p: float | int, digits: int = 3) -> None:
        h_val = float(h)
        p_val = float(p)
        ratio = "-" if h_val == 0.0 else f"{p_val / h_val:.3f}x"
        fmt = f"{{:.{digits}f}}" if digits > 0 else "{:.0f}"
        rows.append(f"| {label} | {fmt.format(h_val)} | {fmt.format(p_val)} | {ratio} |")

    add("Runtime [s]", hypre["runtime_seconds"], pmg["runtime_seconds"])
    add("Continuation wall time [s]", hypre["continuation_total_wall_time"], pmg["continuation_total_wall_time"])
    add("Init linear iterations", hypre["init_linear_iterations"], pmg["init_linear_iterations"], digits=0)
    add("Continuation linear iterations total", hypre["attempt_linear_iterations_total"], pmg["attempt_linear_iterations_total"], digits=0)
    add(
        "Accepted-step Newton iterations total",
        int(np.asarray(hypre["step_newton_iterations_total"], dtype=np.int64).sum()),
        int(np.asarray(pmg["step_newton_iterations_total"], dtype=np.int64).sum()),
        digits=0,
    )
    add(
        "Total linear / Newton",
        float(hypre["attempt_linear_iterations_total"]) / max(float(np.asarray(hypre["step_newton_iterations_total"]).sum()), 1.0),
        float(pmg["attempt_linear_iterations_total"]) / max(float(np.asarray(pmg["step_newton_iterations_total"]).sum()), 1.0),
    )
    add("Init solve [s]", hypre["init_linear_solve_time"], pmg["init_linear_solve_time"])
    add("Init preconditioner collector [s]", hypre["init_linear_preconditioner_time"], pmg["init_linear_preconditioner_time"])
    add("Continuation solve [s]", hypre["attempt_linear_solve_time_total"], pmg["attempt_linear_solve_time_total"])
    add(
        "Continuation preconditioner collector [s]",
        hypre["attempt_linear_preconditioner_time_total"],
        pmg["attempt_linear_preconditioner_time_total"],
    )
    add("Continuation orthogonalization [s]", hypre["attempt_linear_orthogonalization_time_total"], pmg["attempt_linear_orthogonalization_time_total"])
    add("Preconditioner setup diagnostic [s]", hypre["preconditioner_setup_time_total"], pmg["preconditioner_setup_time_total"])
    add("Preconditioner apply diagnostic [s]", hypre["preconditioner_apply_time_total"], pmg["preconditioner_apply_time_total"])
    add("Preconditioner rebuild count", hypre["preconditioner_rebuild_count"], pmg["preconditioner_rebuild_count"], digits=0)
    add("build_tangent_local [s]", hypre["build_tangent_local"], pmg["build_tangent_local"])
    add("build_F [s]", hypre["build_F"], pmg["build_F"])
    add("local_strain [s]", hypre["local_strain"], pmg["local_strain"])
    add("local_constitutive [s]", hypre["local_constitutive"], pmg["local_constitutive"])
    add("stress [s]", hypre["stress"], pmg["stress"])
    add("stress_tangent [s]", hypre["stress_tangent"], pmg["stress_tangent"])
    return rows


def _write_summary(path: Path, *, hypre: dict[str, object], pmg: dict[str, object], mpi_ranks: int, continuation_steps: int, step_max: int) -> None:
    def serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        if isinstance(value, dict):
            return {k: serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(v) for v in value]
        return value

    payload = {
        "mpi_ranks": int(mpi_ranks),
        "continuation_steps_requested": int(continuation_steps),
        "step_max": int(step_max),
        "fine_mesh": str(FINE_MESH),
        "coarse_mesh": str(COARSE_MESH),
        "hypre_default": serializable(hypre),
        "pmg_shell_mixed": serializable(pmg),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_report(
    report_path: Path,
    *,
    out_root: Path,
    plots: dict[str, Path],
    hypre: dict[str, object],
    pmg: dict[str, object],
    mpi_ranks: int,
    continuation_steps: int,
    step_max: int,
    hypre_cmd: str,
    pmg_cmd: str,
) -> None:
    plot_cont = _report_relpath(report_path, plots["continuation"])
    plot_step = _report_relpath(report_path, plots["step_metrics"])
    plot_lin = _report_relpath(report_path, plots["linear_timing"])
    plot_total = _report_relpath(report_path, plots["total_timing"])
    hypre_run = _report_relpath(report_path, hypre["run_info_path"])
    pmg_run = _report_relpath(report_path, pmg["run_info_path"])
    hypre_npz = _report_relpath(report_path, hypre["npz_path"])
    pmg_npz = _report_relpath(report_path, pmg["npz_path"])

    lines = [
        "# P2(L2) Rank-8: Hypre Default vs Mixed PMG-Shell",
        "",
        "## Configuration",
        "",
        f"- Fine mesh: `{FINE_MESH.relative_to(ROOT)}`",
        f"- Mixed PMG coarse mesh: `{COARSE_MESH.relative_to(ROOT)}`",
        f"- MPI ranks: `{int(mpi_ranks)}`",
        f"- Outer solver for both runs: `PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE`",
        f"- Requested continuation advances after init: `{int(continuation_steps)}`",
        f"- Actual runner `step_max`: `{int(step_max)}`",
        "- Note: `step_max` counts accepted states including the 2-state initialization, so `step_max = continuation_steps + 2`.",
        "",
        "## Commands",
        "",
        "```bash",
        hypre_cmd,
        pmg_cmd,
        "```",
        "",
        "## Headline Metrics",
        "",
        * _timing_rows(hypre, pmg),
        "",
        "## Accepted Continuation State Evolution",
        "",
        * _continuation_table_rows(hypre, pmg, limit=int(continuation_steps)),
        "",
        "## Accepted Continuation Iterations And Step Times",
        "",
        * _iteration_table_rows(hypre, pmg, limit=int(continuation_steps)),
        "",
        "## Plots",
        "",
        f"[continuation.png]({plot_cont})",
        "",
        f"![Continuation evolution]({plot_cont})",
        "",
        f"[step_metrics.png]({plot_step})",
        "",
        f"![Per-step metrics]({plot_step})",
        "",
        f"[linear_timing.png]({plot_lin})",
        "",
        f"![Per-step linear timing]({plot_lin})",
        "",
        f"[total_timing.png]({plot_total})",
        "",
        f"![Total timing breakdown]({plot_total})",
        "",
        "## Raw Artifacts",
        "",
        f"- Hypre run info: [run_info.json]({hypre_run})",
        f"- Hypre history: [petsc_run.npz]({hypre_npz})",
        f"- PMG run info: [run_info.json]({pmg_run})",
        f"- PMG history: [petsc_run.npz]({pmg_npz})",
        "",
        "## Notes",
        "",
        "- The Hypre control uses the repo's default direct elasticity-Hypre path for `P2(L2)` with the same outer `DFGMRES` solver.",
        "- The PMG candidate is the mixed hierarchy `P1(L1) -> P1(L2) -> P2(L2)` shell V-cycle with `chebyshev + jacobi` smoothers and `cg + hypre(boomeramg)` on the direct elastic coarse operator.",
        "- The per-step iteration tables use accepted continuation steps only. Newton counts are the total across all attempts that led to that accepted step.",
        "- `* collector [s]` rows come from the solver iteration collector; `* diagnostic [s]` rows come from the preconditioner backend diagnostics.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rank-8 P2(L2) Hypre default against mixed PMG-shell.")
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--continuation-steps", type=int, default=10)
    parser.add_argument("--linear-tolerance", type=float, default=1e-1)
    parser.add_argument("--linear-max-iter", type=int, default=100)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reuse-existing", action="store_true", default=False)
    args = parser.parse_args()

    out_root = _ensure_dir(Path(args.out_root).resolve())
    step_max = int(args.continuation_steps) + 2
    hypre_dir = out_root / f"hypre_default_rank{int(args.mpi_ranks)}_step{int(step_max)}"
    pmg_dir = out_root / f"pmg_shell_mixed_rank{int(args.mpi_ranks)}_step{int(step_max)}"
    pmg_opt_text = " ".join(f"--petsc-opt {opt}" for opt in PMG_PETSC_OPTIONS)

    hypre_cmd = (
        f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src "
        f"mpirun -n {int(args.mpi_ranks)} {sys.executable} -m slope_stability.cli.run_3D_hetero_SSR_capture "
        f"--out_dir {hypre_dir.relative_to(ROOT)} --mesh_path {FINE_MESH.relative_to(ROOT)} --elem_type P2 "
        f"--node_ordering original --step_max {step_max} --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE "
        f"--linear_tolerance {float(args.linear_tolerance)} --linear_max_iter {int(args.linear_max_iter)} "
        f"--no-store_step_u --pc_backend hypre --preconditioner_matrix_source tangent"
    )
    pmg_cmd = (
        f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src "
        f"mpirun -n {int(args.mpi_ranks)} {sys.executable} -m slope_stability.cli.run_3D_hetero_SSR_capture "
        f"--out_dir {pmg_dir.relative_to(ROOT)} --mesh_path {FINE_MESH.relative_to(ROOT)} --elem_type P2 "
        f"--node_ordering original --step_max {step_max} --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE "
        f"--linear_tolerance {float(args.linear_tolerance)} --linear_max_iter {int(args.linear_max_iter)} "
        f"--no-store_step_u --pc_backend pmg_shell --pmg_coarse_mesh_path {COARSE_MESH.relative_to(ROOT)} "
        f"--preconditioner_matrix_source tangent {pmg_opt_text}"
    )

    if not (args.reuse_existing and (hypre_dir / "data" / "run_info.json").exists()):
        _run_case(
            label="hypre_default",
            out_dir=hypre_dir,
            mpi_ranks=int(args.mpi_ranks),
            step_max=step_max,
            fine_mesh=FINE_MESH,
            coarse_mesh=COARSE_MESH,
            linear_tolerance=float(args.linear_tolerance),
            linear_max_iter=int(args.linear_max_iter),
        )
    if not (args.reuse_existing and (pmg_dir / "data" / "run_info.json").exists()):
        _run_case(
            label="pmg_shell_mixed",
            out_dir=pmg_dir,
            mpi_ranks=int(args.mpi_ranks),
            step_max=step_max,
            fine_mesh=FINE_MESH,
            coarse_mesh=COARSE_MESH,
            linear_tolerance=float(args.linear_tolerance),
            linear_max_iter=int(args.linear_max_iter),
        )

    hypre = _load_case(hypre_dir, label="Hypre default")
    pmg = _load_case(pmg_dir, label="Mixed PMG-shell")

    plots_dir = _ensure_dir(out_root / "plots")
    plots = {
        "continuation": plots_dir / "continuation.png",
        "step_metrics": plots_dir / "step_metrics.png",
        "linear_timing": plots_dir / "linear_timing.png",
        "total_timing": plots_dir / "total_timing.png",
    }
    _plot_continuation(plots["continuation"], hypre=hypre, pmg=pmg)
    _plot_step_metrics(plots["step_metrics"], hypre=hypre, pmg=pmg)
    _plot_linear_timing(plots["linear_timing"], hypre=hypre, pmg=pmg)
    _plot_total_timing(plots["total_timing"], hypre=hypre, pmg=pmg)

    _write_summary(
        out_root / "summary.json",
        hypre=hypre,
        pmg=pmg,
        mpi_ranks=int(args.mpi_ranks),
        continuation_steps=int(args.continuation_steps),
        step_max=step_max,
    )
    _write_report(
        Path(args.report_path).resolve(),
        out_root=out_root,
        plots=plots,
        hypre=hypre,
        pmg=pmg,
        mpi_ranks=int(args.mpi_ranks),
        continuation_steps=int(args.continuation_steps),
        step_max=step_max,
        hypre_cmd=hypre_cmd,
        pmg_cmd=pmg_cmd,
    )


if __name__ == "__main__":
    main()
