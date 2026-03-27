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
MESH_DIR = ROOT / "meshes" / "3d_hetero_ssr"

DEFAULT_OUT_ROOT = ROOT / "artifacts" / "pmg_rank8_p2_levels_p4_omega7e6"
DEFAULT_REPORT = SCRIPT_DIR / "report_pmg_rank8_p2_levels_p4_omega7e6.md"

P2_PETSC_OPTIONS = (
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

P4_PETSC_OPTIONS = (
    "pc_hypre_boomeramg_max_iter=4",
    "pc_hypre_boomeramg_tol=0.0",
)

CASE_SPECS: tuple[dict[str, object], ...] = (
    {
        "key": "p2_l1",
        "label": "P2(L1)",
        "hierarchy": "P2(L1) -> P1(L1)",
        "mesh_path": MESH_DIR / "SSR_hetero_ada_L1.msh",
        "elem_type": "P2",
        "node_ordering": "original",
        "pc_backend": "pmg_shell",
        "coarse_mesh_path": None,
        "petsc_opts": P2_PETSC_OPTIONS,
    },
    {
        "key": "p2_l2",
        "label": "P2(L2)",
        "hierarchy": "P2(L2) -> P1(L2) -> P1(L1)",
        "mesh_path": MESH_DIR / "SSR_hetero_ada_L2.msh",
        "elem_type": "P2",
        "node_ordering": "original",
        "pc_backend": "pmg_shell",
        "coarse_mesh_path": MESH_DIR / "SSR_hetero_ada_L1.msh",
        "petsc_opts": P2_PETSC_OPTIONS,
    },
    {
        "key": "p2_l3",
        "label": "P2(L3)",
        "hierarchy": "P2(L3) -> P1(L3) -> P1(L1)",
        "mesh_path": MESH_DIR / "SSR_hetero_ada_L3.msh",
        "elem_type": "P2",
        "node_ordering": "original",
        "pc_backend": "pmg_shell",
        "coarse_mesh_path": MESH_DIR / "SSR_hetero_ada_L1.msh",
        "petsc_opts": P2_PETSC_OPTIONS,
    },
    {
        "key": "p2_l4",
        "label": "P2(L4)",
        "hierarchy": "P2(L4) -> P1(L4) -> P1(L1)",
        "mesh_path": MESH_DIR / "SSR_hetero_ada_L4.msh",
        "elem_type": "P2",
        "node_ordering": "original",
        "pc_backend": "pmg_shell",
        "coarse_mesh_path": MESH_DIR / "SSR_hetero_ada_L1.msh",
        "petsc_opts": P2_PETSC_OPTIONS,
    },
    {
        "key": "p2_l5",
        "label": "P2(L5)",
        "hierarchy": "P2(L5) -> P1(L5) -> P1(L1)",
        "mesh_path": MESH_DIR / "SSR_hetero_ada_L5.msh",
        "elem_type": "P2",
        "node_ordering": "original",
        "pc_backend": "pmg_shell",
        "coarse_mesh_path": MESH_DIR / "SSR_hetero_ada_L1.msh",
        "petsc_opts": P2_PETSC_OPTIONS,
    },
    {
        "key": "p4_l1",
        "label": "P4(L1) baseline",
        "hierarchy": "P4(L1) -> P2(L1) -> P1(L1)",
        "mesh_path": MESH_DIR / "SSR_hetero_ada_L1.msh",
        "elem_type": "P4",
        "node_ordering": "block_metis",
        "pc_backend": "pmg_shell",
        "coarse_mesh_path": None,
        "petsc_opts": P4_PETSC_OPTIONS,
    },
    {
        "key": "p4_l1_newton_caps",
        "label": "P4(L1) + Newton omega caps",
        "hierarchy": "P4(L1) -> P2(L1) -> P1(L1)",
        "mesh_path": MESH_DIR / "SSR_hetero_ada_L1.msh",
        "elem_type": "P4",
        "node_ordering": "block_metis",
        "pc_backend": "pmg_shell",
        "coarse_mesh_path": None,
        "petsc_opts": P4_PETSC_OPTIONS,
        "cli_args": (
            "--omega_no_increase_newton_threshold",
            "10",
            "--omega_half_newton_threshold",
            "20",
        ),
    },
    {
        "key": "p4_l1_newton_caps_uncapped",
        "out_dir_key": "p4_l1_newton_caps",
        "label": "P4(L1) + Newton omega caps, uncapped",
        "hierarchy": "P4(L1) -> P2(L1) -> P1(L1)",
        "mesh_path": MESH_DIR / "SSR_hetero_ada_L1.msh",
        "elem_type": "P4",
        "node_ordering": "block_metis",
        "pc_backend": "pmg_shell",
        "coarse_mesh_path": None,
        "petsc_opts": P4_PETSC_OPTIONS,
        "step_max_override": 100,
        "cli_args": (
            "--omega_no_increase_newton_threshold",
            "10",
            "--omega_half_newton_threshold",
            "20",
        ),
    },
    {
        "key": "p4_l1_smart_controller_v2",
        "label": "P4(L1) + smart omega controller",
        "hierarchy": "P4(L1) -> P2(L1) -> P1(L1)",
        "mesh_path": MESH_DIR / "SSR_hetero_ada_L1.msh",
        "elem_type": "P4",
        "node_ordering": "block_metis",
        "pc_backend": "pmg_shell",
        "coarse_mesh_path": None,
        "petsc_opts": P4_PETSC_OPTIONS,
        "step_max_override": 100,
        "cli_args": (
            "--omega_no_increase_newton_threshold",
            "10",
            "--omega_half_newton_threshold",
            "20",
            "--omega_target_newton_iterations",
            "12",
            "--omega_adapt_min_scale",
            "0.7",
            "--omega_adapt_max_scale",
            "1.25",
            "--omega_hard_newton_threshold",
            "18",
            "--omega_hard_linear_threshold",
            "250",
            "--omega_efficiency_drop_ratio",
            "0.5",
            "--omega_efficiency_window",
            "3",
            "--omega_hard_shrink_scale",
            "0.85",
        ),
    },
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _report_relpath(report_path: Path, target: Path | str) -> str:
    return os.path.relpath(Path(target).resolve(), report_path.parent.resolve())


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _default_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


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


def _run(cmd: list[str], *, env: dict[str, str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _case_command(
    *,
    case_spec: dict[str, object],
    out_dir: Path,
    mpi_ranks: int,
    step_max: int,
    omega_max_stop: float,
) -> list[str]:
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
        str(case_spec["mesh_path"]),
        "--elem_type",
        str(case_spec["elem_type"]),
        "--node_ordering",
        str(case_spec["node_ordering"]),
        "--step_max",
        str(int(step_max)),
        "--omega_max_stop",
        str(float(omega_max_stop)),
        "--solver_type",
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        "--pc_backend",
        str(case_spec["pc_backend"]),
        "--preconditioner_matrix_source",
        "tangent",
        "--no-store_step_u",
    ]
    coarse_mesh_path = case_spec.get("coarse_mesh_path")
    if coarse_mesh_path is not None:
        cmd.extend(["--pmg_coarse_mesh_path", str(coarse_mesh_path)])
    for arg in tuple(case_spec.get("cli_args", ())):
        cmd.append(str(arg))
    for opt in tuple(case_spec.get("petsc_opts", ())):
        cmd.extend(["--petsc-opt", str(opt)])
    return cmd


def _case_step_max(case_spec: dict[str, object], default_step_max: int) -> int:
    return int(case_spec.get("step_max_override", default_step_max))


def _case_out_dir(out_root: Path, case_spec: dict[str, object], mpi_ranks: int, default_step_max: int) -> Path:
    case_step_max = _case_step_max(case_spec, default_step_max)
    out_dir_key = str(case_spec.get("out_dir_key", case_spec["key"]))
    return out_root / f"{out_dir_key}_rank{int(mpi_ranks)}_step{int(case_step_max)}"


def _run_case(
    *,
    case_spec: dict[str, object],
    out_dir: Path,
    mpi_ranks: int,
    step_max: int,
    omega_max_stop: float,
) -> list[str]:
    cmd = _case_command(
        case_spec=case_spec,
        out_dir=out_dir,
        mpi_ranks=mpi_ranks,
        step_max=step_max,
        omega_max_stop=omega_max_stop,
    )
    _run(cmd, env=_default_env(), cwd=ROOT)
    return cmd


def _load_case(out_dir: Path, *, case_spec: dict[str, object]) -> dict[str, object]:
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
    continuation_newton_iterations_total = int(np.sum(step_newton_iterations_total, dtype=np.int64))
    continuation_linear_iterations_total = int(np.sum(step_linear_iterations, dtype=np.int64))
    attempt_newton_iterations_total = int(np.sum(attempt_newton_iterations, dtype=np.int64))
    attempt_linear_iterations_total_npz = int(np.sum(attempt_linear_iterations, dtype=np.int64))
    attempt_linear_per_newton = float(
        attempt_linear_iterations_total_npz / max(float(attempt_newton_iterations_total), 1.0)
    )

    return {
        "key": str(case_spec["key"]),
        "label": str(case_spec["label"]),
        "hierarchy": str(case_spec["hierarchy"]),
        "elem_type": str(case_spec["elem_type"]),
        "node_ordering": str(case_spec["node_ordering"]),
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
        "continuation_newton_iterations_total": continuation_newton_iterations_total,
        "continuation_linear_iterations_total_from_steps": continuation_linear_iterations_total,
        "attempt_newton_iterations_total": attempt_newton_iterations_total,
        "attempt_linear_iterations_total_from_npz": attempt_linear_iterations_total_npz,
        "attempt_linear_per_newton": attempt_linear_per_newton,
        "build_tangent_local": float(constitutive["build_tangent_local"]),
        "build_F": float(constitutive["build_F"]),
        "local_strain": float(constitutive["local_strain"]),
        "local_constitutive": float(constitutive["local_constitutive"]),
        "stress": float(constitutive["stress"]),
        "stress_tangent": float(constitutive["stress_tangent"]),
        "manualmg_levels": int(linear.get("manualmg_levels", 0)),
        "manualmg_level_orders": list(linear.get("manualmg_level_orders", [])),
        "manualmg_level_global_sizes": list(linear.get("manualmg_level_global_sizes", [])),
        "manualmg_coarse_operator_source": str(linear.get("manualmg_coarse_operator_source", "")),
        "manualmg_coarse_ksp_type": str(linear.get("manualmg_coarse_ksp_type", "")),
        "manualmg_coarse_pc_type": str(linear.get("manualmg_coarse_pc_type", "")),
        "manualmg_coarse_hypre_type": str(linear.get("manualmg_coarse_hypre_type", "")),
        "manualmg_fine_ksp_type": str(linear.get("manualmg_fine_ksp_type", "")),
        "manualmg_fine_pc_type": str(linear.get("manualmg_fine_pc_type", "")),
        "manualmg_mid_ksp_type": str(linear.get("manualmg_mid_ksp_type", "")),
        "manualmg_mid_pc_type": str(linear.get("manualmg_mid_pc_type", "")),
        "params": params,
    }


def _plot_lambda_omega(cases: list[dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(cases), 1)))
    for idx, case in enumerate(cases):
        ax.plot(
            case["omega_hist"],
            case["lambda_hist"],
            marker="o",
            markersize=3.5,
            linewidth=2.0,
            color=colors[idx],
            label=f'{case["label"]} ({int(case["unknowns"]):,} dofs)',
        )
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(r"Continuation Curves Up To $\omega_{\max}=7\times 10^6$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_step_metric(
    cases: list[dict[str, object]],
    out_path: Path,
    *,
    value_key: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(cases), 1)))
    for idx, case in enumerate(cases):
        x = np.asarray(case["continuation_index"], dtype=np.int64)
        y = np.asarray(case[value_key], dtype=np.float64)
        if x.size == 0 or y.size == 0:
            continue
        ax.plot(x, y, marker="o", markersize=4.0, linewidth=2.0, color=colors[idx], label=str(case["label"]))
    ax.set_xlabel("Accepted Continuation Advance")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_final_lambda_vs_time(cases: list[dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(cases), 1)))
    for idx, case in enumerate(cases):
        x = float(case["runtime_seconds"])
        y = float(case["lambda_last"])
        ax.scatter([x], [y], s=85, color=colors[idx], label=str(case["label"]))
        ax.annotate(str(case["label"]), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Runtime [s] (log scale)")
    ax.set_ylabel(r"Final $\lambda$")
    ax.set_title(r"Final $\lambda$ vs Runtime")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_report(
    *,
    report_path: Path,
    summary_path: Path,
    out_root: Path,
    cases: list[dict[str, object]],
    commands: dict[str, list[str]],
    plot_paths: dict[str, Path],
    mpi_ranks: int,
    step_max: int,
    omega_max_stop: float,
) -> None:
    lines: list[str] = []
    lines.append("# Rank-8 PMG Comparison: P2(L1..L5) and P4(L1) at `omega_max = 7e6`")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- MPI ranks: `{int(mpi_ranks)}`")
    lines.append(f"- `omega_max_stop`: `{float(omega_max_stop):.1f}`")
    lines.append(f"- Runner `step_max`: `{int(step_max)}`")
    lines.append("- `step_max` counts accepted states including the 2-state initialization, so `step_max = continuation_advances + 2`.")
    lines.append("- `P2(LN)` runs use `pmg_shell` with `P2(LN) -> P1(LN) -> P1(L1)` for `N=2..5`, and same-mesh `P2(L1) -> P1(L1)` for `L1`.")
    lines.append("- `P4(L1)` is shown four times: the existing baseline, a capped rerun with `step_max = 12`, an uncapped capped-controller rerun with `step_max = 100`, and the newer smart-controller rerun with `step_max = 100`.")
    lines.append("- The Newton-based omega caps are `no increase if accepted-step Newton total > 10` and `halve next d_omega if > 20`.")
    lines.append("- `P2` cases use the mixed-hierarchy direct-elastic coarse-Hypre configuration; `P4` uses the previously best working same-mesh shell PMG coarse-Hypre configuration.")
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    for case in cases:
        lines.append(f"### {case['label']}")
        lines.append("")
        lines.append("```bash")
        lines.append(" ".join(commands[str(case["key"])]))
        lines.append("```")
        lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Case | Hierarchy | Unknowns | Accepted advances | Final lambda | Final omega | Runtime [s] |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for case in cases:
        lines.append(
            f"| {case['label']} | {case['hierarchy']} | {int(case['unknowns'])} | "
            f"{int(case['accepted_continuation_advances'])} | {float(case['lambda_last']):.9f} | "
            f"{float(case['omega_last']):.3f} | {float(case['runtime_seconds']):.3f} |"
        )
    lines.append("")
    p4_baseline = next((case for case in cases if str(case["key"]) == "p4_l1"), None)
    p4_capped = next((case for case in cases if str(case["key"]) == "p4_l1_newton_caps"), None)
    p4_uncapped = next((case for case in cases if str(case["key"]) == "p4_l1_newton_caps_uncapped"), None)
    p4_smart = next((case for case in cases if str(case["key"]) == "p4_l1_smart_controller_v2"), None)
    if p4_baseline is not None and p4_capped is not None:
        runtime_ratio = float(p4_capped["runtime_seconds"]) / max(float(p4_baseline["runtime_seconds"]), 1.0e-12)
        linear_ratio = float(p4_capped["attempt_linear_iterations_total"]) / max(
            float(p4_baseline["attempt_linear_iterations_total"]),
            1.0,
        )
        newton_ratio = float(p4_capped["attempt_newton_iterations_total"]) / max(
            float(p4_baseline["attempt_newton_iterations_total"]),
            1.0,
        )
        pc_apply_ratio = float(p4_capped["preconditioner_apply_time_total"]) / max(
            float(p4_baseline["preconditioner_apply_time_total"]),
            1.0e-12,
        )
        lines.append("## P4 Delta")
        lines.append("")
        lines.append("| Metric | P4(L1) baseline | P4(L1) + Newton omega caps | Ratio new/base |")
        lines.append("| --- | ---: | ---: | ---: |")
        lines.append(
            f"| Runtime [s] | {float(p4_baseline['runtime_seconds']):.3f} | "
            f"{float(p4_capped['runtime_seconds']):.3f} | {runtime_ratio:.3f} |"
        )
        lines.append(
            f"| Accepted continuation advances | {int(p4_baseline['accepted_continuation_advances'])} | "
            f"{int(p4_capped['accepted_continuation_advances'])} | - |"
        )
        lines.append(
            f"| Final omega | {float(p4_baseline['omega_last']):.3f} | "
            f"{float(p4_capped['omega_last']):.3f} | - |"
        )
        lines.append(
            f"| Final lambda | {float(p4_baseline['lambda_last']):.9f} | "
            f"{float(p4_capped['lambda_last']):.9f} | - |"
        )
        lines.append(
            f"| Continuation Newton iterations | {int(p4_baseline['attempt_newton_iterations_total'])} | "
            f"{int(p4_capped['attempt_newton_iterations_total'])} | {newton_ratio:.3f} |"
        )
        lines.append(
            f"| Continuation linear iterations | {int(p4_baseline['attempt_linear_iterations_total'])} | "
            f"{int(p4_capped['attempt_linear_iterations_total'])} | {linear_ratio:.3f} |"
        )
        lines.append(
            f"| Preconditioner apply total [s] | {float(p4_baseline['preconditioner_apply_time_total']):.3f} | "
            f"{float(p4_capped['preconditioner_apply_time_total']):.3f} | {pc_apply_ratio:.3f} |"
        )
        lines.append("")
    if p4_baseline is not None and p4_uncapped is not None:
        runtime_ratio = float(p4_uncapped["runtime_seconds"]) / max(float(p4_baseline["runtime_seconds"]), 1.0e-12)
        linear_ratio = float(p4_uncapped["attempt_linear_iterations_total"]) / max(
            float(p4_baseline["attempt_linear_iterations_total"]),
            1.0,
        )
        newton_ratio = float(p4_uncapped["attempt_newton_iterations_total"]) / max(
            float(p4_baseline["attempt_newton_iterations_total"]),
            1.0,
        )
        pc_apply_ratio = float(p4_uncapped["preconditioner_apply_time_total"]) / max(
            float(p4_baseline["preconditioner_apply_time_total"]),
            1.0e-12,
        )
        lines.append("## P4 Delta To Uncapped Rerun")
        lines.append("")
        lines.append("| Metric | P4(L1) baseline | P4(L1) + Newton omega caps, uncapped | Ratio new/base |")
        lines.append("| --- | ---: | ---: | ---: |")
        lines.append(
            f"| Runtime [s] | {float(p4_baseline['runtime_seconds']):.3f} | "
            f"{float(p4_uncapped['runtime_seconds']):.3f} | {runtime_ratio:.3f} |"
        )
        lines.append(
            f"| Accepted continuation advances | {int(p4_baseline['accepted_continuation_advances'])} | "
            f"{int(p4_uncapped['accepted_continuation_advances'])} | - |"
        )
        lines.append(
            f"| Final omega | {float(p4_baseline['omega_last']):.3f} | "
            f"{float(p4_uncapped['omega_last']):.3f} | - |"
        )
        lines.append(
            f"| Final lambda | {float(p4_baseline['lambda_last']):.9f} | "
            f"{float(p4_uncapped['lambda_last']):.9f} | - |"
        )
        lines.append(
            f"| Continuation Newton iterations | {int(p4_baseline['attempt_newton_iterations_total'])} | "
            f"{int(p4_uncapped['attempt_newton_iterations_total'])} | {newton_ratio:.3f} |"
        )
        lines.append(
            f"| Continuation linear iterations | {int(p4_baseline['attempt_linear_iterations_total'])} | "
            f"{int(p4_uncapped['attempt_linear_iterations_total'])} | {linear_ratio:.3f} |"
        )
        lines.append(
            f"| Preconditioner apply total [s] | {float(p4_baseline['preconditioner_apply_time_total']):.3f} | "
            f"{float(p4_uncapped['preconditioner_apply_time_total']):.3f} | {pc_apply_ratio:.3f} |"
        )
        lines.append("")
    if p4_baseline is not None and p4_smart is not None:
        runtime_ratio = float(p4_smart["runtime_seconds"]) / max(float(p4_baseline["runtime_seconds"]), 1.0e-12)
        linear_ratio = float(p4_smart["attempt_linear_iterations_total"]) / max(
            float(p4_baseline["attempt_linear_iterations_total"]),
            1.0,
        )
        newton_ratio = float(p4_smart["attempt_newton_iterations_total"]) / max(
            float(p4_baseline["attempt_newton_iterations_total"]),
            1.0,
        )
        pc_apply_ratio = float(p4_smart["preconditioner_apply_time_total"]) / max(
            float(p4_baseline["preconditioner_apply_time_total"]),
            1.0e-12,
        )
        lines.append("## P4 Delta To Smart Controller")
        lines.append("")
        lines.append("| Metric | P4(L1) baseline | P4(L1) + smart omega controller | Ratio new/base |")
        lines.append("| --- | ---: | ---: | ---: |")
        lines.append(
            f"| Runtime [s] | {float(p4_baseline['runtime_seconds']):.3f} | "
            f"{float(p4_smart['runtime_seconds']):.3f} | {runtime_ratio:.3f} |"
        )
        lines.append(
            f"| Accepted continuation advances | {int(p4_baseline['accepted_continuation_advances'])} | "
            f"{int(p4_smart['accepted_continuation_advances'])} | - |"
        )
        lines.append(
            f"| Final omega | {float(p4_baseline['omega_last']):.3f} | "
            f"{float(p4_smart['omega_last']):.3f} | - |"
        )
        lines.append(
            f"| Final lambda | {float(p4_baseline['lambda_last']):.9f} | "
            f"{float(p4_smart['lambda_last']):.9f} | - |"
        )
        lines.append(
            f"| Continuation Newton iterations | {int(p4_baseline['attempt_newton_iterations_total'])} | "
            f"{int(p4_smart['attempt_newton_iterations_total'])} | {newton_ratio:.3f} |"
        )
        lines.append(
            f"| Continuation linear iterations | {int(p4_baseline['attempt_linear_iterations_total'])} | "
            f"{int(p4_smart['attempt_linear_iterations_total'])} | {linear_ratio:.3f} |"
        )
        lines.append(
            f"| Preconditioner apply total [s] | {float(p4_baseline['preconditioner_apply_time_total']):.3f} | "
            f"{float(p4_smart['preconditioner_apply_time_total']):.3f} | {pc_apply_ratio:.3f} |"
        )
        lines.append("")
    lines.append("## Iteration Totals")
    lines.append("")
    lines.append(
        "| Case | Attempts | Successful attempts | Continuation Newton iters | Continuation linear iters | Linear/Newton | Init linear iters | Preconditioner rebuilds |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in cases:
        lines.append(
            f"| {case['label']} | {int(case['attempt_count'])} | {int(case['successful_attempt_count'])} | "
            f"{int(case['attempt_newton_iterations_total'])} | {int(case['attempt_linear_iterations_total'])} | "
            f"{float(case['attempt_linear_per_newton']):.3f} | {int(case['init_linear_iterations'])} | "
            f"{int(case['preconditioner_rebuild_count'])} |"
        )
    lines.append("")
    lines.append("## Linear Timing Totals")
    lines.append("")
    lines.append(
        "| Case | Init solve [s] | Init PC [s] | Init orthog [s] | Continuation solve [s] | Continuation PC [s] | Continuation orthog [s] | PC setup total [s] | PC apply total [s] |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in cases:
        lines.append(
            f"| {case['label']} | {float(case['init_linear_solve_time']):.3f} | "
            f"{float(case['init_linear_preconditioner_time']):.3f} | {float(case['init_linear_orthogonalization_time']):.3f} | "
            f"{float(case['attempt_linear_solve_time_total']):.3f} | {float(case['attempt_linear_preconditioner_time_total']):.3f} | "
            f"{float(case['attempt_linear_orthogonalization_time_total']):.3f} | "
            f"{float(case['preconditioner_setup_time_total']):.3f} | {float(case['preconditioner_apply_time_total']):.3f} |"
        )
    lines.append("")
    lines.append("## Constitutive Timing Totals")
    lines.append("")
    lines.append(
        "| Case | build_tangent_local [s] | build_F [s] | local_strain [s] | local_constitutive [s] | stress [s] | stress_tangent [s] |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in cases:
        lines.append(
            f"| {case['label']} | {float(case['build_tangent_local']):.3f} | {float(case['build_F']):.3f} | "
            f"{float(case['local_strain']):.3f} | {float(case['local_constitutive']):.3f} | "
            f"{float(case['stress']):.3f} | {float(case['stress_tangent']):.3f} |"
        )
    lines.append("")
    lines.append("## PMG Layout")
    lines.append("")
    lines.append(
        "| Case | ManualMG levels | Level orders | Level global sizes | Coarse operator | Coarse KSP/PC | Fine smoother | Mid smoother |"
    )
    lines.append("| --- | ---: | --- | --- | --- | --- | --- | --- |")
    for case in cases:
        level_orders = "[" + ", ".join(str(int(v)) for v in case["manualmg_level_orders"]) + "]"
        level_sizes = "[" + ", ".join(str(int(v)) for v in case["manualmg_level_global_sizes"]) + "]"
        coarse_desc = str(case["manualmg_coarse_pc_type"])
        if case["manualmg_coarse_hypre_type"]:
            coarse_desc += f"/{case['manualmg_coarse_hypre_type']}"
        fine_smoother = "/".join(
            item for item in (str(case["manualmg_fine_ksp_type"]), str(case["manualmg_fine_pc_type"])) if item
        )
        mid_smoother = "/".join(
            item for item in (str(case["manualmg_mid_ksp_type"]), str(case["manualmg_mid_pc_type"])) if item
        )
        lines.append(
            f"| {case['label']} | {int(case['manualmg_levels'])} | {level_orders} | {level_sizes} | "
            f"{case['manualmg_coarse_operator_source']} | {case['manualmg_coarse_ksp_type']}/{coarse_desc} | "
            f"{fine_smoother or '-'} | {mid_smoother or '-'} |"
        )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append(f"![Lambda Omega]({_report_relpath(report_path, plot_paths['lambda_omega'])})")
    lines.append("")
    lines.append(f"![Step Wall Time]({_report_relpath(report_path, plot_paths['step_wall_time'])})")
    lines.append("")
    lines.append(f"![Step Newton Iterations]({_report_relpath(report_path, plot_paths['step_newton_iterations'])})")
    lines.append("")
    lines.append(f"![Step Linear Iterations]({_report_relpath(report_path, plot_paths['step_linear_iterations'])})")
    lines.append("")
    lines.append(f"![Step Linear Per Newton]({_report_relpath(report_path, plot_paths['step_linear_per_newton'])})")
    lines.append("")
    lines.append(f"![Final Lambda Vs Time]({_report_relpath(report_path, plot_paths['final_lambda_vs_time'])})")
    lines.append("")
    lines.append("## Raw Artifacts")
    lines.append("")
    lines.append(f"- Summary JSON: [{summary_path.name}]({_report_relpath(report_path, summary_path)})")
    for case in cases:
        lines.append(
            f"- {case['label']}: "
            f"[run_info.json]({_report_relpath(report_path, case['run_info_path'])}), "
            f"[petsc_run.npz]({_report_relpath(report_path, case['npz_path'])})"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--step-max", type=int, default=12)
    parser.add_argument("--omega-max-stop", type=float, default=7.0e6)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    out_root = _ensure_dir(Path(args.out_root))
    plots_dir = _ensure_dir(out_root / "plots")

    commands: dict[str, list[str]] = {}
    case_results: list[dict[str, object]] = []
    for case_spec in CASE_SPECS:
        case_step_max = _case_step_max(case_spec, int(args.step_max))
        case_out_dir = _case_out_dir(out_root, case_spec, int(args.mpi_ranks), int(args.step_max))
        commands[str(case_spec["key"])] = _case_command(
            case_spec=case_spec,
            out_dir=case_out_dir,
            mpi_ranks=int(args.mpi_ranks),
            step_max=int(case_step_max),
            omega_max_stop=float(args.omega_max_stop),
        )
        run_info_path = case_out_dir / "data" / "run_info.json"
        if not args.skip_run and not (args.skip_existing and run_info_path.exists()):
            _ensure_dir(case_out_dir)
            _run(
                commands[str(case_spec["key"])],
                env=_default_env(),
                cwd=ROOT,
            )
        case_results.append(_load_case(case_out_dir, case_spec=case_spec))

    plot_paths = {
        "lambda_omega": plots_dir / "lambda_omega.png",
        "step_wall_time": plots_dir / "step_wall_time.png",
        "step_newton_iterations": plots_dir / "step_newton_iterations.png",
        "step_linear_iterations": plots_dir / "step_linear_iterations.png",
        "step_linear_per_newton": plots_dir / "step_linear_per_newton.png",
        "final_lambda_vs_time": plots_dir / "final_lambda_vs_time.png",
    }
    _plot_lambda_omega(case_results, plot_paths["lambda_omega"])
    _plot_step_metric(
        case_results,
        plot_paths["step_wall_time"],
        value_key="step_wall_time",
        ylabel="Wall Time [s]",
        title="Accepted Step Wall Time",
    )
    _plot_step_metric(
        case_results,
        plot_paths["step_newton_iterations"],
        value_key="step_newton_iterations_total",
        ylabel="Newton Iterations",
        title="Accepted Step Newton Iterations",
    )
    _plot_step_metric(
        case_results,
        plot_paths["step_linear_iterations"],
        value_key="step_linear_iterations",
        ylabel="Linear Iterations",
        title="Accepted Step Linear Iterations",
    )
    _plot_step_metric(
        case_results,
        plot_paths["step_linear_per_newton"],
        value_key="step_linear_per_newton",
        ylabel="Linear Iterations / Newton",
        title="Accepted Step Linear Iterations Per Newton",
    )
    _plot_final_lambda_vs_time(case_results, plot_paths["final_lambda_vs_time"])

    summary = {
        "mpi_ranks": int(args.mpi_ranks),
        "step_max": int(args.step_max),
        "omega_max_stop": float(args.omega_max_stop),
        "cases": case_results,
        "commands": commands,
        "plots": {key: str(path) for key, path in plot_paths.items()},
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8")
    _write_report(
        report_path=Path(args.report_path),
        summary_path=summary_path,
        out_root=out_root,
        cases=case_results,
        commands=commands,
        plot_paths=plot_paths,
        mpi_ranks=int(args.mpi_ranks),
        step_max=int(args.step_max),
        omega_max_stop=float(args.omega_max_stop),
    )


if __name__ == "__main__":
    main()
