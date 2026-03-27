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

from slope_stability.mesh import load_mesh_from_file


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "archive" else SCRIPT_DIR
ROOT = BENCHMARK_DIR.parents[1]

MESH_DIR = ROOT / "meshes" / "3d_hetero_ssr"
FINE_MESH = MESH_DIR / "SSR_hetero_ada_L5.msh"
COARSE_CANDIDATES: dict[str, Path | tuple[Path, ...]] = {
    "L4": MESH_DIR / "SSR_hetero_ada_L4.msh",
    "L3": MESH_DIR / "SSR_hetero_ada_L3.msh",
    "L2": MESH_DIR / "SSR_hetero_ada_L2.msh",
    "L1": MESH_DIR / "SSR_hetero_ada_L1.msh",
    "L4_L3_L2_L1_tail": (
        MESH_DIR / "SSR_hetero_ada_L4.msh",
        MESH_DIR / "SSR_hetero_ada_L3.msh",
        MESH_DIR / "SSR_hetero_ada_L2.msh",
        MESH_DIR / "SSR_hetero_ada_L1.msh",
    ),
}
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "p2_l5_rank8_mixed_pmg_step10"
DEFAULT_REPORT = SCRIPT_DIR / "report_p2_l5_rank8_mixed_pmg_step10.md"

STATE_OUT_NAME = "state_hypre_rank8_step1"
HYPRE_SMOKE_NAME = "hypre_frozen"
PMG_CONT_NAME = "pmg_shell_rank8_step12"
CONTINUATION_CANDIDATE_KEY = "L1"

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

REFERENCE_CURVES: tuple[tuple[str, str, Path], ...] = (
    (
        "P2(L1) reference",
        "artifacts/p2_p4_compare_rank8_final_guarded80_v2/p2_rank8_step100/data/petsc_run.npz",
        ROOT / "artifacts" / "p2_p4_compare_rank8_final_guarded80_v2" / "p2_rank8_step100" / "data" / "petsc_run.npz",
    ),
    (
        "P2(L2) PMG-shell",
        "artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/pmg_shell_mixed_rank8_step12/data/petsc_run.npz",
        ROOT / "artifacts" / "p2_l2_rank8_hypre_vs_mixed_pmg_step10" / "pmg_shell_mixed_rank8_step12" / "data" / "petsc_run.npz",
    ),
    (
        "P4(L1) PMG-shell",
        "artifacts/p4_pmg_shell_best_rank8_full/p4_rank8_step100/data/petsc_run.npz",
        ROOT / "artifacts" / "p4_pmg_shell_best_rank8_full" / "p4_rank8_step100" / "data" / "petsc_run.npz",
    ),
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _report_relpath(report_path: Path, target: Path | str) -> str:
    return os.path.relpath(Path(target).resolve(), report_path.parent.resolve())


def _coarse_mesh_tuple(spec: Path | tuple[Path, ...] | list[Path] | None) -> tuple[Path, ...]:
    if spec is None:
        return ()
    if isinstance(spec, Path):
        return (spec,)
    return tuple(Path(path) for path in spec)


def _mesh_level_label(mesh_path: Path) -> str:
    stem = mesh_path.stem
    if "_L" in stem:
        return "L" + stem.rsplit("_L", 1)[1]
    return mesh_path.stem


def _hierarchy_label_for_spec(spec: Path | tuple[Path, ...] | list[Path] | None) -> str:
    coarse_paths = _coarse_mesh_tuple(spec)
    if not coarse_paths:
        return "direct P2(L5)"
    levels = ["P2(L5)", "P1(L5)"] + [f"P1({_mesh_level_label(path)})" for path in coarse_paths]
    return " -> ".join(levels)


def _candidate_display_label(key: str) -> str:
    if key == "L4_L3_L2_L1_tail":
        return "L4->L1 tail"
    return key


def _run(cmd: list[str], *, env: dict[str, str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


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


def _run_state_case(*, out_dir: Path, mpi_ranks: int, step_max: int) -> list[str]:
    cmd = _state_case_command(out_dir=out_dir, mpi_ranks=mpi_ranks, step_max=step_max)
    _run(cmd, env=_default_env(), cwd=ROOT)
    return cmd


def _state_case_command(*, out_dir: Path, mpi_ranks: int, step_max: int) -> list[str]:
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
        str(FINE_MESH),
        "--elem_type",
        "P2",
        "--node_ordering",
        "original",
        "--step_max",
        str(int(step_max)),
        "--solver_type",
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        "--pc_backend",
        "hypre",
    ]
    return cmd


def _write_zero_state_case(*, out_dir: Path) -> list[str]:
    _ensure_dir(out_dir / "data")
    mesh = load_mesh_from_file(FINE_MESH, boundary_type=0, elem_type="P2")
    U = np.zeros_like(np.asarray(mesh.coord, dtype=np.float64))
    np.savez(
        out_dir / "data" / "petsc_run.npz",
        U=U,
        step_U=np.zeros((1,) + U.shape, dtype=np.float64),
        lambda_hist=np.asarray([1.0], dtype=np.float64),
        omega_hist=np.asarray([0.0], dtype=np.float64),
        Umax_hist=np.asarray([0.0], dtype=np.float64),
    )
    payload = {
        "params": {
            "node_ordering": "original",
            "elem_type": "P2",
            "r_min": 1.0e-4,
            "material_rows": None,
            "mesh_path": str(FINE_MESH),
            "state_mode": "synthetic_zero",
        }
    }
    (out_dir / "data" / "run_info.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return _zero_state_command_repr(out_dir=out_dir)


def _zero_state_command_repr(*, out_dir: Path) -> list[str]:
    return ["synthetic_zero_state", str(out_dir / "data" / "petsc_run.npz")]


def _run_frozen_probe(
    *,
    out_dir: Path,
    state_npz: Path,
    state_run_info: Path,
    pc_backend: str,
    linear_tolerance: float,
    linear_max_iter: int,
    coarse_meshes: Path | tuple[Path, ...] | list[Path] | None = None,
) -> list[str]:
    cmd = _frozen_probe_command(
        out_dir=out_dir,
        state_npz=state_npz,
        state_run_info=state_run_info,
        pc_backend=pc_backend,
        linear_tolerance=linear_tolerance,
        linear_max_iter=linear_max_iter,
        coarse_meshes=coarse_meshes,
    )
    _run(cmd, env=_default_env(), cwd=ROOT)
    return cmd


def _frozen_probe_command(
    *,
    out_dir: Path,
    state_npz: Path,
    state_run_info: Path,
    pc_backend: str,
    linear_tolerance: float,
    linear_max_iter: int,
    coarse_meshes: Path | tuple[Path, ...] | list[Path] | None = None,
) -> list[str]:
    cmd = [
        "mpirun",
        "-n",
        "8",
        sys.executable,
        str(SCRIPT_DIR / "probe_hypre_frozen.py"),
        "--out-dir",
        str(out_dir),
        "--state-npz",
        str(state_npz),
        "--state-run-info",
        str(state_run_info),
        "--state-selector",
        "final",
        "--mesh-path",
        str(FINE_MESH),
        "--elem-type",
        "P2",
        "--node-ordering",
        "original",
        "--outer-solver-family",
        "repo",
        "--solver-type",
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        "--pc-backend",
        str(pc_backend),
        "--pmat-source",
        "tangent",
        "--linear-tolerance",
        str(float(linear_tolerance)),
        "--linear-max-iter",
        str(int(linear_max_iter)),
    ]
    for coarse_mesh in _coarse_mesh_tuple(coarse_meshes):
        cmd.extend(["--pmg-coarse-mesh-path", str(coarse_mesh)])
    if str(pc_backend).strip().lower() == "pmg_shell":
        for opt in PMG_PETSC_OPTIONS:
            cmd.extend(["--petsc-opt", opt])
    return cmd


def _run_continuation_case(
    *,
    out_dir: Path,
    mpi_ranks: int,
    step_max: int,
    coarse_mesh: Path,
) -> list[str]:
    cmd = _continuation_case_command(
        out_dir=out_dir,
        mpi_ranks=mpi_ranks,
        step_max=step_max,
        coarse_mesh=coarse_mesh,
    )
    _run(cmd, env=_default_env(), cwd=ROOT)
    return cmd


def _continuation_case_command(
    *,
    out_dir: Path,
    mpi_ranks: int,
    step_max: int,
    coarse_mesh: Path,
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
        str(FINE_MESH),
        "--elem_type",
        "P2",
        "--node_ordering",
        "original",
        "--step_max",
        str(int(step_max)),
        "--solver_type",
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        "--pc_backend",
        "pmg_shell",
        "--preconditioner_matrix_source",
        "tangent",
        "--pmg_coarse_mesh_path",
        str(coarse_mesh),
        "--no-store_step_u",
    ]
    for opt in PMG_PETSC_OPTIONS:
        cmd.extend(["--petsc-opt", opt])
    return cmd


def _load_smoke_case(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_coarse = data.get("pmg_coarse_mesh_path")
    if raw_coarse is None:
        coarse_field: str | list[str] | None = None
    elif isinstance(raw_coarse, list):
        coarse_field = [str(item) for item in raw_coarse]
    else:
        coarse_field = str(raw_coarse)
    return {
        "status": str(data.get("status", "")),
        "mesh_path": str(data.get("mesh_path", "")),
        "pmg_coarse_mesh_path": coarse_field,
        "pc_backend": str(data.get("pc_backend", "")),
        "runtime_seconds": float(data.get("runtime_seconds", 0.0)),
        "runtime_seconds_max": float(data.get("runtime_seconds_max", data.get("runtime_seconds", 0.0))),
        "setup_elapsed_s": float(data.get("setup_elapsed_s", 0.0)),
        "setup_elapsed_s_max": float(data.get("setup_elapsed_s_max", data.get("setup_elapsed_s", 0.0))),
        "solve_elapsed_s": float(data.get("solve_elapsed_s", 0.0)),
        "solve_elapsed_s_max": float(data.get("solve_elapsed_s_max", data.get("solve_elapsed_s", 0.0))),
        "solve_plus_setup_elapsed_s": float(data.get("solve_plus_setup_elapsed_s", 0.0)),
        "solve_plus_setup_elapsed_s_max": float(
            data.get("solve_plus_setup_elapsed_s_max", data.get("solve_plus_setup_elapsed_s", 0.0))
        ),
        "iteration_count": int(data.get("iteration_count", 0)),
        "final_relative_residual": float(data.get("final_relative_residual", np.inf)),
        "preconditioner_setup_time_total": float(data.get("preconditioner_setup_time_total", 0.0)),
        "preconditioner_apply_time_total": float(data.get("preconditioner_apply_time_total", 0.0)),
        "manualmg_setup_time_s": float(data.get("manualmg_setup_time_s", 0.0)),
        "manualmg_level_global_sizes": list(data.get("manualmg_level_global_sizes", [])),
        "manualmg_level_orders": list(data.get("manualmg_level_orders", [])),
        "manualmg_apply_count": int(data.get("manualmg_apply_count", 0)),
        "manualmg_coarse_iterations": int(data.get("manualmg_coarse_iterations", 0)),
        "manualmg_coarse_ksp_iterations_total": int(data.get("manualmg_coarse_ksp_iterations_total", 0)),
        "manualmg_coarse_solve_count": int(data.get("manualmg_coarse_solve_count", 0)),
        "manualmg_coarse_hypre_time_total_s": float(data.get("manualmg_coarse_hypre_time_total_s", 0.0)),
        "manualmg_fine_pre_smoother_time_total_s": float(data.get("manualmg_fine_pre_smoother_time_total_s", 0.0)),
        "manualmg_fine_post_smoother_time_total_s": float(data.get("manualmg_fine_post_smoother_time_total_s", 0.0)),
        "manualmg_mid_pre_smoother_time_total_s": float(data.get("manualmg_mid_pre_smoother_time_total_s", 0.0)),
        "manualmg_mid_post_smoother_time_total_s": float(data.get("manualmg_mid_post_smoother_time_total_s", 0.0)),
        "manualmg_fine_residual_time_total_s": float(data.get("manualmg_fine_residual_time_total_s", 0.0)),
        "manualmg_mid_residual_time_total_s": float(data.get("manualmg_mid_residual_time_total_s", 0.0)),
        "manualmg_restrict_fine_to_mid_time_total_s": float(data.get("manualmg_restrict_fine_to_mid_time_total_s", 0.0)),
        "manualmg_restrict_mid_to_coarse_time_total_s": float(data.get("manualmg_restrict_mid_to_coarse_time_total_s", 0.0)),
        "manualmg_prolong_coarse_to_mid_time_total_s": float(data.get("manualmg_prolong_coarse_to_mid_time_total_s", 0.0)),
        "manualmg_prolong_mid_to_fine_time_total_s": float(data.get("manualmg_prolong_mid_to_fine_time_total_s", 0.0)),
        "manualmg_vector_sum_time_total_s": float(data.get("manualmg_vector_sum_time_total_s", 0.0)),
        "run_info_path": str(path),
    }


def _load_continuation_case(out_dir: Path) -> dict[str, object]:
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

    info = run_info["run_info"]
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
        "step_newton_iterations_total": step_newton_iterations_total,
        "step_linear_iterations": step_linear_iterations,
        "step_linear_solve_time": step_linear_solve_time,
        "step_linear_preconditioner_time": step_linear_preconditioner_time,
        "step_linear_orthogonalization_time": step_linear_orthogonalization_time,
        "step_linear_per_newton": step_linear_per_newton,
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
        "params": run_info["params"],
    }


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {k: _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    return value


def _choose_best_smoke(smokes: dict[str, dict[str, object]], *, tolerance: float) -> tuple[str, dict[str, object]]:
    converged = [
        (label, case)
        for label, case in smokes.items()
        if float(case["final_relative_residual"]) <= float(tolerance)
    ]
    if not converged:
        label, case = min(
            smokes.items(),
            key=lambda item: (float(item[1]["final_relative_residual"]), float(item[1]["solve_plus_setup_elapsed_s_max"])),
        )
        return label, case
    return min(converged, key=lambda item: float(item[1]["solve_plus_setup_elapsed_s_max"]))


def _load_reference_curves() -> list[dict[str, object]]:
    curves: list[dict[str, object]] = []
    for label, artifact_rel, path in REFERENCE_CURVES:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as npz:
            curves.append(
                {
                    "label": label,
                    "artifact_rel": artifact_rel,
                    "path": str(path),
                    "lambda_hist": np.asarray(npz["lambda_hist"], dtype=np.float64),
                    "omega_hist": np.asarray(npz["omega_hist"], dtype=np.float64),
                }
            )
    return curves


def _plot_smoke(out_path: Path, *, smoke_hypre: dict[str, object], pmg_smokes: dict[str, dict[str, object]]) -> None:
    levels = list(pmg_smokes.keys())
    level_labels = [_candidate_display_label(level) for level in levels]
    x = np.arange(len(levels), dtype=np.int64)
    total = np.asarray([pmg_smokes[level]["solve_plus_setup_elapsed_s_max"] for level in levels], dtype=np.float64)
    setup = np.asarray([pmg_smokes[level]["setup_elapsed_s_max"] for level in levels], dtype=np.float64)
    solve = np.asarray([pmg_smokes[level]["solve_elapsed_s_max"] for level in levels], dtype=np.float64)
    iters = np.asarray([pmg_smokes[level]["iteration_count"] for level in levels], dtype=np.float64)
    resid = np.asarray([pmg_smokes[level]["final_relative_residual"] for level in levels], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    width = 0.26
    axes[0].bar(x - width, setup, width=width, label="setup")
    axes[0].bar(x, solve, width=width, label="solve")
    axes[0].bar(x + width, total, width=width, label="setup+solve")
    axes[0].axhline(float(smoke_hypre["solve_plus_setup_elapsed_s_max"]), color="black", linestyle="--", linewidth=1.5, label="Hypre frozen")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(level_labels, rotation=15)
    axes[0].set_xlabel("Lowest PMG mesh level")
    axes[0].set_ylabel("Wall time [s]")
    axes[0].set_title("Frozen smoke wall times")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(level_labels, iters, marker="o", linewidth=2, label="PMG-shell")
    axes[1].axhline(float(smoke_hypre["iteration_count"]), color="black", linestyle="--", linewidth=1.5, label="Hypre frozen")
    axes[1].set_xlabel("Lowest PMG mesh level")
    axes[1].set_ylabel("Outer DFGMRES iterations")
    axes[1].set_title("Frozen smoke iterations")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].semilogy(level_labels, resid, marker="o", linewidth=2, label="PMG-shell")
    axes[2].axhline(float(smoke_hypre["final_relative_residual"]), color="black", linestyle="--", linewidth=1.5, label="Hypre frozen")
    axes[2].axhline(1.0e-3, color="tab:red", linestyle=":", linewidth=1.5, label="target")
    axes[2].set_xlabel("Lowest PMG mesh level")
    axes[2].set_ylabel("Final relative residual")
    axes[2].set_title("Frozen smoke convergence")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_lambda_omega(out_path: Path, *, continuation: dict[str, object], refs: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    ax.plot(
        continuation["omega_hist"],
        continuation["lambda_hist"],
        marker="o",
        linewidth=2.5,
        label="P2(L5) PMG-shell",
    )
    for ref in refs:
        ax.plot(ref["omega_hist"], ref["lambda_hist"], linewidth=2.0, alpha=0.9, label=ref["label"])
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(r"Continuation curves: $\lambda(\omega)$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_step_metrics(out_path: Path, *, continuation: dict[str, object]) -> None:
    x = np.asarray(continuation["continuation_index"], dtype=np.int64)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    series = (
        ("step_wall_time", "Step wall time [s]"),
        ("step_newton_iterations_total", "Newton iterations / accepted step"),
        ("step_linear_iterations", "Linear iterations / accepted step"),
        ("step_linear_per_newton", "Linear iterations / Newton"),
    )
    for ax, (key, ylabel) in zip(axes.ravel(), series):
        y = np.asarray(continuation[key], dtype=np.float64)
        ax.plot(x, y, marker="o", linewidth=2)
        ax.set_xlabel("Accepted continuation step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_timing_breakdown(out_path: Path, *, continuation: dict[str, object]) -> None:
    components = {
        "init solve": float(continuation["init_linear_solve_time"]),
        "init prec": float(continuation["init_linear_preconditioner_time"]),
        "continuation solves": float(continuation["attempt_linear_solve_time_total"]),
        "continuation prec": float(continuation["attempt_linear_preconditioner_time_total"]),
        "build tangent": float(continuation["build_tangent_local"]),
        "build F": float(continuation["build_F"]),
    }
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    labels = list(components.keys())
    values = np.asarray(list(components.values()), dtype=np.float64)
    ax.bar(labels, values)
    ax.set_ylabel("Seconds")
    ax.set_title("Selected total timing components")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _smoke_rows(
    smoke_hypre: dict[str, object],
    pmg_smokes: dict[str, dict[str, object]],
    *,
    candidate_specs: dict[str, Path | tuple[Path, ...]],
    tolerance: float,
    best_level: str,
) -> list[str]:
    rows = [
        "| Case | Hierarchy | Iterations | Final rel. residual | Setup max [s] | Solve max [s] | Setup+solve max [s] | Converged @ 1e-3 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        (
            f"| Hypre frozen | direct P2(L5) | {int(smoke_hypre['iteration_count'])} | "
            f"{float(smoke_hypre['final_relative_residual']):.3e} | "
            f"{float(smoke_hypre['setup_elapsed_s_max']):.3f} | "
            f"{float(smoke_hypre['solve_elapsed_s_max']):.3f} | "
            f"{float(smoke_hypre['solve_plus_setup_elapsed_s_max']):.3f} | "
            f"{'yes' if float(smoke_hypre['final_relative_residual']) <= float(tolerance) else 'no'} |"
        ),
    ]
    for level, case in pmg_smokes.items():
        hierarchy = _hierarchy_label_for_spec(candidate_specs[level])
        converged = float(case["final_relative_residual"]) <= float(tolerance)
        best = " (best)" if level == best_level else ""
        rows.append(
            f"| PMG-shell `{level}`{best} | {hierarchy} | {int(case['iteration_count'])} | "
            f"{float(case['final_relative_residual']):.3e} | "
            f"{float(case['setup_elapsed_s_max']):.3f} | "
            f"{float(case['solve_elapsed_s_max']):.3f} | "
            f"{float(case['solve_plus_setup_elapsed_s_max']):.3f} | "
            f"{'yes' if converged else 'no'} |"
        )
    return rows


def _breakdown_rows(case: dict[str, object]) -> list[str]:
    rows = [
        "| PMG-shell component | Total wall time [s] | Per V-cycle [ms] |",
        "| --- | ---: | ---: |",
    ]
    apply_count = max(int(case.get("manualmg_apply_count", 0)), 1)
    keys = (
        ("Fine pre smoother", "manualmg_fine_pre_smoother_time_total_s"),
        ("Fine post smoother", "manualmg_fine_post_smoother_time_total_s"),
        ("Mid pre smoother", "manualmg_mid_pre_smoother_time_total_s"),
        ("Mid post smoother", "manualmg_mid_post_smoother_time_total_s"),
        ("Fine residual", "manualmg_fine_residual_time_total_s"),
        ("Mid residual", "manualmg_mid_residual_time_total_s"),
        ("Restrict fine->mid", "manualmg_restrict_fine_to_mid_time_total_s"),
        ("Restrict mid->coarse", "manualmg_restrict_mid_to_coarse_time_total_s"),
        ("Prolong coarse->mid", "manualmg_prolong_coarse_to_mid_time_total_s"),
        ("Prolong mid->fine", "manualmg_prolong_mid_to_fine_time_total_s"),
        ("Vector sum", "manualmg_vector_sum_time_total_s"),
        ("Coarse Hypre", "manualmg_coarse_hypre_time_total_s"),
    )
    for label, key in keys:
        total = float(case.get(key, 0.0))
        rows.append(f"| {label} | {total:.3f} | {1.0e3 * total / apply_count:.3f} |")
    return rows


def _timing_rows(cont: dict[str, object]) -> list[str]:
    rows = [
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Runtime [s] | {float(cont['runtime_seconds']):.3f} |",
        f"| Continuation wall time [s] | {float(cont['continuation_total_wall_time']):.3f} |",
        f"| Unknowns | {int(cont['unknowns'])} |",
        f"| Init linear iterations | {int(cont['init_linear_iterations'])} |",
        f"| Continuation linear iterations total | {int(cont['attempt_linear_iterations_total'])} |",
        (
            "| Total linear / Newton | "
            f"{float(cont['attempt_linear_iterations_total']) / max(float(np.asarray(cont['step_newton_iterations_total']).sum()), 1.0):.3f} |"
        ),
        f"| Init solve collector [s] | {float(cont['init_linear_solve_time']):.3f} |",
        f"| Init preconditioner collector [s] | {float(cont['init_linear_preconditioner_time']):.3f} |",
        f"| Continuation solve collector [s] | {float(cont['attempt_linear_solve_time_total']):.3f} |",
        f"| Continuation preconditioner collector [s] | {float(cont['attempt_linear_preconditioner_time_total']):.3f} |",
        f"| Continuation orthogonalization collector [s] | {float(cont['attempt_linear_orthogonalization_time_total']):.3f} |",
        f"| Preconditioner setup diagnostic [s] | {float(cont['preconditioner_setup_time_total']):.3f} |",
        f"| Preconditioner apply diagnostic [s] | {float(cont['preconditioner_apply_time_total']):.3f} |",
        f"| Preconditioner rebuild count | {int(cont['preconditioner_rebuild_count'])} |",
        f"| build_tangent_local [s] | {float(cont['build_tangent_local']):.3f} |",
        f"| build_F [s] | {float(cont['build_F']):.3f} |",
    ]
    return rows


def _step_rows(cont: dict[str, object], *, limit: int) -> list[str]:
    rows = [
        "| Cont. step | Lambda | Omega | Wall [s] | Attempts | Newton | Linear | Linear/Newton |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    x = np.asarray(cont["continuation_index"], dtype=np.int64)
    for idx in range(1, int(limit) + 1):
        mask = x == idx
        if not np.any(mask):
            continue
        rows.append(
            f"| {idx} | "
            f"{float(np.asarray(cont['step_lambda'])[mask][0]):.9f} | "
            f"{float(np.asarray(cont['step_omega'])[mask][0]):.3f} | "
            f"{float(np.asarray(cont['step_wall_time'])[mask][0]):.3f} | "
            f"{int(np.asarray(cont['step_attempt_count'])[mask][0])} | "
            f"{int(np.asarray(cont['step_newton_iterations_total'])[mask][0])} | "
            f"{int(np.asarray(cont['step_linear_iterations'])[mask][0])} | "
            f"{float(np.asarray(cont['step_linear_per_newton'])[mask][0]):.3f} |"
        )
    return rows


def _write_summary(
    path: Path,
    *,
    state_out_dir: Path,
    smoke_hypre: dict[str, object],
    smoke_pmgs: dict[str, dict[str, object]],
    candidate_specs: dict[str, Path | tuple[Path, ...]],
    best_level: str,
    continuation_choice: str,
    continuation: dict[str, object],
    continuation_steps: int,
    step_max: int,
    mpi_ranks: int,
    commands: dict[str, object],
    reference_curves: list[dict[str, object]],
) -> None:
    payload = {
        "mpi_ranks": int(mpi_ranks),
        "continuation_steps_requested": int(continuation_steps),
        "step_max": int(step_max),
        "state_out_dir": str(state_out_dir),
        "fine_mesh": str(FINE_MESH),
        "smoke_hypre": _serializable(smoke_hypre),
        "smoke_pmg_shell": _serializable(smoke_pmgs),
        "smoke_candidate_meshes": {key: [str(path) for path in _coarse_mesh_tuple(spec)] for key, spec in candidate_specs.items()},
        "best_coarse_level": str(best_level),
        "best_coarse_mesh": [str(path) for path in _coarse_mesh_tuple(candidate_specs[best_level])],
        "continuation_choice": str(continuation_choice),
        "continuation_coarse_mesh": [str(path) for path in _coarse_mesh_tuple(candidate_specs[continuation_choice])],
        "continuation": _serializable(continuation),
        "commands": _serializable(commands),
        "reference_curves": _serializable(reference_curves),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_report(
    report_path: Path,
    *,
    out_root: Path,
    plots: dict[str, Path],
    state_out_dir: Path,
    smoke_hypre: dict[str, object],
    smoke_pmgs: dict[str, dict[str, object]],
    candidate_specs: dict[str, Path | tuple[Path, ...]],
    best_level: str,
    continuation_choice: str,
    continuation: dict[str, object],
    continuation_steps: int,
    step_max: int,
    mpi_ranks: int,
    commands: dict[str, object],
    reference_curves: list[dict[str, object]],
) -> None:
    plot_smoke = _report_relpath(report_path, plots["smoke"])
    plot_curve = _report_relpath(report_path, plots["curve"])
    plot_steps = _report_relpath(report_path, plots["steps"])
    plot_timing = _report_relpath(report_path, plots["timing"])
    state_run = _report_relpath(report_path, state_out_dir / "data" / "run_info.json")
    cont_run = _report_relpath(report_path, continuation["run_info_path"])
    cont_npz = _report_relpath(report_path, continuation["npz_path"])

    lines = [
        "# P2(L5) Rank-8 Mixed PMG-Shell",
        "",
        "## Configuration",
        "",
        f"- Fine mesh: `{FINE_MESH.relative_to(ROOT)}`",
        f"- Continuation backend: `pmg_shell`",
        f"- Mixed hierarchy family: `P2(L5) -> P1(L5)` with optional deeper `P1(Lk)` h-tail",
        f"- MPI ranks: `{int(mpi_ranks)}`",
        f"- Outer solver: `PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE`",
        "- Frozen smoke state: synthetic zero displacement on `P2(L5)` with `lambda=1.0`, `omega=0.0`.",
        f"- Frozen smoke tolerance: `1e-3`",
        f"- Requested continuation advances after init: `{int(continuation_steps)}`",
        f"- Actual runner `step_max`: `{int(step_max)}`",
        "- `step_max` counts accepted states including the 2-state initialization, so `step_max = continuation_steps + 2`.",
        "",
        "## Commands",
        "",
        "```bash",
        " ".join(commands["state"]),
        " ".join(commands["smoke_hypre"]),
        *[" ".join(commands["smoke_pmg"][level]) for level in smoke_pmgs],
        " ".join(commands["continuation"]),
        "```",
        "",
        "## Smoke Sweep",
        "",
        f"- State artifact for frozen probes: [`{state_out_dir / 'data' / 'run_info.json'}`]({state_run})",
        f"- Best smoke candidate by converged frozen setup+solve wall time: `{_hierarchy_label_for_spec(candidate_specs[best_level])}`",
        f"- Existing continuation artifact shown below still uses: `{_hierarchy_label_for_spec(candidate_specs[continuation_choice])}`",
        "",
        * _smoke_rows(
            smoke_hypre,
            smoke_pmgs,
            candidate_specs=candidate_specs,
            tolerance=1.0e-3,
            best_level=best_level,
        ),
        "",
        f"![Frozen smoke sweep]({plot_smoke})",
        "",
        "## Continuation Result",
        "",
        f"- Continuation artifact: [`{continuation['run_info_path']}`]({cont_run})",
        f"- History artifact: [`{continuation['npz_path']}`]({cont_npz})",
        f"- Final accepted states: `{int(continuation['final_accepted_states'])}`",
        f"- Accepted continuation advances: `{int(continuation['accepted_continuation_advances'])}`",
        f"- Final lambda: `{float(continuation['lambda_last']):.9f}`",
        f"- Final omega: `{float(continuation['omega_last']):.3f}`",
        "",
        * _timing_rows(continuation),
        "",
        "## Reference Curves Included",
        "",
    ]
    if reference_curves:
        for ref in reference_curves:
            ref_rel = _report_relpath(report_path, ref["path"])
            lines.append(f"- `{ref['label']}` from [`{ref['artifact_rel']}`]({ref_rel})")
    else:
        lines.append("- No additional reference curves were found.")
    lines.extend(
        [
            "",
            f"![Lambda-omega continuation]({plot_curve})",
            "",
            "## Accepted-Step Metrics",
            "",
            * _step_rows(continuation, limit=int(continuation_steps)),
            "",
            f"![Step metrics]({plot_steps})",
            "",
        "## Frozen PMG Timing Breakdown",
        "",
        f"- The breakdown below is for the best frozen smoke case (`{_hierarchy_label_for_spec(candidate_specs[best_level])}`), not the full nonlinear continuation.",
        "",
        * _breakdown_rows(smoke_pmgs[best_level]),
            "",
            f"![Total timing components]({plot_timing})",
            "",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P2(L5) mixed PMG-shell smoke sweep and step-10 continuation.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--state-mode", type=str, default="zero", choices=["zero", "hypre"])
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--continuation-steps", type=int, default=10)
    parser.add_argument("--state-step-max", type=int, default=1)
    parser.add_argument("--smoke-linear-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--smoke-linear-max-iter", type=int, default=100)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    out_root = _ensure_dir(args.out_root)
    plots_dir = _ensure_dir(out_root / "plots")
    state_out_dir = Path(args.state_dir).resolve() if args.state_dir is not None else (out_root / STATE_OUT_NAME)
    smoke_root = _ensure_dir(out_root / "smoke")
    continuation_out_dir = out_root / PMG_CONT_NAME
    step_max = int(args.continuation_steps) + 2
    candidate_specs = dict(COARSE_CANDIDATES)
    continuation_choice = str(CONTINUATION_CANDIDATE_KEY)

    commands: dict[str, object] = {"smoke_pmg": {}}
    if not args.skip_run:
        if not (state_out_dir / "data" / "run_info.json").exists():
            if str(args.state_mode).strip().lower() == "zero":
                commands["state"] = _write_zero_state_case(out_dir=state_out_dir)
            else:
                commands["state"] = _run_state_case(
                    out_dir=state_out_dir,
                    mpi_ranks=int(args.mpi_ranks),
                    step_max=int(args.state_step_max),
                )
        else:
            if str(args.state_mode).strip().lower() == "zero":
                commands["state"] = _zero_state_command_repr(out_dir=state_out_dir)
            else:
                commands["state"] = _state_case_command(
                    out_dir=state_out_dir,
                    mpi_ranks=int(args.mpi_ranks),
                    step_max=int(args.state_step_max),
                )

        state_npz = state_out_dir / "data" / "petsc_run.npz"
        state_run_info = state_out_dir / "data" / "run_info.json"
        hypre_smoke_out = smoke_root / HYPRE_SMOKE_NAME
        if not (hypre_smoke_out / "data" / "run_info.json").exists():
            commands["smoke_hypre"] = _run_frozen_probe(
                out_dir=hypre_smoke_out,
                state_npz=state_npz,
                state_run_info=state_run_info,
                pc_backend="hypre",
                linear_tolerance=float(args.smoke_linear_tolerance),
                linear_max_iter=int(args.smoke_linear_max_iter),
            )
        else:
            commands["smoke_hypre"] = _frozen_probe_command(
                out_dir=hypre_smoke_out,
                state_npz=state_npz,
                state_run_info=state_run_info,
                pc_backend="hypre",
                linear_tolerance=float(args.smoke_linear_tolerance),
                linear_max_iter=int(args.smoke_linear_max_iter),
            )

        for level, coarse_mesh in candidate_specs.items():
            out_dir = smoke_root / f"pmg_shell_{level}"
            if not (out_dir / "data" / "run_info.json").exists():
                commands["smoke_pmg"][level] = _run_frozen_probe(
                    out_dir=out_dir,
                    state_npz=state_npz,
                    state_run_info=state_run_info,
                    pc_backend="pmg_shell",
                    coarse_meshes=coarse_mesh,
                    linear_tolerance=float(args.smoke_linear_tolerance),
                    linear_max_iter=int(args.smoke_linear_max_iter),
                )
            else:
                commands["smoke_pmg"][level] = _frozen_probe_command(
                    out_dir=out_dir,
                    state_npz=state_npz,
                    state_run_info=state_run_info,
                    pc_backend="pmg_shell",
                    coarse_meshes=coarse_mesh,
                    linear_tolerance=float(args.smoke_linear_tolerance),
                    linear_max_iter=int(args.smoke_linear_max_iter),
                )

    if not commands.get("state"):
        if str(args.state_mode).strip().lower() == "zero":
            commands["state"] = _zero_state_command_repr(out_dir=state_out_dir)
        else:
            commands["state"] = _state_case_command(
                out_dir=state_out_dir,
                mpi_ranks=int(args.mpi_ranks),
                step_max=int(args.state_step_max),
            )
    if not commands.get("smoke_hypre"):
        commands["smoke_hypre"] = _frozen_probe_command(
            out_dir=smoke_root / HYPRE_SMOKE_NAME,
            state_npz=state_out_dir / "data" / "petsc_run.npz",
            state_run_info=state_out_dir / "data" / "run_info.json",
            pc_backend="hypre",
            linear_tolerance=float(args.smoke_linear_tolerance),
            linear_max_iter=int(args.smoke_linear_max_iter),
        )
    for level in candidate_specs:
        commands["smoke_pmg"].setdefault(
            level,
            _frozen_probe_command(
                out_dir=smoke_root / f"pmg_shell_{level}",
                state_npz=state_out_dir / "data" / "petsc_run.npz",
                state_run_info=state_out_dir / "data" / "run_info.json",
                pc_backend="pmg_shell",
                coarse_meshes=candidate_specs[level],
                linear_tolerance=float(args.smoke_linear_tolerance),
                linear_max_iter=int(args.smoke_linear_max_iter),
            ),
        )

    smoke_hypre = _load_smoke_case(smoke_root / HYPRE_SMOKE_NAME / "data" / "run_info.json")
    smoke_pmgs = {
        level: _load_smoke_case(smoke_root / f"pmg_shell_{level}" / "data" / "run_info.json")
        for level in candidate_specs
        if (smoke_root / f"pmg_shell_{level}" / "data" / "run_info.json").exists()
    }
    best_level, _best_case = _choose_best_smoke(smoke_pmgs, tolerance=float(args.smoke_linear_tolerance))

    if not args.skip_run:
        if not (continuation_out_dir / "data" / "run_info.json").exists():
            commands["continuation"] = _run_continuation_case(
                out_dir=continuation_out_dir,
                mpi_ranks=int(args.mpi_ranks),
                step_max=int(step_max),
                coarse_mesh=_coarse_mesh_tuple(candidate_specs[continuation_choice])[0],
            )
        else:
            commands["continuation"] = _continuation_case_command(
                out_dir=continuation_out_dir,
                mpi_ranks=int(args.mpi_ranks),
                step_max=int(step_max),
                coarse_mesh=_coarse_mesh_tuple(candidate_specs[continuation_choice])[0],
            )
    if not commands.get("continuation"):
        commands["continuation"] = _continuation_case_command(
            out_dir=continuation_out_dir,
            mpi_ranks=int(args.mpi_ranks),
            step_max=int(step_max),
            coarse_mesh=_coarse_mesh_tuple(candidate_specs[continuation_choice])[0],
        )

    continuation = _load_continuation_case(continuation_out_dir)
    reference_curves = _load_reference_curves()

    plots = {
        "smoke": plots_dir / "smoke_sweep.png",
        "curve": plots_dir / "lambda_omega.png",
        "steps": plots_dir / "step_metrics.png",
        "timing": plots_dir / "timing_breakdown.png",
    }
    _plot_smoke(plots["smoke"], smoke_hypre=smoke_hypre, pmg_smokes=smoke_pmgs)
    _plot_lambda_omega(plots["curve"], continuation=continuation, refs=reference_curves)
    _plot_step_metrics(plots["steps"], continuation=continuation)
    _plot_timing_breakdown(plots["timing"], continuation=continuation)

    _write_summary(
        out_root / "summary.json",
        state_out_dir=state_out_dir,
        smoke_hypre=smoke_hypre,
        smoke_pmgs=smoke_pmgs,
        candidate_specs=candidate_specs,
        best_level=best_level,
        continuation_choice=continuation_choice,
        continuation=continuation,
        continuation_steps=int(args.continuation_steps),
        step_max=int(step_max),
        mpi_ranks=int(args.mpi_ranks),
        commands=commands,
        reference_curves=reference_curves,
    )
    _write_report(
        args.report_path,
        out_root=out_root,
        plots=plots,
        state_out_dir=state_out_dir,
        smoke_hypre=smoke_hypre,
        smoke_pmgs=smoke_pmgs,
        candidate_specs=candidate_specs,
        best_level=best_level,
        continuation_choice=continuation_choice,
        continuation=continuation,
        continuation_steps=int(args.continuation_steps),
        step_max=int(step_max),
        mpi_ranks=int(args.mpi_ranks),
        commands=commands,
        reference_curves=reference_curves,
    )


if __name__ == "__main__":
    main()
