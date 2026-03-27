from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np
import psutil


SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR if SCRIPT_DIR.name == "archive" else SCRIPT_DIR / "archive"
BENCHMARK_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "archive" else SCRIPT_DIR
ROOT = BENCHMARK_DIR.parents[1]
DEFAULT_MESH = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "p4_preconditioner_compare"
DEFAULT_BASELINE_SUMMARY = ROOT / "artifacts" / "p2_p4_compare_rank8_final_memfix" / "summary.json"
DEFAULT_BASELINE_REPORT = REPORT_DIR / "report_p2_vs_p4_rank8_final_memfix.md"
DEFAULT_RECYCLE_FAILURE_REPORT = REPORT_DIR / "report_p4_rank8_recycle_guard80_failed.md"
DEFAULT_FINAL_REPORT = REPORT_DIR / "report_p4_preconditioner_full_trajectory.md"
DEFAULT_STEP2_REPORT = REPORT_DIR / "report_p4_preconditioner_step2.md"
DEFAULT_BDDC_GATE_REPORT = REPORT_DIR / "report_p4_bddc_gate.md"
DEFAULT_BDDC_SHORT_REPORT = REPORT_DIR / "report_bddc_short_runs.md"
DEFAULT_BDDC_FULL_REPORT = REPORT_DIR / "report_bddc_full_trajectory.md"
DEFAULT_BDDC_SWEEP_REPORT = REPORT_DIR / "report_bddc_param_sweep_v2.md"
DEFAULT_SCREEN_RANKS = (1, 8)
DEFAULT_SCALE_RANKS = (1, 2, 4, 8)
DEFAULT_STEP2 = 2
DEFAULT_FULL_STEP_MAX = 100
DEFAULT_STAGES = ("smoke", "screen", "bddc_gate", "full_compare")


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    category: str
    cli_args: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mesh_label(mesh_path: Path) -> str:
    mesh_path = Path(mesh_path)
    if not mesh_path.is_absolute():
        return str(mesh_path)
    try:
        return str(mesh_path.relative_to(ROOT))
    except ValueError:
        return str(mesh_path)


def _path_label(path: Path | str) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _variant_registry(*, include_nongalerkin: bool) -> dict[str, Variant]:
    complexity_args = [
        "--pc_backend",
        "hypre",
        "--preconditioner_matrix_policy",
        "lagged",
        "--preconditioner_rebuild_policy",
        "accepted_step",
        "--pc_hypre_coarsen_type",
        "PMIS",
        "--pc_hypre_interp_type",
        "ext+i-mm",
        "--pc_hypre_P_max",
        "4",
        "--pc_hypre_agg_nl",
        "1",
    ]
    if include_nongalerkin:
        complexity_args.extend(["--pc_hypre_nongalerkin_tol", "0.01"])

    bddc_elastic_common = (
        "--outer_solver_family",
        "native_petsc",
        "--native_ksp_type",
        "cg",
        "--solver_type",
        "PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE",
        "--pc_backend",
        "bddc",
        "--preconditioner_matrix_source",
        "elastic",
        "--preconditioner_matrix_policy",
        "current",
        "--preconditioner_rebuild_policy",
        "every_newton",
        "--pc_bddc_symmetric",
        "--pc_bddc_use_vertices",
        "--pc_bddc_use_edges",
        "--pc_bddc_use_faces",
        "--no-pc_bddc_use_change_of_basis",
        "--no-pc_bddc_use_change_on_faces",
    )

    return {
        "hypre_current": Variant(
            name="hypre_current",
            description="Current Hypre baseline: HMIS + ext+i, current P",
            category="aij",
            cli_args=(
                "--solver_type",
                "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
                "--pc_backend",
                "hypre",
                "--preconditioner_matrix_policy",
                "current",
                "--preconditioner_rebuild_policy",
                "every_newton",
                "--pc_hypre_coarsen_type",
                "HMIS",
                "--pc_hypre_interp_type",
                "ext+i",
            ),
        ),
        "hypre_lagged_current": Variant(
            name="hypre_lagged_current",
            description="Hypre lagged P with current HMIS + ext+i",
            category="aij",
            cli_args=(
                "--solver_type",
                "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
                "--pc_backend",
                "hypre",
                "--preconditioner_matrix_policy",
                "lagged",
                "--preconditioner_rebuild_policy",
                "accepted_step",
                "--pc_hypre_coarsen_type",
                "HMIS",
                "--pc_hypre_interp_type",
                "ext+i",
            ),
        ),
        "hypre_lagged_pmis": Variant(
            name="hypre_lagged_pmis",
            description="Hypre lagged P with PMIS + ext+i-mm",
            category="aij",
            cli_args=(
                "--solver_type",
                "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
                "--pc_backend",
                "hypre",
                "--preconditioner_matrix_policy",
                "lagged",
                "--preconditioner_rebuild_policy",
                "accepted_step",
                "--pc_hypre_coarsen_type",
                "PMIS",
                "--pc_hypre_interp_type",
                "ext+i-mm",
            ),
        ),
        "hypre_lagged_complexity": Variant(
            name="hypre_lagged_complexity",
            description="Hypre lagged P with PMIS + ext+i-mm and lower hierarchy complexity",
            category="aij",
            cli_args=tuple(
                [
                    "--solver_type",
                    "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
                    *complexity_args,
                ]
            ),
        ),
        "gamg_lagged_lowmem": Variant(
            name="gamg_lagged_lowmem",
            description="GAMG lagged P with aggressive low-memory options",
            category="aij",
            cli_args=(
                "--solver_type",
                "PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE",
                "--pc_backend",
                "gamg",
                "--preconditioner_matrix_policy",
                "lagged",
                "--preconditioner_rebuild_policy",
                "accepted_step",
                "--pc_gamg_threshold",
                "0.02",
                "--pc_gamg_aggressive_coarsening",
                "1",
                "--no-pc_gamg_aggressive_square_graph",
                "--pc_gamg_aggressive_mis_k",
                "2",
            ),
        ),
        "bddc": Variant(
            name="bddc",
            description="MATIS + PCBDDC prototype",
            category="bddc",
            cli_args=(
                "--solver_type",
                "PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE",
                "--pc_backend",
                "bddc",
                "--preconditioner_matrix_policy",
                "current",
                "--preconditioner_rebuild_policy",
                "every_newton",
                "--no-pc_bddc_symmetric",
            ),
        ),
        "bddc_exact_current": Variant(
            name="bddc_exact_current",
            description="BDDC with exact local solvers",
            category="bddc",
            cli_args=(
                "--solver_type",
                "PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE",
                "--pc_backend",
                "bddc",
                "--preconditioner_matrix_policy",
                "current",
                "--preconditioner_rebuild_policy",
                "every_newton",
                "--no-pc_bddc_symmetric",
                "--pc_bddc_dirichlet_ksp_type",
                "preonly",
                "--pc_bddc_dirichlet_pc_type",
                "lu",
                "--pc_bddc_neumann_ksp_type",
                "preonly",
                "--pc_bddc_neumann_pc_type",
                "lu",
                "--pc_bddc_coarse_ksp_type",
                "preonly",
                "--pc_bddc_coarse_pc_type",
                "lu",
            ),
        ),
        "bddc_exact_elastic": Variant(
            name="bddc_exact_elastic",
            description="Elastic-first BDDC control with exact local solvers",
            category="bddc_elastic",
            cli_args=(
                *bddc_elastic_common,
                "--pc_bddc_dirichlet_ksp_type",
                "preonly",
                "--pc_bddc_dirichlet_pc_type",
                "lu",
                "--pc_bddc_neumann_ksp_type",
                "preonly",
                "--pc_bddc_neumann_pc_type",
                "lu",
                "--pc_bddc_coarse_ksp_type",
                "preonly",
                "--pc_bddc_coarse_pc_type",
                "lu",
                "--no-pc_bddc_use_deluxe_scaling",
            ),
        ),
        "bddc_gamg_elastic": Variant(
            name="bddc_gamg_elastic",
            description="Elastic-first BDDC with approximate local GAMG and LU coarse solve",
            category="bddc_elastic",
            cli_args=(
                *bddc_elastic_common,
                "--pc_bddc_dirichlet_ksp_type",
                "preonly",
                "--pc_bddc_dirichlet_pc_type",
                "gamg",
                "--pc_bddc_neumann_ksp_type",
                "preonly",
                "--pc_bddc_neumann_pc_type",
                "gamg",
                "--pc_bddc_coarse_ksp_type",
                "preonly",
                "--pc_bddc_coarse_pc_type",
                "lu",
                "--pc_bddc_dirichlet_approximate",
                "--pc_bddc_neumann_approximate",
                "--pc_bddc_switch_static",
                "--no-pc_bddc_use_deluxe_scaling",
            ),
        ),
        "bddc_local_ilu": Variant(
            name="bddc_local_ilu",
            description="BDDC with ILU local solvers and LU coarse solve",
            category="bddc",
            cli_args=(
                "--solver_type",
                "PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE",
                "--pc_backend",
                "bddc",
                "--preconditioner_matrix_policy",
                "current",
                "--preconditioner_rebuild_policy",
                "every_newton",
                "--no-pc_bddc_symmetric",
                "--pc_bddc_dirichlet_ksp_type",
                "preonly",
                "--pc_bddc_dirichlet_pc_type",
                "ilu",
                "--pc_bddc_neumann_ksp_type",
                "preonly",
                "--pc_bddc_neumann_pc_type",
                "ilu",
                "--pc_bddc_coarse_ksp_type",
                "preonly",
                "--pc_bddc_coarse_pc_type",
                "lu",
            ),
        ),
        "bddc_ilu_elastic": Variant(
            name="bddc_ilu_elastic",
            description="Elastic-first BDDC with ILU local solvers and LU coarse solve",
            category="bddc_elastic",
            cli_args=(
                *bddc_elastic_common,
                "--pc_bddc_dirichlet_ksp_type",
                "preonly",
                "--pc_bddc_dirichlet_pc_type",
                "ilu",
                "--pc_bddc_neumann_ksp_type",
                "preonly",
                "--pc_bddc_neumann_pc_type",
                "ilu",
                "--pc_bddc_coarse_ksp_type",
                "preonly",
                "--pc_bddc_coarse_pc_type",
                "lu",
                "--pc_bddc_dirichlet_approximate",
                "--pc_bddc_neumann_approximate",
                "--no-pc_bddc_use_deluxe_scaling",
            ),
        ),
        "bddc_ilu_elastic_deluxe": Variant(
            name="bddc_ilu_elastic_deluxe",
            description="Elastic-first BDDC with ILU local solvers and deluxe scaling",
            category="bddc_elastic",
            cli_args=(
                *bddc_elastic_common,
                "--pc_bddc_dirichlet_ksp_type",
                "preonly",
                "--pc_bddc_dirichlet_pc_type",
                "ilu",
                "--pc_bddc_neumann_ksp_type",
                "preonly",
                "--pc_bddc_neumann_pc_type",
                "ilu",
                "--pc_bddc_coarse_ksp_type",
                "preonly",
                "--pc_bddc_coarse_pc_type",
                "lu",
                "--pc_bddc_dirichlet_approximate",
                "--pc_bddc_neumann_approximate",
                "--pc_bddc_use_deluxe_scaling",
            ),
        ),
        "bddc_local_ilu_lowsetup": Variant(
            name="bddc_local_ilu_lowsetup",
            description="BDDC with ILU local solves and deluxe scaling disabled",
            category="bddc",
            cli_args=(
                "--solver_type",
                "PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE",
                "--pc_backend",
                "bddc",
                "--preconditioner_matrix_policy",
                "current",
                "--preconditioner_rebuild_policy",
                "every_newton",
                "--no-pc_bddc_symmetric",
                "--pc_bddc_dirichlet_ksp_type",
                "preonly",
                "--pc_bddc_dirichlet_pc_type",
                "ilu",
                "--pc_bddc_neumann_ksp_type",
                "preonly",
                "--pc_bddc_neumann_pc_type",
                "ilu",
                "--pc_bddc_coarse_ksp_type",
                "preonly",
                "--pc_bddc_coarse_pc_type",
                "lu",
                "--no-pc_bddc_use_deluxe_scaling",
            ),
        ),
        "bddc_local_ilu_coarse_gamg": Variant(
            name="bddc_local_ilu_coarse_gamg",
            description="BDDC with ILU local solves and GAMG coarse solve",
            category="bddc",
            cli_args=(
                "--solver_type",
                "PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE",
                "--pc_backend",
                "bddc",
                "--preconditioner_matrix_policy",
                "current",
                "--preconditioner_rebuild_policy",
                "every_newton",
                "--no-pc_bddc_symmetric",
                "--pc_bddc_dirichlet_ksp_type",
                "preonly",
                "--pc_bddc_dirichlet_pc_type",
                "ilu",
                "--pc_bddc_neumann_ksp_type",
                "preonly",
                "--pc_bddc_neumann_pc_type",
                "ilu",
                "--pc_bddc_coarse_ksp_type",
                "preonly",
                "--pc_bddc_coarse_pc_type",
                "gamg",
            ),
        ),
        "bddc_local_ilu_coarse_gamg_lowsetup": Variant(
            name="bddc_local_ilu_coarse_gamg_lowsetup",
            description="BDDC with ILU local solves, GAMG coarse solve, and deluxe scaling disabled",
            category="bddc",
            cli_args=(
                "--solver_type",
                "PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE",
                "--pc_backend",
                "bddc",
                "--preconditioner_matrix_policy",
                "current",
                "--preconditioner_rebuild_policy",
                "every_newton",
                "--no-pc_bddc_symmetric",
                "--pc_bddc_dirichlet_ksp_type",
                "preonly",
                "--pc_bddc_dirichlet_pc_type",
                "ilu",
                "--pc_bddc_neumann_ksp_type",
                "preonly",
                "--pc_bddc_neumann_pc_type",
                "ilu",
                "--pc_bddc_coarse_ksp_type",
                "preonly",
                "--pc_bddc_coarse_pc_type",
                "gamg",
                "--no-pc_bddc_use_deluxe_scaling",
            ),
        ),
    }


def _with_petsc_opts(base: tuple[str, ...], *entries: str) -> tuple[str, ...]:
    args = list(base)
    for entry in entries:
        args.extend(["--petsc-opt", str(entry)])
    return tuple(args)


def _without_flags(base: tuple[str, ...], *flags: str) -> tuple[str, ...]:
    remove = set(str(flag) for flag in flags)
    return tuple(token for token in base if token not in remove)


def _with_cli_pairs(base: tuple[str, ...], *pairs: str) -> tuple[str, ...]:
    args = list(base)
    args.extend(str(item) for item in pairs)
    return tuple(args)


def _bddc_sweep_registry(*, include_adaptive: bool) -> dict[str, Variant]:
    probe_common = (
        "--node-ordering",
        "block_metis",
        "--outer_solver_family",
        "native_petsc",
        "--native_ksp_type",
        "cg",
        "--native_ksp_norm_type",
        "unpreconditioned",
    )
    bddc_common = (
        *probe_common,
        "--pc_backend",
        "bddc",
        "--preconditioner_matrix_source",
        "elastic",
        "--pc_bddc_symmetric",
        "--pc_bddc_monolithic",
        "--pc_bddc_coarse_redundant_pc_type",
        "svd",
        "--pc_bddc_use_vertices",
        "--pc_bddc_use_edges",
        "--pc_bddc_use_faces",
        "--no-pc_bddc_use_change_of_basis",
        "--no-pc_bddc_use_change_on_faces",
        "--pc_bddc_check_level",
        "2",
    )
    bddc_doc_base = (
        *bddc_common,
        "--bddc_local_mat_type",
        "aij",
        "--pc_bddc_dirichlet_approximate",
        "--pc_bddc_neumann_approximate",
        "--pc_bddc_switch_static",
        "--pc_bddc_dirichlet_ksp_type",
        "preonly",
        "--pc_bddc_dirichlet_pc_type",
        "gamg",
        "--pc_bddc_neumann_ksp_type",
        "preonly",
        "--pc_bddc_neumann_pc_type",
        "gamg",
        "--pc_bddc_coarse_ksp_type",
        "preonly",
        "--pc_bddc_coarse_pc_type",
        "lu",
        "--no-pc_bddc_use_deluxe_scaling",
    )
    ex56_smooth = _with_petsc_opts(
        tuple(bddc_doc_base),
        "pc_bddc_dirichlet_pc_gamg_type=agg",
        "pc_bddc_neumann_pc_gamg_type=agg",
        "pc_bddc_dirichlet_pc_gamg_threshold=0.05",
        "pc_bddc_neumann_pc_gamg_threshold=0.05",
        "pc_bddc_dirichlet_pc_gamg_threshold_scale=0.0",
        "pc_bddc_neumann_pc_gamg_threshold_scale=0.0",
        "pc_bddc_dirichlet_pc_gamg_aggressive_coarsening=1",
        "pc_bddc_neumann_pc_gamg_aggressive_coarsening=1",
        "pc_bddc_dirichlet_pc_gamg_agg_nsmooths=1",
        "pc_bddc_neumann_pc_gamg_agg_nsmooths=1",
        "pc_bddc_dirichlet_pc_gamg_reuse_interpolation=true",
        "pc_bddc_neumann_pc_gamg_reuse_interpolation=true",
        "pc_bddc_dirichlet_pc_gamg_esteig_ksp_max_it=10",
        "pc_bddc_neumann_pc_gamg_esteig_ksp_max_it=10",
        "pc_bddc_dirichlet_mg_levels_ksp_type=chebyshev",
        "pc_bddc_neumann_mg_levels_ksp_type=chebyshev",
        "pc_bddc_dirichlet_mg_levels_ksp_max_it=1",
        "pc_bddc_neumann_mg_levels_ksp_max_it=1",
    )
    ex56_deluxe = tuple(
        "--pc_bddc_use_deluxe_scaling" if item == "--no-pc_bddc_use_deluxe_scaling" else item
        for item in ex56_smooth
    )
    registry = {
        "hypre_control_v2": Variant(
            name="hypre_control_v2",
            description="Native PETSc CG with Hypre HMIS + ext+i control",
            category="sweep_control",
            cli_args=(
                *probe_common,
                "--pc_backend",
                "hypre",
                "--pc_hypre_coarsen_type",
                "HMIS",
                "--pc_hypre_interp_type",
                "ext+i",
            ),
        ),
        "bddc_exact_lu_ref_v2": Variant(
            name="bddc_exact_lu_ref_v2",
            description="Exact BDDC reference with monolithic correction, coarse redundant SVD, and local LU",
            category="sweep_reference",
            cli_args=(
                *bddc_common,
                "--bddc_local_mat_type",
                "sbaij",
                "--pc_bddc_dirichlet_ksp_type",
                "preonly",
                "--pc_bddc_dirichlet_pc_type",
                "lu",
                "--pc_bddc_neumann_ksp_type",
                "preonly",
                "--pc_bddc_neumann_pc_type",
                "lu",
                "--pc_bddc_coarse_ksp_type",
                "preonly",
                "--pc_bddc_coarse_pc_type",
                "lu",
                "--no-pc_bddc_use_deluxe_scaling",
            ),
        ),
        "bddc_gamg_doc_base_v2": Variant(
            name="bddc_gamg_doc_base_v2",
            description="Approximate local BDDC with monolithic correction, coarse redundant SVD, and local GAMG",
            category="sweep_bddc",
            cli_args=tuple(bddc_doc_base),
        ),
        "bddc_gamg_ex56_v2": Variant(
            name="bddc_gamg_ex56_v2",
            description="Doc-base BDDC plus ex56 local GAMG tuning",
            category="sweep_bddc",
            cli_args=ex56_smooth,
        ),
        "bddc_gamg_ex56_deluxe_v2": Variant(
            name="bddc_gamg_ex56_deluxe_v2",
            description="ex56-tuned BDDC plus deluxe scaling",
            category="sweep_bddc",
            cli_args=ex56_deluxe,
        ),
    }
    if include_adaptive:
        registry["bddc_gamg_ex71_adaptive2_v2"] = Variant(
            name="bddc_gamg_ex71_adaptive2_v2",
            description="ex56 + deluxe + adaptive threshold 2.0 + schur_layers 1",
            category="sweep_bddc",
            cli_args=_with_petsc_opts(
                ex56_deluxe,
                "pc_bddc_adaptive_threshold=2.0",
                "pc_bddc_schur_layers=1",
            ),
        )
    return registry


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


def _load_memory_guard_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    path = path.resolve()
    peak_rss_gib = 0.0
    min_available_gib = np.inf
    sample_count = 0
    guard_triggered = False
    guard_limit_gib = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        event = json.loads(line)
        sample_count += 1
        if str(event.get("event", "")) == "guard_triggered":
            guard_triggered = True
            if event.get("guard_limit_gib") is not None:
                guard_limit_gib = float(event["guard_limit_gib"])
        if "rss_gib" in event:
            peak_rss_gib = max(peak_rss_gib, float(event["rss_gib"]))
        if "mem_available_gib" in event:
            min_available_gib = min(min_available_gib, float(event["mem_available_gib"]))

    if not np.isfinite(min_available_gib):
        min_available_gib = 0.0

    summary = {
        "path": _path_label(path),
        "peak_rss_gib": float(peak_rss_gib),
        "min_available_gib": float(min_available_gib),
        "sample_count": int(sample_count),
        "guard_triggered": bool(guard_triggered),
    }
    if guard_limit_gib is not None:
        summary["guard_limit_gib"] = float(guard_limit_gib)
    return summary


def _load_case_metrics(
    out_dir: Path,
    *,
    memory_summary: dict[str, Any] | None = None,
    startup_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_info = json.loads((out_dir / "data" / "run_info.json").read_text(encoding="utf-8"))
    progress = _load_progress_summary(out_dir)
    with np.load(out_dir / "data" / "petsc_run.npz", allow_pickle=True) as npz:
        lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
        omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
        umax_hist = np.asarray(npz["Umax_hist"], dtype=np.float64)
        step_idx = np.asarray(npz.get("stats_step_index", np.empty(0)), dtype=np.int64)
        step_linear_iterations = np.asarray(npz.get("stats_step_linear_iterations", np.empty(0)), dtype=np.int64)
        step_newton_iterations = np.asarray(npz.get("stats_step_newton_iterations", np.empty(0)), dtype=np.int64)

    info = run_info["run_info"]
    params = run_info["params"]
    timings = run_info["timings"]
    linear = timings["linear"]
    constitutive = timings["constitutive"]
    final_accepted_states = int(progress["final_accepted_states"] or info["step_count"])
    init_accepted_states = int(progress["init_accepted_states"])
    owned_pattern = run_info.get("owned_tangent_pattern") or {}
    owned_stats = dict(owned_pattern.get("stats_max", {}) or {})
    bddc_pattern = run_info.get("bddc_subdomain_pattern") or {}
    bddc_stats = dict(bddc_pattern.get("stats_max", {}) or {})
    metrics = {
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
        "step_linear_iterations_total": int(step_linear_iterations.sum()) if step_linear_iterations.size else 0,
        "step_newton_iterations_total": int(step_newton_iterations.sum()) if step_newton_iterations.size else 0,
        "init_linear_iterations": int(linear["init_linear_iterations"]),
        "init_linear_solve_time": float(linear["init_linear_solve_time"]),
        "init_linear_preconditioner_time": float(linear["init_linear_preconditioner_time"]),
        "attempt_linear_iterations_total": int(linear["attempt_linear_iterations_total"]),
        "attempt_linear_solve_time_total": float(linear["attempt_linear_solve_time_total"]),
        "attempt_linear_preconditioner_time_total": float(linear["attempt_linear_preconditioner_time_total"]),
        "attempt_linear_orthogonalization_time_total": float(linear["attempt_linear_orthogonalization_time_total"]),
        "preconditioner_setup_time_total": float(linear.get("preconditioner_setup_time_total", 0.0)),
        "preconditioner_apply_time_total": float(linear.get("preconditioner_apply_time_total", 0.0)),
        "preconditioner_rebuild_count": int(linear.get("preconditioner_rebuild_count", 0)),
        "preconditioner_reuse_count": int(linear.get("preconditioner_reuse_count", 0)),
        "preconditioner_age_max": int(linear.get("preconditioner_age_max", 0)),
        "pc_backend": str(linear.get("pc_backend", params.get("pc_backend"))),
        "preconditioner_matrix_policy": str(linear.get("preconditioner_matrix_policy", params.get("preconditioner_matrix_policy", "current"))),
        "preconditioner_rebuild_policy": str(linear.get("preconditioner_rebuild_policy", params.get("preconditioner_rebuild_policy", "every_newton"))),
        "preconditioner_rebuild_interval": int(linear.get("preconditioner_rebuild_interval", params.get("preconditioner_rebuild_interval", 1))),
        "build_tangent_local": float(constitutive["build_tangent_local"]),
        "build_F": float(constitutive["build_F"]),
        "local_strain": float(constitutive["local_strain"]),
        "local_constitutive": float(constitutive["local_constitutive"]),
        "local_constitutive_comm": float(constitutive.get("local_constitutive_comm", 0.0)),
        "owned_tangent_pattern_stats_max": owned_stats,
        "bddc_subdomain_pattern_stats_max": bddc_stats,
        "bddc_explicit_primal_vertices": bool(float(bddc_stats.get("explicit_primal_vertices_used", 0.0))),
        "bddc_local_primal_vertices_count": int(float(bddc_stats.get("local_primal_vertices_count", 0.0))),
        "bddc_local_total_bytes": float(bddc_stats.get("local_total_bytes", 0.0)),
        "progress_file_created": bool((startup_summary or {}).get("progress_file_created", True)),
        "first_progress_elapsed_s": (
            None
            if (startup_summary or {}).get("first_progress_elapsed_s") is None
            else float((startup_summary or {})["first_progress_elapsed_s"])
        ),
        "startup_stall_reason": (startup_summary or {}).get("startup_stall_reason"),
    }
    metrics["linear_total_rank_metric"] = (
        float(metrics["attempt_linear_preconditioner_time_total"]) + float(metrics["attempt_linear_solve_time_total"])
    )
    if memory_summary is not None:
        metrics["memory_guard"] = dict(memory_summary)
        metrics["peak_rss_gib"] = float(memory_summary.get("peak_rss_gib", 0.0))
    else:
        metrics["peak_rss_gib"] = 0.0
    return metrics


def _load_probe_metrics(
    out_dir: Path,
    *,
    memory_summary: dict[str, Any] | None = None,
    startup_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = json.loads((out_dir / "data" / "run_info.json").read_text(encoding="utf-8"))
    stdout_path = out_dir.parent / f"{out_dir.name}.stdout.log"
    stderr_path = out_dir.parent / f"{out_dir.name}.stderr.log"
    petsc_log_path = out_dir.parent / f"{out_dir.name}.petsc.log"
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    log_text = ""
    if petsc_log_path.exists():
        log_text = petsc_log_path.read_text(encoding="utf-8", errors="replace")
    else:
        log_text = stdout_text
    coarse_info = _parse_bddc_coarse_info("\n".join([stdout_text, stderr_text]))
    petsc_events = _parse_petsc_log_events(log_text)
    metrics = {
        "runtime_seconds": float(data["runtime_seconds"]),
        "setup_elapsed_s": float(data["setup_elapsed_s"]),
        "solve_times_s": [float(v) for v in data.get("solve_times_s", [])],
        "solve_time": float(sum(float(v) for v in data.get("solve_times_s", []))),
        "iteration_counts": [int(v) for v in data.get("iteration_counts", [])],
        "iteration_count": int(data.get("iteration_counts", [0])[-1]) if data.get("iteration_counts") else 0,
        "converged_reasons": [int(v) for v in data.get("converged_reasons", [])],
        "converged_reason": int(data.get("converged_reasons", [0])[-1]) if data.get("converged_reasons") else 0,
        "final_relative_residual": (
            float(data["final_relative_residual"])
            if data.get("final_relative_residual") is not None
            else (
                float(data.get("relative_residual_norms", [0.0])[-1])
                if data.get("relative_residual_norms")
                else 0.0
            )
        ),
        "reported_residual_history": [float(v) for v in data.get("relative_reported_residual_history", [])],
        "relative_residual_history": [float(v) for v in data.get("relative_reported_residual_history", [])],
        "reported_residual_histories": [
            [float(v) for v in hist] for hist in data.get("relative_reported_residual_histories", [])
        ],
        "pc_backend": str(data.get("pc_backend", "")),
        "outer_solver_family": str(data.get("outer_solver_family", "")),
        "native_ksp_type": data.get("native_ksp_type"),
        "native_ksp_norm_type": data.get("native_ksp_norm_type"),
        "preconditioner_matrix_source": str(data.get("preconditioner_matrix_source", "")),
        "pmat_type": data.get("pmat_type"),
        "pmat_block_size": data.get("pmat_block_size"),
        "local_pmat_type": data.get("local_pmat_type"),
        "local_pmat_block_size": data.get("local_pmat_block_size"),
        "matis_local_mat_type": data.get("matis_local_mat_type"),
        "bddc_local_vertex_major_ordering": bool(data.get("bddc_local_vertex_major_ordering", False)),
        "bddc_local_block_size": data.get("bddc_local_block_size"),
        "bddc_adjacency_source": data.get("bddc_adjacency_source", data.get("adjacency_source")),
        "linear_total_rank_metric": float(data.get("linear_total_rank_metric", 0.0)),
        "attempt_linear_preconditioner_time_total": float(data.get("attempt_linear_preconditioner_time_total", 0.0)),
        "attempt_linear_solve_time_total": float(data.get("attempt_linear_solve_time_total", 0.0)),
        "preconditioner_rebuild_count": int(data.get("preconditioner_rebuild_count", 0)),
        "preconditioner_reuse_count": int(data.get("preconditioner_reuse_count", 0)),
        "preconditioner_age_max": int(data.get("preconditioner_age_max", 0)),
        "progress_file_created": bool((startup_summary or {}).get("progress_file_created", True)),
        "first_progress_elapsed_s": (
            None
            if (startup_summary or {}).get("first_progress_elapsed_s") is None
            else float((startup_summary or {})["first_progress_elapsed_s"])
        ),
        "startup_stall_reason": (startup_summary or {}).get("startup_stall_reason"),
        "bddc_local_total_bytes": float(data.get("bddc_local_total_bytes", 0.0)),
        "bddc_local_primal_vertices_count": int(float(data.get("bddc_local_primal_vertices_count", 0.0))),
        "bddc_candidate_vertices": coarse_info.get("candidate_vertices"),
        "bddc_candidate_edges": coarse_info.get("candidate_edges"),
        "bddc_candidate_faces": coarse_info.get("candidate_faces"),
        "bddc_coarse_size": coarse_info.get("coarse_size"),
        "petsc_bddc_events": petsc_events,
        "petsc_log": (_path_label(petsc_log_path) if petsc_log_path.exists() else None),
    }
    if memory_summary is not None:
        metrics["memory_guard"] = dict(memory_summary)
        metrics["peak_rss_gib"] = float(memory_summary.get("peak_rss_gib", 0.0))
    else:
        metrics["peak_rss_gib"] = 0.0
    return metrics


def _parse_bddc_coarse_info(text: str) -> dict[str, int | None]:
    lowered = str(text)
    patterns = {
        "candidate_vertices": [
            r"got\s+(\d+)\s+local candidate vertices",
            r"candidate vertices[^0-9]*(\d+)",
            r"vertices candidates[^0-9]*(\d+)",
        ],
        "candidate_edges": [
            r"got\s+(\d+)\s+local candidate edges",
            r"candidate edges[^0-9]*(\d+)",
            r"edges candidates[^0-9]*(\d+)",
        ],
        "candidate_faces": [
            r"got\s+(\d+)\s+local candidate faces",
            r"candidate faces[^0-9]*(\d+)",
            r"faces candidates[^0-9]*(\d+)",
        ],
        "coarse_size": [
            r"size of coarse problem is\s+(\d+)",
            r"coarse (?:problem )?(?:dimension|size)[^0-9]*(\d+)",
            r"coarse dofs[^0-9]*(\d+)",
        ],
    }
    parsed: dict[str, int | None] = {}
    for key, variants in patterns.items():
        parsed[key] = None
        for pattern in variants:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if match:
                parsed[key] = int(match.group(1))
                break
    return parsed


def _parse_petsc_log_events(text: str) -> dict[str, float]:
    events = {}
    for event_name in (
        "PC_BDDC_Topology",
        "PC_BDDC_LocalSolvers",
        "PC_BDDC_CorrectionSetUp",
        "PC_BDDC_CoarseSetUp",
        "PC_BDDC_ApproxSetUp",
    ):
        for line in str(text).splitlines():
            if event_name not in line:
                continue
            numbers = re.findall(r"[-+]?(?:\\d+\\.\\d*|\\.\\d+|\\d+)(?:[Ee][-+]?\\d+)?", line)
            if numbers:
                events[event_name] = float(numbers[-1])
                break
    return events


def _classify_probe_failure(*, startup_summary: dict[str, Any], stderr_path: Path) -> str:
    if startup_summary.get("event") == "startup_stall":
        return "startup_stall"
    excerpt = _tail_text(stderr_path).lower()
    if "unknown option" in excerpt or "unused database option" in excerpt:
        return "unsupported_option"
    if startup_summary:
        return str(startup_summary.get("event", "runtime_failure"))
    return "runtime_failure"


def _tail_text(path: Path, *, n_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-int(n_lines):]).strip()


def _sum_process_tree_rss_gib(proc: psutil.Process) -> float:
    rss = 0
    processes = [proc]
    try:
        processes.extend(proc.children(recursive=True))
    except Exception:
        pass
    seen: set[int] = set()
    for entry in processes:
        try:
            if entry.pid in seen:
                continue
            seen.add(entry.pid)
            rss += int(entry.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return float(rss) / float(1024 ** 3)


def _terminate_process_tree(proc: subprocess.Popen[Any]) -> None:
    try:
        root = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return
    children = []
    try:
        children = root.children(recursive=True)
    except Exception:
        children = []
    for child in reversed(children):
        try:
            child.send_signal(signal.SIGTERM)
        except Exception:
            continue
    try:
        root.send_signal(signal.SIGTERM)
    except Exception:
        pass
    _, alive = psutil.wait_procs(children + [root], timeout=5.0)
    for child in alive:
        try:
            child.kill()
        except Exception:
            continue


def _select_backtrace_pid(proc: psutil.Process) -> int | None:
    candidates: list[tuple[int, int]] = []
    try:
        if proc.is_running():
            candidates.append((int(proc.memory_info().rss), int(proc.pid)))
    except Exception:
        pass
    try:
        for child in proc.children(recursive=True):
            try:
                candidates.append((int(child.memory_info().rss), int(child.pid)))
            except Exception:
                continue
    except Exception:
        pass
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def _capture_gdb_backtrace(*, pid: int, out_path: Path) -> Path | None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            ["gdb", "-batch", "-ex", "thread apply all bt", "-p", str(int(pid))],
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except Exception as exc:
        out_path.write_text(f"failed to capture gdb backtrace for pid {int(pid)}: {exc}\n", encoding="utf-8")
        return out_path

    content = proc.stdout
    if proc.stderr:
        content = f"{content}\n[stderr]\n{proc.stderr}"
    if not content.strip():
        content = f"gdb produced no output for pid {int(pid)}\n"
    out_path.write_text(content, encoding="utf-8")
    return out_path


def _run_monitored_command(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    memory_log_path: Path | None,
    guard_limit_gib: float | None,
    wall_time_limit_s: float | None = None,
    sample_interval_s: float = 5.0,
    progress_path: Path | None = None,
    startup_progress_timeout_s: float | None = None,
    startup_backtrace_path: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    if memory_log_path is not None:
        memory_log_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        ps_proc = psutil.Process(proc.pid)
        start = time.monotonic()
        run_summary: dict[str, Any] = {
            "progress_file_created": False,
            "first_progress_elapsed_s": None,
            "startup_stall_reason": None,
        }
        memory_log_handle = None
        if memory_log_path is not None:
            memory_log_handle = memory_log_path.open("w", encoding="utf-8")
        try:
            while True:
                return_code = proc.poll()
                rss_gib = _sum_process_tree_rss_gib(ps_proc)
                mem_available_gib = float(psutil.virtual_memory().available) / float(1024 ** 3)
                elapsed_s = float(time.monotonic() - start)
                if progress_path is not None and not bool(run_summary["progress_file_created"]) and progress_path.exists():
                    run_summary["progress_file_created"] = True
                    run_summary["first_progress_elapsed_s"] = elapsed_s
                if memory_log_handle is not None:
                    memory_log_handle.write(
                        json.dumps(
                            {
                                "timestamp": _utc_now(),
                                "event": "sample",
                                "elapsed_s": elapsed_s,
                                "rss_gib": rss_gib,
                                "mem_available_gib": mem_available_gib,
                            }
                        )
                        + "\n"
                    )
                    memory_log_handle.flush()
                if guard_limit_gib is not None and rss_gib > float(guard_limit_gib):
                    event = {
                        "timestamp": _utc_now(),
                        "event": "guard_triggered",
                        "elapsed_s": elapsed_s,
                        "rss_gib": rss_gib,
                        "mem_available_gib": mem_available_gib,
                        "guard_limit_gib": float(guard_limit_gib),
                    }
                    if memory_log_handle is not None:
                        memory_log_handle.write(json.dumps(event) + "\n")
                        memory_log_handle.flush()
                    _terminate_process_tree(proc)
                    proc.wait(timeout=20.0)
                    return_code = proc.returncode if proc.returncode is not None else 1
                    run_summary.update(event)
                    return int(return_code), run_summary
                if (
                    startup_progress_timeout_s is not None
                    and not bool(run_summary["progress_file_created"])
                    and elapsed_s > float(startup_progress_timeout_s)
                ):
                    backtrace_pid = _select_backtrace_pid(ps_proc)
                    backtrace_label = None
                    if startup_backtrace_path is not None and backtrace_pid is not None:
                        backtrace_path = _capture_gdb_backtrace(pid=backtrace_pid, out_path=startup_backtrace_path)
                        if backtrace_path is not None:
                            backtrace_label = _path_label(backtrace_path)
                    event = {
                        "timestamp": _utc_now(),
                        "event": "startup_stall",
                        "elapsed_s": elapsed_s,
                        "rss_gib": rss_gib,
                        "mem_available_gib": mem_available_gib,
                        "startup_progress_timeout_s": float(startup_progress_timeout_s),
                        "backtrace_pid": backtrace_pid,
                    }
                    if backtrace_label is not None:
                        event["backtrace_path"] = backtrace_label
                    if memory_log_handle is not None:
                        memory_log_handle.write(json.dumps(event) + "\n")
                        memory_log_handle.flush()
                    _terminate_process_tree(proc)
                    proc.wait(timeout=20.0)
                    run_summary.update(event)
                    run_summary["startup_stall_reason"] = "progress_timeout"
                    return_code = proc.returncode if proc.returncode is not None else 1
                    return int(return_code), run_summary
                if wall_time_limit_s is not None and elapsed_s > float(wall_time_limit_s):
                    event = {
                        "timestamp": _utc_now(),
                        "event": "runtime_cutoff",
                        "elapsed_s": elapsed_s,
                        "rss_gib": rss_gib,
                        "mem_available_gib": mem_available_gib,
                        "wall_time_limit_s": float(wall_time_limit_s),
                    }
                    if memory_log_handle is not None:
                        memory_log_handle.write(json.dumps(event) + "\n")
                        memory_log_handle.flush()
                    _terminate_process_tree(proc)
                    proc.wait(timeout=20.0)
                    return_code = proc.returncode if proc.returncode is not None else 1
                    run_summary.update(event)
                    return int(return_code), run_summary
                if return_code is not None:
                    if not bool(run_summary["progress_file_created"]):
                        run_summary["startup_stall_reason"] = "exited_before_progress"
                    return int(return_code), run_summary
                time.sleep(sample_interval_s)
        finally:
            if memory_log_handle is not None:
                memory_log_handle.close()


def _run_case(
    *,
    variant: Variant,
    ranks: int,
    mesh_path: Path,
    step_max: int,
    out_dir: Path,
    guard_limit_gib: float | None,
    max_deflation_basis_vectors: int,
    elem_type: str = "P4",
    wall_time_limit_s: float | None = None,
    reuse_existing: bool,
    startup_progress_timeout_s: float | None = None,
) -> dict[str, Any]:
    if reuse_existing and (out_dir / "data" / "run_info.json").exists():
        memory_summary = _load_memory_guard_summary(out_dir.parent / f"{out_dir.name}.memory_guard.jsonl")
        startup_path = out_dir / "startup_summary.json"
        startup_summary = json.loads(startup_path.read_text(encoding="utf-8")) if startup_path.exists() else None
        return {
            "status": "completed",
            "out_dir": _path_label(out_dir),
            "metrics": _load_case_metrics(out_dir, memory_summary=memory_summary, startup_summary=startup_summary),
            "startup_summary": startup_summary,
        }

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

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
        str(elem_type).upper(),
        "--step_max",
        str(int(step_max)),
        "--node_ordering",
        "block_metis",
        "--tangent_kernel",
        "rows",
        "--constitutive_mode",
        "overlap",
        "--no-recycle_preconditioner",
        "--max_deflation_basis_vectors",
        str(int(max_deflation_basis_vectors)),
        "--no-store_step_u",
        "--out_dir",
        str(out_dir),
        *variant.cli_args,
    ]

    stdout_path = out_dir.parent / f"{out_dir.name}.stdout.log"
    stderr_path = out_dir.parent / f"{out_dir.name}.stderr.log"
    memory_log_path = out_dir.parent / f"{out_dir.name}.memory_guard.jsonl"
    progress_path = out_dir / "data" / "progress.jsonl"
    startup_summary_path = out_dir / "startup_summary.json"
    startup_backtrace_path = out_dir.parent / f"{out_dir.name}.startup_stall.bt.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    return_code, startup_summary = _run_monitored_command(
        cmd=cmd,
        cwd=ROOT,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        memory_log_path=memory_log_path,
        guard_limit_gib=guard_limit_gib,
        wall_time_limit_s=wall_time_limit_s,
        progress_path=progress_path,
        startup_progress_timeout_s=startup_progress_timeout_s,
        startup_backtrace_path=startup_backtrace_path,
    )
    startup_summary_path.write_text(json.dumps(startup_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    memory_summary = _load_memory_guard_summary(memory_log_path)
    if return_code != 0 or not (out_dir / "data" / "run_info.json").exists():
        status = "failed"
        if startup_summary:
            status = str(startup_summary.get("event", status))
        failure_excerpt = _tail_text(stderr_path)
        result = {
            "status": status,
            "out_dir": _path_label(out_dir),
            "stdout_log": _path_label(stdout_path),
            "stderr_log": _path_label(stderr_path),
            "memory_guard": memory_summary,
            "failure_excerpt": failure_excerpt,
            "startup_summary": startup_summary,
        }
        if startup_summary.get("backtrace_path"):
            result["startup_backtrace"] = startup_summary["backtrace_path"]
        return result

    return {
        "status": "completed",
        "out_dir": _path_label(out_dir),
        "stdout_log": _path_label(stdout_path),
        "stderr_log": _path_label(stderr_path),
        "memory_guard": memory_summary,
        "startup_summary": startup_summary,
        "metrics": _load_case_metrics(out_dir, memory_summary=memory_summary, startup_summary=startup_summary),
    }


def _run_probe_case(
    *,
    variant: Variant,
    ranks: int,
    mesh_path: Path,
    out_dir: Path,
    elem_type: str,
    mode: str,
    linear_tolerance: float,
    linear_max_iter: int,
    guard_limit_gib: float | None,
    wall_time_limit_s: float | None,
    reuse_existing: bool,
    startup_progress_timeout_s: float | None = None,
    extra_petsc_opts: tuple[str, ...] = (),
) -> dict[str, Any]:
    stdout_path = out_dir.parent / f"{out_dir.name}.stdout.log"
    stderr_path = out_dir.parent / f"{out_dir.name}.stderr.log"
    memory_log_path = out_dir.parent / f"{out_dir.name}.memory_guard.jsonl"
    startup_summary_path = out_dir / "startup_summary.json"
    petsc_log_path = out_dir.parent / f"{out_dir.name}.petsc.log"

    if reuse_existing and (out_dir / "data" / "run_info.json").exists():
        memory_summary = _load_memory_guard_summary(out_dir.parent / f"{out_dir.name}.memory_guard.jsonl")
        startup_summary = json.loads(startup_summary_path.read_text(encoding="utf-8")) if startup_summary_path.exists() else None
        return {
            "status": "completed",
            "out_dir": _path_label(out_dir),
            "metrics": _load_probe_metrics(out_dir, memory_summary=memory_summary, startup_summary=startup_summary),
            "startup_summary": startup_summary,
        }
    if reuse_existing and startup_summary_path.exists():
        startup_summary = (
            json.loads(startup_summary_path.read_text(encoding="utf-8"))
            if startup_summary_path.exists()
            else {}
        )
        memory_summary = _load_memory_guard_summary(memory_log_path)
        result = {
            "status": _classify_probe_failure(startup_summary=startup_summary, stderr_path=stderr_path),
            "out_dir": _path_label(out_dir),
            "stdout_log": _path_label(stdout_path),
            "stderr_log": _path_label(stderr_path),
            "memory_guard": memory_summary,
            "failure_excerpt": _tail_text(stderr_path),
            "startup_summary": startup_summary,
        }
        if startup_summary.get("backtrace_path"):
            result["startup_backtrace"] = startup_summary["backtrace_path"]
        return result

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    cmd = [
        "mpiexec",
        "-n",
        str(int(ranks)),
        sys.executable,
        str(SCRIPT_DIR / "probe_bddc_elastic.py"),
        "--out-dir",
        str(out_dir),
        "--mesh-path",
        str(mesh_path),
        "--elem-type",
        str(elem_type).upper(),
        "--mode",
        str(mode),
        "--linear_tolerance",
        str(float(linear_tolerance)),
        "--linear_max_iter",
        str(int(linear_max_iter)),
        *variant.cli_args,
    ]
    dynamic_petsc_opts = [
        f"log_view=:{petsc_log_path}:ascii_info_detail",
        "log_view_memory=true",
        *[str(v) for v in extra_petsc_opts],
    ]
    for entry in dynamic_petsc_opts:
        cmd.extend(["--petsc-opt", entry])

    progress_path = out_dir / "data" / "progress.jsonl"
    startup_backtrace_path = out_dir.parent / f"{out_dir.name}.startup_stall.bt.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    return_code, startup_summary = _run_monitored_command(
        cmd=cmd,
        cwd=ROOT,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        memory_log_path=memory_log_path,
        guard_limit_gib=guard_limit_gib,
        wall_time_limit_s=wall_time_limit_s,
        progress_path=progress_path,
        startup_progress_timeout_s=startup_progress_timeout_s,
        startup_backtrace_path=startup_backtrace_path,
    )
    startup_summary_path.write_text(json.dumps(startup_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    memory_summary = _load_memory_guard_summary(memory_log_path)
    if return_code != 0 or not (out_dir / "data" / "run_info.json").exists():
        status = _classify_probe_failure(startup_summary=startup_summary, stderr_path=stderr_path)
        result = {
            "status": status,
            "out_dir": _path_label(out_dir),
            "stdout_log": _path_label(stdout_path),
            "stderr_log": _path_label(stderr_path),
            "memory_guard": memory_summary,
            "failure_excerpt": _tail_text(stderr_path),
            "startup_summary": startup_summary,
        }
        if startup_summary.get("backtrace_path"):
            result["startup_backtrace"] = startup_summary["backtrace_path"]
        return result

    return {
        "status": "completed",
        "out_dir": _path_label(out_dir),
        "stdout_log": _path_label(stdout_path),
        "stderr_log": _path_label(stderr_path),
        "memory_guard": memory_summary,
        "startup_summary": startup_summary,
        "metrics": _load_probe_metrics(out_dir, memory_summary=memory_summary, startup_summary=startup_summary),
    }


def _screen_variant(
    *,
    variant: Variant,
    ranks: tuple[int, ...],
    mesh_path: Path,
    step_max: int,
    out_root: Path,
    max_deflation_basis_vectors: int,
    wall_time_limits_s: dict[int, float] | None,
    reuse_existing: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for ranks_value in ranks:
        out_dir = out_root / "screen" / variant.name / f"rank{int(ranks_value)}_step{int(step_max)}"
        results[str(int(ranks_value))] = _run_case(
            variant=variant,
            ranks=int(ranks_value),
            mesh_path=mesh_path,
            step_max=step_max,
            out_dir=out_dir,
            guard_limit_gib=None,
            max_deflation_basis_vectors=max_deflation_basis_vectors,
            wall_time_limit_s=(None if wall_time_limits_s is None else wall_time_limits_s.get(int(ranks_value))),
            reuse_existing=reuse_existing,
        )
    return results


def _run_option_smoke(
    *,
    variant: Variant,
    mesh_path: Path,
    out_root: Path,
    max_deflation_basis_vectors: int,
    smoke_elem_type: str,
    reuse_existing: bool,
) -> dict[str, Any]:
    out_dir = out_root / "option_smokes" / variant.name / "rank1_step1"
    result = _run_case(
        variant=variant,
        ranks=1,
        mesh_path=mesh_path,
        step_max=1,
        out_dir=out_dir,
        guard_limit_gib=None,
        max_deflation_basis_vectors=max_deflation_basis_vectors,
        elem_type=smoke_elem_type,
        reuse_existing=reuse_existing,
    )
    return result


def _run_bddc_runtime_smokes(
    *,
    variant: Variant,
    mesh_path: Path,
    out_root: Path,
    max_deflation_basis_vectors: int,
    reuse_existing: bool,
) -> dict[str, Any]:
    cases = (
        ("rank1_p2_step1", 1, "P2"),
        ("rank2_p2_step1", 2, "P2"),
        ("rank2_p4_step1", 2, "P4"),
    )
    runs: dict[str, Any] = {}
    for name, ranks, elem_type in cases:
        out_dir = out_root / "bddc_runtime_smokes" / name
        runs[name] = _run_case(
            variant=variant,
            ranks=int(ranks),
            mesh_path=mesh_path,
            step_max=1,
            out_dir=out_dir,
            guard_limit_gib=None,
            max_deflation_basis_vectors=max_deflation_basis_vectors,
            elem_type=elem_type,
            reuse_existing=reuse_existing,
        )
    return {
        "status": "completed" if all(run.get("status") == "completed" for run in runs.values()) else "failed",
        "runs": runs,
    }


def _promote_aij_candidates(screening: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    baseline = screening["hypre_current"]
    if baseline["status"] != "completed":
        raise RuntimeError("The hypre_current baseline screen did not complete successfully.")
    base_rank1 = baseline["runs"]["1"]["metrics"]
    base_rank8 = baseline["runs"]["8"]["metrics"]
    drift_limits = {
        "lambda": abs(float(base_rank8["lambda_last"]) - float(base_rank1["lambda_last"])),
        "omega": abs(float(base_rank8["omega_last"]) - float(base_rank1["omega_last"])),
        "umax": abs(float(base_rank8["umax_last"]) - float(base_rank1["umax_last"])),
    }
    baseline_peak = float(base_rank8.get("peak_rss_gib", 0.0))

    scored: list[tuple[float, str]] = []
    for name, entry in screening.items():
        if name == "hypre_current" or entry.get("variant_category") != "aij":
            continue
        rank1 = entry["runs"].get("1")
        rank8 = entry["runs"].get("8")
        if rank1 is None or rank8 is None or rank1.get("status") != "completed" or rank8.get("status") != "completed":
            continue
        m1 = rank1["metrics"]
        m8 = rank8["metrics"]
        if int(m8["final_accepted_states"]) != int(base_rank8["final_accepted_states"]):
            continue
        if int(m8["successful_attempt_count"]) != int(base_rank8["successful_attempt_count"]):
            continue
        drift_lambda = abs(float(m8["lambda_last"]) - float(m1["lambda_last"]))
        drift_omega = abs(float(m8["omega_last"]) - float(m1["omega_last"]))
        drift_umax = abs(float(m8["umax_last"]) - float(m1["umax_last"]))
        if drift_lambda > drift_limits["lambda"] + 1.0e-10:
            continue
        if drift_omega > drift_limits["omega"] + 1.0e-8:
            continue
        if drift_umax > drift_limits["umax"] + 1.0e-8:
            continue
        peak_rss_gib = float(m8.get("peak_rss_gib", 0.0))
        if baseline_peak > 0.0 and peak_rss_gib > 1.05 * baseline_peak:
            continue
        scored.append((float(m8["linear_total_rank_metric"]), name))

    scored.sort()
    promoted = [name for _score, name in scored[:2]]
    return promoted, drift_limits


def _run_scaling(
    *,
    variant_names: list[str],
    registry: dict[str, Variant],
    ranks: tuple[int, ...],
    mesh_path: Path,
    step_max: int,
    out_root: Path,
    max_deflation_basis_vectors: int,
    reuse_existing: bool,
) -> dict[str, Any]:
    scaling: dict[str, Any] = {}
    for name in variant_names:
        variant = registry[name]
        per_rank: dict[str, Any] = {}
        for ranks_value in ranks:
            out_dir = out_root / "scaling" / variant.name / f"rank{int(ranks_value)}_step{int(step_max)}"
            per_rank[str(int(ranks_value))] = _run_case(
                variant=variant,
                ranks=int(ranks_value),
                mesh_path=mesh_path,
                step_max=step_max,
                out_dir=out_dir,
                guard_limit_gib=None,
                max_deflation_basis_vectors=max_deflation_basis_vectors,
                reuse_existing=reuse_existing,
            )
        scaling[name] = {
            "variant_category": variant.category,
            "description": variant.description,
            "runs": per_rank,
        }
    return scaling


def _select_best_scaled_candidate(scaling: dict[str, Any]) -> str:
    completed = []
    for name, entry in scaling.items():
        rank8 = entry["runs"].get("8")
        if rank8 is None or rank8.get("status") != "completed":
            continue
        completed.append((float(rank8["metrics"]["linear_total_rank_metric"]), name))
    if not completed:
        raise RuntimeError("No scaled step-2 candidate completed successfully at rank 8.")
    completed.sort()
    return completed[0][1]


def _baseline_full_metrics(summary_path: Path, *, elem_type: str = "P4") -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    elem_key = str(elem_type).upper()
    baseline = dict(payload[elem_key])
    memory_guard = payload.get(f"{elem_key}_memory_guard")
    if memory_guard is not None:
        baseline["memory_guard"] = dict(memory_guard)
        baseline["peak_rss_gib"] = float(memory_guard.get("peak_rss_gib", 0.0))
    else:
        baseline["peak_rss_gib"] = 0.0
    baseline["preconditioner_time_total"] = float(baseline.get("init_linear_preconditioner_time", 0.0)) + float(
        baseline.get("attempt_linear_preconditioner_time_total", 0.0)
    )
    baseline["linear_solve_time_total"] = float(baseline.get("init_linear_solve_time", 0.0)) + float(
        baseline.get("attempt_linear_solve_time_total", 0.0)
    )
    return baseline


def _full_run_totals(metrics: dict[str, Any]) -> tuple[float, float]:
    preconditioner_total = float(metrics.get("init_linear_preconditioner_time", 0.0)) + float(
        metrics.get("attempt_linear_preconditioner_time_total", 0.0)
    )
    linear_total = float(metrics.get("init_linear_solve_time", 0.0)) + float(
        metrics.get("attempt_linear_solve_time_total", 0.0)
    )
    return preconditioner_total, linear_total


def _entry_metrics(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not entry:
        return None
    return entry.get("metrics")


def _full_compare_entry(
    *,
    name: str,
    description: str,
    run: dict[str, Any] | None,
    reused_baseline: bool = False,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "description": description,
        "status": "missing" if run is None else str(run.get("status", "missing")),
        "reused_baseline": bool(reused_baseline),
    }
    if run is not None:
        payload.update({key: value for key, value in run.items() if key != "name" and key != "description"})
    return payload


def _format_metric(entry: dict[str, Any] | None, key: str, fmt: str) -> str:
    metrics = _entry_metrics(entry)
    if not metrics or key not in metrics:
        return "-"
    value = metrics[key]
    return format(float(value), fmt) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)


def _run_full_compare(
    *,
    registry: dict[str, Variant],
    best_aij_name: str,
    bddc_gate: dict[str, Any],
    mesh_path: Path,
    out_root: Path,
    full_step_max: int,
    full_memory_guard_gib: float,
    max_deflation_basis_vectors: int,
    reuse_existing: bool,
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    baseline_entry = _full_compare_entry(
        name="hypre_current",
        description=registry["hypre_current"].description,
        run={"status": "reused_baseline", "metrics": dict(baseline_metrics)},
        reused_baseline=True,
    )

    if best_aij_name == "hypre_current":
        best_aij_entry = _full_compare_entry(
            name=best_aij_name,
            description=registry[best_aij_name].description,
            run={"status": "reused_baseline", "metrics": dict(baseline_metrics)},
            reused_baseline=True,
        )
    else:
        best_aij_run = _run_case(
            variant=registry[best_aij_name],
            ranks=8,
            mesh_path=mesh_path,
            step_max=int(full_step_max),
            out_dir=out_root / "full_compare" / f"{best_aij_name}_rank8_step{int(full_step_max)}",
            guard_limit_gib=float(full_memory_guard_gib),
            max_deflation_basis_vectors=max_deflation_basis_vectors,
            reuse_existing=reuse_existing,
        )
        best_aij_entry = _full_compare_entry(
            name=best_aij_name,
            description=registry[best_aij_name].description,
            run=best_aij_run,
        )

    if bddc_gate.get("eligible_for_full_trajectory", False):
        bddc_run = _run_case(
            variant=registry["bddc"],
            ranks=8,
            mesh_path=mesh_path,
            step_max=int(full_step_max),
            out_dir=out_root / "full_compare" / f"bddc_rank8_step{int(full_step_max)}",
            guard_limit_gib=float(full_memory_guard_gib),
            max_deflation_basis_vectors=max_deflation_basis_vectors,
            reuse_existing=reuse_existing,
        )
        bddc_entry = _full_compare_entry(
            name="bddc",
            description=registry["bddc"].description,
            run=bddc_run,
        )
    else:
        bddc_entry = _full_compare_entry(
            name="bddc",
            description=registry["bddc"].description,
            run={
                "status": "not_run_gate_failed",
                "reason": "BDDC did not satisfy the rank-8 step-2 gate.",
            },
        )

    return {
        "baseline": baseline_entry,
        "best_aij": best_aij_entry,
        "bddc": bddc_entry,
    }


BDDCCandidateNames = (
    "bddc_exact_current",
    "bddc_local_ilu",
    "bddc_local_ilu_lowsetup",
    "bddc_local_ilu_coarse_gamg",
    "bddc_local_ilu_coarse_gamg_lowsetup",
)

BDDCElasticCandidateNames = (
    "bddc_exact_elastic",
    "bddc_gamg_elastic",
)

BDDCSweepCandidateNames = (
    "bddc_gamg_doc_base_v2",
    "bddc_gamg_ex56_v2",
    "bddc_gamg_ex56_deluxe_v2",
    "bddc_gamg_ex71_adaptive2_v2",
)


def _bddc_runtime_metric(metrics: dict[str, Any]) -> float:
    return float(metrics.get("attempt_linear_preconditioner_time_total", 0.0)) + float(
        metrics.get("attempt_linear_solve_time_total", 0.0)
    )


def _short_run_acceptance(
    *,
    candidate: dict[str, Any],
    control: dict[str, Any],
    runtime_factor_limit: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate.get("status") != "completed":
        reasons.append(f"status={candidate.get('status', 'missing')}")
        return False, reasons
    if control.get("status") != "completed":
        reasons.append("control_failed")
        return False, reasons

    metrics = candidate["metrics"]
    control_metrics = control["metrics"]
    if not bool(metrics.get("progress_file_created", False)):
        reasons.append("no_progress_file")
    if metrics.get("first_progress_elapsed_s") is None:
        reasons.append("missing_first_progress_time")
    if int(metrics["final_accepted_states"]) != int(control_metrics["final_accepted_states"]):
        reasons.append("accepted_state_mismatch")
    for key in ("lambda_last", "omega_last", "umax_last"):
        if not np.isclose(float(metrics[key]), float(control_metrics[key]), rtol=1.0e-7, atol=1.0e-7):
            reasons.append(f"{key}_drift")
    if float(metrics["runtime_seconds"]) > float(runtime_factor_limit) * float(control_metrics["runtime_seconds"]):
        reasons.append("runtime_limit")
    return len(reasons) == 0, reasons


def _best_bddc_candidate(p2_short: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    control = p2_short.get("hypre_current")
    candidates = p2_short.get("bddc_candidates", {})
    evaluations: dict[str, Any] = {}
    scored: list[tuple[float, str]] = []
    for name, run in candidates.items():
        accepted, reasons = _short_run_acceptance(candidate=run, control=control, runtime_factor_limit=2.0)
        evaluations[name] = {"accepted": bool(accepted), "reasons": reasons}
        if accepted:
            scored.append((_bddc_runtime_metric(run["metrics"]), name))
    scored.sort()
    return (scored[0][1] if scored else None), evaluations


def _p4_gate_acceptance(
    *,
    candidate: dict[str, Any],
    control: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate.get("status") != "completed":
        reasons.append(f"status={candidate.get('status', 'missing')}")
        return False, reasons
    if control.get("status") != "completed":
        reasons.append("control_failed")
        return False, reasons
    metrics = candidate["metrics"]
    control_metrics = control["metrics"]
    if not bool(metrics.get("progress_file_created", False)):
        reasons.append("no_progress_file")
    if metrics.get("first_progress_elapsed_s") is None:
        reasons.append("missing_first_progress_time")
    if int(metrics["final_accepted_states"]) != int(control_metrics["final_accepted_states"]):
        reasons.append("accepted_state_mismatch")
    for key in ("lambda_last", "omega_last", "umax_last"):
        if not np.isclose(float(metrics[key]), float(control_metrics[key]), rtol=1.0e-7, atol=1.0e-7):
            reasons.append(f"{key}_drift")
    if float(metrics["runtime_seconds"]) > 1.5 * float(control_metrics["runtime_seconds"]):
        reasons.append("runtime_limit")
    control_peak = float(control_metrics.get("peak_rss_gib", 0.0))
    if control_peak > 0.0 and float(metrics.get("peak_rss_gib", 0.0)) > 1.10 * control_peak:
        reasons.append("peak_rss_limit")
    return len(reasons) == 0, reasons


def _run_bddc_prototype_workflow(
    *,
    registry: dict[str, Variant],
    mesh_path: Path,
    out_root: Path,
    short_step_max: int,
    full_step_max: int,
    max_deflation_basis_vectors: int,
    full_memory_guard_gib: float,
    reuse_existing: bool,
    startup_progress_timeout_p2_s: float,
    startup_progress_timeout_p4_s: float,
    requested_variants: tuple[str, ...] | None,
) -> dict[str, Any]:
    if requested_variants is None:
        requested_variants = BDDCCandidateNames
    enabled_candidates = [name for name in BDDCCandidateNames if name in requested_variants]
    if not enabled_candidates:
        raise RuntimeError("No BDDC candidates selected for the prototype workflow.")

    summary: dict[str, Any] = {
        "startup_progress_timeout_p2_s": float(startup_progress_timeout_p2_s),
        "startup_progress_timeout_p4_s": float(startup_progress_timeout_p4_s),
        "p2_smokes": {},
        "p2_short": {},
        "p2_full": {"status": "not_run"},
        "p4_smokes": {},
        "p4_short": {},
        "p4_full": {"status": "not_run"},
        "winner": None,
        "winner_evaluation": {},
        "p4_gate": {"eligible": False, "reasons": []},
    }

    p2_smokes: dict[str, Any] = {}
    if "bddc_exact_current" in enabled_candidates:
        p2_smokes["rank1_p2_step1_exact"] = _run_case(
            variant=registry["bddc_exact_current"],
            ranks=1,
            mesh_path=mesh_path,
            step_max=1,
            out_dir=out_root / "p2_smokes" / "rank1_p2_step1_exact",
            guard_limit_gib=None,
            max_deflation_basis_vectors=max_deflation_basis_vectors,
            elem_type="P2",
            reuse_existing=reuse_existing,
            startup_progress_timeout_s=float(startup_progress_timeout_p2_s),
        )
    if "bddc_local_ilu_coarse_gamg" in enabled_candidates:
        p2_smokes["rank1_p2_step1_coarse_gamg"] = _run_case(
            variant=registry["bddc_local_ilu_coarse_gamg"],
            ranks=1,
            mesh_path=mesh_path,
            step_max=1,
            out_dir=out_root / "p2_smokes" / "rank1_p2_step1_coarse_gamg",
            guard_limit_gib=None,
            max_deflation_basis_vectors=max_deflation_basis_vectors,
            elem_type="P2",
            reuse_existing=reuse_existing,
            startup_progress_timeout_s=float(startup_progress_timeout_p2_s),
        )
        if p2_smokes["rank1_p2_step1_coarse_gamg"].get("status") != "completed":
            enabled_candidates = [name for name in enabled_candidates if name != "bddc_local_ilu_coarse_gamg"]
    summary["p2_smokes"] = p2_smokes

    p2_step1_candidates: dict[str, Any] = {}
    p2_step10_candidates: dict[str, Any] = {}
    for name in enabled_candidates:
        p2_step1_candidates[name] = _run_case(
            variant=registry[name],
            ranks=2,
            mesh_path=mesh_path,
            step_max=1,
            out_dir=out_root / "p2_step1" / name,
            guard_limit_gib=None,
            max_deflation_basis_vectors=max_deflation_basis_vectors,
            elem_type="P2",
            reuse_existing=reuse_existing,
            startup_progress_timeout_s=float(startup_progress_timeout_p2_s),
        )
        if p2_step1_candidates[name].get("status") == "completed":
            p2_step10_candidates[name] = _run_case(
                variant=registry[name],
                ranks=2,
                mesh_path=mesh_path,
                step_max=int(short_step_max),
                out_dir=out_root / "p2_step10" / name,
                guard_limit_gib=None,
                max_deflation_basis_vectors=max_deflation_basis_vectors,
                elem_type="P2",
                reuse_existing=reuse_existing,
                startup_progress_timeout_s=float(startup_progress_timeout_p2_s),
            )
        else:
            p2_step10_candidates[name] = {
                "status": "skipped_after_step1_failure",
                "reason": f"rank-2 P2 step_max=1 failed with status {p2_step1_candidates[name].get('status', 'missing')}",
            }
    p2_hypre_current = _run_case(
        variant=registry["hypre_current"],
        ranks=2,
        mesh_path=mesh_path,
        step_max=int(short_step_max),
        out_dir=out_root / "p2_step10" / "hypre_current",
        guard_limit_gib=None,
        max_deflation_basis_vectors=max_deflation_basis_vectors,
        elem_type="P2",
        reuse_existing=reuse_existing,
        startup_progress_timeout_s=float(startup_progress_timeout_p2_s),
    )
    summary["p2_short"] = {
        "step1_candidates": p2_step1_candidates,
        "bddc_candidates": p2_step10_candidates,
        "hypre_current": p2_hypre_current,
    }

    winner, winner_eval = _best_bddc_candidate(summary["p2_short"])
    summary["winner"] = winner
    summary["winner_evaluation"] = winner_eval
    if winner is None:
        return summary

    summary["p2_full"] = _run_case(
        variant=registry[winner],
        ranks=8,
        mesh_path=mesh_path,
        step_max=int(full_step_max),
        out_dir=out_root / "p2_full" / winner,
        guard_limit_gib=float(full_memory_guard_gib),
        max_deflation_basis_vectors=max_deflation_basis_vectors,
        elem_type="P2",
        reuse_existing=reuse_existing,
        startup_progress_timeout_s=float(startup_progress_timeout_p2_s),
    )

    summary["p4_smokes"]["rank2_p4_step1"] = _run_case(
        variant=registry[winner],
        ranks=2,
        mesh_path=mesh_path,
        step_max=1,
        out_dir=out_root / "p4_smokes" / winner,
        guard_limit_gib=None,
        max_deflation_basis_vectors=max_deflation_basis_vectors,
        elem_type="P4",
        reuse_existing=reuse_existing,
        startup_progress_timeout_s=float(startup_progress_timeout_p4_s),
    )
    if summary["p4_smokes"]["rank2_p4_step1"].get("status") != "completed":
        summary["p4_gate"] = {
            "eligible": False,
            "reasons": [
                f"rank2_step1_failed:{summary['p4_smokes']['rank2_p4_step1'].get('status', 'missing')}"
            ],
        }
        return summary
    summary["p4_short"] = {
        "bddc": _run_case(
            variant=registry[winner],
            ranks=8,
            mesh_path=mesh_path,
            step_max=int(short_step_max),
            out_dir=out_root / "p4_step10" / winner,
            guard_limit_gib=None,
            max_deflation_basis_vectors=max_deflation_basis_vectors,
            elem_type="P4",
            reuse_existing=reuse_existing,
            startup_progress_timeout_s=float(startup_progress_timeout_p4_s),
        ),
        "hypre_current": _run_case(
            variant=registry["hypre_current"],
            ranks=8,
            mesh_path=mesh_path,
            step_max=int(short_step_max),
            out_dir=out_root / "p4_step10" / "hypre_current",
            guard_limit_gib=None,
            max_deflation_basis_vectors=max_deflation_basis_vectors,
            elem_type="P4",
            reuse_existing=reuse_existing,
            startup_progress_timeout_s=float(startup_progress_timeout_p4_s),
        ),
    }
    gate_ok, gate_reasons = _p4_gate_acceptance(candidate=summary["p4_short"]["bddc"], control=summary["p4_short"]["hypre_current"])
    summary["p4_gate"] = {"eligible": bool(gate_ok), "reasons": gate_reasons}
    if not gate_ok:
        return summary

    summary["p4_full"] = _run_case(
        variant=registry[winner],
        ranks=8,
        mesh_path=mesh_path,
        step_max=int(full_step_max),
        out_dir=out_root / "p4_full" / winner,
        guard_limit_gib=float(full_memory_guard_gib),
        max_deflation_basis_vectors=max_deflation_basis_vectors,
        elem_type="P4",
        reuse_existing=reuse_existing,
        startup_progress_timeout_s=float(startup_progress_timeout_p4_s),
    )
    return summary


def _petsc_has_external_package(name: str) -> bool:
    try:
        from petsc4py import PETSc  # type: ignore

        return bool(PETSc.Sys.hasExternalPackage(str(name)))
    except Exception:
        return False


def _plot_sweep_histories(
    *,
    plot_script: Path,
    out_path: Path,
    title: str,
    target: float,
    runs: list[tuple[str, Path]],
) -> Path:
    cmd = [
        sys.executable,
        str(plot_script),
        "--out",
        str(out_path),
        "--title",
        str(title),
        "--target",
        str(float(target)),
    ]
    for label, path in runs:
        cmd.extend(["--run", f"{label}={path}"])
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return out_path


def _promote_bddc_sweep_candidates(*, screening: dict[str, Any], control_name: str = "hypre_control_v2") -> list[str]:
    control = screening[control_name]
    if control.get("status") != "completed":
        return []
    control_metrics = control["metrics"]
    control_peak = float(control_metrics.get("peak_rss_gib", 0.0))
    scored: list[tuple[int, float, float, str]] = []
    for name, entry in screening.items():
        if entry.get("variant_category") != "sweep_bddc":
            continue
        if entry.get("status") != "completed":
            continue
        metrics = entry["metrics"]
        if int(metrics.get("converged_reason", 0)) <= 0:
            continue
        if float(metrics.get("final_relative_residual", np.inf)) > 1.0e-5:
            continue
        if control_peak > 0.0 and float(metrics.get("peak_rss_gib", 0.0)) > 1.5 * control_peak:
            continue
        if float(metrics.get("runtime_seconds", np.inf)) > 3.0 * float(control_metrics.get("runtime_seconds", np.inf)):
            continue
        scored.append(
            (
                int(metrics.get("iteration_count", 0)),
                float(metrics.get("solve_time", 0.0)),
                float(metrics.get("setup_elapsed_s", 0.0)),
                name,
            )
        )
    scored.sort()
    if not scored:
        return []
    promoted = [scored[0][3]]
    if len(scored) >= 2:
        promoted.append(scored[1][3])
    return promoted


def _nonlinear_cli_from_linear_probe_variant(variant: Variant) -> tuple[str, ...]:
    args = list(variant.cli_args)
    retained: list[str] = []
    i = 0
    flag_only = {
        "--pc_bddc_symmetric",
        "--pc_bddc_dirichlet_approximate",
        "--pc_bddc_neumann_approximate",
        "--pc_bddc_monolithic",
        "--pc_bddc_switch_static",
        "--pc_bddc_use_deluxe_scaling",
        "--pc_bddc_use_vertices",
        "--pc_bddc_use_edges",
        "--pc_bddc_use_faces",
        "--no-pc_bddc_use_change_of_basis",
        "--no-pc_bddc_use_change_on_faces",
    }
    one_value = {
        "--pc_bddc_check_level",
        "--pc_bddc_dirichlet_ksp_type",
        "--pc_bddc_dirichlet_pc_type",
        "--pc_bddc_neumann_ksp_type",
        "--pc_bddc_neumann_pc_type",
        "--pc_bddc_coarse_ksp_type",
        "--pc_bddc_coarse_pc_type",
        "--pc_bddc_coarse_redundant_pc_type",
        "--petsc-opt",
    }
    while i < len(args):
        token = args[i]
        if token in flag_only:
            retained.append(token)
            i += 1
            continue
        if token in one_value and i + 1 < len(args):
            retained.extend([token, args[i + 1]])
            i += 2
            continue
        i += 1
    return tuple(retained)


def _run_bddc_sweep_workflow(
    *,
    registry: dict[str, Variant],
    mesh_path: Path,
    out_root: Path,
    linear_tolerance: float,
    linear_max_iter: int,
    max_deflation_basis_vectors: int,
    reuse_existing: bool,
    requested_variants: tuple[str, ...] | None,
) -> dict[str, Any]:
    plot_script = SCRIPT_DIR / "plot_p4_linear_convergence.py"
    variant_lookup: dict[str, Variant] = dict(registry)
    summary: dict[str, Any] = {
        "linear_tolerance": float(linear_tolerance),
        "linear_max_iter": int(linear_max_iter),
        "mumps_available": bool(_petsc_has_external_package("mumps")),
        "phase0_note": "Supersedes bddc_param_sweep_v1 because v1 benchmarked BDDC without the corrected symmetric/monolithic/coarse-redundant baseline.",
        "option_smokes": {},
        "linear_screen": {},
        "approximate_side_sweep": {},
        "topology_sweep": {},
        "promoted_candidates": [],
        "diagnostic_linear": {},
        "nonlinear_short": {},
        "nonlinear_step10": {"status": "not_run"},
        "rank8_linear": {"status": "not_run"},
        "plots": {"overlays": {}, "aggregate": None},
    }

    enabled_names = list(BDDCSweepCandidateNames)
    if requested_variants is not None:
        enabled_names = [name for name in enabled_names if name in requested_variants]
    if not bool(summary["mumps_available"]):
        enabled_names = [name for name in enabled_names if name != "bddc_gamg_ex71_adaptive2_v2"]

    smoke_registry = ["hypre_control_v2", "bddc_exact_lu_ref_v2", *enabled_names]
    for name in smoke_registry:
        variant = registry[name]
        result = _run_probe_case(
            variant=variant,
            ranks=1,
            mesh_path=mesh_path,
            out_dir=out_root / "option_smokes" / name / "rank1_p4_single",
            elem_type="P4",
            mode="single_solve",
            linear_tolerance=float(linear_tolerance),
            linear_max_iter=int(linear_max_iter),
            guard_limit_gib=None,
            wall_time_limit_s=1200.0,
            reuse_existing=reuse_existing,
            startup_progress_timeout_s=180.0,
        )
        summary["option_smokes"][name] = {
            "variant_category": variant.category,
            "description": variant.description,
            **result,
        }
    enabled_names = [
        name
        for name in enabled_names
        if summary["option_smokes"].get(name, {}).get("status") == "completed"
    ]

    screen_names = ["hypre_control_v2"]
    if summary["option_smokes"].get("bddc_exact_lu_ref_v2", {}).get("status") == "completed":
        screen_names.append("bddc_exact_lu_ref_v2")
    screen_names.extend(enabled_names)
    for name in screen_names:
        variant = registry[name]
        result = _run_probe_case(
            variant=variant,
            ranks=2,
            mesh_path=mesh_path,
            out_dir=out_root / "linear_screen" / name / "rank2_p4_single",
            elem_type="P4",
            mode="single_solve",
            linear_tolerance=float(linear_tolerance),
            linear_max_iter=int(linear_max_iter),
            guard_limit_gib=None,
            wall_time_limit_s=3600.0,
            reuse_existing=reuse_existing,
            startup_progress_timeout_s=300.0,
        )
        summary["linear_screen"][name] = {
            "variant_category": variant.category,
            "description": variant.description,
            **result,
        }

    hypre_linear = summary["linear_screen"].get("hypre_control_v2")
    hypre_metrics = hypre_linear.get("metrics", {}) if hypre_linear and hypre_linear.get("status") == "completed" else {}
    if hypre_linear and hypre_linear.get("status") == "completed":
        hypre_run_info = out_root / "linear_screen" / "hypre_control_v2" / "rank2_p4_single" / "data" / "run_info.json"
        overlay_dir = out_root / "plots"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        aggregate_runs: list[tuple[str, Path]] = [("hypre", hypre_run_info)]
        for name in enabled_names:
            entry = summary["linear_screen"].get(name, {})
            if entry.get("status") != "completed":
                continue
            run_info_path = out_root / "linear_screen" / name / "rank2_p4_single" / "data" / "run_info.json"
            plot_path = overlay_dir / f"{name}_vs_hypre.png"
            _plot_sweep_histories(
                plot_script=plot_script,
                out_path=plot_path,
                title=f"P4 Linear Elastic Convergence: Hypre vs {name}",
                target=float(linear_tolerance),
                runs=[("hypre", hypre_run_info), (name, run_info_path)],
            )
            summary["plots"]["overlays"][name] = _path_label(plot_path)
            aggregate_runs.append((name, run_info_path))
        if len(aggregate_runs) >= 2:
            aggregate_path = overlay_dir / "p4_linear_top_candidates.png"
            _plot_sweep_histories(
                plot_script=plot_script,
                out_path=aggregate_path,
                title="P4 Linear Elastic Convergence: completed sweep candidates",
                target=float(linear_tolerance),
                runs=aggregate_runs[:4],
            )
            summary["plots"]["aggregate"] = _path_label(aggregate_path)

    phase1_nonadaptive = [
        name
        for name in enabled_names
        if name in summary["linear_screen"] and name != "bddc_gamg_ex71_adaptive2_v2"
    ]
    phase1_nonadaptive = [
        name for name in phase1_nonadaptive if summary["linear_screen"][name].get("status") == "completed"
    ]
    phase1_nonadaptive.sort(
        key=lambda name: (
            int(summary["linear_screen"][name]["metrics"].get("iteration_count", 10**9)),
            float(summary["linear_screen"][name]["metrics"].get("solve_time", np.inf)),
            float(summary["linear_screen"][name]["metrics"].get("setup_elapsed_s", np.inf)),
        )
    )
    best_nonadaptive_name = phase1_nonadaptive[0] if phase1_nonadaptive else None

    if best_nonadaptive_name is not None:
        base_variant = registry[best_nonadaptive_name]
        exact_dir_variant = Variant(
            name=f"{best_nonadaptive_name}_dir_exact_neu_approx",
            description=f"{best_nonadaptive_name} with exact Dirichlet and approximate Neumann",
            category="sweep_bddc",
            cli_args=_without_flags(base_variant.cli_args, "--pc_bddc_dirichlet_approximate"),
        )
        exact_neu_variant = Variant(
            name=f"{best_nonadaptive_name}_dir_approx_neu_exact",
            description=f"{best_nonadaptive_name} with approximate Dirichlet and exact Neumann",
            category="sweep_bddc",
            cli_args=_without_flags(base_variant.cli_args, "--pc_bddc_neumann_approximate"),
        )
        both_variant = Variant(
            name=f"{best_nonadaptive_name}_dir_approx_neu_approx",
            description=f"{best_nonadaptive_name} with approximate Dirichlet and approximate Neumann",
            category="sweep_bddc",
            cli_args=tuple(base_variant.cli_args),
        )
        for variant in (exact_dir_variant, exact_neu_variant, both_variant):
            variant_lookup[variant.name] = variant
            run = _run_probe_case(
                variant=variant,
                ranks=2,
                mesh_path=mesh_path,
                out_dir=out_root / "approximate_side_sweep" / variant.name / "rank2_p4_single",
                elem_type="P4",
                mode="single_solve",
                linear_tolerance=float(linear_tolerance),
                linear_max_iter=int(linear_max_iter),
                guard_limit_gib=None,
                wall_time_limit_s=3600.0,
                reuse_existing=reuse_existing,
                startup_progress_timeout_s=300.0,
            )
            summary["approximate_side_sweep"][variant.name] = {
                "variant_category": variant.category,
                "description": variant.description,
                **run,
            }

    combined_screening = dict(summary["linear_screen"])
    combined_screening.update(summary["approximate_side_sweep"])
    promoted = _promote_bddc_sweep_candidates(screening=combined_screening)

    best_candidate_name = promoted[0] if promoted else None
    best_candidate_metrics = combined_screening.get(best_candidate_name, {}).get("metrics", {}) if best_candidate_name else {}
    needs_topology = False
    if best_candidate_name and hypre_metrics:
        candidate_faces = best_candidate_metrics.get("bddc_candidate_faces")
        coarse_size = best_candidate_metrics.get("bddc_coarse_size")
        iteration_count = float(best_candidate_metrics.get("iteration_count", np.inf))
        hypre_iterations = max(float(hypre_metrics.get("iteration_count", 1.0)), 1.0)
        needs_topology = (
            candidate_faces in (None, 0)
            or coarse_size is None
            or int(coarse_size) < 24
            or iteration_count > 2.0 * hypre_iterations
        )

    if best_candidate_name and needs_topology:
        topology_base = variant_lookup.get(best_candidate_name)
        if topology_base is None and best_nonadaptive_name is not None:
            topology_base = registry[best_nonadaptive_name]
        if topology_base is not None:
            topology_variants = (
                Variant(
                    name=f"{topology_base.name}_adj_none",
                    description=f"{topology_base.name} with PETSc-derived adjacency",
                    category="sweep_bddc",
                    cli_args=_with_cli_pairs(_without_flags(topology_base.cli_args, "--adjacency_source", "csr"), "--adjacency_source", "none"),
                ),
                Variant(
                    name=f"{topology_base.name}_adj_csr",
                    description=f"{topology_base.name} with CSR-derived adjacency",
                    category="sweep_bddc",
                    cli_args=_with_cli_pairs(_without_flags(topology_base.cli_args, "--adjacency_source", "none", "topology"), "--adjacency_source", "csr"),
                ),
                Variant(
                    name=f"{topology_base.name}_adj_topology",
                    description=f"{topology_base.name} with topology-derived adjacency",
                    category="sweep_bddc",
                    cli_args=_with_cli_pairs(_without_flags(topology_base.cli_args, "--adjacency_source", "none", "csr"), "--adjacency_source", "topology"),
                ),
                Variant(
                    name=f"{topology_base.name}_adj_topology_corner",
                    description=f"{topology_base.name} with topology-derived adjacency and corner-only primals",
                    category="sweep_bddc",
                    cli_args=_with_cli_pairs(
                        _without_flags(topology_base.cli_args, "--adjacency_source", "none", "csr"),
                        "--adjacency_source",
                        "topology",
                        "--corner_only_primals",
                    ),
                ),
            )
            for variant in topology_variants:
                variant_lookup[variant.name] = variant
                run = _run_probe_case(
                    variant=variant,
                    ranks=2,
                    mesh_path=mesh_path,
                    out_dir=out_root / "topology_sweep" / variant.name / "rank2_p4_single",
                    elem_type="P4",
                    mode="single_solve",
                    linear_tolerance=float(linear_tolerance),
                    linear_max_iter=int(linear_max_iter),
                    guard_limit_gib=None,
                    wall_time_limit_s=3600.0,
                    reuse_existing=reuse_existing,
                    startup_progress_timeout_s=300.0,
                )
                summary["topology_sweep"][variant.name] = {
                    "variant_category": variant.category,
                    "description": variant.description,
                    **run,
                }

    combined_screening.update(summary["topology_sweep"])
    if hypre_linear and hypre_linear.get("status") == "completed":
        hypre_run_info = out_root / "linear_screen" / "hypre_control_v2" / "rank2_p4_single" / "data" / "run_info.json"
        overlay_dir = out_root / "plots"
        for phase_name, phase_entries in (
            ("approximate_side_sweep", summary["approximate_side_sweep"]),
            ("topology_sweep", summary["topology_sweep"]),
        ):
            for name, entry in phase_entries.items():
                if entry.get("status") != "completed":
                    continue
                run_info_path = out_root / phase_name / name / "rank2_p4_single" / "data" / "run_info.json"
                if not run_info_path.exists():
                    continue
                plot_path = overlay_dir / f"{name}_vs_hypre.png"
                _plot_sweep_histories(
                    plot_script=plot_script,
                    out_path=plot_path,
                    title=f"P4 Linear Elastic Convergence: Hypre vs {name}",
                    target=float(linear_tolerance),
                    runs=[("hypre", hypre_run_info), (name, run_info_path)],
                )
                summary["plots"]["overlays"][name] = _path_label(plot_path)
    promoted = _promote_bddc_sweep_candidates(screening=combined_screening)
    summary["promoted_candidates"] = promoted
    if not promoted:
        return summary

    diag_targets = ["hypre_control_v2", *promoted[:2]]
    for name in diag_targets:
        variant = variant_lookup.get(name)
        if variant is None:
            continue
        run = _run_probe_case(
            variant=variant,
            ranks=2,
            mesh_path=mesh_path,
            out_dir=out_root / "diagnostic_linear" / name / "rank2_p4_single",
            elem_type="P4",
            mode="single_solve",
            linear_tolerance=float(linear_tolerance),
            linear_max_iter=int(linear_max_iter),
            guard_limit_gib=None,
            wall_time_limit_s=3600.0,
            reuse_existing=reuse_existing,
            startup_progress_timeout_s=300.0,
            extra_petsc_opts=("ksp_monitor_singular_value=true", "ksp_view_eigenvalues=true"),
        )
        summary["diagnostic_linear"][name] = run

    best_name = promoted[0]
    best_metrics = combined_screening[best_name]["metrics"]
    if hypre_metrics:
        if int(best_metrics.get("iteration_count", 10**9)) > int(np.ceil(1.5 * float(hypre_metrics.get("iteration_count", 0.0)))):
            return summary
        if float(best_metrics.get("runtime_seconds", np.inf)) > 2.0 * float(hypre_metrics.get("runtime_seconds", np.inf)):
            return summary

    hypre_short = _run_case(
        variant=Variant(
            name="hypre_control_v2_short",
            description="Short nonlinear Hypre control",
            category="sweep_control",
            cli_args=(
                "--solver_type",
                "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
                "--pc_backend",
                "hypre",
                "--linear_tolerance",
                str(float(linear_tolerance)),
                "--pc_hypre_coarsen_type",
                "HMIS",
                "--pc_hypre_interp_type",
                "ext+i",
            ),
        ),
        ranks=2,
        mesh_path=mesh_path,
        step_max=1,
        out_dir=out_root / "nonlinear_short" / "hypre_control_v2",
        guard_limit_gib=None,
        max_deflation_basis_vectors=max_deflation_basis_vectors,
        elem_type="P4",
        reuse_existing=reuse_existing,
        startup_progress_timeout_s=300.0,
    )
    summary["nonlinear_short"]["hypre_control_v2"] = hypre_short
    for name in promoted:
        linear_variant = variant_lookup.get(name)
        if linear_variant is None:
            continue
        nonlinear_opts = _nonlinear_cli_from_linear_probe_variant(linear_variant)
        run = _run_case(
            variant=Variant(
                name=f"{name}_short",
                description=f"Short nonlinear follow-up for {name}",
                category="sweep_bddc",
                cli_args=(
                    "--solver_type",
                    "PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE",
                    "--pc_backend",
                    "bddc",
                    "--preconditioner_matrix_source",
                    "elastic",
                    "--linear_tolerance",
                    str(float(linear_tolerance)),
                    *nonlinear_opts,
                ),
            ),
            ranks=2,
            mesh_path=mesh_path,
            step_max=1,
            out_dir=out_root / "nonlinear_short" / name,
            guard_limit_gib=None,
            max_deflation_basis_vectors=max_deflation_basis_vectors,
            elem_type="P4",
            reuse_existing=reuse_existing,
            startup_progress_timeout_s=300.0,
        )
        accepted, reasons = _short_run_acceptance(candidate=run, control=hypre_short, runtime_factor_limit=2.0)
        summary["nonlinear_short"][name] = {
            **run,
            "accepted": bool(accepted),
            "acceptance_reasons": reasons,
        }

    accepted_candidates = [
        name for name in promoted if bool(summary["nonlinear_short"].get(name, {}).get("accepted", False))
    ]
    if not accepted_candidates:
        return summary

    accepted_candidates.sort(key=lambda name: float(summary["nonlinear_short"][name]["metrics"]["linear_total_rank_metric"]))
    best_name = accepted_candidates[0]
    summary["nonlinear_step10"] = {
        "winner": best_name,
        "run": _run_case(
            variant=Variant(
                name=f"{best_name}_step10",
                description=f"Rank-2 nonlinear step_max=10 for {best_name}",
                category="sweep_bddc",
                cli_args=(
                    "--solver_type",
                    "PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE",
                    "--pc_backend",
                    "bddc",
                    "--preconditioner_matrix_source",
                    "elastic",
                    "--linear_tolerance",
                    str(float(linear_tolerance)),
                    *_nonlinear_cli_from_linear_probe_variant(variant_lookup[best_name]),
                ),
            ),
            ranks=2,
            mesh_path=mesh_path,
            step_max=10,
            out_dir=out_root / "nonlinear_step10" / best_name,
            guard_limit_gib=None,
            max_deflation_basis_vectors=max_deflation_basis_vectors,
            elem_type="P4",
            reuse_existing=reuse_existing,
            startup_progress_timeout_s=300.0,
        ),
    }

    # Stage 4: optional rank-8 linear confirmation.
    summary["rank8_linear"] = {
        "winner": best_name,
        "hypre_control_v2": _run_probe_case(
            variant=registry["hypre_control_v2"],
            ranks=8,
            mesh_path=mesh_path,
            out_dir=out_root / "rank8_linear" / "hypre_control_v2",
            elem_type="P4",
            mode="single_solve",
            linear_tolerance=float(linear_tolerance),
            linear_max_iter=int(linear_max_iter),
            guard_limit_gib=None,
            wall_time_limit_s=5400.0,
            reuse_existing=reuse_existing,
            startup_progress_timeout_s=300.0,
        ),
        "bddc": _run_probe_case(
            variant=variant_lookup.get(best_name, registry[best_nonadaptive_name]) if best_nonadaptive_name is not None else variant_lookup[promoted[0]],
            ranks=8,
            mesh_path=mesh_path,
            out_dir=out_root / "rank8_linear" / best_name,
            elem_type="P4",
            mode="single_solve",
            linear_tolerance=float(linear_tolerance),
            linear_max_iter=int(linear_max_iter),
            guard_limit_gib=None,
            wall_time_limit_s=5400.0,
            reuse_existing=reuse_existing,
            startup_progress_timeout_s=300.0,
        ),
    }
    return summary


def _bddc_sweep_report_lines(*, mesh_path: Path, summary_payload: dict[str, Any]) -> list[str]:
    sweep = summary_payload["bddc_sweep"]
    def _fmt(metrics: dict[str, Any], key: str, fmt: str) -> str:
        if not metrics or metrics.get(key) is None:
            return "-"
        return format(float(metrics[key]), fmt)

    def _fmt_int(metrics: dict[str, Any], key: str) -> str:
        if not metrics or metrics.get(key) is None:
            return "-"
        return str(int(metrics[key]))

    def _append_table(lines: list[str], title: str, entries: dict[str, Any]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Variant | Status | Iter | Setup [s] | Solve [s] | Runtime [s] | Rel. residual | Peak RSS [GiB] | Faces | Coarse |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, entry in entries.items():
            metrics = entry.get("metrics", {})
            lines.append(
                f"| {name} | {entry.get('status', 'missing')} | "
                f"{_fmt_int(metrics, 'iteration_count')} | "
                f"{_fmt(metrics, 'setup_elapsed_s', '.3f')} | "
                f"{_fmt(metrics, 'solve_time', '.3f')} | "
                f"{_fmt(metrics, 'runtime_seconds', '.3f')} | "
                f"{_fmt(metrics, 'final_relative_residual', '.3e')} | "
                f"{_fmt(metrics, 'peak_rss_gib', '.3f')} | "
                f"{_fmt_int(metrics, 'bddc_candidate_faces')} | "
                f"{_fmt_int(metrics, 'bddc_coarse_size')} |"
            )

    lines = [
        "# BDDC Parameter Sweep for P4 (V2)",
        "",
        f"- Mesh: `{_mesh_label(mesh_path)}`",
        f"- Supersession note: {sweep.get('phase0_note')}",
        "- PETSc sources consulted:",
        "  - [`PCBDDC` manual page](https://petsc.org/release/manualpages/PC/PCBDDC/): baseline BDDC requirements and documented option families.",
        "  - [`ex56`](https://petsc.org/main/src/snes/tutorials/ex56.c.html): elasticity-oriented approximate-local BDDC and GAMG tuning.",
        "  - [`ex71`](https://petsc.org/main/src/ksp/ksp/tutorials/ex71.c.html): deluxe scaling and adaptive-threshold elasticity examples.",
        "  - [`ex59`](https://petsc.org/main/src/ksp/ksp/tutorials/ex59.c.html): high-order adjacency/corner-primal customization reserved as the next escalation path.",
        f"- Linear residual target: `{float(sweep['linear_tolerance']):.1e}`",
        "- Linear probe: distributed `P4`, rank `2`, `A=P=K_elast`, native PETSc `CG`, `ksp_norm_type=unpreconditioned`.",
        "- Corrected BDDC baseline: `pc_bddc_symmetric=true`, `pc_bddc_monolithic=true`, `pc_bddc_coarse_redundant_pc_type=svd`, `use_faces=true`.",
        f"- MUMPS available: `{'yes' if sweep.get('mumps_available', False) else 'no'}`",
        "",
        "## Option Smokes",
        "",
        "| Variant | Status | First progress [s] |",
        "| --- | --- | ---: |",
    ]
    for name, entry in sweep.get("option_smokes", {}).items():
        startup = entry.get("startup_summary", {}) or {}
        first_progress = startup.get("first_progress_elapsed_s")
        first_text = "-" if first_progress is None else f"{float(first_progress):.3f}"
        lines.append(f"| {name} | {entry.get('status', 'missing')} | {first_text} |")
    _append_table(lines, "Phase 1: Rank-2 Linear Elastic Screen", sweep.get("linear_screen", {}))
    if sweep.get("approximate_side_sweep"):
        _append_table(lines, "Phase 2: One-Sided Approximate-Local Sweep", sweep.get("approximate_side_sweep", {}))
    if sweep.get("topology_sweep"):
        _append_table(lines, "Phase 3: Topology and Corner-Primal Sweep", sweep.get("topology_sweep", {}))
    if sweep.get("plots", {}).get("overlays"):
        lines.extend(["", "## Convergence Plots", ""])
        for name, path in sweep["plots"]["overlays"].items():
            lines.append(f"- `{name}` vs Hypre: [{Path(path).name}]({path})")
    if sweep.get("plots", {}).get("aggregate"):
        lines.append(f"- Aggregate completed-candidate plot: [{Path(sweep['plots']['aggregate']).name}]({sweep['plots']['aggregate']})")
    lines.extend(
        [
            "",
            f"- Promoted candidates: `{', '.join(sweep.get('promoted_candidates', [])) or 'none'}`",
            "",
            "## Diagnostics",
            "",
            "| Variant | Status | Runtime [s] | PETSc log |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for name, entry in sweep.get("diagnostic_linear", {}).items():
        metrics = entry.get("metrics", {})
        lines.append(
            f"| {name} | {entry.get('status', 'missing')} | {_fmt(metrics, 'runtime_seconds', '.3f')} | "
            f"{metrics.get('petsc_log', '-') if metrics else '-'} |"
        )
    lines.extend(
        [
            "",
            "## Short Nonlinear Follow-up",
            "",
            "| Variant | Status | Accepted | Runtime [s] | Accepted states | Linear total [s] |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for name, entry in sweep.get("nonlinear_short", {}).items():
        metrics = entry.get("metrics", {})
        accepted = entry.get("accepted")
        accepted_text = "-" if accepted is None else ("yes" if accepted else "no")
        lines.append(
            f"| {name} | {entry.get('status', 'missing')} | {accepted_text} | "
            f"{_fmt(metrics, 'runtime_seconds', '.3f')} | "
            f"{_fmt_int(metrics, 'final_accepted_states')} | "
            f"{_fmt(metrics, 'linear_total_rank_metric', '.3f')} |"
        )
    step10 = sweep.get("nonlinear_step10", {})
    if step10.get("run"):
        run = step10["run"]
        lines.extend(
            [
                "",
                f"- Rank-2 `step_max=10` winner: `{step10.get('winner')}`",
                f"- Rank-2 `step_max=10` status: `{run.get('status', 'missing')}`",
            ]
        )
    rank8 = sweep.get("rank8_linear", {})
    if rank8.get("status") != "not_run" and rank8:
        lines.extend(
            [
                "",
                "## Optional Rank-8 Linear Confirmation",
                "",
                f"- Winner: `{rank8.get('winner')}`",
            ]
        )
        for name in ("hypre_control_v2", "bddc"):
            entry = rank8.get(name, {})
            metrics = entry.get("metrics", {})
            lines.append(
                f"- `{name}`: status `{entry.get('status', 'missing')}`, "
                f"runtime `{float(metrics.get('runtime_seconds', 0.0)):.3f} s`, "
                f"iterations `{int(metrics.get('iteration_count', 0))}`"
                if metrics
                else f"- `{name}`: status `{entry.get('status', 'missing')}`"
            )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- This recovery sweep corrected the BDDC baseline before comparing variants: symmetry, monolithic correction, coarse redundant SVD, and faces were treated as mandatory, not optional.",
            "- The next step is determined by the best completed elastic result only: continue local-GAMG tuning if iterations dropped materially, move to nonlinear mismatch only if the elastic probe is close enough to Hypre, or escalate topology classification if coarse information still looks weak.",
        ]
    )
    return lines


def _bddc_gate_report_lines(*, mesh_path: Path, summary_payload: dict[str, Any]) -> list[str]:
    runtime_smokes = summary_payload.get("bddc_runtime_smokes", {})
    gate = summary_payload.get("bddc_gate", {})
    gate_run = gate.get("run", {})
    lines = [
        "# P4 BDDC Gate",
        "",
        f"- Mesh: `{_mesh_label(mesh_path)}`",
        "- Tangent kernel: `rows`",
        "- Constitutive mode: `overlap`",
        "- Node ordering: `block_metis`",
        "- `recycle_preconditioner = false`",
        "",
        "## Runtime Smokes",
        "",
        "| Case | Status | Runtime [s] | Peak RSS [GiB] | Accepted states | Primal vertices | Local subdomain bytes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, run in runtime_smokes.get("runs", {}).items():
        metrics = run.get("metrics", {})
        runtime_text = f"{float(metrics.get('runtime_seconds', 0.0)):.3f}" if metrics else "-"
        peak_text = f"{float(metrics.get('peak_rss_gib', 0.0)):.3f}" if metrics else "-"
        accepted_text = str(int(metrics.get("final_accepted_states", 0))) if metrics else "-"
        primal_text = str(int(metrics.get("bddc_local_primal_vertices_count", 0))) if metrics else "-"
        bytes_text = f"{float(metrics.get('bddc_local_total_bytes', 0.0)):.0f}" if metrics else "-"
        lines.append(
            f"| {name} | {run.get('status', 'missing')} | {runtime_text} | {peak_text} | "
            f"{accepted_text} | {primal_text} | {bytes_text} |"
        )
    lines.extend(
        [
            "",
            "## Rank-8 Step-2 Gate",
            "",
            f"- Status: `{gate.get('status', 'missing')}`",
            f"- Eligible for full trajectory: `{'yes' if gate.get('eligible_for_full_trajectory', False) else 'no'}`",
        ]
    )
    if gate_run.get("status") == "completed":
        metrics = gate_run["metrics"]
        linear_total_text = (
            f"{float(metrics['linear_total_rank_metric']):.3f}"
            if "linear_total_rank_metric" in metrics
            else "-"
        )
        lines.extend(
            [
                "",
                f"- Runtime [s]: `{float(metrics['runtime_seconds']):.3f}`",
                f"- Accepted states: `{int(metrics['final_accepted_states'])}`",
                f"- Peak RSS [GiB]: `{float(metrics.get('peak_rss_gib', 0.0)):.3f}`",
                f"- Total linear solve + preconditioner [s]: `{linear_total_text}`",
                f"- Primal vertices: `{int(metrics.get('bddc_local_primal_vertices_count', 0))}`",
                f"- Local subdomain bytes: `{float(metrics.get('bddc_local_total_bytes', 0.0)):.0f}`",
            ]
        )
    elif gate_run.get("failure_excerpt"):
        lines.extend(
            [
                "",
                "## Failure Excerpt",
                "",
                "```text",
                str(gate_run["failure_excerpt"]),
                "```",
            ]
        )
    return lines


def _bddc_short_report_lines(
    *,
    mesh_path: Path,
    summary_payload: dict[str, Any],
) -> list[str]:
    p2_short = summary_payload["bddc_prototype"]["p2_short"]
    p4_short = summary_payload["bddc_prototype"]["p4_short"]
    winner = summary_payload["bddc_prototype"].get("winner")
    winner_eval = summary_payload["bddc_prototype"].get("winner_evaluation", {})
    p4_gate = summary_payload["bddc_prototype"].get("p4_gate", {})

    def _metric_text(metrics: dict[str, Any], key: str, fmt: str) -> str:
        if not metrics or metrics.get(key) is None:
            return "-"
        return format(float(metrics[key]), fmt)

    def _int_text(metrics: dict[str, Any], key: str) -> str:
        if not metrics or metrics.get(key) is None:
            return "-"
        return str(int(metrics[key]))

    lines = [
        "# BDDC Short-Run Diagnostics",
        "",
        f"- Mesh: `{_mesh_label(mesh_path)}`",
        "- Tangent kernel: `rows`",
        "- Constitutive mode: `overlap`",
        "- Node ordering: `block_metis`",
        "- `recycle_preconditioner = false`",
        "",
        "## P2 Rank-2 Step-10 Comparison",
        "",
        "| Variant | Status | First progress [s] | Runtime [s] | Linear total [s] | Accepted states | Peak RSS [GiB] |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    control = p2_short.get("hypre_current")
    control_metrics = control.get("metrics", {}) if control else {}
    if control:
        lines.append(
            f"| hypre_current | {control.get('status', 'missing')} | "
            f"{_metric_text(control_metrics, 'first_progress_elapsed_s', '.3f')} | "
            f"{_metric_text(control_metrics, 'runtime_seconds', '.3f')} | "
            f"{_metric_text(control_metrics, 'linear_total_rank_metric', '.3f')} | "
            f"{_int_text(control_metrics, 'final_accepted_states')} | "
            f"{_metric_text(control_metrics, 'peak_rss_gib', '.3f')} |"
        )
    for name, run in p2_short.get("bddc_candidates", {}).items():
        metrics = run.get("metrics", {})
        accepted = winner_eval.get(name, {}).get("accepted")
        label = f"{name}{' (winner)' if name == winner else ''}"
        status = f"{run.get('status', 'missing')}"
        if accepted is False:
            status = f"{status} / rejected"
        lines.append(
            f"| {label} | {status} | "
            f"{_metric_text(metrics, 'first_progress_elapsed_s', '.3f')} | "
            f"{_metric_text(metrics, 'runtime_seconds', '.3f')} | "
            f"{_metric_text(metrics, 'linear_total_rank_metric', '.3f')} | "
            f"{_int_text(metrics, 'final_accepted_states')} | "
            f"{_metric_text(metrics, 'peak_rss_gib', '.3f')} |"
        )
    lines.extend(
        [
            "",
            "## P4 Rank-8 Step-10 Gate",
            "",
            f"- Winner carried from P2: `{winner or 'none'}`",
            f"- Gate eligible for P4 full trajectory: `{'yes' if p4_gate.get('eligible', False) else 'no'}`",
        ]
    )
    if p4_gate.get("reasons"):
        lines.append(f"- Gate rejection reasons: `{', '.join(str(v) for v in p4_gate['reasons'])}`")
    lines.extend(
        [
            "",
            "| Variant | Status | First progress [s] | Runtime [s] | Linear total [s] | Accepted states | Peak RSS [GiB] |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, run in (("hypre_current", p4_short.get("hypre_current", {})), ("bddc", p4_short.get("bddc", {}))):
        metrics = run.get("metrics", {})
        lines.append(
            f"| {label} | {run.get('status', 'missing')} | "
            f"{_metric_text(metrics, 'first_progress_elapsed_s', '.3f')} | "
            f"{_metric_text(metrics, 'runtime_seconds', '.3f')} | "
            f"{_metric_text(metrics, 'linear_total_rank_metric', '.3f')} | "
            f"{_int_text(metrics, 'final_accepted_states')} | "
            f"{_metric_text(metrics, 'peak_rss_gib', '.3f')} |"
        )
    return lines


def _bddc_full_report_lines(
    *,
    mesh_path: Path,
    summary_payload: dict[str, Any],
    baseline_summary_path: Path,
    baseline_report_path: Path,
) -> list[str]:
    proto = summary_payload["bddc_prototype"]
    baseline_p2 = summary_payload["baseline_reuse"]["P2"]
    baseline_p4 = summary_payload["baseline_reuse"]["P4"]
    p2_full = proto.get("p2_full", {})
    p4_full = proto.get("p4_full", {})

    def _entry_value(run: dict[str, Any], key: str, fmt: str) -> str:
        metrics = run.get("metrics")
        if not metrics or key not in metrics:
            return "not run"
        return format(float(metrics[key]), fmt) if isinstance(metrics[key], (int, float, np.integer, np.floating)) else str(metrics[key])

    def _pc_total(metrics: dict[str, Any]) -> float:
        return float(metrics.get("init_linear_preconditioner_time", 0.0)) + float(metrics.get("attempt_linear_preconditioner_time_total", 0.0))

    p2_bddc_pc_total = None if not p2_full.get("metrics") else _pc_total(p2_full["metrics"])
    p4_bddc_pc_total = None if not p4_full.get("metrics") else _pc_total(p4_full["metrics"])

    lines = [
        "# BDDC Full-Trajectory Comparison",
        "",
        f"- Mesh: `{_mesh_label(mesh_path)}`",
        "- Tangent kernel: `rows`",
        "- Constitutive mode: `overlap`",
        "- Node ordering: `block_metis`",
        "- `recycle_preconditioner = false`",
        "- Reused baseline summary: "
        f"[summary.json]({baseline_summary_path})",
        "- Reused baseline report: "
        f"[report_p2_vs_p4_rank8_final_memfix.md]({baseline_report_path})",
        "",
        f"- Winning BDDC candidate: `{proto.get('winner') or 'none'}`",
        "",
        "## P2 Full Trajectory",
        "",
        "| Metric | Reused baseline | BDDC |",
        "| --- | ---: | ---: |",
        f"| Runtime [s] | {float(baseline_p2['runtime_seconds']):.3f} | {_entry_value(p2_full, 'runtime_seconds', '.3f')} |",
        f"| Final accepted states | {int(baseline_p2['final_accepted_states'])} | {_entry_value(p2_full, 'final_accepted_states', '.0f')} |",
        f"| Final lambda | {float(baseline_p2['lambda_last']):.9f} | {_entry_value(p2_full, 'lambda_last', '.9f')} |",
        f"| Final omega | {float(baseline_p2['omega_last']):.9f} | {_entry_value(p2_full, 'omega_last', '.9f')} |",
        f"| Final Umax | {float(baseline_p2['umax_last']):.9f} | {_entry_value(p2_full, 'umax_last', '.9f')} |",
        f"| Total preconditioner time [s] | {_pc_total(baseline_p2):.3f} | {('not run' if p2_bddc_pc_total is None else f'{p2_bddc_pc_total:.3f}')} |",
        f"| Peak RSS [GiB] | {float(baseline_p2.get('peak_rss_gib', 0.0)):.3f} | {_entry_value(p2_full, 'peak_rss_gib', '.3f')} |",
        "",
        "## P4 Full Trajectory",
        "",
        "| Metric | Reused baseline | BDDC |",
        "| --- | ---: | ---: |",
        f"| Runtime [s] | {float(baseline_p4['runtime_seconds']):.3f} | {_entry_value(p4_full, 'runtime_seconds', '.3f')} |",
        f"| Final accepted states | {int(baseline_p4['final_accepted_states'])} | {_entry_value(p4_full, 'final_accepted_states', '.0f')} |",
        f"| Final lambda | {float(baseline_p4['lambda_last']):.9f} | {_entry_value(p4_full, 'lambda_last', '.9f')} |",
        f"| Final omega | {float(baseline_p4['omega_last']):.9f} | {_entry_value(p4_full, 'omega_last', '.9f')} |",
        f"| Final Umax | {float(baseline_p4['umax_last']):.9f} | {_entry_value(p4_full, 'umax_last', '.9f')} |",
        f"| Total preconditioner time [s] | {_pc_total(baseline_p4):.3f} | {('not run' if p4_bddc_pc_total is None else f'{p4_bddc_pc_total:.3f}')} |",
        f"| Peak RSS [GiB] | {float(baseline_p4.get('peak_rss_gib', 0.0)):.3f} | {_entry_value(p4_full, 'peak_rss_gib', '.3f')} |",
        "",
    ]
    if p4_full.get("status") != "completed":
        lines.extend(
            [
                "## Conclusion",
                "",
                "- BDDC did not clear the P4 short-run gate or full run, so it is not a viable replacement for the current Hypre production path yet.",
            ]
        )
    else:
        runtime_improvement = (
            float(baseline_p4["runtime_seconds"]) - float(p4_full["metrics"]["runtime_seconds"])
        ) / float(baseline_p4["runtime_seconds"])
        pc_improvement = (
            _pc_total(baseline_p4) - _pc_total(p4_full["metrics"])
        ) / _pc_total(baseline_p4) if _pc_total(baseline_p4) > 0.0 else 0.0
        lines.extend(
            [
                "## Conclusion",
                "",
                f"- P4 runtime improvement vs baseline: `{100.0 * runtime_improvement:.2f}%`",
                f"- P4 preconditioner-time improvement vs baseline: `{100.0 * pc_improvement:.2f}%`",
                "- Recommend switching only if continuation reach is preserved and the improvement clears the strict threshold.",
            ]
        )
    return lines


def _step2_report_lines(*, mesh_path: Path, summary_payload: dict[str, Any]) -> list[str]:
    lines = [
        "# P4 Preconditioner Step-2 Screening",
        "",
        f"- Mesh: `{_mesh_label(mesh_path)}`",
        "- Element order: `P4`",
        "- Tangent kernel: `rows`",
        "- Constitutive mode: `overlap`",
        "- Node ordering: `block_metis`",
        "- `recycle_preconditioner = false`",
        f"- Screen ranks: `{', '.join(str(v) for v in summary_payload.get('screen_ranks', []))}`",
        f"- Scaling ranks: `{', '.join(str(v) for v in summary_payload.get('scale_ranks', []))}`",
        "",
        "## Option Smokes",
        "",
        "| Variant | Status | Rank-1 backend |",
        "| --- | --- | --- |",
    ]
    for name, entry in summary_payload.get("option_smokes", {}).items():
        backend = "-"
        if entry.get("status") == "completed":
            backend = str(entry.get("metrics", {}).get("pc_backend", "-"))
        lines.append(f"| {name} | {entry.get('status', 'missing')} | {backend} |")

    runtime_smokes = summary_payload.get("bddc_runtime_smokes", {})
    if runtime_smokes:
        lines.extend(
            [
                "",
                "## BDDC Runtime Smokes",
                "",
                "| Case | Status | Runtime [s] | Peak RSS [GiB] | Accepted states | Primal vertices |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, run in runtime_smokes.get("runs", {}).items():
            metrics = run.get("metrics", {})
            runtime_text = f"{float(metrics.get('runtime_seconds', 0.0)):.3f}" if metrics else "-"
            peak_text = f"{float(metrics.get('peak_rss_gib', 0.0)):.3f}" if metrics else "-"
            accepted_text = str(int(metrics.get("final_accepted_states", 0))) if metrics else "-"
            primal_text = str(int(metrics.get("bddc_local_primal_vertices_count", 0))) if metrics else "-"
            lines.append(
                f"| {name} | {run.get('status', 'missing')} | {runtime_text} | {peak_text} | {accepted_text} | {primal_text} |"
            )

    lines.extend(
        [
            "",
            "## Rank-8 Screen",
            "",
            "| Variant | Status | Rank-8 linear total [s] | Rank-8 prec [s] | Rank-8 solve [s] | Rank-8 accepted states | Rank-8 peak RSS [GiB] |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, entry in summary_payload.get("screening", {}).items():
        rank8 = entry.get("runs", {}).get("8", {})
        if rank8.get("status") == "completed":
            m = rank8["metrics"]
            lines.append(
                f"| {name} | completed | {float(m['linear_total_rank_metric']):.3f} | "
                f"{float(m['attempt_linear_preconditioner_time_total']):.3f} | "
                f"{float(m['attempt_linear_solve_time_total']):.3f} | "
                f"{int(m['final_accepted_states'])} | {float(m.get('peak_rss_gib', 0.0)):.3f} |"
            )
        else:
            lines.append(f"| {name} | {rank8.get('status', entry.get('status', 'missing'))} | - | - | - | - | - |")

    lines.extend(
        [
            "",
            f"Promoted AIJ variants: `{', '.join(summary_payload.get('promoted_aij_variants', [])) if summary_payload.get('promoted_aij_variants') else 'none'}`",
            f"Best AIJ candidate: `{summary_payload.get('best_aij_name', 'unselected')}`",
            "",
            "## Full Step-2 Scaling",
            "",
            "| Variant | Rank | Runtime [s] | Linear total [s] | Peak RSS [GiB] | Final accepted states |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, entry in summary_payload.get("scaling", {}).items():
        for rank in sorted(entry.get("runs", {}), key=lambda x: int(x)):
            run = entry["runs"][rank]
            if run.get("status") != "completed":
                lines.append(f"| {name} | {rank} | - | - | - | - |")
                continue
            m = run["metrics"]
            lines.append(
                f"| {name} | {rank} | {float(m['runtime_seconds']):.3f} | {float(m['linear_total_rank_metric']):.3f} | "
                f"{float(m.get('peak_rss_gib', 0.0)):.3f} | {int(m['final_accepted_states'])} |"
            )

    gate = summary_payload.get("bddc_gate", {})
    lines.extend(
        [
            "",
            "## BDDC Gate",
            "",
            f"- Status: `{gate.get('status', 'missing')}`",
            f"- Eligible for full trajectory: `{'yes' if gate.get('eligible_for_full_trajectory', False) else 'no'}`",
        ]
    )
    if gate.get("run", {}).get("status") == "completed":
        metrics = gate["run"]["metrics"]
        lines.extend(
            [
                f"- Rank-8 runtime [s]: `{float(metrics['runtime_seconds']):.3f}`",
                f"- Rank-8 accepted states: `{int(metrics['final_accepted_states'])}`",
                f"- Rank-8 peak RSS [GiB]: `{float(metrics.get('peak_rss_gib', 0.0)):.3f}`",
            ]
        )
    return lines


def _final_report_lines(
    *,
    mesh_path: Path,
    summary_payload: dict[str, Any],
    baseline_summary_path: Path,
    baseline_report_path: Path,
    recycle_failure_report: Path,
) -> list[str]:
    baseline = summary_payload["full_compare"]["baseline"]
    best_aij = summary_payload["full_compare"]["best_aij"]
    bddc = summary_payload["full_compare"]["bddc"]
    baseline_metrics = baseline["metrics"]
    best_aij_metrics = _entry_metrics(best_aij)
    bddc_metrics = _entry_metrics(bddc)
    baseline_pc_total, baseline_linear_total = _full_run_totals(baseline_metrics)
    best_aij_pc_total = None if best_aij_metrics is None else _full_run_totals(best_aij_metrics)[0]
    best_aij_linear_total = None if best_aij_metrics is None else _full_run_totals(best_aij_metrics)[1]
    bddc_pc_total = None if bddc_metrics is None else _full_run_totals(bddc_metrics)[0]
    bddc_linear_total = None if bddc_metrics is None else _full_run_totals(bddc_metrics)[1]

    switch_ok = False
    runtime_improvement = None
    pc_improvement = None
    if bddc_metrics is not None:
        runtime_improvement = (
            float(baseline_metrics["runtime_seconds"]) - float(bddc_metrics["runtime_seconds"])
        ) / float(baseline_metrics["runtime_seconds"])
        pc_improvement = (
            baseline_pc_total - float(bddc_pc_total)
        ) / baseline_pc_total if baseline_pc_total > 0.0 else 0.0
        preserves_reach = int(bddc_metrics["final_accepted_states"]) >= int(baseline_metrics["final_accepted_states"])
        peak_not_worse = float(bddc_metrics.get("peak_rss_gib", 0.0)) <= float(baseline_metrics.get("peak_rss_gib", 0.0))
        switch_ok = preserves_reach and peak_not_worse and (runtime_improvement >= 0.10 or pc_improvement >= 0.20)
    else:
        preserves_reach = False
        peak_not_worse = False

    def _full_value(entry: dict[str, Any] | None, key: str, fmt: str) -> str:
        metrics = _entry_metrics(entry)
        if metrics is None:
            return "not run"
        return _format_metric(entry, key, fmt)

    lines = [
        "# P4 Preconditioner Full-Trajectory Comparison",
        "",
        f"- Mesh: `{_mesh_label(mesh_path)}`",
        "- Element order: `P4`",
        "- Tangent kernel: `rows`",
        "- Constitutive mode: `overlap`",
        "- Node ordering: `block_metis`",
        "- `recycle_preconditioner = false`",
        "- Reused baseline summary: "
        f"[summary.json]({baseline_summary_path})",
        "- Reused baseline report: "
        f"[report_p2_vs_p4_rank8_final_memfix.md]({baseline_report_path})",
        "- Recycle-enabled failure context: "
        f"[report_p4_rank8_recycle_guard80_failed.md]({recycle_failure_report})",
        "",
        "## Compared Variants",
        "",
        f"- Reused baseline: `hypre_current`",
        f"- Best AIJ candidate: `{best_aij['name']}`"
        + (" (reused baseline)" if best_aij.get("reused_baseline") else ""),
        f"- BDDC full trajectory: `{bddc.get('status', 'missing')}`",
        "",
        "## Full Trajectory",
        "",
        "| Metric | Reused baseline | Best AIJ candidate | BDDC |",
        "| --- | ---: | ---: | ---: |",
        f"| Runtime [s] | {float(baseline_metrics['runtime_seconds']):.3f} | {_full_value(best_aij, 'runtime_seconds', '.3f')} | {_full_value(bddc, 'runtime_seconds', '.3f')} |",
        f"| Final accepted states | {int(baseline_metrics['final_accepted_states'])} | {_full_value(best_aij, 'final_accepted_states', '.0f')} | {_full_value(bddc, 'final_accepted_states', '.0f')} |",
        f"| Continuation advances after init | {int(baseline_metrics['accepted_continuation_advances'])} | {_full_value(best_aij, 'accepted_continuation_advances', '.0f')} | {_full_value(bddc, 'accepted_continuation_advances', '.0f')} |",
        f"| Final lambda | {float(baseline_metrics['lambda_last']):.9f} | {_full_value(best_aij, 'lambda_last', '.9f')} | {_full_value(bddc, 'lambda_last', '.9f')} |",
        f"| Final omega | {float(baseline_metrics['omega_last']):.9f} | {_full_value(best_aij, 'omega_last', '.9f')} | {_full_value(bddc, 'omega_last', '.9f')} |",
        f"| Final Umax | {float(baseline_metrics['umax_last']):.9f} | {_full_value(best_aij, 'umax_last', '.9f')} | {_full_value(bddc, 'umax_last', '.9f')} |",
        f"| Total preconditioner time [s] | {baseline_pc_total:.3f} | {f'{float(best_aij_pc_total):.3f}' if best_aij_pc_total is not None else 'not run'} | {f'{float(bddc_pc_total):.3f}' if bddc_pc_total is not None else 'not run'} |",
        f"| Total linear solve time [s] | {baseline_linear_total:.3f} | {f'{float(best_aij_linear_total):.3f}' if best_aij_linear_total is not None else 'not run'} | {f'{float(bddc_linear_total):.3f}' if bddc_linear_total is not None else 'not run'} |",
        f"| Preconditioner rebuild count | - | {_full_value(best_aij, 'preconditioner_rebuild_count', '.0f')} | {_full_value(bddc, 'preconditioner_rebuild_count', '.0f')} |",
        f"| Preconditioner reuse count | - | {_full_value(best_aij, 'preconditioner_reuse_count', '.0f')} | {_full_value(bddc, 'preconditioner_reuse_count', '.0f')} |",
        f"| Peak RSS [GiB] | {float(baseline_metrics.get('peak_rss_gib', 0.0)):.3f} | {_full_value(best_aij, 'peak_rss_gib', '.3f')} | {_full_value(bddc, 'peak_rss_gib', '.3f')} |",
        "",
        "## Step-2 Gate Summary",
        "",
        f"- Screened variants: `{', '.join(summary_payload.get('screening', {}).keys())}`",
        f"- Promoted AIJ variants: `{', '.join(summary_payload.get('promoted_aij_variants', [])) if summary_payload.get('promoted_aij_variants') else 'none'}`",
        f"- Best AIJ candidate after scaling: `{summary_payload.get('best_aij_name', 'unselected')}`",
        f"- BDDC gate status: `{summary_payload.get('bddc_gate', {}).get('status', 'missing')}`",
        "",
        "## Decision",
        "",
    ]
    if bddc_metrics is not None and runtime_improvement is not None and pc_improvement is not None:
        lines.extend(
            [
                f"- BDDC runtime improvement vs reused baseline: `{100.0 * runtime_improvement:.2f}%`",
                f"- BDDC total preconditioner-time improvement vs reused baseline: `{100.0 * pc_improvement:.2f}%`",
                f"- BDDC continuation reach preserved: `{'yes' if preserves_reach else 'no'}`",
                f"- BDDC peak RSS not worse: `{'yes' if peak_not_worse else 'no'}`",
                f"- BDDC good enough to replace current Hypre by the strict rule: `{'yes' if switch_ok else 'no'}`",
            ]
        )
    else:
        lines.append("- BDDC did not clear the step-2 gate, so no BDDC full-trajectory replacement decision was made.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The old full trajectory was reused from the current successful implementation; it was not rerun.",
            "- If the best AIJ candidate remained `hypre_current`, the baseline run was reused for both roles.",
            "- The recycle-enabled failure remains context only because the main benchmark target here is the successful no-recycle production baseline.",
        ]
    )
    return lines


def _write_summary_csv(path: Path, *, summary_payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    if "bddc_sweep" in summary_payload:
        sweep = summary_payload["bddc_sweep"]
        for stage_name in ("option_smokes", "linear_screen"):
            for variant_name, entry in sweep.get(stage_name, {}).items():
                rows.append(
                    {
                        "stage": stage_name,
                        "variant": variant_name,
                        "rank": 1 if stage_name == "option_smokes" else 2,
                        "status": entry.get("status"),
                        "runtime_seconds": entry.get("metrics", {}).get("runtime_seconds"),
                        "linear_total_rank_metric": entry.get("metrics", {}).get("linear_total_rank_metric"),
                        "pc_backend": entry.get("metrics", {}).get("pc_backend"),
                        "preconditioner_matrix_policy": entry.get("metrics", {}).get("preconditioner_matrix_policy"),
                        "preconditioner_rebuild_policy": entry.get("metrics", {}).get("preconditioner_rebuild_policy"),
                        "peak_rss_gib": entry.get("metrics", {}).get("peak_rss_gib"),
                        "final_accepted_states": entry.get("metrics", {}).get("final_accepted_states"),
                        "first_progress_elapsed_s": entry.get("metrics", {}).get("first_progress_elapsed_s"),
                        "progress_file_created": entry.get("metrics", {}).get("progress_file_created"),
                        "startup_stall_reason": entry.get("metrics", {}).get("startup_stall_reason"),
                    }
                )
        for variant_name, entry in sweep.get("nonlinear_short", {}).items():
            rows.append(
                {
                    "stage": "nonlinear_short",
                    "variant": variant_name,
                    "rank": 2,
                    "status": entry.get("status"),
                    "runtime_seconds": entry.get("metrics", {}).get("runtime_seconds"),
                    "linear_total_rank_metric": entry.get("metrics", {}).get("linear_total_rank_metric"),
                    "pc_backend": entry.get("metrics", {}).get("pc_backend"),
                    "preconditioner_matrix_policy": entry.get("metrics", {}).get("preconditioner_matrix_policy"),
                    "preconditioner_rebuild_policy": entry.get("metrics", {}).get("preconditioner_rebuild_policy"),
                    "peak_rss_gib": entry.get("metrics", {}).get("peak_rss_gib"),
                    "final_accepted_states": entry.get("metrics", {}).get("final_accepted_states"),
                    "first_progress_elapsed_s": entry.get("metrics", {}).get("first_progress_elapsed_s"),
                    "progress_file_created": entry.get("metrics", {}).get("progress_file_created"),
                    "startup_stall_reason": entry.get("metrics", {}).get("startup_stall_reason"),
                }
            )
        step10 = sweep.get("nonlinear_step10", {}).get("run", {})
        if step10:
            rows.append(
                {
                    "stage": "nonlinear_step10",
                    "variant": sweep.get("nonlinear_step10", {}).get("winner"),
                    "rank": 2,
                    "status": step10.get("status"),
                    "runtime_seconds": step10.get("metrics", {}).get("runtime_seconds"),
                    "linear_total_rank_metric": step10.get("metrics", {}).get("linear_total_rank_metric"),
                    "pc_backend": step10.get("metrics", {}).get("pc_backend"),
                    "preconditioner_matrix_policy": step10.get("metrics", {}).get("preconditioner_matrix_policy"),
                    "preconditioner_rebuild_policy": step10.get("metrics", {}).get("preconditioner_rebuild_policy"),
                    "peak_rss_gib": step10.get("metrics", {}).get("peak_rss_gib"),
                    "final_accepted_states": step10.get("metrics", {}).get("final_accepted_states"),
                    "first_progress_elapsed_s": step10.get("metrics", {}).get("first_progress_elapsed_s"),
                    "progress_file_created": step10.get("metrics", {}).get("progress_file_created"),
                    "startup_stall_reason": step10.get("metrics", {}).get("startup_stall_reason"),
                }
            )
        for variant_name, entry in (
            ("hypre_control_v2", sweep.get("rank8_linear", {}).get("hypre_control_v2", {})),
            ("bddc", sweep.get("rank8_linear", {}).get("bddc", {})),
        ):
            if entry:
                rows.append(
                    {
                        "stage": "rank8_linear",
                        "variant": variant_name,
                        "rank": 8,
                        "status": entry.get("status"),
                        "runtime_seconds": entry.get("metrics", {}).get("runtime_seconds"),
                        "linear_total_rank_metric": entry.get("metrics", {}).get("linear_total_rank_metric"),
                        "pc_backend": entry.get("metrics", {}).get("pc_backend"),
                        "preconditioner_matrix_policy": entry.get("metrics", {}).get("preconditioner_matrix_policy"),
                        "preconditioner_rebuild_policy": entry.get("metrics", {}).get("preconditioner_rebuild_policy"),
                        "peak_rss_gib": entry.get("metrics", {}).get("peak_rss_gib"),
                        "final_accepted_states": entry.get("metrics", {}).get("final_accepted_states"),
                        "first_progress_elapsed_s": entry.get("metrics", {}).get("first_progress_elapsed_s"),
                        "progress_file_created": entry.get("metrics", {}).get("progress_file_created"),
                        "startup_stall_reason": entry.get("metrics", {}).get("startup_stall_reason"),
                    }
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "stage",
                    "variant",
                    "rank",
                    "status",
                    "runtime_seconds",
                    "linear_total_rank_metric",
                    "pc_backend",
                    "preconditioner_matrix_policy",
                    "preconditioner_rebuild_policy",
                    "peak_rss_gib",
                    "final_accepted_states",
                    "first_progress_elapsed_s",
                    "progress_file_created",
                    "startup_stall_reason",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return
    if "bddc_prototype" in summary_payload:
        proto = summary_payload["bddc_prototype"]
        p2_short = proto.get("p2_short", {})
        for stage_name, collection in (
            ("p2_step1_candidates", p2_short.get("step1_candidates", {})),
            ("p2_step10_candidates", p2_short.get("bddc_candidates", {})),
        ):
            for variant_name, run in collection.items():
                rows.append(
                    {
                        "stage": stage_name,
                        "variant": variant_name,
                        "rank": 2,
                        "status": run.get("status"),
                        "runtime_seconds": run.get("metrics", {}).get("runtime_seconds"),
                        "linear_total_rank_metric": run.get("metrics", {}).get("linear_total_rank_metric"),
                        "pc_backend": run.get("metrics", {}).get("pc_backend"),
                        "preconditioner_matrix_policy": run.get("metrics", {}).get("preconditioner_matrix_policy"),
                        "preconditioner_rebuild_policy": run.get("metrics", {}).get("preconditioner_rebuild_policy"),
                        "peak_rss_gib": run.get("metrics", {}).get("peak_rss_gib"),
                        "final_accepted_states": run.get("metrics", {}).get("final_accepted_states"),
                        "first_progress_elapsed_s": run.get("metrics", {}).get("first_progress_elapsed_s"),
                        "progress_file_created": run.get("metrics", {}).get("progress_file_created"),
                        "startup_stall_reason": run.get("metrics", {}).get("startup_stall_reason"),
                    }
                )
        for variant_name, run in (
            ("p2_hypre_current", p2_short.get("hypre_current", {})),
            ("p2_full", proto.get("p2_full", {})),
            ("p4_hypre_current", proto.get("p4_short", {}).get("hypre_current", {})),
            ("p4_bddc", proto.get("p4_short", {}).get("bddc", {})),
            ("p4_full", proto.get("p4_full", {})),
        ):
            rank = 8 if "full" in variant_name or "p4_" in variant_name else 2
            rows.append(
                {
                    "stage": variant_name,
                    "variant": proto.get("winner") if "bddc" in variant_name or variant_name.endswith("full") else "hypre_current",
                    "rank": rank,
                    "status": run.get("status"),
                    "runtime_seconds": run.get("metrics", {}).get("runtime_seconds"),
                    "linear_total_rank_metric": run.get("metrics", {}).get("linear_total_rank_metric"),
                    "pc_backend": run.get("metrics", {}).get("pc_backend"),
                    "preconditioner_matrix_policy": run.get("metrics", {}).get("preconditioner_matrix_policy"),
                    "preconditioner_rebuild_policy": run.get("metrics", {}).get("preconditioner_rebuild_policy"),
                    "peak_rss_gib": run.get("metrics", {}).get("peak_rss_gib"),
                    "final_accepted_states": run.get("metrics", {}).get("final_accepted_states"),
                    "first_progress_elapsed_s": run.get("metrics", {}).get("first_progress_elapsed_s"),
                    "progress_file_created": run.get("metrics", {}).get("progress_file_created"),
                    "startup_stall_reason": run.get("metrics", {}).get("startup_stall_reason"),
                }
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "stage",
                    "variant",
                    "rank",
                    "status",
                    "runtime_seconds",
                    "linear_total_rank_metric",
                    "pc_backend",
                    "preconditioner_matrix_policy",
                    "preconditioner_rebuild_policy",
                    "peak_rss_gib",
                    "final_accepted_states",
                    "first_progress_elapsed_s",
                    "progress_file_created",
                    "startup_stall_reason",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return
    for variant_name, entry in summary_payload.get("option_smokes", {}).items():
        rows.append(
            {
                "stage": "option_smokes",
                "variant": variant_name,
                "rank": 1,
                "status": entry.get("status"),
                "runtime_seconds": entry.get("metrics", {}).get("runtime_seconds"),
                "linear_total_rank_metric": entry.get("metrics", {}).get("linear_total_rank_metric"),
                "pc_backend": entry.get("metrics", {}).get("pc_backend"),
                "preconditioner_matrix_policy": entry.get("metrics", {}).get("preconditioner_matrix_policy"),
                "preconditioner_rebuild_policy": entry.get("metrics", {}).get("preconditioner_rebuild_policy"),
                "peak_rss_gib": entry.get("metrics", {}).get("peak_rss_gib"),
                "final_accepted_states": entry.get("metrics", {}).get("final_accepted_states"),
                "first_progress_elapsed_s": entry.get("metrics", {}).get("first_progress_elapsed_s"),
                "progress_file_created": entry.get("metrics", {}).get("progress_file_created"),
                "startup_stall_reason": entry.get("metrics", {}).get("startup_stall_reason"),
            }
        )
    for name, run in summary_payload.get("bddc_runtime_smokes", {}).get("runs", {}).items():
        rows.append(
            {
                "stage": "bddc_runtime_smokes",
                "variant": "bddc",
                "rank": int(name.split("_")[0].replace("rank", "")),
                "status": run.get("status"),
                "runtime_seconds": run.get("metrics", {}).get("runtime_seconds"),
                "linear_total_rank_metric": run.get("metrics", {}).get("linear_total_rank_metric"),
                "pc_backend": run.get("metrics", {}).get("pc_backend"),
                "preconditioner_matrix_policy": run.get("metrics", {}).get("preconditioner_matrix_policy"),
                "preconditioner_rebuild_policy": run.get("metrics", {}).get("preconditioner_rebuild_policy"),
                "peak_rss_gib": run.get("metrics", {}).get("peak_rss_gib"),
                "final_accepted_states": run.get("metrics", {}).get("final_accepted_states"),
                "first_progress_elapsed_s": run.get("metrics", {}).get("first_progress_elapsed_s"),
                "progress_file_created": run.get("metrics", {}).get("progress_file_created"),
                "startup_stall_reason": run.get("metrics", {}).get("startup_stall_reason"),
            }
        )
    for stage_name in ("screening", "scaling"):
        stage = summary_payload.get(stage_name, {})
        for variant_name, entry in stage.items():
            for rank, run in entry.get("runs", {}).items():
                rows.append(
                    {
                        "stage": stage_name,
                        "variant": variant_name,
                        "rank": int(rank),
                        "status": run.get("status"),
                        "runtime_seconds": run.get("metrics", {}).get("runtime_seconds"),
                        "linear_total_rank_metric": run.get("metrics", {}).get("linear_total_rank_metric"),
                        "pc_backend": run.get("metrics", {}).get("pc_backend"),
                        "preconditioner_matrix_policy": run.get("metrics", {}).get("preconditioner_matrix_policy"),
                        "preconditioner_rebuild_policy": run.get("metrics", {}).get("preconditioner_rebuild_policy"),
                        "peak_rss_gib": run.get("metrics", {}).get("peak_rss_gib"),
                        "final_accepted_states": run.get("metrics", {}).get("final_accepted_states"),
                        "first_progress_elapsed_s": run.get("metrics", {}).get("first_progress_elapsed_s"),
                        "progress_file_created": run.get("metrics", {}).get("progress_file_created"),
                        "startup_stall_reason": run.get("metrics", {}).get("startup_stall_reason"),
                    }
                )
    gate_run = summary_payload.get("bddc_gate", {}).get("run", {})
    if gate_run:
        rows.append(
            {
                "stage": "bddc_gate",
                "variant": "bddc",
                "rank": 8,
                "status": gate_run.get("status"),
                "runtime_seconds": gate_run.get("metrics", {}).get("runtime_seconds"),
                "linear_total_rank_metric": gate_run.get("metrics", {}).get("linear_total_rank_metric"),
                "pc_backend": gate_run.get("metrics", {}).get("pc_backend"),
                "preconditioner_matrix_policy": gate_run.get("metrics", {}).get("preconditioner_matrix_policy"),
                "preconditioner_rebuild_policy": gate_run.get("metrics", {}).get("preconditioner_rebuild_policy"),
                "peak_rss_gib": gate_run.get("metrics", {}).get("peak_rss_gib"),
                "final_accepted_states": gate_run.get("metrics", {}).get("final_accepted_states"),
                "first_progress_elapsed_s": gate_run.get("metrics", {}).get("first_progress_elapsed_s"),
                "progress_file_created": gate_run.get("metrics", {}).get("progress_file_created"),
                "startup_stall_reason": gate_run.get("metrics", {}).get("startup_stall_reason"),
            }
        )
    for variant_name in ("best_aij", "bddc"):
        run = summary_payload.get("full_compare", {}).get(variant_name, {})
        if not run:
            continue
        rows.append(
            {
                "stage": f"full_compare_{variant_name}",
                "variant": run.get("name"),
                "rank": 8,
                "status": run.get("status"),
                "runtime_seconds": run.get("metrics", {}).get("runtime_seconds"),
                "linear_total_rank_metric": run.get("metrics", {}).get("linear_total_rank_metric"),
                "pc_backend": run.get("metrics", {}).get("pc_backend"),
                "preconditioner_matrix_policy": run.get("metrics", {}).get("preconditioner_matrix_policy"),
                "preconditioner_rebuild_policy": run.get("metrics", {}).get("preconditioner_rebuild_policy"),
                "peak_rss_gib": run.get("metrics", {}).get("peak_rss_gib"),
                "final_accepted_states": run.get("metrics", {}).get("final_accepted_states"),
                "first_progress_elapsed_s": run.get("metrics", {}).get("first_progress_elapsed_s"),
                "progress_file_created": run.get("metrics", {}).get("progress_file_created"),
                "startup_stall_reason": run.get("metrics", {}).get("startup_stall_reason"),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "stage",
                "variant",
                "rank",
                "status",
                "runtime_seconds",
                "linear_total_rank_metric",
                "pc_backend",
                "preconditioner_matrix_policy",
                "preconditioner_rebuild_policy",
                "peak_rss_gib",
                "final_accepted_states",
                "first_progress_elapsed_s",
                "progress_file_created",
                "startup_stall_reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen and compare PETSc preconditioner variants for 3D P4.")
    parser.add_argument("--workflow", type=str, choices=["aij_screen", "bddc_proto", "bddc_sweep"], default="aij_screen")
    parser.add_argument("--mesh-path", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--recycle-failure-report", type=Path, default=DEFAULT_RECYCLE_FAILURE_REPORT)
    parser.add_argument("--step2-report", type=Path, default=DEFAULT_STEP2_REPORT)
    parser.add_argument("--bddc-gate-report", type=Path, default=DEFAULT_BDDC_GATE_REPORT)
    parser.add_argument("--final-report", type=Path, default=DEFAULT_FINAL_REPORT)
    parser.add_argument("--bddc-short-report", type=Path, default=DEFAULT_BDDC_SHORT_REPORT)
    parser.add_argument("--bddc-full-report", type=Path, default=DEFAULT_BDDC_FULL_REPORT)
    parser.add_argument("--bddc-sweep-report", type=Path, default=DEFAULT_BDDC_SWEEP_REPORT)
    parser.add_argument("--screen-ranks", type=int, nargs="+", default=DEFAULT_SCREEN_RANKS)
    parser.add_argument("--scale-ranks", type=int, nargs="+", default=DEFAULT_SCALE_RANKS)
    parser.add_argument("--screen-step-max", type=int, default=DEFAULT_STEP2)
    parser.add_argument("--full-step-max", type=int, default=DEFAULT_FULL_STEP_MAX)
    parser.add_argument("--max-deflation-basis-vectors", type=int, default=16)
    parser.add_argument("--smoke-elem-type", type=str, default="P2")
    parser.add_argument("--elem-type", type=str, default=None)
    parser.add_argument("--stages", nargs="+", choices=DEFAULT_STAGES, default=list(DEFAULT_STAGES))
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-memory-guard-gib", type=float, default=80.0)
    parser.add_argument("--screen-runtime-cutoff-factor", type=float, default=1.10)
    parser.add_argument("--startup-progress-timeout-s", type=float, default=None)
    parser.add_argument("--startup-progress-timeout-p4-s", type=float, default=None)
    args = parser.parse_args()

    selected_stages = set(str(v) for v in args.stages)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if str(args.workflow) == "bddc_proto":
        registry = _variant_registry(include_nongalerkin=False)
        baseline_p2 = _baseline_full_metrics(Path(args.baseline_summary), elem_type="P2")
        baseline_p4 = _baseline_full_metrics(Path(args.baseline_summary), elem_type="P4")
        default_timeout = args.startup_progress_timeout_s
        p2_timeout = 180.0 if default_timeout is None else float(default_timeout)
        p4_timeout = (
            float(args.startup_progress_timeout_p4_s)
            if args.startup_progress_timeout_p4_s is not None
            else (300.0 if default_timeout is None else float(default_timeout))
        )
        requested_variants = None if args.variants is None else tuple(str(v) for v in args.variants)
        prototype = _run_bddc_prototype_workflow(
            registry=registry,
            mesh_path=args.mesh_path,
            out_root=out_root,
            short_step_max=int(args.screen_step_max),
            full_step_max=int(args.full_step_max),
            max_deflation_basis_vectors=int(args.max_deflation_basis_vectors),
            full_memory_guard_gib=float(args.full_memory_guard_gib),
            reuse_existing=bool(args.reuse_existing),
            startup_progress_timeout_p2_s=float(p2_timeout),
            startup_progress_timeout_p4_s=float(p4_timeout),
            requested_variants=requested_variants,
        )
        summary_payload = {
            "timestamp": _utc_now(),
            "workflow": "bddc_proto",
            "mesh_path": _mesh_label(args.mesh_path),
            "screen_step_max": int(args.screen_step_max),
            "full_step_max": int(args.full_step_max),
            "bddc_prototype": prototype,
            "baseline_reuse": {
                "summary_path": _path_label(args.baseline_summary),
                "report_path": _path_label(args.baseline_report),
                "P2": baseline_p2,
                "P4": baseline_p4,
            },
        }
        summary_json = out_root / "summary.json"
        summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_summary_csv(out_root / "summary.csv", summary_payload=summary_payload)
        Path(args.bddc_short_report).write_text(
            "\n".join(_bddc_short_report_lines(mesh_path=args.mesh_path, summary_payload=summary_payload)) + "\n",
            encoding="utf-8",
        )
        Path(args.bddc_full_report).write_text(
            "\n".join(
                _bddc_full_report_lines(
                    mesh_path=args.mesh_path,
                    summary_payload=summary_payload,
                    baseline_summary_path=Path(args.baseline_summary),
                    baseline_report_path=Path(args.baseline_report),
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return

    if str(args.workflow) == "bddc_sweep":
        registry = _bddc_sweep_registry(include_adaptive=bool(_petsc_has_external_package("mumps")))
        summary = _run_bddc_sweep_workflow(
            registry=registry,
            mesh_path=args.mesh_path,
            out_root=out_root,
            linear_tolerance=1.0e-5,
            linear_max_iter=500,
            max_deflation_basis_vectors=int(args.max_deflation_basis_vectors),
            reuse_existing=bool(args.reuse_existing),
            requested_variants=(None if args.variants is None else tuple(str(v) for v in args.variants)),
        )
        summary_payload = {
            "timestamp": _utc_now(),
            "workflow": "bddc_sweep",
            "mesh_path": _mesh_label(args.mesh_path),
            "bddc_sweep": summary,
        }
        summary_json = out_root / "summary.json"
        summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_summary_csv(out_root / "summary.csv", summary_payload=summary_payload)
        Path(args.bddc_sweep_report).write_text(
            "\n".join(_bddc_sweep_report_lines(mesh_path=args.mesh_path, summary_payload=summary_payload)) + "\n",
            encoding="utf-8",
        )
        return

    option_smokes: dict[str, Any] = {}
    bddc_runtime_smokes: dict[str, Any] = {}
    screening: dict[str, Any] = {}
    promoted_aij_variants: list[str] = []
    drift_limits: dict[str, float] = {}
    scaling: dict[str, Any] = {}
    bddc_gate: dict[str, Any] = {"status": "not_run", "run": {}, "eligible_for_full_trajectory": False}
    best_aij_name: str | None = None
    full_compare: dict[str, Any] = {}

    registry = _variant_registry(include_nongalerkin=False)

    def _run_aij_option_smokes() -> dict[str, Any]:
        local_registry = _variant_registry(include_nongalerkin=False)
        result: dict[str, Any] = {}
        for name in ("hypre_current", "hypre_lagged_current", "hypre_lagged_pmis", "gamg_lagged_lowmem"):
            result[name] = _run_option_smoke(
                variant=local_registry[name],
                mesh_path=args.mesh_path,
                out_root=out_root,
                max_deflation_basis_vectors=int(args.max_deflation_basis_vectors),
                smoke_elem_type=str(args.smoke_elem_type),
                reuse_existing=bool(args.reuse_existing),
            )
        complexity_variant = _variant_registry(include_nongalerkin=True)["hypre_lagged_complexity"]
        complexity_smoke = _run_option_smoke(
            variant=complexity_variant,
            mesh_path=args.mesh_path,
            out_root=out_root,
            max_deflation_basis_vectors=int(args.max_deflation_basis_vectors),
            smoke_elem_type=str(args.smoke_elem_type),
            reuse_existing=bool(args.reuse_existing),
        )
        if complexity_smoke.get("status") == "completed":
            nonlocal_registry = _variant_registry(include_nongalerkin=True)
            result["hypre_lagged_complexity"] = complexity_smoke
            return result, nonlocal_registry
        result["hypre_lagged_complexity"] = _run_option_smoke(
            variant=local_registry["hypre_lagged_complexity"],
            mesh_path=args.mesh_path,
            out_root=out_root,
            max_deflation_basis_vectors=int(args.max_deflation_basis_vectors),
            smoke_elem_type=str(args.smoke_elem_type),
            reuse_existing=bool(args.reuse_existing),
        )
        return result, local_registry

    need_smoke = bool(selected_stages & {"smoke", "screen", "bddc_gate", "full_compare"})
    need_screen = bool(selected_stages & {"screen", "bddc_gate", "full_compare"})
    need_bddc_gate = bool(selected_stages & {"bddc_gate", "full_compare"})
    need_full_compare = bool("full_compare" in selected_stages)

    if need_smoke:
        option_smokes, registry = _run_aij_option_smokes()
        option_smokes["bddc"] = {
            "status": "covered_by_runtime_smokes",
            "note": "BDDC is validated by dedicated runtime smokes plus serial/MPI pytest coverage.",
        }
        bddc_runtime_smokes = _run_bddc_runtime_smokes(
            variant=registry["bddc"],
            mesh_path=args.mesh_path,
            out_root=out_root,
            max_deflation_basis_vectors=int(args.max_deflation_basis_vectors),
            reuse_existing=bool(args.reuse_existing),
        )

    if need_screen:
        baseline_name = "hypre_current"
        baseline_variant = registry[baseline_name]
        screening[baseline_name] = {
            "variant_category": baseline_variant.category,
            "description": baseline_variant.description,
            "runs": _screen_variant(
                variant=baseline_variant,
                ranks=tuple(int(v) for v in args.screen_ranks),
                mesh_path=args.mesh_path,
                step_max=int(args.screen_step_max),
                out_root=out_root,
                max_deflation_basis_vectors=int(args.max_deflation_basis_vectors),
                wall_time_limits_s=None,
                reuse_existing=bool(args.reuse_existing),
            ),
        }
        screening[baseline_name]["status"] = (
            "completed"
            if all(run.get("status") == "completed" for run in screening[baseline_name]["runs"].values())
            else "failed"
        )

        screen_wall_time_limits_s: dict[int, float] = {}
        if screening[baseline_name]["status"] == "completed":
            for ranks_value in args.screen_ranks:
                rank_key = str(int(ranks_value))
                baseline_metrics = screening[baseline_name]["runs"][rank_key]["metrics"]
                screen_wall_time_limits_s[int(ranks_value)] = (
                    float(baseline_metrics["runtime_seconds"]) * float(args.screen_runtime_cutoff_factor)
                )

        for name in ("hypre_lagged_current", "hypre_lagged_pmis", "hypre_lagged_complexity", "gamg_lagged_lowmem"):
            variant = registry[name]
            screening[name] = {
                "variant_category": variant.category,
                "description": variant.description,
                "runs": _screen_variant(
                    variant=variant,
                    ranks=tuple(int(v) for v in args.screen_ranks),
                    mesh_path=args.mesh_path,
                    step_max=int(args.screen_step_max),
                    out_root=out_root,
                    max_deflation_basis_vectors=int(args.max_deflation_basis_vectors),
                    wall_time_limits_s=screen_wall_time_limits_s,
                    reuse_existing=bool(args.reuse_existing),
                ),
            }
            screening[name]["status"] = (
                "completed"
                if all(run.get("status") == "completed" for run in screening[name]["runs"].values())
                else "failed"
            )

        promoted_aij_variants, drift_limits = _promote_aij_candidates(screening)
        scaling_variant_names = ["hypre_current", *promoted_aij_variants]
        scaling = _run_scaling(
            variant_names=scaling_variant_names,
            registry=registry,
            ranks=tuple(int(v) for v in args.scale_ranks),
            mesh_path=args.mesh_path,
            step_max=int(args.screen_step_max),
            out_root=out_root,
            max_deflation_basis_vectors=int(args.max_deflation_basis_vectors),
            reuse_existing=bool(args.reuse_existing),
        )
        best_aij_name = _select_best_scaled_candidate(scaling)

    if need_bddc_gate:
        if not need_screen:
            raise RuntimeError("bddc_gate/full_compare stages require screening data")
        if bddc_runtime_smokes.get("status") != "completed":
            bddc_gate = {
                "status": "runtime_smoke_failed",
                "run": {},
                "eligible_for_full_trajectory": False,
            }
        else:
            bddc_rank8 = _run_case(
                variant=registry["bddc"],
                ranks=8,
                mesh_path=args.mesh_path,
                step_max=int(args.screen_step_max),
                out_dir=out_root / "bddc_gate" / "rank8_step2",
                guard_limit_gib=None,
                max_deflation_basis_vectors=int(args.max_deflation_basis_vectors),
                reuse_existing=bool(args.reuse_existing),
            )
            bddc_gate = {
                "status": bddc_rank8.get("status"),
                "run": bddc_rank8,
                "eligible_for_full_trajectory": False,
            }
            if bddc_rank8.get("status") == "completed" and best_aij_name is not None:
                best_aij_rank8 = scaling[best_aij_name]["runs"]["8"]["metrics"]
                bddc_metrics = bddc_rank8["metrics"]
                eligible = (
                    int(bddc_metrics["final_accepted_states"]) >= int(best_aij_rank8["final_accepted_states"])
                    and float(bddc_metrics["runtime_seconds"]) <= 1.15 * float(best_aij_rank8["runtime_seconds"])
                    and (
                        float(best_aij_rank8.get("peak_rss_gib", 0.0)) <= 0.0
                        or float(bddc_metrics.get("peak_rss_gib", 0.0)) <= 1.10 * float(best_aij_rank8["peak_rss_gib"])
                    )
                )
                bddc_gate["eligible_for_full_trajectory"] = bool(eligible)

    baseline_p4 = _baseline_full_metrics(Path(args.baseline_summary))
    if need_full_compare:
        if best_aij_name is None:
            raise RuntimeError("full_compare requires a selected best AIJ candidate")
        full_compare = _run_full_compare(
            registry=registry,
            best_aij_name=best_aij_name,
            bddc_gate=bddc_gate,
            mesh_path=args.mesh_path,
            out_root=out_root,
            full_step_max=int(args.full_step_max),
            full_memory_guard_gib=float(args.full_memory_guard_gib),
            max_deflation_basis_vectors=int(args.max_deflation_basis_vectors),
            reuse_existing=bool(args.reuse_existing),
            baseline_metrics=baseline_p4,
        )

    summary_payload = {
        "timestamp": _utc_now(),
        "mesh_path": _mesh_label(args.mesh_path),
        "stages": sorted(selected_stages),
        "screen_ranks": [int(v) for v in args.screen_ranks],
        "scale_ranks": [int(v) for v in args.scale_ranks],
        "screen_step_max": int(args.screen_step_max),
        "full_step_max": int(args.full_step_max),
        "option_smokes": option_smokes,
        "bddc_runtime_smokes": bddc_runtime_smokes,
        "screening": screening,
        "promoted_aij_variants": promoted_aij_variants,
        "screen_drift_limits": drift_limits,
        "scaling": scaling,
        "best_aij_name": best_aij_name,
        "bddc_gate": bddc_gate,
        "full_compare": full_compare,
        "baseline_reuse": {
            "summary_path": _path_label(args.baseline_summary),
            "report_path": _path_label(args.baseline_report),
            "P4": baseline_p4,
        },
        "recycle_failure_report": _path_label(args.recycle_failure_report),
    }

    summary_json = out_root / "summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary_csv(out_root / "summary.csv", summary_payload=summary_payload)

    if selected_stages & {"screen", "bddc_gate", "full_compare"}:
        step2_lines = _step2_report_lines(mesh_path=args.mesh_path, summary_payload=summary_payload)
        Path(args.step2_report).write_text("\n".join(step2_lines) + "\n", encoding="utf-8")

    if selected_stages & {"bddc_gate", "full_compare"}:
        gate_lines = _bddc_gate_report_lines(mesh_path=args.mesh_path, summary_payload=summary_payload)
        Path(args.bddc_gate_report).write_text("\n".join(gate_lines) + "\n", encoding="utf-8")

    if "full_compare" in selected_stages:
        final_lines = _final_report_lines(
            mesh_path=args.mesh_path,
            summary_payload=summary_payload,
            baseline_summary_path=Path(args.baseline_summary),
            baseline_report_path=Path(args.baseline_report),
            recycle_failure_report=Path(args.recycle_failure_report),
        )
        Path(args.final_report).write_text("\n".join(final_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
