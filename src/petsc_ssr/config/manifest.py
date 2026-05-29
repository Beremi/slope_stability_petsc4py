"""Resolved run and environment manifest builders.

These helpers deliberately avoid petsc4py imports so schema, suite, and dry-run
tooling can inspect artifact contracts without constructing native PETSc state.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from petsc_ssr.options import SsrOptions
from petsc_ssr.problem import ProblemSpec
from petsc_ssr.runtime.results import run_artifact_manifest


ENGINE_ROOT = Path(__file__).resolve().parents[3]

ENVIRONMENT_MANIFEST_KIND = "petsc_ssr_environment_manifest"
RUN_COMMAND_MANIFEST_KIND = "petsc_ssr_run_command"
RESOLVED_CONFIG_KIND = "petsc_ssr_resolved_config"
RESOLVED_RUN_MANIFEST_KIND = "petsc_ssr_resolved_run_manifest"
MANIFEST_SCHEMA_VERSION = 1


def build_environment_manifest(
    *,
    mpi_size: int,
    env: Mapping[str, str | None] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the portable runtime environment manifest for one concrete run."""

    env_map = os.environ if env is None else env
    return {
        "kind": ENVIRONMENT_MANIFEST_KIND,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mpi": {
            "size": int(mpi_size),
        },
        "env": {
            key: env_map.get(key)
            for key in (
                "PETSC_DIR",
                "PETSC_ARCH",
                "OMP_NUM_THREADS",
                "SLURM_JOB_ID",
                "SLURM_NTASKS",
            )
            if env_map.get(key) is not None
        },
        "git": build_git_manifest(repo_root or ENGINE_ROOT),
    }


def build_run_command_manifest(
    *,
    output_dir: str | Path,
    mpi_size: int,
    argv: Sequence[str],
    mode: str,
    entrypoint: str,
    resolved_run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the portable command provenance for one direct run invocation."""

    output_path = Path(output_dir)
    linear = dict(resolved_run_manifest.get("linear", {}))
    artifacts = {
        **run_artifact_manifest(output_path),
        **dict(resolved_run_manifest.get("artifacts", {})),
    }
    resolved_profile = {
        section: resolved_run_manifest[section]
        for section in ("continuation", "newton", "linear", "pmg", "seepage", "output")
        if section in resolved_run_manifest
    }
    return {
        "kind": RUN_COMMAND_MANIFEST_KIND,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "mode": str(mode),
        "entrypoint": str(entrypoint),
        "argv": [str(token) for token in argv],
        "command": [str(entrypoint), *[str(token) for token in argv]],
        "case": resolved_run_manifest.get("case"),
        "profile": linear.get("profile", ""),
        "ranks": int(mpi_size),
        "output_dir": str(output_path),
        "resolved_profile": resolved_profile,
        "artifacts": artifacts,
    }


def build_resolved_run_manifest(
    problem: ProblemSpec,
    options: SsrOptions,
    *,
    output_dir: str | Path,
    mpi_size: int,
) -> dict[str, Any]:
    """Build the resolved case/profile/artifact manifest for one concrete run."""

    output_path = Path(output_dir)
    artifacts = run_artifact_manifest(output_path)
    manifest: dict[str, Any] = {
        "kind": RESOLVED_RUN_MANIFEST_KIND,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "case": problem.name,
        "mpi": {
            "size": int(mpi_size),
        },
        "mesh": {
            "path": str(problem.mesh_path),
            "dimension": int(problem.dimension),
            "element_degree": int(problem.element_degree),
            "refine_levels": int(problem.refine_levels),
        },
        "continuation": {
            "profile": problem.metadata.get("continuation_profile", ""),
            "algorithm": options.continuation_algorithm or problem.metadata.get("continuation_algorithm", ""),
            "method": options.continuation_method,
            "step_max": int(options.continuation_step_max),
            "omega_max": float(options.omega_max),
        },
        "newton": {
            "profile": problem.metadata.get("newton_profile", ""),
            "algorithm": options.newton_algorithm or problem.metadata.get("newton_algorithm", ""),
            "max_it": int(options.newton_max_it),
            "stopping_criterion": options.newton_stopping_criterion,
            "stopping_tol": float(options.newton_stopping_tol),
            "line_search": bool(options.line_search),
        },
        "linear": {
            "profile": options.profile_name or problem.metadata.get("linear_profile", ""),
            "algorithm": problem.metadata.get("linear_algorithm", ""),
            "native_algorithm": options.linear_algorithm,
            "rtol": float(options.linear.rtol),
            "max_it": int(options.linear.max_it),
            "deflation": bool(options.linear.deflation),
            "deflation_solver": options.linear.deflation_solver,
            "pc_backend": problem.metadata.get("pc_backend", ""),
            "pc_variant": options.pc_variant,
            "requested_pc_variant": problem.metadata.get("requested_pc_variant", options.pc_variant),
            "pc_variant_fallback_reason": problem.metadata.get("pc_variant_fallback_reason"),
        },
        "pmg": _resolved_pmg_section(options, problem.metadata),
        "output": {
            "preset": problem.metadata.get("output_preset", "standard"),
            "write_solution": _metadata_bool(problem.metadata.get("write_solution_vtu"), default=True),
            "write_history": _metadata_bool(problem.metadata.get("write_history_json"), default=True),
        },
        "compatibility": _compatibility_section(problem.metadata),
        "artifacts": {
            **artifacts,
            "resolved_options": artifacts["resolved_options_txt"],
            "continuation_curve": artifacts["continuation_curve_csv"],
            "stdout": artifacts["stdout_txt"],
            "petsc_log": artifacts["petsc_log_txt"],
            "options_view": artifacts["options_view_txt"],
            "options_left": artifacts["options_left_txt"],
            "native_problem_manifest": problem.metadata.get("native_problem_manifest") or artifacts["native_problem_manifest_json"],
            "mechanics_bc_labels_csv": problem.metadata.get("mechanics_bc_labels_csv"),
            "mechanics_bc_nodes_csv": problem.metadata.get("mechanics_bc_nodes_csv"),
            "mechanics_neumann_labels_csv": problem.metadata.get("mechanics_neumann_labels_csv"),
            "seepage_boundary_labels_csv": problem.metadata.get("seepage_boundary_labels_csv"),
            "seepage_pressure_csv": problem.metadata.get("seepage_pressure_csv"),
        },
    }
    seepage = _resolved_seepage_section(problem.metadata)
    if seepage is not None:
        manifest["seepage"] = seepage
    return manifest


def build_resolved_config(
    problem: ProblemSpec,
    options: SsrOptions,
    *,
    output_dir: str | Path,
    mpi_size: int,
) -> dict[str, Any]:
    """Build the concrete case/profile/options model for one run."""

    metadata = problem.metadata
    resolved: dict[str, Any] = {
        "resolved": {
            "kind": RESOLVED_CONFIG_KIND,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "source_case": metadata.get("case_config"),
        },
        "case": {
            "id": problem.name,
            "analysis": options.analysis,
            "asset": metadata.get("asset"),
            "mesh_variant": metadata.get("mesh_variant"),
            "seepage_coupled": metadata.get("seepage_coupled"),
        },
        "mpi": {
            "size": int(mpi_size),
        },
        "mesh": {
            "path": str(problem.mesh_path),
            "dimension": int(problem.dimension),
            "element_degree": int(problem.element_degree),
            "element": metadata.get("elem_type"),
            "refine_levels": int(problem.refine_levels),
        },
        "continuation": {
            "profile": metadata.get("continuation_profile"),
            "profile_description": metadata.get("continuation_profile_description"),
            "algorithm": options.continuation_algorithm or metadata.get("continuation_algorithm"),
            "method": options.continuation_method,
            "predictor": options.continuation_predictor,
            "omega_step_controller": options.omega_step_controller,
            "omega_max": float(options.omega_max),
            "lambda_init": float(options.lambda_init),
            "d_lambda_init": float(options.d_lambda_init),
            "d_lambda_min": float(options.d_lambda_min),
            "d_lambda_diff_scaled_min": float(options.d_lambda_diff_scaled_min),
            "lambda_ell": float(options.lambda_ell),
            "d_t_min": float(options.d_t_min),
            "d_omega_ini_scale": float(options.d_omega_ini_scale),
            "step_max": int(options.continuation_step_max),
        },
        "newton": {
            "profile": metadata.get("newton_profile"),
            "profile_description": metadata.get("newton_profile_description"),
            "algorithm": options.newton_algorithm or metadata.get("newton_algorithm"),
            "max_it": int(options.newton_max_it),
            "rtol": float(options.newton_rtol),
            "stopping_criterion": options.newton_stopping_criterion,
            "stopping_tol": float(options.newton_stopping_tol),
            "init_stopping_criterion": options.init_newton_stopping_criterion,
            "init_stopping_tol": float(options.init_newton_stopping_tol),
            "it_damp_max": int(options.it_damp_max),
            "r_min": float(options.r_min),
            "damping_min": float(options.damping_min),
            "line_search": bool(options.line_search),
        },
        "linear": {
            "profile": options.profile_name or metadata.get("linear_profile"),
            "profile_description": metadata.get("profile_description"),
            "algorithm": metadata.get("linear_algorithm"),
            "native_algorithm": options.linear_algorithm,
            "rtol": float(options.linear.rtol),
            "max_it": int(options.linear.max_it),
            "ksp_type": options.linear.ksp_type,
            "norm_type": options.linear.norm_type,
            "deflation": bool(options.linear.deflation),
            "deflation_solver": options.linear.deflation_solver,
            "pc_backend": metadata.get("pc_backend"),
            "pc_variant": options.pc_variant,
            "requested_pc_variant": metadata.get("requested_pc_variant", options.pc_variant),
            "pc_variant_fallback_reason": metadata.get("pc_variant_fallback_reason"),
            "partitioner": options.partitioner,
        },
        "pmg": _resolved_pmg_section(options, metadata, include_options_file=True),
        "petsc": {
            "extra_options": list(options.petsc_options),
        },
        "output": {
            "preset": metadata.get("output_preset", "standard"),
            "write_solution": _metadata_bool(metadata.get("write_solution_vtu"), default=True),
            "write_history": _metadata_bool(metadata.get("write_history_json"), default=True),
        },
        "compatibility": _compatibility_section(metadata),
        "artifacts": build_resolved_run_manifest(problem, options, output_dir=output_dir, mpi_size=mpi_size)["artifacts"],
    }
    seepage = _resolved_seepage_section(metadata)
    if seepage is not None:
        resolved["seepage"] = seepage
    return resolved


def _compatibility_section(metadata: Mapping[str, Any]) -> dict[str, Any]:
    label_table = metadata.get("mechanics_bc_labels_csv")
    coordinate_table = metadata.get("mechanics_bc_nodes_csv")
    debug_coordinate = bool(metadata.get("debug_coordinate_bc_table", False))
    pressure_csv = metadata.get("seepage_pressure_csv")
    pressure_source = str(metadata.get("seepage_pressure_source", "") or "").strip()
    if label_table and coordinate_table:
        mechanics_source = "label_table_preferred_coordinate_debug_available"
    elif label_table:
        mechanics_source = "label_table"
    elif coordinate_table:
        mechanics_source = "coordinate_debug_table"
    else:
        mechanics_source = "none"
    return {
        "mechanics_constraint_source": mechanics_source,
        "mechanics_label_table_active": bool(label_table),
        "mechanics_coordinate_debug_table_active": bool(coordinate_table and debug_coordinate),
        "debug_coordinate_bc_table": debug_coordinate,
        "mechanics_label_constraints_csv": label_table,
        "mechanics_coordinate_constraints_csv": coordinate_table,
        "seepage_pressure_coordinate_bridge_active": bool(pressure_csv),
        "seepage_pressure_source": pressure_source or ("none" if not pressure_csv else ""),
        "seepage_pressure_csv": pressure_csv,
    }


def _resolved_seepage_section(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    profile = str(metadata.get("seepage_profile", "") or "").strip()
    coupled = bool(metadata.get("seepage_coupled", False))
    if not profile and not coupled:
        return None
    section: dict[str, Any] = {
        "profile": profile,
        "profile_description": str(metadata.get("seepage_profile_description", "") or ""),
        "coupled": coupled,
    }
    if metadata.get("seepage_linear_tolerance") is not None:
        section["linear_tolerance"] = float(metadata["seepage_linear_tolerance"])
    if metadata.get("seepage_linear_max_iter") is not None:
        section["linear_max_iter"] = int(metadata["seepage_linear_max_iter"])
    if metadata.get("seepage_nonlinear_max_iter") is not None:
        section["nonlinear_max_iter"] = int(metadata["seepage_nonlinear_max_iter"])
    return section


def _resolved_pmg_section(
    options: SsrOptions,
    metadata: Mapping[str, Any],
    *,
    include_options_file: bool = False,
) -> dict[str, Any]:
    section: dict[str, Any] = {}
    if include_options_file:
        section["options_file"] = str(options.pmg.options_file)
    section.update(
        {
            "rank_policy": metadata.get("pmg_rank_policy"),
            "apply_backend": options.pmg.apply_backend,
            "p2_active_ranks": int(options.pmg.p2_active_ranks),
            "p1_active_ranks": int(options.pmg.p1_active_ranks),
            "p2_rank_policy": metadata.get("pmg_shell_p2_rank_policy"),
            "p1_rank_policy": metadata.get("pmg_shell_p1_rank_policy"),
            "subcomm_type": options.pmg.subcomm_type,
            "fine_ksp_max_it": int(options.pmg.fine_ksp_max_it),
            "p2_ksp_max_it": int(options.pmg.p2_ksp_max_it),
            "smoother_ksp_type": options.pmg.smoother_ksp_type,
            "smoother_pc_type": options.pmg.smoother_pc_type,
            "smoother_max_it": int(options.pmg.smoother_max_it),
            "coarse_pc_type": options.pmg.coarse_pc_type,
            "coarse_lu_max_dofs": int(options.pmg.coarse_lu_max_dofs),
            "coarse_redundant_group_size": int(options.pmg.coarse_redundant_group_size),
            "coarse_gamg_aggressive_square_graph": bool(options.pmg.coarse_gamg_aggressive_square_graph),
            "coarse_telescope_active_ranks": int(options.pmg.coarse_telescope_active_ranks),
            "coarse_telescope_subcomm_type": options.pmg.coarse_telescope_subcomm_type,
            "coarse_telescope_ksp_type": options.pmg.coarse_telescope_ksp_type,
            "coarse_telescope_ksp_rtol": float(options.pmg.coarse_telescope_ksp_rtol),
            "coarse_telescope_ksp_max_it": int(options.pmg.coarse_telescope_ksp_max_it),
            "coarse_telescope_pc_type": options.pmg.coarse_telescope_pc_type,
            "p2_telescope_active_ranks": int(options.pmg.p2_telescope_active_ranks),
            "p2_telescope_subcomm_type": options.pmg.p2_telescope_subcomm_type,
            "p2_telescope_ksp_type": options.pmg.p2_telescope_ksp_type,
            "p2_telescope_ksp_rtol": float(options.pmg.p2_telescope_ksp_rtol),
            "p2_telescope_ksp_max_it": int(options.pmg.p2_telescope_ksp_max_it),
            "p2_telescope_pc_type": options.pmg.p2_telescope_pc_type,
            "p1_pc_type": options.pmg.p1_pc_type,
            "p1_redundant_number": options.pmg.p1_redundant_number,
            "p1_redundant_ksp_type": options.pmg.p1_redundant_ksp_type,
            "p1_redundant_ksp_rtol": options.pmg.p1_redundant_ksp_rtol,
            "p1_redundant_ksp_max_it": options.pmg.p1_redundant_ksp_max_it,
            "p1_redundant_pc_type": options.pmg.p1_redundant_pc_type,
        }
    )
    return section


def _metadata_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def dumps_resolved_config_toml(resolved: Mapping[str, Any]) -> str:
    """Serialize the resolved config model as simple TOML without extra deps."""

    lines: list[str] = []
    for section, payload in resolved.items():
        if not isinstance(payload, Mapping):
            continue
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        for key, value in payload.items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def build_git_manifest(repo_root: str | Path) -> dict[str, Any]:
    """Capture lightweight Git provenance without failing outside a worktree."""

    root = Path(repo_root)

    def _git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": (_git(["status", "--short"]) or "") != "",
    }


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
