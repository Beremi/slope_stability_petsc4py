"""Resolve case TOMLs, solver profiles, and mesh assets into a run model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .case_schema import RunCaseConfig, load_run_case_config
from .profiles import ResolvedPcVariant, native_linear_algorithm_selector, pc_variant_from_backend
from petsc_ssr.problem_asset_runtime import ResolvedAsset, resolve_problem_asset_from_config


@dataclass(frozen=True, slots=True)
class ResolvedCaseModel:
    """Concrete public case/profile/asset model before PETSc state exists."""

    source: Path
    config: RunCaseConfig
    asset: ResolvedAsset
    pc_policy: ResolvedPcVariant

    def validation_payload(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "case": cfg.problem.name,
            "analysis": cfg.problem.analysis,
            "dimension": self.asset.dimension,
            "asset": self.asset.asset_name,
            "mesh_variant": self.asset.variant_name,
            "mesh_path": None if self.asset.mesh_path is None else str(self.asset.mesh_path),
            "element": cfg.problem.elem_type,
            "refine_levels": cfg.problem.refine_levels,
            "partitioner": cfg.problem.partitioner,
            "linear_profile": cfg.linear_solver.profile,
            "linear_algorithm": cfg.linear_solver.algorithm or None,
            "native_linear_algorithm": self.native_linear_algorithm(),
            "pc_backend": cfg.linear_solver.pc_backend,
            "pc_variant": self.pc_policy.variant,
            "requested_pc_variant": self.pc_policy.requested_variant,
            "pc_variant_fallback_reason": self.pc_policy.fallback_reason,
            "continuation_profile": cfg.continuation.profile or None,
            "continuation_algorithm": cfg.continuation.algorithm or None,
            "newton_profile": cfg.newton.profile or None,
            "newton_algorithm": cfg.newton.algorithm or None,
            "seepage_profile": cfg.seepage.profile or None,
            "resolved_world_size": cfg.linear_solver.resolved_world_size,
            "resolved_pmg": _resolved_pmg_payload(cfg),
            "output": _output_payload(cfg),
        }

    def explanation_payload(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "case": cfg.problem.name,
            "asset": self.asset.asset_name,
            "mesh_variant": self.asset.variant_name,
            "mesh_path": None if self.asset.mesh_path is None else str(self.asset.mesh_path),
            "element": cfg.problem.elem_type,
            "analysis": cfg.problem.analysis,
            "continuation": _continuation_payload(cfg),
            "newton": _newton_payload(cfg),
            "seepage": _seepage_payload(cfg),
            "linear": {
                "profile": cfg.linear_solver.profile,
                "algorithm": cfg.linear_solver.algorithm,
                "native_algorithm": self.native_linear_algorithm(),
                "solver_type": cfg.linear_solver.solver_type,
                "ksp_type": cfg.linear_solver.ksp_type,
                "norm_type": cfg.linear_solver.norm_type,
                "deflation": cfg.linear_solver.deflation,
                "deflation_solver": cfg.linear_solver.deflation_solver,
                "rtol": cfg.linear_solver.tolerance,
                "max_it": cfg.linear_solver.max_iterations,
                "pc_backend": cfg.linear_solver.pc_backend,
                "pc_variant": self.pc_policy.variant,
                "requested_pc_variant": self.pc_policy.requested_variant,
                "pc_variant_fallback_reason": self.pc_policy.fallback_reason,
                "petsc_options_file": None
                if cfg.linear_solver.pmg_options_file is None
                else str(cfg.linear_solver.pmg_options_file),
            },
            "resolved_pmg": {
                "world_size": cfg.linear_solver.resolved_world_size,
                **_resolved_pmg_payload(cfg),
            },
            "output": _output_payload(cfg),
        }

    def native_linear_algorithm(self) -> str:
        cfg = self.config
        return native_linear_algorithm_selector(
            cfg.linear_solver.algorithm or cfg.linear_solver.solver_type,
            pc_variant=self.pc_policy.variant,
            deflation=bool(cfg.linear_solver.deflation),
        )


def resolve_case_model(case_toml: str | Path) -> ResolvedCaseModel:
    source = Path(case_toml).resolve()
    cfg = load_run_case_config(source).validate()
    asset = resolve_problem_asset_from_config(cfg)
    return ResolvedCaseModel(
        source=source,
        config=cfg,
        asset=asset,
        pc_policy=resolved_pc_policy(cfg),
    )


def validate_case_payload(case_toml: str | Path) -> dict[str, Any]:
    return resolve_case_model(case_toml).validation_payload()


def explain_case_payload(case_toml: str | Path) -> dict[str, Any]:
    return resolve_case_model(case_toml).explanation_payload()


def resolved_pc_policy(cfg: RunCaseConfig) -> ResolvedPcVariant:
    element_degree = int(str(cfg.problem.elem_type).strip().upper()[1:])
    supported = (
        ("gamg", "pmg", "none")
        if str(cfg.problem.analysis).strip().lower() == "seepage"
        else ("gamg", "bddc", "fetidp", "pmg", "none")
    )
    return pc_variant_from_backend(
        cfg.linear_solver.pc_backend,
        element_degree=element_degree,
        supported=supported,
    )


def _resolved_pmg_payload(cfg: RunCaseConfig) -> dict[str, Any]:
    return {
        "rank_policy": cfg.linear_solver.pmg_rank_policy,
        "apply_backend": cfg.linear_solver.pmg_apply_backend,
        "p2_active_ranks": cfg.linear_solver.pmg_shell_p2_active_ranks,
        "p1_active_ranks": cfg.linear_solver.pmg_shell_p1_active_ranks,
        "p2_policy": cfg.linear_solver.pmg_shell_p2_rank_policy,
        "p1_policy": cfg.linear_solver.pmg_shell_p1_rank_policy,
        "subcomm_type": cfg.linear_solver.pmg_shell_subcomm_type,
        "fine_ksp_max_it": cfg.linear_solver.pmg_shell_fine_ksp_max_it,
        "p2_ksp_max_it": cfg.linear_solver.pmg_shell_p2_ksp_max_it,
        "smoother_ksp_type": cfg.linear_solver.pmg_smoother_ksp_type,
        "smoother_pc_type": cfg.linear_solver.pmg_smoother_pc_type,
        "smoother_max_it": cfg.linear_solver.pmg_smoother_max_it,
        "coarse_pc_type": cfg.linear_solver.pmg_coarse_pc_type,
        "coarse_lu_max_dofs": cfg.linear_solver.pmg_coarse_lu_max_dofs,
        "coarse_redundant_group_size": cfg.linear_solver.pmg_coarse_redundant_group_size,
        "coarse_gamg_aggressive_square_graph": cfg.linear_solver.pmg_coarse_gamg_aggressive_square_graph,
        "coarse_telescope_active_ranks": cfg.linear_solver.pmg_coarse_telescope_active_ranks,
        "coarse_telescope_subcomm_type": cfg.linear_solver.pmg_coarse_telescope_subcomm_type,
        "coarse_telescope_ksp_type": cfg.linear_solver.pmg_coarse_telescope_ksp_type,
        "coarse_telescope_ksp_rtol": cfg.linear_solver.pmg_coarse_telescope_ksp_rtol,
        "coarse_telescope_ksp_max_it": cfg.linear_solver.pmg_coarse_telescope_ksp_max_it,
        "coarse_telescope_pc_type": cfg.linear_solver.pmg_coarse_telescope_pc_type,
        "p2_telescope_active_ranks": cfg.linear_solver.pmg_p2_telescope_active_ranks,
        "p2_telescope_subcomm_type": cfg.linear_solver.pmg_p2_telescope_subcomm_type,
        "p2_telescope_ksp_type": cfg.linear_solver.pmg_p2_telescope_ksp_type,
        "p2_telescope_ksp_rtol": cfg.linear_solver.pmg_p2_telescope_ksp_rtol,
        "p2_telescope_ksp_max_it": cfg.linear_solver.pmg_p2_telescope_ksp_max_it,
        "p2_telescope_pc_type": cfg.linear_solver.pmg_p2_telescope_pc_type,
    }


def _output_payload(cfg: RunCaseConfig) -> dict[str, Any]:
    return {
        "preset": cfg.export.preset,
        "write_solution": bool(cfg.export.write_solution_vtu),
        "write_history": bool(cfg.export.write_history_json),
    }


def _continuation_payload(cfg: RunCaseConfig) -> dict[str, Any] | None:
    if not cfg.continuation.profile:
        return None
    return {
        "profile": cfg.continuation.profile,
        "algorithm": cfg.continuation.algorithm,
        "method": cfg.continuation.method,
        "omega_max": cfg.continuation.omega_max,
        "step_max": cfg.continuation.step_max,
    }


def _newton_payload(cfg: RunCaseConfig) -> dict[str, Any] | None:
    if not cfg.newton.profile:
        return None
    return {
        "profile": cfg.newton.profile,
        "algorithm": cfg.newton.algorithm,
        "stopping_criterion": cfg.newton.stopping_criterion,
        "stopping_tol": cfg.newton.stopping_tol,
        "line_search": cfg.newton.line_search,
    }


def _seepage_payload(cfg: RunCaseConfig) -> dict[str, Any] | None:
    if not cfg.seepage.profile:
        return None
    return {
        "profile": cfg.seepage.profile,
        "linear_tolerance": cfg.seepage.linear_tolerance,
        "linear_max_iter": cfg.seepage.linear_max_iter,
        "nonlinear_max_iter": cfg.seepage.nonlinear_max_iter,
    }


__all__ = [
    "ResolvedCaseModel",
    "explain_case_payload",
    "resolve_case_model",
    "resolved_pc_policy",
    "validate_case_payload",
]
