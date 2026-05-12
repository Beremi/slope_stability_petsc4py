"""Asset-backed case execution routes used by the public config CLI."""

from __future__ import annotations

import inspect
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from petsc4py import PETSc

from slope_stability.core.run_config import RunCaseConfig
from slope_stability.export import write_debug_bundle_h5, write_history_csv_tables, write_history_json, write_vtu
from slope_stability.postprocess import build_field_exports, rebuild_case_mesh, validate_case_mesh_alignment
from slope_stability.problem_asset_runtime import resolve_problem_asset_from_config

from .mechanics_2d import run_capture as run_mechanics_2d
from .mechanics_3d import run_capture as run_mechanics_3d
from .seepage_2d import run_capture as run_seepage_2d
from .seepage_3d import run_capture as run_seepage_3d
from .seepage_ssr_3d import run_capture as run_seepage_ssr_3d


class RouteKind(str, Enum):
    MECHANICS_2D = "2d_mechanics"
    SEEPAGE_2D = "2d_seepage"
    MECHANICS_3D = "3d_mechanics"
    SEEPAGE_3D = "3d_seepage"
    SEEPAGE_SSR_3D = "3d_seepage_ssr"


def _optional_int(value) -> int | None:
    if value is None:
        return None
    return int(value)


def select_case_route(cfg: RunCaseConfig) -> RouteKind:
    resolved = resolve_problem_asset_from_config(cfg)
    analysis = cfg.problem.analysis.lower()
    if resolved.dimension == 2 and analysis == "seepage":
        return RouteKind.SEEPAGE_2D
    if resolved.dimension == 2:
        return RouteKind.MECHANICS_2D
    if resolved.dimension == 3 and analysis == "seepage":
        return RouteKind.SEEPAGE_3D
    if resolved.dimension == 3 and analysis == "ssr" and "seepage" in resolved.definition.capabilities:
        return RouteKind.SEEPAGE_SSR_3D
    if resolved.dimension == 3 and "seepage" in resolved.definition.capabilities:
        raise ValueError(
            f"Analysis {cfg.problem.analysis!r} is not supported for seepage-capable 3D asset "
            f"{resolved.asset_name!r}; supported analyses are 'seepage' and seepage-coupled 'ssr'."
        )
    if resolved.dimension == 3:
        return RouteKind.MECHANICS_3D
    raise KeyError(
        f"Unsupported asset routing for asset={resolved.asset_name!r}, "
        f"dimension={resolved.dimension}, source_kind={resolved.source_kind!r}, analysis={cfg.problem.analysis!r}."
    )


def case_runner_kwargs(cfg: RunCaseConfig) -> tuple[Callable[..., dict], dict]:
    resolved = resolve_problem_asset_from_config(cfg)
    profile = resolved.resolved_variant.profile
    linear = cfg.linear_solver
    common_linear = {
        "solver_type": linear.solver_type,
        "linear_tolerance": linear.tolerance,
        "linear_max_iter": linear.max_iterations,
    }

    if resolved.dimension == 2 and cfg.problem.analysis.lower() == "seepage":
        kwargs = {
            "asset_name": resolved.asset_name,
            "mesh_variant": resolved.variant_name,
            "profile": profile,
            "elem_type": cfg.problem.elem_type,
            "node_ordering": cfg.execution.node_ordering,
            "solver_type": linear.solver_type.replace("_NULLSPACE", ""),
            "linear_tolerance": cfg.seepage.linear_tolerance,
            "linear_max_iter": cfg.seepage.linear_max_iter,
            "nonlinear_max_iter": cfg.seepage.nonlinear_max_iter,
        }
        return run_seepage_2d, kwargs

    if resolved.dimension == 2:
        kwargs = {
            "asset_name": resolved.asset_name,
            "mesh_variant": resolved.variant_name,
            "profile": profile,
            "analysis": cfg.problem.analysis,
            "continuation_method": cfg.continuation.method,
            "elem_type": cfg.problem.elem_type,
            "davis_type": cfg.problem.davis_type,
            "node_ordering": cfg.execution.node_ordering,
            "lambda_init": cfg.continuation.lambda_init,
            "d_lambda_init": cfg.continuation.d_lambda_init,
            "d_lambda_min": cfg.continuation.d_lambda_min,
            "d_lambda_diff_scaled_min": cfg.continuation.d_lambda_diff_scaled_min,
            "lambda_ell": cfg.continuation.lambda_ell,
            "d_omega_ini_scale": cfg.continuation.d_omega_ini_scale,
            "d_t_min": cfg.continuation.d_t_min,
            "omega_max_stop": cfg.continuation.omega_max,
            "continuation_predictor": cfg.continuation.predictor,
            "omega_step_controller": cfg.continuation.omega_step_controller,
            "continuation_secant_correction_mode": cfg.continuation.secant_correction_mode,
            "continuation_first_newton_warm_start_mode": cfg.continuation.first_newton_warm_start_mode,
            "continuation_secant_correction_mode": cfg.continuation.secant_correction_mode,
            "continuation_first_newton_warm_start_mode": cfg.continuation.first_newton_warm_start_mode,
            "omega_no_increase_newton_threshold": cfg.continuation.omega_no_increase_newton_threshold,
            "omega_half_newton_threshold": cfg.continuation.omega_half_newton_threshold,
            "omega_target_newton_iterations": cfg.continuation.omega_target_newton_iterations,
            "omega_adapt_min_scale": cfg.continuation.omega_adapt_min_scale,
            "omega_adapt_max_scale": cfg.continuation.omega_adapt_max_scale,
            "omega_hard_newton_threshold": cfg.continuation.omega_hard_newton_threshold,
            "omega_hard_linear_threshold": cfg.continuation.omega_hard_linear_threshold,
            "omega_efficiency_floor": cfg.continuation.omega_efficiency_floor,
            "omega_efficiency_drop_ratio": cfg.continuation.omega_efficiency_drop_ratio,
            "omega_efficiency_window": cfg.continuation.omega_efficiency_window,
            "omega_hard_shrink_scale": cfg.continuation.omega_hard_shrink_scale,
            "step_length_cap_mode": cfg.continuation.step_length_cap_mode,
            "step_length_cap_factor": cfg.continuation.step_length_cap_factor,
            "step_max": cfg.continuation.step_max,
            "it_newt_max": cfg.newton.it_max,
            "it_damp_max": cfg.newton.it_damp_max,
            "tol": cfg.newton.tol,
            "r_min": cfg.newton.r_min,
            "mpi_distribute_by_nodes": cfg.execution.mpi_distribute_by_nodes,
            "pc_hypre_coarsen_type": linear.pc_hypre_coarsen_type,
            "pc_hypre_interp_type": linear.pc_hypre_interp_type,
            "pc_hypre_strong_threshold": linear.pc_hypre_strong_threshold,
            "pc_hypre_boomeramg_max_iter": linear.pc_hypre_boomeramg_max_iter or 1,
            "recycle_preconditioner": linear.recycle_preconditioner,
            "constitutive_mode": cfg.execution.constitutive_mode,
            "tangent_kernel": cfg.execution.tangent_kernel,
            "seepage_linear_tolerance": cfg.seepage.linear_tolerance,
            "seepage_linear_max_iter": cfg.seepage.linear_max_iter,
            **common_linear,
        }
        return run_mechanics_2d, kwargs

    if resolved.dimension == 3 and cfg.problem.analysis.lower() == "seepage":
        kwargs = {
            "asset_name": resolved.asset_name,
            "mesh_variant": resolved.variant_name,
            "profile": profile,
            "elem_type": cfg.problem.elem_type,
            "node_ordering": cfg.execution.node_ordering,
            "solver_type": linear.solver_type.replace("_NULLSPACE", ""),
            "pc_backend": linear.pc_backend or "hypre",
            "linear_tolerance": cfg.seepage.linear_tolerance,
            "linear_max_iter": cfg.seepage.linear_max_iter,
            "petsc_opt": linear.petsc_opt,
        }
        return run_seepage_3d, kwargs

    if resolved.dimension == 3 and cfg.problem.analysis.lower() == "ssr" and "seepage" in resolved.definition.capabilities:
        kwargs = {
            "asset_name": resolved.asset_name,
            "mesh_variant": resolved.variant_name,
            "profile": profile,
            "elem_type": cfg.problem.elem_type,
            "node_ordering": cfg.execution.node_ordering,
            "lambda_init": cfg.continuation.lambda_init,
            "d_lambda_init": cfg.continuation.d_lambda_init,
            "d_lambda_min": cfg.continuation.d_lambda_min,
            "d_lambda_diff_scaled_min": cfg.continuation.d_lambda_diff_scaled_min,
            "omega_max_stop": cfg.continuation.omega_max,
            "continuation_predictor": cfg.continuation.predictor,
            "omega_step_controller": cfg.continuation.omega_step_controller,
            "step_max": cfg.continuation.step_max,
            "it_newt_max": cfg.newton.it_max,
            "it_damp_max": cfg.newton.it_damp_max,
            "tol": cfg.newton.tol,
            "r_min": cfg.newton.r_min,
            "newton_stopping_criterion": cfg.newton.stopping_criterion,
            "newton_stopping_tol": cfg.newton.stopping_tol,
            "mpi_distribute_by_nodes": cfg.execution.mpi_distribute_by_nodes,
            "pc_backend": linear.pc_backend,
            "pmg_coarse_mesh_variant": linear.pmg_coarse_mesh_variant,
            "pmg_fine_hierarchy_mode": linear.pmg_fine_hierarchy_mode,
            "pc_hypre_coarsen_type": linear.pc_hypre_coarsen_type or "HMIS",
            "pc_hypre_interp_type": linear.pc_hypre_interp_type or "ext+i",
            "pc_hypre_strong_threshold": linear.pc_hypre_strong_threshold,
            "pc_hypre_boomeramg_max_iter": linear.pc_hypre_boomeramg_max_iter or 1,
            "preconditioner_threads": linear.threads,
            "pc_hypre_P_max": linear.pc_hypre_P_max,
            "pc_hypre_agg_nl": linear.pc_hypre_agg_nl,
            "pc_hypre_nongalerkin_tol": linear.pc_hypre_nongalerkin_tol,
            "recycle_preconditioner": linear.recycle_preconditioner,
            "constitutive_mode": cfg.execution.constitutive_mode,
            "tangent_kernel": cfg.execution.tangent_kernel,
            "seepage_linear_tolerance": cfg.seepage.linear_tolerance,
            "seepage_linear_max_iter": cfg.seepage.linear_max_iter,
            "petsc_opt": linear.petsc_opt,
            "max_deflation_basis_vectors": linear.max_deflation_basis_vectors,
            **common_linear,
        }
        return run_seepage_ssr_3d, kwargs

    if resolved.dimension == 3 and "seepage" in resolved.definition.capabilities:
        raise ValueError(
            f"Analysis {cfg.problem.analysis!r} is not supported for seepage-capable 3D asset "
            f"{resolved.asset_name!r}; supported analyses are 'seepage' and seepage-coupled 'ssr'."
        )

    if resolved.dimension == 3:
        kwargs = {
            "analysis": cfg.problem.analysis,
            "asset_name": resolved.asset_name,
            "mesh_variant": resolved.variant_name,
            "profile": profile,
            "elem_type": cfg.problem.elem_type,
            "quadrature_rule": _optional_int(cfg.geometry.get("quadrature_rule")),
            "davis_type": cfg.problem.davis_type,
            "node_ordering": cfg.execution.node_ordering,
            "lambda_init": cfg.continuation.lambda_init,
            "d_lambda_init": cfg.continuation.d_lambda_init,
            "d_lambda_min": cfg.continuation.d_lambda_min,
            "d_lambda_diff_scaled_min": cfg.continuation.d_lambda_diff_scaled_min,
            "lambda_ell": cfg.continuation.lambda_ell,
            "d_omega_ini_scale": cfg.continuation.d_omega_ini_scale,
            "d_t_min": cfg.continuation.d_t_min,
            "omega_max_stop": cfg.continuation.omega_max,
            "continuation_predictor": cfg.continuation.predictor,
            "omega_step_controller": cfg.continuation.omega_step_controller,
            "omega_no_increase_newton_threshold": cfg.continuation.omega_no_increase_newton_threshold,
            "omega_half_newton_threshold": cfg.continuation.omega_half_newton_threshold,
            "omega_target_newton_iterations": cfg.continuation.omega_target_newton_iterations,
            "omega_adapt_min_scale": cfg.continuation.omega_adapt_min_scale,
            "omega_adapt_max_scale": cfg.continuation.omega_adapt_max_scale,
            "omega_hard_newton_threshold": cfg.continuation.omega_hard_newton_threshold,
            "omega_hard_linear_threshold": cfg.continuation.omega_hard_linear_threshold,
            "omega_efficiency_floor": cfg.continuation.omega_efficiency_floor,
            "omega_efficiency_drop_ratio": cfg.continuation.omega_efficiency_drop_ratio,
            "omega_efficiency_window": cfg.continuation.omega_efficiency_window,
            "omega_hard_shrink_scale": cfg.continuation.omega_hard_shrink_scale,
            "step_length_cap_mode": cfg.continuation.step_length_cap_mode,
            "step_length_cap_factor": cfg.continuation.step_length_cap_factor,
            "step_max": cfg.continuation.step_max,
            "it_newt_max": cfg.newton.it_max,
            "it_damp_max": cfg.newton.it_damp_max,
            "tol": cfg.newton.tol,
            "r_min": cfg.newton.r_min,
            "newton_stopping_criterion": cfg.newton.stopping_criterion,
            "newton_stopping_tol": cfg.newton.stopping_tol,
            "newton_line_search": cfg.newton.line_search,
            "newton_armijo_alpha0": cfg.newton.armijo_alpha0,
            "newton_armijo_c1": cfg.newton.armijo_c1,
            "newton_armijo_shrink": cfg.newton.armijo_shrink,
            "newton_armijo_max_ls": cfg.newton.armijo_max_ls,
            "newton_armijo_rescale_trial_to_omega": cfg.newton.armijo_rescale_trial_to_omega,
            "newton_armijo_fallback_to_alg5": cfg.newton.armijo_fallback_to_alg5,
            "init_newton_stopping_criterion": cfg.continuation.init_newton_stopping_criterion,
            "init_newton_stopping_tol": cfg.continuation.init_newton_stopping_tol,
            "fine_newton_stopping_criterion": cfg.continuation.fine_newton_stopping_criterion,
            "fine_newton_stopping_tol": cfg.continuation.fine_newton_stopping_tol,
            "fine_switch_mode": cfg.continuation.fine_switch_mode,
            "fine_switch_distance_factor": cfg.continuation.fine_switch_distance_factor,
            "factor_solver_type": linear.factor_solver_type,
            "preconditioner_threads": linear.threads,
            "pc_backend": linear.pc_backend,
            "pmg_coarse_mesh_variant": linear.pmg_coarse_mesh_variant,
            "pmg_fine_hierarchy_mode": linear.pmg_fine_hierarchy_mode,
            "numa_domains_per_node": linear.numa_domains_per_node,
            "pmg_smoother_pc_type": linear.pmg_smoother_pc_type,
            "pmg_smoother_gasm_total_subdomains": linear.pmg_smoother_gasm_total_subdomains,
            "pmg_smoother_gasm_grouping": linear.pmg_smoother_gasm_grouping,
            "pmg_smoother_gasm_overlap": linear.pmg_smoother_gasm_overlap,
            "pmg_smoother_gasm_type": linear.pmg_smoother_gasm_type,
            "pmg_smoother_gasm_sub_ksp_type": linear.pmg_smoother_gasm_sub_ksp_type,
            "pmg_smoother_gasm_sub_ksp_max_it": linear.pmg_smoother_gasm_sub_ksp_max_it,
            "pmg_smoother_gasm_sub_pc_type": linear.pmg_smoother_gasm_sub_pc_type,
            "pmg_smoother_gasm_view_subdomains": linear.pmg_smoother_gasm_view_subdomains,
            "preconditioner_matrix_source": linear.preconditioner_matrix_source,
            "preconditioner_matrix_policy": linear.preconditioner_matrix_policy,
            "preconditioner_rebuild_policy": linear.preconditioner_rebuild_policy,
            "preconditioner_rebuild_interval": linear.preconditioner_rebuild_interval,
            "mpi_distribute_by_nodes": cfg.execution.mpi_distribute_by_nodes,
            "pc_gamg_process_eq_limit": linear.pc_gamg_process_eq_limit,
            "pc_gamg_threshold": linear.pc_gamg_threshold,
            "pc_gamg_aggressive_coarsening": linear.pc_gamg_aggressive_coarsening,
            "pc_gamg_aggressive_square_graph": linear.pc_gamg_aggressive_square_graph,
            "pc_gamg_aggressive_mis_k": linear.pc_gamg_aggressive_mis_k,
            "pc_hypre_coarsen_type": linear.pc_hypre_coarsen_type,
            "pc_hypre_interp_type": linear.pc_hypre_interp_type,
            "pc_hypre_strong_threshold": linear.pc_hypre_strong_threshold,
            "pc_hypre_boomeramg_max_iter": linear.pc_hypre_boomeramg_max_iter or 1,
            "pc_hypre_P_max": linear.pc_hypre_P_max,
            "pc_hypre_agg_nl": linear.pc_hypre_agg_nl,
            "pc_hypre_nongalerkin_tol": linear.pc_hypre_nongalerkin_tol,
            "pc_bddc_symmetric": linear.pc_bddc_symmetric,
            "pc_bddc_dirichlet_ksp_type": linear.pc_bddc_dirichlet_ksp_type,
            "pc_bddc_dirichlet_pc_type": linear.pc_bddc_dirichlet_pc_type,
            "pc_bddc_neumann_ksp_type": linear.pc_bddc_neumann_ksp_type,
            "pc_bddc_neumann_pc_type": linear.pc_bddc_neumann_pc_type,
            "pc_bddc_coarse_ksp_type": linear.pc_bddc_coarse_ksp_type,
            "pc_bddc_coarse_pc_type": linear.pc_bddc_coarse_pc_type,
            "pc_bddc_dirichlet_approximate": linear.pc_bddc_dirichlet_approximate,
            "pc_bddc_neumann_approximate": linear.pc_bddc_neumann_approximate,
            "pc_bddc_monolithic": linear.pc_bddc_monolithic,
            "pc_bddc_coarse_redundant_pc_type": linear.pc_bddc_coarse_redundant_pc_type,
            "pc_bddc_switch_static": linear.pc_bddc_switch_static,
            "pc_bddc_use_deluxe_scaling": linear.pc_bddc_use_deluxe_scaling,
            "pc_bddc_use_vertices": linear.pc_bddc_use_vertices,
            "pc_bddc_use_edges": linear.pc_bddc_use_edges,
            "pc_bddc_use_faces": linear.pc_bddc_use_faces,
            "pc_bddc_use_change_of_basis": linear.pc_bddc_use_change_of_basis,
            "pc_bddc_use_change_on_faces": linear.pc_bddc_use_change_on_faces,
            "pc_bddc_check_level": linear.pc_bddc_check_level,
            "petsc_opt": linear.petsc_opt,
            "max_deflation_basis_vectors": linear.max_deflation_basis_vectors,
            "compiled_outer": linear.compiled_outer,
            "recycle_preconditioner": linear.recycle_preconditioner,
            "constitutive_mode": cfg.execution.constitutive_mode,
            "tangent_kernel": cfg.execution.tangent_kernel,
            "store_step_u": cfg.execution.store_step_u,
            **common_linear,
        }
        return run_mechanics_3d, kwargs

    raise KeyError(
        f"Unsupported asset routing for asset={resolved.asset_name!r}, "
        f"dimension={resolved.dimension}, source_kind={resolved.source_kind!r}, analysis={cfg.problem.analysis!r}."
    )


def run_case_config(cfg: RunCaseConfig, output_dir: Path) -> dict:
    runner, kwargs = case_runner_kwargs(cfg)
    sig = inspect.signature(runner)
    accepted = set(sig.parameters)
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    if "output_dir" in sig.parameters:
        return runner(Path(output_dir), **filtered_kwargs)
    if "out_dir" in sig.parameters:
        return runner(out_dir=Path(output_dir), **filtered_kwargs)
    raise TypeError(f"Unsupported runner signature for {runner.__module__}.{runner.__name__}")


def _load_export_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as npz:
        return {name: np.asarray(npz[name]) for name in npz.files}


def _build_field_exports(
    arrays: dict[str, np.ndarray],
    *,
    n_cells: int,
    coord: np.ndarray | None = None,
    elem: np.ndarray | None = None,
    elem_type: str | None = None,
    dim: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return build_field_exports(
        arrays,
        n_cells=n_cells,
        coord=coord,
        elem=elem,
        elem_type=elem_type,
        dim=dim,
    )


def _export_outputs(cfg: RunCaseConfig, config_path: Path, output_dir: Path) -> None:
    data_dir = output_dir / "data"
    npz_path = data_dir / "petsc_run.npz"
    run_info_path = data_dir / "run_info.json"
    progress_path = data_dir / "progress.jsonl"
    exports_dir = output_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    config_text = config_path.read_text(encoding="utf-8")
    (output_dir / "generated_case.toml").write_text(config_text, encoding="utf-8")
    (exports_dir / "resolved_config.toml").write_text(config_text, encoding="utf-8")

    if cfg.export.write_custom_debug_bundle and npz_path.exists() and run_info_path.exists():
        write_debug_bundle_h5(
            out_path=exports_dir / cfg.export.custom_debug_name,
            config_text=config_text,
            run_info_path=run_info_path,
            npz_path=npz_path,
            progress_path=progress_path if progress_path.exists() else None,
        )
    if cfg.export.write_history_json and npz_path.exists() and run_info_path.exists():
        history_path = write_history_json(
            out_path=exports_dir / cfg.export.history_name,
            run_info_path=run_info_path,
            npz_path=npz_path,
            progress_path=progress_path if progress_path.exists() else None,
        )
        write_history_csv_tables(
            out_dir=exports_dir,
            history_json_path=history_path,
        )
    if cfg.export.write_solution_vtu and npz_path.exists():
        case_mesh = rebuild_case_mesh(cfg, mpi_size=int(PETSc.COMM_WORLD.getSize()))
        arrays = _load_export_arrays(npz_path)
        validate_case_mesh_alignment(case_mesh, arrays)
        point_data, cell_data = _build_field_exports(
            arrays,
            n_cells=sum(block.shape[0] for _, block in case_mesh.cell_blocks),
            coord=case_mesh.coord,
            elem=case_mesh.elem,
            elem_type=cfg.problem.elem_type,
            dim=case_mesh.dim,
        )
        cell_data = {"material_id": case_mesh.material_id, **cell_data}
        write_vtu(
            exports_dir / cfg.export.solution_name,
            points=case_mesh.points,
            cell_blocks=case_mesh.cell_blocks,
            point_data=point_data,
            cell_data=cell_data,
        )
