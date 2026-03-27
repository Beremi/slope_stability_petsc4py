#!/usr/bin/env python
"""Run a supported benchmark case from a TOML config file."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from petsc4py import PETSc

ROOT = Path(__file__).resolve().parents[3]

from slope_stability.core.run_config import RunCaseConfig, load_run_case_config
from slope_stability.export import write_debug_bundle_h5, write_history_json, write_vtu
from slope_stability.postprocess import build_field_exports, rebuild_case_mesh

from .run_2D_homo_SSR_capture import run_capture as run_2d_homo_ssr_capture
from .run_2D_textmesh_case_capture import run_capture as run_2d_textmesh_case_capture
from .run_2D_sloan2013_seepage_capture import run_capture as run_2d_sloan2013_seepage_capture
from .run_3D_hetero_SSR_capture import run_capture as run_3d_ssr_capture
from .run_3D_hetero_seepage_capture import run_capture as run_3d_hetero_seepage_capture
from .run_3D_hetero_seepage_SSR_comsol_capture import run_capture as run_3d_comsol_ssr_capture


def _case_runner_kwargs(cfg: RunCaseConfig) -> tuple[callable, dict]:
    linear = cfg.linear_solver
    common_linear = {
        "solver_type": linear.solver_type,
        "linear_tolerance": linear.tolerance,
        "linear_max_iter": linear.max_iterations,
    }
    if cfg.problem.case == "2d_homo_ssr":
        geom = cfg.geometry
        kwargs = {
            "analysis": cfg.problem.analysis,
            "elem_type": cfg.problem.elem_type,
            "davis_type": cfg.problem.davis_type,
            "h": float(geom.get("h", 1.0)),
            "x1": float(geom.get("x1", 15.0)),
            "x3": float(geom.get("x3", 15.0)),
            "y1": float(geom.get("y1", 10.0)),
            "y2": float(geom.get("y2", 10.0)),
            "beta_deg": float(geom.get("beta_deg", 45.0)),
            "material_row": cfg.material_rows()[0],
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
            **common_linear,
        }
        return run_2d_homo_ssr_capture, kwargs
    if cfg.problem.case == "2d_sloan2013_seepage":
        kwargs = {
            "elem_type": cfg.problem.elem_type,
            "solver_type": linear.solver_type.replace("_NULLSPACE", ""),
            "linear_tolerance": cfg.seepage.linear_tolerance,
            "linear_max_iter": cfg.seepage.linear_max_iter,
            "nonlinear_max_iter": cfg.seepage.nonlinear_max_iter,
        }
        return run_2d_sloan2013_seepage_capture, kwargs
    if cfg.problem.case in {"2d_kozinec_ssr", "2d_kozinec_ll", "2d_luzec_ssr", "2d_franz_dam_ssr"}:
        kwargs = {
            "case_name": cfg.case_data.get("case_name", cfg.problem.case.split("_", 1)[1].rsplit("_", 1)[0]),
            "analysis": cfg.problem.analysis,
            "continuation_method": cfg.continuation.method,
            "mesh_dir": cfg.case_data["mesh_dir"],
            "elem_type": cfg.problem.elem_type,
            "davis_type": cfg.problem.davis_type,
            "material_rows": cfg.material_rows(),
            "hydraulic_conductivity": (
                None if not cfg.seepage.conductivity else list(cfg.seepage.conductivity)
            ),
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
            "seepage_water_unit_weight": cfg.seepage.water_unit_weight,
            **common_linear,
        }
        return run_2d_textmesh_case_capture, kwargs
    if cfg.problem.case in {"3d_homo_ssr", "3d_hetero_ssr", "3d_siopt_ssr"}:
        kwargs = {
            "analysis": cfg.problem.analysis,
            "mesh_path": cfg.problem.mesh_path,
            "mesh_boundary_type": cfg.problem.mesh_boundary_type,
            "elem_type": cfg.problem.elem_type,
            "davis_type": cfg.problem.davis_type,
            "material_rows": cfg.material_rows(),
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
            "step_max": cfg.continuation.step_max,
            "it_newt_max": cfg.newton.it_max,
            "it_damp_max": cfg.newton.it_damp_max,
            "tol": cfg.newton.tol,
            "r_min": cfg.newton.r_min,
            "factor_solver_type": linear.factor_solver_type,
            "pc_backend": linear.pc_backend,
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
            "compiled_outer": linear.compiled_outer,
            "recycle_preconditioner": linear.recycle_preconditioner,
            "constitutive_mode": cfg.execution.constitutive_mode,
            "tangent_kernel": cfg.execution.tangent_kernel,
            **common_linear,
        }
        return run_3d_ssr_capture, kwargs
    if cfg.problem.case == "3d_hetero_seepage":
        kwargs = {
            "mesh_path": cfg.problem.mesh_path,
            "elem_type": cfg.problem.elem_type,
            "solver_type": linear.solver_type.replace("_NULLSPACE", ""),
            "linear_tolerance": cfg.seepage.linear_tolerance,
            "linear_max_iter": cfg.seepage.linear_max_iter,
        }
        return run_3d_hetero_seepage_capture, kwargs
    if cfg.problem.case in {"3d_hetero_seepage_ssr_comsol", "3d_homo_seepage_ssr"}:
        kwargs = {
            "mesh_path": cfg.problem.mesh_path,
            "elem_type": cfg.problem.elem_type,
            "node_ordering": cfg.execution.node_ordering,
            "lambda_init": cfg.continuation.lambda_init,
            "d_lambda_init": cfg.continuation.d_lambda_init,
            "d_lambda_min": cfg.continuation.d_lambda_min,
            "d_lambda_diff_scaled_min": cfg.continuation.d_lambda_diff_scaled_min,
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
            "step_max": cfg.continuation.step_max,
            "it_newt_max": cfg.newton.it_max,
            "it_damp_max": cfg.newton.it_damp_max,
            "tol": cfg.newton.tol,
            "r_min": cfg.newton.r_min,
            "mpi_distribute_by_nodes": cfg.execution.mpi_distribute_by_nodes,
            "pc_backend": linear.pc_backend,
            "preconditioner_matrix_source": linear.preconditioner_matrix_source,
            "preconditioner_matrix_policy": linear.preconditioner_matrix_policy,
            "preconditioner_rebuild_policy": linear.preconditioner_rebuild_policy,
            "preconditioner_rebuild_interval": linear.preconditioner_rebuild_interval,
            "pc_hypre_coarsen_type": linear.pc_hypre_coarsen_type or "HMIS",
            "pc_hypre_interp_type": linear.pc_hypre_interp_type or "ext+i",
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
            "recycle_preconditioner": linear.recycle_preconditioner,
            "constitutive_mode": cfg.execution.constitutive_mode,
            "tangent_kernel": cfg.execution.tangent_kernel,
            "seepage_linear_tolerance": cfg.seepage.linear_tolerance,
            "seepage_linear_max_iter": cfg.seepage.linear_max_iter,
            **common_linear,
        }
        return run_3d_comsol_ssr_capture, kwargs
    raise KeyError(f"Unsupported case id {cfg.problem.case!r}")


def _build_field_exports(npz_path: Path, n_cells: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(npz_path, allow_pickle=True) as npz:
        arrays = {name: np.asarray(npz[name]) for name in npz.files}
    return build_field_exports(arrays, n_cells=n_cells)


def _export_outputs(cfg: RunCaseConfig, config_path: Path, output_dir: Path) -> None:
    data_dir = output_dir / "data"
    npz_path = data_dir / "petsc_run.npz"
    run_info_path = data_dir / "run_info.json"
    progress_path = data_dir / "progress.jsonl"
    exports_dir = output_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    config_text = config_path.read_text(encoding="utf-8")
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
        write_history_json(
            out_path=exports_dir / cfg.export.history_name,
            run_info_path=run_info_path,
            npz_path=npz_path,
            progress_path=progress_path if progress_path.exists() else None,
        )
    if cfg.export.write_solution_vtu and npz_path.exists():
        case_mesh = rebuild_case_mesh(cfg, mpi_size=int(PETSc.COMM_WORLD.getSize()))
        point_data, cell_data = _build_field_exports(
            npz_path,
            sum(block.shape[0] for _, block in case_mesh.cell_blocks),
        )
        cell_data = {"material_id": case_mesh.material_id, **cell_data}
        write_vtu(
            exports_dir / cfg.export.solution_name,
            points=case_mesh.points,
            cell_blocks=case_mesh.cell_blocks,
            point_data=point_data,
            cell_data=cell_data,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PETSc slope-stability case from a TOML config.")
    parser.add_argument("config", type=Path, help="Path to the TOML config.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Optional output directory override.")
    args = parser.parse_args()

    cfg = load_run_case_config(args.config)
    runner, kwargs = _case_runner_kwargs(cfg)
    out_dir = args.out_dir
    if out_dir is None:
        safe_ts = np.datetime64("now").astype(str).replace(":", "-")
        out_dir = ROOT / "artifacts" / "config_runs" / cfg.problem.name / safe_ts

    sig = inspect.signature(runner)
    accepted = set(sig.parameters)
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    if "output_dir" in sig.parameters:
        result = runner(Path(out_dir), **filtered_kwargs)
    elif "out_dir" in sig.parameters:
        result = runner(out_dir=Path(out_dir), **filtered_kwargs)
    else:
        raise TypeError(f"Unsupported runner signature for {runner.__module__}.{runner.__name__}")
    if PETSc.COMM_WORLD.getRank() == 0:
        output_path = Path(result["output"]) if isinstance(result, dict) and "output" in result else Path(out_dir)
        _export_outputs(cfg, args.config.resolve(), output_path)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
