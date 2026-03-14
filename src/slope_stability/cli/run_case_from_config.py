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
from slope_stability.core.elements import simplex_vtk_cell_block
from slope_stability.export import write_debug_bundle_h5, write_history_json, write_vtu
from slope_stability.mesh import (
    franz_dam_pressure_boundary,
    generate_homogeneous_slope_mesh_2d,
    generate_sloan2013_mesh_2d,
    load_mesh_franz_dam_2d,
    load_mesh_from_file,
    load_mesh_gmsh_waterlevels,
    load_mesh_kozinec_2d,
    load_mesh_luzec_2d,
    load_mesh_p2_comsol,
    luzec_pressure_boundary,
    reorder_mesh_nodes,
)

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
            "step_max": cfg.continuation.step_max,
            "it_newt_max": cfg.newton.it_max,
            "it_damp_max": cfg.newton.it_damp_max,
            "tol": cfg.newton.tol,
            "r_min": cfg.newton.r_min,
            "mpi_distribute_by_nodes": cfg.execution.mpi_distribute_by_nodes,
            "pc_hypre_coarsen_type": linear.pc_hypre_coarsen_type,
            "pc_hypre_interp_type": linear.pc_hypre_interp_type,
            "pc_hypre_strong_threshold": linear.pc_hypre_strong_threshold,
            "recycle_preconditioner": linear.recycle_preconditioner,
            "constitutive_mode": cfg.execution.constitutive_mode,
            **common_linear,
        }
        return run_2d_homo_ssr_capture, kwargs
    if cfg.problem.case == "2d_sloan2013_seepage":
        kwargs = {
            "elem_type": cfg.problem.elem_type,
            "solver_type": linear.solver_type.replace("_NULLSPACE", ""),
            "linear_tolerance": cfg.seepage.linear_tolerance,
            "linear_max_iter": cfg.seepage.linear_max_iter,
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
            "step_max": cfg.continuation.step_max,
            "it_newt_max": cfg.newton.it_max,
            "it_damp_max": cfg.newton.it_damp_max,
            "tol": cfg.newton.tol,
            "r_min": cfg.newton.r_min,
            "mpi_distribute_by_nodes": cfg.execution.mpi_distribute_by_nodes,
            "pc_hypre_coarsen_type": linear.pc_hypre_coarsen_type,
            "pc_hypre_interp_type": linear.pc_hypre_interp_type,
            "pc_hypre_strong_threshold": linear.pc_hypre_strong_threshold,
            "recycle_preconditioner": linear.recycle_preconditioner,
            "constitutive_mode": cfg.execution.constitutive_mode,
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
            "step_max": cfg.continuation.step_max,
            "it_newt_max": cfg.newton.it_max,
            "it_damp_max": cfg.newton.it_damp_max,
            "tol": cfg.newton.tol,
            "r_min": cfg.newton.r_min,
            "factor_solver_type": linear.factor_solver_type,
            "mpi_distribute_by_nodes": cfg.execution.mpi_distribute_by_nodes,
            "pc_gamg_process_eq_limit": linear.pc_gamg_process_eq_limit,
            "pc_gamg_threshold": linear.pc_gamg_threshold,
            "pc_hypre_coarsen_type": linear.pc_hypre_coarsen_type,
            "pc_hypre_interp_type": linear.pc_hypre_interp_type,
            "pc_hypre_strong_threshold": linear.pc_hypre_strong_threshold,
            "compiled_outer": linear.compiled_outer,
            "recycle_preconditioner": linear.recycle_preconditioner,
            "constitutive_mode": cfg.execution.constitutive_mode,
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
            "step_max": cfg.continuation.step_max,
            "it_newt_max": cfg.newton.it_max,
            "it_damp_max": cfg.newton.it_damp_max,
            "tol": cfg.newton.tol,
            "r_min": cfg.newton.r_min,
            "mpi_distribute_by_nodes": cfg.execution.mpi_distribute_by_nodes,
            "pc_hypre_coarsen_type": linear.pc_hypre_coarsen_type or "HMIS",
            "pc_hypre_interp_type": linear.pc_hypre_interp_type or "ext+i",
            "pc_hypre_strong_threshold": linear.pc_hypre_strong_threshold,
            "recycle_preconditioner": linear.recycle_preconditioner,
            "constitutive_mode": cfg.execution.constitutive_mode,
            "seepage_linear_tolerance": cfg.seepage.linear_tolerance,
            "seepage_linear_max_iter": cfg.seepage.linear_max_iter,
            **common_linear,
        }
        return run_3d_comsol_ssr_capture, kwargs
    raise KeyError(f"Unsupported case id {cfg.problem.case!r}")


def _rebuild_case_mesh(cfg: RunCaseConfig) -> tuple[np.ndarray, list[tuple[str, np.ndarray]], dict[str, np.ndarray]]:
    case = cfg.problem.case
    size = int(PETSc.COMM_WORLD.getSize())
    part_count = size if cfg.execution.node_ordering.lower() == "block_metis" else None

    if case == "2d_homo_ssr":
        geom = cfg.geometry
        beta_deg = float(geom.get("beta_deg", 45.0))
        y2 = float(geom.get("y2", 10.0))
        x2 = y2 / np.tan(np.deg2rad(beta_deg))
        mesh = generate_homogeneous_slope_mesh_2d(
            elem_type=cfg.problem.elem_type,
            h=float(geom.get("h", 1.0)),
            x1=float(geom.get("x1", 15.0)),
            x2=float(x2),
            x3=float(geom.get("x3", 15.0)),
            y1=float(geom.get("y1", 10.0)),
            y2=y2,
        )
        reordered = reorder_mesh_nodes(
            mesh.coord,
            mesh.elem,
            mesh.surf,
            mesh.q_mask,
            strategy=cfg.execution.node_ordering,
            n_parts=part_count,
        )
        points = _points_2d(reordered.coord)
        cell_type, cells = simplex_vtk_cell_block(2, reordered.elem, cfg.problem.elem_type)
        return points, [(cell_type, cells)], {"material_id": np.asarray(mesh.material, dtype=np.int64)}

    if case == "2d_sloan2013_seepage":
        mesh = generate_sloan2013_mesh_2d(elem_type=cfg.problem.elem_type)
        points = _points_2d(mesh.coord)
        cell_type, cells = simplex_vtk_cell_block(2, mesh.elem, cfg.problem.elem_type)
        return points, [(cell_type, cells)], {"material_id": np.asarray(mesh.material, dtype=np.int64)}

    if case in {"2d_kozinec_ssr", "2d_kozinec_ll", "2d_luzec_ssr", "2d_franz_dam_ssr"}:
        mesh_dir = Path(cfg.case_data["mesh_dir"])
        if case.startswith("2d_kozinec"):
            mesh = load_mesh_kozinec_2d(cfg.problem.elem_type, mesh_dir)
        elif case == "2d_luzec_ssr":
            mesh = load_mesh_luzec_2d(cfg.problem.elem_type, mesh_dir)
        else:
            mesh = load_mesh_franz_dam_2d(cfg.problem.elem_type, mesh_dir)
        reordered = reorder_mesh_nodes(
            mesh.coord,
            mesh.elem,
            mesh.surf,
            mesh.q_mask,
            strategy=cfg.execution.node_ordering,
            n_parts=part_count,
        )
        points = _points_2d(reordered.coord)
        cell_type, cells = simplex_vtk_cell_block(2, reordered.elem, cfg.problem.elem_type)
        cell_blocks = [(cell_type, cells)]
        return points, cell_blocks, {"material_id": np.asarray(mesh.material, dtype=np.int64)}

    if case in {"3d_homo_ssr", "3d_hetero_ssr", "3d_siopt_ssr"}:
        mesh = load_mesh_from_file(cfg.problem.mesh_path, boundary_type=cfg.problem.mesh_boundary_type)
        reordered = reorder_mesh_nodes(
            mesh.coord,
            mesh.elem,
            mesh.surf,
            mesh.q_mask,
            strategy=cfg.execution.node_ordering,
            n_parts=part_count,
        )
        cell_type, cells = simplex_vtk_cell_block(3, reordered.elem, cfg.problem.elem_type)
        return reordered.coord.T, [(cell_type, cells)], {"material_id": np.asarray(mesh.material, dtype=np.int64)}

    if case == "3d_hetero_seepage":
        mesh = load_mesh_gmsh_waterlevels(cfg.problem.mesh_path)
        cell_type, cells = simplex_vtk_cell_block(3, mesh.elem, cfg.problem.elem_type)
        return mesh.coord.T, [(cell_type, cells)], {"material_id": np.asarray(mesh.material, dtype=np.int64)}

    if case in {"3d_hetero_seepage_ssr_comsol", "3d_homo_seepage_ssr"}:
        mesh = load_mesh_p2_comsol(cfg.problem.mesh_path, boundary_type=1)
        reordered = reorder_mesh_nodes(
            mesh.coord,
            mesh.elem,
            mesh.surf,
            mesh.q_mask,
            strategy=cfg.execution.node_ordering,
            n_parts=part_count,
        )
        cell_type, cells = simplex_vtk_cell_block(3, reordered.elem, cfg.problem.elem_type)
        return reordered.coord.T, [(cell_type, cells)], {"material_id": np.asarray(mesh.material, dtype=np.int64)}

    raise KeyError(f"No mesh reconstruction registered for case {case!r}")


def _points_2d(coord: np.ndarray) -> np.ndarray:
    pts = np.zeros((coord.shape[1], 3), dtype=np.float64)
    pts[:, :2] = coord.T
    return pts


def _build_field_exports(npz_path: Path, n_cells: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    point_data: dict[str, np.ndarray] = {}
    cell_data: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=True) as npz:
        if "U" in npz:
            U = np.asarray(npz["U"], dtype=np.float64)
            disp = np.zeros((U.shape[1], 3), dtype=np.float64)
            disp[:, : U.shape[0]] = U.T
            point_data["displacement"] = disp
            point_data["displacement_magnitude"] = np.linalg.norm(disp, axis=1)
        pore_key = "pw" if "pw" in npz else "seepage_pw" if "seepage_pw" in npz else None
        if pore_key is not None:
            point_data["pore_pressure"] = np.asarray(npz[pore_key], dtype=np.float64).reshape(-1)
        grad_key = "grad_p" if "grad_p" in npz else "seepage_grad_p" if "seepage_grad_p" in npz else None
        if grad_key is not None:
            grad = np.asarray(npz[grad_key], dtype=np.float64)
            if grad.ndim == 2 and grad.shape[1] % max(n_cells, 1) == 0:
                n_q = grad.shape[1] // max(n_cells, 1)
                grad_cell = grad.reshape(grad.shape[0], n_q, n_cells, order="F").mean(axis=1).T
                pad = np.zeros((n_cells, 3), dtype=np.float64)
                pad[:, : grad_cell.shape[1]] = grad_cell
                cell_data["pressure_gradient"] = pad
        sat_key = "mater_sat" if "mater_sat" in npz else "seepage_mater_sat" if "seepage_mater_sat" in npz else None
        if sat_key is not None:
            sat = np.asarray(npz[sat_key], dtype=np.float64).reshape(-1)
            if sat.size == n_cells:
                cell_data["saturation"] = sat
    return point_data, cell_data


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
        points, cell_blocks, base_cell_data = _rebuild_case_mesh(cfg)
        point_data, cell_data = _build_field_exports(npz_path, sum(block.shape[0] for _, block in cell_blocks))
        cell_data = {**base_cell_data, **cell_data}
        write_vtu(
            exports_dir / cfg.export.solution_name,
            points=points,
            cell_blocks=cell_blocks,
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
    if "output_dir" in sig.parameters:
        result = runner(Path(out_dir), **kwargs)
    elif "out_dir" in sig.parameters:
        result = runner(out_dir=Path(out_dir), **kwargs)
    else:
        raise TypeError(f"Unsupported runner signature for {runner.__module__}.{runner.__name__}")
    if PETSc.COMM_WORLD.getRank() == 0:
        output_path = Path(result["output"]) if isinstance(result, dict) and "output" in result else Path(out_dir)
        _export_outputs(cfg, args.config.resolve(), output_path)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
