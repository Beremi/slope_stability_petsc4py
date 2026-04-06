#!/usr/bin/env python
"""Convenience entrypoint for the 3D homogeneous SSR capture case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from petsc4py import PETSc

from .run_3D_hetero_SSR_capture import run_capture as _run_generic_capture


def run_capture(output_dir: Path, **kwargs):
    kwargs.setdefault(
        "mesh_path",
        Path(__file__).resolve().parents[3] / "meshes" / "3d_homo_ssr" / "SSR_homo_ada_L1.msh",
    )
    kwargs.setdefault("analysis", "ssr")
    kwargs.setdefault("mesh_boundary_type", 0)
    return _run_generic_capture(output_dir, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 3D homogeneous SSR capture.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--mesh_path", type=Path, default=None)
    parser.add_argument("--mesh_boundary_type", type=int, default=0)
    parser.add_argument("--elem_type", type=str, default="P2")
    parser.add_argument("--node_ordering", type=str, default="block_metis")
    parser.add_argument("--lambda_init", type=float, default=0.9)
    parser.add_argument("--d_lambda_init", type=float, default=0.1)
    parser.add_argument("--d_lambda_min", type=float, default=1e-5)
    parser.add_argument("--d_lambda_diff_scaled_min", type=float, default=1e-3)
    parser.add_argument("--omega_max_stop", type=float, default=1e6)
    parser.add_argument("--continuation_predictor", type=str, default="secant")
    parser.add_argument("--omega_step_controller", type=str, default="legacy")
    parser.add_argument("--step_max", type=int, default=100)
    parser.add_argument("--it_newt_max", type=int, default=50)
    parser.add_argument("--it_damp_max", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--r_min", type=float, default=1e-4)
    parser.add_argument(
        "--newton_stopping_criterion",
        type=str,
        default="relative_residual",
        choices=["relative_residual", "relative_correction", "absolute_delta_lambda"],
    )
    parser.add_argument("--newton_stopping_tol", type=float, default=None)
    parser.add_argument("--linear_tolerance", type=float, default=1e-1)
    parser.add_argument("--linear_max_iter", type=int, default=100)
    parser.add_argument("--preconditioner_threads", type=int, default=16)
    parser.add_argument("--solver_type", type=str, default="PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE")
    parser.add_argument("--pc_backend", type=str, default="hypre")
    parser.add_argument("--pmg_coarse_mesh_path", type=Path, default=None)
    parser.add_argument("--pmg_fine_hierarchy_mode", type=str, default="default")
    parser.add_argument("--mpi_distribute_by_nodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pc_hypre_coarsen_type", type=str, default="HMIS")
    parser.add_argument("--pc_hypre_interp_type", type=str, default="ext+i")
    parser.add_argument("--pc_hypre_strong_threshold", type=float, default=None)
    parser.add_argument("--pc_hypre_boomeramg_max_iter", type=int, default=1)
    parser.add_argument("--pc_hypre_P_max", type=int, default=None)
    parser.add_argument("--pc_hypre_agg_nl", type=int, default=None)
    parser.add_argument("--pc_hypre_nongalerkin_tol", type=float, default=None)
    parser.add_argument("--recycle_preconditioner", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--constitutive_mode", type=str, default="overlap")
    parser.add_argument("--tangent_kernel", type=str, default="rows")
    parser.add_argument("--petsc-opt", action="append", default=[], dest="petsc_opt")
    args = parser.parse_args()

    result = run_capture(
        args.out_dir,
        mesh_path=args.mesh_path,
        mesh_boundary_type=args.mesh_boundary_type,
        elem_type=args.elem_type,
        node_ordering=args.node_ordering,
        lambda_init=args.lambda_init,
        d_lambda_init=args.d_lambda_init,
        d_lambda_min=args.d_lambda_min,
        d_lambda_diff_scaled_min=args.d_lambda_diff_scaled_min,
        omega_max_stop=args.omega_max_stop,
        continuation_predictor=args.continuation_predictor,
        omega_step_controller=args.omega_step_controller,
        step_max=args.step_max,
        it_newt_max=args.it_newt_max,
        it_damp_max=args.it_damp_max,
        tol=args.tol,
        r_min=args.r_min,
        newton_stopping_criterion=args.newton_stopping_criterion,
        newton_stopping_tol=args.newton_stopping_tol,
        linear_tolerance=args.linear_tolerance,
        linear_max_iter=args.linear_max_iter,
        preconditioner_threads=args.preconditioner_threads,
        solver_type=args.solver_type,
        pc_backend=args.pc_backend,
        pmg_coarse_mesh_path=args.pmg_coarse_mesh_path,
        pmg_fine_hierarchy_mode=args.pmg_fine_hierarchy_mode,
        mpi_distribute_by_nodes=args.mpi_distribute_by_nodes,
        pc_hypre_coarsen_type=args.pc_hypre_coarsen_type,
        pc_hypre_interp_type=args.pc_hypre_interp_type,
        pc_hypre_strong_threshold=args.pc_hypre_strong_threshold,
        pc_hypre_boomeramg_max_iter=args.pc_hypre_boomeramg_max_iter,
        pc_hypre_P_max=args.pc_hypre_P_max,
        pc_hypre_agg_nl=args.pc_hypre_agg_nl,
        pc_hypre_nongalerkin_tol=args.pc_hypre_nongalerkin_tol,
        recycle_preconditioner=args.recycle_preconditioner,
        constitutive_mode=args.constitutive_mode,
        tangent_kernel=args.tangent_kernel,
        petsc_opt=args.petsc_opt,
    )
    if PETSc.COMM_WORLD.getRank() == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
