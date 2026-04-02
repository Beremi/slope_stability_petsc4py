#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from petsc4py import PETSc

from slope_stability.cli.run_3D_hetero_SSR_capture import run_capture


BASELINE_RUN_INFO = Path(
    "artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_rank8_step12/data/run_info.json"
)


def _load_baseline_params() -> dict[str, object]:
    payload = json.loads(BASELINE_RUN_INFO.read_text(encoding="utf-8"))
    return dict(payload["params"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single P4(L1) secant-history patch case.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--step-max", type=int, default=15)
    parser.add_argument("--omega-max-stop", type=float, default=7.0e6)
    parser.add_argument("--lambda-init", type=float, default=1.0)
    parser.add_argument("--d-lambda-init", type=float, default=0.05)
    parser.add_argument(
        "--continuation-mode",
        type=str,
        default="classic",
        choices=("classic", "streaming_microstep"),
    )
    parser.add_argument(
        "--secant-correction-mode",
        type=str,
        default="none",
        choices=("none", "orthogonal_increment_ls"),
    )
    parser.add_argument(
        "--first-newton-warm-start-mode",
        type=str,
        default="none",
        choices=("none", "history_deflation"),
    )
    parser.add_argument("--streaming-micro-target-length", type=float, default=0.15)
    parser.add_argument("--streaming-micro-min-length", type=float, default=0.05)
    parser.add_argument("--streaming-micro-max-length", type=float, default=0.30)
    parser.add_argument("--streaming-move-relres-threshold", type=float, default=5.0e-3)
    parser.add_argument("--streaming-alpha-advance-threshold", type=float, default=0.5)
    parser.add_argument("--streaming-micro-max-corrections", type=int, default=40)
    parser.add_argument("--streaming-basis-max-vectors", type=int, default=8)
    parser.add_argument("--store-step-u", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    base = _load_baseline_params()
    run_capture(
        args.out_dir,
        analysis="ssr",
        mesh_path=None,
        mesh_boundary_type=int(base["mesh_boundary_type"]),
        elem_type=str(base["elem_type"]),
        davis_type=str(base["davis_type"]),
        material_rows=base["material_rows"],
        node_ordering=str(base["node_ordering"]),
        lambda_init=float(args.lambda_init),
        d_lambda_init=float(args.d_lambda_init),
        d_lambda_min=float(base["d_lambda_min"]),
        d_lambda_diff_scaled_min=float(base["d_lambda_diff_scaled_min"]),
        lambda_ell=float(base["lambda_ell"]),
        d_omega_ini_scale=float(base["d_omega_ini_scale"]),
        d_t_min=float(base["d_t_min"]),
        omega_max_stop=float(args.omega_max_stop),
        continuation_predictor="secant",
        continuation_mode=str(args.continuation_mode),
        omega_step_controller="legacy",
        continuation_secant_correction_mode=str(args.secant_correction_mode),
        continuation_first_newton_warm_start_mode=str(args.first_newton_warm_start_mode),
        streaming_micro_target_length=float(args.streaming_micro_target_length),
        streaming_micro_min_length=float(args.streaming_micro_min_length),
        streaming_micro_max_length=float(args.streaming_micro_max_length),
        streaming_move_relres_threshold=float(args.streaming_move_relres_threshold),
        streaming_alpha_advance_threshold=float(args.streaming_alpha_advance_threshold),
        streaming_micro_max_corrections=int(args.streaming_micro_max_corrections),
        streaming_basis_max_vectors=int(args.streaming_basis_max_vectors),
        step_max=int(args.step_max),
        it_newt_max=int(base["it_newt_max"]),
        it_damp_max=int(base["it_damp_max"]),
        tol=float(base["tol"]),
        r_min=float(base["r_min"]),
        linear_tolerance=1.0e-1,
        linear_max_iter=100,
        solver_type="PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        factor_solver_type=base["factor_solver_type"],
        pc_backend="pmg_shell",
        preconditioner_matrix_source=str(base["preconditioner_matrix_source"]),
        preconditioner_matrix_policy=str(base["preconditioner_matrix_policy"]),
        preconditioner_rebuild_policy=str(base["preconditioner_rebuild_policy"]),
        preconditioner_rebuild_interval=int(base["preconditioner_rebuild_interval"]),
        mpi_distribute_by_nodes=bool(base["mpi_distribute_by_nodes"]),
        pc_gamg_process_eq_limit=base["pc_gamg_process_eq_limit"],
        pc_gamg_threshold=base["pc_gamg_threshold"],
        pc_gamg_aggressive_coarsening=base["pc_gamg_aggressive_coarsening"],
        pc_gamg_aggressive_square_graph=base["pc_gamg_aggressive_square_graph"],
        pc_gamg_aggressive_mis_k=base["pc_gamg_aggressive_mis_k"],
        pc_hypre_coarsen_type=base["pc_hypre_coarsen_type"],
        pc_hypre_interp_type=base["pc_hypre_interp_type"],
        pc_hypre_strong_threshold=base["pc_hypre_strong_threshold"],
        pc_hypre_boomeramg_max_iter=base["pc_hypre_boomeramg_max_iter"],
        pc_hypre_P_max=base["pc_hypre_P_max"],
        pc_hypre_agg_nl=base["pc_hypre_agg_nl"],
        pc_hypre_nongalerkin_tol=base["pc_hypre_nongalerkin_tol"],
        petsc_opt=list(base["petsc_opt"]),
        compiled_outer=bool(base["compiled_outer"]),
        recycle_preconditioner=bool(base["recycle_preconditioner"]),
        constitutive_mode=str(base["constitutive_mode"]),
        tangent_kernel=str(base["tangent_kernel"]),
        store_step_u=bool(args.store_step_u),
    )
    PETSc.COMM_WORLD.Barrier()


if __name__ == "__main__":
    main()
