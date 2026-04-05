#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from petsc4py import PETSc

from slope_stability.cli.run_3D_hetero_SSR_capture import run_capture


BASELINE_RUN_INFO = Path(
    "artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_smart_controller_v2_rank8_step100/data/run_info.json"
)


def _load_baseline_params() -> dict[str, object]:
    payload = json.loads(BASELINE_RUN_INFO.read_text(encoding="utf-8"))
    return dict(payload["params"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single P4(L1) smart-controller predictor case.")
    parser.add_argument(
        "--predictor",
        required=True,
        choices=(
            "secant",
            "secant_energy_alpha",
            "three_param_penalty",
            "coarse_p1_solution",
            "coarse_p1_reduced_newton",
            "reduced_newton_all_prev",
            "reduced_newton_affine_all_prev",
            "reduced_newton_window",
            "reduced_newton_increment_power",
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--step-max", type=int, default=100)
    parser.add_argument("--omega-max-stop", type=float, default=7.0e6)
    parser.add_argument("--lambda-init", type=float, default=None)
    parser.add_argument("--d-lambda-init", type=float, default=None)
    parser.add_argument("--predictor-window-size", type=int, default=None)
    parser.add_argument(
        "--predictor-use-projected-lambda",
        dest="predictor_use_projected_lambda",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--predictor-use-current-lambda",
        dest="predictor_use_projected_lambda",
        action="store_false",
    )
    parser.add_argument(
        "--predictor-refine-lambda-for-fixed-u",
        dest="predictor_refine_lambda_for_fixed_u",
        action="store_true",
        default=False,
    )
    parser.add_argument("--predictor-reduced-max-iterations", type=int, default=None)
    parser.add_argument(
        "--predictor-reduced-use-partial-result",
        dest="predictor_reduced_use_partial_result",
        action="store_true",
        default=False,
    )
    parser.add_argument("--predictor-reduced-tolerance", type=float, default=None)
    parser.add_argument("--predictor-power-order", type=int, default=None)
    parser.add_argument(
        "--predictor-power-init",
        type=str,
        default=None,
        choices=("secant", "equal_split"),
    )
    parser.add_argument("--predictor-switch-ordinal", type=int, default=None)
    parser.add_argument(
        "--predictor-switch-to",
        type=str,
        default=None,
        choices=(
            "secant",
            "secant_energy_alpha",
            "three_param_penalty",
            "coarse_p1_solution",
            "coarse_p1_reduced_newton",
            "two_step",
            "reduced_all_prev",
            "reduced_newton_all_prev",
            "reduced_newton_affine_all_prev",
            "reduced_newton_window",
            "reduced_newton_increment_power",
        ),
    )
    args = parser.parse_args()

    base_params = _load_baseline_params()
    run_capture(
        args.out_dir,
        analysis="ssr",
        mesh_path=None,
        mesh_boundary_type=int(base_params["mesh_boundary_type"]),
        elem_type=str(base_params["elem_type"]),
        davis_type=str(base_params["davis_type"]),
        material_rows=base_params["material_rows"],
        node_ordering=str(base_params["node_ordering"]),
        lambda_init=float(base_params["lambda_init"] if args.lambda_init is None else args.lambda_init),
        d_lambda_init=float(base_params["d_lambda_init"] if args.d_lambda_init is None else args.d_lambda_init),
        d_lambda_min=float(base_params["d_lambda_min"]),
        d_lambda_diff_scaled_min=float(base_params["d_lambda_diff_scaled_min"]),
        omega_max_stop=float(args.omega_max_stop),
        continuation_predictor=str(args.predictor),
        continuation_predictor_switch_ordinal=args.predictor_switch_ordinal,
        continuation_predictor_switch_to=args.predictor_switch_to,
        continuation_predictor_window_size=args.predictor_window_size,
        continuation_predictor_use_projected_lambda=args.predictor_use_projected_lambda,
        continuation_predictor_refine_lambda_for_fixed_u=args.predictor_refine_lambda_for_fixed_u,
        continuation_predictor_reduced_max_iterations=args.predictor_reduced_max_iterations,
        continuation_predictor_reduced_use_partial_result=args.predictor_reduced_use_partial_result,
        continuation_predictor_reduced_tolerance=args.predictor_reduced_tolerance,
        continuation_predictor_power_order=args.predictor_power_order,
        continuation_predictor_power_init=args.predictor_power_init,
        omega_no_increase_newton_threshold=base_params["omega_no_increase_newton_threshold"],
        omega_half_newton_threshold=base_params["omega_half_newton_threshold"],
        omega_target_newton_iterations=base_params["omega_target_newton_iterations"],
        omega_adapt_min_scale=base_params["omega_adapt_min_scale"],
        omega_adapt_max_scale=base_params["omega_adapt_max_scale"],
        omega_hard_newton_threshold=base_params["omega_hard_newton_threshold"],
        omega_hard_linear_threshold=base_params["omega_hard_linear_threshold"],
        omega_efficiency_floor=base_params["omega_efficiency_floor"],
        omega_efficiency_drop_ratio=base_params["omega_efficiency_drop_ratio"],
        omega_efficiency_window=int(base_params["omega_efficiency_window"]),
        omega_hard_shrink_scale=base_params["omega_hard_shrink_scale"],
        step_max=int(args.step_max),
        it_newt_max=int(base_params["it_newt_max"]),
        it_damp_max=int(base_params["it_damp_max"]),
        tol=float(base_params["tol"]),
        r_min=float(base_params["r_min"]),
        linear_tolerance=1.0e-1,
        linear_max_iter=100,
        solver_type="PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        pc_backend="pmg_shell",
        preconditioner_matrix_source="tangent",
        preconditioner_matrix_policy=str(base_params["preconditioner_matrix_policy"]),
        preconditioner_rebuild_policy=str(base_params["preconditioner_rebuild_policy"]),
        preconditioner_rebuild_interval=int(base_params["preconditioner_rebuild_interval"]),
        mpi_distribute_by_nodes=bool(base_params["mpi_distribute_by_nodes"]),
        pc_hypre_coarsen_type=str(base_params["pc_hypre_coarsen_type"]),
        pc_hypre_interp_type=str(base_params["pc_hypre_interp_type"]),
        pc_hypre_strong_threshold=base_params["pc_hypre_strong_threshold"],
        pc_hypre_boomeramg_max_iter=base_params["pc_hypre_boomeramg_max_iter"],
        pc_hypre_P_max=base_params["pc_hypre_P_max"],
        pc_hypre_agg_nl=base_params["pc_hypre_agg_nl"],
        pc_hypre_nongalerkin_tol=base_params["pc_hypre_nongalerkin_tol"],
        petsc_opt=list(base_params["petsc_opt"]),
        compiled_outer=bool(base_params["compiled_outer"]),
        recycle_preconditioner=bool(base_params["recycle_preconditioner"]),
        constitutive_mode=str(base_params["constitutive_mode"]),
        tangent_kernel=str(base_params["tangent_kernel"]),
        store_step_u=True,
    )
    PETSc.COMM_WORLD.Barrier()


if __name__ == "__main__":
    main()
