#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from petsc4py import PETSc

from slope_stability.cli.assembly_policy import use_lightweight_mpi_elastic_path, use_owned_tangent_path
from slope_stability.cli.run_3D_hetero_SSR_capture import (
    _collector_delta,
    _collector_snapshot,
    _newton_guess_difference_volume_integrals,
    _parse_petsc_opt_entries,
)
from slope_stability.constitutive import ConstitutiveOperator
from slope_stability.continuation.indirect import _secant_predictor, _three_param_penalty_predictor
from slope_stability.fem import (
    assemble_owned_elastic_rows_for_comm,
    assemble_strain_operator,
    prepare_owned_tangent_pattern,
    quadrature_volume_3d,
    vector_volume,
)
from slope_stability.linear import SolverFactory
from slope_stability.linear.pmg import build_3d_pmg_hierarchy
from slope_stability.mesh import MaterialSpec, heterogenous_materials, load_mesh_from_file, reorder_mesh_nodes
from slope_stability.nonlinear.newton import newton_ind_ssr
from slope_stability.utils import local_csr_to_petsc_aij_matrix


ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = ROOT / "artifacts/p4_l1_alpha_refine_compare/rank8_secant_step12/data"
TARGET13_DIR = ROOT / "artifacts/p4_l1_step13_three_param_compare/rank8_secant_step13/data"
OUT_DIR = ROOT / "artifacts/p4_l1_step13_three_param_replay"


def _load_params() -> tuple[dict, dict, np.lib.npyio.NpzFile, np.lib.npyio.NpzFile]:
    run_info = json.loads((STATE_DIR / "run_info.json").read_text())
    target_info = json.loads((TARGET13_DIR / "run_info.json").read_text())
    state_npz = np.load(STATE_DIR / "petsc_run.npz", allow_pickle=True)
    target_npz = np.load(TARGET13_DIR / "petsc_run.npz", allow_pickle=True)
    return run_info, target_info, state_npz, target_npz


def _material_specs(rows: list[list[float]]) -> list[MaterialSpec]:
    return [
        MaterialSpec(
            c0=float(row[0]),
            phi=float(row[1]),
            psi=float(row[2]),
            young=float(row[3]),
            poisson=float(row[4]),
            gamma_sat=float(row[5]),
            gamma_unsat=float(row[6]),
        )
        for row in rows
    ]


def _build_case(run_info: dict):
    params = run_info["params"]
    mesh_info = run_info["mesh"]
    mesh_path = Path(mesh_info["mesh_file"])
    node_ordering = str(params["node_ordering"])
    elem_type = str(params["elem_type"]).upper()
    mesh_boundary_type = int(params["mesh_boundary_type"])
    mat_props = params["material_rows"]
    materials = _material_specs(mat_props)
    partition_count = int(PETSc.COMM_WORLD.getSize()) if node_ordering.lower() == "block_metis" else None
    pmg_hierarchy = build_3d_pmg_hierarchy(
        mesh_path,
        boundary_type=mesh_boundary_type,
        node_ordering=node_ordering,
        reorder_parts=partition_count,
        material_rows=np.asarray(mat_props, dtype=np.float64).tolist(),
        comm=PETSc.COMM_WORLD,
    )
    coord = pmg_hierarchy.fine_level.coord.astype(np.float64)
    elem = pmg_hierarchy.fine_level.elem.astype(np.int64)
    surf = pmg_hierarchy.fine_level.surf.astype(np.int64)
    q_mask = pmg_hierarchy.fine_level.q_mask.astype(bool)
    mesh = load_mesh_from_file(mesh_path, boundary_type=mesh_boundary_type, elem_type=elem_type)
    material_identifier = np.asarray(mesh.material, dtype=np.int64).ravel()

    n_q = int(quadrature_volume_3d(elem_type)[0].shape[1])
    n_int = int(elem.shape[1] * n_q)
    c0, phi, psi, shear, bulk, lame, gamma = heterogenous_materials(
        material_identifier,
        np.ones(n_int, dtype=bool),
        n_q,
        materials,
    )

    solver_type = str(run_info["run_info"]["solver_type"])
    mpi_distribute_by_nodes = bool(params["mpi_distribute_by_nodes"])
    use_owned_mpi_tangent_path = use_owned_tangent_path(
        solver_type=solver_type,
        mpi_distribute_by_nodes=mpi_distribute_by_nodes,
    )
    use_lightweight_mpi_path = use_lightweight_mpi_elastic_path(
        solver_type=solver_type,
        mpi_distribute_by_nodes=mpi_distribute_by_nodes,
        constitutive_mode=str(params["constitutive_mode"]),
    )

    B = None
    weight = np.zeros(n_int, dtype=np.float64)
    elastic_rows = None
    tangent_pattern = None
    if use_lightweight_mpi_path:
        elastic_rows = assemble_owned_elastic_rows_for_comm(
            coord,
            elem,
            q_mask,
            material_identifier,
            materials,
            PETSc.COMM_WORLD,
            elem_type=elem_type,
        )
        global_size = int(coord.shape[0] * coord.shape[1])
        K_elast = local_csr_to_petsc_aij_matrix(
            elastic_rows.local_matrix,
            global_shape=(global_size, global_size),
            comm=PETSc.COMM_WORLD,
            block_size=coord.shape[0],
        )
        rhs_parts = PETSc.COMM_WORLD.tompi4py().allgather(np.asarray(elastic_rows.local_rhs, dtype=np.float64))
        f_V = np.concatenate(rhs_parts).reshape(coord.shape[0], coord.shape[1], order="F")
    else:
        assembly = assemble_strain_operator(coord, elem, elem_type, dim=3)
        from slope_stability.fem.assembly import build_elastic_stiffness_matrix

        K_elast, weight, B = build_elastic_stiffness_matrix(assembly, shear, lame, bulk)
        f_v_int = np.vstack(
            (
                np.zeros(assembly.n_int, dtype=np.float64),
                -gamma.astype(np.float64),
                np.zeros(assembly.n_int, dtype=np.float64),
            )
        )
        f_V = vector_volume(assembly, f_v_int, weight)

    const_builder = ConstitutiveOperator(
        B=B,
        c0=c0,
        phi=phi,
        psi=psi,
        Davis_type=str(params["davis_type"]),
        shear=shear,
        bulk=bulk,
        lame=lame,
        WEIGHT=weight,
        n_strain=6,
        n_int=n_int,
        dim=3,
        q_mask=q_mask,
    )

    if use_owned_mpi_tangent_path:
        from slope_stability.utils import owned_block_range

        row0, row1 = owned_block_range(coord.shape[1], coord.shape[0], PETSc.COMM_WORLD)
        tangent_pattern = prepare_owned_tangent_pattern(
            coord,
            elem,
            q_mask,
            material_identifier,
            materials,
            (row0 // coord.shape[0], row1 // coord.shape[0]),
            elem_type=elem_type,
            include_unique=(str(params["constitutive_mode"]).lower() != "overlap"),
            include_legacy_scatter=(str(params["tangent_kernel"]).lower() == "legacy"),
            include_overlap_B=(str(params["tangent_kernel"]).lower() == "legacy"),
            elastic_rows=elastic_rows if use_lightweight_mpi_path else None,
        )
        const_builder.set_owned_tangent_pattern(
            tangent_pattern,
            use_compiled=True,
            tangent_kernel=str(params["tangent_kernel"]),
            constitutive_mode=str(params["constitutive_mode"]),
            use_compiled_constitutive=True,
        )

    preconditioner_options = {
        "threads": 16,
        "print_level": 0,
        "use_as_preconditioner": True,
        "factor_solver_type": params["factor_solver_type"],
        "pc_backend": params["pc_backend"],
        "pmg_coarse_mesh_path": None,
        "preconditioner_matrix_source": str(params["preconditioner_matrix_source"]),
        "preconditioner_matrix_policy": str(params["preconditioner_matrix_policy"]),
        "preconditioner_rebuild_policy": str(params["preconditioner_rebuild_policy"]),
        "preconditioner_rebuild_interval": int(params["preconditioner_rebuild_interval"]),
        "mpi_distribute_by_nodes": bool(params["mpi_distribute_by_nodes"]),
        "use_coordinates": True,
        "max_deflation_basis_vectors": 16,
        "full_system_preconditioner": False,
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_max_it": 3,
        "mg_levels_pc_type": "sor",
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "hypre",
        "mg_coarse_pc_hypre_type": "boomeramg",
        "pmg_hierarchy": pmg_hierarchy,
    }
    preconditioner_options.update(_parse_petsc_opt_entries(params["petsc_opt"]))

    linear_system_solver = SolverFactory.create(
        solver_type,
        tolerance=1.0e-1,
        max_iterations=100,
        deflation_basis_tolerance=1e-3,
        verbose=False,
        q_mask=q_mask,
        coord=coord,
        preconditioner_options=preconditioner_options,
    )
    return {
        "coord": coord,
        "elem": elem,
        "q_mask": q_mask,
        "f_V": f_V,
        "K_elast": K_elast,
        "const_builder": const_builder,
        "solver": linear_system_solver,
    }


def _run_one_case(case_name: str, run_info: dict, state_npz, target_npz, predictor: str) -> dict:
    built = _build_case(run_info)
    coord = built["coord"]
    elem = built["elem"]
    q_mask = built["q_mask"]
    f_V = built["f_V"]
    K_elast = built["K_elast"]
    const_builder = built["const_builder"]
    solver = built["solver"]

    step_u = np.asarray(state_npz["step_U"], dtype=np.float64)
    omega_hist = np.asarray(state_npz["omega_hist"], dtype=np.float64)
    lambda_hist = np.asarray(state_npz["lambda_hist"], dtype=np.float64)
    U_im2 = np.asarray(step_u[9], dtype=np.float64)
    U_im1 = np.asarray(step_u[10], dtype=np.float64)
    U_i = np.asarray(step_u[11], dtype=np.float64)
    omega_old = float(omega_hist[10])
    omega_now = float(omega_hist[11])
    lambda_now = float(lambda_hist[11])
    omega_target = float(np.asarray(target_npz["omega_hist"], dtype=np.float64)[12])

    predictor_u_hist = tuple(np.asarray(u, dtype=np.float64) for u in step_u[:12])

    if predictor == "secant":
        t0 = perf_counter()
        U_ini, lambda_ini, predictor_kind = _secant_predictor(
            omega_old=omega_old,
            omega=omega_now,
            omega_target=omega_target,
            U_old=U_im1,
            U=U_i,
            lambda_value=lambda_now,
        )
        predictor_info = {
            "predictor_alpha": float((omega_target - omega_now) / (omega_now - omega_old)),
            "predictor_beta": np.nan,
            "predictor_gamma": np.nan,
            "energy_eval_count": np.nan,
            "energy_value": np.nan,
            "predictor_wall_time": float(perf_counter() - t0),
        }
    else:
        U_ini, lambda_ini, predictor_kind, predictor_info = _three_param_penalty_predictor(
            omega_old=omega_old,
            omega=omega_now,
            omega_target=omega_target,
            U_old=U_im1,
            U=U_i,
            lambda_value=lambda_now,
            predictor_u_hist=predictor_u_hist,
            Q=q_mask,
            f=f_V,
            constitutive_matrix_builder=const_builder,
        )

    snap_before = _collector_snapshot(solver)
    t_newton = perf_counter()
    U_sol, lambda_sol, flag, it_newt, history = newton_ind_ssr(
        U_ini,
        omega_target,
        lambda_ini,
        int(run_info["params"]["it_newt_max"]),
        int(run_info["params"]["it_damp_max"]),
        float(run_info["params"]["tol"]),
        float(run_info["params"]["r_min"]),
        K_elast,
        q_mask,
        f_V,
        const_builder,
        solver,
    )
    newton_wall = float(perf_counter() - t_newton)
    snap_after = _collector_snapshot(solver)
    delta = _collector_delta(snap_before, snap_after)
    guess_diag = _newton_guess_difference_volume_integrals(coord, elem, str(run_info["params"]["elem_type"]), U_ini, U_sol)

    close_solver = getattr(solver, "close", None)
    if callable(close_solver):
        close_solver()

    return {
        "case": case_name,
        "predictor_kind": predictor_kind,
        "predictor_alpha": float(predictor_info.get("predictor_alpha", np.nan)),
        "predictor_beta": float(predictor_info.get("predictor_beta", np.nan)),
        "predictor_gamma": float(predictor_info.get("predictor_gamma", np.nan)),
        "predictor_eval_count": float(predictor_info.get("energy_eval_count", np.nan)),
        "predictor_merit": float(predictor_info.get("energy_value", np.nan)),
        "predictor_wall_time": float(predictor_info.get("predictor_wall_time", np.nan)),
        "omega_target": omega_target,
        "lambda_initial": float(lambda_ini),
        "lambda_solution": float(lambda_sol),
        "newton_flag": int(flag),
        "newton_iterations": int(it_newt),
        "newton_wall_time": newton_wall,
        "linear_iterations": int(delta["iterations"]),
        "linear_solve_time": float(delta["solve_time"]),
        "linear_preconditioner_time": float(delta["preconditioner_time"]),
        "linear_orthogonalization_time": float(delta["orthogonalization_time"]),
        "u_init_to_solution_displacement_integral": float(guess_diag["displacement_diff_volume_integral"]),
        "u_init_to_solution_deviatoric_integral": float(guess_diag["deviatoric_strain_diff_volume_integral"]),
        "lambda_initial_abs_error": abs(float(lambda_sol) - float(lambda_ini)),
    }


def _write_report(secant: dict, new_case: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots = OUT_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    labels = ["predictor", "linear solve", "pc", "orth", "other newton"]
    secant_other = secant["newton_wall_time"] - secant["linear_solve_time"] - secant["linear_preconditioner_time"] - secant["linear_orthogonalization_time"]
    new_other = new_case["newton_wall_time"] - new_case["linear_solve_time"] - new_case["linear_preconditioner_time"] - new_case["linear_orthogonalization_time"]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, [secant["predictor_wall_time"], secant["linear_solve_time"], secant["linear_preconditioner_time"], secant["linear_orthogonalization_time"], secant_other], width, label="secant")
    plt.bar(x + width / 2, [new_case["predictor_wall_time"], new_case["linear_solve_time"], new_case["linear_preconditioner_time"], new_case["linear_orthogonalization_time"], new_other], width, label="3-param")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Time [s]")
    plt.title("One-Step Replay: Accepted Step 13 Timing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "timing_split.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    labels2 = ["Newton", "Linear", "u diff", "dev diff"]
    x2 = np.arange(len(labels2))
    plt.bar(x2 - width / 2, [secant["newton_iterations"], secant["linear_iterations"], secant["u_init_to_solution_displacement_integral"], secant["u_init_to_solution_deviatoric_integral"]], width, label="secant")
    plt.bar(x2 + width / 2, [new_case["newton_iterations"], new_case["linear_iterations"], new_case["u_init_to_solution_displacement_integral"], new_case["u_init_to_solution_deviatoric_integral"]], width, label="3-param")
    plt.xticks(x2, labels2, rotation=20)
    plt.title("One-Step Replay: Accepted Step 13 Outcome")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "outcome.png", dpi=180)
    plt.close()

    summary = {"secant": secant, "three_param": new_case}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    report = f"""# P4(L1) One-Step Replay of Accepted Step 13

Source state:

- saved secant step-12 artifact: [run_info.json](../p4_l1_alpha_refine_compare/rank8_secant_step12/data/run_info.json)

This report replays only the next continuation step from that saved state on rank `8`, with a cold solver in both cases:

- standard secant predictor
- new 3-parameter residual-plus-penalty predictor

The target omega is the secant-branch accepted-step-13 target from [run_info.json](../p4_l1_step13_three_param_compare/rank8_secant_step13/data/run_info.json).

## Comparison

| Metric | Secant | 3-Parameter |
| --- | ---: | ---: |
| Predictor kind | `{secant["predictor_kind"]}` | `{new_case["predictor_kind"]}` |
| Predictor wall time [s] | `{secant["predictor_wall_time"]:.3f}` | `{new_case["predictor_wall_time"]:.3f}` |
| Chosen alpha | `{secant["predictor_alpha"]:.6f}` | `{new_case["predictor_alpha"]:.6f}` |
| Chosen beta | `-` | `{new_case["predictor_beta"]:.6f}` |
| Chosen gamma | `-` | `{new_case["predictor_gamma"]:.6f}` |
| Merit evaluations | `{secant["predictor_eval_count"]:.0f}` | `{new_case["predictor_eval_count"]:.0f}` |
| Merit value | `{secant["predictor_merit"]:.6e}` | `{new_case["predictor_merit"]:.6e}` |
| Newton flag | `{secant["newton_flag"]}` | `{new_case["newton_flag"]}` |
| Newton iterations | `{secant["newton_iterations"]}` | `{new_case["newton_iterations"]}` |
| Linear iterations | `{secant["linear_iterations"]}` | `{new_case["linear_iterations"]}` |
| Newton wall time [s] | `{secant["newton_wall_time"]:.3f}` | `{new_case["newton_wall_time"]:.3f}` |
| Linear solve [s] | `{secant["linear_solve_time"]:.3f}` | `{new_case["linear_solve_time"]:.3f}` |
| PC apply [s] | `{secant["linear_preconditioner_time"]:.3f}` | `{new_case["linear_preconditioner_time"]:.3f}` |
| Orthogonalization [s] | `{secant["linear_orthogonalization_time"]:.3f}` | `{new_case["linear_orthogonalization_time"]:.3f}` |
| `u_ini -> u_newton` displacement integral | `{secant["u_init_to_solution_displacement_integral"]:.3f}` | `{new_case["u_init_to_solution_displacement_integral"]:.3f}` |
| `u_ini -> u_newton` deviatoric integral | `{secant["u_init_to_solution_deviatoric_integral"]:.3f}` | `{new_case["u_init_to_solution_deviatoric_integral"]:.3f}` |
| `|lambda_ini - lambda_newton|` | `{secant["lambda_initial_abs_error"]:.6e}` | `{new_case["lambda_initial_abs_error"]:.6e}` |

## Predictor Cost Relative to Newton

| Metric | Secant | 3-Parameter |
| --- | ---: | ---: |
| Predictor / Newton wall | `{(secant["predictor_wall_time"] / secant["newton_wall_time"] if secant["newton_wall_time"] else 0.0):.4f}` | `{(new_case["predictor_wall_time"] / new_case["newton_wall_time"] if new_case["newton_wall_time"] else 0.0):.4f}` |
| Predictor / linear solve | `{(secant["predictor_wall_time"] / secant["linear_solve_time"] if secant["linear_solve_time"] else 0.0):.4f}` | `{(new_case["predictor_wall_time"] / new_case["linear_solve_time"] if new_case["linear_solve_time"] else 0.0):.4f}` |

## Plots

![Timing split](plots/timing_split.png)

![Outcome](plots/outcome.png)
"""
    (OUT_DIR / "README.md").write_text(report)


def main() -> None:
    run_info, _target_info, state_npz, target_npz = _load_params()
    secant = _run_one_case("secant", run_info, state_npz, target_npz, predictor="secant")
    three_param = _run_one_case("three_param_penalty", run_info, state_npz, target_npz, predictor="three_param_penalty")
    if PETSc.COMM_WORLD.getRank() == 0:
        _write_report(secant, three_param)
    PETSc.COMM_WORLD.Barrier()


if __name__ == "__main__":
    main()
