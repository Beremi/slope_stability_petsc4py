#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from petsc4py import PETSc

from slope_stability.cli.assembly_policy import use_lightweight_mpi_elastic_path, use_owned_tangent_path
from slope_stability.cli.run_3D_hetero_SSR_capture import (
    _build_p4_l1_coarse_predictor_context,
    _collector_delta,
    _collector_snapshot,
    _newton_guess_difference_volume_integrals,
    _parse_petsc_opt_entries,
)
from slope_stability.constitutive import ConstitutiveOperator
from slope_stability.continuation.indirect import _secant_predictor
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
from slope_stability.utils import q_to_free_indices, flatten_field


ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = ROOT / "artifacts/p4_l1_alpha_refine_compare/rank8_secant_step12/data"
OUT_DIR = ROOT / "artifacts/p4_l1_coarse_predictor_replays"
TARGET_STEPS = (3, 4, 5, 6, 7)


def _load_params() -> tuple[dict, np.lib.npyio.NpzFile]:
    run_info = json.loads((STATE_DIR / "run_info.json").read_text())
    state_npz = np.load(STATE_DIR / "petsc_run.npz", allow_pickle=True)
    return run_info, state_npz


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
        "pmg_hierarchy": pmg_hierarchy,
        "preconditioner_options": preconditioner_options,
        "mesh_path": mesh_path,
        "partition_count": partition_count,
    }


def _step_inputs(state_npz, target_step: int) -> dict[str, object]:
    step_u = np.asarray(state_npz["step_U"], dtype=np.float64)
    omega_hist = np.asarray(state_npz["omega_hist"], dtype=np.float64)
    lambda_hist = np.asarray(state_npz["lambda_hist"], dtype=np.float64)
    current_index = target_step - 2
    previous_index = target_step - 3
    return {
        "U_old": np.asarray(step_u[previous_index], dtype=np.float64),
        "U": np.asarray(step_u[current_index], dtype=np.float64),
        "omega_old": float(omega_hist[previous_index]),
        "omega": float(omega_hist[current_index]),
        "omega_target": float(omega_hist[target_step - 1]),
        "lambda_value": float(lambda_hist[current_index]),
        "predictor_omega_hist": tuple(float(v) for v in omega_hist[: target_step - 1]),
        "predictor_lambda_hist": tuple(float(v) for v in lambda_hist[: target_step - 1]),
        "predictor_u_hist": tuple(np.asarray(v, dtype=np.float64) for v in step_u[: target_step - 1]),
        "continuation_increment_hist": tuple(
            np.asarray(step_u[i] - step_u[i - 1], dtype=np.float64) for i in range(2, target_step - 1)
        ),
    }


def _run_newton_from_predictor(run_info: dict, state_npz, target_step: int, predictor_kind: str, predictor_fn=None) -> dict[str, object]:
    built = _build_case(run_info)
    coord = built["coord"]
    elem = built["elem"]
    q_mask = built["q_mask"]
    f_V = built["f_V"]
    K_elast = built["K_elast"]
    const_builder = built["const_builder"]
    solver = built["solver"]
    params = run_info["params"]
    step_inputs = _step_inputs(state_npz, target_step)
    free_idx = q_to_free_indices(np.asarray(q_mask, dtype=bool))
    increments_free = tuple(
        np.asarray(flatten_field(np.asarray(v, dtype=np.float64))[free_idx], dtype=np.float64)
        for v in step_inputs["continuation_increment_hist"]
    )

    if predictor_kind == "secant":
        t0 = perf_counter()
        U_ini, lambda_ini, predictor_kind_actual = _secant_predictor(
            omega_old=float(step_inputs["omega_old"]),
            omega=float(step_inputs["omega"]),
            omega_target=float(step_inputs["omega_target"]),
            U_old=np.asarray(step_inputs["U_old"], dtype=np.float64),
            U=np.asarray(step_inputs["U"], dtype=np.float64),
            lambda_value=float(step_inputs["lambda_value"]),
        )
        predictor_info = {"predictor_wall_time": float(perf_counter() - t0)}
    else:
        U_ini, lambda_ini, predictor_kind_actual, predictor_info = predictor_fn(
            omega_old=float(step_inputs["omega_old"]),
            omega=float(step_inputs["omega"]),
            omega_target=float(step_inputs["omega_target"]),
            U_old=np.asarray(step_inputs["U_old"], dtype=np.float64),
            U=np.asarray(step_inputs["U"], dtype=np.float64),
            lambda_value=float(step_inputs["lambda_value"]),
            Q=q_mask,
            f=f_V,
            predictor_omega_hist=step_inputs["predictor_omega_hist"],
            predictor_lambda_hist=step_inputs["predictor_lambda_hist"],
            predictor_u_hist=step_inputs["predictor_u_hist"],
            continuation_increment_hist=step_inputs["continuation_increment_hist"],
            continuation_increment_free_hist=increments_free,
            it_newt_max=int(params["it_newt_max"]),
            it_damp_max=int(params["it_damp_max"]),
            tol=float(params["tol"]),
            r_min=float(params["r_min"]),
        )

    snap_before = _collector_snapshot(solver)
    t_newton = perf_counter()
    error = None
    try:
        U_sol, lambda_sol, flag, it_newt, _history = newton_ind_ssr(
            np.asarray(U_ini, dtype=np.float64),
            float(step_inputs["omega_target"]),
            float(lambda_ini),
            int(params["it_newt_max"]),
            int(params["it_damp_max"]),
            float(params["tol"]),
            float(params["r_min"]),
            K_elast,
            q_mask,
            f_V,
            const_builder,
            solver,
        )
    except Exception as exc:
        U_sol = None
        lambda_sol = np.nan
        flag = -999
        it_newt = np.nan
        error = repr(exc)
    newton_wall = float(perf_counter() - t_newton)
    snap_after = _collector_snapshot(snap_before | {}) if False else _collector_snapshot(solver)
    delta = _collector_delta(snap_before, snap_after)

    if U_sol is not None:
        guess_diag = _newton_guess_difference_volume_integrals(coord, elem, str(params["elem_type"]), U_ini, U_sol)
        disp_int = float(guess_diag["displacement_diff_volume_integral"])
        dev_int = float(guess_diag["deviatoric_strain_diff_volume_integral"])
    else:
        disp_int = np.nan
        dev_int = np.nan

    close_solver = getattr(solver, "close", None)
    if callable(close_solver):
        close_solver()

    return {
        "target_step": int(target_step),
        "predictor_kind": str(predictor_kind_actual),
        "predictor_wall_time": float(predictor_info.get("predictor_wall_time", np.nan)),
        "predictor_coarse_solve_wall_time": float(predictor_info.get("coarse_solve_wall_time", np.nan)),
        "predictor_coarse_newton_iterations": float(predictor_info.get("coarse_newton_iterations", np.nan)),
        "predictor_coarse_residual_end": float(predictor_info.get("coarse_residual_end", np.nan)),
        "predictor_error": error if error is not None else predictor_info.get("fallback_error"),
        "lambda_initial": float(lambda_ini),
        "lambda_solution": float(lambda_sol) if np.isfinite(lambda_sol) else np.nan,
        "newton_flag": int(flag),
        "newton_iterations": int(it_newt) if np.isfinite(it_newt) else None,
        "newton_wall_time": float(newton_wall),
        "linear_iterations": int(delta["iterations"]),
        "linear_solve_time": float(delta["solve_time"]),
        "linear_preconditioner_time": float(delta["preconditioner_time"]),
        "linear_orthogonalization_time": float(delta["orthogonalization_time"]),
        "u_init_to_solution_displacement_integral": float(disp_int),
        "u_init_to_solution_deviatoric_integral": float(dev_int),
    }


def _write_report(results: dict[int, dict[str, dict[str, object]]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(results, indent=2))
    lines = [
        "# P4(L1) Coarse-P1 Predictor Replays",
        "",
        "Source saved secant branch:",
        f"- [run_info.json]({STATE_DIR / 'run_info.json'})",
        "",
        "Each row is a cold rank-8 one-step replay from the saved secant branch.",
        "The coarse predictor uses a persistent coarse `P1(L1)` branch and prolongates its converged displacement to the fine `P4(L1)` problem.",
        "",
        "| Step | Predictor | Flag | Newton iters | Linear iters | Predictor wall [s] | Coarse solve wall [s] | Newton wall [s] | `u_ini -> u_final` disp. int. | `u_ini -> u_final` dev. int. | Error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for step in sorted(results):
        for kind in ("secant", "coarse"):
            row = results[step][kind]
            lines.append(
                f"| {step} | `{row['predictor_kind']}` | {row['newton_flag']} | "
                f"{row['newton_iterations'] if row['newton_iterations'] is not None else '-'} | "
                f"{row['linear_iterations']} | {row['predictor_wall_time']:.3f} | "
                f"{row['predictor_coarse_solve_wall_time']:.3f} | {row['newton_wall_time']:.3f} | "
                f"{row['u_init_to_solution_displacement_integral']:.3f} | {row['u_init_to_solution_deviatoric_integral']:.3f} | "
                f"{row['predictor_error'] or ''} |"
            )
    (OUT_DIR / "README.md").write_text("\n".join(lines))


def main() -> None:
    run_info, state_npz = _load_params()
    context_case = _build_case(run_info)
    params = run_info["params"]
    predictor_context = _build_p4_l1_coarse_predictor_context(
        mesh_path=context_case["mesh_path"],
        mesh_boundary_type=int(params["mesh_boundary_type"]),
        node_ordering=str(params["node_ordering"]),
        reorder_parts=context_case["partition_count"],
        material_rows=np.asarray(params["material_rows"], dtype=np.float64).tolist(),
        davis_type=str(params["davis_type"]),
        constitutive_mode=str(params["constitutive_mode"]),
        tangent_kernel=str(params["tangent_kernel"]),
        solver_type=str(run_info["run_info"]["solver_type"]),
        linear_tolerance=1.0e-1,
        linear_max_iter=100,
        lambda_init=float(params["lambda_init"]),
        d_lambda_init=float(params["d_lambda_init"]),
        d_lambda_min=float(params["d_lambda_min"]),
        it_newt_max=int(params["it_newt_max"]),
        it_damp_max=int(params["it_damp_max"]),
        tol=float(params["tol"]),
        r_min=float(params["r_min"]),
        fine_q_mask=context_case["q_mask"],
        fine_f_V=context_case["f_V"],
        fine_constitutive_matrix_builder=context_case["const_builder"],
        pmg_hierarchy=context_case["pmg_hierarchy"],
        preconditioner_options=context_case["preconditioner_options"],
    )
    close_solver = getattr(context_case["solver"], "close", None)
    if callable(close_solver):
        close_solver()

    coarse_predictor = predictor_context["coarse_p1_solution"]
    results: dict[int, dict[str, dict[str, object]]] = {}
    for step in TARGET_STEPS:
        if PETSc.COMM_WORLD.rank == 0:
            print(f"[replay] accepted step {step}")
        results[step] = {
            "secant": _run_newton_from_predictor(run_info, state_npz, step, "secant"),
            "coarse": _run_newton_from_predictor(run_info, state_npz, step, "coarse_p1_solution", predictor_fn=coarse_predictor),
        }
    if PETSc.COMM_WORLD.rank == 0:
        _write_report(results)
    cleanup = predictor_context.get("cleanup")
    if callable(cleanup):
        cleanup()


if __name__ == "__main__":
    main()
