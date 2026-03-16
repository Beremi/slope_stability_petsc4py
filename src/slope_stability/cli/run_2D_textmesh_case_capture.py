#!/usr/bin/env python
"""Run MATLAB-style 2D text-mesh cases through the PETSc stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from slope_stability.constitutive import ConstitutiveOperator
from slope_stability.continuation import LL_indirect_continuation, SSR_direct_continuation, SSR_indirect_continuation
from slope_stability.fem import (
    assemble_owned_elastic_rows_for_comm,
    assemble_strain_operator,
    local_basis_volume_2d,
    prepare_owned_tangent_pattern,
    quadrature_volume_2d,
    vector_volume,
)
from slope_stability.linear import SolverFactory
from slope_stability.mesh import (
    MaterialSpec,
    franz_dam_pressure_boundary,
    heterogenous_materials,
    load_mesh_franz_dam_2d,
    load_mesh_kozinec_2d,
    load_mesh_luzec_2d,
    luzec_pressure_boundary,
    reorder_mesh_nodes,
)
from slope_stability.nonlinear.newton import _destroy_petsc_mat, _prefers_full_system_operator, _setup_linear_system, _solve_linear_system
from slope_stability.seepage import heter_conduct, seepage_problem_2d
from slope_stability.utils import extract_submatrix_free, local_csr_to_petsc_aij_matrix, owned_block_range, q_to_free_indices


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_progress_logger(progress_dir: Path):
    progress_jsonl = progress_dir / "progress.jsonl"
    progress_latest = progress_dir / "progress_latest.json"

    def _write(event: dict) -> None:
        payload = {"timestamp": np.datetime64("now").astype(str), **event}
        with progress_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
            handle.flush()
        progress_latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return _write


def _collector_snapshot(solver) -> dict:
    collector = solver.iteration_collector
    return {
        "iterations": collector.get_total_iterations(),
        "solve_time": collector.get_total_solve_time(),
        "preconditioner_time": collector.get_total_preconditioner_time(),
        "orthogonalization_time": collector.get_total_orthogonalization_time(),
    }


def _collector_delta(before: dict, after: dict) -> dict:
    return {
        "iterations": after["iterations"] - before["iterations"],
        "solve_time": after["solve_time"] - before["solve_time"],
        "preconditioner_time": after["preconditioner_time"] - before["preconditioner_time"],
        "orthogonalization_time": after["orthogonalization_time"] - before["orthogonalization_time"],
    }


def _kozinec_saturation(coord: np.ndarray, elem: np.ndarray, hatp: np.ndarray) -> np.ndarray:
    n_p = int(elem.shape[0])
    n_e = int(elem.shape[1])
    n_q = int(hatp.shape[1])
    hatphi = np.tile(np.asarray(hatp, dtype=np.float64), (1, n_e))
    coord_x = np.reshape(coord[0, elem.reshape(-1, order="F")], (n_p, n_e), order="F")
    coord_y = np.reshape(coord[1, elem.reshape(-1, order="F")], (n_p, n_e), order="F")
    coord_x_int = np.sum(np.kron(coord_x, np.ones((1, n_q), dtype=np.float64)) * hatphi, axis=0)
    coord_y_int = np.sum(np.kron(coord_y, np.ones((1, n_q), dtype=np.float64)) * hatphi, axis=0)

    level = np.empty_like(coord_x_int)
    x = coord_x_int
    level[x <= 44.0] = 59.0 - (59.0 - 55.0) * (x[x <= 44.0] / 44.0)
    mask = (x > 44.0) & (x <= 116.0)
    level[mask] = 55.0 - (55.0 - 39.0) * ((x[mask] - 44.0) / (116.0 - 44.0))
    mask = (x > 116.0) & (x <= 149.0)
    level[mask] = 39.0 - (39.0 - 32.0) * ((x[mask] - 116.0) / (149.0 - 116.0))
    mask = (x > 149.0) & (x <= 165.0)
    level[mask] = 32.0 - (32.0 - 27.0) * ((x[mask] - 149.0) / (165.0 - 149.0))
    mask = (x > 165.0) & (x <= 194.0)
    level[mask] = 27.0 - (27.0 - 24.0) * ((x[mask] - 165.0) / (194.0 - 165.0))
    mask = (x > 194.0) & (x <= 232.0)
    level[mask] = 24.0 - (24.0 - 20.0) * ((x[mask] - 194.0) / (232.0 - 194.0))
    level[x > 232.0] = 20.0
    return coord_y_int <= level


def _load_case_mesh(case_name: str, elem_type: str, mesh_dir: Path):
    case_key = case_name.lower()
    if case_key == "kozinec":
        return load_mesh_kozinec_2d(elem_type, mesh_dir)
    if case_key == "luzec":
        return load_mesh_luzec_2d(elem_type, mesh_dir)
    if case_key == "franz_dam":
        return load_mesh_franz_dam_2d(elem_type, mesh_dir)
    raise KeyError(f"Unsupported 2D text-mesh case {case_name!r}")


def run_capture(
    output_dir: Path,
    *,
    case_name: str,
    analysis: str = "ssr",
    continuation_method: str = "indirect",
    mesh_dir: Path,
    elem_type: str = "P2",
    davis_type: str = "B",
    material_rows: list[list[float]] | np.ndarray,
    hydraulic_conductivity: list[float] | np.ndarray | None = None,
    node_ordering: str = "block_metis",
    lambda_init: float = 0.7,
    d_lambda_init: float = 0.1,
    d_lambda_min: float = 1e-5,
    d_lambda_diff_scaled_min: float = 1e-3,
    lambda_ell: float = 1.0,
    d_omega_ini_scale: float = 1.0 / 30.0,
    d_t_min: float = 1e-3,
    omega_max_stop: float = 7.0e7,
    step_max: int = 100,
    it_newt_max: int = 50,
    it_damp_max: int = 10,
    tol: float = 1e-4,
    r_min: float = 1e-4,
    linear_tolerance: float = 1e-1,
    linear_max_iter: int = 100,
    solver_type: str = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
    mpi_distribute_by_nodes: bool = True,
    pc_hypre_coarsen_type: str | None = "HMIS",
    pc_hypre_interp_type: str | None = "ext+i",
    pc_hypre_strong_threshold: float | None = None,
    recycle_preconditioner: bool = True,
    constitutive_mode: str = "overlap",
    tangent_kernel: str = "rows",
    seepage_linear_tolerance: float = 1e-10,
    seepage_linear_max_iter: int = 500,
    seepage_water_unit_weight: float = 9.81,
) -> dict:
    rank = int(PETSc.COMM_WORLD.getRank())
    out_dir = _ensure_dir(output_dir) if rank == 0 else output_dir
    data_dir = out_dir / "data"
    progress_callback = None
    if rank == 0:
        _ensure_dir(data_dir)
        progress_callback = _make_progress_logger(data_dir)

    case_key = str(case_name).lower()
    analysis_key = str(analysis).lower()
    method_key = str(continuation_method).lower()

    mesh = _load_case_mesh(case_key, elem_type, Path(mesh_dir))
    partition_count = int(PETSc.COMM_WORLD.getSize()) if str(node_ordering).lower() == "block_metis" else None
    reordered = reorder_mesh_nodes(mesh.coord, mesh.elem, mesh.surf, mesh.q_mask, strategy=node_ordering, n_parts=partition_count)

    coord = reordered.coord.astype(np.float64)
    elem = reordered.elem.astype(np.int64)
    surf = reordered.surf.astype(np.int64)
    q_mask = reordered.q_mask.astype(bool)
    material_identifier = np.asarray(mesh.material, dtype=np.int64)

    material_rows_arr = np.asarray(material_rows, dtype=np.float64)
    materials = [
        MaterialSpec(
            c0=float(row[0]),
            phi=float(row[1]),
            psi=float(row[2]),
            young=float(row[3]),
            poisson=float(row[4]),
            gamma_sat=float(row[5]),
            gamma_unsat=float(row[6]),
        )
        for row in material_rows_arr
    ]

    xi, wf = quadrature_volume_2d(elem_type)
    n_q = int(wf.shape[0])
    n_int = int(elem.shape[1] * n_q)

    seepage_payload: dict[str, np.ndarray] = {}
    if case_key == "kozinec":
        hatp, _, _ = local_basis_volume_2d(elem_type, xi)
        saturation = _kozinec_saturation(coord, elem, hatp)
    else:
        if hydraulic_conductivity is None:
            raise ValueError(f"hydraulic_conductivity is required for seepage-coupled case {case_name!r}")
        conduct0 = heter_conduct(material_identifier, n_q, np.asarray(hydraulic_conductivity, dtype=np.float64))
        if case_key == "luzec":
            q_w, pw_d = luzec_pressure_boundary(coord, surf, float(seepage_water_unit_weight))
        elif case_key == "franz_dam":
            q_w, pw_d = franz_dam_pressure_boundary(coord, surf, float(seepage_water_unit_weight))
        else:
            raise KeyError(case_name)
        seepage_solver = SolverFactory.create(
            solver_type.replace("_NULLSPACE", ""),
            tolerance=seepage_linear_tolerance,
            max_iterations=seepage_linear_max_iter,
            deflation_basis_tolerance=1.0e-3,
            verbose=False,
            q_mask=None,
            coord=None,
            preconditioner_options={
                "threads": 16,
                "print_level": 0,
                "use_as_preconditioner": True,
                "pc_hypre_boomeramg_coarsen_type": pc_hypre_coarsen_type,
                "pc_hypre_boomeramg_interp_type": pc_hypre_interp_type,
            },
        )
        pw, grad_p, mater_sat, seep_history, _ = seepage_problem_2d(
            coord,
            elem,
            q_w,
            pw_d,
            float(seepage_water_unit_weight),
            conduct0,
            elem_type=elem_type,
            linear_system_solver=seepage_solver,
            it_max=50,
            tol=1.0e-10,
        )
        saturation = np.repeat(np.asarray(mater_sat, dtype=bool), n_q)
        seepage_payload = {
            "pw": np.asarray(pw, dtype=np.float64),
            "grad_p": np.asarray(grad_p, dtype=np.float64),
            "mater_sat": np.asarray(mater_sat, dtype=np.float64),
            "seepage_linear_iterations": np.asarray(seep_history["linear_iterations"], dtype=np.int64),
            "seepage_criterion": np.asarray(seep_history["criterion"], dtype=np.float64),
        }

    c0, phi, psi, shear, bulk, lame, gamma = heterogenous_materials(
        material_identifier,
        saturation,
        n_q,
        materials,
    )

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
    rhs_parts = MPI.COMM_WORLD.allgather(np.asarray(elastic_rows.local_rhs, dtype=np.float64))
    f_V = np.concatenate(rhs_parts).reshape(coord.shape[0], coord.shape[1], order="F")
    if seepage_payload:
        grad_p = seepage_payload["grad_p"]
        f_V_int = np.vstack((-grad_p[0, :], -grad_p[1, :] - gamma))
        f_asm = assemble_strain_operator(coord, elem, elem_type, dim=2)
        f_V = vector_volume(f_asm, f_V_int)

    const_builder = ConstitutiveOperator(
        B=None,
        c0=c0,
        phi=phi,
        psi=psi,
        Davis_type=davis_type,
        shear=shear,
        bulk=bulk,
        lame=lame,
        WEIGHT=np.zeros(n_int, dtype=np.float64),
        n_strain=3,
        n_int=n_int,
        dim=2,
        q_mask=q_mask,
    )

    row0, row1 = owned_block_range(coord.shape[1], coord.shape[0], PETSc.COMM_WORLD)
    tangent_pattern = prepare_owned_tangent_pattern(
        coord,
        elem,
        q_mask,
        material_identifier,
        materials,
        (row0 // coord.shape[0], row1 // coord.shape[0]),
        elem_type=elem_type,
        include_unique=(str(constitutive_mode).lower() != "overlap"),
        include_legacy_scatter=(str(tangent_kernel).lower() == "legacy"),
        elastic_rows=elastic_rows,
    )
    const_builder.set_owned_tangent_pattern(
        tangent_pattern,
        use_compiled=True,
        tangent_kernel=tangent_kernel,
        constitutive_mode=constitutive_mode,
        use_compiled_constitutive=True,
    )

    preconditioner_options = {
        "threads": 16,
        "print_level": 0,
        "use_as_preconditioner": True,
        "mpi_distribute_by_nodes": bool(mpi_distribute_by_nodes),
        "use_coordinates": True,
        "pc_hypre_boomeramg_coarsen_type": pc_hypre_coarsen_type,
        "pc_hypre_boomeramg_interp_type": pc_hypre_interp_type,
    }
    if recycle_preconditioner:
        preconditioner_options["recycle_preconditioner"] = True
    if pc_hypre_strong_threshold is not None:
        preconditioner_options["pc_hypre_boomeramg_strong_threshold"] = float(pc_hypre_strong_threshold)

    linear_system_solver = SolverFactory.create(
        solver_type,
        tolerance=linear_tolerance,
        max_iterations=linear_max_iter,
        deflation_basis_tolerance=1.0e-3,
        verbose=False,
        q_mask=q_mask,
        coord=coord,
        preconditioner_options=preconditioner_options,
    )

    params = {
        "case_name": case_key,
        "analysis": analysis_key,
        "continuation_method": method_key,
        "elem_type": elem_type,
        "davis_type": davis_type,
        "material_rows": material_rows_arr.tolist(),
        "hydraulic_conductivity": None if hydraulic_conductivity is None else np.asarray(hydraulic_conductivity, dtype=np.float64).tolist(),
        "node_ordering": node_ordering,
        "lambda_init": float(lambda_init),
        "d_lambda_init": float(d_lambda_init),
        "d_lambda_min": float(d_lambda_min),
        "d_lambda_diff_scaled_min": float(d_lambda_diff_scaled_min),
        "lambda_ell": float(lambda_ell),
        "d_omega_ini_scale": float(d_omega_ini_scale),
        "d_t_min": float(d_t_min),
        "omega_max_stop": float(omega_max_stop),
        "step_max": int(step_max),
        "it_newt_max": int(it_newt_max),
        "it_damp_max": int(it_damp_max),
        "tol": float(tol),
        "r_min": float(r_min),
        "linear_tolerance": float(linear_tolerance),
        "linear_max_iter": int(linear_max_iter),
        "solver_type": solver_type,
        "pc_hypre_coarsen_type": pc_hypre_coarsen_type,
        "pc_hypre_interp_type": pc_hypre_interp_type,
        "pc_hypre_strong_threshold": pc_hypre_strong_threshold,
        "recycle_preconditioner": bool(recycle_preconditioner),
        "constitutive_mode": constitutive_mode,
        "tangent_kernel": str(tangent_kernel),
        "mesh_dir": str(mesh_dir),
    }

    t0 = perf_counter()
    init_linear = {
        "init_linear_iterations": 0,
        "init_linear_solve_time": 0.0,
        "init_linear_preconditioner_time": 0.0,
        "init_linear_orthogonalization_time": 0.0,
    }
    stats: dict = {}
    work_hist = None
    if analysis_key == "ssr" and method_key == "direct":
        U, lambda_hist, omega_hist, Umax_hist, work_hist = SSR_direct_continuation(
            lambda_init,
            d_lambda_init,
            d_lambda_min,
            d_lambda_diff_scaled_min,
            step_max,
            it_newt_max,
            it_damp_max,
            tol,
            r_min,
            K_elast,
            q_mask,
            f_V,
            const_builder,
            linear_system_solver.copy(),
        )
        step_u = np.empty((0, 2, 0), dtype=np.float64)
    elif analysis_key == "ssr":
        U, lambda_hist, omega_hist, Umax_hist, stats = SSR_indirect_continuation(
            lambda_init,
            d_lambda_init,
            d_lambda_min,
            d_lambda_diff_scaled_min,
            step_max,
            omega_max_stop,
            it_newt_max,
            it_damp_max,
            tol,
            r_min,
            K_elast,
            q_mask,
            f_V,
            const_builder,
            linear_system_solver.copy(),
            progress_callback=progress_callback,
        )
        step_u = np.asarray(stats.pop("step_U"), dtype=np.float64) if isinstance(stats.get("step_U", None), list) else np.empty((0, 2, 0), dtype=np.float64)
    else:
        free_idx = q_to_free_indices(q_mask)
        f_full = np.asarray(f_V, dtype=np.float64).reshape(-1, order="F")
        f_free = f_full[free_idx]
        snap_init_0 = _collector_snapshot(linear_system_solver)
        U_elast_free = None
        K_free = None
        const_builder.reduction(float(lambda_ell))
        try:
            if _prefers_full_system_operator(linear_system_solver, K_elast):
                _setup_linear_system(linear_system_solver, K_elast, A_full=K_elast, free_idx=free_idx)
                U_elast_free = _solve_linear_system(
                    linear_system_solver,
                    K_elast,
                    f_free,
                    b_full=f_full,
                    free_idx=free_idx,
                )
            else:
                K_free = extract_submatrix_free(K_elast, free_idx)
                _setup_linear_system(linear_system_solver, K_free, A_full=K_elast, free_idx=free_idx)
                U_elast_free = _solve_linear_system(
                    linear_system_solver,
                    K_free,
                    f_free,
                    b_full=f_full,
                    free_idx=free_idx,
                )
        finally:
            _destroy_petsc_mat(K_free)
            release = getattr(linear_system_solver, "release_iteration_resources", None)
            if callable(release):
                release()
        init_delta = _collector_delta(snap_init_0, _collector_snapshot(linear_system_solver))
        init_linear = {
            "init_linear_iterations": int(init_delta["iterations"]),
            "init_linear_solve_time": float(init_delta["solve_time"]),
            "init_linear_preconditioner_time": float(init_delta["preconditioner_time"]),
            "init_linear_orthogonalization_time": float(init_delta["orthogonalization_time"]),
        }
        U_elast = np.zeros_like(f_V)
        U_elast.reshape(-1, order="F")[free_idx] = np.asarray(U_elast_free, dtype=np.float64)
        linear_system_solver.expand_deflation_basis(np.asarray(U_elast_free, dtype=np.float64))
        omega_el = float(np.dot(f_free, np.asarray(U_elast_free, dtype=np.float64)))
        U_elast = U_elast * float(d_omega_ini_scale)
        U, lambda_hist, omega_hist, Umax_hist, stats = LL_indirect_continuation(
            omega_el * float(d_omega_ini_scale),
            d_t_min,
            step_max,
            omega_max_stop,
            it_newt_max,
            it_damp_max,
            tol,
            r_min,
            K_elast,
            U_elast,
            q_mask,
            f_V,
            const_builder,
            linear_system_solver.copy(),
            progress_callback=progress_callback,
        )
        step_u = np.asarray(stats.pop("step_U"), dtype=np.float64) if isinstance(stats.get("step_U", None), list) else np.empty((0, 2, 0), dtype=np.float64)

    runtime = perf_counter() - t0
    mpi_comm = PETSc.COMM_WORLD.tompi4py()
    const_times = const_builder.get_total_time()
    const_times_max = {key: float(mpi_comm.allreduce(float(val), op=MPI.MAX)) for key, val in const_times.items()}
    linear_summary = {
        **init_linear,
        "attempt_linear_iterations_total": int(np.sum(np.asarray(stats.get("attempt_linear_iterations", []), dtype=np.int64))),
        "attempt_linear_solve_time_total": float(np.sum(np.asarray(stats.get("attempt_linear_solve_time", []), dtype=np.float64))),
        "attempt_linear_preconditioner_time_total": float(np.sum(np.asarray(stats.get("attempt_linear_preconditioner_time", []), dtype=np.float64))),
        "attempt_linear_orthogonalization_time_total": float(np.sum(np.asarray(stats.get("attempt_linear_orthogonalization_time", []), dtype=np.float64))),
    }

    run_payload = {
        "run_info": {
            "timestamp": np.datetime64("now").astype(str),
            "runtime_seconds": float(runtime),
            "solver_type": solver_type,
            "analysis": analysis_key,
            "continuation_method": method_key,
            "case_name": case_key,
            "rank_count": int(PETSc.COMM_WORLD.getSize()),
            "step_count": int(len(lambda_hist)),
        },
        "params": params,
        "mesh": {
            "coord_shape": coord.shape,
            "elem_shape": elem.shape,
            "surf_shape": surf.shape,
        },
        "timings": {
            "constitutive": const_times_max,
            "linear": linear_summary,
            "continuation_total_wall_time": float(stats.get("total_wall_time", runtime)),
        },
    }

    if rank == 0:
        np.savez_compressed(
            data_dir / "petsc_run.npz",
            U=U,
            lambda_hist=lambda_hist,
            load_factor_hist=lambda_hist,
            omega_hist=omega_hist,
            Umax_hist=Umax_hist,
            step_U=step_u,
            **seepage_payload,
            **({"work_hist": np.asarray(work_hist, dtype=np.float64)} if work_hist is not None else {}),
            **{"stats_" + key: np.asarray(value) for key, value in stats.items()},
        )
        (data_dir / "run_info.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

    return {
        "output": str(out_dir),
        "npz": str(data_dir / "petsc_run.npz"),
        "json": str(data_dir / "run_info.json"),
        "runtime": float(runtime),
        "lambda_last": float(lambda_hist[-1]),
        "omega_last": float(omega_hist[-1]),
        "steps": int(len(lambda_hist)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PETSc 2D text-mesh continuation cases.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--case_name", type=str, required=True, choices=["kozinec", "luzec", "franz_dam"])
    parser.add_argument("--analysis", type=str, default="ssr", choices=["ssr", "ll"])
    parser.add_argument("--continuation_method", type=str, default="indirect", choices=["indirect", "direct"])
    parser.add_argument("--mesh_dir", type=Path, required=True)
    parser.add_argument("--elem_type", type=str, default="P2", choices=["P1", "P2", "P4"])
    args = parser.parse_args()
    raise SystemExit(
        "Use the config-driven entrypoint or call run_capture() directly; the CLI wrapper is intentionally minimal."
    )


if __name__ == "__main__":
    main()
