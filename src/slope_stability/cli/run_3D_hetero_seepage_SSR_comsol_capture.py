#!/usr/bin/env python
"""Run the COMSOL-based 3D heterogeneous seepage SSR continuation case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from slope_stability.core.elements import validate_supported_elem_type
from slope_stability.cli.progress import make_progress_logger
from slope_stability.constitutive import ConstitutiveOperator
from slope_stability.continuation import SSR_indirect_continuation
from slope_stability.fem import (
    assemble_owned_elastic_rows_for_comm,
    quadrature_volume_3d,
    prepare_owned_tangent_pattern,
)
from slope_stability.linear import SolverFactory
from slope_stability.mesh import (
    MaterialSpec,
    heterogenous_materials,
    load_mesh_p2_comsol,
    reorder_mesh_nodes,
    seepage_boundary_3d_hetero_comsol,
)
from slope_stability.problem_assets import load_material_rows_for_path
from slope_stability.seepage import heter_conduct, seepage_problem_3d
from slope_stability.utils import local_csr_to_petsc_aij_matrix, owned_block_range


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_progress_logger(progress_dir: Path):
    return make_progress_logger(progress_dir)


def run_capture(
    output_dir: Path,
    *,
    mesh_path: Path | None = None,
    elem_type: str = "P2",
    node_ordering: str = "block_metis",
    lambda_init: float = 1.0,
    d_lambda_init: float = 0.1,
    d_lambda_min: float = 1e-5,
    d_lambda_diff_scaled_min: float = 0.005,
    omega_max_stop: float = 3.407e8,
    continuation_predictor: str = "secant",
    step_max: int = 100,
    it_newt_max: int = 50,
    it_damp_max: int = 10,
    tol: float = 1e-4,
    r_min: float = 1e-4,
    linear_tolerance: float = 1e-1,
    linear_max_iter: int = 100,
    solver_type: str = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
    mpi_distribute_by_nodes: bool = True,
    pc_hypre_coarsen_type: str = "HMIS",
    pc_hypre_interp_type: str = "ext+i",
    pc_hypre_strong_threshold: float | None = None,
    recycle_preconditioner: bool = True,
    constitutive_mode: str = "overlap",
    tangent_kernel: str = "rows",
    seepage_linear_tolerance: float = 1e-10,
    seepage_linear_max_iter: int = 500,
) -> dict:
    rank = int(PETSc.COMM_WORLD.getRank())
    out_dir = _ensure_dir(output_dir) if rank == 0 else output_dir
    data_dir = out_dir / "data"
    progress_callback = None
    if rank == 0:
        _ensure_dir(data_dir)
        progress_callback = _make_progress_logger(data_dir)

    if mesh_path is None:
        mesh_path = Path(__file__).resolve().parents[3] / "meshes" / "3d_hetero_seepage_ssr_comsol" / "comsol_mesh.msh"
    mesh_path = Path(mesh_path)
    elem_type = validate_supported_elem_type(3, elem_type)
    if elem_type != "P2":
        raise NotImplementedError(
            f"COMSOL seepage+SSR currently uses the exported P2 tetrahedral mesh family; requested {elem_type!r}."
        )

    material_rows = load_material_rows_for_path(mesh_path)
    if material_rows is None:
        material_rows = [
            [15.0, 30.0, 0.0, 10000.0, 0.33, 19.0, 19.0],
            [15.0, 38.0, 0.0, 50000.0, 0.30, 22.0, 22.0],
            [10.0, 35.0, 0.0, 50000.0, 0.30, 21.0, 21.0],
            [18.0, 32.0, 0.0, 20000.0, 0.33, 20.0, 20.0],
        ]
    material_rows = np.asarray(material_rows, dtype=np.float64)
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
        for row in material_rows
    ]

    mesh = load_mesh_p2_comsol(mesh_path, boundary_type=1)
    coord0 = np.asarray(mesh.coord, dtype=np.float64)
    elem0 = np.asarray(mesh.elem, dtype=np.int64)
    surf0 = np.asarray(mesh.surf, dtype=np.int64)
    q_mask0 = np.asarray(mesh.q_mask, dtype=bool)
    material_identifier = np.asarray(mesh.material, dtype=np.int64).ravel()
    triangle_labels = np.asarray(mesh.triangle_labels, dtype=np.int64).ravel()

    grho = 9.81
    n_q = int(quadrature_volume_3d(elem_type)[0].shape[1])
    seepage_mat = np.zeros(elem0.shape[1], dtype=np.int64)
    conduct0 = heter_conduct(seepage_mat, n_q, np.array([1.0], dtype=np.float64))
    q_w, pw_d = seepage_boundary_3d_hetero_comsol(coord0, surf0, triangle_labels, grho)

    seepage_solver = SolverFactory.create(
        "PETSC_MATLAB_DFGMRES_HYPRE",
        tolerance=seepage_linear_tolerance,
        max_iterations=seepage_linear_max_iter,
        deflation_basis_tolerance=1e-3,
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
    pw, grad_p, mater_sat, seep_history, _seep_assembly = seepage_problem_3d(
        coord0,
        elem0,
        q_w,
        pw_d,
        grho,
        conduct0,
        elem_type=elem_type,
        linear_system_solver=seepage_solver,
        it_max=50,
        tol=1e-10,
    )
    saturation = np.repeat(np.asarray(mater_sat, dtype=bool), n_q)

    partition_count = int(PETSc.COMM_WORLD.getSize()) if str(node_ordering).lower() == "block_metis" else None
    reordered = reorder_mesh_nodes(
        coord0,
        elem0,
        surf0,
        q_mask0,
        strategy=node_ordering,
        n_parts=partition_count,
    )
    coord = reordered.coord.astype(np.float64)
    elem = reordered.elem.astype(np.int64)
    q_mask = reordered.q_mask.astype(bool)

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

    const_builder = ConstitutiveOperator(
        B=None,
        c0=c0,
        phi=phi,
        psi=psi,
        Davis_type="B",
        shear=shear,
        bulk=bulk,
        lame=lame,
        WEIGHT=np.zeros(elem.shape[1] * n_q, dtype=np.float64),
        n_strain=6,
        n_int=elem.shape[1] * n_q,
        dim=3,
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
        include_overlap_B=(str(tangent_kernel).lower() == "legacy"),
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
        deflation_basis_tolerance=1e-3,
        verbose=False,
        q_mask=q_mask,
        coord=coord,
        preconditioner_options=preconditioner_options,
    )

    params = {
        "lambda_init": float(lambda_init),
        "d_lambda_init": float(d_lambda_init),
        "d_lambda_min": float(d_lambda_min),
        "d_lambda_diff_scaled_min": float(d_lambda_diff_scaled_min),
        "omega_max_stop": float(omega_max_stop),
        "step_max": int(step_max),
        "it_newt_max": int(it_newt_max),
        "it_damp_max": int(it_damp_max),
        "tol": float(tol),
        "r_min": float(r_min),
        "elem_type": elem_type,
        "davis_type": "B",
        "material_rows": material_rows.tolist(),
        "node_ordering": node_ordering,
        "mpi_distribute_by_nodes": bool(mpi_distribute_by_nodes),
        "pc_hypre_coarsen_type": pc_hypre_coarsen_type,
        "pc_hypre_interp_type": pc_hypre_interp_type,
        "pc_hypre_strong_threshold": pc_hypre_strong_threshold,
        "recycle_preconditioner": bool(recycle_preconditioner),
        "constitutive_mode": constitutive_mode,
        "tangent_kernel": str(tangent_kernel),
        "mesh_file": str(mesh_path),
        "seepage_linear_tolerance": float(seepage_linear_tolerance),
        "seepage_linear_max_iter": int(seepage_linear_max_iter),
    }

    t0 = perf_counter()
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
        continuation_predictor=str(continuation_predictor),
    )
    runtime = perf_counter() - t0
    mpi_comm = PETSc.COMM_WORLD.tompi4py()
    const_times = const_builder.get_total_time()
    const_times_max = {key: float(mpi_comm.allreduce(float(val), op=MPI.MAX)) for key, val in const_times.items()}
    linear_summary = {
        "init_linear_iterations": int(stats.get("init_linear_iterations", 0)),
        "init_linear_solve_time": float(stats.get("init_linear_solve_time", 0.0)),
        "init_linear_preconditioner_time": float(stats.get("init_linear_preconditioner_time", 0.0)),
        "init_linear_orthogonalization_time": float(stats.get("init_linear_orthogonalization_time", 0.0)),
        "attempt_linear_iterations_total": int(np.sum(np.asarray(stats.get("attempt_linear_iterations", []), dtype=np.int64))),
        "attempt_linear_solve_time_total": float(np.sum(np.asarray(stats.get("attempt_linear_solve_time", []), dtype=np.float64))),
        "attempt_linear_preconditioner_time_total": float(np.sum(np.asarray(stats.get("attempt_linear_preconditioner_time", []), dtype=np.float64))),
        "attempt_linear_orthogonalization_time_total": float(np.sum(np.asarray(stats.get("attempt_linear_orthogonalization_time", []), dtype=np.float64))),
    }

    step_u = np.asarray(stats.pop("step_U"), dtype=np.float64) if isinstance(stats.get("step_U", None), list) else np.empty((0, 3, 0), dtype=np.float64)
    run_payload = {
        "run_info": {
            "timestamp": np.datetime64("now").astype(str),
            "runtime_seconds": float(runtime),
            "mpi_size": int(PETSc.COMM_WORLD.getSize()),
            "mesh_nodes": int(coord.shape[1]),
            "mesh_elements": int(elem.shape[1]),
            "unknowns": int(q_mask.astype(bool).sum()),
            "solver_type": solver_type,
            "step_count": int(len(lambda_hist)),
        },
        "params": params,
        "mesh": {
            "mesh_file": str(mesh_path),
            "coord_shape": coord.shape,
            "elem_shape": elem.shape,
            "surf_shape": surf0.shape,
        },
        "timings": {
            "constitutive": const_times_max,
            "linear": linear_summary,
            "continuation_total_wall_time": float(stats.get("total_wall_time", runtime)),
            "seepage_runtime": float(np.sum(np.asarray(seep_history.get("linear_solve_time", []), dtype=np.float64))),
        },
        "seepage": {
            "criterion": [float(x) for x in seep_history.get("criterion", [])],
            "iterations": int(seep_history.get("iterations", 0)),
            "converged": bool(seep_history.get("converged", False)),
        },
    }

    if rank == 0:
        seepage_pw_reordered = np.asarray(pw, dtype=np.float64)[np.asarray(reordered.permutation, dtype=np.int64)]
        np.savez_compressed(
            data_dir / "petsc_run.npz",
            U=U,
            lambda_hist=lambda_hist,
            omega_hist=omega_hist,
            Umax_hist=Umax_hist,
            step_U=step_u,
            seepage_pw=pw,
            seepage_pw_reordered=seepage_pw_reordered,
            seepage_grad_p=grad_p,
            seepage_mater_sat=mater_sat,
            **{"stats_" + key: np.asarray(value) for key, value in stats.items() if key != "step_U"},
        )
        (data_dir / "run_info.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
        try:
            linear_system_solver.release_iteration_resources()
        except Exception:
            pass
        try:
            seepage_solver.release_iteration_resources()
        except Exception:
            pass
        try:
            const_builder.release_petsc_caches()
        except Exception:
            pass

    return {
        "output": str(out_dir),
        "npz": str(data_dir / "petsc_run.npz"),
        "json": str(data_dir / "run_info.json"),
        "runtime": runtime,
        "lambda_last": float(lambda_hist[-1]),
        "omega_last": float(omega_hist[-1]),
        "steps": int(len(lambda_hist)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the COMSOL 3D hetero seepage SSR capture.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--mesh_path", type=Path, default=None)
    parser.add_argument("--elem_type", type=str, default="P2", choices=["P1", "P2", "P4"])
    parser.add_argument("--node_ordering", type=str, default="block_metis", choices=["original", "xyz", "block_xyz", "morton", "rcm", "block_rcm", "block_metis"])
    parser.add_argument("--step_max", type=int, default=100)
    parser.add_argument("--lambda_init", type=float, default=1.0)
    parser.add_argument("--d_lambda_init", type=float, default=0.1)
    parser.add_argument("--d_lambda_min", type=float, default=1e-5)
    parser.add_argument("--d_lambda_diff_scaled_min", type=float, default=0.005)
    parser.add_argument("--omega_max_stop", type=float, default=3.407e8)
    parser.add_argument("--it_newt_max", type=int, default=50)
    parser.add_argument("--it_damp_max", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--r_min", type=float, default=1e-4)
    parser.add_argument("--linear_tolerance", type=float, default=1e-1)
    parser.add_argument("--linear_max_iter", type=int, default=100)
    parser.add_argument("--solver_type", type=str, default="PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE")
    parser.add_argument("--mpi_distribute_by_nodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pc_hypre_coarsen_type", type=str, default="HMIS")
    parser.add_argument("--pc_hypre_interp_type", type=str, default="ext+i")
    parser.add_argument("--pc_hypre_strong_threshold", type=float, default=None)
    parser.add_argument("--recycle_preconditioner", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--constitutive_mode", type=str, default="overlap", choices=["global", "overlap", "unique_gather", "unique_exchange"])
    parser.add_argument("--tangent_kernel", type=str, default="rows", choices=["legacy", "rows"])
    parser.add_argument("--seepage_linear_tolerance", type=float, default=1e-10)
    parser.add_argument("--seepage_linear_max_iter", type=int, default=500)
    args = parser.parse_args()

    result = run_capture(
        args.out_dir,
        mesh_path=args.mesh_path,
        elem_type=args.elem_type,
        node_ordering=args.node_ordering,
        step_max=args.step_max,
        lambda_init=args.lambda_init,
        d_lambda_init=args.d_lambda_init,
        d_lambda_min=args.d_lambda_min,
        d_lambda_diff_scaled_min=args.d_lambda_diff_scaled_min,
        omega_max_stop=args.omega_max_stop,
        it_newt_max=args.it_newt_max,
        it_damp_max=args.it_damp_max,
        tol=args.tol,
        r_min=args.r_min,
        linear_tolerance=args.linear_tolerance,
        linear_max_iter=args.linear_max_iter,
        solver_type=args.solver_type,
        mpi_distribute_by_nodes=args.mpi_distribute_by_nodes,
        pc_hypre_coarsen_type=args.pc_hypre_coarsen_type,
        pc_hypre_interp_type=args.pc_hypre_interp_type,
        pc_hypre_strong_threshold=args.pc_hypre_strong_threshold,
        recycle_preconditioner=args.recycle_preconditioner,
        constitutive_mode=args.constitutive_mode,
        tangent_kernel=args.tangent_kernel,
        seepage_linear_tolerance=args.seepage_linear_tolerance,
        seepage_linear_max_iter=args.seepage_linear_max_iter,
    )
    if PETSc.COMM_WORLD.getRank() == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
