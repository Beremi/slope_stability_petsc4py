#!/usr/bin/env python
"""Run the 2D homogeneous SSR benchmark with the PETSc solver stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from slope_stability.mesh import MaterialSpec, generate_homogeneous_slope_mesh_2d, heterogenous_materials, reorder_mesh_nodes
from slope_stability.fem import (
    assemble_owned_elastic_rows_for_comm,
    assemble_strain_operator,
    prepare_owned_tangent_pattern,
    quadrature_volume_2d,
    vector_volume,
)
from slope_stability.constitutive import ConstitutiveOperator
from slope_stability.continuation import SSR_indirect_continuation
from slope_stability.continuation import LL_indirect_continuation
from slope_stability.linear import SolverFactory
from slope_stability.nonlinear.newton import _destroy_petsc_mat, _prefers_full_system_operator, _setup_linear_system, _solve_linear_system
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


def _split_triangles_for_plot(elem: np.ndarray) -> np.ndarray:
    elem = np.asarray(elem, dtype=np.int64)
    if elem.shape[0] == 3:
        return elem.T
    if elem.shape[0] != 6:
        return elem[:3, :].T
    split = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2], [5, 3, 4]], dtype=np.int64)
    tri = np.empty((4 * elem.shape[1], 3), dtype=np.int64)
    for i in range(elem.shape[1]):
        tri[4 * i : 4 * (i + 1), :] = elem[:, i][split]
    return tri


def _deviatoric_strain_norm_2d(E: np.ndarray) -> np.ndarray:
    iota = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    dev = np.diag([1.0, 1.0, 0.5]) - np.outer(iota, iota) / 2.0
    dev_e = dev @ E
    return np.sqrt(np.maximum(0.0, np.sum(E * dev_e, axis=0)))


def _save_plots(
    coord: np.ndarray,
    elem: np.ndarray,
    U: np.ndarray,
    lambda_hist: np.ndarray,
    omega_hist: np.ndarray,
    B,
    out_dir: Path,
    step_u: np.ndarray,
    *,
    load_label: str = r"$\lambda$",
    title_prefix: str = "Indirect continuation",
):
    tri = _split_triangles_for_plot(elem)
    triangulation = mtri.Triangulation(coord[0, :], coord[1, :], triangles=tri)

    disp_mag = np.linalg.norm(U, axis=0)
    coord_def = coord + 0.15 * U
    triangulation_def = mtri.Triangulation(coord_def[0, :], coord_def[1, :], triangles=tri)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    tpc = ax.tripcolor(triangulation_def, disp_mag, shading="gouraud", cmap="viridis")
    ax.triplot(triangulation_def, color="k", linewidth=0.2, alpha=0.25)
    ax.set_aspect("equal")
    ax.set_title("Displacement magnitude (deformed mesh)")
    fig.colorbar(tpc, ax=ax, label=r"$\|U\|$")
    fig.tight_layout()
    fig.savefig(out_dir / "petsc_displacements_2D.png")
    plt.close(fig)

    E = (B @ U.reshape(-1, order="F")).reshape(3, -1, order="F")
    dev_norm = _deviatoric_strain_norm_2d(E)
    n_q = int(dev_norm.size // elem.shape[1])
    elem_strain = np.mean(dev_norm.reshape(n_q, elem.shape[1], order="F"), axis=0)
    centroids = np.mean(coord[:, elem[:3, :]], axis=1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    sc = ax.scatter(centroids[0, :], centroids[1, :], c=elem_strain, s=10, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_title("Deviatoric strain norm (element mean)")
    fig.colorbar(sc, ax=ax, label="deviatoric strain")
    fig.tight_layout()
    fig.savefig(out_dir / "petsc_deviatoric_strain_2D.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.plot(omega_hist, lambda_hist, marker="o", linewidth=1.25)
    ax.grid(True)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(load_label)
    ax.set_title(title_prefix)
    fig.tight_layout()
    fig.savefig(out_dir / "petsc_omega_lambda_2D.png")
    plt.close(fig)

    step_norm = np.max(np.linalg.norm(step_u, axis=1), axis=1)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.plot(step_norm, marker="o", linewidth=1.0)
    ax.grid(True)
    ax.set_xlabel("accepted step")
    ax.set_ylabel(r"max $\|U\|$")
    ax.set_title("Converged-step displacement growth")
    fig.tight_layout()
    fig.savefig(out_dir / "petsc_step_displacement_2D.png")
    plt.close(fig)


def run_capture(
    output_dir: Path,
    *,
    analysis: str = "ssr",
    elem_type: str = "P2",
    davis_type: str = "B",
    h: float = 1.0,
    x1: float = 15.0,
    x3: float = 15.0,
    y1: float = 10.0,
    y2: float = 10.0,
    beta_deg: float = 45.0,
    material_row: list[float] | np.ndarray | None = None,
    node_ordering: str = "block_metis",
    lambda_init: float = 0.9,
    d_lambda_init: float = 0.1,
    d_lambda_min: float = 1e-5,
    d_lambda_diff_scaled_min: float = 0.001,
    lambda_ell: float = 1.0,
    d_omega_ini_scale: float = 0.2,
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
) -> dict:
    rank = int(PETSc.COMM_WORLD.getRank())
    out_dir = _ensure_dir(output_dir) if rank == 0 else output_dir
    data_dir = out_dir / "data"
    progress_callback = None
    if rank == 0:
        _ensure_dir(data_dir)
        progress_callback = _make_progress_logger(data_dir)

    beta_rad = np.deg2rad(float(beta_deg))
    x2 = float(y2) / np.tan(beta_rad)
    mesh = generate_homogeneous_slope_mesh_2d(elem_type=elem_type, h=h, x1=x1, x2=x2, x3=x3, y1=y1, y2=y2)
    partition_count = int(PETSc.COMM_WORLD.getSize()) if str(node_ordering).lower() == "block_metis" else None
    reordered = reorder_mesh_nodes(mesh.coord, mesh.elem, mesh.surf, mesh.q_mask, strategy=node_ordering, n_parts=partition_count)

    coord = reordered.coord.astype(np.float64)
    elem = reordered.elem.astype(np.int64)
    surf = reordered.surf.astype(np.int64)
    q_mask = reordered.q_mask.astype(bool)
    material_identifier = np.asarray(mesh.material, dtype=np.int64)

    if material_row is None:
        material_row = [6.0, 45.0, 0.0, 40000.0, 0.3, 20.0, 20.0]
    row = np.asarray(material_row, dtype=np.float64)
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
    ]

    n_q = int(quadrature_volume_2d(elem_type)[0].shape[1])
    n_int = int(elem.shape[1] * n_q)
    c0, phi, psi, shear, bulk, lame, gamma = heterogenous_materials(
        material_identifier,
        np.ones(n_int, dtype=bool),
        n_q,
        materials,
    )

    use_lightweight_mpi_path = bool(mpi_distribute_by_nodes and str(constitutive_mode).lower() != "global")
    B = None
    weight = np.zeros(n_int, dtype=np.float64)

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
        rhs_parts = MPI.COMM_WORLD.allgather(np.asarray(elastic_rows.local_rhs, dtype=np.float64))
        f_V = np.concatenate(rhs_parts).reshape(coord.shape[0], coord.shape[1], order="F")
    else:
        assembly = assemble_strain_operator(coord, elem, elem_type, dim=2)
        from slope_stability.fem.assembly import build_elastic_stiffness_matrix

        K_elast, weight, B = build_elastic_stiffness_matrix(assembly, shear, lame, bulk)
        f_v_int = np.vstack((np.zeros(assembly.n_int, dtype=np.float64), -gamma.astype(np.float64)))
        f_V = vector_volume(assembly, f_v_int, weight)

    const_builder = ConstitutiveOperator(
        B=B,
        c0=c0,
        phi=phi,
        psi=psi,
        Davis_type=str(davis_type),
        shear=shear,
        bulk=bulk,
        lame=lame,
        WEIGHT=weight,
        n_strain=3,
        n_int=n_int,
        dim=2,
        q_mask=q_mask,
    )

    if mpi_distribute_by_nodes:
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
        )
        const_builder.set_owned_tangent_pattern(
            tangent_pattern,
            use_compiled=False,
            constitutive_mode=constitutive_mode,
            use_compiled_constitutive=False,
        )

    preconditioner_options = {
        "threads": 16,
        "print_level": 0,
        "use_as_preconditioner": True,
        "mpi_distribute_by_nodes": bool(mpi_distribute_by_nodes),
        "use_coordinates": True,
        "recycle_preconditioner": bool(recycle_preconditioner),
    }
    if pc_hypre_coarsen_type is not None:
        preconditioner_options["pc_hypre_boomeramg_coarsen_type"] = str(pc_hypre_coarsen_type)
    if pc_hypre_interp_type is not None:
        preconditioner_options["pc_hypre_boomeramg_interp_type"] = str(pc_hypre_interp_type)
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

    analysis_key = str(analysis).lower()
    if analysis_key not in {"ssr", "ll"}:
        raise ValueError(f"Unsupported analysis {analysis!r}.")

    params = {
        "analysis": analysis_key,
        "elem_type": str(elem_type),
        "davis_type": str(davis_type),
        "h": float(h),
        "x1": float(x1),
        "x2": float(x2),
        "x3": float(x3),
        "y1": float(y1),
        "y2": float(y2),
        "beta_deg": float(beta_deg),
        "material_row": row.tolist(),
        "node_ordering": str(node_ordering),
        "solver_type": str(solver_type),
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
        "mpi_distribute_by_nodes": bool(mpi_distribute_by_nodes),
        "recycle_preconditioner": bool(recycle_preconditioner),
        "constitutive_mode": str(constitutive_mode),
        "pc_hypre_coarsen_type": pc_hypre_coarsen_type,
        "pc_hypre_interp_type": pc_hypre_interp_type,
        "pc_hypre_strong_threshold": pc_hypre_strong_threshold,
    }

    t0 = perf_counter()
    init_linear = {
        "init_linear_iterations": 0,
        "init_linear_solve_time": 0.0,
        "init_linear_preconditioner_time": 0.0,
        "init_linear_orthogonalization_time": 0.0,
    }
    if analysis_key == "ssr":
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

        snap_init_1 = _collector_snapshot(linear_system_solver)
        init_delta = _collector_delta(snap_init_0, snap_init_1)
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
    runtime = perf_counter() - t0
    mpi_comm = PETSc.COMM_WORLD.tompi4py()

    const_times = const_builder.get_total_time()
    const_times_max = {key: float(mpi_comm.allreduce(float(val), op=MPI.MAX)) for key, val in const_times.items()}
    linear_summary = {
        "init_linear_iterations": int(stats.get("init_linear_iterations", init_linear["init_linear_iterations"])),
        "init_linear_solve_time": float(stats.get("init_linear_solve_time", init_linear["init_linear_solve_time"])),
        "init_linear_preconditioner_time": float(stats.get("init_linear_preconditioner_time", init_linear["init_linear_preconditioner_time"])),
        "init_linear_orthogonalization_time": float(stats.get("init_linear_orthogonalization_time", init_linear["init_linear_orthogonalization_time"])),
        "attempt_linear_iterations_total": int(np.sum(np.asarray(stats.get("attempt_linear_iterations", []), dtype=np.int64))),
        "attempt_linear_solve_time_total": float(np.sum(np.asarray(stats.get("attempt_linear_solve_time", []), dtype=np.float64))),
        "attempt_linear_preconditioner_time_total": float(np.sum(np.asarray(stats.get("attempt_linear_preconditioner_time", []), dtype=np.float64))),
        "attempt_linear_orthogonalization_time_total": float(np.sum(np.asarray(stats.get("attempt_linear_orthogonalization_time", []), dtype=np.float64))),
    }

    step_u = np.asarray(stats.pop("step_U"), dtype=np.float64) if isinstance(stats.get("step_U", None), list) else np.empty((0, 2, 0), dtype=np.float64)

    run_payload = {
        "run_info": {
            "timestamp": np.datetime64("now").astype(str),
            "runtime_seconds": float(runtime),
            "mpi_size": int(PETSc.COMM_WORLD.getSize()),
            "mesh_nodes": int(coord.shape[1]),
            "mesh_elements": int(elem.shape[1]),
            "unknowns": int(q_mask.sum()),
            "analysis": analysis_key,
            "solver_type": solver_type,
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
            **{"stats_" + key: np.asarray(value) for key, value in stats.items() if key != "step_U"},
        )
        (data_dir / "run_info.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

        try:
            linear_system_solver.release_iteration_resources()
        except Exception:
            pass
        try:
            const_builder.release_petsc_caches()
        except Exception:
            pass

        _ensure_dir(out_dir / "plots")
        plot_B = B
        if plot_B is None:
            plot_B = assemble_strain_operator(coord, elem, elem_type, dim=2).B
        _save_plots(
            coord,
            elem,
            U,
            lambda_hist,
            omega_hist,
            plot_B,
            out_dir / "plots",
            step_u=step_u,
            load_label=(r"$t$" if analysis_key == "ll" else r"$\lambda$"),
            title_prefix=("Indirect continuation (LL)" if analysis_key == "ll" else "Indirect continuation"),
        )

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
    parser = argparse.ArgumentParser(description="Run PETSc 2D homogeneous mechanics continuation case.")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--analysis", type=str, default="ssr", choices=["ssr", "ll"])
    parser.add_argument("--elem_type", type=str, default="P2", choices=["P1", "P2", "P4"])
    parser.add_argument("--davis_type", type=str, default="B")
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--x1", type=float, default=15.0)
    parser.add_argument("--x3", type=float, default=15.0)
    parser.add_argument("--y1", type=float, default=10.0)
    parser.add_argument("--y2", type=float, default=10.0)
    parser.add_argument("--beta_deg", type=float, default=45.0)
    parser.add_argument("--node_ordering", type=str, default="block_metis", choices=["original", "xyz", "block_xyz", "rcm", "block_rcm", "block_metis"])
    parser.add_argument("--step_max", type=int, default=100)
    parser.add_argument("--lambda_init", type=float, default=0.9)
    parser.add_argument("--d_lambda_init", type=float, default=0.1)
    parser.add_argument("--d_lambda_min", type=float, default=1e-5)
    parser.add_argument("--d_lambda_diff_scaled_min", type=float, default=0.001)
    parser.add_argument("--lambda_ell", type=float, default=1.0)
    parser.add_argument("--d_omega_ini_scale", type=float, default=0.2)
    parser.add_argument("--d_t_min", type=float, default=1e-3)
    parser.add_argument("--omega_max_stop", type=float, default=7e7)
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
    parser.add_argument("--constitutive_mode", type=str, default="overlap", choices=["global", "overlap", "unique_gather"])
    args = parser.parse_args()

    if args.out_dir is None:
        ts = np.datetime64("now").astype(str).replace(":", "-")
        args.out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "2D_homo_SSR_capture" / ts

    result = run_capture(
        args.out_dir,
        analysis=args.analysis,
        elem_type=args.elem_type,
        davis_type=args.davis_type,
        h=args.h,
        x1=args.x1,
        x3=args.x3,
        y1=args.y1,
        y2=args.y2,
        beta_deg=args.beta_deg,
        node_ordering=args.node_ordering,
        step_max=args.step_max,
        lambda_init=args.lambda_init,
        d_lambda_init=args.d_lambda_init,
        d_lambda_min=args.d_lambda_min,
        d_lambda_diff_scaled_min=args.d_lambda_diff_scaled_min,
        lambda_ell=args.lambda_ell,
        d_omega_ini_scale=args.d_omega_ini_scale,
        d_t_min=args.d_t_min,
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
    )
    if PETSc.COMM_WORLD.getRank() == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
