#!/usr/bin/env python
"""Run a PETSc-side 3D heterogeneous SSR indirect-continuation case and export artifacts."""

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from slope_stability.core.elements import normalize_elem_type, validate_supported_elem_type
from slope_stability.mesh import load_mesh_from_file, heterogenous_materials, MaterialSpec, reorder_mesh_nodes
from slope_stability.fem import (
    assemble_owned_elastic_rows_for_comm,
    assemble_strain_operator,
    prepare_owned_tangent_pattern,
    quadrature_volume_3d,
    vector_volume,
)
from slope_stability.linear import SolverFactory
from slope_stability.constitutive import ConstitutiveOperator
from slope_stability.continuation import LL_indirect_continuation, SSR_indirect_continuation
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


def _build_plotting_mesh(surf: np.ndarray) -> np.ndarray:
    """Return a simple triangulated boundary mesh for P1/P2/P4 triangle faces."""
    surf = np.asarray(surf, dtype=np.int64)
    if surf.ndim != 2:
        raise ValueError(f"Expected a 2D surface array, got shape {surf.shape}")
    if surf.shape[0] == 6:
        surf_faces = surf.T
    elif surf.shape[1] == 6:
        surf_faces = surf
    elif surf.shape[0] == 15:
        return surf[:3, :].T.astype(np.int64)
    elif surf.shape[1] == 15:
        return surf[:, :3].astype(np.int64)
    elif surf.shape[0] == 3:
        return surf.T.astype(np.int64)
    elif surf.shape[1] == 3:
        return surf.astype(np.int64)
    else:
        raise ValueError(f"Unsupported surface array shape {surf.shape}")
    if surf_faces.shape[1] != 6:
        # Already triangular.
        return surf_faces.astype(np.int64)
    split = np.array([[0, 3, 5], [3, 1, 4], [3, 4, 5], [5, 4, 2]], dtype=np.int64)
    triangles = []
    for face in surf_faces:
        triangles.append(face[split[0]])
        triangles.append(face[split[1]])
        triangles.append(face[split[2]])
        triangles.append(face[split[3]])
    return np.asarray(triangles, dtype=np.int64)


def _set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range) * 0.6
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d(x_mid - max_range, x_mid + max_range)
    ax.set_ylim3d(y_mid - max_range, y_mid + max_range)
    ax.set_zlim3d(z_mid - max_range, z_mid + max_range)


def _deviatoric_strain_norm(E: np.ndarray) -> np.ndarray:
    iota = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    dev = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - np.outer(iota, iota) / 3.0
    dev_e = dev @ E
    norm = np.sqrt(np.maximum(0.0, np.sum(E * dev_e, axis=0)))
    return norm


def _save_plots(
    coord,
    surf,
    U,
    lambda_hist,
    omega_hist,
    B,
    out_dir: Path,
    step_u: np.ndarray,
    elem: np.ndarray,
    n_q: int,
    *,
    load_label: str = r"$\lambda$",
    title_prefix: str = r"Indirect continuation: $\omega$ vs $\lambda$",
):
    tri = _build_plotting_mesh(surf.astype(np.int64))

    disp_mag = np.linalg.norm(U, axis=0)
    defo_scale = 0.05
    coord_def = coord + defo_scale * U

    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    triangles = [coord_def[:, t].T for t in tri]
    tri_vals = np.mean(disp_mag[tri], axis=1)
    face_colors = cmap((tri_vals - tri_vals.min()) / (tri_vals.max() - tri_vals.min() + 1e-15))
    mesh = Poly3DCollection(triangles, facecolors=face_colors, edgecolor="none", alpha=0.95)
    ax.add_collection3d(mesh)
    ax.set_title("Displacement magnitude (deformed mesh)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(tri_vals)
    fig.colorbar(mappable, ax=ax, pad=0.1, shrink=0.7, label=r"$\|U\|$")
    _set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "petsc_displacements_3D.png")
    plt.close(fig)

    # Deviatoric strain proxy as element-centered scatter, similar intent to MATLAB boundary coloring.
    E = B @ U.reshape(-1, order="F")
    E = E.reshape(6, -1, order="F")
    dev_norm = _deviatoric_strain_norm(E)
    n_e = elem.shape[1]
    elem_strain = dev_norm.reshape(n_q, n_e, order="F")
    # Aggregate by mean of IPs and reorder for consistency with MATLAB ordering used there.
    elem_strain = np.mean(elem_strain, axis=0)

    centroid = np.mean(coord[:, elem.astype(np.int64)], axis=1).T

    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(centroid[:, 0], centroid[:, 1], centroid[:, 2], c=elem_strain, cmap="viridis", s=2)
    ax.set_title("Deviatoric strain (element mean proxy)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(p, ax=ax, pad=0.1, shrink=0.7, label="deviatoric strain norm")
    _set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "petsc_deviatoric_strain_3D.png")
    plt.close(fig)

    # Continuation curve.
    fig = plt.figure(figsize=(8, 6), dpi=160)
    plt.plot(omega_hist, lambda_hist, marker="o", linewidth=1.5)
    plt.xlabel(r"$\xi$")
    plt.ylabel(load_label)
    plt.title(title_prefix)
    plt.grid(True)
    fig.savefig(out_dir / "petsc_omega_lambda.png")
    plt.close(fig)

    # Optional: save per-step displacement maxima as lightweight visual summary.
    step_norm = np.max(np.linalg.norm(step_u, axis=1), axis=1)
    fig = plt.figure(figsize=(8, 6), dpi=160)
    plt.plot(step_norm, marker="o", linewidth=1.0)
    plt.xlabel("accepted step")
    plt.ylabel(r"max $\|U\|$")
    plt.title("Converged-step displacement growth")
    plt.grid(True)
    fig.savefig(out_dir / "petsc_step_displacement.png")
    plt.close(fig)


def run_capture(
    output_dir: Path,
    *,
    analysis: str = "ssr",
    mesh_path: Path | None = None,
    mesh_boundary_type: int = 0,
    elem_type: str = "P2",
    davis_type: str = "B",
    material_rows: list[list[float]] | np.ndarray | None = None,
    node_ordering: str = "block_metis",
    lambda_init: float = 1.0,
    d_lambda_init: float = 0.1,
    d_lambda_min: float = 1e-3,
    d_lambda_diff_scaled_min: float = 0.001,
    lambda_ell: float = 1.0,
    d_omega_ini_scale: float = 0.2,
    d_t_min: float = 1e-3,
    omega_max_stop: float = 1.20e7,
    step_max: int = 100,
    it_newt_max: int = 200,
    it_damp_max: int = 10,
    tol: float = 1e-4,
    r_min: float = 1e-4,
    linear_tolerance: float = 1e-1,
    linear_max_iter: int = 100,
    solver_type: str = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
    factor_solver_type: str | None = None,
    mpi_distribute_by_nodes: bool = True,
    pc_gamg_process_eq_limit: int | None = None,
    pc_gamg_threshold: float | None = None,
    pc_hypre_coarsen_type: str | None = None,
    pc_hypre_interp_type: str | None = None,
    pc_hypre_strong_threshold: float | None = None,
    compiled_outer: bool = False,
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

    elem_type = validate_supported_elem_type(3, elem_type)
    if elem_type == "P4":
        raise NotImplementedError("3D P4 is wired in the config interface, but the mechanics benchmark path is not implemented yet.")

    if material_rows is None:
        material_rows = [
            [15.0, 30.0, 0.0, 10000.0, 0.33, 19.0, 19.0],
            [15.0, 38.0, 0.0, 50000.0, 0.30, 22.0, 22.0],
            [10.0, 35.0, 0.0, 50000.0, 0.30, 21.0, 21.0],
            [18.0, 32.0, 0.0, 20000.0, 0.33, 20.0, 20.0],
        ]
    mat_props = np.asarray(material_rows, dtype=np.float64)
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
        for row in mat_props
    ]

    if mesh_path is None:
        mesh_path = Path(__file__).resolve().parents[3] / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.h5"
    mesh_path = Path(mesh_path)
    mesh = load_mesh_from_file(mesh_path, boundary_type=int(mesh_boundary_type))
    if mesh.elem_type is not None and normalize_elem_type(mesh.elem_type) != elem_type:
        raise ValueError(
            f"Requested elem_type {elem_type!r}, but mesh {mesh_path.name} contains {mesh.elem_type!r} elements."
        )
    partition_count = int(PETSc.COMM_WORLD.getSize()) if str(node_ordering).lower() == "block_metis" else None
    reordered = reorder_mesh_nodes(
        mesh.coord,
        mesh.elem,
        mesh.surf,
        mesh.q_mask,
        strategy=node_ordering,
        n_parts=partition_count,
    )

    coord = reordered.coord.astype(np.float64)
    elem = reordered.elem.astype(np.int64)
    surf = reordered.surf.astype(np.int64)
    q_mask = reordered.q_mask.astype(bool)
    material_identifier = mesh.material.astype(np.int64).ravel()

    n_q = int(quadrature_volume_3d(elem_type)[0].shape[1])
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
        phi=phi,  # builder expects radians for phi
        psi=psi,  # builder expects radians for psi
        Davis_type=str(davis_type),
        shear=shear,
        bulk=bulk,
        lame=lame,
        WEIGHT=weight,
        n_strain=6,
        n_int=n_int,
        dim=3,
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
            use_compiled=True,
            constitutive_mode=constitutive_mode,
            use_compiled_constitutive=True,
        )

    preconditioner_options = {
        "threads": 16,
        "print_level": 0,
        "use_as_preconditioner": True,
        "factor_solver_type": factor_solver_type,
        "mpi_distribute_by_nodes": bool(mpi_distribute_by_nodes),
        "use_coordinates": True,
    }
    if compiled_outer:
        preconditioner_options["compiled_outer"] = True
    if recycle_preconditioner:
        preconditioner_options["recycle_preconditioner"] = True
    if "GAMG" in str(solver_type).upper():
        preconditioner_options.update(
            {
                "mg_levels_ksp_type": "richardson",
                "mg_levels_pc_type": "jacobi",
                "mg_levels_pc_jacobi_type": "rowl1",
                "mg_levels_pc_jacobi_rowl1_scale": 0.5,
                "mg_levels_pc_jacobi_fixdiagonal": True,
                "pc_gamg_agg_nsmooths": 1,
                "pc_gamg_esteig_ksp_max_it": 10,
                "pc_gamg_use_sa_esteig": False,
                "pc_gamg_coarse_eq_limit": 1000,
                "pc_mg_cycle_type": "v",
            }
        )
        if recycle_preconditioner:
            preconditioner_options["pc_gamg_reuse_interpolation"] = True
        if pc_gamg_process_eq_limit is not None:
            preconditioner_options["pc_gamg_process_eq_limit"] = int(pc_gamg_process_eq_limit)
        if pc_gamg_threshold is not None:
            preconditioner_options["pc_gamg_threshold"] = float(pc_gamg_threshold)
    if "HYPRE" in str(solver_type).upper():
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
        "factor_solver_type": factor_solver_type,
        "elem_type": str(elem_type),
        "davis_type": str(davis_type),
        "material_rows": np.asarray(mat_props, dtype=np.float64).tolist(),
        "mesh_boundary_type": int(mesh_boundary_type),
        "node_ordering": node_ordering,
        "mpi_distribute_by_nodes": bool(mpi_distribute_by_nodes),
        "pc_gamg_process_eq_limit": pc_gamg_process_eq_limit,
        "pc_gamg_threshold": pc_gamg_threshold,
        "pc_hypre_coarsen_type": pc_hypre_coarsen_type,
        "pc_hypre_interp_type": pc_hypre_interp_type,
        "pc_hypre_strong_threshold": pc_hypre_strong_threshold,
        "compiled_outer": bool(compiled_outer),
        "recycle_preconditioner": bool(recycle_preconditioner),
        "constitutive_mode": constitutive_mode,
    }

    t0 = perf_counter()
    init_linear = {
        "init_linear_iterations": 0,
        "init_linear_solve_time": 0.0,
        "init_linear_preconditioner_time": 0.0,
        "init_linear_orthogonalization_time": 0.0,
    }
    if analysis_key == "ssr":
        U3, lambda_hist3, omega_hist3, Umax_hist3, stats = SSR_indirect_continuation(
            lambda_init,
            d_lambda_init,
            d_lambda_min,
            d_lambda_diff_scaled_min,
            params["step_max"],
            params["omega_max_stop"],
            params["it_newt_max"],
            params["it_damp_max"],
            params["tol"],
            params["r_min"],
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

        U3, lambda_hist3, omega_hist3, Umax_hist3, stats = LL_indirect_continuation(
            omega_el * float(d_omega_ini_scale),
            d_t_min,
            params["step_max"],
            params["omega_max_stop"],
            params["it_newt_max"],
            params["it_damp_max"],
            params["tol"],
            params["r_min"],
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

    step_u = np.asarray(stats.pop("step_U"), dtype=np.float64) if isinstance(stats.get("step_U", None), list) else np.empty((0, 3, 0), dtype=np.float64)

    run_payload = {
        "run_info": {
            "timestamp": np.datetime64("now").astype(str),
            "python_version": "petsc driver",
            "runtime_seconds": float(runtime),
            "mpi_size": int(PETSc.COMM_WORLD.getSize()),
            "mesh_nodes": int(coord.shape[1]),
            "mesh_elements": int(elem.shape[1]),
            "unknowns": int(q_mask.astype(bool).sum()),
            "analysis": analysis_key,
            "solver_type": solver_type,
            "step_count": int(len(lambda_hist3)),
        },
        "params": params,
        "mesh": {
            "mesh_file": str(mesh_path),
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
            U=U3,
            lambda_hist=lambda_hist3,
            load_factor_hist=lambda_hist3,
            omega_hist=omega_hist3,
            Umax_hist=Umax_hist3,
            step_U=step_u,
            init_meta=np.array([1]),
            **{
                "stats_" + key: np.asarray(value) for key, value in stats.items() if key != "step_U"
            },
        )
        (data_dir / "run_info.json").write_text(json.dumps(run_payload, indent=2))

        # Release solver-side PETSc objects before rank-0 postprocessing.
        # This avoids carrying multigrid/HYPRE state through plotting/export.
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
            plot_B = assemble_strain_operator(coord, elem, elem_type, dim=3).B
        _save_plots(
            coord,
            surf,
            U3,
            lambda_hist3,
            omega_hist3,
            plot_B,
            out_dir / "plots",
            step_u=step_u,
            elem=elem,
            n_q=n_q,
            load_label=(r"$t$" if analysis_key == "ll" else r"$\lambda$"),
            title_prefix=(
                r"Indirect continuation: $\omega$ vs $t$"
                if analysis_key == "ll"
                else r"Indirect continuation: $\omega$ vs $\lambda$"
            ),
        )

    return {
        "output": str(out_dir),
        "npz": str(data_dir / "petsc_run.npz"),
        "json": str(data_dir / "run_info.json"),
        "runtime": runtime,
        "lambda_last": float(lambda_hist3[-1]),
        "omega_last": float(omega_hist3[-1]),
        "steps": int(len(lambda_hist3)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PETSc 3D mechanics continuation case.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory for artifacts.")
    parser.add_argument("--analysis", type=str, default="ssr", choices=["ssr", "ll"])
    parser.add_argument("--mesh_path", type=Path, default=None, help="Optional mesh override.")
    parser.add_argument("--mesh_boundary_type", type=int, default=0)
    parser.add_argument("--elem_type", type=str, default="P2", choices=["P1", "P2", "P4"])
    parser.add_argument("--davis_type", type=str, default="B")
    parser.add_argument(
        "--node_ordering",
        type=str,
        default="block_metis",
        choices=["original", "xyz", "block_xyz", "morton", "rcm", "block_rcm", "block_metis"],
    )
    parser.add_argument("--step_max", type=int, default=100)
    parser.add_argument("--lambda_init", type=float, default=1.0)
    parser.add_argument("--d_lambda_init", type=float, default=0.1)
    parser.add_argument("--d_lambda_min", type=float, default=1e-3)
    parser.add_argument("--d_lambda_diff_scaled_min", type=float, default=0.001)
    parser.add_argument("--lambda_ell", type=float, default=1.0)
    parser.add_argument("--d_omega_ini_scale", type=float, default=0.2)
    parser.add_argument("--d_t_min", type=float, default=1e-3)
    parser.add_argument("--omega_max_stop", type=float, default=1.2e7)
    parser.add_argument("--it_newt_max", type=int, default=200)
    parser.add_argument("--it_damp_max", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--r_min", type=float, default=1e-4)
    parser.add_argument("--linear_tolerance", type=float, default=1e-1)
    parser.add_argument("--linear_max_iter", type=int, default=100)
    parser.add_argument("--solver_type", type=str, default="PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE")
    parser.add_argument("--factor_solver_type", type=str, default=None)
    parser.add_argument("--mpi_distribute_by_nodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pc_gamg_process_eq_limit", type=int, default=None)
    parser.add_argument("--pc_gamg_threshold", type=float, default=None)
    parser.add_argument("--pc_hypre_coarsen_type", type=str, default="HMIS")
    parser.add_argument("--pc_hypre_interp_type", type=str, default="ext+i")
    parser.add_argument("--pc_hypre_strong_threshold", type=float, default=None)
    parser.add_argument("--compiled_outer", action="store_true", default=False)
    parser.add_argument("--recycle_preconditioner", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--constitutive_mode",
        type=str,
        default="overlap",
        choices=["global", "overlap", "unique_gather"],
    )
    args = parser.parse_args()

    if args.out_dir is None:
        ts = np.datetime64("now").astype(str).replace(":", "-")
        args.out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "3D_hetero_SSR_capture" / ts
    args.out_dir = Path(args.out_dir)
    result = run_capture(
        args.out_dir,
        analysis=args.analysis,
        mesh_path=args.mesh_path,
        mesh_boundary_type=args.mesh_boundary_type,
        elem_type=args.elem_type,
        davis_type=args.davis_type,
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
        factor_solver_type=args.factor_solver_type,
        mpi_distribute_by_nodes=args.mpi_distribute_by_nodes,
        pc_gamg_process_eq_limit=args.pc_gamg_process_eq_limit,
        pc_gamg_threshold=args.pc_gamg_threshold,
        pc_hypre_coarsen_type=args.pc_hypre_coarsen_type,
        pc_hypre_interp_type=args.pc_hypre_interp_type,
        pc_hypre_strong_threshold=args.pc_hypre_strong_threshold,
        compiled_outer=args.compiled_outer,
        recycle_preconditioner=args.recycle_preconditioner,
        constitutive_mode=args.constitutive_mode,
    )
    if PETSc.COMM_WORLD.getRank() == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
