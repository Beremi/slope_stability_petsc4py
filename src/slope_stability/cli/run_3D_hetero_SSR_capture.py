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
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from slope_stability.cli.assembly_policy import use_lightweight_mpi_elastic_path, use_owned_tangent_path
from slope_stability.core.elements import normalize_elem_type, validate_supported_elem_type
from slope_stability.cli.progress import make_progress_logger
from slope_stability.mesh import load_mesh_from_file, heterogenous_materials, MaterialSpec, reorder_mesh_nodes
from slope_stability.fem import (
    assemble_owned_elastic_rows_for_comm,
    assemble_strain_operator,
    prepare_owned_tangent_pattern,
    quadrature_volume_3d,
    vector_volume,
)
from slope_stability.fem.basis import local_basis_volume_3d
from slope_stability.fem.distributed_tangent import prepare_bddc_subdomain_pattern
from slope_stability.linear import SolverFactory
from slope_stability.linear.pmg import (
    build_3d_mixed_pmg_hierarchy,
    build_3d_mixed_pmg_hierarchy_with_intermediate_p2,
    build_3d_pmg_hierarchy,
    build_3d_same_mesh_pmg_hierarchy,
)
from slope_stability.constitutive import ConstitutiveOperator
from slope_stability.continuation import indirect as indirect_module
from slope_stability.continuation import LL_indirect_continuation, SSR_indirect_continuation
from slope_stability.nonlinear.newton import (
    _destroy_petsc_mat,
    _prefers_full_system_operator,
    _setup_linear_system,
    _solve_linear_system,
)
from slope_stability.problem_assets import load_material_rows_for_path
from slope_stability.utils import (
    extract_submatrix_free,
    flatten_field,
    full_field_from_free_values,
    local_csr_to_petsc_aij_matrix,
    matvec_to_numpy,
    owned_block_range,
    q_to_free_indices,
    to_petsc_aij_matrix,
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_progress_logger(progress_dir: Path):
    return make_progress_logger(progress_dir)


def _parse_petsc_opt_entries(entries: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in entries or []:
        text = str(raw).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Expected PETSc option in key=value form, got {raw!r}")
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Expected non-empty PETSc option key in {raw!r}")
        parsed[key] = value
    return parsed


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


def _nan_last(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr[-1]) if arr.size else np.nan


def _nan_sum(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nansum(arr))


def _nan_max(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.max(finite))


def _stats_value_to_npz(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        if not value:
            return np.asarray(value)
        if any(isinstance(v, (list, tuple, np.ndarray, dict)) or v is None for v in value):
            return np.asarray(value, dtype=object)
        return np.asarray(value)
    return np.asarray(value)


def _predictor_info_defaults() -> dict[str, object]:
    return {
        "basis_dim": np.nan,
        "basis_condition": np.nan,
        "predictor_wall_time": 0.0,
        "predictor_alpha": np.nan,
        "projected_delta_lambda": np.nan,
        "projected_correction_norm": np.nan,
        "energy_eval_count": np.nan,
        "energy_value": np.nan,
        "coarse_solve_wall_time": np.nan,
        "coarse_newton_iterations": np.nan,
        "coarse_residual_end": np.nan,
        "reduced_newton_iterations": np.nan,
        "reduced_gmres_iterations": np.nan,
        "reduced_projected_residual": np.nan,
        "reduced_omega_residual": np.nan,
        "fallback_used": False,
        "fallback_error": None,
    }


def _rescale_to_target_omega(U: np.ndarray, omega_target: float, f: np.ndarray, q_mask: np.ndarray) -> np.ndarray:
    U_arr = np.asarray(U, dtype=np.float64)
    free_idx = q_to_free_indices(np.asarray(q_mask, dtype=bool))
    denom = float(np.dot(flatten_field(np.asarray(f, dtype=np.float64))[free_idx], flatten_field(U_arr)[free_idx]))
    if abs(denom) <= 1.0e-30:
        return U_arr.copy()
    return U_arr * (float(omega_target) / denom)


def _surface_faces_by_width(surf: np.ndarray) -> np.ndarray:
    """Return boundary faces as (n_faces, nodes_per_face)."""
    surf = np.asarray(surf, dtype=np.int64)
    if surf.ndim != 2:
        raise ValueError(f"Expected a 2D surface array, got shape {surf.shape}")
    if surf.shape[0] == 6:
        return surf.T.astype(np.int64)
    if surf.shape[1] == 6:
        return surf.astype(np.int64)
    elif surf.shape[0] == 15:
        return surf[:3, :].T.astype(np.int64)
    if surf.shape[1] == 15:
        return surf[:, :3].astype(np.int64)
    if surf.shape[0] == 3:
        return surf.T.astype(np.int64)
    if surf.shape[1] == 3:
        return surf.astype(np.int64)
    raise ValueError(f"Unsupported surface array shape {surf.shape}")


def _build_plotting_mesh(surf: np.ndarray) -> np.ndarray:
    """Return a simple triangulated boundary mesh for P1/P2/P4 triangle faces."""
    surf_faces = _surface_faces_by_width(surf)
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


def _build_plotting_mesh_with_face_ids(surf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return triangulated boundary faces and the owning surface-face index for each triangle."""
    surf_faces = _surface_faces_by_width(surf)
    if surf_faces.shape[1] != 6:
        tri = surf_faces.astype(np.int64)
        face_ids = np.arange(tri.shape[0], dtype=np.int64)
        return tri, face_ids

    split = np.array([[0, 3, 5], [3, 1, 4], [3, 4, 5], [5, 4, 2]], dtype=np.int64)
    triangles: list[np.ndarray] = []
    face_ids: list[int] = []
    for face_id, face in enumerate(surf_faces):
        for local in split:
            triangles.append(face[local])
            face_ids.append(face_id)
    return np.asarray(triangles, dtype=np.int64), np.asarray(face_ids, dtype=np.int64)


def _surface_parent_elements(elem: np.ndarray, surf: np.ndarray) -> np.ndarray:
    """Map each boundary face to its owning tetrahedron via corner-node triples."""
    tet = np.asarray(elem, dtype=np.int64)
    faces = _surface_faces_by_width(surf)
    if tet.ndim != 2 or tet.shape[0] < 4:
        raise ValueError(f"Expected tetrahedral connectivity, got shape {tet.shape}")
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise ValueError(f"Expected triangular faces, got shape {faces.shape}")

    lookup: dict[tuple[int, int, int], int] = {}
    local_faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    corner_tet = tet[:4, :]
    for elem_id in range(corner_tet.shape[1]):
        nodes = corner_tet[:, elem_id]
        for local in local_faces:
            key = tuple(sorted(int(nodes[idx]) for idx in local))
            lookup[key] = elem_id

    parent = np.empty(faces.shape[0], dtype=np.int64)
    for face_id, face in enumerate(faces):
        key = tuple(sorted(int(v) for v in face[:3]))
        if key not in lookup:
            raise KeyError(f"Boundary face {key} was not found in any tetrahedron.")
        parent[face_id] = lookup[key]
    return parent


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


def _newton_guess_difference_volume_integrals(
    coord: np.ndarray,
    elem: np.ndarray,
    elem_type: str,
    U_guess: np.ndarray,
    U_solution: np.ndarray,
) -> dict[str, float]:
    """Return ∫||Δu|| dV and ∫||dev(Δε)|| dV for the Newton predictor error."""

    coord = np.asarray(coord, dtype=np.float64)
    elem = np.asarray(elem, dtype=np.int64)
    delta_u = np.asarray(U_solution, dtype=np.float64) - np.asarray(U_guess, dtype=np.float64)
    if coord.shape[0] != 3 or delta_u.shape[0] != 3:
        raise ValueError("3D predictor diagnostics require (3, n_nodes) coordinates and fields")

    xi, wf = quadrature_volume_3d(elem_type)
    hatp, dhat1, dhat2, dhat3 = local_basis_volume_3d(elem_type, xi)
    n_p = int(hatp.shape[0])
    n_q = int(xi.shape[1])
    if elem.shape[0] != n_p:
        raise ValueError(f"Element connectivity width {elem.shape[0]} does not match {elem_type} basis width {n_p}")

    # Keep the temporary dphi/evaluation buffers bounded on large P4 MPI runs.
    chunk_elems = max(32, min(1024, int(max(1, 2_000_000 // max(n_p * n_q, 1)))))
    disp_integral = 0.0
    dev_integral = 0.0

    for elem0 in range(0, int(elem.shape[1]), chunk_elems):
        elem1 = min(elem0 + chunk_elems, int(elem.shape[1]))
        elem_chunk = elem[:, elem0:elem1]
        n_elem_chunk = int(elem_chunk.shape[1])
        u_chunk = delta_u[:, elem_chunk]

        x = coord[0, elem_chunk]
        y = coord[1, elem_chunk]
        z = coord[2, elem_chunk]

        j11 = np.einsum("pe,pq->eq", x, dhat1, optimize=True)
        j12 = np.einsum("pe,pq->eq", y, dhat1, optimize=True)
        j13 = np.einsum("pe,pq->eq", z, dhat1, optimize=True)
        j21 = np.einsum("pe,pq->eq", x, dhat2, optimize=True)
        j22 = np.einsum("pe,pq->eq", y, dhat2, optimize=True)
        j23 = np.einsum("pe,pq->eq", z, dhat2, optimize=True)
        j31 = np.einsum("pe,pq->eq", x, dhat3, optimize=True)
        j32 = np.einsum("pe,pq->eq", y, dhat3, optimize=True)
        j33 = np.einsum("pe,pq->eq", z, dhat3, optimize=True)

        det_j = j11 * (j22 * j33 - j23 * j32) - j12 * (j21 * j33 - j23 * j31) + j13 * (j21 * j32 - j22 * j31)
        inv_det = 1.0 / det_j
        weight = np.abs(det_j) * wf[None, :]

        u_q = np.einsum("dpe,pq->deq", u_chunk, hatp, optimize=True)
        disp_integral += float(np.sum(np.linalg.norm(u_q, axis=0) * weight))

        c11 = (j22 * j33 - j23 * j32) * inv_det
        c12 = (j12 * j33 - j13 * j32) * inv_det
        c13 = (j12 * j23 - j13 * j22) * inv_det
        c21 = -(j21 * j33 - j23 * j31) * inv_det
        c22 = (j11 * j33 - j13 * j31) * inv_det
        c23 = -(j11 * j23 - j13 * j21) * inv_det
        c31 = (j21 * j32 - j22 * j31) * inv_det
        c32 = -(j11 * j32 - j12 * j31) * inv_det
        c33 = (j11 * j22 - j12 * j21) * inv_det

        dphi1 = (
            dhat1[:, None, :] * c11[None, :, :]
            - dhat2[:, None, :] * c12[None, :, :]
            + dhat3[:, None, :] * c13[None, :, :]
        )
        dphi2 = (
            dhat1[:, None, :] * c21[None, :, :]
            + dhat2[:, None, :] * c22[None, :, :]
            + dhat3[:, None, :] * c23[None, :, :]
        )
        dphi3 = (
            dhat1[:, None, :] * c31[None, :, :]
            + dhat2[:, None, :] * c32[None, :, :]
            + dhat3[:, None, :] * c33[None, :, :]
        )

        ux = u_chunk[0, :, :]
        uy = u_chunk[1, :, :]
        uz = u_chunk[2, :, :]
        e11 = np.einsum("pe,peq->eq", ux, dphi1, optimize=True)
        e22 = np.einsum("pe,peq->eq", uy, dphi2, optimize=True)
        e33 = np.einsum("pe,peq->eq", uz, dphi3, optimize=True)
        g12 = np.einsum("pe,peq->eq", ux, dphi2, optimize=True) + np.einsum("pe,peq->eq", uy, dphi1, optimize=True)
        g23 = np.einsum("pe,peq->eq", uy, dphi3, optimize=True) + np.einsum("pe,peq->eq", uz, dphi2, optimize=True)
        g13 = np.einsum("pe,peq->eq", ux, dphi3, optimize=True) + np.einsum("pe,peq->eq", uz, dphi1, optimize=True)
        strain = np.vstack(
            [
                e11.reshape(1, n_elem_chunk * n_q, order="C"),
                e22.reshape(1, n_elem_chunk * n_q, order="C"),
                e33.reshape(1, n_elem_chunk * n_q, order="C"),
                g12.reshape(1, n_elem_chunk * n_q, order="C"),
                g23.reshape(1, n_elem_chunk * n_q, order="C"),
                g13.reshape(1, n_elem_chunk * n_q, order="C"),
            ]
        )
        dev_norm = _deviatoric_strain_norm(strain).reshape(n_elem_chunk, n_q, order="C")
        dev_integral += float(np.sum(dev_norm * weight))

    return {
        "displacement_diff_volume_integral": float(disp_integral),
        "deviatoric_strain_diff_volume_integral": float(dev_integral),
    }


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

    # MATLAB-style surface coloring on the undeformed boundary mesh.
    E = B @ U.reshape(-1, order="F")
    E = E.reshape(6, -1, order="F")
    dev_norm = _deviatoric_strain_norm(E)
    n_e = elem.shape[1]
    elem_strain = dev_norm.reshape(n_q, n_e, order="F")
    elem_strain = np.mean(elem_strain, axis=0)
    tri_surface, tri_face_ids = _build_plotting_mesh_with_face_ids(surf.astype(np.int64))
    face_parent = _surface_parent_elements(elem.astype(np.int64), surf.astype(np.int64))
    tri_vals = elem_strain[face_parent[tri_face_ids]]
    strain_cmap = plt.get_cmap("jet")
    strain_norm = mcolors.Normalize(vmin=0.0, vmax=max(float(elem_strain.max()), 1e-12))

    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    triangles = [coord[:, tri_nodes].T for tri_nodes in tri_surface]
    face_colors = strain_cmap(strain_norm(tri_vals))
    mesh = Poly3DCollection(triangles, facecolors=face_colors, edgecolor="none", alpha=0.95)
    ax.add_collection3d(mesh)
    ax.set_title("Deviatoric strain (undeformed boundary surface)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    mappable = plt.cm.ScalarMappable(cmap=strain_cmap, norm=strain_norm)
    mappable.set_array(np.asarray([strain_norm.vmin, strain_norm.vmax], dtype=np.float64))
    fig.colorbar(mappable, ax=ax, pad=0.1, shrink=0.7, label="deviatoric strain norm")
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
    if step_u.ndim == 3 and step_u.size:
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
    continuation_predictor: str = "secant",
    omega_step_controller: str = "legacy",
    omega_no_increase_newton_threshold: int | None = None,
    omega_half_newton_threshold: int | None = None,
    omega_target_newton_iterations: float | None = None,
    omega_adapt_min_scale: float | None = None,
    omega_adapt_max_scale: float | None = None,
    omega_hard_newton_threshold: int | None = None,
    omega_hard_linear_threshold: int | None = None,
    omega_efficiency_floor: float | None = None,
    omega_efficiency_drop_ratio: float | None = None,
    omega_efficiency_window: int = 3,
    omega_hard_shrink_scale: float | None = None,
    step_length_cap_mode: str = "none",
    step_length_cap_factor: float = 1.0,
    init_newton_stopping_criterion: str | None = None,
    init_newton_stopping_tol: float | None = None,
    fine_newton_stopping_criterion: str | None = None,
    fine_newton_stopping_tol: float | None = None,
    fine_switch_mode: str = "none",
    fine_switch_distance_factor: float = 2.0,
    continuation_predictor_switch_ordinal: int | None = None,
    continuation_predictor_switch_to: str | None = None,
    continuation_predictor_window_size: int | None = None,
    continuation_predictor_use_projected_lambda: bool = True,
    continuation_predictor_refine_lambda_for_fixed_u: bool = False,
    continuation_predictor_reduced_max_iterations: int | None = None,
    continuation_predictor_reduced_use_partial_result: bool = False,
    continuation_predictor_reduced_tolerance: float | None = None,
    continuation_predictor_power_order: int | None = None,
    continuation_predictor_power_init: str | None = None,
    continuation_secant_correction_mode: str = "none",
    continuation_first_newton_warm_start_mode: str = "none",
    continuation_mode: str = "classic",
    streaming_micro_target_length: float = 0.15,
    streaming_micro_min_length: float = 0.05,
    streaming_micro_max_length: float = 0.30,
    streaming_move_relres_threshold: float = 5.0e-3,
    streaming_alpha_advance_threshold: float = 0.5,
    streaming_micro_max_corrections: int = 40,
    streaming_basis_max_vectors: int = 8,
    step_max: int = 100,
    it_newt_max: int = 200,
    it_damp_max: int = 10,
    tol: float = 1e-4,
    r_min: float = 1e-4,
    newton_stopping_criterion: str = "relative_residual",
    newton_stopping_tol: float | None = None,
    linear_tolerance: float = 1e-1,
    linear_max_iter: int = 100,
    solver_type: str = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
    factor_solver_type: str | None = None,
    pc_backend: str | None = "hypre",
    pmg_coarse_mesh_path: Path | None = None,
    pmg_fine_hierarchy_mode: str = "default",
    preconditioner_matrix_source: str = "tangent",
    preconditioner_matrix_policy: str = "current",
    preconditioner_rebuild_policy: str = "every_newton",
    preconditioner_rebuild_interval: int = 1,
    mpi_distribute_by_nodes: bool = True,
    pc_gamg_process_eq_limit: int | None = None,
    pc_gamg_threshold: float | None = None,
    pc_gamg_aggressive_coarsening: int | None = None,
    pc_gamg_aggressive_square_graph: bool | None = None,
    pc_gamg_aggressive_mis_k: int | None = None,
    pc_hypre_coarsen_type: str | None = "HMIS",
    pc_hypre_interp_type: str | None = "ext+i",
    pc_hypre_strong_threshold: float | None = None,
    pc_hypre_boomeramg_max_iter: int | None = 1,
    pc_hypre_P_max: int | None = None,
    pc_hypre_agg_nl: int | None = None,
    pc_hypre_nongalerkin_tol: float | None = None,
    pc_bddc_symmetric: bool | None = None,
    pc_bddc_dirichlet_ksp_type: str | None = None,
    pc_bddc_dirichlet_pc_type: str | None = None,
    pc_bddc_neumann_ksp_type: str | None = None,
    pc_bddc_neumann_pc_type: str | None = None,
    pc_bddc_coarse_ksp_type: str | None = None,
    pc_bddc_coarse_pc_type: str | None = None,
    pc_bddc_dirichlet_approximate: bool | None = None,
    pc_bddc_neumann_approximate: bool | None = None,
    pc_bddc_monolithic: bool | None = None,
    pc_bddc_coarse_redundant_pc_type: str | None = None,
    pc_bddc_switch_static: bool | None = None,
    pc_bddc_use_deluxe_scaling: bool | None = None,
    pc_bddc_use_vertices: bool | None = None,
    pc_bddc_use_edges: bool | None = None,
    pc_bddc_use_faces: bool | None = None,
    pc_bddc_use_change_of_basis: bool | None = None,
    pc_bddc_use_change_on_faces: bool | None = None,
    pc_bddc_check_level: int | None = None,
    petsc_opt: list[str] | None = None,
    compiled_outer: bool = False,
    recycle_preconditioner: bool = True,
    constitutive_mode: str = "overlap",
    tangent_kernel: str = "rows",
    max_deflation_basis_vectors: int = 48,
    store_step_u: bool = True,
) -> dict:
    rank = int(PETSc.COMM_WORLD.getRank())
    out_dir = _ensure_dir(output_dir) if rank == 0 else output_dir
    data_dir = out_dir / "data"
    progress_callback = None
    if rank == 0:
        _ensure_dir(data_dir)
        progress_callback = _make_progress_logger(data_dir)

    elem_type = validate_supported_elem_type(3, elem_type)

    if mesh_path is None:
        mesh_path = Path(__file__).resolve().parents[3] / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
    mesh_path = Path(mesh_path)

    solver_type_upper = str(solver_type).upper()
    effective_pc_backend = None
    if pc_backend is not None:
        effective_pc_backend = str(pc_backend).strip().lower()
    else:
        if "HYPRE" in solver_type_upper:
            effective_pc_backend = "hypre"
        elif "GAMG" in solver_type_upper:
            effective_pc_backend = "gamg"

    if effective_pc_backend in {"pmg", "pmg_shell"}:
        mixed_pmg_requested = pmg_coarse_mesh_path is not None
        if mixed_pmg_requested:
            if str(elem_type).upper() not in {"P2", "P4"}:
                raise ValueError(
                    f"{effective_pc_backend} backend with pmg_coarse_mesh_path currently supports only 3D P2 or P4 fine spaces."
                )
        elif effective_pc_backend == "pmg":
            if str(elem_type).upper() != "P4":
                raise ValueError(f"{effective_pc_backend} backend currently supports only 3D P4 fine spaces.")
        elif str(elem_type).upper() not in {"P2", "P4"}:
            raise ValueError(f"{effective_pc_backend} backend currently supports only 3D P2 or P4 fine spaces.")
        if str(preconditioner_matrix_source).strip().lower() != "tangent":
            raise ValueError(f"{effective_pc_backend} backend currently supports only preconditioner_matrix_source='tangent'.")
        if not bool(mpi_distribute_by_nodes):
            raise ValueError(f"{effective_pc_backend} backend requires mpi_distribute_by_nodes=true.")
        if "PETSC_MATLAB_DFGMRES" not in solver_type_upper and not solver_type_upper.startswith("KSPFGMRES"):
            raise ValueError(
                f"{effective_pc_backend} backend is currently supported only with PETSC_MATLAB_DFGMRES* or KSPFGMRES* solver types."
            )

    if material_rows is None:
        material_rows = load_material_rows_for_path(mesh_path)
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

    partition_count = int(PETSc.COMM_WORLD.getSize()) if str(node_ordering).lower() == "block_metis" else None
    pmg_hierarchy = None
    if effective_pc_backend in {"pmg", "pmg_shell"}:
        fine_hierarchy_mode = str(pmg_fine_hierarchy_mode).strip().lower()
        if pmg_coarse_mesh_path is None:
            if effective_pc_backend == "pmg_shell" and str(elem_type).upper() == "P2":
                pmg_hierarchy = build_3d_same_mesh_pmg_hierarchy(
                    mesh_path,
                    fine_elem_type=elem_type,
                    boundary_type=int(mesh_boundary_type),
                    node_ordering=node_ordering,
                    reorder_parts=partition_count,
                    material_rows=np.asarray(mat_props, dtype=np.float64).tolist(),
                    comm=PETSc.COMM_WORLD,
                )
            else:
                pmg_hierarchy = build_3d_pmg_hierarchy(
                    mesh_path,
                    boundary_type=int(mesh_boundary_type),
                    node_ordering=node_ordering,
                    reorder_parts=partition_count,
                    material_rows=np.asarray(mat_props, dtype=np.float64).tolist(),
                    comm=PETSc.COMM_WORLD,
                )
        else:
            if fine_hierarchy_mode == "p4_p2_intermediate":
                if str(elem_type).upper() != "P4":
                    raise ValueError("pmg_fine_hierarchy_mode='p4_p2_intermediate' requires elem_type='P4'.")
                pmg_hierarchy = build_3d_mixed_pmg_hierarchy_with_intermediate_p2(
                    mesh_path,
                    pmg_coarse_mesh_path,
                    boundary_type=int(mesh_boundary_type),
                    node_ordering=node_ordering,
                    reorder_parts=partition_count,
                    material_rows=np.asarray(mat_props, dtype=np.float64).tolist(),
                    comm=PETSc.COMM_WORLD,
                )
            else:
                pmg_hierarchy = build_3d_mixed_pmg_hierarchy(
                    mesh_path,
                    pmg_coarse_mesh_path,
                    fine_elem_type=elem_type,
                    boundary_type=int(mesh_boundary_type),
                    node_ordering=node_ordering,
                    reorder_parts=partition_count,
                    material_rows=np.asarray(mat_props, dtype=np.float64).tolist(),
                    comm=PETSc.COMM_WORLD,
                )
        coord = pmg_hierarchy.fine_level.coord.astype(np.float64)
        elem = pmg_hierarchy.fine_level.elem.astype(np.int64)
        surf = pmg_hierarchy.fine_level.surf.astype(np.int64)
        q_mask = pmg_hierarchy.fine_level.q_mask.astype(bool)
        material_identifier = pmg_hierarchy.fine_level.material_identifier.astype(np.int64).ravel()
    else:
        mesh = load_mesh_from_file(mesh_path, boundary_type=int(mesh_boundary_type), elem_type=elem_type)
        if mesh.elem_type is not None and normalize_elem_type(mesh.elem_type) != elem_type:
            raise ValueError(
                f"Requested elem_type {elem_type!r}, but mesh {mesh_path.name} contains {mesh.elem_type!r} elements."
            )
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

    use_owned_mpi_tangent_path = use_owned_tangent_path(
        solver_type=solver_type,
        mpi_distribute_by_nodes=mpi_distribute_by_nodes,
    )
    use_lightweight_mpi_path = use_lightweight_mpi_elastic_path(
        solver_type=solver_type,
        mpi_distribute_by_nodes=mpi_distribute_by_nodes,
        constitutive_mode=constitutive_mode,
    )

    B = None
    weight = np.zeros(n_int, dtype=np.float64)
    elastic_rows = None
    tangent_pattern = None
    bddc_pattern = None

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

    if use_owned_mpi_tangent_path:
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
            elastic_rows=elastic_rows if use_lightweight_mpi_path else None,
        )
        const_builder.set_owned_tangent_pattern(
            tangent_pattern,
            use_compiled=True,
            tangent_kernel=tangent_kernel,
            constitutive_mode=constitutive_mode,
            use_compiled_constitutive=True,
        )
        if effective_pc_backend == "bddc":
            bddc_pattern = prepare_bddc_subdomain_pattern(
                coord,
                elem,
                q_mask,
                material_identifier,
                materials,
                (row0 // coord.shape[0], row1 // coord.shape[0]),
                elem_type=elem_type,
                overlap_local_int_indices=tangent_pattern.local_int_indices,
            )
            const_builder.set_bddc_subdomain_pattern(bddc_pattern)

    preconditioner_options = {
        "threads": 16,
        "print_level": 0,
        "use_as_preconditioner": True,
        "factor_solver_type": factor_solver_type,
        "pc_backend": effective_pc_backend,
        "pmg_coarse_mesh_path": None if pmg_coarse_mesh_path is None else str(pmg_coarse_mesh_path),
        "pmg_fine_hierarchy_mode": str(pmg_fine_hierarchy_mode),
        "preconditioner_matrix_source": str(preconditioner_matrix_source),
        "preconditioner_matrix_policy": preconditioner_matrix_policy,
        "preconditioner_rebuild_policy": preconditioner_rebuild_policy,
        "preconditioner_rebuild_interval": int(preconditioner_rebuild_interval),
        "mpi_distribute_by_nodes": bool(mpi_distribute_by_nodes),
        "use_coordinates": True,
        # P4 continuation can accumulate many recycle vectors; cap them to keep memory bounded.
        "max_deflation_basis_vectors": int(max_deflation_basis_vectors),
    }
    if compiled_outer:
        preconditioner_options["compiled_outer"] = True
    if recycle_preconditioner:
        preconditioner_options["recycle_preconditioner"] = True
    mixed_parallel_shell = (
        pmg_hierarchy is not None
        and tuple(int(getattr(level, "order", -1)) for level in getattr(pmg_hierarchy, "levels", ())) == (1, 1, 2)
        and int(PETSc.COMM_WORLD.getSize()) > 1
    )
    if effective_pc_backend == "pmg":
        preconditioner_options.update(
            {
                "full_system_preconditioner": False,
                "pc_mg_galerkin": "both",
                "pc_mg_cycle_type": "v",
                "mg_levels_ksp_type": "richardson",
                "mg_levels_ksp_max_it": 3,
                "mg_levels_pc_type": "sor",
                "mg_coarse_ksp_type": "preonly",
                "mg_coarse_pc_type": "lu" if int(PETSc.COMM_WORLD.getSize()) == 1 else "redundant",
                "mg_coarse_redundant_ksp_type": "preonly",
                "mg_coarse_redundant_pc_type": "lu",
                "pmg_hierarchy": pmg_hierarchy,
            }
        )
    if effective_pc_backend == "pmg_shell":
        preconditioner_options.update(
            {
                "full_system_preconditioner": False,
                "mg_levels_ksp_type": "chebyshev" if mixed_parallel_shell else "richardson",
                "mg_levels_ksp_max_it": 3,
                "mg_levels_pc_type": "jacobi" if mixed_parallel_shell else "sor",
                "mg_coarse_ksp_type": "preonly",
                "mg_coarse_pc_type": "hypre",
                "mg_coarse_pc_hypre_type": "boomeramg",
                "pmg_hierarchy": pmg_hierarchy,
            }
        )
    if effective_pc_backend == "gamg":
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
        if pc_gamg_aggressive_coarsening is not None:
            preconditioner_options["pc_gamg_aggressive_coarsening"] = int(pc_gamg_aggressive_coarsening)
        if pc_gamg_aggressive_square_graph is not None:
            preconditioner_options["pc_gamg_aggressive_square_graph"] = bool(pc_gamg_aggressive_square_graph)
        if pc_gamg_aggressive_mis_k is not None:
            preconditioner_options["pc_gamg_aggressive_mis_k"] = int(pc_gamg_aggressive_mis_k)
    if effective_pc_backend == "hypre":
        if pc_hypre_coarsen_type is not None:
            preconditioner_options["pc_hypre_boomeramg_coarsen_type"] = str(pc_hypre_coarsen_type)
        if pc_hypre_interp_type is not None:
            preconditioner_options["pc_hypre_boomeramg_interp_type"] = str(pc_hypre_interp_type)
        if pc_hypre_strong_threshold is not None:
            preconditioner_options["pc_hypre_boomeramg_strong_threshold"] = float(pc_hypre_strong_threshold)
        if pc_hypre_boomeramg_max_iter is not None:
            preconditioner_options["pc_hypre_boomeramg_max_iter"] = int(pc_hypre_boomeramg_max_iter)
        if pc_hypre_P_max is not None:
            preconditioner_options["pc_hypre_boomeramg_P_max"] = int(pc_hypre_P_max)
        if pc_hypre_agg_nl is not None:
            preconditioner_options["pc_hypre_boomeramg_agg_nl"] = int(pc_hypre_agg_nl)
        if pc_hypre_nongalerkin_tol is not None:
            preconditioner_options["pc_hypre_boomeramg_nongalerkin_tol"] = float(pc_hypre_nongalerkin_tol)
    if pc_bddc_symmetric is not None:
        preconditioner_options["pc_bddc_symmetric"] = bool(pc_bddc_symmetric)
    if pc_bddc_dirichlet_ksp_type is not None:
        preconditioner_options["pc_bddc_dirichlet_ksp_type"] = str(pc_bddc_dirichlet_ksp_type)
    if pc_bddc_dirichlet_pc_type is not None:
        preconditioner_options["pc_bddc_dirichlet_pc_type"] = str(pc_bddc_dirichlet_pc_type)
    if pc_bddc_neumann_ksp_type is not None:
        preconditioner_options["pc_bddc_neumann_ksp_type"] = str(pc_bddc_neumann_ksp_type)
    if pc_bddc_neumann_pc_type is not None:
        preconditioner_options["pc_bddc_neumann_pc_type"] = str(pc_bddc_neumann_pc_type)
    if pc_bddc_coarse_ksp_type is not None:
        preconditioner_options["pc_bddc_coarse_ksp_type"] = str(pc_bddc_coarse_ksp_type)
    if pc_bddc_coarse_pc_type is not None:
        preconditioner_options["pc_bddc_coarse_pc_type"] = str(pc_bddc_coarse_pc_type)
    if pc_bddc_dirichlet_approximate is not None:
        preconditioner_options["pc_bddc_dirichlet_approximate"] = bool(pc_bddc_dirichlet_approximate)
    if pc_bddc_neumann_approximate is not None:
        preconditioner_options["pc_bddc_neumann_approximate"] = bool(pc_bddc_neumann_approximate)
    if pc_bddc_monolithic is not None:
        preconditioner_options["pc_bddc_monolithic"] = bool(pc_bddc_monolithic)
    if pc_bddc_coarse_redundant_pc_type is not None:
        preconditioner_options["pc_bddc_coarse_redundant_pc_type"] = str(pc_bddc_coarse_redundant_pc_type)
    if pc_bddc_switch_static is not None:
        preconditioner_options["pc_bddc_switch_static"] = bool(pc_bddc_switch_static)
    if pc_bddc_use_deluxe_scaling is not None:
        preconditioner_options["pc_bddc_use_deluxe_scaling"] = bool(pc_bddc_use_deluxe_scaling)
    if pc_bddc_use_vertices is not None:
        preconditioner_options["pc_bddc_use_vertices"] = bool(pc_bddc_use_vertices)
    if pc_bddc_use_edges is not None:
        preconditioner_options["pc_bddc_use_edges"] = bool(pc_bddc_use_edges)
    if pc_bddc_use_faces is not None:
        preconditioner_options["pc_bddc_use_faces"] = bool(pc_bddc_use_faces)
    if pc_bddc_use_change_of_basis is not None:
        preconditioner_options["pc_bddc_use_change_of_basis"] = bool(pc_bddc_use_change_of_basis)
    if pc_bddc_use_change_on_faces is not None:
        preconditioner_options["pc_bddc_use_change_on_faces"] = bool(pc_bddc_use_change_on_faces)
    if pc_bddc_check_level is not None:
        preconditioner_options["pc_bddc_check_level"] = int(pc_bddc_check_level)
    preconditioner_options.update(_parse_petsc_opt_entries(petsc_opt))

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
        "continuation_predictor": str(continuation_predictor),
        "omega_step_controller": str(omega_step_controller),
        "continuation_predictor_switch_ordinal": (
            None if continuation_predictor_switch_ordinal is None else int(continuation_predictor_switch_ordinal)
        ),
        "continuation_predictor_switch_to": (
            None if continuation_predictor_switch_to is None else str(continuation_predictor_switch_to)
        ),
        "continuation_predictor_window_size": (
            None if continuation_predictor_window_size is None else int(continuation_predictor_window_size)
        ),
        "continuation_predictor_use_projected_lambda": bool(continuation_predictor_use_projected_lambda),
        "continuation_predictor_refine_lambda_for_fixed_u": bool(continuation_predictor_refine_lambda_for_fixed_u),
        "continuation_predictor_reduced_max_iterations": (
            None if continuation_predictor_reduced_max_iterations is None else int(continuation_predictor_reduced_max_iterations)
        ),
        "continuation_predictor_reduced_use_partial_result": bool(continuation_predictor_reduced_use_partial_result),
        "continuation_predictor_reduced_tolerance": (
            None if continuation_predictor_reduced_tolerance is None else float(continuation_predictor_reduced_tolerance)
        ),
        "continuation_predictor_power_order": (
            None if continuation_predictor_power_order is None else int(continuation_predictor_power_order)
        ),
        "continuation_predictor_power_init": (
            None if continuation_predictor_power_init is None else str(continuation_predictor_power_init)
        ),
        "continuation_secant_correction_mode": str(continuation_secant_correction_mode),
        "continuation_first_newton_warm_start_mode": str(continuation_first_newton_warm_start_mode),
        "continuation_mode": str(continuation_mode),
        "streaming_micro_target_length": float(streaming_micro_target_length),
        "streaming_micro_min_length": float(streaming_micro_min_length),
        "streaming_micro_max_length": float(streaming_micro_max_length),
        "streaming_move_relres_threshold": float(streaming_move_relres_threshold),
        "streaming_alpha_advance_threshold": float(streaming_alpha_advance_threshold),
        "streaming_micro_max_corrections": int(streaming_micro_max_corrections),
        "streaming_basis_max_vectors": int(streaming_basis_max_vectors),
        "omega_no_increase_newton_threshold": (
            None if omega_no_increase_newton_threshold is None else int(omega_no_increase_newton_threshold)
        ),
        "omega_half_newton_threshold": (
            None if omega_half_newton_threshold is None else int(omega_half_newton_threshold)
        ),
        "omega_target_newton_iterations": (
            None if omega_target_newton_iterations is None else float(omega_target_newton_iterations)
        ),
        "omega_adapt_min_scale": None if omega_adapt_min_scale is None else float(omega_adapt_min_scale),
        "omega_adapt_max_scale": None if omega_adapt_max_scale is None else float(omega_adapt_max_scale),
        "omega_hard_newton_threshold": (
            None if omega_hard_newton_threshold is None else int(omega_hard_newton_threshold)
        ),
        "omega_hard_linear_threshold": (
            None if omega_hard_linear_threshold is None else int(omega_hard_linear_threshold)
        ),
        "omega_efficiency_floor": None if omega_efficiency_floor is None else float(omega_efficiency_floor),
        "omega_efficiency_drop_ratio": (
            None if omega_efficiency_drop_ratio is None else float(omega_efficiency_drop_ratio)
        ),
        "omega_efficiency_window": int(omega_efficiency_window),
        "omega_hard_shrink_scale": (
            None if omega_hard_shrink_scale is None else float(omega_hard_shrink_scale)
        ),
        "step_length_cap_mode": str(step_length_cap_mode),
        "step_length_cap_factor": float(step_length_cap_factor),
        "init_newton_stopping_criterion": (
            None if init_newton_stopping_criterion is None else str(init_newton_stopping_criterion)
        ),
        "init_newton_stopping_tol": (
            None if init_newton_stopping_tol is None else float(init_newton_stopping_tol)
        ),
        "fine_newton_stopping_criterion": (
            None if fine_newton_stopping_criterion is None else str(fine_newton_stopping_criterion)
        ),
        "fine_newton_stopping_tol": (
            None if fine_newton_stopping_tol is None else float(fine_newton_stopping_tol)
        ),
        "fine_switch_mode": str(fine_switch_mode),
        "fine_switch_distance_factor": float(fine_switch_distance_factor),
        "step_max": int(step_max),
        "it_newt_max": int(it_newt_max),
        "it_damp_max": int(it_damp_max),
        "tol": float(tol),
        "r_min": float(r_min),
        "newton_stopping_criterion": str(newton_stopping_criterion),
        "newton_stopping_tol": (
            None if newton_stopping_tol is None else float(newton_stopping_tol)
        ),
        "factor_solver_type": factor_solver_type,
        "pc_backend": effective_pc_backend,
        "pmg_fine_hierarchy_mode": str(pmg_fine_hierarchy_mode),
        "preconditioner_matrix_source": str(preconditioner_matrix_source),
        "preconditioner_matrix_policy": str(preconditioner_matrix_policy),
        "preconditioner_rebuild_policy": str(preconditioner_rebuild_policy),
        "preconditioner_rebuild_interval": int(preconditioner_rebuild_interval),
        "elem_type": str(elem_type),
        "davis_type": str(davis_type),
        "material_rows": np.asarray(mat_props, dtype=np.float64).tolist(),
        "mesh_boundary_type": int(mesh_boundary_type),
        "node_ordering": node_ordering,
        "mpi_distribute_by_nodes": bool(mpi_distribute_by_nodes),
        "pc_gamg_process_eq_limit": pc_gamg_process_eq_limit,
        "pc_gamg_threshold": pc_gamg_threshold,
        "pc_gamg_aggressive_coarsening": pc_gamg_aggressive_coarsening,
        "pc_gamg_aggressive_square_graph": pc_gamg_aggressive_square_graph,
        "pc_gamg_aggressive_mis_k": pc_gamg_aggressive_mis_k,
        "pc_hypre_coarsen_type": pc_hypre_coarsen_type,
        "pc_hypre_interp_type": pc_hypre_interp_type,
        "pc_hypre_strong_threshold": pc_hypre_strong_threshold,
        "pc_hypre_boomeramg_max_iter": pc_hypre_boomeramg_max_iter,
        "pc_hypre_P_max": pc_hypre_P_max,
        "pc_hypre_agg_nl": pc_hypre_agg_nl,
        "pc_hypre_nongalerkin_tol": pc_hypre_nongalerkin_tol,
        "pc_bddc_symmetric": pc_bddc_symmetric,
        "pc_bddc_dirichlet_ksp_type": pc_bddc_dirichlet_ksp_type,
        "pc_bddc_dirichlet_pc_type": pc_bddc_dirichlet_pc_type,
        "pc_bddc_neumann_ksp_type": pc_bddc_neumann_ksp_type,
        "pc_bddc_neumann_pc_type": pc_bddc_neumann_pc_type,
        "pc_bddc_coarse_ksp_type": pc_bddc_coarse_ksp_type,
        "pc_bddc_coarse_pc_type": pc_bddc_coarse_pc_type,
        "pc_bddc_dirichlet_approximate": pc_bddc_dirichlet_approximate,
        "pc_bddc_neumann_approximate": pc_bddc_neumann_approximate,
        "pc_bddc_monolithic": pc_bddc_monolithic,
        "pc_bddc_coarse_redundant_pc_type": pc_bddc_coarse_redundant_pc_type,
        "pc_bddc_switch_static": pc_bddc_switch_static,
        "pc_bddc_use_deluxe_scaling": pc_bddc_use_deluxe_scaling,
        "pc_bddc_use_vertices": pc_bddc_use_vertices,
        "pc_bddc_use_edges": pc_bddc_use_edges,
        "pc_bddc_use_faces": pc_bddc_use_faces,
        "pc_bddc_use_change_of_basis": pc_bddc_use_change_of_basis,
        "pc_bddc_use_change_on_faces": pc_bddc_use_change_on_faces,
        "pc_bddc_check_level": pc_bddc_check_level,
        "petsc_opt": list(petsc_opt or []),
        "compiled_outer": bool(compiled_outer),
        "recycle_preconditioner": bool(recycle_preconditioner),
        "constitutive_mode": constitutive_mode,
        "tangent_kernel": str(tangent_kernel),
    }

    t0 = perf_counter()
    init_linear = {
        "init_linear_iterations": 0,
        "init_linear_solve_time": 0.0,
        "init_linear_preconditioner_time": 0.0,
        "init_linear_orthogonalization_time": 0.0,
    }
    step_guess_diagnostics = None
    if analysis_key == "ssr":
        step_guess_diagnostics = lambda U_guess, U_solution: _newton_guess_difference_volume_integrals(  # noqa: E731
            coord,
            elem,
            elem_type,
            U_guess,
            U_solution,
        )
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
            linear_system_solver,
            progress_callback=progress_callback,
            store_step_u=bool(store_step_u),
            continuation_predictor=str(continuation_predictor),
            omega_step_controller=str(omega_step_controller),
            step_guess_diagnostics=step_guess_diagnostics,
            omega_no_increase_newton_threshold=omega_no_increase_newton_threshold,
            omega_half_newton_threshold=omega_half_newton_threshold,
            omega_target_newton_iterations=omega_target_newton_iterations,
            omega_adapt_min_scale=omega_adapt_min_scale,
            omega_adapt_max_scale=omega_adapt_max_scale,
            omega_hard_newton_threshold=omega_hard_newton_threshold,
            omega_hard_linear_threshold=omega_hard_linear_threshold,
            omega_efficiency_floor=omega_efficiency_floor,
            omega_efficiency_drop_ratio=omega_efficiency_drop_ratio,
            omega_efficiency_window=omega_efficiency_window,
            omega_hard_shrink_scale=omega_hard_shrink_scale,
            step_length_cap_mode=str(step_length_cap_mode),
            step_length_cap_factor=float(step_length_cap_factor),
            newton_stopping_criterion=str(newton_stopping_criterion),
            newton_stopping_tol=newton_stopping_tol,
            init_newton_stopping_criterion=init_newton_stopping_criterion,
            init_newton_stopping_tol=init_newton_stopping_tol,
            fine_newton_stopping_criterion=fine_newton_stopping_criterion,
            fine_newton_stopping_tol=fine_newton_stopping_tol,
            fine_switch_mode=str(fine_switch_mode),
            fine_switch_distance_factor=float(fine_switch_distance_factor),
            continuation_predictor_switch_ordinal=continuation_predictor_switch_ordinal,
            continuation_predictor_switch_to=continuation_predictor_switch_to,
            continuation_predictor_window_size=continuation_predictor_window_size,
            continuation_predictor_use_projected_lambda=continuation_predictor_use_projected_lambda,
            continuation_predictor_refine_lambda_for_fixed_u=continuation_predictor_refine_lambda_for_fixed_u,
            continuation_predictor_reduced_max_iterations=continuation_predictor_reduced_max_iterations,
            continuation_predictor_reduced_use_partial_result=continuation_predictor_reduced_use_partial_result,
            continuation_predictor_reduced_tolerance=continuation_predictor_reduced_tolerance,
            continuation_predictor_power_order=continuation_predictor_power_order,
            continuation_predictor_power_init=continuation_predictor_power_init,
            continuation_secant_correction_mode=str(continuation_secant_correction_mode),
            continuation_first_newton_warm_start_mode=str(continuation_first_newton_warm_start_mode),
            continuation_mode=str(continuation_mode),
            streaming_micro_target_length=float(streaming_micro_target_length),
            streaming_micro_min_length=float(streaming_micro_min_length),
            streaming_micro_max_length=float(streaming_micro_max_length),
            streaming_move_relres_threshold=float(streaming_move_relres_threshold),
            streaming_alpha_advance_threshold=float(streaming_alpha_advance_threshold),
            streaming_micro_max_corrections=int(streaming_micro_max_corrections),
            streaming_basis_max_vectors=int(streaming_basis_max_vectors),
        )
    else:
        free_idx = q_to_free_indices(q_mask)
        f_full = np.asarray(f_V, dtype=np.float64).reshape(-1, order="F")
        f_free = f_full[free_idx]
        init_linear_solver = linear_system_solver
        if effective_pc_backend in {"pmg", "pmg_shell"}:
            init_preconditioner_options = dict(preconditioner_options)
            init_preconditioner_options["pc_backend"] = "hypre"
            init_preconditioner_options.pop("pmg_hierarchy", None)
            for key in tuple(init_preconditioner_options.keys()):
                if key.startswith("mg_") or key.startswith("pc_mg_"):
                    init_preconditioner_options.pop(key, None)
            init_linear_solver = SolverFactory.create(
                solver_type,
                tolerance=linear_tolerance,
                max_iterations=linear_max_iter,
                deflation_basis_tolerance=1e-3,
                verbose=False,
                q_mask=q_mask,
                coord=coord,
                preconditioner_options=init_preconditioner_options,
            )

        snap_init_0 = _collector_snapshot(init_linear_solver)
        U_elast_free = None
        K_free = None
        const_builder.reduction(float(lambda_ell))
        try:
            if _prefers_full_system_operator(init_linear_solver, K_elast):
                _setup_linear_system(init_linear_solver, K_elast, A_full=K_elast, free_idx=free_idx)
                U_elast_free = _solve_linear_system(
                    init_linear_solver,
                    K_elast,
                    f_free,
                    b_full=f_full,
                    free_idx=free_idx,
                )
            else:
                K_free = extract_submatrix_free(K_elast, free_idx)
                _setup_linear_system(init_linear_solver, K_free, A_full=K_elast, free_idx=free_idx)
                U_elast_free = _solve_linear_system(
                    init_linear_solver,
                    K_free,
                    f_free,
                    b_full=f_full,
                    free_idx=free_idx,
                )
        finally:
            release = getattr(init_linear_solver, "release_iteration_resources", None)
            if callable(release):
                release()
            _destroy_petsc_mat(K_free)

        snap_init_1 = _collector_snapshot(init_linear_solver)
        init_delta = _collector_delta(snap_init_0, snap_init_1)
        init_linear = {
            "init_linear_iterations": int(init_delta["iterations"]),
            "init_linear_solve_time": float(init_delta["solve_time"]),
            "init_linear_preconditioner_time": float(init_delta["preconditioner_time"]),
            "init_linear_orthogonalization_time": float(init_delta["orthogonalization_time"]),
        }

        U_elast = full_field_from_free_values(np.asarray(U_elast_free, dtype=np.float64), free_idx, f_V.shape)
        if getattr(linear_system_solver, "supports_dynamic_deflation_basis", lambda: True)():
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
            linear_system_solver,
            progress_callback=progress_callback,
        )
    runtime = perf_counter() - t0
    mpi_comm = PETSc.COMM_WORLD.tompi4py()

    const_times = const_builder.get_total_time()
    const_times_max = {key: float(mpi_comm.allreduce(float(val), op=MPI.MAX)) for key, val in const_times.items()}
    tangent_pattern_stats_max = None
    tangent_pattern_stats_sum = None
    tangent_pattern_timings_max = None
    if tangent_pattern is not None:
        tangent_pattern_stats_max = {
            key: float(mpi_comm.allreduce(float(val), op=MPI.MAX))
            for key, val in tangent_pattern.stats.items()
        }
        tangent_pattern_stats_sum = {
            key: float(mpi_comm.allreduce(float(val), op=MPI.SUM))
            for key, val in tangent_pattern.stats.items()
        }
        tangent_pattern_timings_max = {
            key: float(mpi_comm.allreduce(float(val), op=MPI.MAX))
            for key, val in tangent_pattern.timings.items()
        }
    bddc_pattern_stats_max = None
    bddc_pattern_stats_sum = None
    bddc_pattern_timings_max = None
    if bddc_pattern is not None:
        bddc_pattern_stats_max = {
            key: float(mpi_comm.allreduce(float(val), op=MPI.MAX))
            for key, val in bddc_pattern.stats.items()
        }
        bddc_pattern_stats_sum = {
            key: float(mpi_comm.allreduce(float(val), op=MPI.SUM))
            for key, val in bddc_pattern.stats.items()
        }
        bddc_pattern_timings_max = {
            key: float(mpi_comm.allreduce(float(val), op=MPI.MAX))
            for key, val in bddc_pattern.timings.items()
        }
    linear_summary = {
        "init_linear_iterations": int(stats.get("init_linear_iterations", init_linear["init_linear_iterations"])),
        "init_linear_solve_time": float(stats.get("init_linear_solve_time", init_linear["init_linear_solve_time"])),
        "init_linear_preconditioner_time": float(stats.get("init_linear_preconditioner_time", init_linear["init_linear_preconditioner_time"])),
        "init_linear_orthogonalization_time": float(stats.get("init_linear_orthogonalization_time", init_linear["init_linear_orthogonalization_time"])),
        "attempt_linear_iterations_total": int(np.sum(np.asarray(stats.get("attempt_linear_iterations", []), dtype=np.int64))),
        "attempt_linear_solve_time_total": float(np.sum(np.asarray(stats.get("attempt_linear_solve_time", []), dtype=np.float64))),
        "attempt_linear_preconditioner_time_total": float(np.sum(np.asarray(stats.get("attempt_linear_preconditioner_time", []), dtype=np.float64))),
        "attempt_linear_orthogonalization_time_total": float(np.sum(np.asarray(stats.get("attempt_linear_orthogonalization_time", []), dtype=np.float64))),
        **linear_system_solver.get_preconditioner_diagnostics(),
    }
    attempt_u_diff = np.asarray(stats.get("attempt_initial_guess_displacement_diff_volume_integral", []), dtype=np.float64)
    attempt_dev_diff = np.asarray(stats.get("attempt_initial_guess_deviatoric_strain_diff_volume_integral", []), dtype=np.float64)
    step_u_diff = np.asarray(stats.get("step_initial_guess_displacement_diff_volume_integral", []), dtype=np.float64)
    step_dev_diff = np.asarray(stats.get("step_initial_guess_deviatoric_strain_diff_volume_integral", []), dtype=np.float64)
    step_secant_u_diff = np.asarray(stats.get("step_secant_reference_displacement_diff_volume_integral", []), dtype=np.float64)
    step_secant_dev_diff = np.asarray(stats.get("step_secant_reference_deviatoric_strain_diff_volume_integral", []), dtype=np.float64)
    step_predictor_basis_dim = np.asarray(stats.get("step_predictor_basis_dim", []), dtype=np.float64)
    step_predictor_wall = np.asarray(stats.get("step_predictor_wall_time", []), dtype=np.float64)
    step_predictor_alpha = np.asarray(stats.get("step_predictor_alpha", []), dtype=np.float64)
    step_predictor_beta = np.asarray(stats.get("step_predictor_beta", []), dtype=np.float64)
    step_predictor_gamma = np.asarray(stats.get("step_predictor_gamma", []), dtype=np.float64)
    step_predictor_energy_eval_count = np.asarray(stats.get("step_predictor_energy_eval_count", []), dtype=np.float64)
    step_predictor_energy_value = np.asarray(stats.get("step_predictor_energy_value", []), dtype=np.float64)
    step_predictor_coarse_solve_wall = np.asarray(stats.get("step_predictor_coarse_solve_wall_time", []), dtype=np.float64)
    step_predictor_coarse_newton = np.asarray(stats.get("step_predictor_coarse_newton_iterations", []), dtype=np.float64)
    step_predictor_coarse_residual = np.asarray(stats.get("step_predictor_coarse_residual_end", []), dtype=np.float64)
    step_predictor_basis_condition = np.asarray(stats.get("step_predictor_basis_condition", []), dtype=np.float64)
    step_predictor_reduced_newton = np.asarray(stats.get("step_predictor_reduced_newton_iterations", []), dtype=np.float64)
    step_predictor_reduced_gmres = np.asarray(stats.get("step_predictor_reduced_gmres_iterations", []), dtype=np.float64)
    step_predictor_reduced_projected = np.asarray(stats.get("step_predictor_reduced_projected_residual", []), dtype=np.float64)
    step_predictor_reduced_omega = np.asarray(stats.get("step_predictor_reduced_omega_residual", []), dtype=np.float64)
    step_predictor_state_coefficients = list(stats.get("step_predictor_state_coefficients", []))
    step_predictor_state_coefficients_ref = list(stats.get("step_predictor_state_coefficients_ref", []))
    step_predictor_state_coefficient_sum = np.asarray(stats.get("step_predictor_state_coefficient_sum", []), dtype=np.float64)
    step_secant_correction_active = np.asarray(stats.get("step_secant_correction_active", []), dtype=bool)
    step_secant_correction_basis_dim = np.asarray(stats.get("step_secant_correction_basis_dim", []), dtype=np.float64)
    step_secant_correction_trust_clipped = np.asarray(stats.get("step_secant_correction_trust_region_clipped", []), dtype=bool)
    step_secant_correction_predicted_decrease = np.asarray(
        stats.get("step_secant_correction_predicted_residual_decrease", []), dtype=np.float64
    )
    step_first_newton_warm_start_active = np.asarray(stats.get("step_first_newton_warm_start_active", []), dtype=bool)
    step_first_newton_warm_start_basis_dim = np.asarray(stats.get("step_first_newton_warm_start_basis_dim", []), dtype=np.float64)
    step_first_newton_linear_iterations = np.asarray(stats.get("step_first_newton_linear_iterations", []), dtype=np.float64)
    step_first_newton_linear_solve_time = np.asarray(stats.get("step_first_newton_linear_solve_time", []), dtype=np.float64)
    step_first_newton_linear_preconditioner_time = np.asarray(
        stats.get("step_first_newton_linear_preconditioner_time", []), dtype=np.float64
    )
    step_first_newton_linear_orthogonalization_time = np.asarray(
        stats.get("step_first_newton_linear_orthogonalization_time", []), dtype=np.float64
    )
    step_first_newton_correction_norm = np.asarray(stats.get("step_first_newton_correction_norm", []), dtype=np.float64)
    step_predictor_fallback_used = np.asarray(stats.get("step_predictor_fallback_used", []), dtype=bool)
    step_lambda_guess_abs_error = np.asarray(stats.get("step_lambda_initial_guess_abs_error", []), dtype=np.float64)
    predictor_summary = {
        "attempt_initial_guess_displacement_diff_volume_integral_total": _nan_sum(attempt_u_diff),
        "attempt_initial_guess_deviatoric_strain_diff_volume_integral_total": _nan_sum(attempt_dev_diff),
        "step_initial_guess_displacement_diff_volume_integral_total": _nan_sum(step_u_diff),
        "step_initial_guess_deviatoric_strain_diff_volume_integral_total": _nan_sum(step_dev_diff),
        "step_initial_guess_displacement_diff_volume_integral_last": _nan_last(step_u_diff),
        "step_initial_guess_deviatoric_strain_diff_volume_integral_last": _nan_last(step_dev_diff),
        "step_secant_reference_displacement_diff_volume_integral_total": _nan_sum(step_secant_u_diff),
        "step_secant_reference_deviatoric_strain_diff_volume_integral_total": _nan_sum(step_secant_dev_diff),
        "step_secant_reference_displacement_diff_volume_integral_last": _nan_last(step_secant_u_diff),
        "step_secant_reference_deviatoric_strain_diff_volume_integral_last": _nan_last(step_secant_dev_diff),
        "step_predictor_basis_dim_last": _nan_last(step_predictor_basis_dim),
        "step_predictor_basis_dim_max": _nan_max(step_predictor_basis_dim),
        "step_predictor_wall_time_total": _nan_sum(step_predictor_wall),
        "step_predictor_wall_time_last": _nan_last(step_predictor_wall),
        "step_predictor_alpha_last": _nan_last(step_predictor_alpha),
        "step_predictor_beta_last": _nan_last(step_predictor_beta),
        "step_predictor_gamma_last": _nan_last(step_predictor_gamma),
        "step_predictor_energy_eval_count_total": _nan_sum(step_predictor_energy_eval_count),
        "step_predictor_energy_value_last": _nan_last(step_predictor_energy_value),
        "step_predictor_coarse_solve_wall_time_total": _nan_sum(step_predictor_coarse_solve_wall),
        "step_predictor_coarse_newton_iterations_total": _nan_sum(step_predictor_coarse_newton),
        "step_predictor_coarse_residual_last": _nan_last(step_predictor_coarse_residual),
        "step_predictor_basis_condition_last": _nan_last(step_predictor_basis_condition),
        "step_predictor_basis_condition_max": _nan_max(step_predictor_basis_condition),
        "step_predictor_reduced_newton_iterations_total": _nan_sum(step_predictor_reduced_newton),
        "step_predictor_reduced_gmres_iterations_total": _nan_sum(step_predictor_reduced_gmres),
        "step_predictor_reduced_projected_residual_last": _nan_last(step_predictor_reduced_projected),
        "step_predictor_reduced_omega_residual_last": _nan_last(step_predictor_reduced_omega),
        "step_predictor_state_coefficients_last": None
        if not step_predictor_state_coefficients
        else step_predictor_state_coefficients[-1],
        "step_predictor_state_coefficients_ref_last": None
        if not step_predictor_state_coefficients_ref
        else step_predictor_state_coefficients_ref[-1],
        "step_predictor_state_coefficient_sum_last": _nan_last(step_predictor_state_coefficient_sum),
        "step_secant_correction_active_count": int(np.count_nonzero(step_secant_correction_active)),
        "step_secant_correction_basis_dim_last": _nan_last(step_secant_correction_basis_dim),
        "step_secant_correction_trust_region_clipped_count": int(np.count_nonzero(step_secant_correction_trust_clipped)),
        "step_secant_correction_predicted_residual_decrease_last": _nan_last(step_secant_correction_predicted_decrease),
        "step_first_newton_warm_start_active_count": int(np.count_nonzero(step_first_newton_warm_start_active)),
        "step_first_newton_warm_start_basis_dim_last": _nan_last(step_first_newton_warm_start_basis_dim),
        "step_first_newton_linear_iterations_total": _nan_sum(step_first_newton_linear_iterations),
        "step_first_newton_linear_solve_time_total": _nan_sum(step_first_newton_linear_solve_time),
        "step_first_newton_linear_preconditioner_time_total": _nan_sum(step_first_newton_linear_preconditioner_time),
        "step_first_newton_linear_orthogonalization_time_total": _nan_sum(step_first_newton_linear_orthogonalization_time),
        "step_first_newton_correction_norm_last": _nan_last(step_first_newton_correction_norm),
        "step_predictor_fallback_count": int(np.count_nonzero(step_predictor_fallback_used)),
        "step_lambda_initial_guess_abs_error_total": _nan_sum(step_lambda_guess_abs_error),
        "step_lambda_initial_guess_abs_error_last": _nan_last(step_lambda_guess_abs_error),
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
        "predictor_diagnostics": predictor_summary,
        "owned_tangent_pattern": {
            "stats_max": tangent_pattern_stats_max,
            "stats_sum": tangent_pattern_stats_sum,
            "timings_max": tangent_pattern_timings_max,
        } if tangent_pattern is not None else None,
        "bddc_subdomain_pattern": {
            "stats_max": bddc_pattern_stats_max,
            "stats_sum": bddc_pattern_stats_sum,
            "timings_max": bddc_pattern_timings_max,
        } if bddc_pattern is not None else None,
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
                "stats_" + key: _stats_value_to_npz(value) for key, value in stats.items() if key != "step_U"
            },
        )
        (data_dir / "run_info.json").write_text(json.dumps(run_payload, indent=2))

        # Release solver-side PETSc objects before rank-0 postprocessing.
        # This avoids carrying multigrid/HYPRE state through plotting/export.
        try:
            close_solver = getattr(linear_system_solver, "close", None)
            if callable(close_solver):
                close_solver()
            else:
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
    parser.add_argument(
        "--continuation_predictor",
        type=str,
        default="secant",
        choices=[
            "secant",
            "reduced_newton_all_prev",
            "reduced_newton_affine_all_prev",
            "reduced_newton_window",
            "reduced_newton_increment_power",
        ],
    )
    parser.add_argument("--omega_step_controller", type=str, default="legacy", choices=["legacy", "adaptive"])
    parser.add_argument("--continuation_predictor_switch_ordinal", type=int, default=None)
    parser.add_argument(
        "--continuation_predictor_switch_to",
        type=str,
        default=None,
        choices=[
            "secant",
            "reduced_newton_all_prev",
            "reduced_newton_affine_all_prev",
            "reduced_newton_window",
            "reduced_newton_increment_power",
        ],
    )
    parser.add_argument("--continuation_predictor_window_size", type=int, default=None)
    parser.add_argument(
        "--continuation_predictor_use_projected_lambda",
        dest="continuation_predictor_use_projected_lambda",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--continuation_predictor_use_current_lambda",
        dest="continuation_predictor_use_projected_lambda",
        action="store_false",
    )
    parser.add_argument(
        "--continuation_predictor_refine_lambda_for_fixed_u",
        dest="continuation_predictor_refine_lambda_for_fixed_u",
        action="store_true",
        default=False,
    )
    parser.add_argument("--continuation_predictor_reduced_max_iterations", type=int, default=None)
    parser.add_argument(
        "--continuation_predictor_reduced_use_partial_result",
        dest="continuation_predictor_reduced_use_partial_result",
        action="store_true",
        default=False,
    )
    parser.add_argument("--continuation_predictor_reduced_tolerance", type=float, default=None)
    parser.add_argument("--continuation_predictor_power_order", type=int, default=None)
    parser.add_argument(
        "--continuation_predictor_power_init",
        type=str,
        default=None,
        choices=["secant", "equal_split"],
    )
    parser.add_argument(
        "--continuation_secant_correction_mode",
        type=str,
        default="none",
        choices=["none", "orthogonal_increment_ls"],
    )
    parser.add_argument(
        "--continuation_first_newton_warm_start_mode",
        type=str,
        default="none",
        choices=["none", "history_deflation"],
    )
    parser.add_argument(
        "--continuation_mode",
        type=str,
        default="classic",
        choices=["classic", "streaming_microstep"],
    )
    parser.add_argument("--streaming_micro_target_length", type=float, default=0.15)
    parser.add_argument("--streaming_micro_min_length", type=float, default=0.05)
    parser.add_argument("--streaming_micro_max_length", type=float, default=0.30)
    parser.add_argument("--streaming_move_relres_threshold", type=float, default=5.0e-3)
    parser.add_argument("--streaming_alpha_advance_threshold", type=float, default=0.5)
    parser.add_argument("--streaming_micro_max_corrections", type=int, default=40)
    parser.add_argument("--streaming_basis_max_vectors", type=int, default=8)
    parser.add_argument("--omega_no_increase_newton_threshold", type=int, default=None)
    parser.add_argument("--omega_half_newton_threshold", type=int, default=None)
    parser.add_argument("--omega_target_newton_iterations", type=float, default=None)
    parser.add_argument("--omega_adapt_min_scale", type=float, default=None)
    parser.add_argument("--omega_adapt_max_scale", type=float, default=None)
    parser.add_argument("--omega_hard_newton_threshold", type=int, default=None)
    parser.add_argument("--omega_hard_linear_threshold", type=int, default=None)
    parser.add_argument("--omega_efficiency_floor", type=float, default=None)
    parser.add_argument("--omega_efficiency_drop_ratio", type=float, default=None)
    parser.add_argument("--omega_efficiency_window", type=int, default=3)
    parser.add_argument("--omega_hard_shrink_scale", type=float, default=None)
    parser.add_argument("--step_length_cap_mode", type=str, default="none", choices=["none", "initial_segment", "history_box"])
    parser.add_argument("--step_length_cap_factor", type=float, default=1.0)
    parser.add_argument("--it_newt_max", type=int, default=200)
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
    parser.add_argument(
        "--init_newton_stopping_criterion",
        type=str,
        default=None,
        choices=["relative_residual", "relative_correction", "absolute_delta_lambda"],
    )
    parser.add_argument("--init_newton_stopping_tol", type=float, default=None)
    parser.add_argument(
        "--fine_newton_stopping_criterion",
        type=str,
        default=None,
        choices=["relative_residual", "relative_correction", "absolute_delta_lambda"],
    )
    parser.add_argument("--fine_newton_stopping_tol", type=float, default=None)
    parser.add_argument("--fine_switch_mode", type=str, default="none", choices=["none", "history_box_cumulative_distance"])
    parser.add_argument("--fine_switch_distance_factor", type=float, default=2.0)
    parser.add_argument("--linear_tolerance", type=float, default=1e-1)
    parser.add_argument("--linear_max_iter", type=int, default=100)
    parser.add_argument("--solver_type", type=str, default="PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE")
    parser.add_argument("--factor_solver_type", type=str, default=None)
    parser.add_argument("--pc_backend", type=str, default="hypre", choices=["hypre", "gamg", "bddc", "pmg", "pmg_shell"])
    parser.add_argument("--pmg_coarse_mesh_path", type=Path, default=None)
    parser.add_argument(
        "--pmg_fine_hierarchy_mode",
        type=str,
        default="default",
        choices=["default", "p4_p2_intermediate"],
    )
    parser.add_argument(
        "--preconditioner_matrix_source",
        type=str,
        default="tangent",
        choices=["tangent", "regularized", "elastic"],
    )
    parser.add_argument("--preconditioner_matrix_policy", type=str, default="current", choices=["current", "lagged"])
    parser.add_argument(
        "--preconditioner_rebuild_policy",
        type=str,
        default="every_newton",
        choices=["every_newton", "every_n_newton", "accepted_step", "accepted_or_rejected_step"],
    )
    parser.add_argument("--preconditioner_rebuild_interval", type=int, default=1)
    parser.add_argument("--mpi_distribute_by_nodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pc_gamg_process_eq_limit", type=int, default=None)
    parser.add_argument("--pc_gamg_threshold", type=float, default=None)
    parser.add_argument("--pc_gamg_aggressive_coarsening", type=int, default=None)
    parser.add_argument("--pc_gamg_aggressive_square_graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pc_gamg_aggressive_mis_k", type=int, default=None)
    parser.add_argument("--pc_hypre_coarsen_type", type=str, default="HMIS")
    parser.add_argument("--pc_hypre_interp_type", type=str, default="ext+i")
    parser.add_argument("--pc_hypre_strong_threshold", type=float, default=None)
    parser.add_argument("--pc_hypre_boomeramg_max_iter", type=int, default=1)
    parser.add_argument("--pc_hypre_P_max", type=int, default=None)
    parser.add_argument("--pc_hypre_agg_nl", type=int, default=None)
    parser.add_argument("--pc_hypre_nongalerkin_tol", type=float, default=None)
    parser.add_argument("--pc_bddc_symmetric", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pc_bddc_dirichlet_ksp_type", type=str, default=None)
    parser.add_argument("--pc_bddc_dirichlet_pc_type", type=str, default=None)
    parser.add_argument("--pc_bddc_neumann_ksp_type", type=str, default=None)
    parser.add_argument("--pc_bddc_neumann_pc_type", type=str, default=None)
    parser.add_argument("--pc_bddc_coarse_ksp_type", type=str, default=None)
    parser.add_argument("--pc_bddc_coarse_pc_type", type=str, default=None)
    parser.add_argument("--pc_bddc_dirichlet_approximate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pc_bddc_neumann_approximate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pc_bddc_monolithic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pc_bddc_coarse_redundant_pc_type", type=str, default=None)
    parser.add_argument("--pc_bddc_switch_static", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pc_bddc_use_deluxe_scaling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pc_bddc_use_vertices", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pc_bddc_use_edges", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pc_bddc_use_faces", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pc_bddc_use_change_of_basis", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pc_bddc_use_change_on_faces", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pc_bddc_check_level", type=int, default=None)
    parser.add_argument("--petsc-opt", action="append", default=[], dest="petsc_opt")
    parser.add_argument("--compiled_outer", action="store_true", default=False)
    parser.add_argument("--recycle_preconditioner", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_deflation_basis_vectors", type=int, default=48)
    parser.add_argument("--store_step_u", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--constitutive_mode",
        type=str,
        default="overlap",
        choices=["global", "overlap", "unique_gather", "unique_exchange"],
    )
    parser.add_argument("--tangent_kernel", type=str, default="rows", choices=["legacy", "rows"])
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
        continuation_predictor=args.continuation_predictor,
        omega_step_controller=args.omega_step_controller,
        continuation_predictor_switch_ordinal=args.continuation_predictor_switch_ordinal,
        continuation_predictor_switch_to=args.continuation_predictor_switch_to,
        continuation_predictor_window_size=args.continuation_predictor_window_size,
        continuation_predictor_use_projected_lambda=args.continuation_predictor_use_projected_lambda,
        continuation_predictor_refine_lambda_for_fixed_u=args.continuation_predictor_refine_lambda_for_fixed_u,
        continuation_predictor_reduced_max_iterations=args.continuation_predictor_reduced_max_iterations,
        continuation_predictor_reduced_use_partial_result=args.continuation_predictor_reduced_use_partial_result,
        continuation_predictor_reduced_tolerance=args.continuation_predictor_reduced_tolerance,
        continuation_predictor_power_order=args.continuation_predictor_power_order,
        continuation_predictor_power_init=args.continuation_predictor_power_init,
        continuation_secant_correction_mode=args.continuation_secant_correction_mode,
        continuation_first_newton_warm_start_mode=args.continuation_first_newton_warm_start_mode,
        continuation_mode=args.continuation_mode,
        streaming_micro_target_length=args.streaming_micro_target_length,
        streaming_micro_min_length=args.streaming_micro_min_length,
        streaming_micro_max_length=args.streaming_micro_max_length,
        streaming_move_relres_threshold=args.streaming_move_relres_threshold,
        streaming_alpha_advance_threshold=args.streaming_alpha_advance_threshold,
        streaming_micro_max_corrections=args.streaming_micro_max_corrections,
        streaming_basis_max_vectors=args.streaming_basis_max_vectors,
        omega_no_increase_newton_threshold=args.omega_no_increase_newton_threshold,
        omega_half_newton_threshold=args.omega_half_newton_threshold,
        omega_target_newton_iterations=args.omega_target_newton_iterations,
        omega_adapt_min_scale=args.omega_adapt_min_scale,
        omega_adapt_max_scale=args.omega_adapt_max_scale,
        omega_hard_newton_threshold=args.omega_hard_newton_threshold,
        omega_hard_linear_threshold=args.omega_hard_linear_threshold,
        omega_efficiency_floor=args.omega_efficiency_floor,
        omega_efficiency_drop_ratio=args.omega_efficiency_drop_ratio,
        omega_efficiency_window=args.omega_efficiency_window,
        omega_hard_shrink_scale=args.omega_hard_shrink_scale,
        step_length_cap_mode=args.step_length_cap_mode,
        step_length_cap_factor=args.step_length_cap_factor,
        it_newt_max=args.it_newt_max,
        it_damp_max=args.it_damp_max,
        tol=args.tol,
        r_min=args.r_min,
        newton_stopping_criterion=args.newton_stopping_criterion,
        newton_stopping_tol=args.newton_stopping_tol,
        init_newton_stopping_criterion=args.init_newton_stopping_criterion,
        init_newton_stopping_tol=args.init_newton_stopping_tol,
        fine_newton_stopping_criterion=args.fine_newton_stopping_criterion,
        fine_newton_stopping_tol=args.fine_newton_stopping_tol,
        fine_switch_mode=args.fine_switch_mode,
        fine_switch_distance_factor=args.fine_switch_distance_factor,
        linear_tolerance=args.linear_tolerance,
        linear_max_iter=args.linear_max_iter,
        solver_type=args.solver_type,
        factor_solver_type=args.factor_solver_type,
        pc_backend=args.pc_backend,
        pmg_coarse_mesh_path=args.pmg_coarse_mesh_path,
        pmg_fine_hierarchy_mode=args.pmg_fine_hierarchy_mode,
        preconditioner_matrix_source=args.preconditioner_matrix_source,
        preconditioner_matrix_policy=args.preconditioner_matrix_policy,
        preconditioner_rebuild_policy=args.preconditioner_rebuild_policy,
        preconditioner_rebuild_interval=args.preconditioner_rebuild_interval,
        mpi_distribute_by_nodes=args.mpi_distribute_by_nodes,
        pc_gamg_process_eq_limit=args.pc_gamg_process_eq_limit,
        pc_gamg_threshold=args.pc_gamg_threshold,
        pc_gamg_aggressive_coarsening=args.pc_gamg_aggressive_coarsening,
        pc_gamg_aggressive_square_graph=args.pc_gamg_aggressive_square_graph,
        pc_gamg_aggressive_mis_k=args.pc_gamg_aggressive_mis_k,
        pc_hypre_coarsen_type=args.pc_hypre_coarsen_type,
        pc_hypre_interp_type=args.pc_hypre_interp_type,
        pc_hypre_strong_threshold=args.pc_hypre_strong_threshold,
        pc_hypre_boomeramg_max_iter=args.pc_hypre_boomeramg_max_iter,
        pc_hypre_P_max=args.pc_hypre_P_max,
        pc_hypre_agg_nl=args.pc_hypre_agg_nl,
        pc_hypre_nongalerkin_tol=args.pc_hypre_nongalerkin_tol,
        pc_bddc_symmetric=args.pc_bddc_symmetric,
        pc_bddc_dirichlet_ksp_type=args.pc_bddc_dirichlet_ksp_type,
        pc_bddc_dirichlet_pc_type=args.pc_bddc_dirichlet_pc_type,
        pc_bddc_neumann_ksp_type=args.pc_bddc_neumann_ksp_type,
        pc_bddc_neumann_pc_type=args.pc_bddc_neumann_pc_type,
        pc_bddc_coarse_ksp_type=args.pc_bddc_coarse_ksp_type,
        pc_bddc_coarse_pc_type=args.pc_bddc_coarse_pc_type,
        pc_bddc_dirichlet_approximate=args.pc_bddc_dirichlet_approximate,
        pc_bddc_neumann_approximate=args.pc_bddc_neumann_approximate,
        pc_bddc_monolithic=args.pc_bddc_monolithic,
        pc_bddc_coarse_redundant_pc_type=args.pc_bddc_coarse_redundant_pc_type,
        pc_bddc_switch_static=args.pc_bddc_switch_static,
        pc_bddc_use_deluxe_scaling=args.pc_bddc_use_deluxe_scaling,
        pc_bddc_use_vertices=args.pc_bddc_use_vertices,
        pc_bddc_use_edges=args.pc_bddc_use_edges,
        pc_bddc_use_faces=args.pc_bddc_use_faces,
        pc_bddc_use_change_of_basis=args.pc_bddc_use_change_of_basis,
        pc_bddc_use_change_on_faces=args.pc_bddc_use_change_on_faces,
        pc_bddc_check_level=args.pc_bddc_check_level,
        petsc_opt=args.petsc_opt,
        compiled_outer=args.compiled_outer,
        recycle_preconditioner=args.recycle_preconditioner,
        constitutive_mode=args.constitutive_mode,
        tangent_kernel=args.tangent_kernel,
        max_deflation_basis_vectors=args.max_deflation_basis_vectors,
        store_step_u=args.store_step_u,
    )
    if PETSc.COMM_WORLD.getRank() == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
