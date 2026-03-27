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
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.sparse import linalg as sparse_linalg

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
    build_3d_pmg_hierarchy,
    build_3d_same_mesh_pmg_hierarchy,
)
from slope_stability.constitutive import ConstitutiveOperator
from slope_stability.continuation import indirect as indirect_module
from slope_stability.continuation import LL_indirect_continuation, SSR_indirect_continuation
from slope_stability.nonlinear.newton import (
    _destroy_petsc_mat,
    _is_builder_cached_matrix,
    _prefers_full_system_operator,
    _setup_linear_system,
    _solve_linear_system,
    newton_ind_ssr,
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


def _retrospective_predictor_specs() -> tuple[tuple[str, int], ...]:
    return (
        ("secant", 2),
        ("two_step_quadratic", 3),
        ("three_step_cubic", 4),
    )


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


def _lagrange_coefficients(xs: np.ndarray, xt: float) -> np.ndarray:
    coeff = np.ones(xs.size, dtype=np.float64)
    for j in range(xs.size):
        for k in range(xs.size):
            if j != k:
                coeff[j] *= (xt - xs[k]) / (xs[j] - xs[k])
    return coeff


def _retrospective_predict_vector(
    method: str,
    omega_hist: np.ndarray,
    U_flat: np.ndarray,
    state_idx: int,
) -> np.ndarray:
    if method == "secant":
        omega0 = float(omega_hist[state_idx - 2])
        omega1 = float(omega_hist[state_idx - 1])
        scale = (float(omega_hist[state_idx]) - omega1) / (omega1 - omega0)
        u0 = U_flat[state_idx - 2]
        u1 = U_flat[state_idx - 1]
        return u1 + scale * (u1 - u0)
    if method == "two_step_quadratic":
        xs = np.asarray(omega_hist[state_idx - 3 : state_idx], dtype=np.float64)
        coeff = _lagrange_coefficients(xs, float(omega_hist[state_idx]))
        return coeff @ U_flat[state_idx - 3 : state_idx]
    if method == "three_step_cubic":
        xs = np.asarray(omega_hist[state_idx - 4 : state_idx], dtype=np.float64)
        coeff = _lagrange_coefficients(xs, float(omega_hist[state_idx]))
        return coeff @ U_flat[state_idx - 4 : state_idx]
    raise ValueError(f"Unsupported retrospective predictor {method!r}")


def _retrospective_predict_scalar(
    method: str,
    omega_hist: np.ndarray,
    value_hist: np.ndarray,
    state_idx: int,
) -> float:
    if method == "secant":
        omega0 = float(omega_hist[state_idx - 2])
        omega1 = float(omega_hist[state_idx - 1])
        scale = (float(omega_hist[state_idx]) - omega1) / (omega1 - omega0)
        return float(value_hist[state_idx - 1] + scale * (value_hist[state_idx - 1] - value_hist[state_idx - 2]))
    if method == "two_step_quadratic":
        xs = np.asarray(omega_hist[state_idx - 3 : state_idx], dtype=np.float64)
        coeff = _lagrange_coefficients(xs, float(omega_hist[state_idx]))
        return float(np.dot(coeff, value_hist[state_idx - 3 : state_idx]))
    if method == "three_step_cubic":
        xs = np.asarray(omega_hist[state_idx - 4 : state_idx], dtype=np.float64)
        coeff = _lagrange_coefficients(xs, float(omega_hist[state_idx]))
        return float(np.dot(coeff, value_hist[state_idx - 4 : state_idx]))
    raise ValueError(f"Unsupported retrospective predictor {method!r}")


def _compute_retrospective_predictor_export(
    *,
    step_u: np.ndarray,
    omega_hist: np.ndarray,
    lambda_hist: np.ndarray,
    step_index: np.ndarray,
    step_omega: np.ndarray,
    step_lambda: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if step_u.ndim != 3 or step_u.shape[0] == 0:
        return {}, {
            "available": False,
            "reason": "step_U_unavailable",
            "methods": {},
        }

    state_count = int(step_u.shape[0])
    cont_count = int(step_index.size)
    expected_state_count = cont_count + 2
    if state_count != expected_state_count:
        return {}, {
            "available": False,
            "reason": "state_count_mismatch",
            "state_count": state_count,
            "expected_state_count": expected_state_count,
            "methods": {},
        }

    U_flat = np.asarray(step_u, dtype=np.float64).reshape(state_count, -1)
    omega_hist = np.asarray(omega_hist, dtype=np.float64)
    lambda_hist = np.asarray(lambda_hist, dtype=np.float64)
    step_index = np.asarray(step_index, dtype=np.int64)
    step_omega = np.asarray(step_omega, dtype=np.float64)
    step_lambda = np.asarray(step_lambda, dtype=np.float64)

    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, object] = {
        "available": True,
        "state_count": state_count,
        "continuation_state_count": cont_count,
        "methods": {},
    }

    for method, min_state_index in _retrospective_predictor_specs():
        u_l2_abs = np.full(cont_count, np.nan, dtype=np.float64)
        u_l2_rel = np.full(cont_count, np.nan, dtype=np.float64)
        u_increment_rel = np.full(cont_count, np.nan, dtype=np.float64)
        u_max_node_abs = np.full(cont_count, np.nan, dtype=np.float64)
        lambda_pred = np.full(cont_count, np.nan, dtype=np.float64)
        lambda_abs_err = np.full(cont_count, np.nan, dtype=np.float64)

        for local_idx in range(cont_count):
            state_idx = local_idx + 2
            if state_idx < min_state_index:
                continue
            predicted = _retrospective_predict_vector(method, omega_hist, U_flat, state_idx)
            truth = U_flat[state_idx]
            previous = U_flat[state_idx - 1]
            diff = predicted - truth
            l2_abs = float(np.linalg.norm(diff))
            truth_norm = max(float(np.linalg.norm(truth)), 1.0e-30)
            increment_norm = max(float(np.linalg.norm(truth - previous)), 1.0e-30)
            diff_vec = diff.reshape(step_u.shape[1], -1)

            u_l2_abs[local_idx] = l2_abs
            u_l2_rel[local_idx] = l2_abs / truth_norm
            u_increment_rel[local_idx] = l2_abs / increment_norm
            u_max_node_abs[local_idx] = float(np.linalg.norm(diff_vec, axis=0).max())
            lambda_pred[local_idx] = _retrospective_predict_scalar(method, omega_hist, lambda_hist, state_idx)
            lambda_abs_err[local_idx] = abs(lambda_pred[local_idx] - float(lambda_hist[state_idx]))

        prefix = f"retrospective_predictor_{method}"
        arrays[f"{prefix}_u_l2_abs"] = u_l2_abs
        arrays[f"{prefix}_u_l2_rel"] = u_l2_rel
        arrays[f"{prefix}_u_increment_rel"] = u_increment_rel
        arrays[f"{prefix}_u_max_node_abs"] = u_max_node_abs
        arrays[f"{prefix}_lambda_pred"] = lambda_pred
        arrays[f"{prefix}_lambda_abs_err"] = lambda_abs_err

        finite_mask = np.isfinite(u_l2_rel)
        targets = int(np.count_nonzero(finite_mask))
        if targets == 0:
            method_summary = {
                "targets": 0,
                "min_state_index": min_state_index,
            }
        else:
            method_summary = {
                "targets": targets,
                "min_state_index": min_state_index,
                "mean_u_l2_rel": float(np.nanmean(u_l2_rel)),
                "max_u_l2_rel": float(np.nanmax(u_l2_rel)),
                "mean_u_increment_rel": float(np.nanmean(u_increment_rel)),
                "max_u_increment_rel": float(np.nanmax(u_increment_rel)),
                "mean_u_l2_abs": float(np.nanmean(u_l2_abs)),
                "max_u_l2_abs": float(np.nanmax(u_l2_abs)),
                "mean_u_max_node_abs": float(np.nanmean(u_max_node_abs)),
                "max_u_max_node_abs": float(np.nanmax(u_max_node_abs)),
                "mean_lambda_abs_err": float(np.nanmean(lambda_abs_err)),
                "max_lambda_abs_err": float(np.nanmax(lambda_abs_err)),
            }
        summary["methods"][method] = method_summary

    return arrays, summary


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


def _pmg_transfer_to_global_csr(transfer) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (
            np.asarray(transfer.coo_data, dtype=np.float64),
            (
                np.asarray(transfer.coo_rows, dtype=np.int64),
                np.asarray(transfer.coo_cols, dtype=np.int64),
            ),
        ),
        shape=tuple(int(v) for v in transfer.global_shape),
        dtype=np.float64,
    ).tocsr()


def _build_owned_problem(
    *,
    mesh_path: Path,
    elem_type: str,
    mesh_boundary_type: int,
    node_ordering: str,
    reorder_parts: int | None,
    material_rows: list[list[float]] | None,
    davis_type: str,
    constitutive_mode: str,
    tangent_kernel: str,
    comm=None,
) -> dict[str, object]:
    if comm is None:
        comm = PETSc.COMM_WORLD
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
    mesh = load_mesh_from_file(mesh_path, boundary_type=int(mesh_boundary_type), elem_type=str(elem_type))
    reordered = reorder_mesh_nodes(
        mesh.coord,
        mesh.elem,
        mesh.surf,
        mesh.q_mask,
        strategy=node_ordering,
        n_parts=reorder_parts if str(node_ordering).lower() == "block_metis" else None,
    )
    coord = np.asarray(reordered.coord, dtype=np.float64)
    elem = np.asarray(reordered.elem, dtype=np.int64)
    surf = np.asarray(reordered.surf, dtype=np.int64)
    q_mask = np.asarray(reordered.q_mask, dtype=bool)
    material_identifier = np.asarray(mesh.material, dtype=np.int64).ravel()

    elastic_rows = assemble_owned_elastic_rows_for_comm(
        coord,
        elem,
        q_mask,
        material_identifier,
        materials,
        comm,
        elem_type=str(elem_type),
    )
    n_q = int(quadrature_volume_3d(elem_type)[0].shape[1])
    n_int = int(elem.shape[1] * n_q)
    c0, phi, psi, shear, bulk, lame, gamma = heterogenous_materials(
        material_identifier,
        np.ones(n_int, dtype=bool),
        n_q,
        materials,
    )
    global_size = int(coord.shape[0] * coord.shape[1])
    if int(comm.getSize()) == 1:
        K_elast = to_petsc_aij_matrix(
            elastic_rows.local_matrix,
            comm=comm,
            block_size=coord.shape[0],
        )
        rhs_parts = [np.asarray(elastic_rows.local_rhs, dtype=np.float64)]
    else:
        K_elast = local_csr_to_petsc_aij_matrix(
            elastic_rows.local_matrix,
            global_shape=(global_size, global_size),
            comm=comm,
            block_size=coord.shape[0],
        )
        mpi_comm = comm.tompi4py() if hasattr(comm, "tompi4py") else comm
        rhs_parts = mpi_comm.allgather(np.asarray(elastic_rows.local_rhs, dtype=np.float64))
    f_V = np.concatenate(rhs_parts).reshape(coord.shape[0], coord.shape[1], order="F")
    const_builder = ConstitutiveOperator(
        B=None,
        c0=c0,
        phi=phi,
        psi=psi,
        Davis_type=str(davis_type),
        shear=shear,
        bulk=bulk,
        lame=lame,
        WEIGHT=np.zeros(n_int, dtype=np.float64),
        n_strain=6,
        n_int=n_int,
        dim=3,
        q_mask=q_mask,
    )
    row0, row1 = owned_block_range(coord.shape[1], coord.shape[0], comm)
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
    const_builder._owned_comm = comm
    return {
        "coord": coord,
        "elem": elem,
        "surf": surf,
        "q_mask": q_mask,
        "K_elast": K_elast,
        "f_V": f_V,
        "const_builder": const_builder,
        "tangent_pattern": tangent_pattern,
        "free_idx": q_to_free_indices(q_mask),
    }


def _clone_hypre_preconditioner_options(preconditioner_options: dict[str, object]) -> dict[str, object]:
    coarse_options = dict(preconditioner_options)
    coarse_options["pc_backend"] = "hypre"
    coarse_options.pop("pmg_hierarchy", None)
    coarse_options.pop("pmg_coarse_mesh_path", None)
    for key in tuple(coarse_options.keys()):
        if key.startswith("mg_") or key.startswith("pc_mg_"):
            coarse_options.pop(key, None)
    return coarse_options


def _solver_basis_snapshot(solver):
    getter = getattr(solver, "get_deflation_basis_snapshot", None)
    if callable(getter):
        return getter()
    basis = getattr(solver, "deflation_basis", None)
    if basis is None:
        return None
    return np.array(basis, dtype=np.float64, copy=True)


def _solver_basis_restore(solver, snapshot) -> None:
    restore = getattr(solver, "restore_deflation_basis", None)
    if callable(restore):
        restore(snapshot)
        return
    if hasattr(solver, "deflation_basis"):
        solver.deflation_basis = np.array(snapshot, dtype=np.float64, copy=True)


def _solver_notify_attempt(solver, *, success: bool) -> None:
    notify = getattr(solver, "notify_continuation_attempt", None)
    if callable(notify):
        notify(success=bool(success))


def _coarse_free_orthonormal_basis(vectors: list[np.ndarray], *, tol: float = 1.0e-10) -> tuple[np.ndarray | None, float]:
    if not vectors:
        return None, np.nan
    cols = [np.asarray(v, dtype=np.float64).reshape(-1) for v in vectors]
    raw = np.column_stack(cols)
    if raw.size == 0:
        return None, np.nan
    singular = np.linalg.svd(raw, compute_uv=False)
    cond = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0.0 else np.inf
    qmat, rmat = np.linalg.qr(raw, mode="reduced")
    diag = np.abs(np.diag(rmat))
    scale = float(np.max(diag)) if diag.size else 0.0
    if scale <= 0.0:
        return None, cond
    keep = diag > tol * scale
    if not np.any(keep):
        return None, cond
    return np.asarray(qmat[:, keep], dtype=np.float64), cond


def _build_p4_l1_coarse_predictor_context(
    *,
    mesh_path: Path,
    mesh_boundary_type: int,
    node_ordering: str,
    reorder_parts: int | None,
    material_rows: list[list[float]],
    davis_type: str,
    constitutive_mode: str,
    tangent_kernel: str,
    solver_type: str,
    linear_tolerance: float,
    linear_max_iter: int,
    lambda_init: float,
    d_lambda_init: float,
    d_lambda_min: float,
    it_newt_max: int,
    it_damp_max: int,
    tol: float,
    r_min: float,
    fine_q_mask: np.ndarray,
    fine_f_V: np.ndarray,
    fine_constitutive_matrix_builder,
    pmg_hierarchy,
    preconditioner_options: dict[str, object],
) -> dict[str, object]:
    if pmg_hierarchy is None:
        return {}
    level_orders = tuple(int(getattr(level, "order", -1)) for level in getattr(pmg_hierarchy, "levels", ()))
    if level_orders != (1, 2, 4):
        return {}

    p21 = _pmg_transfer_to_global_csr(pmg_hierarchy.prolongation_p21)
    p42 = _pmg_transfer_to_global_csr(pmg_hierarchy.prolongation_p42)
    p41 = (p42 @ p21).tocsr()
    world_comm = MPI.COMM_WORLD
    world_rank = int(world_comm.Get_rank())
    is_coarse_root = world_rank == 0
    coarse_problem = None
    coarse_solver = None
    if is_coarse_root:
        coarse_problem = _build_owned_problem(
            mesh_path=Path(mesh_path),
            elem_type="P1",
            mesh_boundary_type=int(mesh_boundary_type),
            node_ordering=node_ordering,
            reorder_parts=reorder_parts,
            material_rows=[list(map(float, row)) for row in material_rows],
            davis_type=davis_type,
            constitutive_mode=constitutive_mode,
            tangent_kernel=tangent_kernel,
            comm=PETSc.COMM_SELF,
        )
        coarse_preconditioner_options = _clone_hypre_preconditioner_options(preconditioner_options)
        coarse_solver = SolverFactory.create(
            "KSPPREONLY_LU",
            tolerance=float(linear_tolerance),
            max_iterations=int(linear_max_iter),
            deflation_basis_tolerance=1e-3,
            verbose=False,
            q_mask=coarse_problem["q_mask"],
            coord=coarse_problem["coord"],
            preconditioner_options=coarse_preconditioner_options,
        )

    fine_q_mask = np.asarray(fine_q_mask, dtype=bool)
    fine_free_idx = q_to_free_indices(fine_q_mask)
    fine_shape = tuple(int(v) for v in np.asarray(fine_f_V).shape)
    if is_coarse_root:
        coarse_q_mask = np.asarray(coarse_problem["q_mask"], dtype=bool)
        coarse_free_idx = np.asarray(coarse_problem["free_idx"], dtype=np.int64)
        coarse_shape = tuple(int(v) for v in np.asarray(coarse_problem["f_V"]).shape)
        coarse_f_full = flatten_field(np.asarray(coarse_problem["f_V"], dtype=np.float64))
        coarse_f_free = np.asarray(coarse_f_full[coarse_free_idx], dtype=np.float64)
        coarse_norm_f = max(float(np.linalg.norm(coarse_f_free)), 1.0)
    else:
        coarse_q_mask = None
        coarse_free_idx = None
        coarse_shape = None
        coarse_f_full = None
        coarse_f_free = None
        coarse_norm_f = np.nan
    coarse_state = {
        "initialized": False,
        "init_error": None,
        "U_hist": [],
        "omega_hist": [],
        "lambda_hist": [],
        "pending": None,
    }

    def _project_fine_full_to_coarse_full(U_fine: np.ndarray) -> np.ndarray:
        fine_free = np.asarray(flatten_field(U_fine)[fine_free_idx], dtype=np.float64)
        coarse_free = np.asarray(
            sparse_linalg.lsmr(
                p41,
                fine_free,
                atol=1.0e-12,
                btol=1.0e-12,
                maxiter=200,
            )[0],
            dtype=np.float64,
        ).reshape(-1)
        return full_field_from_free_values(coarse_free, coarse_free_idx, coarse_shape)

    def _project_increment_to_coarse_free(dU_fine: np.ndarray) -> np.ndarray:
        fine_free = np.asarray(flatten_field(dU_fine)[fine_free_idx], dtype=np.float64)
        return np.asarray(
            sparse_linalg.lsmr(
                p41,
                fine_free,
                atol=1.0e-12,
                btol=1.0e-12,
                maxiter=200,
            )[0],
            dtype=np.float64,
        ).reshape(-1)

    def _prolongate_coarse_full_to_fine_full(U_coarse: np.ndarray) -> np.ndarray:
        coarse_free = np.asarray(flatten_field(U_coarse)[coarse_free_idx], dtype=np.float64)
        fine_free = np.asarray(p41 @ coarse_free, dtype=np.float64).reshape(-1)
        return full_field_from_free_values(fine_free, fine_free_idx, fine_shape)

    def _coarse_secant_full(*, omega_old: float, omega: float, omega_target: float, U_old: np.ndarray, U: np.ndarray) -> np.ndarray:
        denom = float(omega - omega_old)
        if denom == 0.0:
            return np.asarray(U, dtype=np.float64)
        return np.asarray(U, dtype=np.float64) + ((float(omega_target) - float(omega)) / denom) * (
            np.asarray(U, dtype=np.float64) - np.asarray(U_old, dtype=np.float64)
        )

    def _coarse_secant_lambda(*, omega_old: float, omega: float, omega_target: float, lambda_old: float, lambda_value: float) -> float:
        denom = float(omega - omega_old)
        if abs(denom) <= 1.0e-30:
            return float(lambda_value)
        alpha = (float(omega_target) - float(omega)) / denom
        return float(lambda_value + alpha * (float(lambda_value) - float(lambda_old)))

    def _refine_fine_lambda_for_fixed_u(
        *,
        U_pred: np.ndarray,
        omega_old: float,
        omega_now: float,
        omega_target: float,
        lambda_old: float,
        lambda_now: float,
        Q: np.ndarray,
        f: np.ndarray,
    ) -> tuple[float, float, int]:
        lambda_center = _coarse_secant_lambda(
            omega_old=float(omega_old),
            omega=float(omega_now),
            omega_target=float(omega_target),
            lambda_old=float(lambda_old),
            lambda_value=float(lambda_now),
        )
        lam_candidates = np.asarray([float(lambda_old), float(lambda_now), float(lambda_center)], dtype=np.float64)
        lam_pos = lam_candidates[np.isfinite(lam_candidates) & (lam_candidates > 1.0e-8)]
        if lam_pos.size == 0:
            return max(float(lambda_now), 1.0e-6), np.inf, 0
        lam_lo = max(1.0e-6, 0.5 * float(np.min(lam_pos)))
        lam_hi = max(lam_lo * 1.01, 2.0 * float(np.max(lam_pos)))

        def _merit(lambda_trial: float) -> float:
            return float(
                indirect_module._predictor_residual_penalty_merit(
                    U=np.asarray(U_pred, dtype=np.float64),
                    lambda_value=float(lambda_trial),
                    omega_target=float(omega_target),
                    Q=np.asarray(Q, dtype=bool),
                    f=np.asarray(f, dtype=np.float64),
                    constitutive_matrix_builder=fine_constitutive_matrix_builder,
                    penalty_weight=0.0,
                )
            )

        lambda_star, merit_star, n_eval = indirect_module._bounded_scalar_minimize(
            _merit,
            lower=float(lam_lo),
            upper=float(lam_hi),
            max_evals=8,
        )
        return float(lambda_star), float(merit_star), int(n_eval)

    def _coarse_expand_basis(U_full: np.ndarray) -> None:
        if coarse_solver is None:
            return
        if getattr(coarse_solver, "supports_dynamic_deflation_basis", lambda: True)():
            coarse_solver.expand_deflation_basis(flatten_field(np.asarray(U_full, dtype=np.float64))[coarse_free_idx])

    def _ensure_coarse_initialized() -> None:
        if not is_coarse_root:
            return
        if coarse_state["initialized"] or coarse_state["init_error"] is not None:
            return
        try:
            U1, U2, omega1, omega2, lambda1, lambda2, _ = indirect_module.init_phase_SSR_indirect_continuation(
                lambda_init=float(lambda_init),
                d_lambda_init=float(d_lambda_init),
                d_lambda_min=float(d_lambda_min),
                it_newt_max=min(int(it_newt_max), 20),
                it_damp_max=min(int(it_damp_max), 8),
                tol=max(float(tol) * 10.0, 1.0e-3),
                r_min=float(r_min),
                K_elast=coarse_problem["K_elast"],
                Q=coarse_problem["q_mask"],
                f=coarse_problem["f_V"],
                constitutive_matrix_builder=coarse_problem["const_builder"],
                linear_system_solver=coarse_solver,
            )
            coarse_state["U_hist"] = [np.asarray(U1, dtype=np.float64), np.asarray(U2, dtype=np.float64)]
            coarse_state["omega_hist"] = [float(omega1), float(omega2)]
            coarse_state["lambda_hist"] = [float(lambda1), float(lambda2)]
            coarse_state["initialized"] = True
            coarse_state["pending"] = None
        except Exception as exc:
            coarse_state["init_error"] = repr(exc)

    def _coarse_solve_from_history(
        *,
        omega_target: float,
        U_old: np.ndarray,
        U_now: np.ndarray,
        omega_old: float,
        omega_now: float,
        lambda_old: float,
        lambda_now: float,
    ) -> dict[str, object]:
        info: dict[str, object] = {
            "success": False,
            "wall_time": 0.0,
            "newton_iterations": np.nan,
            "residual_end": np.nan,
            "error": None,
        }
        U_ini = _coarse_secant_full(
            omega_old=float(omega_old),
            omega=float(omega_now),
            omega_target=float(omega_target),
            U_old=np.asarray(U_old, dtype=np.float64),
            U=np.asarray(U_now, dtype=np.float64),
        )
        U_ini = _rescale_to_target_omega(U_ini, float(omega_target), coarse_problem["f_V"], coarse_problem["q_mask"])
        lambda_ini = _coarse_secant_lambda(
            omega_old=float(omega_old),
            omega=float(omega_now),
            omega_target=float(omega_target),
            lambda_old=float(lambda_old),
            lambda_value=float(lambda_now),
        )
        basis_before = _solver_basis_snapshot(coarse_solver)
        reduction_orig = coarse_problem["const_builder"].reduction
        coarse_lambda_clip_count = {"value": 0}

        def _safe_reduction(lambda_value: float):
            lam = max(float(lambda_value), 1.0e-6)
            if lam != float(lambda_value):
                coarse_lambda_clip_count["value"] += 1
            return reduction_orig(lam)

        coarse_problem["const_builder"].reduction = _safe_reduction
        solve_t0 = perf_counter()
        try:
            U_sol, lambda_sol, flag, it_used, history = newton_ind_ssr(
                U_ini,
                float(omega_target),
                float(lambda_ini),
                min(int(it_newt_max), 20),
                min(int(it_damp_max), 8),
                max(float(tol) * 10.0, 1.0e-3),
                float(r_min),
                coarse_problem["K_elast"],
                coarse_problem["q_mask"],
                coarse_problem["f_V"],
                coarse_problem["const_builder"],
                coarse_solver,
            )
        finally:
            coarse_problem["const_builder"].reduction = reduction_orig
        info["wall_time"] = float(perf_counter() - solve_t0)
        info["newton_iterations"] = float(it_used)
        if history["residual"].size:
            info["residual_end"] = float(history["residual"][-1])
        info["lambda_clip_count"] = float(coarse_lambda_clip_count["value"])
        _solver_notify_attempt(coarse_solver, success=(flag == 0))
        if flag == 0:
            _coarse_expand_basis(U_sol)
            info["success"] = True
            info["U"] = np.asarray(U_sol, dtype=np.float64)
            info["lambda"] = float(lambda_sol)
            info["omega"] = float(omega_target)
        else:
            _solver_basis_restore(coarse_solver, basis_before)
            info["error"] = "coarse_solution_newton_failed"
        return info

    def _coarse_commit_pending_if_matching(fine_omega_hist: tuple[float, ...]) -> None:
        pending = coarse_state.get("pending")
        if pending is None:
            return
        coarse_len = len(coarse_state["omega_hist"])
        if coarse_len >= len(fine_omega_hist):
            return
        next_target = float(fine_omega_hist[coarse_len])
        if abs(float(pending["omega"]) - next_target) > max(1.0e-6 * max(abs(next_target), 1.0), 1.0e-8):
            return
        coarse_state["U_hist"].append(np.asarray(pending["U"], dtype=np.float64))
        coarse_state["omega_hist"].append(float(pending["omega"]))
        coarse_state["lambda_hist"].append(float(pending["lambda"]))
        coarse_state["pending"] = None

    def _coarse_sync_to_fine_history(fine_omega_hist: tuple[float, ...]) -> dict[str, float]:
        _ensure_coarse_initialized()
        if coarse_state["init_error"] is not None:
            raise RuntimeError(str(coarse_state["init_error"]))
        _coarse_commit_pending_if_matching(fine_omega_hist)
        accum = {"wall_time": 0.0, "newton_iterations": 0.0, "residual_end": np.nan}
        while len(coarse_state["omega_hist"]) < len(fine_omega_hist):
            coarse_len = len(coarse_state["omega_hist"])
            target = float(fine_omega_hist[coarse_len])
            result = _coarse_solve_from_history(
                omega_target=target,
                U_old=np.asarray(coarse_state["U_hist"][-2], dtype=np.float64),
                U_now=np.asarray(coarse_state["U_hist"][-1], dtype=np.float64),
                omega_old=float(coarse_state["omega_hist"][-2]),
                omega_now=float(coarse_state["omega_hist"][-1]),
                lambda_old=float(coarse_state["lambda_hist"][-2]),
                lambda_now=float(coarse_state["lambda_hist"][-1]),
            )
            accum["wall_time"] += float(result["wall_time"])
            accum["newton_iterations"] += float(result["newton_iterations"])
            if np.isfinite(float(result["residual_end"])):
                accum["residual_end"] = float(result["residual_end"])
            if not bool(result["success"]):
                raise RuntimeError(str(result.get("error") or "coarse_history_sync_failed"))
            coarse_state["U_hist"].append(np.asarray(result["U"], dtype=np.float64))
            coarse_state["omega_hist"].append(float(result["omega"]))
            coarse_state["lambda_hist"].append(float(result["lambda"]))
        return accum

    def _coarse_solution_predictor(**kwargs):
        payload = None
        if is_coarse_root:
            info = _predictor_info_defaults()
            t0 = perf_counter()
            try:
                lambda_ini = float(kwargs["lambda_value"])
                fine_omega_hist = tuple(float(v) for v in kwargs.get("predictor_omega_hist", ()))
                sync = _coarse_sync_to_fine_history(fine_omega_hist)
                info["coarse_solve_wall_time"] = float(sync["wall_time"])
                info["coarse_newton_iterations"] = float(sync["newton_iterations"])
                if np.isfinite(float(sync["residual_end"])):
                    info["coarse_residual_end"] = float(sync["residual_end"])

                coarse_result = _coarse_solve_from_history(
                    omega_target=float(kwargs["omega_target"]),
                    U_old=np.asarray(coarse_state["U_hist"][-2], dtype=np.float64),
                    U_now=np.asarray(coarse_state["U_hist"][-1], dtype=np.float64),
                    omega_old=float(coarse_state["omega_hist"][-2]),
                    omega_now=float(coarse_state["omega_hist"][-1]),
                    lambda_old=float(coarse_state["lambda_hist"][-2]),
                    lambda_now=float(coarse_state["lambda_hist"][-1]),
                )
                info["coarse_solve_wall_time"] += float(coarse_result["wall_time"])
                info["coarse_newton_iterations"] += float(coarse_result["newton_iterations"])
                if np.isfinite(float(coarse_result["residual_end"])):
                    info["coarse_residual_end"] = float(coarse_result["residual_end"])
                if not bool(coarse_result["success"]):
                    info["fallback_used"] = True
                    info["fallback_error"] = str(coarse_result.get("error") or "coarse_solution_newton_failed")
                    U_fallback, lambda_sec, _ = indirect_module._secant_predictor(
                        omega_old=float(kwargs["omega_old"]),
                        omega=float(kwargs["omega"]),
                        omega_target=float(kwargs["omega_target"]),
                        U_old=np.asarray(kwargs["U_old"], dtype=np.float64),
                        U=np.asarray(kwargs["U"], dtype=np.float64),
                        lambda_value=lambda_ini,
                    )
                    info["predictor_wall_time"] = float(perf_counter() - t0)
                    payload = (U_fallback, lambda_sec, "coarse_p1_solution_fallback_secant", info)
                else:
                    coarse_state["pending"] = {
                        "U": np.asarray(coarse_result["U"], dtype=np.float64),
                        "omega": float(coarse_result["omega"]),
                        "lambda": float(coarse_result["lambda"]),
                    }
                    U_pred = _prolongate_coarse_full_to_fine_full(np.asarray(coarse_result["U"], dtype=np.float64))
                    U_pred = _rescale_to_target_omega(
                        U_pred,
                        float(kwargs["omega_target"]),
                        np.asarray(kwargs["f"], dtype=np.float64),
                        np.asarray(kwargs["Q"], dtype=bool),
                    )
                    info["projected_delta_lambda"] = float(coarse_result["lambda"] - lambda_ini)
                    info["predictor_wall_time"] = float(perf_counter() - t0)
                    payload = (U_pred, float(lambda_ini), "coarse_p1_solution", info)
            except Exception as exc:
                info["predictor_wall_time"] = float(perf_counter() - t0)
                info["fallback_used"] = True
                info["fallback_error"] = repr(exc)
                U_sec, lambda_ini, _ = indirect_module._secant_predictor(
                    omega_old=float(kwargs["omega_old"]),
                    omega=float(kwargs["omega"]),
                    omega_target=float(kwargs["omega_target"]),
                    U_old=np.asarray(kwargs["U_old"], dtype=np.float64),
                    U=np.asarray(kwargs["U"], dtype=np.float64),
                    lambda_value=float(kwargs["lambda_value"]),
                )
                payload = (U_sec, lambda_ini, "coarse_p1_solution_fallback_secant", info)
        U_pred, lambda_pred, kind, info = world_comm.bcast(payload, root=0)
        return np.asarray(U_pred, dtype=np.float64), float(lambda_pred), str(kind), dict(info)

    def _coarse_reduced_newton_predictor(**kwargs):
        payload = None
        if is_coarse_root:
            info = _predictor_info_defaults()
            t0 = perf_counter()
            try:
                basis_free_raw = [
                    _project_increment_to_coarse_free(np.asarray(dU, dtype=np.float64))
                    for dU in kwargs.get("continuation_increment_hist", ())
                ]
                basis_free, basis_cond = _coarse_free_orthonormal_basis(basis_free_raw)
                info["basis_dim"] = float(0 if basis_free is None else basis_free.shape[1])
                info["basis_condition"] = float(basis_cond)
                if basis_free is None or basis_free.shape[1] == 0:
                    info["fallback_used"] = True
                    info["fallback_error"] = "empty_reduced_basis"
                    U_pred, lambda_pred, kind, coarse_info = _coarse_solution_predictor(**kwargs)
                    coarse_info.update(info)
                    coarse_info["fallback_used"] = True
                    coarse_info["fallback_error"] = "empty_reduced_basis"
                    payload = (U_pred, lambda_pred, "coarse_p1_reduced_newton_fallback_solution", coarse_info)
                else:
                    omega_old = float(kwargs["omega_old"])
                    omega = float(kwargs["omega"])
                    omega_target = float(kwargs["omega_target"])
                    U_old_coarse = _project_fine_full_to_coarse_full(np.asarray(kwargs["U_old"], dtype=np.float64))
                    U_coarse = _project_fine_full_to_coarse_full(np.asarray(kwargs["U"], dtype=np.float64))
                    U_it = _coarse_secant_full(
                        omega_old=omega_old,
                        omega=omega,
                        omega_target=omega_target,
                        U_old=U_old_coarse,
                        U=U_coarse,
                    )
                    U_it = _rescale_to_target_omega(
                        U_it,
                        omega_target,
                        coarse_problem["f_V"],
                        coarse_problem["q_mask"],
                    )
                    lambda_it = float(kwargs["lambda_value"])
                    tol_projected = 1.0e-2
                    tol_omega = 1.0e-2
                    gmres_total = 0
                    converged = False
                    reduced_iter = 0

                    for reduced_iter in range(1, 11):
                        coarse_problem["const_builder"].reduction(max(float(lambda_it), 1.0e-6))
                        K_tangent = None
                        K_free = None
                        try:
                            if hasattr(coarse_problem["const_builder"], "build_F_K_tangent_all_free") and callable(
                                coarse_problem["const_builder"].build_F_K_tangent_all_free
                            ):
                                F_free, K_free = coarse_problem["const_builder"].build_F_K_tangent_all_free(float(lambda_it), U_it)
                                F_free = np.asarray(F_free, dtype=np.float64).reshape(-1)
                                eps = max(float(kwargs.get("tol", 1.0e-4)) / 1000.0, 1.0e-12)
                                F_eps = np.asarray(
                                    coarse_problem["const_builder"].build_F_all_free(float(lambda_it) + eps, U_it),
                                    dtype=np.float64,
                                ).reshape(-1)
                            else:
                                F_full, K_tangent = coarse_problem["const_builder"].build_F_K_tangent_all(float(lambda_it), U_it)
                                F_free = np.asarray(flatten_field(F_full)[coarse_free_idx], dtype=np.float64)
                                K_free = extract_submatrix_free(K_tangent, coarse_free_idx)
                                eps = max(float(kwargs.get("tol", 1.0e-4)) / 1000.0, 1.0e-12)
                                F_eps_full = coarse_problem["const_builder"].build_F_all(float(lambda_it) + eps, U_it)
                                F_eps = np.asarray(flatten_field(F_eps_full)[coarse_free_idx], dtype=np.float64)
                            residual_free = coarse_f_free - F_free
                            U_free = np.asarray(flatten_field(U_it)[coarse_free_idx], dtype=np.float64)
                            omega_err = float(np.dot(coarse_f_free, U_free) - omega_target)
                            projected_res = np.asarray(basis_free.T @ residual_free, dtype=np.float64)
                            projected_rel = float(np.linalg.norm(projected_res) / coarse_norm_f)
                            omega_rel = abs(float(omega_err)) / max(abs(float(omega_target)), 1.0)
                            info["reduced_projected_residual"] = float(projected_rel)
                            info["reduced_omega_residual"] = float(omega_rel)
                            if projected_rel <= tol_projected and omega_rel <= tol_omega:
                                converged = True
                                break

                            basis_dim = int(basis_free.shape[1])
                            wtf = np.asarray(basis_free.T @ coarse_f_free, dtype=np.float64)
                            rhs = np.concatenate([projected_res, np.asarray([-omega_err], dtype=np.float64)])

                            def _reduced_matvec(x_vec: np.ndarray) -> np.ndarray:
                                x_arr = np.asarray(x_vec, dtype=np.float64).reshape(-1)
                                coeff = x_arr[:basis_dim]
                                d_lambda = float(x_arr[basis_dim])
                                basis_combination = np.asarray(basis_free @ coeff, dtype=np.float64)
                                k_times = np.asarray(matvec_to_numpy(K_free, basis_combination), dtype=np.float64).reshape(-1)
                                top = -(basis_free.T @ k_times) + wtf * d_lambda
                                bottom = np.asarray([float(np.dot(coarse_f_free, basis_combination))], dtype=np.float64)
                                return np.concatenate([top, bottom])

                            iteration_counter = {"value": 0}

                            def _gmres_callback(_unused) -> None:
                                iteration_counter["value"] += 1

                            linop = sparse_linalg.LinearOperator(
                                shape=(basis_dim + 1, basis_dim + 1),
                                matvec=_reduced_matvec,
                                dtype=np.float64,
                            )
                            delta_red, gmres_info = sparse_linalg.gmres(
                                linop,
                                rhs,
                                restart=basis_dim + 1,
                                rtol=1.0e-10,
                                atol=0.0,
                                maxiter=max(20, 2 * (basis_dim + 1)),
                                callback=_gmres_callback,
                                callback_type="pr_norm",
                            )
                            gmres_total += int(iteration_counter["value"])
                            if gmres_info not in (0, None):
                                raise RuntimeError(f"reduced_gmres_failed:{gmres_info}")

                            coeff = np.asarray(delta_red[:basis_dim], dtype=np.float64)
                            d_lambda = float(delta_red[basis_dim])
                            dU_free = np.asarray(basis_free @ coeff, dtype=np.float64)
                            dU_full = full_field_from_free_values(dU_free, coarse_free_idx, coarse_shape)
                            U_it = np.asarray(U_it, dtype=np.float64) + dU_full
                            lambda_it = float(lambda_it + d_lambda)
                        finally:
                            if K_free is not None and K_free is not K_tangent and not _is_builder_cached_matrix(K_free, coarse_problem["const_builder"]):
                                _destroy_petsc_mat(K_free)
                            if K_tangent is not None and not _is_builder_cached_matrix(K_tangent, coarse_problem["const_builder"]):
                                _destroy_petsc_mat(K_tangent)

                    info["reduced_newton_iterations"] = float(reduced_iter)
                    info["reduced_gmres_iterations"] = float(gmres_total)
                    if not converged:
                        info["fallback_used"] = True
                        info["fallback_error"] = "reduced_newton_not_converged"
                        U_pred, lambda_pred, _kind, coarse_info = _coarse_solution_predictor(**kwargs)
                        coarse_info.update(info)
                        coarse_info["fallback_used"] = True
                        coarse_info["fallback_error"] = "reduced_newton_not_converged"
                        payload = (U_pred, lambda_pred, "coarse_p1_reduced_newton_fallback_solution", coarse_info)
                    else:
                        U_pred = _prolongate_coarse_full_to_fine_full(U_it)
                        U_pred = _rescale_to_target_omega(
                            U_pred,
                            float(kwargs["omega_target"]),
                            np.asarray(kwargs["f"], dtype=np.float64),
                            np.asarray(kwargs["Q"], dtype=bool),
                        )
                        info["predictor_wall_time"] = float(perf_counter() - t0)
                        payload = (U_pred, float(kwargs["lambda_value"]), "coarse_p1_reduced_newton", info)
            except Exception as exc:
                info["fallback_used"] = True
                info["fallback_error"] = repr(exc)
                U_pred, lambda_pred, _kind, coarse_info = _coarse_solution_predictor(**kwargs)
                coarse_info.update(info)
                coarse_info["fallback_used"] = True
                coarse_info["fallback_error"] = repr(exc)
                payload = (U_pred, lambda_pred, "coarse_p1_reduced_newton_fallback_solution", coarse_info)
        U_pred, lambda_pred, kind, info = world_comm.bcast(payload, root=0)
        return np.asarray(U_pred, dtype=np.float64), float(lambda_pred), str(kind), dict(info)

    def _cleanup() -> None:
        if not is_coarse_root:
            return
        try:
            close_solver = getattr(coarse_solver, "close", None)
            if callable(close_solver):
                close_solver()
            else:
                release = getattr(coarse_solver, "release_iteration_resources", None)
                if callable(release):
                    release()
        except Exception:
            pass
        try:
            coarse_problem["const_builder"].release_petsc_caches()
        except Exception:
            pass
        try:
            _destroy_petsc_mat(coarse_problem["K_elast"])
        except Exception:
            pass

    return {
        "coarse_p1_solution": _coarse_solution_predictor,
        "coarse_p1_reduced_newton": _coarse_reduced_newton_predictor,
        "cleanup": _cleanup,
    }


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
    step_max: int = 100,
    it_newt_max: int = 200,
    it_damp_max: int = 10,
    tol: float = 1e-4,
    r_min: float = 1e-4,
    linear_tolerance: float = 1e-1,
    linear_max_iter: int = 100,
    solver_type: str = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
    factor_solver_type: str | None = None,
    pc_backend: str | None = "hypre",
    pmg_coarse_mesh_path: Path | None = None,
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
    predictor_context: dict[str, object] | None = None
    if analysis_key == "ssr":
        predictor_mode = str(continuation_predictor).strip().lower()
        if predictor_mode in {"coarse_p1_solution", "coarse_p1_reduced_newton"}:
            if effective_pc_backend != "pmg_shell" or str(elem_type).upper() != "P4" or pmg_coarse_mesh_path is not None:
                raise ValueError(
                    f"{predictor_mode} currently requires same-mesh P4(L1) with pc_backend='pmg_shell'."
                )
            predictor_context = _build_p4_l1_coarse_predictor_context(
                mesh_path=mesh_path,
                mesh_boundary_type=int(mesh_boundary_type),
                node_ordering=str(node_ordering),
                reorder_parts=partition_count,
                material_rows=np.asarray(mat_props, dtype=np.float64).tolist(),
                davis_type=str(davis_type),
                constitutive_mode=str(constitutive_mode),
                tangent_kernel=str(tangent_kernel),
                solver_type=str(solver_type),
                linear_tolerance=float(linear_tolerance),
                linear_max_iter=int(linear_max_iter),
                lambda_init=float(lambda_init),
                d_lambda_init=float(d_lambda_init),
                d_lambda_min=float(d_lambda_min),
                it_newt_max=int(it_newt_max),
                it_damp_max=int(it_damp_max),
                tol=float(tol),
                r_min=float(r_min),
                fine_q_mask=q_mask,
                fine_f_V=f_V,
                fine_constitutive_matrix_builder=constitutive_matrix_builder,
                pmg_hierarchy=pmg_hierarchy,
                preconditioner_options=preconditioner_options,
            )

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
        "step_max": int(step_max),
        "it_newt_max": int(it_newt_max),
        "it_damp_max": int(it_damp_max),
        "tol": float(tol),
        "r_min": float(r_min),
        "factor_solver_type": factor_solver_type,
        "pc_backend": effective_pc_backend,
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
            predictor_context=predictor_context,
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
        "step_predictor_fallback_count": int(np.count_nonzero(step_predictor_fallback_used)),
        "step_lambda_initial_guess_abs_error_total": _nan_sum(step_lambda_guess_abs_error),
        "step_lambda_initial_guess_abs_error_last": _nan_last(step_lambda_guess_abs_error),
    }

    step_u = np.asarray(stats.pop("step_U"), dtype=np.float64) if isinstance(stats.get("step_U", None), list) else np.empty((0, 3, 0), dtype=np.float64)
    retrospective_predictor_arrays, retrospective_predictor_summary = _compute_retrospective_predictor_export(
        step_u=step_u,
        omega_hist=np.asarray(omega_hist3, dtype=np.float64),
        lambda_hist=np.asarray(lambda_hist3, dtype=np.float64),
        step_index=np.asarray(stats.get("step_index", []), dtype=np.int64),
        step_omega=np.asarray(stats.get("step_omega", []), dtype=np.float64),
        step_lambda=np.asarray(stats.get("step_lambda", []), dtype=np.float64),
    )

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
        "retrospective_predictor_export": retrospective_predictor_summary,
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
            **retrospective_predictor_arrays,
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

    cleanup_predictor = None if predictor_context is None else predictor_context.get("cleanup")
    if callable(cleanup_predictor):
        try:
            cleanup_predictor()
        except Exception:
            pass

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
            "two_step",
            "reduced_all_prev",
            "reduced_newton_all_prev",
            "reduced_newton_affine_all_prev",
            "reduced_newton_window",
            "reduced_newton_increment_power",
            "secant_energy_alpha",
            "three_param_penalty",
            "coarse_p1_solution",
            "coarse_p1_reduced_newton",
        ],
    )
    parser.add_argument("--continuation_predictor_switch_ordinal", type=int, default=None)
    parser.add_argument(
        "--continuation_predictor_switch_to",
        type=str,
        default=None,
        choices=[
            "secant",
            "two_step",
            "reduced_all_prev",
            "reduced_newton_all_prev",
            "reduced_newton_affine_all_prev",
            "reduced_newton_window",
            "reduced_newton_increment_power",
            "secant_energy_alpha",
            "three_param_penalty",
            "coarse_p1_solution",
            "coarse_p1_reduced_newton",
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
    parser.add_argument("--it_newt_max", type=int, default=200)
    parser.add_argument("--it_damp_max", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--r_min", type=float, default=1e-4)
    parser.add_argument("--linear_tolerance", type=float, default=1e-1)
    parser.add_argument("--linear_max_iter", type=int, default=100)
    parser.add_argument("--solver_type", type=str, default="PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE")
    parser.add_argument("--factor_solver_type", type=str, default=None)
    parser.add_argument("--pc_backend", type=str, default="hypre", choices=["hypre", "gamg", "bddc", "pmg", "pmg_shell"])
    parser.add_argument("--pmg_coarse_mesh_path", type=Path, default=None)
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
        it_newt_max=args.it_newt_max,
        it_damp_max=args.it_damp_max,
        tol=args.tol,
        r_min=args.r_min,
        linear_tolerance=args.linear_tolerance,
        linear_max_iter=args.linear_max_iter,
        solver_type=args.solver_type,
        factor_solver_type=args.factor_solver_type,
        pc_backend=args.pc_backend,
        pmg_coarse_mesh_path=args.pmg_coarse_mesh_path,
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
