#!/usr/bin/env python
"""Run 3D seepage-coupled SSR continuation on supported mesh families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from slope_stability.cli.progress import make_progress_logger
from slope_stability.constitutive import ConstitutiveOperator
from slope_stability.continuation import SSR_indirect_continuation
from slope_stability.core.elements import validate_supported_elem_type
from slope_stability.fem import (
    assemble_owned_elastic_rows_for_comm,
    assemble_strain_operator,
    prepare_owned_tangent_pattern,
    quadrature_volume_3d,
    vector_volume,
)
from slope_stability.linear import SolverFactory
from slope_stability.linear.pmg import (
    build_3d_mixed_pmg_hierarchy,
    build_3d_same_mesh_pmg_hierarchy,
    build_3d_same_mesh_scalar_pmg_hierarchy,
    validate_pmg_fine_level_alignment,
)
from slope_stability.mesh import MaterialSpec, heterogenous_materials, reorder_mesh_nodes
from slope_stability.problem_asset_runtime import build_mesh_for_path
from slope_stability.problem_assets import (
    build_seepage_boundary_for_path,
    load_hydraulic_conductivity_for_path,
    load_material_rows_for_path,
    load_water_unit_weight_for_path,
)
from slope_stability.seepage import heter_conduct, seepage_problem_3d
from slope_stability.utils import local_csr_to_petsc_aij_matrix, owned_block_range, release_petsc_aij_matrix


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_progress_logger(progress_dir: Path):
    return make_progress_logger(progress_dir)


def _stage_log(rank: int, label: str, t0: float) -> None:
    if rank == 0:
        print(f"[stage] {label} | t={perf_counter() - t0:.1f}s", flush=True)


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


def _release_rank_local_resources(*, solvers: tuple[object, ...], const_builder, mats: tuple[object, ...]) -> None:
    for solver in solvers:
        if solver is None:
            continue
        try:
            close_solver = getattr(solver, "close", None)
            if callable(close_solver):
                close_solver()
            else:
                solver.release_iteration_resources()
        except Exception:
            pass
    if const_builder is not None:
        try:
            const_builder.release_petsc_caches()
        except Exception:
            pass
    if PETSc is not None:
        for mat in mats:
            if isinstance(mat, PETSc.Mat):
                try:
                    release_petsc_aij_matrix(mat)
                    mat.destroy()
                except Exception:
                    pass
        cleanup = getattr(PETSc, "garbage_cleanup", None)
        if callable(cleanup):
            try:
                cleanup(PETSc.COMM_WORLD)
            except TypeError:
                cleanup()
            except Exception:
                pass


def _resolve_boundary_mode(mesh_path: Path, boundary_mode: str) -> str:
    raw = str(boundary_mode).strip().lower()
    if raw and raw != "auto":
        return raw
    return "canonical"


def _load_labeled_mesh(
    mesh_path: Path,
    *,
    elem_type: str,
    profile: str | None,
    node_ordering: str,
    partition_count: int | None,
):
    mesh = build_mesh_for_path(mesh_path, elem_type=elem_type, profile=profile)
    reordered = reorder_mesh_nodes(
        mesh.coord,
        mesh.elem,
        mesh.surf,
        mesh.q_mask,
        strategy=node_ordering,
        n_parts=partition_count,
    )
    return {
        "coord": np.asarray(reordered.coord, dtype=np.float64),
        "elem": np.asarray(reordered.elem, dtype=np.int64),
        "surf": np.asarray(reordered.surf, dtype=np.int64),
        "q_mask": np.asarray(reordered.q_mask, dtype=bool),
        "material_identifier": np.asarray(mesh.material_id, dtype=np.int64).ravel(),
        "triangle_labels": np.asarray(mesh.boundary_labels, dtype=np.int64).ravel(),
        "mesh_boundary_type": 0,
    }


def run_capture(
    output_dir: Path,
    *,
    mesh_path: Path | None = None,
    profile: str | None = None,
    boundary_mode: str = "auto",
    elem_type: str = "P2",
    node_ordering: str = "block_metis",
    lambda_init: float = 1.0,
    d_lambda_init: float = 0.1,
    d_lambda_min: float = 1e-5,
    d_lambda_diff_scaled_min: float = 5e-3,
    omega_max_stop: float = 7.0e7,
    continuation_predictor: str = "secant",
    omega_step_controller: str = "legacy",
    step_max: int = 100,
    it_newt_max: int = 50,
    it_damp_max: int = 10,
    tol: float = 1e-4,
    r_min: float = 1e-4,
    newton_stopping_criterion: str = "relative_residual",
    newton_stopping_tol: float | None = None,
    linear_tolerance: float = 1e-1,
    linear_max_iter: int = 100,
    preconditioner_threads: int = 16,
    solver_type: str = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
    pc_backend: str | None = "hypre",
    pmg_coarse_mesh_path: Path | None = None,
    pmg_fine_hierarchy_mode: str = "default",
    mpi_distribute_by_nodes: bool = True,
    pc_hypre_coarsen_type: str = "HMIS",
    pc_hypre_interp_type: str = "ext+i",
    pc_hypre_strong_threshold: float | None = None,
    pc_hypre_boomeramg_max_iter: int | None = 1,
    pc_hypre_P_max: int | None = None,
    pc_hypre_agg_nl: int | None = None,
    pc_hypre_nongalerkin_tol: float | None = None,
    recycle_preconditioner: bool = True,
    constitutive_mode: str = "overlap",
    tangent_kernel: str = "rows",
    petsc_opt: list[str] | None = None,
    seepage_linear_tolerance: float = 1e-10,
    seepage_linear_max_iter: int = 500,
    seepage_pc_backend: str = "hypre",
    seepage_max_deflation_basis_vectors: int = 48,
    max_deflation_basis_vectors: int = 48,
    water_unit_weight: float | None = None,
    conductivity: list[float] | np.ndarray | None = None,
) -> dict:
    rank = int(PETSc.COMM_WORLD.getRank())
    out_dir = _ensure_dir(output_dir) if rank == 0 else output_dir
    data_dir = out_dir / "data"
    progress_callback = None
    if rank == 0:
        _ensure_dir(data_dir)
        progress_callback = _make_progress_logger(data_dir)

    if mesh_path is None:
        mesh_path = Path(__file__).resolve().parents[3] / "meshes" / "3d_hetero_seepage" / "concave_family_b.msh"
    mesh_path = Path(mesh_path)
    run_t0 = perf_counter()
    _stage_log(rank, "start", run_t0)
    elem_type = validate_supported_elem_type(3, elem_type)
    if elem_type not in {"P2", "P4"}:
        raise NotImplementedError(f"3D seepage+SSR study runner currently supports only 'P2' and 'P4', got {elem_type!r}.")

    solver_type_upper = str(solver_type).upper()
    effective_pc_backend = None if pc_backend is None else str(pc_backend).strip().lower()
    if effective_pc_backend is None:
        if "HYPRE" in solver_type_upper:
            effective_pc_backend = "hypre"
        elif "GAMG" in solver_type_upper:
            effective_pc_backend = "gamg"

    if effective_pc_backend in {"pmg", "pmg_shell"}:
        if not bool(mpi_distribute_by_nodes):
            raise ValueError(f"{effective_pc_backend} backend requires mpi_distribute_by_nodes=true.")
        if "PETSC_MATLAB_DFGMRES" not in solver_type_upper and not solver_type_upper.startswith("KSPFGMRES"):
            raise ValueError(
                f"{effective_pc_backend} backend is currently supported only with PETSC_MATLAB_DFGMRES* or KSPFGMRES* solver types."
            )

    boundary_mode_name = _resolve_boundary_mode(mesh_path, boundary_mode)
    _stage_log(rank, "resolved_boundary_mode", run_t0)
    material_rows = load_material_rows_for_path(mesh_path)
    if material_rows is None:
        raise ValueError(f"No material rows found in mesh-family definition for {mesh_path}.")
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
    labeled = _load_labeled_mesh(
        mesh_path,
        elem_type=elem_type,
        profile=profile,
        node_ordering=node_ordering,
        partition_count=partition_count,
    )
    _stage_log(rank, "loaded_labeled_mesh", run_t0)
    coord = labeled["coord"]
    elem = labeled["elem"]
    surf = labeled["surf"]
    q_mask = labeled["q_mask"]
    material_identifier = labeled["material_identifier"]
    triangle_labels = labeled["triangle_labels"]
    mesh_boundary_type = int(labeled["mesh_boundary_type"])

    pmg_hierarchy = None
    if effective_pc_backend in {"pmg", "pmg_shell"}:
        if pmg_coarse_mesh_path is None:
            pmg_hierarchy = build_3d_same_mesh_pmg_hierarchy(
                mesh_path,
                fine_elem_type=elem_type,
                profile=profile,
                boundary_type=mesh_boundary_type,
                node_ordering=node_ordering,
                reorder_parts=partition_count,
                material_rows=mat_props.tolist(),
                comm=PETSc.COMM_WORLD,
            )
        else:
            pmg_hierarchy = build_3d_mixed_pmg_hierarchy(
                mesh_path,
                pmg_coarse_mesh_path,
                fine_elem_type=elem_type,
                profile=profile,
                boundary_type=mesh_boundary_type,
                node_ordering=node_ordering,
                reorder_parts=partition_count,
                material_rows=mat_props.tolist(),
                comm=PETSc.COMM_WORLD,
            )

        fine_level = pmg_hierarchy.fine_level
        if (
            fine_level.coord.shape != coord.shape
            or fine_level.elem.shape != elem.shape
            or fine_level.surf.shape != surf.shape
            or fine_level.q_mask.shape != q_mask.shape
            or not np.allclose(fine_level.coord, coord)
            or not np.array_equal(fine_level.elem, elem)
            or not np.array_equal(fine_level.surf, surf)
            or not np.array_equal(fine_level.q_mask, q_mask)
        ):
            raise ValueError(
                "PMG fine-level hierarchy does not match the reordered seepage mesh. "
                "Use the same mesh family and node ordering for the hierarchy and seepage run."
            )
        _stage_log(rank, "built_pmg_hierarchy", run_t0)

    n_q = int(quadrature_volume_3d(elem_type)[0].shape[1])
    resolved_grho = load_water_unit_weight_for_path(mesh_path, required=True)
    grho = float(resolved_grho if water_unit_weight is None else water_unit_weight)
    conductivity_values = np.asarray(
        load_hydraulic_conductivity_for_path(mesh_path, required=True) if conductivity is None else conductivity,
        dtype=np.float64,
    ).ravel()
    required_conductivity_count = int(material_identifier.max()) + 1 if material_identifier.size else 1
    if conductivity_values.size == 1 and required_conductivity_count > 1:
        conductivity_values = np.repeat(conductivity_values, required_conductivity_count)
    if conductivity_values.size < required_conductivity_count:
        raise ValueError(
            f"Conductivity vector has {conductivity_values.size} entries, "
            f"but seepage material ids require {required_conductivity_count}."
        )
    conduct0 = heter_conduct(material_identifier, n_q, conductivity_values)
    q_w, pw_d = build_seepage_boundary_for_path(mesh_path, coord, surf, triangle_labels, grho=grho)

    seepage_solver = None
    if rank == 0:
        seepage_q_mask = np.asarray(q_w, dtype=bool).reshape(1, -1)
        seepage_preconditioner_options = {
            "threads": int(preconditioner_threads),
            "print_level": 0,
            "use_as_preconditioner": True,
            "pc_hypre_boomeramg_coarsen_type": pc_hypre_coarsen_type,
            "pc_hypre_boomeramg_interp_type": pc_hypre_interp_type,
            "max_deflation_basis_vectors": int(seepage_max_deflation_basis_vectors),
            "null_space": np.empty((0, 0), dtype=np.float64),
        }
        seepage_pc_backend_norm = str(seepage_pc_backend or "hypre").strip().lower()
        if seepage_pc_backend_norm in {"pmg", "pmg_shell"}:
            def _seepage_q_mask_builder(level_coord, level_surf, level_triangle_labels):
                if level_triangle_labels is None:
                    raise ValueError("Seepage PMG hierarchy requires triangle labels for boundary detection.")
                level_q_w, _ = build_seepage_boundary_for_path(
                    mesh_path,
                    level_coord,
                    level_surf,
                    level_triangle_labels,
                    grho=grho,
                )
                return np.asarray(level_q_w, dtype=bool).reshape(1, -1)

            seepage_pmg_hierarchy = build_3d_same_mesh_scalar_pmg_hierarchy(
                mesh_path,
                fine_elem_type=elem_type,
                profile=profile,
                boundary_type=mesh_boundary_type,
                node_ordering=node_ordering,
                reorder_parts=partition_count,
                material_rows=mat_props.tolist(),
                q_mask_builder=_seepage_q_mask_builder,
                comm=PETSc.COMM_SELF,
            )
            validate_pmg_fine_level_alignment(
                seepage_pmg_hierarchy,
                coord=coord,
                elem=elem,
                surf=surf,
                q_mask=seepage_q_mask,
                comm=PETSc.COMM_SELF,
            )
            seepage_preconditioner_options.update(
                {
                    "pc_backend": seepage_pc_backend_norm,
                    "pmg_hierarchy": seepage_pmg_hierarchy,
                }
            )
        seepage_solver = SolverFactory.create(
            "PETSC_MATLAB_DFGMRES_HYPRE",
            tolerance=seepage_linear_tolerance,
            max_iterations=seepage_linear_max_iter,
            deflation_basis_tolerance=1e-3,
            verbose=False,
            q_mask=seepage_q_mask,
            coord=coord,
            preconditioner_options=seepage_preconditioner_options,
        )
        if hasattr(seepage_solver, "enable_diagnostics"):
            seepage_solver.enable_diagnostics(True)
        pw, grad_p, mater_sat, seep_history, _seep_assembly = seepage_problem_3d(
            coord,
            elem,
            q_w,
            pw_d,
            grho,
            conduct0,
            elem_type=elem_type,
            linear_system_solver=seepage_solver,
            it_max=50,
            tol=1e-10,
        )
        seepage_payload = {
            "pw": np.asarray(pw, dtype=np.float64),
            "grad_p": np.asarray(grad_p, dtype=np.float64),
            "mater_sat": np.asarray(mater_sat, dtype=bool),
            "history": dict(seep_history),
        }
    else:
        seepage_payload = None
    seepage_payload = PETSc.COMM_WORLD.tompi4py().bcast(seepage_payload, root=0)
    pw = np.asarray(seepage_payload["pw"], dtype=np.float64)
    grad_p = np.asarray(seepage_payload["grad_p"], dtype=np.float64)
    mater_sat = np.asarray(seepage_payload["mater_sat"], dtype=bool)
    seep_history = dict(seepage_payload["history"])
    _stage_log(rank, "solved_seepage", run_t0)
    saturation = np.repeat(np.asarray(mater_sat, dtype=bool), n_q)

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
    _stage_log(rank, "assembled_elastic_rows", run_t0)
    global_size = int(coord.shape[0] * coord.shape[1])
    K_elast = local_csr_to_petsc_aij_matrix(
        elastic_rows.local_matrix,
        global_shape=(global_size, global_size),
        comm=PETSc.COMM_WORLD,
        block_size=coord.shape[0],
    )

    mech_assembly = assemble_strain_operator(coord, elem, elem_type, dim=3)
    if boundary_mode_name == "comsol":
        f_v_int = np.vstack(
            (
                np.zeros(mech_assembly.n_int, dtype=np.float64),
                -gamma.astype(np.float64),
                np.zeros(mech_assembly.n_int, dtype=np.float64),
            )
        )
    else:
        f_v_int = np.vstack(
            (
                -np.asarray(grad_p[0, :], dtype=np.float64),
                -np.asarray(grad_p[1, :], dtype=np.float64) - gamma.astype(np.float64),
                -np.asarray(grad_p[2, :], dtype=np.float64),
            )
        )
    f_V = vector_volume(mech_assembly, f_v_int)

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
    _stage_log(rank, "prepared_tangent_pattern", run_t0)
    const_builder.set_owned_tangent_pattern(
        tangent_pattern,
        use_compiled=True,
        tangent_kernel=tangent_kernel,
        constitutive_mode=constitutive_mode,
        use_compiled_constitutive=True,
    )

    preconditioner_options = {
        "threads": int(preconditioner_threads),
        "print_level": 0,
        "use_as_preconditioner": True,
        "pc_backend": effective_pc_backend,
        "pmg_coarse_mesh_path": None if pmg_coarse_mesh_path is None else str(pmg_coarse_mesh_path),
        "pmg_fine_hierarchy_mode": str(pmg_fine_hierarchy_mode),
        "preconditioner_matrix_source": "tangent",
        "preconditioner_matrix_policy": "current",
        "preconditioner_rebuild_policy": "every_newton",
        "preconditioner_rebuild_interval": 1,
        "mpi_distribute_by_nodes": bool(mpi_distribute_by_nodes),
        "use_coordinates": True,
        "max_deflation_basis_vectors": int(max_deflation_basis_vectors),
    }
    if recycle_preconditioner:
        preconditioner_options["recycle_preconditioner"] = True
    robust_parallel_shell = (
        pmg_hierarchy is not None
        and str(elem_type).upper() == "P4"
        and tuple(int(getattr(level, "order", -1)) for level in getattr(pmg_hierarchy, "levels", ())) in {(1, 1, 2), (1, 2, 4)}
        and int(PETSc.COMM_WORLD.getSize()) > 1
    )
    if effective_pc_backend == "pmg_shell":
        preconditioner_options.update(
            {
                "full_system_preconditioner": False,
                "mg_levels_ksp_type": "chebyshev" if robust_parallel_shell else "richardson",
                "mg_levels_ksp_max_it": 3,
                "mg_levels_pc_type": "jacobi" if robust_parallel_shell else "sor",
                "mg_coarse_ksp_type": "preonly",
                "mg_coarse_pc_type": "hypre",
                "mg_coarse_pc_hypre_type": "boomeramg",
                "pmg_hierarchy": pmg_hierarchy,
            }
        )
    if effective_pc_backend == "hypre":
        preconditioner_options["pc_hypre_boomeramg_coarsen_type"] = str(pc_hypre_coarsen_type)
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
    _stage_log(rank, "created_outer_solver", run_t0)

    params = {
        "lambda_init": float(lambda_init),
        "d_lambda_init": float(d_lambda_init),
        "d_lambda_min": float(d_lambda_min),
        "d_lambda_diff_scaled_min": float(d_lambda_diff_scaled_min),
        "omega_max_stop": float(omega_max_stop),
        "continuation_predictor": str(continuation_predictor),
        "omega_step_controller": str(omega_step_controller),
        "step_max": int(step_max),
        "it_newt_max": int(it_newt_max),
        "it_damp_max": int(it_damp_max),
        "tol": float(tol),
        "r_min": float(r_min),
        "newton_stopping_criterion": str(newton_stopping_criterion),
        "newton_stopping_tol": None if newton_stopping_tol is None else float(newton_stopping_tol),
        "elem_type": elem_type,
        "davis_type": "B",
        "material_rows": mat_props.tolist(),
        "node_ordering": node_ordering,
        "mesh_boundary_type": mesh_boundary_type,
        "mpi_distribute_by_nodes": bool(mpi_distribute_by_nodes),
        "pc_backend": effective_pc_backend,
        "pmg_coarse_mesh_path": None if pmg_coarse_mesh_path is None else str(pmg_coarse_mesh_path),
        "pmg_fine_hierarchy_mode": str(pmg_fine_hierarchy_mode),
        "pc_hypre_coarsen_type": pc_hypre_coarsen_type,
        "pc_hypre_interp_type": pc_hypre_interp_type,
        "pc_hypre_strong_threshold": pc_hypre_strong_threshold,
        "pc_hypre_boomeramg_max_iter": pc_hypre_boomeramg_max_iter,
        "pc_hypre_P_max": pc_hypre_P_max,
        "pc_hypre_agg_nl": pc_hypre_agg_nl,
        "pc_hypre_nongalerkin_tol": pc_hypre_nongalerkin_tol,
        "recycle_preconditioner": bool(recycle_preconditioner),
        "constitutive_mode": str(constitutive_mode),
        "tangent_kernel": str(tangent_kernel),
        "mesh_file": str(mesh_path),
        "boundary_mode": boundary_mode_name,
        "seepage_linear_tolerance": float(seepage_linear_tolerance),
        "seepage_linear_max_iter": int(seepage_linear_max_iter),
        "seepage_pc_backend": str(seepage_pc_backend),
        "water_unit_weight": float(grho),
        "conductivity": conductivity_values.tolist(),
        "petsc_opt": list(petsc_opt or []),
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
        linear_system_solver,
        progress_callback=progress_callback,
        continuation_predictor=str(continuation_predictor),
        omega_step_controller=str(omega_step_controller),
        newton_stopping_criterion=str(newton_stopping_criterion),
        newton_stopping_tol=newton_stopping_tol,
    )
    _stage_log(rank, "finished_continuation", run_t0)
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

    step_u = (
        np.asarray(stats.pop("step_U"), dtype=np.float64)
        if isinstance(stats.get("step_U", None), list)
        else np.empty((0, 3, 0), dtype=np.float64)
    )
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
            "stop_reason": str(stats.get("stop_reason", "")),
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
            "seepage_runtime": float(np.sum(np.asarray(seep_history.get("linear_solve_time", []), dtype=np.float64))),
        },
        "seepage": {
            "criterion": [float(x) for x in seep_history.get("criterion", [])],
            "iterations": int(seep_history.get("iterations", 0)),
            "converged": bool(seep_history.get("converged", False)),
            "boundary_mode": boundary_mode_name,
        },
    }

    _release_rank_local_resources(
        solvers=(linear_system_solver, seepage_solver),
        const_builder=const_builder,
        mats=(K_elast,),
    )
    mpi_comm.Barrier()

    if rank == 0:
        np.savez_compressed(
            data_dir / "petsc_run.npz",
            U=U,
            lambda_hist=lambda_hist,
            omega_hist=omega_hist,
            Umax_hist=Umax_hist,
            step_U=step_u,
            seepage_pw=pw,
            seepage_grad_p=grad_p,
            seepage_mater_sat=mater_sat,
            **{"stats_" + key: _stats_value_to_npz(value) for key, value in stats.items() if key != "step_U"},
        )
        (data_dir / "run_info.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

    mpi_comm.Barrier()

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
    parser = argparse.ArgumentParser(description="Run a 3D seepage SSR capture.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--mesh_path", type=Path, default=None)
    parser.add_argument("--boundary_mode", type=str, default="auto", choices=["auto", "waterlevels", "comsol"])
    parser.add_argument("--elem_type", type=str, default="P2", choices=["P1", "P2", "P4"])
    parser.add_argument(
        "--node_ordering",
        type=str,
        default="block_metis",
        choices=["original", "xyz", "block_xyz", "morton", "rcm", "block_rcm", "block_metis"],
    )
    parser.add_argument("--lambda_init", type=float, default=1.0)
    parser.add_argument("--d_lambda_init", type=float, default=0.1)
    parser.add_argument("--d_lambda_min", type=float, default=1e-5)
    parser.add_argument("--d_lambda_diff_scaled_min", type=float, default=5e-3)
    parser.add_argument("--omega_max_stop", type=float, default=7e7)
    parser.add_argument("--continuation_predictor", type=str, default="secant")
    parser.add_argument("--omega_step_controller", type=str, default="legacy")
    parser.add_argument("--step_max", type=int, default=100)
    parser.add_argument("--it_newt_max", type=int, default=50)
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
    parser.add_argument("--linear_tolerance", type=float, default=1e-1)
    parser.add_argument("--linear_max_iter", type=int, default=100)
    parser.add_argument("--preconditioner_threads", type=int, default=16)
    parser.add_argument("--solver_type", type=str, default="PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE")
    parser.add_argument("--pc_backend", type=str, default="hypre", choices=["hypre", "gamg", "pmg", "pmg_shell"])
    parser.add_argument("--pmg_coarse_mesh_path", type=Path, default=None)
    parser.add_argument("--pmg_fine_hierarchy_mode", type=str, default="default")
    parser.add_argument("--mpi_distribute_by_nodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pc_hypre_coarsen_type", type=str, default="HMIS")
    parser.add_argument("--pc_hypre_interp_type", type=str, default="ext+i")
    parser.add_argument("--pc_hypre_strong_threshold", type=float, default=None)
    parser.add_argument("--pc_hypre_boomeramg_max_iter", type=int, default=1)
    parser.add_argument("--pc_hypre_P_max", type=int, default=None)
    parser.add_argument("--pc_hypre_agg_nl", type=int, default=None)
    parser.add_argument("--pc_hypre_nongalerkin_tol", type=float, default=None)
    parser.add_argument("--recycle_preconditioner", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--constitutive_mode",
        type=str,
        default="overlap",
        choices=["global", "overlap", "unique_gather", "unique_exchange"],
    )
    parser.add_argument("--tangent_kernel", type=str, default="rows", choices=["legacy", "rows"])
    parser.add_argument("--petsc-opt", action="append", default=[], dest="petsc_opt")
    parser.add_argument("--seepage_linear_tolerance", type=float, default=1e-10)
    parser.add_argument("--seepage_linear_max_iter", type=int, default=500)
    parser.add_argument("--seepage_pc_backend", type=str, default="hypre")
    parser.add_argument("--seepage_max_deflation_basis_vectors", type=int, default=48)
    parser.add_argument("--max_deflation_basis_vectors", type=int, default=48)
    parser.add_argument("--water_unit_weight", type=float, default=None)
    parser.add_argument("--conductivity", type=float, action="append", default=None)
    args = parser.parse_args()

    result = run_capture(
        args.out_dir,
        mesh_path=args.mesh_path,
        boundary_mode=args.boundary_mode,
        elem_type=args.elem_type,
        node_ordering=args.node_ordering,
        lambda_init=args.lambda_init,
        d_lambda_init=args.d_lambda_init,
        d_lambda_min=args.d_lambda_min,
        d_lambda_diff_scaled_min=args.d_lambda_diff_scaled_min,
        omega_max_stop=args.omega_max_stop,
        continuation_predictor=args.continuation_predictor,
        omega_step_controller=args.omega_step_controller,
        step_max=args.step_max,
        it_newt_max=args.it_newt_max,
        it_damp_max=args.it_damp_max,
        tol=args.tol,
        r_min=args.r_min,
        newton_stopping_criterion=args.newton_stopping_criterion,
        newton_stopping_tol=args.newton_stopping_tol,
        linear_tolerance=args.linear_tolerance,
        linear_max_iter=args.linear_max_iter,
        preconditioner_threads=args.preconditioner_threads,
        solver_type=args.solver_type,
        pc_backend=args.pc_backend,
        pmg_coarse_mesh_path=args.pmg_coarse_mesh_path,
        pmg_fine_hierarchy_mode=args.pmg_fine_hierarchy_mode,
        mpi_distribute_by_nodes=args.mpi_distribute_by_nodes,
        pc_hypre_coarsen_type=args.pc_hypre_coarsen_type,
        pc_hypre_interp_type=args.pc_hypre_interp_type,
        pc_hypre_strong_threshold=args.pc_hypre_strong_threshold,
        pc_hypre_boomeramg_max_iter=args.pc_hypre_boomeramg_max_iter,
        pc_hypre_P_max=args.pc_hypre_P_max,
        pc_hypre_agg_nl=args.pc_hypre_agg_nl,
        pc_hypre_nongalerkin_tol=args.pc_hypre_nongalerkin_tol,
        recycle_preconditioner=args.recycle_preconditioner,
        constitutive_mode=args.constitutive_mode,
        tangent_kernel=args.tangent_kernel,
        petsc_opt=args.petsc_opt,
        seepage_linear_tolerance=args.seepage_linear_tolerance,
        seepage_linear_max_iter=args.seepage_linear_max_iter,
        seepage_pc_backend=args.seepage_pc_backend,
        seepage_max_deflation_basis_vectors=args.seepage_max_deflation_basis_vectors,
        max_deflation_basis_vectors=args.max_deflation_basis_vectors,
        water_unit_weight=args.water_unit_weight,
        conductivity=args.conductivity,
    )
    if PETSc.COMM_WORLD.getRank() == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
