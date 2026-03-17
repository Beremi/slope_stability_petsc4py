from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from slope_stability.cli.run_3D_hetero_SSR_capture import load_material_rows_for_path
from slope_stability.constitutive.problem import ConstitutiveOperator
from slope_stability.core.elements import validate_supported_elem_type
from slope_stability.fem.assembly import assemble_strain_operator, build_elastic_stiffness_matrix
from slope_stability.fem.distributed_elastic import assemble_owned_elastic_rows_for_comm
from slope_stability.fem.distributed_tangent import prepare_bddc_subdomain_pattern
from slope_stability.mesh import load_mesh_from_file
from slope_stability.linear.solver import SolverFactory
from slope_stability.mesh.materials import MaterialSpec, heterogenous_materials
from slope_stability.mesh.reorder import reorder_mesh_nodes
from slope_stability.utils import (
    bddc_pc_coordinates_from_metadata,
    extract_submatrix_free,
    get_petsc_is_local_mat,
    get_petsc_matrix_metadata,
    global_array_to_petsc_vec,
    local_csr_to_petsc_aij_matrix,
    matvec_to_numpy,
    owned_block_range,
    petsc_vec_to_global_array,
    q_to_free_indices,
    release_petsc_aij_matrix,
)
from slope_stability.fem.quadrature import quadrature_volume_3d


ROOT = Path(__file__).resolve().parents[2]


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


def _native_ksp_norm_type(name: str):
    normalized = str(name).strip().lower()
    mapping = {
        "default": PETSc.KSP.NormType.DEFAULT,
        "preconditioned": PETSc.KSP.NormType.PRECONDITIONED,
        "unpreconditioned": PETSc.KSP.NormType.UNPRECONDITIONED,
        "natural": PETSc.KSP.NormType.NATURAL,
        "none": PETSc.KSP.NormType.NONE,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported native KSP norm type {name!r}")
    return mapping[normalized]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _progress_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _emit(event: str, **payload) -> None:
        row = {"event": str(event), **payload}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    return _emit


def _collector_snapshot(solver) -> dict[str, float]:
    collector = solver.iteration_collector
    return {
        "iterations": float(collector.get_total_iterations()),
        "solve_time": float(collector.get_total_solve_time()),
        "preconditioner_time": float(collector.get_total_preconditioner_time()),
        "orthogonalization_time": float(collector.get_total_orthogonalization_time()),
    }


def _collector_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: float(after[key] - before[key]) for key in before}


def _set_petsc_option(options, key: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        options[key] = "true" if value else "false"
    else:
        options[key] = value


def _is_global_petsc_option(key: str) -> bool:
    normalized = str(key).strip().lower()
    return normalized.startswith("log_") or normalized in {
        "options_left",
        "memory_view",
        "malloc_view",
        "help",
        "pc_view",
        "ksp_view",
        "ksp_converged_reason",
        "ksp_monitor_singular_value",
        "ksp_view_eigenvalues",
        "log_trace",
    }


def _configure_bddc_pc_from_metadata(pc: PETSc.PC, Pmat: PETSc.Mat, *, use_coordinates: bool = True) -> None:
    metadata = get_petsc_matrix_metadata(Pmat)
    coordinates = bddc_pc_coordinates_from_metadata(Pmat)
    if coordinates is not None and use_coordinates:
        pc.setCoordinates(np.asarray(coordinates, dtype=np.float64))
    field_is = metadata.get("bddc_field_is_local")
    if field_is:
        if not all(isinstance(v, PETSc.IS) for v in field_is):
            field_is = tuple(
                PETSc.IS().createGeneral(np.asarray(v, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
                for v in field_is
            )
        pc.setBDDCDofsSplittingLocal(field_is)
    dirichlet = metadata.get("bddc_dirichlet_local")
    if dirichlet is not None:
        if not isinstance(dirichlet, PETSc.IS):
            dirichlet = PETSc.IS().createGeneral(np.asarray(dirichlet, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
        pc.setBDDCDirichletBoundariesLocal(dirichlet)
    adjacency = metadata.get("bddc_local_adjacency")
    if adjacency is not None:
        pc.setBDDCLocalAdjacency(adjacency)
    primal_vertices = metadata.get("bddc_primal_vertices_local")
    if primal_vertices is not None:
        if not isinstance(primal_vertices, PETSc.IS):
            primal_vertices = PETSc.IS().createGeneral(
                np.asarray(primal_vertices, dtype=PETSc.IntType),
                comm=PETSc.COMM_SELF,
            )
        pc.setBDDCPrimalVerticesLocalIS(primal_vertices)


def _native_ksp_type(name: str):
    normalized = str(name).strip().lower()
    mapping = {
        "cg": PETSc.KSP.Type.CG,
        "fgmres": PETSc.KSP.Type.FGMRES,
        "gmres": PETSc.KSP.Type.GMRES,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported native KSP type {name!r}")
    return mapping[normalized]


def _build_native_petsc_ksp(args, *, operator_matrix: PETSc.Mat, preconditioning_matrix: PETSc.Mat | None) -> tuple[PETSc.KSP, float]:
    t0 = perf_counter()
    ksp = PETSc.KSP().create(comm=operator_matrix.getComm())
    prefix = f"probe_native_{int(ksp.handle)}_"
    ksp.setOptionsPrefix(prefix)
    pmat = operator_matrix if preconditioning_matrix is None else preconditioning_matrix
    ksp.setOperators(operator_matrix, pmat)
    ksp.setType(_native_ksp_type(args.native_ksp_type))
    ksp.setInitialGuessNonzero(False)
    ksp.setTolerances(rtol=float(args.linear_tolerance), atol=1e-30, max_it=int(args.linear_max_iter))
    ksp.setNormType(_native_ksp_norm_type(args.native_ksp_norm_type))
    pc = ksp.getPC()
    backend = str(args.pc_backend).lower()
    if backend == "bddc":
        if preconditioning_matrix is None:
            raise ValueError("native PETSc BDDC probe requires an explicit MATIS preconditioning matrix")
        pc.setType(PETSc.PC.Type.BDDC)
        _configure_bddc_pc_from_metadata(
            pc,
            preconditioning_matrix,
            use_coordinates=bool(getattr(args, "use_coordinates", True)),
        )
    elif backend == "hypre":
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType("boomeramg")
    elif backend == "gamg":
        pc.setType(PETSc.PC.Type.GAMG)
    else:
        pc.setType(PETSc.PC.Type.NONE)

    opts = PETSc.Options()
    for key in (
        "pc_bddc_symmetric",
        "pc_bddc_dirichlet_ksp_type",
        "pc_bddc_dirichlet_pc_type",
        "pc_bddc_neumann_ksp_type",
        "pc_bddc_neumann_pc_type",
        "pc_bddc_coarse_ksp_type",
        "pc_bddc_coarse_pc_type",
        "pc_bddc_dirichlet_approximate",
        "pc_bddc_neumann_approximate",
        "pc_bddc_monolithic",
        "pc_bddc_coarse_redundant_pc_type",
        "pc_bddc_switch_static",
        "pc_bddc_use_deluxe_scaling",
        "pc_bddc_use_vertices",
        "pc_bddc_use_edges",
        "pc_bddc_use_faces",
        "pc_bddc_use_change_of_basis",
        "pc_bddc_use_change_on_faces",
        "pc_bddc_check_level",
        "pc_hypre_coarsen_type",
        "pc_hypre_interp_type",
        "pc_hypre_boomeramg_max_iter",
    ):
        _set_petsc_option(opts, f"{prefix}{key}", getattr(args, key, None))
    for key, value in getattr(args, "petsc_opt_map", {}).items():
        target_key = str(key) if _is_global_petsc_option(str(key)) else f"{prefix}{key}"
        _set_petsc_option(opts, target_key, value)
    ksp.setFromOptions()
    ksp.setUp()
    return ksp, float(perf_counter() - t0)


def _native_petsc_ksp_solve_once(
    ksp: PETSc.KSP,
    A: PETSc.Mat,
    rhs_full: np.ndarray,
) -> tuple[np.ndarray, float, int, int, list[float], list[float]]:
    rhs = global_array_to_petsc_vec(
        np.asarray(rhs_full, dtype=np.float64),
        comm=A.getComm(),
        ownership_range=A.getOwnershipRange() if int(A.getComm().getSize()) > 1 else None,
        bsize=A.getBlockSize() or None,
    )
    x = A.createVecRight()
    x.set(0.0)
    residual_history: list[float] = []
    rhs_norm = float(np.linalg.norm(np.asarray(rhs_full, dtype=np.float64).reshape(-1)))
    rhs_norm = max(rhs_norm, 1.0)

    def _monitor(_ksp, _it, rnorm):
        residual_history.append(float(rnorm))

    ksp.setMonitor(_monitor)
    t0 = perf_counter()
    ksp.solve(rhs, x)
    elapsed = float(perf_counter() - t0)
    solution = petsc_vec_to_global_array(x)
    reason = int(ksp.getConvergedReason())
    iterations = int(ksp.getIterationNumber())
    if not residual_history:
        residual = np.asarray(matvec_to_numpy(A, solution), dtype=np.float64).reshape(-1) - np.asarray(rhs_full, dtype=np.float64).reshape(-1)
        residual_history = [float(np.linalg.norm(residual))]
    relative_residual_history = [float(v / rhs_norm) for v in residual_history]
    ksp.cancelMonitor()
    rhs.destroy()
    x.destroy()
    return solution, elapsed, iterations, reason, residual_history, relative_residual_history


def _build_problem(
    *,
    mesh_path: Path,
    elem_type: str,
    node_ordering: str,
    material_rows: list[list[float]] | None,
    adjacency_source: str,
    corner_only_primals: bool,
):
    if material_rows is None:
        material_rows = load_material_rows_for_path(mesh_path)
    if material_rows is None:
        raise ValueError(f"No material rows found for {mesh_path}")
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
    mesh = load_mesh_from_file(mesh_path, boundary_type=0, elem_type=elem_type)
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

    row0, row1 = owned_block_range(coord.shape[1], coord.shape[0], PETSc.COMM_WORLD)
    bddc_pattern = prepare_bddc_subdomain_pattern(
        coord,
        elem,
        q_mask,
        material_identifier,
        materials,
        (row0 // coord.shape[0], row1 // coord.shape[0]),
        elem_type=elem_type,
        adjacency_source=adjacency_source,
        corner_only_primals=corner_only_primals,
    )

    const_builder = ConstitutiveOperator(
        B=None,
        c0=c0,
        phi=phi,
        psi=psi,
        Davis_type="B",
        shear=shear,
        bulk=bulk,
        lame=lame,
        WEIGHT=np.zeros(n_int, dtype=np.float64),
        n_strain=6,
        n_int=n_int,
        dim=3,
        q_mask=q_mask,
    )
    const_builder.set_bddc_subdomain_pattern(bddc_pattern)
    return coord, q_mask, K_elast, f_V, const_builder, bddc_pattern


def _build_solver(args, *, q_mask: np.ndarray, coord: np.ndarray):
    options = {
        "threads": 16,
        "print_level": 0,
        "use_as_preconditioner": True,
        "pc_backend": args.pc_backend,
        "preconditioner_matrix_source": args.preconditioner_matrix_source,
        "preconditioner_matrix_policy": args.preconditioner_matrix_policy,
        "preconditioner_rebuild_policy": args.preconditioner_rebuild_policy,
        "preconditioner_rebuild_interval": int(args.preconditioner_rebuild_interval),
        "mpi_distribute_by_nodes": True,
        "use_coordinates": True,
        "max_deflation_basis_vectors": 16,
    }
    for key in (
        "pc_bddc_symmetric",
        "pc_bddc_dirichlet_ksp_type",
        "pc_bddc_dirichlet_pc_type",
        "pc_bddc_neumann_ksp_type",
        "pc_bddc_neumann_pc_type",
        "pc_bddc_coarse_ksp_type",
        "pc_bddc_coarse_pc_type",
        "pc_bddc_dirichlet_approximate",
        "pc_bddc_neumann_approximate",
        "pc_bddc_monolithic",
        "pc_bddc_coarse_redundant_pc_type",
        "pc_bddc_use_deluxe_scaling",
        "pc_bddc_use_vertices",
        "pc_bddc_use_edges",
        "pc_bddc_use_faces",
        "pc_bddc_use_change_of_basis",
        "pc_bddc_use_change_on_faces",
        "pc_bddc_check_level",
        "pc_hypre_coarsen_type",
        "pc_hypre_interp_type",
    ):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    options.update(getattr(args, "petsc_opt_map", {}))
    return SolverFactory.create(
        args.solver_type,
        tolerance=float(args.linear_tolerance),
        max_iterations=int(args.linear_max_iter),
        deflation_basis_tolerance=1e-3,
        verbose=False,
        q_mask=q_mask,
        coord=coord,
        preconditioner_options=options,
    )


def run_probe(args) -> dict[str, object]:
    rank = int(PETSc.COMM_WORLD.getRank())
    mpi_comm = PETSc.COMM_WORLD.tompi4py()
    out_dir = _ensure_dir(Path(args.out_dir)) if rank == 0 else Path(args.out_dir)
    data_dir = out_dir / "data"
    if rank == 0:
        _ensure_dir(data_dir)
        emit = _progress_writer(data_dir / "progress.jsonl")
    else:
        emit = None

    t0 = perf_counter()
    coord, q_mask, K_elast, f_V, const_builder, bddc_pattern = _build_problem(
        mesh_path=Path(args.mesh_path),
        elem_type=str(args.elem_type),
        node_ordering=str(args.node_ordering),
        material_rows=None,
        adjacency_source=str(args.adjacency_source),
        corner_only_primals=bool(args.corner_only_primals),
    )
    free_idx = q_to_free_indices(q_mask)
    f_full = np.asarray(f_V, dtype=np.float64).reshape(-1, order="F")
    f_free = f_full[free_idx]
    if rank == 0 and emit is not None:
        emit(
            "elastic_problem_built",
            total_wall_time=float(perf_counter() - t0),
            mode=str(args.mode),
            outer_solver_family=str(args.outer_solver_family),
        )

    Pmat = None
    if str(args.pc_backend).lower() == "bddc":
        Pmat = const_builder.build_bddc_elastic_matrix(local_mat_type=str(args.bddc_local_mat_type))
        if rank == 0 and emit is not None:
            emit(
                "elastic_pmat_built",
                total_wall_time=float(perf_counter() - t0),
                mat_type=str(Pmat.getType()),
            )

    K_free = None
    solve_times: list[float] = []
    residual_norms: list[float] = []
    relative_residual_norms: list[float] = []
    residual_histories: list[list[float]] = []
    relative_residual_histories: list[list[float]] = []
    solve_deltas: list[dict[str, float]] = []
    iteration_counts: list[int] = []
    converged_reasons: list[int] = []
    solve_count = 3 if str(args.mode) == "repeat_solve" else 1
    solution_norm = 0.0
    if str(args.outer_solver_family) == "native_petsc":
        rhs_norm = float(np.linalg.norm(f_full))
        rhs_norm = max(rhs_norm, 1.0)
        K_elast.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        K_elast.setOption(PETSc.Mat.Option.SPD, True)
        native_ksp, setup_elapsed = _build_native_petsc_ksp(
            args,
            operator_matrix=K_elast,
            preconditioning_matrix=Pmat,
        )
        if rank == 0 and emit is not None:
            emit(
                "elastic_init_complete",
                total_wall_time=float(perf_counter() - t0),
                setup_elapsed_s=float(setup_elapsed),
                mode=str(args.mode),
            )

        for solve_idx in range(solve_count):
            x_full, elapsed, iterations, reason, residual_history, relative_residual_history = _native_petsc_ksp_solve_once(
                native_ksp,
                K_elast,
                f_full,
            )
            residual = np.asarray(matvec_to_numpy(K_elast, x_full), dtype=np.float64).reshape(-1) - f_full
            residual_norm = float(np.linalg.norm(residual))
            relative_residual_norm = float(residual_norm / rhs_norm)
            solve_times.append(float(elapsed))
            residual_norms.append(float(residual_norm))
            relative_residual_norms.append(float(relative_residual_norm))
            residual_histories.append([float(v) for v in residual_history])
            relative_residual_histories.append([float(v) for v in relative_residual_history])
            iteration_counts.append(int(iterations))
            converged_reasons.append(int(reason))
            solution_norm = float(np.linalg.norm(x_full))
            solve_deltas.append(
                {
                    "iterations": float(iterations),
                    "solve_time": float(elapsed),
                    "preconditioner_time": float(setup_elapsed if solve_idx == 0 else 0.0),
                    "orthogonalization_time": 0.0,
                    "converged_reason": float(reason),
                }
            )
            if rank == 0 and emit is not None:
                emit(
                    "elastic_solve_complete",
                    solve_index=int(solve_idx),
                    total_wall_time=float(perf_counter() - t0),
                    solve_elapsed_s=float(elapsed),
                    residual_norm=float(residual_norm),
                    relative_residual_norm=float(relative_residual_norm),
                    iterations=int(iterations),
                    converged_reason=int(reason),
                )

        diagnostics = {
            "pc_backend": str(args.pc_backend),
            "preconditioner_matrix_source": str(args.preconditioner_matrix_source),
            "preconditioner_matrix_policy": str(args.preconditioner_matrix_policy),
            "preconditioner_rebuild_policy": str(args.preconditioner_rebuild_policy),
            "preconditioner_rebuild_interval": int(args.preconditioner_rebuild_interval),
            "preconditioner_rebuild_count": 1,
            "preconditioner_reuse_count": max(0, int(solve_count - 1)),
            "preconditioner_age_max": max(0, int(solve_count - 1)),
            "preconditioner_setup_time_total": float(setup_elapsed),
            "preconditioner_apply_time_total": 0.0,
            "preconditioner_last_rebuild_reason": "initial",
        }
    else:
        solver = _build_solver(args, q_mask=q_mask, coord=coord)
        prefers_full = bool(getattr(solver, "prefers_full_system_operator", lambda: False)())
        setup_start = perf_counter()
        if prefers_full:
            solver.setup_preconditioner(
                K_elast,
                full_matrix=K_elast,
                free_indices=free_idx,
                preconditioning_matrix=Pmat,
            )
            A_solve = K_elast
        else:
            K_free = extract_submatrix_free(K_elast, free_idx)
            solver.setup_preconditioner(K_free, preconditioning_matrix=Pmat)
            A_solve = K_free
        setup_elapsed = perf_counter() - setup_start
        if rank == 0 and emit is not None:
            emit(
                "elastic_init_complete",
                total_wall_time=float(perf_counter() - t0),
                setup_elapsed_s=float(setup_elapsed),
                mode=str(args.mode),
            )

        rhs_norm = float(np.linalg.norm(f_free))
        rhs_norm = max(rhs_norm, 1.0)
        for solve_idx in range(solve_count):
            snap0 = _collector_snapshot(solver)
            ts = perf_counter()
            if prefers_full:
                x_free = np.asarray(
                    solver.solve(A_solve, f_free, full_rhs=f_full, free_indices=free_idx),
                    dtype=np.float64,
                ).reshape(-1)
                x_full = np.zeros_like(f_full)
                x_full[free_idx] = x_free
                residual = np.asarray(matvec_to_numpy(K_elast, x_full), dtype=np.float64).reshape(-1) - f_full
                residual_norm = float(np.linalg.norm(residual[free_idx]))
                solution_norm = float(np.linalg.norm(x_free))
            else:
                x_free = np.asarray(solver.solve(A_solve, f_free, free_indices=free_idx), dtype=np.float64).reshape(-1)
                residual = np.asarray(K_free @ x_free, dtype=np.float64).reshape(-1) - f_free
                residual_norm = float(np.linalg.norm(residual))
                solution_norm = float(np.linalg.norm(x_free))
            relative_residual_norm = float(residual_norm / rhs_norm)
            elapsed = perf_counter() - ts
            snap1 = _collector_snapshot(solver)
            solve_times.append(float(elapsed))
            residual_norms.append(float(residual_norm))
            relative_residual_norms.append(float(relative_residual_norm))
            solve_deltas.append(_collector_delta(snap0, snap1))
            if rank == 0 and emit is not None:
                emit(
                    "elastic_solve_complete",
                    solve_index=int(solve_idx),
                    total_wall_time=float(perf_counter() - t0),
                    solve_elapsed_s=float(elapsed),
                    residual_norm=float(residual_norm),
                    relative_residual_norm=float(relative_residual_norm),
                )

        diagnostics = solver.get_preconditioner_diagnostics()
    result = {
        "status": "completed",
        "mode": str(args.mode),
        "mesh_path": str(Path(args.mesh_path)),
        "elem_type": str(args.elem_type),
        "pc_backend": str(args.pc_backend),
        "outer_solver_family": str(args.outer_solver_family),
        "native_ksp_type": None if str(args.outer_solver_family) != "native_petsc" else str(args.native_ksp_type),
        "native_ksp_norm_type": None
        if str(args.outer_solver_family) != "native_petsc"
        else str(args.native_ksp_norm_type),
        "preconditioner_matrix_source": str(args.preconditioner_matrix_source),
        "petsc_opt_map": dict(getattr(args, "petsc_opt_map", {})),
        "pc_hypre_boomeramg_max_iter": (
            None if args.pc_hypre_boomeramg_max_iter is None else int(args.pc_hypre_boomeramg_max_iter)
        ),
        "adjacency_source": str(args.adjacency_source),
        "corner_only_primals": bool(args.corner_only_primals),
        "bddc_local_mat_type": str(args.bddc_local_mat_type),
        "runtime_seconds": float(perf_counter() - t0),
        "setup_elapsed_s": float(setup_elapsed),
        "solve_count": int(solve_count),
        "solve_times_s": [float(v) for v in solve_times],
        "residual_norms": [float(v) for v in residual_norms],
        "relative_residual_norms": [float(v) for v in relative_residual_norms],
        "reported_residual_histories": [[float(v) for v in hist] for hist in residual_histories],
        "relative_reported_residual_histories": [[float(v) for v in hist] for hist in relative_residual_histories],
        "residual_histories": [[float(v) for v in hist] for hist in residual_histories],
        "relative_residual_histories": [[float(v) for v in hist] for hist in relative_residual_histories],
        "reported_residual_history": ([] if not residual_histories else [float(v) for v in residual_histories[0]]),
        "relative_reported_residual_history": (
            [] if not relative_residual_histories else [float(v) for v in relative_residual_histories[0]]
        ),
        "final_relative_residual": (None if not relative_residual_norms else float(relative_residual_norms[-1])),
        "rhs_norm": float(rhs_norm),
        "solution_norm": float(solution_norm),
        "iteration_counts": [int(v) for v in iteration_counts],
        "converged_reasons": [int(v) for v in converged_reasons],
        "linear_total_rank_metric": float(sum(d["solve_time"] + d["preconditioner_time"] for d in solve_deltas)),
        "attempt_linear_preconditioner_time_total": float(sum(d["preconditioner_time"] for d in solve_deltas)),
        "attempt_linear_solve_time_total": float(sum(d["solve_time"] for d in solve_deltas)),
        "solve_deltas": solve_deltas,
        "progress_file_created": True,
        "first_progress_elapsed_s": float(setup_elapsed),
        **diagnostics,
    }
    if Pmat is not None:
        pmat_metadata = get_petsc_matrix_metadata(Pmat)
        local_mat = get_petsc_is_local_mat(Pmat)
        result.update(
            {
                "pmat_type": str(Pmat.getType()),
                "pmat_block_size": int(Pmat.getBlockSize()),
                "local_pmat_type": (None if local_mat is None else str(local_mat.getType())),
                "local_pmat_block_size": (None if local_mat is None else int(local_mat.getBlockSize())),
                "matis_local_mat_type": pmat_metadata.get("matis_local_mat_type"),
                "bddc_local_vertex_major_ordering": bool(pmat_metadata.get("bddc_local_vertex_major_ordering", False)),
                "bddc_local_block_size": (
                    None if pmat_metadata.get("bddc_local_block_size") is None else int(pmat_metadata["bddc_local_block_size"])
                ),
            }
        )
    if bddc_pattern is not None:
        result.update(
            {
                "bddc_local_total_bytes": float(bddc_pattern.stats.get("local_total_bytes", 0.0)),
                "bddc_local_primal_vertices_count": float(bddc_pattern.stats.get("local_primal_vertices_count", 0.0)),
                "bddc_local_interface_nodes_count": float(bddc_pattern.stats.get("local_interface_nodes_count", 0.0)),
                "bddc_adjacency_source": str(bddc_pattern.adjacency_source),
            }
        )

    result_max = {
        key: float(mpi_comm.allreduce(float(value), op=MPI.MAX))
        for key, value in result.items()
        if isinstance(value, (int, float, np.floating))
    }
    result["runtime_seconds_max"] = float(result_max.get("runtime_seconds", result["runtime_seconds"]))
    result["setup_elapsed_s_max"] = float(result_max.get("setup_elapsed_s", result["setup_elapsed_s"]))
    result["residual_norm_max"] = float(max(residual_norms) if residual_norms else 0.0)
    result["residual_norm_max"] = float(mpi_comm.allreduce(result["residual_norm_max"], op=MPI.MAX))

    if rank == 0:
        (data_dir / "run_info.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        md_lines = [
            "# BDDC Elastic Probe",
            "",
            f"- Element type: `{args.elem_type}`",
            f"- Mode: `{args.mode}`",
            f"- Backend: `{args.pc_backend}`",
            f"- Outer solver family: `{args.outer_solver_family}`",
            f"- Preconditioner matrix source: `{args.preconditioner_matrix_source}`",
            f"- Setup elapsed: `{result['setup_elapsed_s']:.6f} s`",
            f"- Solve times: `{', '.join(f'{v:.6f}' for v in solve_times)}`",
            f"- Residual norms: `{', '.join(f'{v:.3e}' for v in residual_norms)}`",
            f"- Relative residual norms: `{', '.join(f'{v:.3e}' for v in relative_residual_norms)}`",
            f"- Preconditioner rebuild count: `{diagnostics['preconditioner_rebuild_count']}`",
            f"- Preconditioner reuse count: `{diagnostics['preconditioner_reuse_count']}`",
        ]
        (out_dir / "elastic_probe.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        if emit is not None:
            emit("finished", total_wall_time=float(perf_counter() - t0), residual_norm_max=float(result["residual_norm_max"]))

    if str(args.outer_solver_family) == "native_petsc":
        native_ksp.destroy()
    else:
        solver.release_iteration_resources()
        if K_free is not None and isinstance(K_free, PETSc.Mat):
            release_petsc_aij_matrix(K_free)
            K_free.destroy()
    if isinstance(K_elast, PETSc.Mat):
        release_petsc_aij_matrix(K_elast)
        K_elast.destroy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Contained elastic-only PETSc/BDDC probe.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mesh-path", type=Path, default=ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh")
    parser.add_argument("--elem-type", type=str, default="P2", choices=["P2", "P4"])
    parser.add_argument("--mode", type=str, default="single_solve", choices=["single_solve", "repeat_solve"])
    parser.add_argument("--node-ordering", type=str, default="block_metis")
    parser.add_argument("--linear_tolerance", type=float, default=1e-8)
    parser.add_argument("--linear_max_iter", type=int, default=200)
    parser.add_argument("--outer_solver_family", type=str, default="repo", choices=["repo", "native_petsc"])
    parser.add_argument("--native_ksp_type", type=str, default="cg", choices=["cg", "fgmres", "gmres"])
    parser.add_argument(
        "--native_ksp_norm_type",
        type=str,
        default="unpreconditioned",
        choices=["default", "preconditioned", "unpreconditioned", "natural", "none"],
    )
    parser.add_argument("--solver_type", type=str, default="PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE")
    parser.add_argument("--pc_backend", type=str, default="bddc", choices=["hypre", "gamg", "bddc"])
    parser.add_argument("--preconditioner_matrix_source", type=str, default="elastic", choices=["tangent", "regularized", "elastic"])
    parser.add_argument("--preconditioner_matrix_policy", type=str, default="current", choices=["current", "lagged"])
    parser.add_argument(
        "--preconditioner_rebuild_policy",
        type=str,
        default="every_newton",
        choices=["every_newton", "every_n_newton", "accepted_step", "accepted_or_rejected_step"],
    )
    parser.add_argument("--preconditioner_rebuild_interval", type=int, default=1)
    parser.add_argument("--pc_hypre_coarsen_type", type=str, default=None)
    parser.add_argument("--pc_hypre_interp_type", type=str, default=None)
    parser.add_argument("--pc_hypre_boomeramg_max_iter", type=int, default=None)
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
    parser.add_argument("--adjacency_source", type=str, default="csr", choices=["csr", "none", "topology"])
    parser.add_argument("--corner_only_primals", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bddc_local_mat_type", type=str, default="aij", choices=["aij", "sbaij"])
    parser.add_argument("--petsc-opt", action="append", default=[], dest="petsc_opt")
    args = parser.parse_args()
    args.elem_type = validate_supported_elem_type(3, args.elem_type)
    args.petsc_opt_map = _parse_petsc_opt_entries(args.petsc_opt)
    result = run_probe(args)
    if int(PETSc.COMM_WORLD.getRank()) == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
