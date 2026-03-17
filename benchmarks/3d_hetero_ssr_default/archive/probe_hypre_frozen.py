from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from slope_stability.constitutive.problem import ConstitutiveOperator
from slope_stability.core.elements import validate_supported_elem_type
from slope_stability.fem.distributed_elastic import assemble_owned_elastic_rows_for_comm
from slope_stability.fem.distributed_tangent import prepare_owned_tangent_pattern
from slope_stability.fem.quadrature import quadrature_volume_3d
from slope_stability.linear.solver import SolverFactory
from slope_stability.linear.preconditioners import attach_near_nullspace, make_near_nullspace_elasticity
from slope_stability.mesh import load_mesh_from_file
from slope_stability.mesh.materials import MaterialSpec, heterogenous_materials
from slope_stability.mesh.reorder import reorder_mesh_nodes
from slope_stability.problem_assets import load_material_rows_for_path
from slope_stability.utils import (
    extract_submatrix_free,
    global_array_to_petsc_vec,
    local_csr_to_petsc_aij_matrix,
    matvec_to_numpy,
    owned_block_range,
    petsc_vec_to_global_array,
    q_to_free_indices,
    release_petsc_aij_matrix,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "archive" else SCRIPT_DIR
ROOT = BENCHMARK_DIR.parents[1]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _set_petsc_option(options, key: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        options[key] = "true" if value else "false"
    else:
        options[key] = value


def _native_ksp_type(name: str):
    normalized = str(name).strip().lower()
    mapping = {
        "fgmres": PETSc.KSP.Type.FGMRES,
        "gmres": PETSc.KSP.Type.GMRES,
        "cg": PETSc.KSP.Type.CG,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported native KSP type {name!r}")
    return mapping[normalized]


def _native_pc_type(name: str):
    normalized = str(name).strip().lower()
    mapping = {
        "hypre": PETSc.PC.Type.HYPRE,
        "gamg": PETSc.PC.Type.GAMG,
        "hmg": PETSc.PC.Type.HMG,
        "mg": PETSc.PC.Type.MG,
        "deflation": PETSc.PC.Type.DEFLATION,
        "composite": PETSc.PC.Type.COMPOSITE,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported native PC type {name!r}")
    return mapping[normalized]


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


def _build_native_petsc_ksp(args, *, operator_matrix: PETSc.Mat, preconditioning_matrix: PETSc.Mat) -> tuple[PETSc.KSP, float]:
    t0 = perf_counter()
    ksp = PETSc.KSP().create(comm=operator_matrix.getComm())
    prefix = f"probe_native_{int(ksp.handle)}_"
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(operator_matrix, preconditioning_matrix)
    ksp.setType(_native_ksp_type(args.native_ksp_type))
    ksp.setInitialGuessNonzero(False)
    ksp.setTolerances(rtol=float(args.linear_tolerance), atol=1e-30, max_it=int(args.linear_max_iter))
    ksp.setNormType(_native_ksp_norm_type(args.native_ksp_norm_type))
    pc = ksp.getPC()
    native_pc_type = str(getattr(args, "native_pc_type", "hypre")).strip().lower()
    pc.setType(_native_pc_type(native_pc_type))
    if native_pc_type == "hypre":
        pc.setHYPREType("boomeramg")
    opts = PETSc.Options()
    if native_pc_type == "hypre":
        for key, value in (
            ("pc_hypre_boomeramg_coarsen_type", args.pc_hypre_coarsen_type),
            ("pc_hypre_boomeramg_interp_type", args.pc_hypre_interp_type),
            ("pc_hypre_boomeramg_strong_threshold", args.pc_hypre_strong_threshold),
            ("pc_hypre_boomeramg_max_iter", args.pc_hypre_boomeramg_max_iter),
            ("pc_hypre_boomeramg_P_max", args.pc_hypre_P_max),
            ("pc_hypre_boomeramg_agg_nl", args.pc_hypre_agg_nl),
            ("pc_hypre_boomeramg_nongalerkin_tol", args.pc_hypre_nongalerkin_tol),
            ("pc_hypre_boomeramg_nodal_coarsen", 4),
            ("pc_hypre_boomeramg_nodal_coarsen_diag", 1),
            ("pc_hypre_boomeramg_vec_interp_variant", 2),
            ("pc_hypre_boomeramg_vec_interp_qmax", 4),
            ("pc_hypre_boomeramg_vec_interp_smooth", True),
            ("pc_hypre_boomeramg_numfunctions", 3),
            ("pc_hypre_boomeramg_tol", 0.0),
        ):
            _set_petsc_option(opts, f"{prefix}{key}", value)
    for key, value in getattr(args, "petsc_opt_map", {}).items():
        target_key = str(key) if _is_global_petsc_option(str(key)) else f"{prefix}{key}"
        _set_petsc_option(opts, target_key, value)
    ksp.setFromOptions()
    ksp.setUp()
    return ksp, float(perf_counter() - t0)


def _native_ksp_solve_once(ksp: PETSc.KSP, A: PETSc.Mat, rhs_full: np.ndarray) -> tuple[np.ndarray, float, int, int, list[float], list[float]]:
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
    # Collect the distributed solution back to a global dense array so later
    # diagnostics and free-index slicing operate on consistent indexing.
    solution = np.asarray(petsc_vec_to_global_array(x), dtype=np.float64).copy()
    reason = int(ksp.getConvergedReason())
    iterations = int(ksp.getIterationNumber())
    if not residual_history:
        residual = np.asarray(matvec_to_numpy(A, solution), dtype=np.float64).reshape(-1) - np.asarray(rhs_full, dtype=np.float64).reshape(-1)
        residual_history = [float(np.linalg.norm(residual))]
    relative_history = [float(v / rhs_norm) for v in residual_history]
    ksp.cancelMonitor()
    rhs.destroy()
    x.destroy()
    return solution, elapsed, iterations, reason, residual_history, relative_history


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


def _rank_hint_from_path(path: Path) -> int | None:
    text = str(path)
    match = re.search(r"rank(\d+)", text)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_state_run_info(args) -> dict:
    run_info_path = args.state_run_info
    if run_info_path is None:
        sibling = Path(args.state_npz).with_name("run_info.json")
        if sibling.exists():
            run_info_path = sibling
    if run_info_path is None or not Path(run_info_path).exists():
        return {}
    return json.loads(Path(run_info_path).read_text(encoding="utf-8"))


def _select_state(args, run_info: dict) -> dict[str, object]:
    npz = np.load(args.state_npz)
    step_u = np.asarray(npz["step_U"]) if "step_U" in npz else np.empty((0,), dtype=np.float64)
    lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
    omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64) if "omega_hist" in npz else np.empty(0, dtype=np.float64)
    final_u = np.asarray(npz["U"], dtype=np.float64)

    selector = str(args.state_selector).strip().lower()
    use_final = False
    selected_index = None
    if selector == "easy":
        if step_u.ndim >= 3 and step_u.shape[0] >= 2:
            selected_index = 1
        elif step_u.ndim >= 3 and step_u.shape[0] == 1:
            selected_index = 0
        else:
            use_final = True
    elif selector == "hard":
        if step_u.ndim >= 3 and step_u.shape[0] >= 1:
            selected_index = int(step_u.shape[0] - 1)
        else:
            use_final = True
    elif selector == "final":
        use_final = True
    elif selector == "index":
        if args.state_index is None:
            raise ValueError("--state-index is required when --state-selector=index")
        if step_u.ndim < 3:
            raise ValueError("Selected indexed state, but source artifact has no step_U history")
        selected_index = int(args.state_index)
        if selected_index < 0 or selected_index >= int(step_u.shape[0]):
            raise ValueError(f"State index {selected_index} out of range for step_U with shape {step_u.shape}")
    else:
        raise ValueError(f"Unsupported state selector {args.state_selector!r}")

    if use_final:
        state_u = final_u
        lambda_value = float(lambda_hist[-1])
        omega_value = None if omega_hist.size == 0 else float(omega_hist[-1])
        selected_label = "final"
    else:
        state_u = np.asarray(step_u[selected_index], dtype=np.float64)
        lambda_value = float(lambda_hist[selected_index])
        omega_value = None if omega_hist.size <= selected_index else float(omega_hist[selected_index])
        selected_label = f"step_{selected_index}"

    params = dict(run_info.get("params", {}))
    source_node_ordering = str(params.get("node_ordering", args.node_ordering))
    source_elem_type = str(params.get("elem_type", args.elem_type))
    raw_r_min = params.get("r_min", (1e-4 if args.regularization_r is None else args.regularization_r))
    source_r_min = float(raw_r_min)
    source_material_rows = params.get("material_rows")

    reorder_parts = args.reorder_parts
    if reorder_parts is None and source_node_ordering.lower() == "block_metis":
        reorder_parts = _rank_hint_from_path(Path(args.state_npz))
    if reorder_parts is None and source_node_ordering.lower() == "block_metis":
        reorder_parts = int(PETSc.COMM_WORLD.getSize())

    return {
        "state_u": state_u,
        "lambda_value": lambda_value,
        "omega_value": omega_value,
        "selected_label": selected_label,
        "selected_index": selected_index,
        "source_node_ordering": source_node_ordering,
        "source_elem_type": source_elem_type,
        "regularization_r": source_r_min,
        "material_rows": source_material_rows,
        "reorder_parts": reorder_parts,
        "step_u_shape": tuple(int(v) for v in getattr(step_u, "shape", ())),
        "lambda_hist_len": int(lambda_hist.size),
    }


def _build_problem(
    *,
    mesh_path: Path,
    elem_type: str,
    node_ordering: str,
    reorder_parts: int | None,
    material_rows: list[list[float]] | None,
    davis_type: str,
    constitutive_mode: str,
    tangent_kernel: str,
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
    reordered = reorder_mesh_nodes(
        mesh.coord,
        mesh.elem,
        mesh.surf,
        mesh.q_mask,
        strategy=node_ordering,
        n_parts=reorder_parts if str(node_ordering).lower() == "block_metis" else None,
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
    return {
        "coord": coord,
        "elem": elem,
        "q_mask": q_mask,
        "K_elast": K_elast,
        "f_V": f_V,
        "const_builder": const_builder,
        "tangent_pattern": tangent_pattern,
        "materials": material_rows,
    }


def _build_preconditioner_options(args) -> dict[str, object]:
    options: dict[str, object] = {
        "threads": 16,
        "print_level": 0,
        "use_as_preconditioner": True,
        "pc_backend": "hypre",
        "preconditioner_matrix_source": str(args.pmat_source),
        "preconditioner_matrix_policy": "current",
        "preconditioner_rebuild_policy": "every_newton",
        "preconditioner_rebuild_interval": 1,
        "mpi_distribute_by_nodes": True,
        "use_coordinates": True,
        "recycle_preconditioner": False,
        "max_deflation_basis_vectors": int(args.max_deflation_basis_vectors),
    }
    if args.pc_hypre_coarsen_type is not None:
        options["pc_hypre_boomeramg_coarsen_type"] = str(args.pc_hypre_coarsen_type)
    if args.pc_hypre_interp_type is not None:
        options["pc_hypre_boomeramg_interp_type"] = str(args.pc_hypre_interp_type)
    if args.pc_hypre_strong_threshold is not None:
        options["pc_hypre_boomeramg_strong_threshold"] = float(args.pc_hypre_strong_threshold)
    if args.pc_hypre_boomeramg_max_iter is not None:
        options["pc_hypre_boomeramg_max_iter"] = int(args.pc_hypre_boomeramg_max_iter)
    if args.pc_hypre_P_max is not None:
        options["pc_hypre_boomeramg_P_max"] = int(args.pc_hypre_P_max)
    if args.pc_hypre_agg_nl is not None:
        options["pc_hypre_boomeramg_agg_nl"] = int(args.pc_hypre_agg_nl)
    if args.pc_hypre_nongalerkin_tol is not None:
        options["pc_hypre_boomeramg_nongalerkin_tol"] = float(args.pc_hypre_nongalerkin_tol)
    options.update(_parse_petsc_opt_entries(args.petsc_opt))
    return options


def _dump_ksp_view(path: Path, solver) -> None:
    ksp = getattr(solver, "_ksp", None)
    inner_ksp = getattr(solver, "_inner_ksp", None)
    comm = None
    if ksp is not None:
        comm = ksp.getComm()
    elif inner_ksp is not None:
        comm = inner_ksp.getComm()
    else:
        comm = PETSc.COMM_WORLD
    viewer = PETSc.Viewer().createASCII(str(path), mode="w", comm=comm)
    if ksp is not None:
        viewer.printfASCII("=== outer_ksp ===\n")
        ksp.view(viewer)
    if inner_ksp is not None:
        viewer.printfASCII("\n=== inner_ksp ===\n")
        inner_ksp.view(viewer)
    viewer.destroy()


def run_probe(args) -> dict[str, object]:
    rank = int(PETSc.COMM_WORLD.getRank())
    mpi_comm = MPI.COMM_WORLD
    out_dir = _ensure_dir(Path(args.out_dir)) if rank == 0 else Path(args.out_dir)
    data_dir = out_dir / "data"
    if rank == 0:
        _ensure_dir(data_dir)

    run_info = _resolve_state_run_info(args)
    state = _select_state(args, run_info)
    elem_type = validate_supported_elem_type(3, args.elem_type or state["source_elem_type"])
    mesh_path = Path(args.mesh_path)
    node_ordering = str(args.node_ordering or state["source_node_ordering"])
    material_rows = state["material_rows"]
    if material_rows is not None:
        material_rows = [list(map(float, row)) for row in material_rows]
    regularization_r = float(args.regularization_r if args.regularization_r is not None else state["regularization_r"])

    t0 = perf_counter()
    problem = _build_problem(
        mesh_path=mesh_path,
        elem_type=elem_type,
        node_ordering=node_ordering,
        reorder_parts=state["reorder_parts"],
        material_rows=material_rows,
        davis_type=str(args.davis_type),
        constitutive_mode=str(args.constitutive_mode),
        tangent_kernel=str(args.tangent_kernel),
    )
    coord = problem["coord"]
    q_mask = problem["q_mask"]
    K_elast = problem["K_elast"]
    f_V = problem["f_V"]
    const_builder = problem["const_builder"]
    state_u = np.asarray(state["state_u"], dtype=np.float64)
    if state_u.shape != (coord.shape[0], coord.shape[1]):
        raise ValueError(
            f"Frozen state shape {state_u.shape} does not match reordered mesh {(coord.shape[0], coord.shape[1])}. "
            "Check node ordering / reorder_parts against the source artifact."
        )

    const_builder.reduction(float(state["lambda_value"]))
    F_state, K_tangent = const_builder.build_F_K_tangent_reduced(state_u)
    load_full = np.asarray(f_V, dtype=np.float64).reshape(-1, order="F")
    internal_full = np.asarray(F_state, dtype=np.float64).reshape(-1, order="F")
    if str(args.rhs_source).strip().lower() == "body":
        rhs_full = load_full
    else:
        rhs_full = float(state["lambda_value"]) * load_full - internal_full
    free_idx = q_to_free_indices(q_mask)
    f_free = np.asarray(rhs_full[free_idx], dtype=np.float64)
    if str(args.pmat_source) == "tangent":
        Pmat = K_tangent
    elif str(args.pmat_source) == "regularized":
        Pmat = const_builder.build_K_regularized(regularization_r)
    elif str(args.pmat_source) == "elastic":
        Pmat = K_elast
    else:
        raise ValueError(f"Unsupported pmat_source {args.pmat_source!r}")

    K_free = None
    P_free = None
    native_ksp = None
    solver = None
    solve_info: dict[str, object] = {}

    outer_solver_family = str(args.outer_solver_family).strip().lower()
    if outer_solver_family == "native_petsc":
        null_space = make_near_nullspace_elasticity(
            coord,
            q_mask=q_mask,
            center_coordinates=True,
            return_full=True,
        )
        K_tangent, _, _ = attach_near_nullspace(K_tangent, null_space)
        if Pmat is not K_tangent:
            Pmat, _, _ = attach_near_nullspace(Pmat, null_space)
        native_ksp, setup_elapsed = _build_native_petsc_ksp(
            args,
            operator_matrix=K_tangent,
            preconditioning_matrix=Pmat,
        )
        viewer = PETSc.Viewer().createASCII(str(data_dir / "ksp_view.txt"), mode="w", comm=K_tangent.getComm())
        native_ksp.view(viewer)
        viewer.destroy()
        x_full, solve_elapsed, iteration_count, converged_reason, history, true_history = _native_ksp_solve_once(
            native_ksp,
            K_tangent,
            rhs_full,
        )
        x_free = np.asarray(x_full[free_idx], dtype=np.float64)
        residual = np.asarray(matvec_to_numpy(K_tangent, x_full), dtype=np.float64).reshape(-1) - rhs_full
        residual_norm = float(np.linalg.norm(residual[free_idx]))
        solve_delta = {
            "iterations": float(iteration_count),
            "solve_time": float(solve_elapsed),
            "preconditioner_time": float(setup_elapsed),
            "orthogonalization_time": 0.0,
            "converged_reason": float(converged_reason),
        }
        diagnostics = {
            "pc_backend": str(getattr(args, "native_pc_type", "hypre")),
            "preconditioner_matrix_source": str(args.pmat_source),
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "preconditioner_rebuild_count": 1,
            "preconditioner_reuse_count": 0,
            "preconditioner_age_max": 0,
            "preconditioner_apply_time_total": 0.0,
            "preconditioner_setup_time_total": float(setup_elapsed),
            "preconditioner_last_rebuild_reason": "initial",
        }
        solve_info = {
            "reported_residual_history": history,
            "true_residual_history": true_history,
            "true_residual_final": float(true_history[-1] if true_history else residual_norm / max(np.linalg.norm(f_free), 1.0)),
        }
    elif outer_solver_family == "repo":
        preconditioner_options = _build_preconditioner_options(args)
        solver = SolverFactory.create(
            args.solver_type,
            tolerance=float(args.linear_tolerance),
            max_iterations=int(args.linear_max_iter),
            deflation_basis_tolerance=1e-3,
            verbose=False,
            q_mask=q_mask,
            coord=coord,
            preconditioner_options=preconditioner_options,
        )
        enable_diagnostics = getattr(solver, "enable_diagnostics", None)
        if callable(enable_diagnostics):
            enable_diagnostics(True)
        snap0 = _collector_snapshot(solver)
        solve_setup_t0 = perf_counter()
        prefers_full = bool(getattr(solver, "prefers_full_system_operator", lambda: False)())
        if prefers_full:
            solver.setup_preconditioner(
                K_tangent,
                full_matrix=K_tangent,
                free_indices=free_idx,
                preconditioning_matrix=Pmat,
            )
            _dump_ksp_view(data_dir / "ksp_view.txt", solver)
            x_free = np.asarray(
                solver.solve(K_tangent, f_free, full_rhs=rhs_full, free_indices=free_idx),
                dtype=np.float64,
            ).reshape(-1)
            x_full = np.zeros_like(rhs_full)
            x_full[free_idx] = x_free
            residual = np.asarray(matvec_to_numpy(K_tangent, x_full), dtype=np.float64).reshape(-1) - rhs_full
            residual_norm = float(np.linalg.norm(residual[free_idx]))
        else:
            K_free = extract_submatrix_free(K_tangent, free_idx)
            if Pmat is K_tangent:
                P_free = K_free
            elif isinstance(Pmat, PETSc.Mat):
                P_free = extract_submatrix_free(Pmat, free_idx)
            else:
                P_free = Pmat
            solver.setup_preconditioner(
                K_free,
                full_matrix=K_tangent,
                free_indices=free_idx,
                preconditioning_matrix=P_free,
            )
            if rank == 0:
                _dump_ksp_view(data_dir / "ksp_view.txt", solver)
            x_free = np.asarray(solver.solve(K_free, f_free, free_indices=free_idx), dtype=np.float64).reshape(-1)
            residual = np.asarray(K_free @ x_free, dtype=np.float64).reshape(-1) - f_free
            residual_norm = float(np.linalg.norm(residual))
        _solve_elapsed_total = float(perf_counter() - solve_setup_t0)
        snap1 = _collector_snapshot(solver)
        solve_delta = _collector_delta(snap0, snap1)
        diagnostics = solver.get_preconditioner_diagnostics()
        solve_info = getattr(solver, "get_last_solve_info", lambda: {})()
        solve_elapsed = float(solve_delta["solve_time"])
        setup_elapsed = float(solve_delta["preconditioner_time"])
    else:
        raise ValueError(f"Unsupported outer_solver_family {args.outer_solver_family!r}")

    rhs_norm = float(np.linalg.norm(f_free))
    rhs_norm = max(rhs_norm, 1.0)
    final_relative_residual = float(residual_norm / rhs_norm)
    history = [float(v) for v in solve_info.get("reported_residual_history", [])]
    true_history = [float(v) for v in solve_info.get("true_residual_history", [])]
    if not true_history and final_relative_residual >= 0.0:
        true_history = [final_relative_residual]

    result: dict[str, object] = {
        "status": "completed",
        "mesh_path": str(mesh_path),
        "elem_type": str(elem_type),
        "node_ordering": str(node_ordering),
        "reorder_parts": (None if state["reorder_parts"] is None else int(state["reorder_parts"])),
        "solver_type": str(args.solver_type),
        "outer_solver_family": outer_solver_family,
        "native_ksp_type": (None if outer_solver_family != "native_petsc" else str(args.native_ksp_type)),
        "native_ksp_norm_type": (None if outer_solver_family != "native_petsc" else str(args.native_ksp_norm_type)),
        "pc_backend": ("hypre" if outer_solver_family != "native_petsc" else str(getattr(args, "native_pc_type", "hypre"))),
        "native_pc_type": (None if outer_solver_family != "native_petsc" else str(getattr(args, "native_pc_type", "hypre"))),
        "rhs_source": str(args.rhs_source),
        "pmat_source": str(args.pmat_source),
        "regularization_r": float(regularization_r),
        "pc_hypre_coarsen_type": args.pc_hypre_coarsen_type,
        "pc_hypre_interp_type": args.pc_hypre_interp_type,
        "pc_hypre_strong_threshold": args.pc_hypre_strong_threshold,
        "pc_hypre_boomeramg_max_iter": args.pc_hypre_boomeramg_max_iter,
        "pc_hypre_P_max": args.pc_hypre_P_max,
        "pc_hypre_agg_nl": args.pc_hypre_agg_nl,
        "pc_hypre_nongalerkin_tol": args.pc_hypre_nongalerkin_tol,
        "petsc_opt_map": _parse_petsc_opt_entries(args.petsc_opt),
        "state_npz": str(Path(args.state_npz)),
        "state_run_info": (None if args.state_run_info is None else str(Path(args.state_run_info))),
        "state_selector": str(args.state_selector),
        "selected_state_label": str(state["selected_label"]),
        "selected_state_index": state["selected_index"],
        "state_lambda": float(state["lambda_value"]),
        "state_omega": state["omega_value"],
        "state_step_u_shape": list(state["step_u_shape"]),
        "state_lambda_hist_len": int(state["lambda_hist_len"]),
        "runtime_seconds": float(perf_counter() - t0),
        "setup_elapsed_s": float(solve_delta["preconditioner_time"]),
        "solve_elapsed_s": float(solve_delta["solve_time"]),
        "orthogonalization_elapsed_s": float(solve_delta["orthogonalization_time"]),
        "solve_plus_setup_elapsed_s": float(float(solve_delta["preconditioner_time"]) + float(solve_delta["solve_time"])),
        "iteration_count": int(solve_delta["iterations"]),
        "rhs_norm": float(rhs_norm),
        "residual_norm": float(residual_norm),
        "final_relative_residual": float(final_relative_residual),
        "reported_residual_history": history,
        "true_residual_history": true_history,
        "true_residual_final": float(solve_info.get("true_residual_final", final_relative_residual)),
        "solve_delta": solve_delta,
        **diagnostics,
    }
    result["runtime_seconds_max"] = float(mpi_comm.allreduce(result["runtime_seconds"], op=MPI.MAX))
    result["setup_elapsed_s_max"] = float(mpi_comm.allreduce(result["setup_elapsed_s"], op=MPI.MAX))
    result["solve_elapsed_s_max"] = float(mpi_comm.allreduce(result["solve_elapsed_s"], op=MPI.MAX))
    result["residual_norm_max"] = float(mpi_comm.allreduce(result["residual_norm"], op=MPI.MAX))

    if rank == 0:
        (data_dir / "run_info.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        md_lines = [
            "# Hypre Frozen-State Probe",
            "",
            f"- Element type: `{elem_type}`",
            f"- Frozen state: `{state['selected_label']}`",
            f"- Lambda: `{state['lambda_value']:.9f}`",
            f"- Outer solver family: `{outer_solver_family}`",
            f"- Pmat source: `{args.pmat_source}`",
            f"- BoomerAMG max_iter: `{args.pc_hypre_boomeramg_max_iter}`",
            f"- Iterations: `{result['iteration_count']}`",
            f"- Setup elapsed: `{result['setup_elapsed_s']:.6f} s`",
            f"- Solve elapsed: `{result['solve_elapsed_s']:.6f} s`",
            f"- Final relative residual: `{result['final_relative_residual']:.3e}`",
        ]
        (out_dir / "frozen_probe.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    release = getattr(solver, "release_iteration_resources", None)
    if callable(release):
        release()
    if native_ksp is not None:
        native_ksp.destroy()

    destroyed_ids: set[int] = set()

    def _destroy_mat(mat) -> None:
        if not isinstance(mat, PETSc.Mat):
            return
        mat_id = id(mat)
        if mat_id in destroyed_ids:
            return
        release_petsc_aij_matrix(mat)
        mat.destroy()
        destroyed_ids.add(mat_id)

    _destroy_mat(K_free)
    _destroy_mat(P_free)
    _destroy_mat(K_elast)
    _destroy_mat(K_tangent)
    _destroy_mat(Pmat)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-state Hypre explicit-Pmat probe for 3D SSR.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--state-npz", type=Path, required=True)
    parser.add_argument("--state-run-info", type=Path, default=None)
    parser.add_argument("--state-selector", type=str, default="easy", choices=["easy", "hard", "final", "index"])
    parser.add_argument("--state-index", type=int, default=None)
    parser.add_argument("--mesh-path", type=Path, default=ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh")
    parser.add_argument("--elem-type", type=str, default="P4", choices=["P1", "P2", "P4"])
    parser.add_argument(
        "--node-ordering",
        type=str,
        default="block_metis",
        choices=["original", "xyz", "block_xyz", "morton", "rcm", "block_rcm", "block_metis"],
    )
    parser.add_argument("--reorder-parts", type=int, default=None)
    parser.add_argument("--davis-type", type=str, default="B")
    parser.add_argument("--constitutive-mode", type=str, default="overlap")
    parser.add_argument("--tangent-kernel", type=str, default="rows", choices=["legacy", "rows"])
    parser.add_argument("--pmat-source", type=str, default="tangent", choices=["tangent", "regularized", "elastic"])
    parser.add_argument("--regularization-r", type=float, default=None)
    parser.add_argument("--linear-tolerance", type=float, default=1e-5)
    parser.add_argument("--linear-max-iter", type=int, default=300)
    parser.add_argument("--outer-solver-family", type=str, default="native_petsc", choices=["native_petsc", "repo"])
    parser.add_argument("--native-ksp-type", type=str, default="fgmres", choices=["fgmres", "gmres", "cg"])
    parser.add_argument("--native-pc-type", type=str, default="hypre", choices=["hypre", "gamg", "hmg", "mg", "deflation", "composite"])
    parser.add_argument(
        "--native-ksp-norm-type",
        type=str,
        default="unpreconditioned",
        choices=["default", "preconditioned", "unpreconditioned", "natural", "none"],
    )
    parser.add_argument("--rhs-source", type=str, default="residual", choices=["residual", "body"])
    parser.add_argument("--solver-type", type=str, default="PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE")
    parser.add_argument("--pc-hypre-coarsen-type", type=str, default="HMIS")
    parser.add_argument("--pc-hypre-interp-type", type=str, default="ext+i")
    parser.add_argument("--pc-hypre-strong-threshold", type=float, default=None)
    parser.add_argument("--pc-hypre-boomeramg-max-iter", type=int, default=None)
    parser.add_argument("--pc-hypre-P-max", type=int, default=None)
    parser.add_argument("--pc-hypre-agg-nl", type=int, default=None)
    parser.add_argument("--pc-hypre-nongalerkin-tol", type=float, default=None)
    parser.add_argument("--max-deflation-basis-vectors", type=int, default=16)
    parser.add_argument("--petsc-opt", action="append", default=[])
    args = parser.parse_args()
    args.elem_type = validate_supported_elem_type(3, args.elem_type)
    result = run_probe(args)
    if int(PETSc.COMM_WORLD.getRank()) == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
