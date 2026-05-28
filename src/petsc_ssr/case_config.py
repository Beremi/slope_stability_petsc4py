from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .context import EngineRunResult
from .options import LinearOptions, PmgOptions, SsrOptions
from .problem import BoundarySpec, MaterialSpec, ProblemSpec


ENGINE_ROOT = Path(__file__).resolve().parents[2]
STANDALONE_SRC = ENGINE_ROOT / "src"
STANDALONE_CASES = ENGINE_ROOT / "benchmarks" / "cases"


def ensure_engine_imports() -> None:
    """Ensure the standalone engine package is importable from helper scripts."""

    if STANDALONE_SRC.is_dir() and str(STANDALONE_SRC) not in sys.path:
        sys.path.insert(0, str(STANDALONE_SRC))


@dataclass(frozen=True, slots=True)
class CaseTranslation:
    supported: bool
    reason: str
    problem: ProblemSpec | None = None
    options: SsrOptions | None = None
    config: Any | None = None
    mesh_info: dict[str, Any] | None = None


def translate_case_toml(config_path: str | Path, *, refine_levels: int | None = None, force_full_c_baseline: bool = False) -> CaseTranslation:
    ensure_engine_imports()
    try:
        from petsc_ssr.config import load_run_case_config
        from petsc_ssr.problem_asset_runtime import resolve_problem_asset_from_config
    except Exception as exc:
        return CaseTranslation(False, f"case TOML support is not importable: {exc}")

    cfg = load_run_case_config(Path(config_path)).validate()
    resolved = resolve_problem_asset_from_config(cfg)
    analysis = str(cfg.problem.analysis).strip().lower()
    method = str(cfg.continuation.method).strip().lower()
    elem_type = str(cfg.problem.elem_type).strip().upper()
    degree = _degree_from_elem_type(elem_type)
    mesh_path = resolved.mesh_path
    mesh_info = _parse_gmsh_physical_names(mesh_path) if mesh_path else {"regions": {}, "boundaries": {}}

    if analysis not in {"ssr", "ll"}:
        return CaseTranslation(False, f"analysis={analysis!r} is not in the current C engine mechanics scope", config=cfg, mesh_info=mesh_info)
    if method not in {"indirect", "direct"}:
        return CaseTranslation(False, f"continuation.method={method!r} is not implemented in the C engine yet", config=cfg, mesh_info=mesh_info)
    dimension = int(resolved.dimension)
    if dimension not in (2, 3):
        return CaseTranslation(False, f"dimension={resolved.dimension} is not supported by the C engine", config=cfg, mesh_info=mesh_info)
    seepage_coupled = bool(cfg.problem.seepage_coupled)
    if mesh_path is None:
        return CaseTranslation(False, "resolved asset does not provide a mesh file path for DMPlex", config=cfg, mesh_info=mesh_info)

    try:
        materials = tuple(_material_specs_from_asset(resolved, mesh_info))
        boundary = _boundary_spec_from_asset(resolved, mesh_info)
    except Exception as exc:
        return CaseTranslation(False, f"case asset translation failed: {exc}", config=cfg, mesh_info=mesh_info)

    linear = LinearOptions(
        rtol=float(cfg.linear_solver.tolerance),
        max_it=int(cfg.linear_solver.max_iterations),
        deflation=True,
        deflation_solver=_deflation_solver_from_solver_name(str(cfg.linear_solver.solver_type)),
    )
    pmg = PmgOptions.current_baseline()
    for name, value in (
        ("p2_active_ranks", cfg.linear_solver.pmg_shell_p2_active_ranks),
        ("p1_active_ranks", cfg.linear_solver.pmg_shell_p1_active_ranks),
        ("subcomm_type", cfg.linear_solver.pmg_shell_subcomm_type),
        ("fine_ksp_max_it", cfg.linear_solver.pmg_shell_fine_ksp_max_it),
        ("p2_ksp_max_it", cfg.linear_solver.pmg_shell_p2_ksp_max_it),
        ("p1_pc_type", cfg.linear_solver.pmg_shell_p1_pc_type),
        ("p1_redundant_number", cfg.linear_solver.pmg_shell_p1_redundant_number),
        ("p1_redundant_ksp_type", cfg.linear_solver.pmg_shell_p1_redundant_ksp_type),
        ("p1_redundant_ksp_rtol", cfg.linear_solver.pmg_shell_p1_redundant_ksp_rtol),
        ("p1_redundant_ksp_max_it", cfg.linear_solver.pmg_shell_p1_redundant_ksp_max_it),
        ("p1_redundant_pc_type", cfg.linear_solver.pmg_shell_p1_redundant_pc_type),
    ):
        if value is not None:
            setattr(pmg, name, value)

    pc_variant = "gamg" if degree == 1 else "pmg"
    options = SsrOptions(
        analysis=analysis,
        continuation_method=method,
        omega_max=float(cfg.continuation.omega_max),
        lambda_init=float(cfg.continuation.lambda_init),
        d_lambda_init=float(cfg.continuation.d_lambda_init),
        d_lambda_min=float(cfg.continuation.d_lambda_min),
        d_lambda_diff_scaled_min=float(cfg.continuation.d_lambda_diff_scaled_min),
        lambda_ell=float(cfg.continuation.lambda_ell),
        d_t_min=float(cfg.continuation.d_t_min),
        d_omega_ini_scale=float(cfg.continuation.d_omega_ini_scale),
        continuation_step_max=int(cfg.continuation.step_max),
        newton_max_it=int(cfg.newton.it_max),
        newton_rtol=float(cfg.newton.tol),
        newton_stopping_criterion=_normalize_stopping(cfg.newton.stopping_criterion),
        newton_stopping_tol=float(cfg.newton.stopping_tol if cfg.newton.stopping_tol is not None else cfg.newton.tol),
        init_newton_stopping_criterion=_normalize_stopping(cfg.continuation.init_newton_stopping_criterion or "relative_correction"),
        init_newton_stopping_tol=float(cfg.continuation.init_newton_stopping_tol if cfg.continuation.init_newton_stopping_tol is not None else 1.0e-3),
        it_damp_max=int(cfg.newton.it_damp_max),
        r_min=float(cfg.newton.r_min),
        line_search=(str(cfg.newton.line_search).strip().lower() == "alg5"),
        continuation_predictor=str(cfg.continuation.predictor),
        omega_step_controller=str(cfg.continuation.omega_step_controller),
        pc_variant=pc_variant,
        partitioner=str(cfg.problem.partitioner),
        linear=linear,
        pmg=pmg,
        petsc_options=list(cfg.linear_solver.petsc_opt),
    )
    if force_full_c_baseline:
        options.linear.deflation_solver = "fgmres"

    problem = ProblemSpec(
        name=str(cfg.problem.name or cfg.problem.case or Path(config_path).parent.name),
        mesh_path=Path(mesh_path),
        dimension=dimension,
        element_degree=degree,
        refine_levels=int(cfg.problem.refine_levels if refine_levels is None else refine_levels),
        boundary=boundary,
        materials=materials,
        metadata={
            "case_config": str(Path(config_path).resolve()),
            "asset": resolved.asset_name,
            "mesh_variant": resolved.variant_name,
            "elem_type": elem_type,
            "seepage_coupled": seepage_coupled,
        },
    )
    suffix = "_seepage_coupled" if seepage_coupled else ""
    return CaseTranslation(True, f"supported_{dimension}d_{method}_{analysis}{suffix}", problem=problem, options=options, config=cfg, mesh_info=mesh_info)


def write_case_outputs(result: EngineRunResult, translation: CaseTranslation, config_path: str | Path) -> None:
    rank = _rank()
    if rank != 0:
        return

    output_dir = Path(result.output_dir)
    data_dir = output_dir / "data"
    exports_dir = output_dir / "exports"
    data_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    curve_rows = _read_curve(result.curve_csv)
    steps = np.asarray([int(row["step"]) for row in curve_rows], dtype=np.int64)
    omega = np.asarray([float(row["omega"]) for row in curve_rows], dtype=np.float64)
    lambdas = np.asarray([float(row["lambda"]) for row in curve_rows], dtype=np.float64)
    umax = np.asarray([float(row["u_max"]) for row in curve_rows], dtype=np.float64)
    newton = np.asarray([int(row["newton_iterations"]) for row in curve_rows], dtype=np.int64)
    linear = np.asarray([int(row["linear_iterations"]) for row in curve_rows], dtype=np.int64)
    case_mesh = _load_case_mesh_for_output(translation)
    displacement = _load_case_ordered_displacement(result, translation, case_mesh)
    coupled_fields = _load_coupled_hydro_fields(output_dir)

    np.savez_compressed(
        data_dir / "petsc_run.npz",
        step=steps,
        omega_hist=omega,
        lambda_hist=lambdas,
        load_factor_hist=lambdas,
        Umax_hist=umax,
        U=displacement,
        step_U=np.empty((0, displacement.shape[0] if displacement.size else 0, displacement.shape[1] if displacement.size else 0), dtype=np.float64),
        stats_step_index=steps,
        stats_step_omega=omega,
        stats_step_lambda=lambdas,
        stats_step_newton_iterations=newton,
        stats_step_linear_iterations=linear,
        **coupled_fields,
    )
    payload = {
        "run_info": {
            "python_version": "petsc_ssr full-C DMPlex backend",
            "mpi_size": int(result.summary.get("ranks", 1)),
            "analysis": str(translation.config.problem.analysis if translation.config is not None else result.summary.get("analysis", "ssr")),
            "mechanics_backend": "petsc_ssr_full_c",
            "solver_type": "PETSC_DMPLEX_C_PMG",
            "unknowns": int(result.summary.get("global_dofs", 0)),
            "step_count": int(result.summary.get("accepted_steps", len(curve_rows))),
        },
        "params": {
            "case_config": str(Path(config_path).resolve()),
            "mesh_file": "" if translation.problem is None else str(translation.problem.mesh_path),
            "elem_type": "" if translation.problem is None else f"P{translation.problem.element_degree}",
        },
        "timings": {
            "continuation_total_wall_time": float(result.summary.get("continuation_wall_time", 0.0)),
            "wall_time": float(result.summary.get("wall_time", result.wall_time)),
            "linear": {
                "attempt_linear_iterations_total": int(result.summary.get("total_linear_its", 0)),
                "deflation_orthogonalization_time": float(result.summary.get("deflation_orthogonalization_time", 0.0)),
                "deflation_pc_apply_time": float(result.summary.get("deflation_pc_apply_time", 0.0)),
                "deflation_projector_time": float(result.summary.get("deflation_projector_time", 0.0)),
            },
        },
        "c_hotpath_summary": result.summary,
        "lambda_last": float(result.summary.get("lambda_last", lambdas[-1] if lambdas.size else 0.0)),
        "omega_last": float(result.summary.get("omega_last", omega[-1] if omega.size else 0.0)),
        "final_rel": float(result.summary.get("final_rel", 0.0)),
        "final_rel_correction": float(result.summary.get("final_rel_correction", 0.0)),
        "newton_iterations_total": int(result.summary.get("total_newton_its", 0)),
    }
    (data_dir / "run_info.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_case_vtu(exports_dir / "final_solution.vtu", translation, case_mesh, displacement, extra_arrays=coupled_fields)
    _copy_config_if_different(config_path, exports_dir / "resolved_config.toml")
    _copy_config_if_different(config_path, output_dir / "generated_case.toml")


def _copy_config_if_different(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src_path.resolve() == dst_path.resolve():
            return
    except FileNotFoundError:
        pass
    shutil.copyfile(src_path, dst_path)


def _load_coupled_hydro_fields(output_dir: Path) -> dict[str, np.ndarray]:
    hydro_npz = output_dir / "hydro_prepass" / "data" / "petsc_run.npz"
    if not hydro_npz.exists():
        return {}
    with np.load(hydro_npz, allow_pickle=True) as npz_file:
        arrays = {name: np.asarray(npz_file[name]) for name in npz_file.files}
    fields: dict[str, np.ndarray] = {}
    pressure = None
    for key in ("pore_pressure_export", "pw_export", "seepage_pw_reordered", "pw_reordered", "pw", "pressure"):
        if key in arrays:
            pressure = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
            break
    if pressure is not None:
        fields["pore_pressure_export"] = pressure
        fields["pw_export"] = pressure
        fields["seepage_pw_reordered"] = pressure
    saturation = None
    for key in ("saturation", "mater_sat", "seepage_mater_sat"):
        if key in arrays:
            saturation = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
            break
    if saturation is not None:
        fields["mater_sat"] = saturation
        fields["seepage_mater_sat"] = saturation
        fields["saturation"] = saturation
    return fields


def _load_case_mesh_for_output(translation: CaseTranslation) -> Any | None:
    if translation.config is None or translation.problem is None:
        return None
    if int(translation.problem.refine_levels) != 0:
        return None
    ensure_engine_imports()
    from petsc_ssr.problem_asset_runtime import build_mesh_for_resolved_asset, resolve_problem_asset_from_config

    resolved = resolve_problem_asset_from_config(translation.config)
    elem_type = str(translation.problem.metadata.get("elem_type", f"P{translation.problem.element_degree}"))
    return build_mesh_for_resolved_asset(resolved, elem_type=elem_type)


def _load_case_ordered_displacement(result: EngineRunResult, translation: CaseTranslation, case_mesh: Any | None) -> np.ndarray:
    if translation.problem is None or case_mesh is None:
        return np.empty((0, 0), dtype=np.float64)
    csv_raw = str(result.summary.get("solution_points_csv", "") or "").strip()
    csv_path = Path(csv_raw) if csv_raw else Path()
    if not csv_raw or not csv_path.exists() or csv_path.is_dir():
        csv_path = Path(result.output_dir) / "data" / "final_displacement_points.csv"
    if not csv_path.exists() or csv_path.is_dir():
        print(
            "ROOT_OUTPUT_DISPLACEMENT_MAP "
            f"points=0 matched=0 missing=0 source=missing reason=no_solution_points_csv",
            flush=True,
        )
        return np.empty((0, 0), dtype=np.float64)

    coord = np.asarray(case_mesh.coord, dtype=np.float64)
    dim = int(translation.problem.dimension)
    if coord.ndim != 2 or coord.shape[0] < dim:
        return np.empty((0, 0), dtype=np.float64)
    span = float(np.max(np.ptp(coord[:dim, :], axis=1))) if coord.size else 1.0
    tol = max(1.0e-10, 1.0e-8 * max(span, 1.0))
    table: dict[tuple[int, int, int], tuple[np.ndarray, int]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            xyz = np.array([float(row.get(name, 0.0) or 0.0) for name in ("x", "y", "z")], dtype=np.float64)
            disp = np.array([float(row.get(name, 0.0) or 0.0) for name in ("ux", "uy", "uz")], dtype=np.float64)
            key = _coord_key(xyz, tol)
            acc = table.get(key)
            if acc is None:
                table[key] = (disp.copy(), 1)
            else:
                acc[0][:] += disp
                table[key] = (acc[0], acc[1] + 1)

    U = np.zeros((dim, coord.shape[1]), dtype=np.float64)
    matched = 0
    for i in range(coord.shape[1]):
        xyz = np.zeros(3, dtype=np.float64)
        xyz[:dim] = coord[:dim, i]
        acc = table.get(_coord_key(xyz, tol))
        if acc is None:
            continue
        U[:, i] = (acc[0] / max(acc[1], 1))[:dim]
        matched += 1
    print(
        "ROOT_OUTPUT_DISPLACEMENT_MAP "
        f"points={coord.shape[1]} matched={matched} missing={coord.shape[1] - matched} "
        f"source={csv_path} tol={tol:.3e}",
        flush=True,
    )
    return U


def _coord_key(xyz: np.ndarray, tol: float) -> tuple[int, int, int]:
    return tuple(int(round(float(xyz[d]) / tol)) for d in range(3))


def _write_case_vtu(
    path: Path,
    translation: CaseTranslation,
    case_mesh: Any | None,
    displacement: np.ndarray,
    *,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    if translation.problem is None or case_mesh is None or displacement.size == 0:
        return
    try:
        import meshio
        ensure_engine_imports()
        from petsc_ssr.postprocess.field_exports import build_field_exports
    except Exception as exc:
        print(f"ROOT_OUTPUT_VTU skipped=true reason=import_failed detail={exc}", flush=True)
        return

    coord = np.asarray(case_mesh.coord, dtype=np.float64)
    elem = np.asarray(case_mesh.elem, dtype=np.int64)
    if coord.ndim != 2 or elem.ndim != 2:
        return
    points = np.zeros((coord.shape[1], 3), dtype=np.float64)
    points[:, : coord.shape[0]] = coord.T
    dim = int(translation.problem.dimension)
    degree = int(translation.problem.element_degree)
    cell_type = _meshio_cell_type(dim, degree)
    if cell_type is None:
        print("ROOT_OUTPUT_VTU skipped=true reason=unsupported_cell_type", flush=True)
        return
    cells = elem.T
    linearized = False
    if degree > 2:
        # meshio/VTK do not consistently support high-order simplex cell names
        # such as tetra35. Keep all high-order points and point fields, but use
        # the vertex subset for a robust visualization mesh.
        cell_type = "triangle" if dim == 2 else "tetra"
        cells = elem[: dim + 1, :].T
        linearized = True

    arrays = {"U": displacement}
    if extra_arrays:
        arrays.update(_filter_vtu_extra_arrays(extra_arrays, n_points=coord.shape[1], n_cells=elem.shape[1]))
    max_strain_cells = int(os.environ.get("SSR_ROOT_STRAIN_EXPORT_MAX_CELLS", "50000"))
    if elem.shape[1] <= max_strain_cells:
        point_data, cell_data = build_field_exports(
            arrays,
            n_cells=elem.shape[1],
            coord=coord,
            elem=elem,
            elem_type=str(translation.problem.metadata.get("elem_type", f"P{translation.problem.element_degree}")),
            dim=int(translation.problem.dimension),
        )
    else:
        disp = np.zeros((displacement.shape[1], 3), dtype=np.float64)
        disp[:, : displacement.shape[0]] = displacement.T
        point_data = {
            "displacement": disp,
            "displacement_magnitude": np.linalg.norm(disp, axis=1),
        }
        cell_data = {}
    mesh = meshio.Mesh(points=points, cells=[(cell_type, cells)], point_data=point_data, cell_data={name: [values] for name, values in cell_data.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(path, mesh, file_format="vtu", binary=False)
    print(
        "ROOT_OUTPUT_VTU "
        f"path={path} points={points.shape[0]} cells={elem.shape[1]} cell_type={cell_type} "
        f"linearized={str(linearized).lower()} "
        f"strain_exported={str('deviatoric_strain' in point_data or 'deviatoric_strain' in cell_data).lower()}",
        flush=True,
    )


def _filter_vtu_extra_arrays(extra_arrays: dict[str, np.ndarray], *, n_points: int, n_cells: int) -> dict[str, np.ndarray]:
    filtered: dict[str, np.ndarray] = {}
    for key in ("pore_pressure_export", "pw_export", "seepage_pw_reordered", "pw_reordered"):
        if key not in extra_arrays:
            continue
        values = np.asarray(extra_arrays[key], dtype=np.float64).reshape(-1)
        if values.size == n_points:
            filtered[key] = values
            break
    for key in ("saturation", "mater_sat", "seepage_mater_sat"):
        if key not in extra_arrays:
            continue
        values = np.asarray(extra_arrays[key], dtype=np.float64).reshape(-1)
        if values.size == n_cells:
            filtered[key] = values
            break
    return filtered


def _meshio_cell_type(dim: int, degree: int) -> str | None:
    if dim == 2:
        return {1: "triangle", 2: "triangle6", 4: "triangle15"}.get(degree)
    if dim == 3:
        return {1: "tetra", 2: "tetra10", 4: "tetra35"}.get(degree)
    return None


def benchmark_capability_rows(root: str | Path | None = None) -> list[dict[str, str]]:
    ensure_engine_imports()
    root_path = STANDALONE_CASES if root is None else Path(root)
    rows: list[dict[str, str]] = []
    for path in sorted(root_path.glob("*/case.toml")):
        try:
            from .hydro_cases import translate_hydro_case_toml

            hydro = translate_hydro_case_toml(path)
        except Exception:
            hydro = None
        if hydro is not None and hydro.supported:
            cfg = hydro.config
            rows.append(
                {
                    "case": path.parent.name,
                    "config": str(path),
                    "supported": "true",
                    "reason": hydro.reason,
                    "analysis": "" if cfg is None else str(cfg.problem.analysis),
                    "dimension": "" if cfg is None else str(cfg.problem.dimension),
                    "elem_type": hydro.elem_type,
                }
            )
            continue
        trans = translate_case_toml(path)
        cfg = trans.config
        rows.append(
            {
                "case": path.parent.name,
                "config": str(path),
                "supported": str(trans.supported).lower(),
                "reason": trans.reason,
                "analysis": "" if cfg is None else str(cfg.problem.analysis),
                "dimension": "" if cfg is None else str(cfg.problem.dimension),
                "elem_type": "" if cfg is None else str(cfg.problem.elem_type),
            }
        )
    return rows


def write_capability_report(path: str | Path, rows: list[dict[str, str]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["case", "supported", "analysis", "dimension", "elem_type", "reason", "config"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_mechanics_constraint_table(translation: CaseTranslation, output_dir: str | Path) -> Path:
    """Write case q_mask constraints as coordinate/component rows for the C engine.

    Root assets can constrain named node sets as well as boundary faces. The C
    DMPlex path consumes this compact table as an additional algebraic
    constraint layer, so the benchmark translation follows the case q_mask
    exactly without duplicating asset-specific node-set expansion rules in C.
    """

    if translation.config is None or translation.problem is None:
        raise ValueError("mechanics constraint table requires a supported mechanics translation")
    ensure_engine_imports()
    from petsc_ssr.problem_asset_runtime import build_mesh_for_resolved_asset, resolve_problem_asset_from_config

    resolved = resolve_problem_asset_from_config(translation.config)
    elem_type = str(translation.problem.metadata.get("elem_type", f"P{translation.problem.element_degree}"))
    mesh = build_mesh_for_resolved_asset(resolved, elem_type=elem_type)
    coord = np.asarray(mesh.coord, dtype=np.float64)
    q_mask = np.asarray(mesh.q_mask, dtype=bool)
    constrained = ~q_mask
    selected = np.any(constrained, axis=0)

    out = Path(output_dir) / "data" / "mechanics_bc_nodes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["x", "y", "z", "cx", "cy", "cz"])
        for node in np.flatnonzero(selected):
            xyz = [float(coord[d, node]) if d < coord.shape[0] else 0.0 for d in range(3)]
            flags = [1 if d < constrained.shape[0] and bool(constrained[d, node]) else 0 for d in range(3)]
            writer.writerow([f"{xyz[0]:.17g}", f"{xyz[1]:.17g}", f"{xyz[2]:.17g}", *flags])
    return out


def _parse_gmsh_physical_names(path: Path | None) -> dict[str, dict[str, int]]:
    regions: dict[str, int] = {}
    boundaries: dict[str, int] = {}
    nodesets: dict[str, int] = {}
    if path is None or not Path(path).exists():
        return {"regions": regions, "boundaries": boundaries, "nodesets": nodesets}
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = lines.index("$PhysicalNames")
    except ValueError:
        return {"regions": regions, "boundaries": boundaries, "nodesets": nodesets}
    n = int(lines[start + 1].strip())
    for raw in lines[start + 2 : start + 2 + n]:
        parts = raw.split(maxsplit=2)
        if len(parts) < 3:
            continue
        dim = int(parts[0])
        tag = int(parts[1])
        name = parts[2].strip().strip('"')
        if name.startswith("region:"):
            regions[name.split(":", 1)[1]] = tag
        elif name.startswith("boundary:"):
            boundaries[name.split(":", 1)[1]] = tag
        elif name.startswith("nodeset:"):
            nodesets[name.split(":", 1)[1]] = tag
        elif dim == 3:
            regions[name] = tag
        elif dim == 2:
            boundaries[name] = tag
        elif dim == 0:
            nodesets[name] = tag
    return {"regions": regions, "boundaries": boundaries, "nodesets": nodesets}


def _material_specs_from_asset(resolved: Any, mesh_info: dict[str, Any]) -> list[MaterialSpec]:
    mechanics = resolved.definition.mechanics_spec()
    if mechanics is None:
        raise ValueError(f"asset {resolved.asset_name!r} has no mechanical model")
    regions: dict[str, int] = dict(mesh_info.get("regions", {}))
    if not regions and len(mechanics.materials) == 1:
        regions = {"region_1": 1}
    out: list[MaterialSpec] = []
    for region_name, region_id in sorted(regions.items(), key=lambda item: int(item[1])):
        material_name = mechanics.region_assignment.get(region_name)
        if material_name is None and len(mechanics.materials) == 1:
            material_name = next(iter(mechanics.materials))
        if material_name is None:
            raise ValueError(f"no material assignment for mesh region {region_name!r}")
        model = mechanics.materials[material_name]
        row = model.mechanical_row()
        if row is None:
            raise ValueError(f"material {material_name!r} is not mechanical")
        out.append(
            MaterialSpec(
                region=int(region_id),
                c0=float(row[0]),
                phi_deg=float(row[1]),
                psi_deg=float(row[2]),
                young=float(row[3]),
                poisson=float(row[4]),
                gamma_sat=float(row[5]),
                gamma_unsat=float(row[6]) if len(row) > 6 else float(row[5]),
            )
        )
    return out


def _boundary_spec_from_asset(resolved: Any, mesh_info: dict[str, Any]) -> BoundarySpec:
    mechanics = resolved.definition.mechanics_spec()
    if mechanics is None:
        return BoundarySpec()
    profile = resolved.resolved_variant.profile or mechanics.default_profile
    spec = mechanics.profiles.get(profile) or mechanics.profiles[mechanics.default_profile]
    rules = {rule.target: tuple(rule.components) for rule in spec.dirichlet}
    boundaries: dict[str, int] = dict(mesh_info.get("boundaries", {}))

    base_components = set(rules.get("base", ()))
    if int(resolved.dimension) == 2:
        has_left_x = "x" in set(rules.get("left", ()))
        has_right_x = "x" in set(rules.get("right", ()))
        if base_components == {"x", "y"} and has_left_x and has_right_x:
            return BoundarySpec(
                mode="2d_rollers",
                tag_base=int(boundaries.get("base", 1)),
                tag_x_min=int(boundaries.get("left", 0)),
                tag_x_max=int(boundaries.get("right", 0)),
                tag_z_min=0,
                tag_z_max=0,
            )
        if base_components == {"y"} and has_left_x and has_right_x:
            return BoundarySpec(
                mode="rollers",
                tag_base=int(boundaries.get("base", 1)),
                tag_x_min=int(boundaries.get("left", 0)),
                tag_x_max=int(boundaries.get("right", 0)),
                tag_z_min=0,
                tag_z_max=0,
            )
        raise ValueError(f"unsupported 2D Dirichlet rules for C boundary modes: {rules!r}")

    has_x = "x" in set(rules.get("x_lock", ()))
    has_z = "z" in set(rules.get("z_lock", ()))
    if base_components == {"y"} and has_x and has_z:
        mode = "rollers"
    elif base_components == {"x", "y", "z"} and has_x and has_z:
        mode = "fixed_base_rollers"
    elif base_components == {"x", "y", "z"}:
        mode = "fixed_base_rollers"
    elif base_components == {"y"}:
        mode = "base_only"
    else:
        raise ValueError(f"unsupported Dirichlet rules for C boundary modes: {rules!r}")

    return BoundarySpec(
        mode=mode,
        tag_base=int(boundaries.get("base", 5)),
        tag_x_min=int(boundaries.get("x_min", 0)),
        tag_x_max=int(boundaries.get("x_max", 0)),
        tag_z_min=int(boundaries.get("z_min", 0)),
        tag_z_max=int(boundaries.get("z_max", 0)),
    )


def _has_seepage_coupling(resolved: Any) -> bool:
    try:
        return resolved.definition.seepage_spec() is not None
    except Exception:
        return False


def _degree_from_elem_type(elem_type: str) -> int:
    key = elem_type.strip().upper()
    if key not in {"P1", "P2", "P4"}:
        raise ValueError(f"unsupported element type {elem_type!r}; current C engine supports P1/P2/P4 simplex elements")
    return int(key[1:])


def _normalize_stopping(value: str) -> str:
    key = str(value).strip().lower()
    if key in {"residual", "rel_residual", "relative_residual"}:
        return "relative_residual"
    if key in {"correction", "rel_correction", "relative_correction", "relative_newton_correction"}:
        return "relative_correction"
    if key in {"delta_lambda", "abs_delta_lambda", "absolute_delta_lambda"}:
        return "absolute_delta_lambda"
    raise ValueError(f"unsupported stopping criterion {value!r}")


def _deflation_solver_from_solver_name(solver_type: str) -> str:
    text = solver_type.upper()
    if "DFGMRES" in text:
        return "matlab_dfgmres"
    if "CG" in text:
        return "cg"
    return "fgmres"


def _read_curve(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _rank() -> int:
    try:
        from petsc4py import PETSc

        return int(PETSc.COMM_WORLD.getRank())
    except Exception:
        return 0
