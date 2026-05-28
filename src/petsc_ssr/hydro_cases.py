from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from .context import EngineRunResult


ENGINE_ROOT = Path(__file__).resolve().parents[2]
STANDALONE_SRC = ENGINE_ROOT / "src"


def ensure_engine_imports() -> None:
    if STANDALONE_SRC.is_dir() and str(STANDALONE_SRC) not in sys.path:
        sys.path.insert(0, str(STANDALONE_SRC))


@dataclass(frozen=True, slots=True)
class HydroCaseTranslation:
    supported: bool
    reason: str
    config: Any | None = None
    resolved: Any | None = None
    elem_type: str = "P2"


def translate_hydro_case_toml(config_path: str | Path, *, allow_coupled: bool = False) -> HydroCaseTranslation:
    ensure_engine_imports()
    try:
        from petsc_ssr.config import load_run_case_config
        from petsc_ssr.problem_asset_runtime import load_seepage_problem_spec, resolve_problem_asset_from_config
    except Exception as exc:
        return HydroCaseTranslation(False, f"case TOML support is not importable: {exc}")

    cfg = load_run_case_config(Path(config_path)).validate()
    analysis = str(cfg.problem.analysis).strip().lower()
    if analysis != "seepage" and not (allow_coupled and analysis == "ssr"):
        return HydroCaseTranslation(False, "not a seepage-only case", config=cfg)

    resolved = resolve_problem_asset_from_config(cfg)
    if resolved.mesh_path is None:
        return HydroCaseTranslation(False, "resolved asset does not provide a mesh file path", config=cfg, resolved=resolved)

    seepage = load_seepage_problem_spec(resolved).seepage
    kinds = {str(bc.kind).strip().lower() for bc in seepage.head_bcs}
    targets = {str(bc.target).strip().lower() for bc in seepage.head_bcs}
    dim = int(resolved.dimension)
    if dim == 3 and (not {"dry", "constant_level"}.issuperset(kinds) or not {"head_dry", "head_porous", "head_free"}.issubset(targets)):
        return HydroCaseTranslation(
            False,
            "3D C hydro path currently supports head_dry/head_porous/head_free constant-head assets",
            config=cfg,
            resolved=resolved,
        )
    if dim == 2:
        if len(seepage.head_bcs) != 1 or "head_support" not in targets or kinds != {"piecewise_linear_level"}:
            return HydroCaseTranslation(
                False,
                "2D C hydro path currently supports one head_support piecewise_linear_level boundary with domain_below_head",
                config=cfg,
                resolved=resolved,
            )
        scope = str(seepage.head_bcs[0].value_model.get("scope", "support_only")).strip().lower()
        if scope != "domain_below_head":
            return HydroCaseTranslation(False, "2D C hydro path requires scope='domain_below_head'", config=cfg, resolved=resolved)
    if dim not in (2, 3):
        return HydroCaseTranslation(False, f"unsupported seepage dimension {resolved.dimension}", config=cfg, resolved=resolved)

    elem_type = str(cfg.problem.elem_type).strip().upper()
    if elem_type not in {"P1", "P2", "P4"}:
        return HydroCaseTranslation(False, f"unsupported hydro element type {elem_type!r}", config=cfg, resolved=resolved)
    return HydroCaseTranslation(True, f"supported_{dim}d_seepage_petsc", config=cfg, resolved=resolved, elem_type=elem_type)


def _quote_tokens(tokens: list[str]) -> str:
    return " ".join(shlex.quote(str(token)) for token in tokens)


def _read_distributed_vec_binary(path: Path) -> np.ndarray:
    comm = PETSc.COMM_WORLD.tompi4py()
    viewer = PETSc.Viewer().createBinary(str(path), "r", comm=PETSc.COMM_WORLD)
    vec = PETSc.Vec().load(viewer)
    viewer.destroy()
    local = np.asarray(vec.getArray(readonly=True), dtype=np.float64).copy()
    chunks = comm.gather(local, root=0)
    vec.destroy()
    if comm.rank == 0:
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
    return np.empty(0, dtype=np.float64)


def _read_dof_coords_csv(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 4), dtype=np.float64)
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)


def _map_pressure_to_case_order(pressure: np.ndarray, dof_coords: np.ndarray, coord: np.ndarray) -> tuple[np.ndarray, float]:
    if pressure.size == 0 or dof_coords.size == 0:
        return pressure, float("nan")
    dof_coords = np.atleast_2d(dof_coords)
    order = np.argsort(dof_coords[:, 0].astype(np.int64))
    dof_coords = dof_coords[order]
    if int(dof_coords[-1, 0]) >= pressure.size:
        return pressure, float("nan")
    values_by_global = pressure[dof_coords[:, 0].astype(np.int64)]

    dim = int(coord.shape[0])

    def key(values: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(values[:dim], dtype=np.float64), 11))

    lookup: dict[tuple[float, ...], int] = {}
    for i, row in enumerate(dof_coords):
        lookup[key(row[1 : 1 + dim])] = i
    mapped = np.empty(coord.shape[1], dtype=np.float64)
    max_error = 0.0
    for j in range(coord.shape[1]):
        k = key(coord[:, j])
        idx = lookup.get(k)
        if idx is None:
            raise RuntimeError(f"Could not map PETSc hydro DOF coordinate {coord[:, j]} to case mesh node {j}")
        mapped[j] = values_by_global[idx]
        max_error = max(max_error, float(np.max(np.abs(dof_coords[idx, 1 : 1 + dim] - coord[:, j]))))
    return mapped, max_error


def _build_case_hydro_arrays(translation: HydroCaseTranslation, pressure: np.ndarray, dof_coords_csv: Path) -> dict[str, np.ndarray]:
    ensure_engine_imports()
    from petsc_ssr.problem_asset_runtime import (
        build_mesh_for_resolved_asset,
        build_seepage_boundary_for_resolved_asset,
        load_seepage_problem_spec,
    )
    from petsc_ssr.seepage.flow import (
        _finalize_seepage_outputs,
        assemble_auxiliary_matrices,
        heter_conduct,
        penalty_parameters_2d,
        penalty_parameters_3d,
    )

    resolved = translation.resolved
    if resolved is None:
        return {"pw": pressure}
    built = build_mesh_for_resolved_asset(resolved, elem_type=translation.elem_type)
    coord = np.asarray(built.coord, dtype=np.float64)
    elem = np.asarray(built.elem, dtype=np.int64)
    surf = np.asarray(built.surf, dtype=np.int64)
    boundary_labels = np.asarray(getattr(built, "boundary_labels", np.empty(0, dtype=np.int64)), dtype=np.int64)
    seepage_spec = load_seepage_problem_spec(resolved)
    grho = float(seepage_spec.seepage.water_unit_weight)
    q_w, pw_d = build_seepage_boundary_for_resolved_asset(resolved, coord, surf, boundary_labels, grho=grho)
    dof_coords = _read_dof_coords_csv(dof_coords_csv)
    pw, map_error = _map_pressure_to_case_order(pressure, dof_coords, coord)
    assembly = assemble_auxiliary_matrices(coord, elem, translation.elem_type)
    eps = penalty_parameters_2d(coord, elem) if int(resolved.dimension) == 2 else penalty_parameters_3d(coord, elem)
    _, grad_p, mater_sat = _finalize_seepage_outputs(assembly, pw, eps)
    material_id = np.asarray(getattr(built, "material_id", np.zeros(elem.shape[1], dtype=np.int64)), dtype=np.int64)
    conductivity = np.asarray(seepage_spec.conductivity, dtype=np.float64).ravel()
    required = int(material_id.max()) + 1 if material_id.size else 1
    if conductivity.size == 1 and required > 1:
        conductivity = np.repeat(conductivity, required)
    conduct0 = heter_conduct(material_id, int(assembly.n_q), conductivity)
    return {
        "coord": coord,
        "elem": elem,
        "surf": surf,
        "material_identifier": material_id,
        "q_w": np.asarray(q_w, dtype=bool),
        "pw_d": np.asarray(pw_d, dtype=np.float64),
        "conduct0": np.asarray(conduct0, dtype=np.float64),
        "pw": np.asarray(pw, dtype=np.float64),
        "pressure": np.asarray(pw, dtype=np.float64),
        "grad_p": np.asarray(grad_p, dtype=np.float64),
        "mater_sat": np.asarray(mater_sat, dtype=bool),
        "dof_map_max_error": np.asarray([map_error], dtype=np.float64),
    }


def run_hydro_case(translation: HydroCaseTranslation, output_dir: str | Path) -> EngineRunResult:
    if not translation.supported or translation.config is None or translation.resolved is None:
        raise RuntimeError(f"Cannot run unsupported hydro case: {translation.reason}")

    from .native import _core

    comm = MPI.COMM_WORLD
    rank = comm.rank
    cfg = translation.config
    resolved = translation.resolved
    ensure_engine_imports()
    from petsc_ssr.problem_asset_runtime import load_seepage_problem_spec

    seepage = load_seepage_problem_spec(resolved).seepage
    output = Path(output_dir)
    data_dir = output / "data"
    exports_dir = output / "exports"
    summary_json = data_dir / "hydro_summary.json"
    pressure_binary = data_dir / "hydro_pressure.bin"
    dof_coords_csv = data_dir / "hydro_dof_coords.csv"
    if rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        exports_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    degree = int(translation.elem_type[1:])
    pc_variant = "gamg" if degree == 1 else "pmg"
    tokens = [
        "-hydro_mesh",
        str(resolved.mesh_path),
        "-hydro_dim",
        str(int(resolved.dimension)),
        "-hydro_elem_type",
        translation.elem_type,
        "-hydro_pc_variant",
        pc_variant,
        "-hydro_grho",
        f"{float(seepage.water_unit_weight):.17g}",
        "-hydro_newton_tol",
        f"{float(cfg.seepage.linear_tolerance):.17g}",
        "-hydro_newton_max_it",
        str(int(cfg.seepage.nonlinear_max_iter)),
        "-hydro_ksp_rtol",
        f"{float(cfg.seepage.linear_tolerance):.17g}",
        "-hydro_ksp_max_it",
        str(int(cfg.seepage.linear_max_iter)),
        "-hydro_summary_json",
        str(summary_json),
        "-hydro_pressure_binary",
        str(pressure_binary),
        "-hydro_dof_coords_csv",
        str(dof_coords_csv),
    ]
    if int(resolved.dimension) == 2:
        bc = seepage.head_bcs[0]
        model = dict(bc.value_model)
        points = model.get("points")
        if not isinstance(points, (list, tuple)) or len(points) < 2:
            raise RuntimeError("2D piecewise seepage head model must provide at least two [x, level] points")
        p0 = points[0]
        p1 = points[-1]
        tokens.extend(
            [
                "-hydro_head_mode",
                "support_piecewise",
                "-hydro_head_x0",
                f"{float(p0[0]):.17g}",
                "-hydro_head_y0",
                f"{float(p0[1]):.17g}",
                "-hydro_head_x1",
                f"{float(p1[0]):.17g}",
                "-hydro_head_y1",
                f"{float(p1[1]):.17g}",
            ]
        )
    else:
        tokens.extend(["-hydro_head_mode", "comsol3d"])
    if rank == 0:
        (data_dir / "hydro_options.txt").write_text(_quote_tokens(tokens) + "\n", encoding="utf-8")

    t0 = perf_counter()
    _core.run_hydro_options(_quote_tokens(tokens))
    wall = perf_counter() - t0
    comm.Barrier()

    summary: dict[str, Any] = {}
    if rank == 0 and summary_json.exists():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        summary["analysis"] = "seepage"
        summary["engine"] = "petsc_ssr_hydro_c"
        summary["asset"] = str(resolved.asset_name)
        summary["mesh_variant"] = str(resolved.variant_name)
        summary["wall_time_python"] = float(wall)
        summary["dof_coords_csv"] = str(dof_coords_csv)
    summary = comm.bcast(summary, root=0)
    result = EngineRunResult(output, pressure_binary, summary_json, wall, summary)
    return result


def write_hydro_case_outputs(result: EngineRunResult, translation: HydroCaseTranslation, config_path: str | Path) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    output = Path(result.output_dir)
    data_dir = output / "data"
    exports_dir = output / "exports"
    pressure_binary = Path(result.summary.get("pressure_binary", ""))
    dof_coords_csv = Path(result.summary.get("dof_coords_csv", ""))
    pressure = _read_distributed_vec_binary(pressure_binary) if pressure_binary.exists() else np.empty(0, dtype=np.float64)

    if rank != 0:
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    cfg = translation.config
    resolved = translation.resolved
    try:
        arrays = _build_case_hydro_arrays(translation, pressure, dof_coords_csv)
    except Exception as exc:
        arrays = {
            "pw": pressure,
            "pressure": pressure,
            "grad_p": np.empty((0, 0), dtype=np.float64),
            "mater_sat": np.empty(0, dtype=bool),
            "output_mapping_error": np.asarray([str(exc)], dtype=object),
        }
    np.savez_compressed(
        data_dir / "petsc_run.npz",
        **arrays,
        criterion=np.asarray([float(result.summary.get("final_criterion", np.nan))], dtype=np.float64),
        linear_iterations=np.asarray([int(result.summary.get("linear_iterations", 0))], dtype=np.int64),
    )
    payload = {
        "run_info": {
            "runtime_seconds": float(result.summary.get("wall_time", result.wall_time)),
            "mpi_size": int(result.summary.get("ranks", comm.size)),
            "mpi_mode": "distributed_petsc",
            "mesh_file": "" if resolved is None else str(resolved.mesh_path),
            "mesh_nodes": int(result.summary.get("global_dofs", pressure.size)),
            "mesh_elements": int(result.summary.get("cells", 0)),
            "solver_type": "PETSC_DMPLEX_C_HYDRO",
            "pc_backend": str(result.summary.get("pc_variant", "")),
        },
        "params": {
            "elem_type": translation.elem_type,
            "asset_name": "" if resolved is None else str(resolved.asset_name),
            "mesh_variant": "" if resolved is None else str(resolved.variant_name),
            "linear_tolerance": None if cfg is None else float(cfg.seepage.linear_tolerance),
            "linear_max_iter": None if cfg is None else int(cfg.seepage.linear_max_iter),
            "nonlinear_max_iter": None if cfg is None else int(cfg.seepage.nonlinear_max_iter),
        },
        "history": {
            "criterion": [float(result.summary.get("final_criterion", np.nan))],
            "iterations": int(result.summary.get("newton_iterations", 0)),
            "converged": bool(float(result.summary.get("final_criterion", np.inf)) <= (float(cfg.seepage.linear_tolerance) if cfg is not None else 1.0e-10)),
        },
        "timings": {
            "linear": {
                "total_linear_iterations": int(result.summary.get("linear_iterations", 0)),
                "solve_time": float(result.summary.get("solve_time", 0.0)),
                "assembly_time": float(result.summary.get("assembly_time", 0.0)),
            }
        },
        "c_hydro_summary": result.summary,
    }
    if "dof_map_max_error" in arrays:
        payload["run_info"]["dof_map_max_error"] = float(np.asarray(arrays["dof_map_max_error"]).ravel()[0])
    (data_dir / "run_info.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_case_hydro_vtu(exports_dir / "final_solution.vtu", arrays, translation)
    _write_config_copy(config_path, exports_dir / "resolved_config.toml")
    _write_config_copy(config_path, output / "generated_case.toml")


def _write_case_hydro_vtu(path: Path, arrays: dict[str, np.ndarray], translation: HydroCaseTranslation) -> None:
    ensure_engine_imports()
    from petsc_ssr.export import write_vtu
    from petsc_ssr.postprocess.field_exports import build_field_exports

    coord = np.asarray(arrays.get("coord", np.empty((0, 0))), dtype=np.float64)
    elem = np.asarray(arrays.get("elem", np.empty((0, 0), dtype=np.int64)), dtype=np.int64)
    if coord.ndim != 2 or elem.ndim != 2 or coord.size == 0 or elem.size == 0:
        raise RuntimeError("Cannot write hydro VTU without case-order coord/elem arrays.")
    dim = int(coord.shape[0])
    points = np.zeros((coord.shape[1], 3), dtype=np.float64)
    points[:, :dim] = coord.T
    elem_type = str(translation.elem_type).strip().upper()
    point_data, cell_data = build_field_exports(
        arrays,
        n_cells=int(elem.shape[1]),
        coord=coord,
        elem=elem,
        elem_type=elem_type,
        dim=dim,
    )
    if "material_identifier" in arrays:
        cell_data["material_id"] = np.asarray(arrays["material_identifier"], dtype=np.int64).reshape(-1)
    write_vtu(
        path,
        points=points,
        cell_blocks=[(_hydro_vtu_cell_type(dim, elem_type), elem.T)],
        point_data=point_data,
        cell_data=cell_data,
    )


def _hydro_vtu_cell_type(dim: int, elem_type: str) -> str:
    key = str(elem_type).strip().upper()
    if int(dim) == 2:
        if key == "P1":
            return "triangle"
        if key == "P2":
            return "triangle6"
        if key == "P4":
            return "VTK_LAGRANGE_TRIANGLE"
    if int(dim) == 3:
        if key == "P1":
            return "tetra"
        if key == "P2":
            return "tetra10"
        if key == "P4":
            return "VTK_LAGRANGE_TETRAHEDRON"
    raise ValueError(f"Unsupported hydro VTU cell type for dim={dim}, elem_type={elem_type!r}")


def _write_config_copy(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src_path.resolve() == dst_path.resolve():
            return
    except FileNotFoundError:
        pass
    dst_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")


def write_coupled_pressure_table(result: EngineRunResult, translation: HydroCaseTranslation) -> Path:
    """Write x,y,z,pressure rows for the mechanics C assembly pressure lookup."""

    comm = MPI.COMM_WORLD
    rank = comm.rank
    output = Path(result.output_dir)
    data_dir = output / "data"
    table = data_dir / "coupled_pressure_nodes.csv"
    pressure_binary = Path(result.summary.get("pressure_binary", ""))
    dof_coords_csv = Path(result.summary.get("dof_coords_csv", ""))
    pressure = _read_distributed_vec_binary(pressure_binary) if pressure_binary.exists() else np.empty(0, dtype=np.float64)

    if rank == 0:
        arrays = _build_case_hydro_arrays(translation, pressure, dof_coords_csv)
        coord = np.asarray(arrays["coord"], dtype=np.float64)
        pw = np.asarray(arrays["pressure"], dtype=np.float64).reshape(-1)
        if coord.shape[1] != pw.size:
            raise RuntimeError(f"Cannot write coupled pressure table: coord nodes={coord.shape[1]} pressure={pw.size}")
        data_dir.mkdir(parents=True, exist_ok=True)
        with table.open("w", encoding="utf-8") as fh:
            fh.write("x,y,z,pressure\n")
            for i in range(pw.size):
                x = float(coord[0, i])
                y = float(coord[1, i]) if coord.shape[0] >= 2 else 0.0
                z = float(coord[2, i]) if coord.shape[0] >= 3 else 0.0
                fh.write(f"{x:.17e},{y:.17e},{z:.17e},{float(pw[i]):.17e}\n")
    comm.Barrier()
    return table
