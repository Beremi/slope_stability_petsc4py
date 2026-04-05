"""Case-mesh reconstruction shared by exports and notebook tooling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slope_stability.core.elements import simplex_vtk_cell_block
from slope_stability.core.run_config import RunCaseConfig
from slope_stability.mesh import (
    generate_homogeneous_slope_mesh_2d,
    generate_sloan2013_mesh_2d,
    load_mesh_franz_dam_2d,
    load_mesh_from_file,
    load_mesh_gmsh_waterlevels,
    load_mesh_kozinec_2d,
    load_mesh_luzec_2d,
    load_mesh_p2_comsol,
    reorder_mesh_nodes,
)


@dataclass(frozen=True)
class CaseMesh:
    dim: int
    coord: np.ndarray
    elem: np.ndarray
    surf: np.ndarray | None
    q_mask: np.ndarray | None
    material_id: np.ndarray
    points: np.ndarray
    cell_blocks: list[tuple[str, np.ndarray]]


def rebuild_case_mesh(cfg: RunCaseConfig, *, mpi_size: int = 1) -> CaseMesh:
    case = cfg.problem.case
    part_count = int(mpi_size) if cfg.execution.node_ordering.lower() == "block_metis" else None

    if case == "2d_homo_ssr":
        geom = cfg.geometry
        beta_deg = float(geom.get("beta_deg", 45.0))
        y2 = float(geom.get("y2", 10.0))
        x2 = y2 / np.tan(np.deg2rad(beta_deg))
        mesh = generate_homogeneous_slope_mesh_2d(
            elem_type=cfg.problem.elem_type,
            h=float(geom.get("h", 1.0)),
            x1=float(geom.get("x1", 15.0)),
            x2=float(x2),
            x3=float(geom.get("x3", 15.0)),
            y1=float(geom.get("y1", 10.0)),
            y2=y2,
        )
        reordered = _maybe_reorder(mesh.coord, mesh.elem, mesh.surf, mesh.q_mask, cfg, part_count)
        return _build_case_mesh(
            dim=2,
            coord=reordered[0],
            elem=reordered[1],
            surf=reordered[2],
            q_mask=reordered[3],
            elem_type=cfg.problem.elem_type,
            material=np.asarray(mesh.material, dtype=np.int64),
        )

    if case == "2d_sloan2013_seepage":
        mesh = generate_sloan2013_mesh_2d(elem_type=cfg.problem.elem_type)
        return _build_case_mesh(
            dim=2,
            coord=np.asarray(mesh.coord, dtype=np.float64),
            elem=np.asarray(mesh.elem, dtype=np.int64),
            surf=np.asarray(mesh.surf, dtype=np.int64),
            q_mask=np.asarray(mesh.q_mask, dtype=bool),
            elem_type=cfg.problem.elem_type,
            material=np.asarray(mesh.material, dtype=np.int64),
        )

    if case in {"2d_kozinec_ssr", "2d_kozinec_ll", "2d_luzec_ssr", "2d_franz_dam_ssr"}:
        mesh_dir = cfg.case_data["mesh_dir"]
        if case.startswith("2d_kozinec"):
            mesh = load_mesh_kozinec_2d(cfg.problem.elem_type, mesh_dir)
        elif case == "2d_luzec_ssr":
            mesh = load_mesh_luzec_2d(cfg.problem.elem_type, mesh_dir)
        else:
            mesh = load_mesh_franz_dam_2d(cfg.problem.elem_type, mesh_dir)
        reordered = _maybe_reorder(mesh.coord, mesh.elem, mesh.surf, mesh.q_mask, cfg, part_count)
        return _build_case_mesh(
            dim=2,
            coord=reordered[0],
            elem=reordered[1],
            surf=reordered[2],
            q_mask=reordered[3],
            elem_type=cfg.problem.elem_type,
            material=np.asarray(mesh.material, dtype=np.int64),
        )

    if case in {"3d_homo_ssr", "3d_hetero_ssr", "3d_siopt_ssr"}:
        mesh = load_mesh_from_file(
            cfg.problem.mesh_path,
            boundary_type=cfg.problem.mesh_boundary_type,
            elem_type=cfg.problem.elem_type,
        )
        reordered = _maybe_reorder(mesh.coord, mesh.elem, mesh.surf, mesh.q_mask, cfg, part_count)
        return _build_case_mesh(
            dim=3,
            coord=reordered[0],
            elem=reordered[1],
            surf=reordered[2],
            q_mask=reordered[3],
            elem_type=cfg.problem.elem_type,
            material=np.asarray(mesh.material, dtype=np.int64),
        )

    if case == "3d_hetero_seepage":
        mesh = load_mesh_gmsh_waterlevels(cfg.problem.mesh_path)
        return _build_case_mesh(
            dim=3,
            coord=np.asarray(mesh.coord, dtype=np.float64),
            elem=np.asarray(mesh.elem, dtype=np.int64),
            surf=np.asarray(mesh.surf, dtype=np.int64),
            q_mask=np.asarray(mesh.q_mask, dtype=bool),
            elem_type=cfg.problem.elem_type,
            material=np.asarray(mesh.material, dtype=np.int64),
        )

    if case in {"3d_hetero_seepage_ssr_comsol", "3d_homo_seepage_ssr", "3d_concave_seepage_ssr"}:
        mesh = load_mesh_p2_comsol(cfg.problem.mesh_path, boundary_type=1)
        reordered = _maybe_reorder(mesh.coord, mesh.elem, mesh.surf, mesh.q_mask, cfg, part_count)
        return _build_case_mesh(
            dim=3,
            coord=reordered[0],
            elem=reordered[1],
            surf=reordered[2],
            q_mask=reordered[3],
            elem_type=cfg.problem.elem_type,
            material=np.asarray(mesh.material, dtype=np.int64),
        )

    raise KeyError(f"No mesh reconstruction registered for case {case!r}")


def _maybe_reorder(
    coord: np.ndarray,
    elem: np.ndarray,
    surf: np.ndarray | None,
    q_mask: np.ndarray | None,
    cfg: RunCaseConfig,
    part_count: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    reordered = reorder_mesh_nodes(
        coord,
        elem,
        surf,
        q_mask,
        strategy=cfg.execution.node_ordering,
        n_parts=part_count,
    )
    return (
        np.asarray(reordered.coord, dtype=np.float64),
        np.asarray(reordered.elem, dtype=np.int64),
        None if reordered.surf is None else np.asarray(reordered.surf, dtype=np.int64),
        None if reordered.q_mask is None else np.asarray(reordered.q_mask, dtype=bool),
    )


def _build_case_mesh(
    *,
    dim: int,
    coord: np.ndarray,
    elem: np.ndarray,
    surf: np.ndarray | None,
    q_mask: np.ndarray | None,
    elem_type: str,
    material: np.ndarray,
) -> CaseMesh:
    cell_type, cells = simplex_vtk_cell_block(dim, elem, elem_type)
    points = _points_2d(coord) if dim == 2 else coord.T
    return CaseMesh(
        dim=int(dim),
        coord=np.asarray(coord, dtype=np.float64),
        elem=np.asarray(elem, dtype=np.int64),
        surf=None if surf is None else np.asarray(surf, dtype=np.int64),
        q_mask=None if q_mask is None else np.asarray(q_mask, dtype=bool),
        material_id=np.asarray(material, dtype=np.int64),
        points=np.asarray(points, dtype=np.float64),
        cell_blocks=[(cell_type, np.asarray(cells, dtype=np.int64))],
    )


def _points_2d(coord: np.ndarray) -> np.ndarray:
    pts = np.zeros((coord.shape[1], 3), dtype=np.float64)
    pts[:, :2] = coord.T
    return pts
