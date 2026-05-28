"""Case-mesh reconstruction shared by exports and notebook tooling."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from petsc_ssr.core.elements import simplex_vtk_cell_block
from petsc_ssr.core.run_config import RunCaseConfig
from petsc_ssr.mesh import reorder_mesh_nodes
from petsc_ssr.problem_asset_runtime import build_mesh_for_resolved_asset, resolve_problem_asset_from_config


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
    resolved = resolve_problem_asset_from_config(cfg)
    built = build_mesh_for_resolved_asset(resolved, elem_type=cfg.problem.elem_type)
    part_count = int(mpi_size) if cfg.execution.node_ordering.lower() == "block_metis" else None

    coord, elem, surf, q_mask = _maybe_reorder(
        built.coord,
        built.elem,
        built.surf,
        built.q_mask,
        cfg,
        part_count,
    )

    return _build_case_mesh(
        dim=int(resolved.dimension),
        coord=np.asarray(coord, dtype=np.float64),
        elem=np.asarray(elem, dtype=np.int64),
        surf=np.asarray(surf, dtype=np.int64),
        q_mask=np.asarray(q_mask, dtype=bool),
        elem_type=cfg.problem.elem_type,
        material=np.asarray(built.material_id, dtype=np.int64),
    )


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


def validate_case_mesh_alignment(case_mesh: CaseMesh, arrays: Mapping[str, np.ndarray]) -> None:
    coord = arrays.get("coord")
    if coord is not None:
        coord_arr = np.asarray(coord, dtype=np.float64)
        if coord_arr.shape != case_mesh.coord.shape:
            raise ValueError(
                "Export mesh coordinate shape mismatch: "
                f"saved {coord_arr.shape}, rebuilt {case_mesh.coord.shape}."
            )
        if not np.array_equal(coord_arr, case_mesh.coord):
            raise ValueError("Export mesh coordinates do not match the saved solver ordering.")

    elem = arrays.get("elem")
    if elem is not None:
        elem_arr = np.asarray(elem, dtype=np.int64)
        if elem_arr.shape != case_mesh.elem.shape:
            raise ValueError(
                "Export mesh connectivity shape mismatch: "
                f"saved {elem_arr.shape}, rebuilt {case_mesh.elem.shape}."
            )
        if not np.array_equal(elem_arr, case_mesh.elem):
            raise ValueError("Export mesh connectivity does not match the saved solver ordering.")
