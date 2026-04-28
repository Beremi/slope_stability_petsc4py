"""Shared helpers for canonical mesh conversion scripts."""

from __future__ import annotations

from pathlib import Path
import tempfile

import gmsh
import meshio
import numpy as np


def ensure_2d_points(coord: np.ndarray) -> np.ndarray:
    coord_arr = np.asarray(coord, dtype=np.float64)
    if coord_arr.shape[0] != 2:
        raise ValueError(f"Expected 2D coordinates with shape (2, n), got {coord_arr.shape}.")
    out = np.zeros((coord_arr.shape[1], 3), dtype=np.float64)
    out[:, :2] = coord_arr.T
    return out


def ensure_3d_points(coord: np.ndarray) -> np.ndarray:
    coord_arr = np.asarray(coord, dtype=np.float64)
    if coord_arr.shape[0] != 3:
        raise ValueError(f"Expected 3D coordinates with shape (3, n), got {coord_arr.shape}.")
    return coord_arr.T.copy()


def orient_triangles_positive(coord: np.ndarray, elem: np.ndarray) -> np.ndarray:
    coord_arr = np.asarray(coord, dtype=np.float64)
    tri = np.asarray(elem, dtype=np.int64).copy()
    for idx in range(tri.shape[1]):
        a, b, c = tri[:, idx]
        area2 = np.cross(coord_arr[:, b] - coord_arr[:, a], coord_arr[:, c] - coord_arr[:, a])
        if float(area2) < 0.0:
            tri[[1, 2], idx] = tri[[2, 1], idx]
    return tri


def orient_tetrahedra_positive(coord: np.ndarray, elem: np.ndarray) -> np.ndarray:
    coord_arr = np.asarray(coord, dtype=np.float64)
    tet = np.asarray(elem, dtype=np.int64).copy()
    for idx in range(tet.shape[1]):
        a, b, c, d = tet[:, idx]
        J = np.column_stack((coord_arr[:, b] - coord_arr[:, a], coord_arr[:, c] - coord_arr[:, a], coord_arr[:, d] - coord_arr[:, a]))
        if float(np.linalg.det(J)) < 0.0:
            tet[[0, 1], idx] = tet[[1, 0], idx]
    return tet


def write_canonical_msh(
    *,
    out_path: str | Path,
    points: np.ndarray,
    volume_type: str,
    volume_cells: np.ndarray,
    region_by_entity: list[str],
    boundary_type: str | None = None,
    boundary_cells: np.ndarray | None = None,
    boundary_by_entity: list[str] | None = None,
    nodesets: dict[str, np.ndarray] | None = None,
    boundary_geom_type: str | None = None,
    boundary_geom_cells: np.ndarray | None = None,
    boundary_geom_by_entity: list[str] | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    points_arr = np.asarray(points, dtype=np.float64)
    cells: list[tuple[str, np.ndarray]] = []
    physical_blocks: list[np.ndarray] = []
    geometrical_blocks: list[np.ndarray] = []
    field_data: dict[str, np.ndarray] = {}

    def add_block(*, cell_type: str, data: np.ndarray, names: list[str], prefix: str, dim: int) -> None:
        if data.size == 0:
            return
        tag_by_name: dict[str, int] = {}
        tags = np.empty(len(names), dtype=np.int32)
        for idx, name in enumerate(names):
            tag = tag_by_name.get(name)
            if tag is None:
                tag = len(tag_by_name) + 1
                tag_by_name[name] = tag
                field_data[f"{prefix}:{name}"] = np.asarray([tag, dim], dtype=np.int32)
            tags[idx] = tag
        cells.append((cell_type, np.asarray(data, dtype=np.int64)))
        physical_blocks.append(tags)
        geometrical_blocks.append(tags.copy())

    dim = 2 if volume_type == "triangle" else 3
    add_block(cell_type=volume_type, data=np.asarray(volume_cells, dtype=np.int64).T, names=list(region_by_entity), prefix="region", dim=dim)
    if boundary_type is not None and boundary_cells is not None and boundary_by_entity is not None:
        add_block(
            cell_type=boundary_type,
            data=np.asarray(boundary_cells, dtype=np.int64).T,
            names=list(boundary_by_entity),
            prefix="boundary",
            dim=dim - 1,
        )
    if nodesets:
        vertex_cells: list[list[int]] = []
        vertex_names: list[str] = []
        for name, nodes in nodesets.items():
            for node in np.asarray(nodes, dtype=np.int64).ravel():
                vertex_cells.append([int(node)])
                vertex_names.append(str(name))
        if vertex_cells:
            add_block(
                cell_type="vertex",
                data=np.asarray(vertex_cells, dtype=np.int64),
                names=vertex_names,
                prefix="nodeset",
                dim=0,
            )
    if boundary_geom_type is not None and boundary_geom_cells is not None and boundary_geom_by_entity is not None:
        add_block(
            cell_type=boundary_geom_type,
            data=np.asarray(boundary_geom_cells, dtype=np.int64).T,
            names=list(boundary_geom_by_entity),
            prefix="boundary_geom",
            dim=dim - 1,
        )

    mesh = meshio.Mesh(
        points=points_arr,
        cells=cells,
        cell_data={
            "gmsh:physical": physical_blocks,
            "gmsh:geometrical": geometrical_blocks,
        },
        field_data=field_data,
    )

    with tempfile.TemporaryDirectory(prefix="canonical_msh_") as tmpdir:
        tmp_path = Path(tmpdir) / f"{out_path.stem}_legacy22.msh"
        meshio.write(tmp_path, mesh, file_format="gmsh22")
        gmsh.initialize()
        try:
            gmsh.open(str(tmp_path))
            gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
            gmsh.write(str(out_path))
        finally:
            gmsh.finalize()
