"""Mesh IO helpers for project mesh formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np

from .core.elements import infer_simplex_elem_type, SIMPLEX_NODES_PER_SURFACE
from .core.simplex_lagrange import triangle_lagrange_interior_tuples
from .problem_assets import load_problem_asset_definition_for_path


@dataclass
class MeshData:
    coord: np.ndarray
    elem: np.ndarray
    surf: np.ndarray
    q_mask: np.ndarray
    material: np.ndarray
    boundary: np.ndarray
    elem_type: str | None = None


_AXES_BY_DIM: dict[int, tuple[str, ...]] = {
    2: ("x", "y"),
    3: ("x", "y", "z"),
}


def _default_dirichlet_labels(dim: int) -> dict[str, tuple[int, ...]]:
    axes = _AXES_BY_DIM[int(dim)]
    if int(dim) == 2:
        return {"x": (1, 2), "y": (3,)}
    if int(dim) == 3:
        return {"x": (1, 2), "y": (5,), "z": (3, 4)}
    raise ValueError(f"Unsupported dimension {dim}.")


def _dirichlet_labels_for_path(path: Path, dim: int) -> dict[str, tuple[int, ...]]:
    asset = load_problem_asset_definition_for_path(path)
    if asset is None:
        return _default_dirichlet_labels(dim)

    raw = asset.payload.get("dirichlet_labels")
    if not isinstance(raw, dict):
        return _default_dirichlet_labels(dim)

    labels = _default_dirichlet_labels(dim)
    for axis in _AXES_BY_DIM[int(dim)]:
        value = raw.get(axis)
        if value is None:
            continue
        labels[axis] = tuple(int(v) for v in value)
    return labels


def _build_dirichlet_mask(
    dim: int,
    n_nodes: int,
    surf: np.ndarray,
    boundary: np.ndarray,
    *,
    path: Path,
    boundary_type: int = 0,
) -> np.ndarray:
    face = np.asarray(surf, dtype=np.int64)
    labels = np.asarray(boundary, dtype=np.int64).ravel()
    q = np.ones((int(dim), int(n_nodes)), dtype=bool)
    axis_to_labels = _dirichlet_labels_for_path(path, dim)
    axis_names = _AXES_BY_DIM[int(dim)]

    for axis_idx, axis_name in enumerate(axis_names):
        constrained = tuple(int(v) for v in axis_to_labels.get(axis_name, ()))
        if not constrained or face.size == 0:
            continue
        mask = np.isin(labels, np.asarray(constrained, dtype=np.int64))
        if np.any(mask):
            q[axis_idx, face[:, mask].ravel()] = False

    # Replicate the MATLAB tetrahedral mesh loader convention: when boundary_type
    # is enabled, the family-local "bottom" labels (carried on the y-axis label set
    # for 3D slope meshes) are glued in all displacement components.
    if int(dim) == 3 and int(boundary_type) and face.size:
        glued = tuple(int(v) for v in axis_to_labels.get("y", ()))
        if glued:
            mask = np.isin(labels, np.asarray(glued, dtype=np.int64))
            if np.any(mask):
                q[:, face[:, mask].ravel()] = False
    return q


def _to_zero_based(indices: np.ndarray) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64)
    return arr


def _orient_connectivity(connectivity: np.ndarray, valid_nodes_per_entity: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(connectivity, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D connectivity array, got shape {arr.shape}.")
    valid = {int(v) for v in valid_nodes_per_entity}
    if arr.shape[0] in valid and arr.shape[1] not in valid:
        return arr
    if arr.shape[1] in valid and arr.shape[0] not in valid:
        return arr.T
    if arr.shape[0] in valid:
        return arr
    if arr.shape[1] in valid:
        return arr.T
    raise ValueError(
        f"Cannot orient connectivity array of shape {arr.shape}; "
        f"expected one dimension in {tuple(sorted(valid))}."
    )


def _load_lagrange_tet_mesh(path: Path, *, boundary_type: int = 0) -> MeshData:
    """Load a MATLAB-exported HDF5 tetrahedral mesh for P1/P2/P4 families."""

    with h5py.File(str(path), "r") as h5:
        boundary = np.asarray(h5["boundary"][:], dtype=np.int64).ravel()
        elem = _orient_connectivity(_to_zero_based(np.asarray(h5["elem"][:])), (4, 10, 35))
        face = _orient_connectivity(_to_zero_based(np.asarray(h5["face"][:])), (3, 6, 15))
        material = np.asarray(h5["material"][:], dtype=np.int64).ravel()
        node = np.asarray(h5["node"][:], dtype=np.float64)
    if node.ndim == 2 and node.shape[1] < node.shape[0]:
        node = node.T
    elem_type = infer_simplex_elem_type(3, int(elem.shape[0]))
    expected_face_nodes = SIMPLEX_NODES_PER_SURFACE[(3, elem_type)]
    if face.ndim == 2 and face.shape[0] != expected_face_nodes and face.shape[1] == expected_face_nodes:
        face = face.T
    if face.ndim != 2 or face.shape[0] != expected_face_nodes:
        raise ValueError(
            f"Unexpected face connectivity shape {face.shape} for 3D {elem_type} mesh; "
            f"expected {expected_face_nodes} nodes per face."
        )

    # MATLAB exports stored as (x, z, y) in this helper.
    coord = np.asarray(node[[0, 2, 1], :], dtype=np.float64)
    q = _build_dirichlet_mask(3, coord.shape[1], face, boundary, path=path, boundary_type=boundary_type)

    return MeshData(
        coord=coord,
        elem=elem,
        surf=face,
        q_mask=q,
        material=material,
        boundary=boundary,
        elem_type=elem_type,
    )


def load_mesh_p2(file_path: str | Path, boundary_type: int = 0) -> MeshData:
    """Backwards-compatible name for MATLAB HDF5 tetrahedral meshes."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return _load_lagrange_tet_mesh(path, boundary_type=boundary_type)


def _physical_group_name_map(field_data: dict[str, np.ndarray], dim: int, prefix: str) -> dict[int, int]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    mapping: dict[int, int] = {}
    for name, meta in field_data.items():
        arr = np.asarray(meta, dtype=np.int64).ravel()
        if arr.size < 2 or int(arr[1]) != int(dim):
            continue
        match = pattern.match(str(name).strip().lower())
        if match is None:
            continue
        mapping[int(arr[0])] = int(match.group(1))
    return mapping


def _map_physical_ids(physical_ids: np.ndarray, field_data: dict[str, np.ndarray], dim: int, prefix: str) -> np.ndarray:
    ids = np.asarray(physical_ids, dtype=np.int64).ravel()
    mapping = _physical_group_name_map(field_data, dim, prefix)
    if not mapping:
        return ids
    missing = sorted(int(v) for v in np.unique(ids) if int(v) not in mapping)
    if missing:
        raise ValueError(
            f"Physical group mapping for prefix {prefix!r} is incomplete; missing logical ids for physical tags {missing}."
        )
    return np.asarray([mapping[int(v)] for v in ids], dtype=np.int64)


def _collect_meshio_blocks(mesh, cell_type: str) -> tuple[np.ndarray, np.ndarray]:
    physical_blocks = mesh.cell_data.get("gmsh:physical")
    if physical_blocks is None:
        raise ValueError("Gmsh mesh must carry 'gmsh:physical' cell data.")

    cells_out: list[np.ndarray] = []
    tags_out: list[np.ndarray] = []
    for block, physical in zip(mesh.cells, physical_blocks, strict=False):
        if str(block.type) != str(cell_type):
            continue
        cell_arr = np.asarray(block.data, dtype=np.int64)
        physical_arr = np.asarray(physical, dtype=np.int64).ravel()
        if cell_arr.shape[0] != physical_arr.size:
            raise ValueError(
                f"Cell-data size mismatch for {cell_type}: cells {cell_arr.shape[0]}, physical tags {physical_arr.size}."
            )
        cells_out.append(cell_arr)
        tags_out.append(physical_arr)

    if not cells_out:
        return np.empty((0, 0), dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.vstack(cells_out), np.concatenate(tags_out)


def _midpoint_node_index(
    coord: np.ndarray,
    edge_map: dict[tuple[int, int], int],
    extra_points: list[np.ndarray],
    a: int,
    b: int,
) -> int:
    i = int(a)
    j = int(b)
    key = (i, j) if i < j else (j, i)
    idx = edge_map.get(key)
    if idx is not None:
        return idx
    idx = int(coord.shape[1] + len(extra_points))
    edge_map[key] = idx
    extra_points.append(0.5 * (coord[:, key[0]] + coord[:, key[1]]))
    return idx


def _elevate_tet4_mesh_to_tet10(
    coord: np.ndarray,
    elem: np.ndarray,
    surf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_arr = np.asarray(coord, dtype=np.float64)
    tet4 = np.asarray(elem, dtype=np.int64)
    tri3 = np.asarray(surf, dtype=np.int64)
    if tet4.shape[0] != 4:
        raise ValueError(f"tet4 elevation expects 4-node tetrahedra, got shape {tet4.shape}.")
    if tri3.size and tri3.shape[0] != 3:
        raise ValueError(f"tet4 elevation expects 3-node surface triangles, got shape {tri3.shape}.")

    edge_map: dict[tuple[int, int], int] = {}
    extra_points: list[np.ndarray] = []

    tet10 = np.empty((10, tet4.shape[1]), dtype=np.int64)
    tet10[:4, :] = tet4
    for idx in range(tet4.shape[1]):
        v0, v1, v2, v3 = (int(v) for v in tet4[:, idx])
        tet10[4, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v0, v1)
        tet10[5, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v1, v2)
        tet10[6, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v0, v2)
        tet10[7, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v1, v3)
        tet10[8, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v2, v3)
        tet10[9, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v0, v3)

    tri6 = np.empty((6, tri3.shape[1]), dtype=np.int64)
    if tri3.shape[1]:
        tri6[:3, :] = tri3
        for idx in range(tri3.shape[1]):
            v0, v1, v2 = (int(v) for v in tri3[:, idx])
            tri6[3, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v0, v1)
            tri6[4, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v1, v2)
            tri6[5, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v0, v2)

    if extra_points:
        coord_new = np.hstack((coord_arr, np.column_stack(extra_points)))
    else:
        coord_new = coord_arr.copy()
    return coord_new, tet10, tri6


def _edge_lagrange_node_indices(
    coord: np.ndarray,
    edge_map: dict[tuple[int, int], tuple[int, ...]],
    extra_points: list[np.ndarray],
    a: int,
    b: int,
    *,
    order: int,
) -> tuple[int, ...]:
    i = int(a)
    j = int(b)
    key = (i, j) if i < j else (j, i)
    stored = edge_map.get(key)
    if stored is None:
        lo, hi = key
        ids: list[int] = []
        for step in range(1, int(order)):
            idx = int(coord.shape[1] + len(extra_points))
            alpha = float(int(order) - step) / float(order)
            beta = float(step) / float(order)
            extra_points.append(alpha * coord[:, lo] + beta * coord[:, hi])
            ids.append(idx)
        stored = tuple(ids)
        edge_map[key] = stored
    if (i, j) == key:
        return stored
    return tuple(reversed(stored))


def _face_interior_node_indices(
    coord: np.ndarray,
    face_map: dict[tuple[int, int, int], dict[tuple[int, int, int], int]],
    extra_points: list[np.ndarray],
    verts: tuple[int, int, int],
    *,
    order: int,
) -> tuple[int, ...]:
    local_verts = tuple(int(v) for v in verts)
    canonical = tuple(sorted(local_verts))
    stored = face_map.get(canonical)
    if stored is None:
        stored = {}
        for tri_counts in triangle_lagrange_interior_tuples(int(order)):
            point = np.zeros(coord.shape[0], dtype=np.float64)
            for count, node in zip(tri_counts, canonical, strict=False):
                point += (float(count) / float(order)) * coord[:, int(node)]
            idx = int(coord.shape[1] + len(extra_points))
            extra_points.append(point)
            stored[tuple(int(v) for v in tri_counts)] = idx
        face_map[canonical] = stored

    local_to_canonical = [canonical.index(v) for v in local_verts]
    out: list[int] = []
    for tri_counts in triangle_lagrange_interior_tuples(int(order)):
        canonical_counts = [0, 0, 0]
        for local_idx, canonical_idx in enumerate(local_to_canonical):
            canonical_counts[canonical_idx] = int(tri_counts[local_idx])
        out.append(int(stored[tuple(canonical_counts)]))
    return tuple(out)


def _elevate_tet4_mesh_to_tet35(
    coord: np.ndarray,
    elem: np.ndarray,
    surf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_arr = np.asarray(coord, dtype=np.float64)
    tet4 = np.asarray(elem, dtype=np.int64)
    tri3 = np.asarray(surf, dtype=np.int64)
    if tet4.shape[0] != 4:
        raise ValueError(f"tet35 elevation expects 4-node tetrahedra, got shape {tet4.shape}.")
    if tri3.size and tri3.shape[0] != 3:
        raise ValueError(f"tet35 elevation expects 3-node surface triangles, got shape {tri3.shape}.")

    edge_map: dict[tuple[int, int], tuple[int, ...]] = {}
    face_map: dict[tuple[int, int, int], dict[tuple[int, int, int], int]] = {}
    extra_points: list[np.ndarray] = []

    tet35 = np.empty((35, tet4.shape[1]), dtype=np.int64)
    tet35[:4, :] = tet4
    for idx in range(tet4.shape[1]):
        v0, v1, v2, v3 = (int(v) for v in tet4[:, idx])

        tet35[4:7, idx] = _edge_lagrange_node_indices(coord_arr, edge_map, extra_points, v0, v1, order=4)
        tet35[7:10, idx] = _edge_lagrange_node_indices(coord_arr, edge_map, extra_points, v1, v2, order=4)
        tet35[10:13, idx] = _edge_lagrange_node_indices(coord_arr, edge_map, extra_points, v0, v2, order=4)
        tet35[13:16, idx] = _edge_lagrange_node_indices(coord_arr, edge_map, extra_points, v1, v3, order=4)
        tet35[16:19, idx] = _edge_lagrange_node_indices(coord_arr, edge_map, extra_points, v2, v3, order=4)
        tet35[19:22, idx] = _edge_lagrange_node_indices(coord_arr, edge_map, extra_points, v0, v3, order=4)

        faces = (
            (v0, v1, v2),
            (v0, v1, v3),
            (v0, v2, v3),
            (v1, v2, v3),
        )
        cursor = 22
        for face in faces:
            interior = _face_interior_node_indices(coord_arr, face_map, extra_points, face, order=4)
            tet35[cursor : cursor + len(interior), idx] = interior
            cursor += len(interior)

        centroid_idx = int(coord_arr.shape[1] + len(extra_points))
        extra_points.append(0.25 * (coord_arr[:, v0] + coord_arr[:, v1] + coord_arr[:, v2] + coord_arr[:, v3]))
        tet35[34, idx] = centroid_idx

    tri15 = np.empty((15, tri3.shape[1]), dtype=np.int64)
    if tri3.shape[1]:
        tri15[:3, :] = tri3
        for idx in range(tri3.shape[1]):
            v0, v1, v2 = (int(v) for v in tri3[:, idx])
            tri15[3:6, idx] = _edge_lagrange_node_indices(coord_arr, edge_map, extra_points, v0, v1, order=4)
            tri15[6:9, idx] = _edge_lagrange_node_indices(coord_arr, edge_map, extra_points, v1, v2, order=4)
            tri15[9:12, idx] = _edge_lagrange_node_indices(coord_arr, edge_map, extra_points, v0, v2, order=4)
            tri15[12:15, idx] = _face_interior_node_indices(coord_arr, face_map, extra_points, (v0, v1, v2), order=4)

    if extra_points:
        coord_new = np.hstack((coord_arr, np.column_stack(extra_points)))
    else:
        coord_new = coord_arr.copy()
    return coord_new, tet35, tri15


def _load_gmsh_simplex_mesh(path: Path, *, elem_type: str | None = None, boundary_type: int = 0) -> MeshData:
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - runtime dependency in normal use
        raise ImportError("Reading .msh files requires the 'meshio' package.") from exc

    msh = meshio.read(path)
    points = np.asarray(msh.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected 3D point coordinates in {path}, got shape {points.shape}.")

    tetra_cells, tetra_tags = _collect_meshio_blocks(msh, "tetra")
    tri_cells, tri_tags = _collect_meshio_blocks(msh, "triangle")
    if tetra_cells.size == 0:
        raise ValueError(f"No tetrahedral cells found in {path}.")

    coord = points[:, :3].T.copy()
    material = _map_physical_ids(tetra_tags, msh.field_data, 3, "material")
    boundary = _map_physical_ids(tri_tags, msh.field_data, 2, "boundary")
    elem = np.asarray(tetra_cells.T, dtype=np.int64)
    surf = np.asarray(tri_cells.T, dtype=np.int64) if tri_cells.size else np.empty((3, 0), dtype=np.int64)

    target = None if elem_type is None else str(elem_type).strip().upper()
    if target in {None, "", "P1"}:
        q_mask = _build_dirichlet_mask(3, coord.shape[1], surf, boundary, path=path, boundary_type=boundary_type)
        return MeshData(
            coord=coord,
            elem=elem,
            surf=surf,
            q_mask=q_mask,
            material=material,
            boundary=boundary,
            elem_type="P1",
        )
    if target == "P2":
        coord_p2, elem_p2, surf_p2 = _elevate_tet4_mesh_to_tet10(coord, elem, surf)
        q_mask = _build_dirichlet_mask(3, coord_p2.shape[1], surf_p2, boundary, path=path, boundary_type=boundary_type)
        return MeshData(
            coord=coord_p2,
            elem=elem_p2,
            surf=surf_p2,
            q_mask=q_mask,
            material=material,
            boundary=boundary,
            elem_type="P2",
        )
    if target == "P4":
        coord_p4, elem_p4, surf_p4 = _elevate_tet4_mesh_to_tet35(coord, elem, surf)
        q_mask = _build_dirichlet_mask(3, coord_p4.shape[1], surf_p4, boundary, path=path, boundary_type=boundary_type)
        return MeshData(
            coord=coord_p4,
            elem=elem_p4,
            surf=surf_p4,
            q_mask=q_mask,
            material=material,
            boundary=boundary,
            elem_type="P4",
        )
    raise NotImplementedError(
        f"Gmsh simplex loader currently supports P1 source meshes elevated to P2/P4; requested {target!r}."
    )


def load_mesh_file(mesh_file: str | Path, *, elem_type: str | None = None, boundary_type: int = 0) -> MeshData:
    path = Path(mesh_file)
    lower = path.name.lower()
    if path.suffix.lower() == ".h5":
        with h5py.File(str(path), "r") as h5:
            keys = set(h5.keys())
        if {"boundary", "elem", "face", "material", "node"} <= keys:
            return _load_lagrange_tet_mesh(path, boundary_type=boundary_type)
    if path.suffix.lower() == ".msh":
        return _load_gmsh_simplex_mesh(path, elem_type=elem_type, boundary_type=boundary_type)
    if "p2" in lower:
        return load_mesh_p2(path, boundary_type=boundary_type)
    raise ValueError(f"Unsupported mesh format for {path}")
