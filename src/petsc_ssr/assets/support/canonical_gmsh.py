"""Canonical Gmsh mesh loading and solver-mesh promotion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...core.simplex_lagrange import triangle_lagrange_node_tuples
from ...io import (
    _collect_meshio_blocks,
    _elevate_tet4_mesh_to_tet10,
    _elevate_tet4_mesh_to_tet20,
    _elevate_tet4_mesh_to_tet35,
)
from ..api import BoundaryGeometryPatch, CanonicalMesh, SolverMesh


def gmsh_variants_from_dir(asset_dir: Path, pattern: str = "*.msh") -> dict[str, dict]:
    return {path.name: {"source": {"path": path.name}} for path in sorted(asset_dir.glob(pattern))}


def _edge_node_triplet(
    *,
    coord: np.ndarray,
    key: tuple[int, int],
    edge_cache: dict[tuple[int, int], tuple[int, int, int]],
    next_node: int,
) -> tuple[tuple[int, int, int], int, np.ndarray]:
    existing = edge_cache.get(key)
    if existing is not None:
        return existing, next_node, coord

    a, b = key
    midpoint = next_node
    quarter_a = next_node + 1
    quarter_b = next_node + 2
    next_node += 3
    new_points = np.column_stack(
        (
            (coord[:, a] + coord[:, b]) / 2.0,
            0.75 * coord[:, a] + 0.25 * coord[:, b],
            0.25 * coord[:, a] + 0.75 * coord[:, b],
        )
    )
    coord = np.hstack((coord, new_points))
    nodes = (midpoint, quarter_a, quarter_b)
    edge_cache[key] = nodes
    return nodes, next_node, coord


@dataclass(frozen=True)
class _NamedEntities:
    connectivity: np.ndarray
    names: tuple[str, ...]


def _normalize_name(value: str, *, prefix: str) -> str | None:
    text = str(value).strip()
    lower = text.lower()
    if lower.startswith(f"{prefix}:"):
        return text.split(":", 1)[1]
    legacy_prefix = {
        "region": "material_",
        "boundary": "boundary_",
        "nodeset": "nodeset_",
        "boundary_geom": "boundary_geom_",
    }[prefix]
    if ":" not in text and lower.startswith(legacy_prefix):
        return text.split("_", 1)[1]
    return None


def _field_name_map(field_data: dict[str, np.ndarray], *, prefix: str, dim: int) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for name, meta in field_data.items():
        arr = np.asarray(meta, dtype=np.int64).ravel()
        if arr.size < 2 or int(arr[1]) != int(dim):
            continue
        logical = _normalize_name(str(name), prefix=prefix)
        if logical is not None:
            mapping[int(arr[0])] = logical
    return mapping


def _named_blocks(mesh, cell_type: str, *, tag_map: dict[int, str]) -> _NamedEntities:
    cells, tags = _collect_meshio_blocks(mesh, cell_type)
    if cells.size == 0:
        width = {"vertex": 1, "line": 2, "line3": 3, "triangle": 3, "triangle6": 6, "tetra": 4}.get(cell_type, 0)
        return _NamedEntities(np.empty((0, width), dtype=np.int64), ())
    missing = sorted(int(v) for v in np.unique(tags) if int(v) not in tag_map)
    if missing:
        raise ValueError(f"Missing logical names for {cell_type} physical tags {missing}.")
    names = tuple(tag_map[int(tag)] for tag in np.asarray(tags, dtype=np.int64).ravel())
    return _NamedEntities(np.asarray(cells, dtype=np.int64), names)


def _compact_base_nodes(
    points: np.ndarray,
    *connectivity_blocks: np.ndarray,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    used_parts = [np.asarray(block, dtype=np.int64).reshape(-1) for block in connectivity_blocks if np.asarray(block).size]
    if not used_parts:
        raise ValueError("Canonical mesh has no nodes referenced by volume/boundary/node-set entities.")
    used = np.unique(np.concatenate(used_parts)).astype(np.int64)
    old_to_new = np.full(points.shape[0], -1, dtype=np.int64)
    old_to_new[used] = np.arange(used.size, dtype=np.int64)
    coord = np.asarray(points[used, :dim], dtype=np.float64).T
    return coord, old_to_new


def load_canonical_gmsh_mesh(path: str | Path, *, dimension: int) -> CanonicalMesh:
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - runtime dependency in normal use
        raise ImportError("Reading .msh files requires the 'meshio' package.") from exc

    path = Path(path)
    msh = meshio.read(path)
    points = np.asarray(msh.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < int(dimension):
        raise ValueError(f"Expected at least {dimension}D point coordinates in {path}, got shape {points.shape}.")

    volume_type = "triangle" if int(dimension) == 2 else "tetra"
    boundary_cell_type = "line" if int(dimension) == 2 else "triangle"
    geom_type = "line3" if int(dimension) == 2 else "triangle6"
    region_map = _field_name_map(msh.field_data, prefix="region", dim=int(dimension))
    if int(dimension) == 3 and not region_map:
        region_map = _field_name_map(msh.field_data, prefix="material", dim=3)
    boundary_map = _field_name_map(msh.field_data, prefix="boundary", dim=int(dimension) - 1)
    nodeset_map = _field_name_map(msh.field_data, prefix="nodeset", dim=0)
    geom_map = _field_name_map(msh.field_data, prefix="boundary_geom", dim=int(dimension) - 1)

    volume = _named_blocks(msh, volume_type, tag_map=region_map)
    boundary = _named_blocks(msh, boundary_cell_type, tag_map=boundary_map)
    nodesets = _named_blocks(msh, "vertex", tag_map=nodeset_map) if nodeset_map else _NamedEntities(
        np.empty((0, 1), dtype=np.int64), ()
    )
    geom = _named_blocks(msh, geom_type, tag_map=geom_map) if geom_map else _NamedEntities(
        np.empty((0, 3 if int(dimension) == 2 else 6), dtype=np.int64), ()
    )

    coord, old_to_new = _compact_base_nodes(
        points,
        volume.connectivity,
        boundary.connectivity,
        nodesets.connectivity,
        dim=int(dimension),
    )
    elem = old_to_new[np.asarray(volume.connectivity, dtype=np.int64)].T
    surf = old_to_new[np.asarray(boundary.connectivity, dtype=np.int64)].T if boundary.connectivity.size else np.empty(
        (int(dimension), 0), dtype=np.int64
    )

    region_groups: dict[str, np.ndarray] = {}
    for name in dict.fromkeys(volume.names):
        region_groups[str(name)] = np.flatnonzero(np.asarray(volume.names, dtype=object) == str(name)).astype(np.int64)

    boundary_groups: dict[str, np.ndarray] = {}
    for name in dict.fromkeys(boundary.names):
        boundary_groups[str(name)] = np.flatnonzero(np.asarray(boundary.names, dtype=object) == str(name)).astype(np.int64)

    nodeset_groups: dict[str, np.ndarray] = {}
    if nodesets.connectivity.size:
        node_names = np.asarray(nodesets.names, dtype=object)
        node_indices = old_to_new[np.asarray(nodesets.connectivity, dtype=np.int64).ravel()]
        for name in dict.fromkeys(nodesets.names):
            selected = node_indices[node_names == str(name)]
            nodeset_groups[str(name)] = np.unique(selected[selected >= 0]).astype(np.int64)

    boundary_geometry: dict[str, BoundaryGeometryPatch] = {}
    if geom.connectivity.size:
        geom_names = np.asarray(geom.names, dtype=object)
        corners = 2 if int(dimension) == 2 else 3
        for name in dict.fromkeys(geom.names):
            idx = np.flatnonzero(geom_names == str(name)).astype(np.int64)
            conn = np.asarray(geom.connectivity[idx], dtype=np.int64)
            control_points = np.asarray(points[conn, : int(dimension)], dtype=np.float64)
            boundary_geometry[str(name)] = BoundaryGeometryPatch(
                name=str(name),
                cell_type=geom_type,
                corner_nodes=old_to_new[conn[:, :corners]].T,
                control_points=np.transpose(control_points, (2, 1, 0)),
            )

    return CanonicalMesh(
        coord=np.asarray(coord, dtype=np.float64),
        elem=np.asarray(elem, dtype=np.int64),
        surf=np.asarray(surf, dtype=np.int64),
        region_name_by_elem=tuple(str(name) for name in volume.names),
        boundary_name_by_entity=tuple(str(name) for name in boundary.names),
        region_groups=region_groups,
        boundary_groups=boundary_groups,
        nodesets=nodeset_groups,
        boundary_geometry=boundary_geometry,
    )


def _elevate_triangle_mesh_to_tri6(
    coord: np.ndarray,
    elem: np.ndarray,
    surf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_arr = np.asarray(coord, dtype=np.float64)
    tri3 = np.asarray(elem, dtype=np.int64)
    line2 = np.asarray(surf, dtype=np.int64)
    edge_nodes: dict[tuple[int, int], int] = {}
    next_node = int(coord_arr.shape[1])
    coord_out = coord_arr

    def midpoint(a: int, b: int) -> int:
        nonlocal next_node, coord_out
        key = (int(a), int(b))
        sorted_key = tuple(sorted(key))
        idx = edge_nodes.get(sorted_key)
        if idx is None:
            idx = next_node
            next_node += 1
            edge_nodes[sorted_key] = idx
            coord_out = np.hstack((coord_out, (0.5 * (coord_arr[:, sorted_key[0]] + coord_arr[:, sorted_key[1]]))[:, None]))
        return idx

    tri6 = np.empty((6, tri3.shape[1]), dtype=np.int64)
    tri6[:3, :] = tri3
    for idx in range(tri3.shape[1]):
        v0, v1, v2 = (int(v) for v in tri3[:, idx])
        tri6[3, idx] = midpoint(v1, v2)
        tri6[4, idx] = midpoint(v2, v0)
        tri6[5, idx] = midpoint(v0, v1)

    line3 = np.empty((3, line2.shape[1]), dtype=np.int64)
    if line2.shape[1]:
        line3[:2, :] = line2
        for idx in range(line2.shape[1]):
            a, b = (int(v) for v in line2[:, idx])
            line3[2, idx] = midpoint(a, b)
    return coord_out, tri6, line3


def _elevate_triangle_mesh_to_tri15(
    coord: np.ndarray,
    elem: np.ndarray,
    surf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_arr = np.asarray(coord, dtype=np.float64)
    tri3 = np.asarray(elem, dtype=np.int64)
    line2 = np.asarray(surf, dtype=np.int64)
    coord_out = coord_arr
    edge_cache: dict[tuple[int, int], tuple[int, int, int]] = {}
    next_node = int(coord_out.shape[1])

    tri15 = np.empty((15, tri3.shape[1]), dtype=np.int64)
    tri15[:3, :] = tri3
    for idx in range(tri3.shape[1]):
        v0, v1, v2 = (int(v) for v in tri3[:, idx])
        edge_nodes: list[int] = []
        for edge in ((v0, v1), (v1, v2), (v2, v0)):
            key = tuple(sorted(edge))
            (midpoint, quarter_a, quarter_b), next_node, coord_out = _edge_node_triplet(
                coord=coord_out,
                key=key,
                edge_cache=edge_cache,
                next_node=next_node,
            )
            if edge == key:
                edge_nodes.extend((midpoint, quarter_a, quarter_b))
            else:
                edge_nodes.extend((midpoint, quarter_b, quarter_a))

        interior = np.column_stack(
            (
                0.50 * coord_arr[:, v0] + 0.25 * coord_arr[:, v1] + 0.25 * coord_arr[:, v2],
                0.25 * coord_arr[:, v0] + 0.50 * coord_arr[:, v1] + 0.25 * coord_arr[:, v2],
                0.25 * coord_arr[:, v0] + 0.25 * coord_arr[:, v1] + 0.50 * coord_arr[:, v2],
            )
        )
        interior_ids = np.arange(next_node, next_node + 3, dtype=np.int64)
        next_node += 3
        coord_out = np.hstack((coord_out, interior))

        tri15[3:, idx] = np.asarray(
            [
                edge_nodes[0],
                edge_nodes[3],
                edge_nodes[6],
                edge_nodes[1],
                edge_nodes[2],
                edge_nodes[4],
                edge_nodes[5],
                edge_nodes[7],
                edge_nodes[8],
                interior_ids[0],
                interior_ids[1],
                interior_ids[2],
            ],
            dtype=np.int64,
        )

    line5 = np.empty((5, line2.shape[1]), dtype=np.int64)
    if line2.shape[1]:
        line5[:2, :] = line2
        for idx in range(line2.shape[1]):
            a, b = (int(v) for v in line2[:, idx])
            key = tuple(sorted((a, b)))
            midpoint, quarter_a, quarter_b = edge_cache[key]
            if (a, b) == key:
                line5[2:, idx] = np.asarray([midpoint, quarter_a, quarter_b], dtype=np.int64)
            else:
                line5[2:, idx] = np.asarray([midpoint, quarter_b, quarter_a], dtype=np.int64)
    return coord_out, tri15, line5


def _line_positions_for_surf_width(width: int) -> tuple[float, ...]:
    if width == 2:
        return (0.0, 1.0)
    if width == 3:
        return (0.0, 1.0, 0.5)
    if width == 5:
        return (0.0, 1.0, 0.5, 0.25, 0.75)
    raise ValueError(f"Unsupported 2D boundary width {width} for curved-boundary promotion.")


def _apply_curved_line_geometry(
    coord: np.ndarray,
    surf: np.ndarray,
    canonical: CanonicalMesh,
    *,
    geometry_name: str,
    support_boundary: str,
) -> np.ndarray:
    patch = canonical.boundary_geometry.get(geometry_name)
    if patch is None:
        raise ValueError(f"Mesh does not define boundary geometry patch {geometry_name!r}.")
    support = canonical.boundary_groups.get(support_boundary)
    if support is None:
        raise ValueError(f"Mesh does not define support boundary {support_boundary!r}.")
    support_map = {tuple(int(v) for v in canonical.surf[:2, idx]): int(idx) for idx in support}
    out = np.asarray(coord, dtype=np.float64).copy()
    positions = _line_positions_for_surf_width(int(surf.shape[0]))
    for col in range(int(patch.corner_nodes.shape[1])):
        key = tuple(int(v) for v in patch.corner_nodes[:, col])
        support_idx = support_map.get(key)
        if support_idx is None:
            raise ValueError(
                f"Curved boundary geometry {geometry_name!r} has no matching support edge with ordered corners {key}."
            )
        ctrl = patch.control_points[:, :, col]
        for local_idx, t in enumerate(positions):
            N0 = (1.0 - t) * (1.0 - 2.0 * t)
            N1 = t * (2.0 * t - 1.0)
            N2 = 4.0 * t * (1.0 - t)
            out[:, surf[local_idx, support_idx]] = (N0 * ctrl[:, 0]) + (N1 * ctrl[:, 1]) + (N2 * ctrl[:, 2])
    return out


def _apply_curved_triangle_geometry(
    coord: np.ndarray,
    surf: np.ndarray,
    canonical: CanonicalMesh,
    *,
    geometry_name: str,
    support_boundary: str,
    order: int,
) -> np.ndarray:
    patch = canonical.boundary_geometry.get(geometry_name)
    if patch is None:
        raise ValueError(f"Mesh does not define boundary geometry patch {geometry_name!r}.")
    support = canonical.boundary_groups.get(support_boundary)
    if support is None:
        raise ValueError(f"Mesh does not define support boundary {support_boundary!r}.")
    support_map = {tuple(int(v) for v in canonical.surf[:3, idx]): int(idx) for idx in support}
    out = np.asarray(coord, dtype=np.float64).copy()
    tuples = triangle_lagrange_node_tuples(int(order))
    for col in range(int(patch.corner_nodes.shape[1])):
        key = tuple(int(v) for v in patch.corner_nodes[:, col])
        support_idx = support_map.get(key)
        if support_idx is None:
            raise ValueError(
                f"Curved boundary geometry {geometry_name!r} has no matching support face with ordered corners {key}."
            )
        ctrl = patch.control_points[:, :, col]
        for local_idx, counts in enumerate(tuples):
            L0 = float(counts[0]) / float(order)
            L1 = float(counts[1]) / float(order)
            L2 = float(counts[2]) / float(order)
            N = np.array(
                [
                    L0 * (2.0 * L0 - 1.0),
                    L1 * (2.0 * L1 - 1.0),
                    L2 * (2.0 * L2 - 1.0),
                    4.0 * L0 * L1,
                    4.0 * L1 * L2,
                    4.0 * L0 * L2,
                ],
                dtype=np.float64,
            )
            out[:, surf[local_idx, support_idx]] = ctrl @ N
    return out


def build_solver_mesh(
    canonical: CanonicalMesh,
    *,
    elem_type: str,
    boundary_geometry_specs: dict[str, tuple[str, int]] | None = None,
    region_id_by_name: dict[str, int],
    boundary_id_by_name: dict[str, int],
) -> SolverMesh:
    elem_key = str(elem_type).strip().upper()
    dim = int(canonical.coord.shape[0])
    if dim == 2:
        if elem_key == "P1":
            coord = np.asarray(canonical.coord, dtype=np.float64)
            elem = np.asarray(canonical.elem, dtype=np.int64)
            surf = np.asarray(canonical.surf, dtype=np.int64)
        elif elem_key == "P2":
            coord, elem, surf = _elevate_triangle_mesh_to_tri6(canonical.coord, canonical.elem, canonical.surf)
        elif elem_key == "P4":
            coord, elem, surf = _elevate_triangle_mesh_to_tri15(canonical.coord, canonical.elem, canonical.surf)
        else:
            raise ValueError(f"Unsupported 2D element type {elem_type!r}; expected P1, P2, or P4.")
    else:
        if elem_key == "P1":
            coord = np.asarray(canonical.coord, dtype=np.float64)
            elem = np.asarray(canonical.elem, dtype=np.int64)
            surf = np.asarray(canonical.surf, dtype=np.int64)
        elif elem_key == "P2":
            coord, elem, surf = _elevate_tet4_mesh_to_tet10(canonical.coord, canonical.elem, canonical.surf)
        elif elem_key == "P3":
            coord, elem, surf = _elevate_tet4_mesh_to_tet20(canonical.coord, canonical.elem, canonical.surf)
        elif elem_key == "P4":
            coord, elem, surf = _elevate_tet4_mesh_to_tet35(canonical.coord, canonical.elem, canonical.surf)
        else:
            raise ValueError(f"Unsupported 3D element type {elem_type!r}; expected P1, P2, P3, or P4.")

    for geometry_name, (support_boundary, geometry_order) in (boundary_geometry_specs or {}).items():
        if dim == 2:
            coord = _apply_curved_line_geometry(coord, surf, canonical, geometry_name=geometry_name, support_boundary=support_boundary)
        else:
            coord = _apply_curved_triangle_geometry(
                coord,
                surf,
                canonical,
                geometry_name=geometry_name,
                support_boundary=support_boundary,
                order=int(elem_key[1:]),
            )

    material_id = np.asarray([region_id_by_name[name] for name in canonical.region_name_by_elem], dtype=np.int64)
    boundary_labels = np.asarray([boundary_id_by_name[name] for name in canonical.boundary_name_by_entity], dtype=np.int64)
    return SolverMesh(
        coord=np.asarray(coord, dtype=np.float64),
        elem=np.asarray(elem, dtype=np.int64),
        surf=np.asarray(surf, dtype=np.int64),
        q_mask=np.ones((dim, coord.shape[1]), dtype=bool),
        material_id=material_id,
        boundary_labels=boundary_labels,
        elem_type=elem_key,
        region_id_by_name=dict(region_id_by_name),
        boundary_id_by_name=dict(boundary_id_by_name),
        boundary_groups={name: np.asarray(idx, dtype=np.int64) for name, idx in canonical.boundary_groups.items()},
        nodesets={name: np.asarray(nodes, dtype=np.int64) for name, nodes in canonical.nodesets.items()},
        boundary_geometry=dict(canonical.boundary_geometry),
    )
