"""MATLAB-compatible 2D text-mesh loaders for Kozinec/Luzec/Franz cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TextMesh2D:
    coord: np.ndarray
    elem: np.ndarray
    surf: np.ndarray
    q_mask: np.ndarray
    material: np.ndarray
    boundary: np.ndarray


def _load_base_triangle_mesh(
    coordinates_path: Path,
    elements_path: Path,
    materials_path: Path,
    *,
    shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord = np.loadtxt(coordinates_path, dtype=np.float64).T
    elem = np.loadtxt(elements_path, dtype=np.int64).T
    material = np.loadtxt(materials_path, dtype=np.int64).ravel() - 1
    coord[0, :] += float(shift[0])
    coord[1, :] += float(shift[1])
    return coord, elem, material


def _boundary_edges(elem: np.ndarray) -> tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    counts: dict[tuple[int, int], int] = {}
    oriented: dict[tuple[int, int], tuple[int, int]] = {}
    for tri in np.asarray(elem[:3, :], dtype=np.int64).T:
        local_edges = ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
        for edge in local_edges:
            key = tuple(sorted(edge))
            counts[key] = counts.get(key, 0) + 1
            oriented.setdefault(key, edge)
    boundary_keys = [key for key, count in counts.items() if count == 1]
    return boundary_keys, oriented


def _expand_p2(
    coord: np.ndarray,
    elem: np.ndarray,
    boundary_keys: list[tuple[int, int]],
    oriented_edges: dict[tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_out = np.asarray(coord, dtype=np.float64)
    edge_nodes: dict[tuple[int, int], int] = {}
    next_node = int(coord_out.shape[1])

    boundary_set = set(boundary_keys)
    surf_parts: list[list[int]] = []
    elem_out = np.empty((6, elem.shape[1]), dtype=np.int64)
    elem_out[:3, :] = elem[:3, :]

    for e, tri in enumerate(np.asarray(elem[:3, :], dtype=np.int64).T):
        local_order = ((tri[1], tri[2]), (tri[2], tri[0]), (tri[0], tri[1]))
        edge_midpoints: list[int] = []
        for edge in local_order:
            key = tuple(sorted(edge))
            midpoint = edge_nodes.get(key)
            if midpoint is None:
                midpoint = next_node
                next_node += 1
                edge_nodes[key] = midpoint
                a, b = key
                coord_out = np.hstack((coord_out, ((coord[:, a] + coord[:, b]) / 2.0)[:, None]))
            edge_midpoints.append(midpoint)
        elem_out[3:, e] = np.asarray(edge_midpoints, dtype=np.int64)

    for key in boundary_keys:
        a, b = oriented_edges[key]
        surf_parts.append([a, b, edge_nodes[key]])

    surf = np.asarray(surf_parts, dtype=np.int64).T if surf_parts else np.empty((3, 0), dtype=np.int64)
    return coord_out, elem_out, surf


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


def _expand_p4(
    coord: np.ndarray,
    elem: np.ndarray,
    boundary_keys: list[tuple[int, int]],
    oriented_edges: dict[tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_out = np.asarray(coord, dtype=np.float64)
    edge_cache: dict[tuple[int, int], tuple[int, int, int]] = {}
    next_node = int(coord_out.shape[1])
    surf_parts: list[list[int]] = []
    elem_out = np.empty((15, elem.shape[1]), dtype=np.int64)
    elem_out[:3, :] = elem[:3, :]

    for e, tri in enumerate(np.asarray(elem[:3, :], dtype=np.int64).T):
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
        local_edges = ((v0, v1), (v1, v2), (v2, v0))
        edge_nodes: list[int] = []
        for edge in local_edges:
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
                0.50 * coord[:, v0] + 0.25 * coord[:, v1] + 0.25 * coord[:, v2],
                0.25 * coord[:, v0] + 0.50 * coord[:, v1] + 0.25 * coord[:, v2],
                0.25 * coord[:, v0] + 0.25 * coord[:, v1] + 0.50 * coord[:, v2],
            )
        )
        interior_ids = np.arange(next_node, next_node + 3, dtype=np.int64)
        next_node += 3
        coord_out = np.hstack((coord_out, interior))

        elem_out[3:, e] = np.asarray(
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

    for key in boundary_keys:
        a, b = oriented_edges[key]
        midpoint, quarter_a, quarter_b = edge_cache[key]
        if (a, b) == key:
            surf_parts.append([a, b, midpoint, quarter_a, quarter_b])
        else:
            surf_parts.append([a, b, midpoint, quarter_b, quarter_a])

    surf = np.asarray(surf_parts, dtype=np.int64).T if surf_parts else np.empty((5, 0), dtype=np.int64)
    return coord_out, elem_out, surf


def _boundary_mask(coord: np.ndarray, *, y_floor: float) -> np.ndarray:
    x_max = float(np.max(coord[0, :]))
    x_min = float(np.min(coord[0, :]))
    q_mask = np.zeros((2, coord.shape[1]), dtype=bool)
    q_mask[0, :] = (coord[0, :] > x_min + 0.2) & (coord[1, :] > y_floor + 0.2) & (coord[0, :] < x_max - 0.2)
    q_mask[1, :] = coord[1, :] > y_floor + 0.2
    return q_mask


def _load_text_mesh_case(
    base_dir: Path,
    *,
    elem_type: str,
    coordinates_name: str,
    elements_name: str,
    materials_name: str,
    shift: tuple[float, float] = (0.0, 0.0),
    y_floor: float = 0.0,
) -> TextMesh2D:
    coord, elem, material = _load_base_triangle_mesh(
        base_dir / coordinates_name,
        base_dir / elements_name,
        base_dir / materials_name,
        shift=shift,
    )
    boundary_keys, oriented_edges = _boundary_edges(elem)
    elem_type_up = str(elem_type).upper()

    if elem_type_up == "P1":
        surf_parts = [list(oriented_edges[key]) for key in boundary_keys]
        surf = np.asarray(surf_parts, dtype=np.int64).T if surf_parts else np.empty((2, 0), dtype=np.int64)
        coord_out = coord
        elem_out = elem
    elif elem_type_up == "P2":
        coord_out, elem_out, surf = _expand_p2(coord, elem, boundary_keys, oriented_edges)
    elif elem_type_up == "P4":
        coord_out, elem_out, surf = _expand_p4(coord, elem, boundary_keys, oriented_edges)
    else:
        raise ValueError(f"Unsupported 2D text-mesh element type {elem_type!r}")

    q_mask = _boundary_mask(coord_out, y_floor=y_floor)
    return TextMesh2D(
        coord=coord_out,
        elem=elem_out,
        surf=surf,
        q_mask=q_mask,
        material=material,
        boundary=np.zeros(surf.shape[1], dtype=np.int64),
    )


def load_mesh_kozinec_2d(elem_type: str, base_dir: str | Path) -> TextMesh2D:
    base_dir = Path(base_dir)
    return _load_text_mesh_case(
        base_dir,
        elem_type=elem_type,
        coordinates_name="coordinates3.txt",
        elements_name="elements3.txt",
        materials_name="materials3.txt",
        shift=(0.0, 0.0),
        y_floor=0.0,
    )


def load_mesh_luzec_2d(elem_type: str, base_dir: str | Path) -> TextMesh2D:
    base_dir = Path(base_dir)
    return _load_text_mesh_case(
        base_dir,
        elem_type=elem_type,
        coordinates_name="coordinates.txt",
        elements_name="elements.txt",
        materials_name="materials.txt",
        shift=(-200.0, -30.0),
        y_floor=0.0,
    )


def load_mesh_franz_dam_2d(elem_type: str, base_dir: str | Path) -> TextMesh2D:
    base_dir = Path(base_dir)
    raw_coord = np.loadtxt(base_dir / "coordinates.txt", dtype=np.float64)
    y_floor = float(np.min(raw_coord[:, 1]))
    return _load_text_mesh_case(
        base_dir,
        elem_type=elem_type,
        coordinates_name="coordinates.txt",
        elements_name="elements.txt",
        materials_name="materials.txt",
        shift=(0.0, 0.0),
        y_floor=y_floor,
    )


def luzec_pressure_boundary(coord: np.ndarray, surf: np.ndarray, water_unit_weight: float) -> tuple[np.ndarray, np.ndarray]:
    q_d = (coord[1, surf[0, :]] + coord[1, surf[1, :]]) / 2.0 >= 1.0e-1
    q_w = np.ones(coord.shape[1], dtype=bool)
    q_w[np.unique(surf[:, q_d])] = False
    q_w[(np.abs(coord[0, :] - 93.07) < 0.05) & (np.abs(coord[1, :] - 18.16) < 0.05)] = False

    x1, y1 = 91.12, 15.75
    x2, y2 = 101.845, 22.40
    pw_d = np.zeros(coord.shape[1], dtype=np.float64)
    part1 = (coord[0, :] < x1 + 1.0e-9) & (coord[1, :] < y1)
    part2 = (
        (coord[0, :] >= x1 + 1.0e-9)
        & (coord[0, :] < x2 + 1.0e-9)
        & (coord[1, :] < ((y2 - y1) / (x2 - x1)) * (coord[0, :] - x1) + y1)
    )
    part3 = coord[0, :] >= x2 + 1.0e-9
    pw_d[part1] = water_unit_weight * (y1 - coord[1, part1])
    pw_d[part2] = water_unit_weight * (
        ((y2 - y1) / (x2 - x1)) * (coord[0, part2] - x1) + y1 - coord[1, part2]
    )
    pw_d[part3] = water_unit_weight * (y2 - coord[1, part3])
    return q_w, pw_d


def franz_dam_pressure_boundary(coord: np.ndarray, surf: np.ndarray, water_unit_weight: float) -> tuple[np.ndarray, np.ndarray]:
    q_d = (coord[1, surf[0, :]] + coord[1, surf[1, :]]) / 2.0 >= -400.0 + 1.0e-1
    q_w = np.ones(coord.shape[1], dtype=bool)
    q_w[np.unique(surf[:, q_d])] = False

    x1, y1 = -82.5, -50.0
    x2, y2 = 172.5, -112.0
    pw_d = np.zeros(coord.shape[1], dtype=np.float64)
    part1 = (coord[0, :] < x1 + 1.0e-9) & (coord[1, :] < y1)
    part2 = (
        (coord[0, :] >= x1 + 1.0e-9)
        & (coord[0, :] < x2 + 1.0e-9)
        & (coord[1, :] < ((y2 - y1) / (x2 - x1)) * (coord[0, :] - x1) + y1)
    )
    part3 = (coord[0, :] >= x2 + 1.0e-9) & (coord[1, :] < y2)
    pw_d[part1] = water_unit_weight * (y1 - coord[1, part1])
    pw_d[part2] = water_unit_weight * (
        ((y2 - y1) / (x2 - x1)) * (coord[0, part2] - x1) + y1 - coord[1, part2]
    )
    pw_d[part3] = water_unit_weight * (y2 - coord[1, part3])
    return q_w, pw_d
