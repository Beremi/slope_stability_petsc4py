"""2D Sloan2013 weak-layer slope mesh."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .textmesh_2d import _boundary_edges as _triangle_boundary_edges
from .textmesh_2d import _expand_p2 as _expand_triangle_mesh_p2
from .textmesh_2d import _expand_p4 as _expand_triangle_mesh_p4


@dataclass(frozen=True)
class Sloan2013Mesh2D:
    coord: np.ndarray
    elem: np.ndarray
    q_mask: np.ndarray
    material: np.ndarray
    surf: np.ndarray


def _boundary_edges(elem: np.ndarray) -> np.ndarray:
    tri = np.asarray(elem[:3, :], dtype=np.int64)
    edges = np.hstack((tri[[1, 2], :], tri[[2, 0], :], tri[[0, 1], :]))
    edges_sorted = np.sort(edges, axis=0)
    uniq, inv, counts = np.unique(edges_sorted.T, axis=0, return_inverse=True, return_counts=True)
    boundary = counts[inv] == 1
    return edges[:, boundary]


def generate_sloan2013_mesh_2d(
    *,
    elem_type: str = "P1",
    h: float = 0.5,
    x1: float = 15.0,
    x3: float = 20.0,
    y11: float = 6.75,
    y12: float = 0.5,
    y13: float = 0.75,
    y21: float = 1.0,
    y22: float = 9.25,
    y23: float = 2.0,
    beta_deg: float = 26.6,
) -> Sloan2013Mesh2D:
    """Replicate MATLAB ``MESH.create_mesh_Sloan2013`` for P1/P2/P4."""

    elem_type = str(elem_type).upper()
    if elem_type not in {"P1", "P2", "P4"}:
        raise ValueError(f"Unsupported Sloan2013 element type {elem_type!r}.")

    beta = np.deg2rad(beta_deg)
    y1 = y11 + y12 + y13
    y2 = y21 + y22 + y23
    x2 = y2 / np.tan(beta)

    nx12 = int(round((x1 + x2) / h))
    nx3 = int(round(x3 / h))
    nx = nx12 + nx3
    ny11 = int(np.ceil(y11 / h))
    ny12 = int(np.ceil(y12 / h))
    ny13 = int(np.ceil(y13 / h))
    ny1 = ny11 + ny12 + ny13
    ny21 = int(np.ceil(y21 / h))
    ny22 = int(np.ceil(y22 / h))
    ny23 = int(np.ceil(y23 / h))
    ny2 = ny21 + ny22 + ny23
    ny = ny1 + ny2

    coord_x12 = np.linspace(0.0, x1 + x2, nx12 + 1)
    coord_x3 = np.linspace(x1 + x2, x1 + x2 + x3, nx3 + 1)
    coord_x = np.concatenate((coord_x12, coord_x3[1:]))

    coord_y11 = np.linspace(0.0, y11, ny11 + 1)
    coord_y12 = np.linspace(y11, y11 + y12, ny12 + 1)
    coord_y13 = np.linspace(y11 + y12, y1, ny13 + 1)
    coord_y21 = np.linspace(y1, y1 + y21, ny21 + 1)
    coord_y22 = np.linspace(y1 + y21, y1 + y21 + y22, ny22 + 1)
    coord_y23 = np.linspace(y1 + y21 + y22, y1 + y2, ny23 + 1)
    coord_y = np.concatenate((coord_y11, coord_y12[1:], coord_y13[1:], coord_y21[1:], coord_y22[1:], coord_y23[1:]))

    n_nodes_est = (ny1 + 1) * (nx + 1) + ny2 * (nx12 + 1)
    coord = np.zeros((2, n_nodes_est), dtype=np.float64)
    V = np.zeros((nx + 1, ny + 1), dtype=np.int64)
    n_n = 0

    for j in range(ny1 + 1):
        for i in range(nx + 1):
            V[i, j] = n_n
            coord[:, n_n] = [coord_x[i], coord_y[j]]
            n_n += 1

    for j in range(ny1 + 1, ny + 1):
        x_max = x1 + x2 * (y1 + y2 - coord_y[j]) / y2
        coord_x_row = np.linspace(0.0, x_max, nx12 + 1)
        for i in range(nx12 + 1):
            V[i, j] = n_n
            coord[:, n_n] = [coord_x_row[i], coord_y[j]]
            n_n += 1

    coord = coord[:, :n_n]
    elems: list[list[int]] = []
    mater: list[int] = []

    def add_cell(i: int, j: int, mat_id: int) -> None:
        elems.append([int(V[i + 1, j]), int(V[i + 1, j + 1]), int(V[i, j])])
        mater.append(mat_id)
        elems.append([int(V[i, j + 1]), int(V[i, j]), int(V[i + 1, j + 1])])
        mater.append(mat_id)

    for j in range(ny11):
        for i in range(nx):
            add_cell(i, j, 0)
    for j in range(ny11, ny11 + ny12):
        for i in range(nx):
            add_cell(i, j, 1)
    for j in range(ny11 + ny12, ny1):
        for i in range(nx):
            add_cell(i, j, 0)
    for j in range(ny1, ny):
        for i in range(nx12):
            add_cell(i, j, 0)

    elem = np.asarray(elems, dtype=np.int64).T
    material = np.asarray(mater, dtype=np.int64)
    surf = _boundary_edges(elem)

    x_max = float(np.max(coord[0, :]))
    y_min = float(np.min(coord[1, :]))
    if elem_type == "P1":
        coord_out = coord
        elem_out = elem
        surf_out = surf
    else:
        boundary_keys, oriented_edges = _triangle_boundary_edges(elem)
        if elem_type == "P2":
            coord_out, elem_out, surf_out = _expand_triangle_mesh_p2(coord, elem, boundary_keys, oriented_edges)
        else:
            coord_out, elem_out, surf_out = _expand_triangle_mesh_p4(coord, elem, boundary_keys, oriented_edges)

    q_mask = np.zeros((2, coord_out.shape[1]), dtype=bool)
    q_mask[0, :] = (coord_out[0, :] > 0.01) & (coord_out[1, :] > y_min + 0.01) & (coord_out[0, :] < x_max - 0.01)
    q_mask[1, :] = coord_out[1, :] > y_min + 0.01

    return Sloan2013Mesh2D(coord=coord_out, elem=elem_out, q_mask=q_mask, material=material, surf=surf_out)
