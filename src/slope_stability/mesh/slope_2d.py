"""Procedural 2D slope meshes matching the MATLAB homogeneous benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .textmesh_2d import _boundary_edges as _triangle_boundary_edges
from .textmesh_2d import _expand_p4 as _expand_triangle_mesh_p4


@dataclass(frozen=True)
class Slope2DMesh:
    coord: np.ndarray
    elem: np.ndarray
    surf: np.ndarray
    q_mask: np.ndarray
    material: np.ndarray


def _boundary_mask(coord: np.ndarray) -> np.ndarray:
    x_max = float(np.max(coord[0, :])) - 1.0e-9
    q_mask = np.zeros((2, coord.shape[1]), dtype=bool)
    q_mask[0, :] = (coord[0, :] > 0.0) & (coord[1, :] > 0.0) & (coord[0, :] < x_max)
    q_mask[1, :] = coord[1, :] > 0.0
    return q_mask


def _boundary_edges_from_edge_el(edge_nodes: np.ndarray, edge_el: np.ndarray) -> np.ndarray:
    boundary = np.any(np.asarray(edge_el, dtype=np.int64) == 0, axis=0)
    return np.asarray(edge_nodes, dtype=np.int64)[:, boundary]


def mesh_p1_2d(h: float, x1: float, x2: float, x3: float, y1: float, y2: float) -> Slope2DMesh:
    """Replicate MATLAB ``MESH.mesh_P1_2D``."""

    nx12 = int(round((x1 + x2) / h))
    nx3 = int(round(x3 / h))
    nx = nx12 + nx3
    ny1 = int(round(y1 / h))
    ny2 = int(round(y2 / h))
    ny = ny1 + ny2

    coord_x12 = np.linspace(0.0, x1 + x2, nx12 + 1, dtype=np.float64)
    coord_x3 = np.linspace(x1 + x2, x1 + x2 + x3, nx3 + 1, dtype=np.float64)
    coord_x = np.concatenate((coord_x12, coord_x3[1:]))

    coord_y1 = np.linspace(0.0, y1, ny1 + 1, dtype=np.float64)
    coord_y2 = np.linspace(y1, y1 + y2, ny2 + 1, dtype=np.float64)
    coord_y = np.concatenate((coord_y1, coord_y2[1:]))

    coord = np.zeros((2, (ny1 + 1) * (nx + 1) + ny2 * (nx12 + 1)), dtype=np.float64)
    V = np.zeros((nx + 1, ny + 1), dtype=np.int64)
    n_n = 0

    for j in range(ny1 + 1):
        for i in range(nx + 1):
            V[i, j] = n_n
            coord[:, n_n] = np.array([coord_x[i], coord_y[j]], dtype=np.float64)
            n_n += 1

    for j in range(ny1 + 1, ny + 1):
        x_max = x1 + x2 * (y1 + y2 - coord_y[j]) / y2
        local_x = np.linspace(0.0, x_max, nx12 + 1, dtype=np.float64)
        for i in range(nx12 + 1):
            V[i, j] = n_n
            coord[:, n_n] = np.array([local_x[i], coord_y[j]], dtype=np.float64)
            n_n += 1

    elem = np.zeros((3, 2 * nx * ny), dtype=np.int64)
    n_e = 0
    for j in range(ny1):
        for i in range(nx):
            elem[:, n_e] = np.array([V[i, j], V[i + 1, j], V[i, j + 1]], dtype=np.int64)
            n_e += 1
            elem[:, n_e] = np.array([V[i + 1, j + 1], V[i, j + 1], V[i + 1, j]], dtype=np.int64)
            n_e += 1
    n1_e = n_e
    for j in range(ny1, ny):
        for i in range(nx12):
            elem[:, n_e] = np.array([V[i, j], V[i + 1, j], V[i, j + 1]], dtype=np.int64)
            n_e += 1
            elem[:, n_e] = np.array([V[i + 1, j + 1], V[i, j + 1], V[i + 1, j]], dtype=np.int64)
            n_e += 1
    elem = elem[:, :n_e]
    n2_e = n_e - n1_e

    n_ed = 0
    n1_ed = nx * (ny1 + 1)
    Eh1 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx, ny1 + 1, order="F")
    n_ed += n1_ed
    n1_ed = (nx + 1) * ny1
    Ev1 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx + 1, ny1, order="F")
    n_ed += n1_ed
    n1_ed = nx * ny1
    Ed1 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx, ny1, order="F")
    n_ed += n1_ed

    E12 = Eh1[:, :ny1].reshape(-1, order="F")
    E23 = Ev1[1:, :ny1].reshape(-1, order="F")
    E34 = Eh1[:, 1 : ny1 + 1].reshape(-1, order="F")
    E14 = Ev1[:nx, :ny1].reshape(-1, order="F")
    E24 = Ed1.reshape(-1, order="F")
    aux_elem_ed = np.vstack((E12, E24, E14, E34, E24, E23))
    elem1_ed = aux_elem_ed.reshape(3, n1_e, order="F")

    n1_ed = nx12 * ny2
    Eh2 = np.concatenate(
        (
            Eh1[:nx12, -1][:, None],
            np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx12, ny2, order="F"),
        ),
        axis=1,
    )
    n_ed += n1_ed
    n1_ed = (nx12 + 1) * ny2
    Ev2 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx12 + 1, ny2, order="F")
    n_ed += n1_ed
    n1_ed = nx12 * ny2
    Ed2 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx12, ny2, order="F")

    E12 = Eh2[:, :ny2].reshape(-1, order="F")
    E23 = Ev2[1:, :ny2].reshape(-1, order="F")
    E34 = Eh2[:, 1 : ny2 + 1].reshape(-1, order="F")
    E14 = Ev2[:nx12, :ny2].reshape(-1, order="F")
    E24 = Ed2.reshape(-1, order="F")
    aux_elem_ed = np.vstack((E12, E24, E14, E34, E24, E23))
    elem2_ed = aux_elem_ed.reshape(3, n2_e, order="F")
    elem_ed = np.hstack((elem1_ed, elem2_ed))

    Tlb1 = np.zeros((nx + 1, ny1 + 1), dtype=np.int64)
    Trt1 = np.zeros((nx + 1, ny1 + 1), dtype=np.int64)
    if n1_e:
        Tlb1[:nx, :ny1] = np.arange(0, n1_e, 2, dtype=np.int64).reshape(nx, ny1, order="F") + 1
        Tlb1[:nx12, -1] = np.arange(0, 2 * nx12, 2, dtype=np.int64) + n1_e + 1
        Trt1[1:, 1 : ny1 + 1] = np.arange(1, n1_e, 2, dtype=np.int64).reshape(nx, ny1, order="F") + 1

    Tlb2 = np.zeros((nx12 + 1, ny2 + 1), dtype=np.int64)
    Trt2 = np.zeros((nx12 + 1, ny2 + 1), dtype=np.int64)
    if n2_e:
        Tlb2[:nx12, :ny2] = np.arange(n1_e, n_e, 2, dtype=np.int64).reshape(nx12, ny2, order="F") + 1
        Trt2[1:, 1 : ny2 + 1] = np.arange(n1_e + 1, n_e, 2, dtype=np.int64).reshape(nx12, ny2, order="F") + 1
        Trt2[1:, 0] = np.arange(2, 2 * nx12 + 1, 2, dtype=np.int64) + n1_e - 2 * nx

    edge1_el_h = np.vstack((Tlb1[:nx, :].reshape(-1, order="F"), Trt1[1:, :].reshape(-1, order="F")))
    edge2_el_h = np.vstack((Tlb2[:nx12, 1:].reshape(-1, order="F"), Trt2[1:, 1:].reshape(-1, order="F")))
    edge1_el_v = np.vstack((Trt1[:, 1:].reshape(-1, order="F"), Tlb1[:, :ny1].reshape(-1, order="F")))
    edge2_el_v = np.vstack((Trt2[:, 1:].reshape(-1, order="F"), Tlb2[:, :ny2].reshape(-1, order="F")))
    edge1_el_d = np.vstack((Tlb1[:nx, :ny1].reshape(-1, order="F"), Trt1[1:, 1:].reshape(-1, order="F")))
    edge2_el_d = np.vstack((Tlb2[:nx12, :ny2].reshape(-1, order="F"), Trt2[1:, 1:].reshape(-1, order="F")))
    edge_el = np.hstack((edge1_el_h, edge1_el_v, edge1_el_d, edge2_el_h, edge2_el_v, edge2_el_d))

    edge_nodes = np.zeros((2, edge_el.shape[1]), dtype=np.int64)
    for e in range(elem_ed.shape[1]):
        tri = elem[:, e]
        for local_edge, edge_id in enumerate(elem_ed[:, e]):
            if edge_nodes[0, edge_id] != 0 or edge_nodes[1, edge_id] != 0 or (tri[0] == 0 and tri[1] == 0):
                continue
            if local_edge == 0:
                edge_nodes[:, edge_id] = np.array([tri[0], tri[1]], dtype=np.int64)
            elif local_edge == 1:
                edge_nodes[:, edge_id] = np.array([tri[1], tri[2]], dtype=np.int64)
            else:
                edge_nodes[:, edge_id] = np.array([tri[0], tri[2]], dtype=np.int64)

    surf = _boundary_edges_from_edge_el(edge_nodes, edge_el)
    q_mask = _boundary_mask(coord)
    material = np.zeros(elem.shape[1], dtype=np.int64)
    return Slope2DMesh(coord=coord, elem=elem, surf=surf, q_mask=q_mask, material=material)


def mesh_p2_2d(h: float, x1: float, x2: float, x3: float, y1: float, y2: float) -> Slope2DMesh:
    """Replicate MATLAB ``MESH.mesh_P2_2D``."""

    nx12 = int(round((x1 + x2) / h))
    nx3 = int(round(x3 / h))
    nx = nx12 + nx3
    ny1 = int(round(y1 / h))
    ny2 = int(round(y2 / h))
    ny = ny1 + ny2

    coord_x12 = np.linspace(0.0, x1 + x2, 2 * nx12 + 1, dtype=np.float64)
    coord_x3 = np.linspace(x1 + x2, x1 + x2 + x3, 2 * nx3 + 1, dtype=np.float64)
    coord_x = np.concatenate((coord_x12, coord_x3[1:]))

    coord_y1 = np.linspace(0.0, y1, 2 * ny1 + 1, dtype=np.float64)
    coord_y2 = np.linspace(y1, y1 + y2, 2 * ny2 + 1, dtype=np.float64)
    coord_y = np.concatenate((coord_y1, coord_y2[1:]))

    coord = np.zeros((2, (2 * ny1 + 1) * (2 * nx + 1) + 2 * ny2 * (2 * nx12 + 1)), dtype=np.float64)
    C = np.zeros((2 * nx + 1, 2 * ny + 1), dtype=np.int64)
    n_n = 0

    for j in range(2 * ny1 + 1):
        for i in range(2 * nx + 1):
            C[i, j] = n_n
            coord[:, n_n] = np.array([coord_x[i], coord_y[j]], dtype=np.float64)
            n_n += 1

    for j in range(2 * ny1 + 1, 2 * ny + 1):
        x_max = x1 + x2 * (y1 + y2 - coord_y[j]) / y2
        local_x = np.linspace(0.0, x_max, 2 * nx12 + 1, dtype=np.float64)
        for i in range(2 * nx12 + 1):
            C[i, j] = n_n
            coord[:, n_n] = np.array([local_x[i], coord_y[j]], dtype=np.float64)
            n_n += 1

    elem = np.zeros((6, 2 * nx * ny), dtype=np.int64)
    n_e = 0
    for j in range(ny1):
        for i in range(nx):
            elem[:, n_e] = np.array(
                [
                    C[2 * i, 2 * j],
                    C[2 * i + 2, 2 * j],
                    C[2 * i, 2 * j + 2],
                    C[2 * i + 1, 2 * j + 1],
                    C[2 * i, 2 * j + 1],
                    C[2 * i + 1, 2 * j],
                ],
                dtype=np.int64,
            )
            n_e += 1
            elem[:, n_e] = np.array(
                [
                    C[2 * i + 2, 2 * j + 2],
                    C[2 * i, 2 * j + 2],
                    C[2 * i + 2, 2 * j],
                    C[2 * i + 1, 2 * j + 1],
                    C[2 * i + 2, 2 * j + 1],
                    C[2 * i + 1, 2 * j + 2],
                ],
                dtype=np.int64,
            )
            n_e += 1
    n1_e = n_e
    for j in range(ny1, ny):
        for i in range(nx12):
            elem[:, n_e] = np.array(
                [
                    C[2 * i, 2 * j],
                    C[2 * i + 2, 2 * j],
                    C[2 * i, 2 * j + 2],
                    C[2 * i + 1, 2 * j + 1],
                    C[2 * i, 2 * j + 1],
                    C[2 * i + 1, 2 * j],
                ],
                dtype=np.int64,
            )
            n_e += 1
            elem[:, n_e] = np.array(
                [
                    C[2 * i + 2, 2 * j + 2],
                    C[2 * i, 2 * j + 2],
                    C[2 * i + 2, 2 * j],
                    C[2 * i + 1, 2 * j + 1],
                    C[2 * i + 2, 2 * j + 1],
                    C[2 * i + 1, 2 * j + 2],
                ],
                dtype=np.int64,
            )
            n_e += 1
    elem = elem[:, :n_e]
    n2_e = n_e - n1_e

    n_ed = 0
    n1_ed = nx * (ny1 + 1)
    Eh1 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx, ny1 + 1, order="F")
    n_ed += n1_ed
    n1_ed = (nx + 1) * ny1
    Ev1 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx + 1, ny1, order="F")
    n_ed += n1_ed
    n1_ed = nx * ny1
    Ed1 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx, ny1, order="F")
    n_ed += n1_ed

    E12 = Eh1[:, :ny1].reshape(-1, order="F")
    E23 = Ev1[1:, :ny1].reshape(-1, order="F")
    E34 = Eh1[:, 1 : ny1 + 1].reshape(-1, order="F")
    E14 = Ev1[:nx, :ny1].reshape(-1, order="F")
    E24 = Ed1.reshape(-1, order="F")
    aux_elem_ed = np.vstack((E12, E24, E14, E34, E24, E23))
    elem1_ed = aux_elem_ed.reshape(3, n1_e, order="F")

    n1_ed = nx12 * ny2
    Eh2 = np.concatenate(
        (
            Eh1[:nx12, -1][:, None],
            np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx12, ny2, order="F"),
        ),
        axis=1,
    )
    n_ed += n1_ed
    n1_ed = (nx12 + 1) * ny2
    Ev2 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx12 + 1, ny2, order="F")
    n_ed += n1_ed
    n1_ed = nx12 * ny2
    Ed2 = np.arange(n_ed, n_ed + n1_ed, dtype=np.int64).reshape(nx12, ny2, order="F")

    E12 = Eh2[:, :ny2].reshape(-1, order="F")
    E23 = Ev2[1:, :ny2].reshape(-1, order="F")
    E34 = Eh2[:, 1 : ny2 + 1].reshape(-1, order="F")
    E14 = Ev2[:nx12, :ny2].reshape(-1, order="F")
    E24 = Ed2.reshape(-1, order="F")
    aux_elem_ed = np.vstack((E12, E24, E14, E34, E24, E23))
    elem2_ed = aux_elem_ed.reshape(3, n2_e, order="F")
    elem_ed = np.hstack((elem1_ed, elem2_ed))

    Tlb1 = np.zeros((nx + 1, ny1 + 1), dtype=np.int64)
    Trt1 = np.zeros((nx + 1, ny1 + 1), dtype=np.int64)
    if n1_e:
        Tlb1[:nx, :ny1] = np.arange(0, n1_e, 2, dtype=np.int64).reshape(nx, ny1, order="F") + 1
        Tlb1[:nx12, -1] = np.arange(0, 2 * nx12, 2, dtype=np.int64) + n1_e + 1
        Trt1[1:, 1 : ny1 + 1] = np.arange(1, n1_e, 2, dtype=np.int64).reshape(nx, ny1, order="F") + 1

    Tlb2 = np.zeros((nx12 + 1, ny2 + 1), dtype=np.int64)
    Trt2 = np.zeros((nx12 + 1, ny2 + 1), dtype=np.int64)
    if n2_e:
        Tlb2[:nx12, :ny2] = np.arange(n1_e, n_e, 2, dtype=np.int64).reshape(nx12, ny2, order="F") + 1
        Trt2[1:, 1 : ny2 + 1] = np.arange(n1_e + 1, n_e, 2, dtype=np.int64).reshape(nx12, ny2, order="F") + 1
        Trt2[1:, 0] = np.arange(2, 2 * nx12 + 1, 2, dtype=np.int64) + n1_e - 2 * nx

    edge1_el_h = np.vstack((Tlb1[:nx, :].reshape(-1, order="F"), Trt1[1:, :].reshape(-1, order="F")))
    edge2_el_h = np.vstack((Tlb2[:nx12, 1:].reshape(-1, order="F"), Trt2[1:, 1:].reshape(-1, order="F")))
    edge1_el_v = np.vstack((Trt1[:, 1:].reshape(-1, order="F"), Tlb1[:, :ny1].reshape(-1, order="F")))
    edge2_el_v = np.vstack((Trt2[:, 1:].reshape(-1, order="F"), Tlb2[:, :ny2].reshape(-1, order="F")))
    edge1_el_d = np.vstack((Tlb1[:nx, :ny1].reshape(-1, order="F"), Trt1[1:, 1:].reshape(-1, order="F")))
    edge2_el_d = np.vstack((Tlb2[:nx12, :ny2].reshape(-1, order="F"), Trt2[1:, 1:].reshape(-1, order="F")))
    edge_el = np.hstack((edge1_el_h, edge1_el_v, edge1_el_d, edge2_el_h, edge2_el_v, edge2_el_d))

    edge_nodes = np.zeros((3, edge_el.shape[1]), dtype=np.int64)
    for e in range(elem_ed.shape[1]):
        tri = elem[:, e]
        for local_edge, edge_id in enumerate(elem_ed[:, e]):
            if np.any(edge_nodes[:, edge_id] != 0) or (tri[0] == 0 and tri[1] == 0 and tri[2] == 0):
                continue
            if local_edge == 0:
                edge_nodes[:, edge_id] = np.array([tri[0], tri[1], tri[5]], dtype=np.int64)
            elif local_edge == 1:
                edge_nodes[:, edge_id] = np.array([tri[1], tri[2], tri[3]], dtype=np.int64)
            else:
                edge_nodes[:, edge_id] = np.array([tri[0], tri[2], tri[4]], dtype=np.int64)

    surf = _boundary_edges_from_edge_el(edge_nodes, edge_el)
    q_mask = _boundary_mask(coord)
    material = np.zeros(elem.shape[1], dtype=np.int64)
    return Slope2DMesh(coord=coord, elem=elem, surf=surf, q_mask=q_mask, material=material)


def generate_homogeneous_slope_mesh_2d(
    *,
    elem_type: str = "P2",
    h: float = 1.0,
    x1: float = 15.0,
    x2: float = 10.0,
    x3: float = 15.0,
    y1: float = 10.0,
    y2: float = 10.0,
) -> Slope2DMesh:
    elem_key = str(elem_type).upper()
    if elem_key == "P1":
        return mesh_p1_2d(h, x1, x2, x3, y1, y2)
    if elem_key == "P2":
        return mesh_p2_2d(h, x1, x2, x3, y1, y2)
    if elem_key == "P4":
        base = mesh_p1_2d(h, x1, x2, x3, y1, y2)
        boundary_keys, oriented_edges = _triangle_boundary_edges(base.elem)
        coord, elem, surf = _expand_triangle_mesh_p4(base.coord, base.elem, boundary_keys, oriented_edges)
        return Slope2DMesh(
            coord=coord,
            elem=elem,
            surf=surf,
            q_mask=_boundary_mask(coord),
            material=np.asarray(base.material, dtype=np.int64),
        )
    raise ValueError(f"Unsupported 2D element type {elem_type!r}")
