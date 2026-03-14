"""COMSOL-specific P2 mesh loading and seepage boundary helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class ComsolP2Mesh3D:
    coord: np.ndarray
    elem: np.ndarray
    surf: np.ndarray
    q_mask: np.ndarray
    material: np.ndarray
    triangle_labels: np.ndarray


def _as_nodes_by_count(arr: np.ndarray, width: int) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {out.shape}")
    if out.shape[0] == width:
        return out
    if out.shape[1] == width:
        return out.T
    raise ValueError(f"Unsupported shape {out.shape} for width {width}")


def load_mesh_p2_comsol(path: str | Path, boundary_type: int = 1) -> ComsolP2Mesh3D:
    """Replicate MATLAB ``MESH.load_mesh_P2(file_path, boundary_type)`` for COMSOL meshes."""

    path = Path(path)
    with h5py.File(str(path), "r") as h5:
        boundary = np.asarray(h5["boundary"][:], dtype=np.int64).ravel()
        elem = _as_nodes_by_count(np.asarray(h5["elem"][:], dtype=np.int64), 10)
        face = _as_nodes_by_count(np.asarray(h5["face"][:], dtype=np.int64), 6)
        material = np.asarray(h5["material"][:], dtype=np.int64).ravel()
        node = _as_nodes_by_count(np.asarray(h5["node"][:], dtype=np.float64), 3)

    q = np.ones_like(node, dtype=bool)

    tmp = face[:, boundary == 1]
    q[0, tmp.ravel()] = False
    tmp = face[:, boundary == 2]
    q[0, tmp.ravel()] = False
    tmp = face[:, boundary == 3]
    q[1, tmp.ravel()] = False
    tmp = face[:, boundary == 4]
    q[1, tmp.ravel()] = False
    tmp = face[:, boundary == 5]
    if int(boundary_type):
        q[:, tmp.ravel()] = False
    else:
        q[2, tmp.ravel()] = False

    coord = np.asarray(node[[0, 2, 1], :], dtype=np.float64)
    q = q[[0, 2, 1], :]
    return ComsolP2Mesh3D(
        coord=coord,
        elem=np.asarray(elem, dtype=np.int64),
        surf=np.asarray(face, dtype=np.int64),
        q_mask=np.asarray(q, dtype=bool),
        material=material,
        triangle_labels=boundary,
    )


def seepage_boundary_3d_hetero_comsol(
    coord: np.ndarray,
    surf: np.ndarray,
    triangle_labels: np.ndarray,
    grho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate MATLAB ``MESH.seepage_boundary_3D_hetero_comsol`` exactly."""

    coord = np.asarray(coord, dtype=np.float64)
    surf = np.asarray(surf, dtype=np.int64)
    triangle_labels = np.asarray(triangle_labels, dtype=np.int64).ravel()

    n_n = int(coord.shape[1])
    y_free_water_level = 35.0
    y_porous_water_level = 55.0

    pw_d = np.zeros(n_n, dtype=np.float64)
    x = coord[0, :]
    y = coord[1, :]
    boundary_nodes = np.unique(surf.ravel())

    q_dry = np.zeros(n_n, dtype=bool)
    tmp = surf[:, triangle_labels == 6]
    q_dry[tmp.ravel()] = True

    q_wet1 = np.zeros(n_n, dtype=bool)
    tmp = surf[:, triangle_labels == 2]
    nodes_tmp = np.unique(tmp.ravel())
    selected_nodes_wet = nodes_tmp[y[nodes_tmp] < y_porous_water_level]
    q_wet1[selected_nodes_wet] = True
    pw_d[q_wet1] = grho * (y_porous_water_level - coord[1, q_wet1])
    dry = nodes_tmp[y[nodes_tmp] >= y_porous_water_level]
    q_dry[dry] = True

    q_wet2 = np.zeros(n_n, dtype=bool)
    triangles = surf[:3, :]
    v1 = coord[:, triangles[0, :]]
    v2 = coord[:, triangles[1, :]]
    v3 = coord[:, triangles[2, :]]
    e1 = v2 - v1
    e2 = v3 - v1
    normals = np.cross(e1.T, e2.T).T
    tol = 1.0e-1
    condition = np.all(np.abs(normals) > tol, axis=0)
    selected_triangles = surf[:, condition]
    nodes_tmp = np.unique(selected_triangles.ravel())

    tol = 1.0e-6
    c = np.array([55.0, 30.0, 0.0], dtype=np.float64)
    t = np.array([115.0, 60.0, 0.0], dtype=np.float64)
    a_left = np.array([30.0, 30.0, 43.3], dtype=np.float64)
    a_right = a_left.copy()
    a_right[2] = -a_right[2]
    normal_left = np.cross(t - c, a_left - c)
    normal_right = np.cross(t - c, a_right - c)
    X = coord[:, nodes_tmp].T
    V = X - c
    d_left = np.abs(V @ normal_left)
    d_right = np.abs(V @ normal_right)
    mask = (d_left < tol) | (d_right < tol)
    nodes_tmp = nodes_tmp[mask]

    selected_nodes = nodes_tmp[y[nodes_tmp] < y_free_water_level]
    q_wet2[selected_nodes] = True
    selected_nodes_dry = nodes_tmp[y[nodes_tmp] >= y_free_water_level]
    q_dry[selected_nodes_dry] = True

    tol = 1.0e-1
    y_bed = 30.0
    selected = boundary_nodes[np.abs(y[boundary_nodes] - y_bed) < tol]

    tol = 1.0e-10
    c = np.array([55.0, 30.0, 0.0], dtype=np.float64)
    a_left = np.array([30.0, 30.0, 43.3], dtype=np.float64)
    a_right = a_left.copy()
    a_right[2] = -a_right[2]
    n = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    X = coord[:, selected].T
    dL = a_left - c
    dR = a_right - c
    VL = X - c
    kL = np.cross(n, dL)
    kR = np.cross(n, dR)
    kL = kL / np.linalg.norm(kL)
    kR = kR / np.linalg.norm(kR)
    sL = VL @ kL
    sR = VL @ kR
    sL = sL / np.linalg.norm(sL)
    sR = sR / np.linalg.norm(sR)
    mask = (sL < tol) & (sR > tol)
    selected = selected[mask]
    q_wet2[selected] = True

    tmp = surf[:, triangle_labels == 1]
    q_wet2[tmp.ravel()] = True
    pw_d[q_wet2] = grho * (y_free_water_level - coord[1, q_wet2])

    q_w = np.ones(n_n, dtype=bool)
    q_w[q_dry] = False
    q_w[q_wet2] = False
    q_w[q_wet1] = False
    return q_w, pw_d
