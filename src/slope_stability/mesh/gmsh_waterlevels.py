"""Loader for GMSH HDF5 meshes with water-level labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class WaterlevelsMesh3D:
    coord: np.ndarray
    elem: np.ndarray
    surf: np.ndarray
    q_mask: np.ndarray
    material: np.ndarray
    triangle_labels: np.ndarray


def seepage_boundary_3d_hetero(
    coord: np.ndarray,
    surf: np.ndarray,
    triangle_labels: np.ndarray,
    grho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate MATLAB ``MESH.seepage_boundary_3D_hetero``."""

    coord = np.asarray(coord, dtype=np.float64)
    surf = np.asarray(surf, dtype=np.int64)
    triangle_labels = np.asarray(triangle_labels, dtype=np.int64).ravel()

    n_n = int(coord.shape[1])
    y_free_water_level = 35.0
    y_porous_water_level = 55.0

    pw_d = np.zeros(n_n, dtype=np.float64)

    q_dry = np.zeros(n_n, dtype=bool)
    for label in (13, 5, 7):
        tmp = surf[:, triangle_labels == label]
        q_dry[tmp.ravel()] = True

    q_wet1 = np.zeros(n_n, dtype=bool)
    tmp = surf[:, triangle_labels == 8]
    q_wet1[tmp.ravel()] = True
    pw_d[q_wet1] = grho * (y_porous_water_level - coord[1, q_wet1])

    q_wet2 = np.zeros(n_n, dtype=bool)
    for label in (6, 14, 9):
        tmp = surf[:, triangle_labels == label]
        q_wet2[tmp.ravel()] = True
    pw_d[q_wet2] = grho * (y_free_water_level - coord[1, q_wet2])

    q_w = np.ones(n_n, dtype=bool)
    q_w[q_dry] = False
    q_w[q_wet1] = False
    q_w[q_wet2] = False
    return q_w, pw_d


def load_mesh_gmsh_waterlevels(path: str | Path) -> WaterlevelsMesh3D:
    path = Path(path)
    with h5py.File(str(path), "r") as h5:
        node = np.asarray(h5["points"][:], dtype=np.float64)
        tetra_cells = np.asarray(h5["tetra_cells"][:], dtype=np.int64).T
        tetra_labels = np.asarray(h5["tetra_labels"][:], dtype=np.int64).ravel() - 1
        triangle_cells = np.asarray(h5["triangles"][:], dtype=np.int64).T
        triangle_labels = np.asarray(h5["triangle_labels"][:], dtype=np.int64).ravel()

    q_mask = np.ones((3, node.shape[0]), dtype=bool)
    tmp = triangle_cells[:, triangle_labels == 7]
    q_mask[0, tmp.ravel()] = False
    tmp = triangle_cells[:, triangle_labels == 8]
    q_mask[0, tmp.ravel()] = False
    tmp = triangle_cells[:, triangle_labels == 9]
    q_mask[0, tmp.ravel()] = False
    tmp = triangle_cells[:, triangle_labels == 10]
    q_mask[2, tmp.ravel()] = False
    tmp = triangle_cells[:, triangle_labels == 11]
    q_mask[2, tmp.ravel()] = False
    tmp = triangle_cells[:, triangle_labels == 12]
    q_mask[:, tmp.ravel()] = False

    coord = np.asarray(node[:, [0, 2, 1]].T, dtype=np.float64)
    q_mask = q_mask[[0, 2, 1], :]
    elem = tetra_cells[[0, 1, 2, 3, 4, 5, 6, 9, 8, 7], :]
    surf = triangle_cells

    return WaterlevelsMesh3D(
        coord=coord,
        elem=elem,
        surf=surf,
        q_mask=q_mask,
        material=tetra_labels,
        triangle_labels=triangle_labels,
    )
