"""Mesh IO helpers for MATLAB-compatible HDF5 inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .core.elements import infer_simplex_elem_type, SIMPLEX_NODES_PER_SURFACE


@dataclass
class MeshData:
    coord: np.ndarray
    elem: np.ndarray
    surf: np.ndarray
    q_mask: np.ndarray
    material: np.ndarray
    boundary: np.ndarray
    elem_type: str | None = None


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


def _load_lagrange_tet_mesh(path: Path) -> MeshData:
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

    q = np.ones_like(node, dtype=bool)

    face_1 = face[:, boundary == 1].ravel()
    face_2 = face[:, boundary == 2].ravel()
    face_3 = face[:, boundary == 3].ravel()
    face_4 = face[:, boundary == 4].ravel()
    face_5 = face[:, boundary == 5].ravel()

    # Boundary mask behavior follows legacy MATLAB orientation convention.
    q[0, face_1] = 0
    q[0, face_2] = 0
    q[1, face_3] = 0
    q[1, face_4] = 0
    q[2, face_5] = 0

    # MATLAB exports stored as (x, z, y) in this helper.
    coord = np.asarray(node[[0, 2, 1], :], dtype=np.float64)
    q = q[[0, 2, 1], :]

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
    return _load_lagrange_tet_mesh(path)


def load_mesh_file(mesh_file: str | Path) -> MeshData:
    path = Path(mesh_file)
    lower = path.name.lower()
    if path.suffix.lower() == ".h5":
        with h5py.File(str(path), "r") as h5:
            keys = set(h5.keys())
        if {"boundary", "elem", "face", "material", "node"} <= keys:
            return _load_lagrange_tet_mesh(path)
    if "p2" in lower:
        return load_mesh_p2(path)
    raise ValueError(f"Unsupported mesh format for {path}")
