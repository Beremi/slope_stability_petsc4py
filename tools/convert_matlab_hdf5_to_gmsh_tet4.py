#!/usr/bin/env python3
"""Convert supported HDF5 tetrahedral meshes into Gmsh tet4/tri3 meshes."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import meshio
import numpy as np


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
        f"Cannot orient connectivity array of shape {arr.shape}; expected one dimension in {tuple(sorted(valid))}."
    )


def _load_legacy_hdf5_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(str(path), "r") as h5:
        node = np.asarray(h5["node"][:], dtype=np.float64)
        elem = _orient_connectivity(np.asarray(h5["elem"][:], dtype=np.int64), (4, 10, 35))
        face = _orient_connectivity(np.asarray(h5["face"][:], dtype=np.int64), (3, 6, 15))
        material = np.asarray(h5["material"][:], dtype=np.int64).ravel()
        boundary = np.asarray(h5["boundary"][:], dtype=np.int64).ravel()

    if node.ndim != 2:
        raise ValueError(f"Expected node array to be 2D, got shape {node.shape}.")
    if node.shape[1] != 3 and node.shape[0] == 3:
        node = node.T
    if node.shape[1] != 3:
        raise ValueError(f"Expected 3D coordinates, got shape {node.shape}.")

    # Match the framework's coordinate convention.
    points = np.asarray(node[:, [0, 2, 1]], dtype=np.float64)
    tet4 = np.asarray(elem[:4, :], dtype=np.int64)
    tri3 = np.asarray(face[:3, :], dtype=np.int64)
    return points, tet4, tri3, material, boundary


def _load_waterlevels_hdf5_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(str(path), "r") as h5:
        points = np.asarray(h5["points"][:], dtype=np.float64)
        tetra = _orient_connectivity(np.asarray(h5["tetra_cells"][:], dtype=np.int64), (4, 10, 35))
        tetra_labels = np.asarray(h5["tetra_labels"][:], dtype=np.int64).ravel() - 1
        triangles = _orient_connectivity(np.asarray(h5["triangles"][:], dtype=np.int64), (3, 6, 15))
        triangle_labels = np.asarray(h5["triangle_labels"][:], dtype=np.int64).ravel()

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points array with shape (n, 3), got {points.shape}.")

    points_xyz = np.asarray(points[:, [0, 2, 1]], dtype=np.float64)
    tet4 = np.asarray(tetra[:4, :], dtype=np.int64)
    tri3 = np.asarray(triangles[:3, :], dtype=np.int64)
    return points_xyz, tet4, tri3, tetra_labels, triangle_labels


def _load_hdf5_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(str(path), "r") as h5:
        keys = set(h5.keys())
    if {"boundary", "elem", "face", "material", "node"} <= keys:
        return _load_legacy_hdf5_mesh(path)
    if {"points", "tetra_cells", "tetra_labels", "triangles", "triangle_labels"} <= keys:
        return _load_waterlevels_hdf5_mesh(path)
    raise ValueError(f"Unsupported HDF5 mesh schema in {path}; keys are {sorted(keys)}.")


def _compact_vertices(points: np.ndarray, tet4: np.ndarray, tri3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    used = np.unique(np.concatenate((tet4.reshape(-1), tri3.reshape(-1) if tri3.size else np.empty(0, dtype=np.int64))))
    remap = np.full(points.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    points_new = points[used, :]
    tet4_new = remap[tet4]
    tri3_new = remap[tri3] if tri3.size else tri3.copy()
    return points_new, tet4_new, tri3_new


def _field_data(tags: np.ndarray, dim: int, prefix: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    unique = sorted(int(v) for v in np.unique(np.asarray(tags, dtype=np.int64)))
    physical = np.empty_like(np.asarray(tags, dtype=np.int64))
    metadata: dict[str, np.ndarray] = {}
    for offset, logical in enumerate(unique, start=1):
        physical[np.asarray(tags, dtype=np.int64) == logical] = int(offset)
        metadata[f"{prefix}_{logical}"] = np.asarray([offset, dim], dtype=np.int64)
    return physical, metadata


def convert_mesh(input_path: Path, output_path: Path) -> None:
    points, tet4, tri3, material, boundary = _load_hdf5_mesh(input_path)
    points, tet4, tri3 = _compact_vertices(points, tet4, tri3)
    tet_phys, tet_meta = _field_data(material, 3, "material")
    tri_phys, tri_meta = _field_data(boundary, 2, "boundary")

    mesh = meshio.Mesh(
        points=points,
        cells=[
            ("tetra", tet4.T.astype(np.int64)),
            ("triangle", tri3.T.astype(np.int64)),
        ],
        cell_data={
            "gmsh:physical": [
                tet_phys.astype(np.int64),
                tri_phys.astype(np.int64),
            ],
            "gmsh:geometrical": [
                tet_phys.astype(np.int64),
                tri_phys.astype(np.int64),
            ],
        },
        field_data={**tet_meta, **tri_meta},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.write(output_path, file_format="gmsh22", binary=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a supported HDF5 tetrahedral mesh into a Gmsh tet4 mesh.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    convert_mesh(args.input_path.resolve(), args.output_path.resolve())


if __name__ == "__main__":
    main()
