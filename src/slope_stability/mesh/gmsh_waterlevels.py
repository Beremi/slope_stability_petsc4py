"""Loader for GMSH HDF5 meshes with water-level labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from ..io import _collect_meshio_blocks, _elevate_tet4_mesh_to_tet10, _map_physical_ids, load_mesh_file
from ..problem_assets import build_dirichlet_mask_for_path, build_seepage_boundary_for_path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WATERLEVELS_MESH_PATH = ROOT / "meshes" / "3d_hetero_seepage" / "concave_family_b.msh"


@dataclass(frozen=True)
class WaterlevelsMesh3D:
    coord: np.ndarray
    elem: np.ndarray
    surf: np.ndarray
    q_mask: np.ndarray
    material: np.ndarray
    triangle_labels: np.ndarray


def _normalize_elem_type(elem_type: str | None) -> str:
    if elem_type is None:
        return "P2"
    text = str(elem_type).strip().upper()
    if text not in {"P1", "P2", "P4"}:
        raise NotImplementedError(f"Waterlevels mesh loader currently supports only 'P1', 'P2', and 'P4', got {elem_type!r}.")
    return text


def _compact_p1_connectivity(
    coord: np.ndarray,
    elem: np.ndarray,
    surf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop unused higher-order nodes after projecting a P2 waterlevels mesh to P1."""

    used_nodes = np.unique(
        np.concatenate(
            [
                np.asarray(elem, dtype=np.int64).reshape(-1, order="F"),
                np.asarray(surf, dtype=np.int64).reshape(-1, order="F"),
            ]
        )
    ).astype(np.int64)
    if used_nodes.size == int(coord.shape[1]) and np.array_equal(
        used_nodes, np.arange(int(coord.shape[1]), dtype=np.int64)
    ):
        return coord, elem, surf

    old_to_new = np.full(int(coord.shape[1]), -1, dtype=np.int64)
    old_to_new[used_nodes] = np.arange(used_nodes.size, dtype=np.int64)
    coord_compact = np.asarray(coord[:, used_nodes], dtype=np.float64)
    elem_compact = old_to_new[np.asarray(elem, dtype=np.int64)]
    surf_compact = old_to_new[np.asarray(surf, dtype=np.int64)]
    return coord_compact, elem_compact, surf_compact


def seepage_boundary_3d_hetero(
    coord: np.ndarray,
    surf: np.ndarray,
    triangle_labels: np.ndarray,
    grho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve seepage BCs from the waterlevels mesh-family definition."""

    return build_seepage_boundary_for_path(
        DEFAULT_WATERLEVELS_MESH_PATH,
        coord,
        surf,
        triangle_labels,
        grho=grho,
    )


def load_mesh_gmsh_waterlevels(path: str | Path, elem_type: str | None = "P2") -> WaterlevelsMesh3D:
    path = Path(path)
    elem_type_norm = _normalize_elem_type(elem_type)
    if path.suffix.lower() == ".msh":
        if elem_type_norm == "P4":
            mesh = load_mesh_file(path, elem_type="P4", boundary_type=0)
            triangle_labels = np.asarray(mesh.boundary, dtype=np.int64).ravel()
            return WaterlevelsMesh3D(
                coord=np.asarray(mesh.coord, dtype=np.float64),
                elem=np.asarray(mesh.elem, dtype=np.int64),
                surf=np.asarray(mesh.surf, dtype=np.int64),
                q_mask=np.asarray(mesh.q_mask, dtype=bool),
                material=np.asarray(mesh.material, dtype=np.int64),
                triangle_labels=triangle_labels,
            )
        try:
            import meshio
        except ImportError as exc:  # pragma: no cover - runtime dependency in normal use
            raise ImportError("Reading .msh files requires the 'meshio' package.") from exc

        msh = meshio.read(path)
        tetra_cells, tetra_tags = _collect_meshio_blocks(msh, "tetra")
        triangle_cells, triangle_tags = _collect_meshio_blocks(msh, "triangle")
        if tetra_cells.size == 0:
            raise ValueError(f"No tetrahedral cells found in {path}.")

        coord_p1 = np.asarray(msh.points[:, :3], dtype=np.float64).T
        elem_p1 = np.asarray(tetra_cells.T, dtype=np.int64)
        surf_p1 = np.asarray(triangle_cells.T, dtype=np.int64) if triangle_cells.size else np.empty((3, 0), dtype=np.int64)
        triangle_labels = _map_physical_ids(triangle_tags, msh.field_data, 2, "boundary")
        material = _map_physical_ids(tetra_tags, msh.field_data, 3, "material")

        if elem_type_norm == "P1":
            coord = coord_p1
            elem = elem_p1
            surf = surf_p1
        else:
            coord, elem, surf = _elevate_tet4_mesh_to_tet10(coord_p1, elem_p1, surf_p1)
        q_mask = build_dirichlet_mask_for_path(
            path,
            dim=3,
            n_nodes=coord.shape[1],
            surf=surf,
            boundary=triangle_labels,
            coord=coord,
            boundary_type=0,
        )
        return WaterlevelsMesh3D(
            coord=coord,
            elem=elem,
            surf=surf,
            q_mask=q_mask,
            material=np.asarray(material, dtype=np.int64),
            triangle_labels=np.asarray(triangle_labels, dtype=np.int64),
        )

    if elem_type_norm == "P4":
        raise NotImplementedError("Waterlevels HDF5 meshes currently support only 'P1' and 'P2'; use the .msh family for 'P4'.")

    with h5py.File(str(path), "r") as h5:
        node = np.asarray(h5["points"][:], dtype=np.float64)
        tetra_cells = np.asarray(h5["tetra_cells"][:], dtype=np.int64).T
        tetra_labels = np.asarray(h5["tetra_labels"][:], dtype=np.int64).ravel() - 1
        triangle_cells = np.asarray(h5["triangles"][:], dtype=np.int64).T
        triangle_labels = np.asarray(h5["triangle_labels"][:], dtype=np.int64).ravel()

    coord = np.asarray(node[:, [0, 2, 1]].T, dtype=np.float64)
    if elem_type_norm == "P1":
        elem = tetra_cells[:4, :]
        surf = triangle_cells[:3, :]
        coord, elem, surf = _compact_p1_connectivity(coord, elem, surf)
    else:
        elem = tetra_cells[[0, 1, 2, 3, 4, 5, 6, 9, 8, 7], :]
        surf = triangle_cells
    q_mask = build_dirichlet_mask_for_path(
        path,
        dim=3,
        n_nodes=coord.shape[1],
        surf=surf,
        boundary=triangle_labels,
        coord=coord,
        boundary_type=0,
    )

    return WaterlevelsMesh3D(
        coord=coord,
        elem=elem,
        surf=surf,
        q_mask=q_mask,
        material=tetra_labels,
        triangle_labels=triangle_labels,
    )
