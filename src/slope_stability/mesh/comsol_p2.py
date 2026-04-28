"""COMSOL-specific P2 mesh loading and seepage boundary helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from ..io import _collect_meshio_blocks, _elevate_tet4_mesh_to_tet10, _map_physical_ids
from ..problem_assets import build_dirichlet_mask_for_path, build_seepage_boundary_for_path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMSOL_MESH_PATH = ROOT / "meshes" / "3d_hetero_seepage_transition" / "transition_default.msh"


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
    if path.suffix.lower() == ".msh":
        try:
            import meshio
        except ImportError as exc:  # pragma: no cover - runtime dependency in normal use
            raise ImportError("Reading .msh files requires the 'meshio' package.") from exc

        msh = meshio.read(path)
        tetra_cells, tetra_tags = _collect_meshio_blocks(msh, "tetra")
        face_cells, boundary = _collect_meshio_blocks(msh, "triangle")
        if tetra_cells.size == 0:
            raise ValueError(f"No tetrahedral cells found in {path}.")

        coord_p1 = np.asarray(msh.points[:, :3], dtype=np.float64).T
        elem_p1 = np.asarray(tetra_cells.T, dtype=np.int64)
        surf_p1 = np.asarray(face_cells.T, dtype=np.int64) if face_cells.size else np.empty((3, 0), dtype=np.int64)
        coord, elem, surf = _elevate_tet4_mesh_to_tet10(coord_p1, elem_p1, surf_p1)
        boundary = _map_physical_ids(boundary, msh.field_data, 2, "boundary")
        material = _map_physical_ids(tetra_tags, msh.field_data, 3, "material")
        q = build_dirichlet_mask_for_path(
            path,
            dim=3,
            n_nodes=coord.shape[1],
            surf=surf,
            boundary=boundary,
            coord=coord,
            boundary_type=boundary_type,
        )
        return ComsolP2Mesh3D(
            coord=coord,
            elem=elem,
            surf=surf,
            q_mask=q,
            material=np.asarray(material, dtype=np.int64),
            triangle_labels=np.asarray(boundary, dtype=np.int64),
        )

    with h5py.File(str(path), "r") as h5:
        boundary = np.asarray(h5["boundary"][:], dtype=np.int64).ravel()
        elem = _as_nodes_by_count(np.asarray(h5["elem"][:], dtype=np.int64), 10)
        face = _as_nodes_by_count(np.asarray(h5["face"][:], dtype=np.int64), 6)
        material = np.asarray(h5["material"][:], dtype=np.int64).ravel()
        node = _as_nodes_by_count(np.asarray(h5["node"][:], dtype=np.float64), 3)

    coord = np.asarray(node[[0, 2, 1], :], dtype=np.float64)
    q = build_dirichlet_mask_for_path(
        path,
        dim=3,
        n_nodes=coord.shape[1],
        surf=face,
        boundary=boundary,
        coord=coord,
        boundary_type=boundary_type,
    )
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
    """Resolve seepage BCs from the COMSOL mesh-family definition."""

    return build_seepage_boundary_for_path(
        DEFAULT_COMSOL_MESH_PATH,
        coord,
        surf,
        triangle_labels,
        grho=grho,
    )
