"""Mesh dataset loader abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import load_mesh_file, load_mesh_p2


@dataclass(frozen=True)
class LoadedMesh:
    coord: np.ndarray
    elem: np.ndarray
    surf: np.ndarray
    q_mask: np.ndarray
    material: np.ndarray
    boundary: np.ndarray
    elem_type: str | None = None


def load_mesh_from_file(path: str | Path, *, elem_type: str | None = None) -> LoadedMesh:
    path = Path(path)
    if path.suffix.lower() == ".h5":
        data = load_mesh_p2(path)
    else:
        data = load_mesh_file(path, elem_type=elem_type)
    return LoadedMesh(
        coord=np.asarray(data.coord, dtype=np.float64),
        elem=np.asarray(data.elem, dtype=np.int64),
        surf=np.asarray(data.surf, dtype=np.int64),
        q_mask=np.asarray(data.q_mask, dtype=bool),
        material=np.asarray(data.material, dtype=np.int64),
        boundary=np.asarray(data.boundary, dtype=np.int64),
        elem_type=data.elem_type,
    )
