"""Generic text-bundle mesh helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..api import MeshBuildResult, ResolvedVariant


def textmesh_variant(
    *,
    coordinates: str,
    elements: str,
    materials: str,
    shift: tuple[float, float] = (0.0, 0.0),
    y_floor: float = 0.0,
) -> dict:
    return {
        "source": {
            "coordinates": coordinates,
            "elements": elements,
            "materials": materials,
            "shift": list(shift),
            "y_floor": float(y_floor),
        }
    }


def build_textmesh_bundle_mesh(asset_dir: Path, variant: ResolvedVariant, *, elem_type: str) -> MeshBuildResult:
    from ...mesh.textmesh_2d import load_text_mesh_bundle

    source = dict(variant.source)
    mesh = load_text_mesh_bundle(
        elem_type,
        asset_dir,
        coordinates_name=str(source["coordinates"]),
        elements_name=str(source["elements"]),
        materials_name=str(source["materials"]),
        shift=tuple(float(v) for v in source.get("shift", (0.0, 0.0))),
        y_floor=float(source.get("y_floor", 0.0)),
    )
    return MeshBuildResult(
        coord=np.asarray(mesh.coord, dtype=np.float64),
        elem=np.asarray(mesh.elem, dtype=np.int64),
        surf=np.asarray(mesh.surf, dtype=np.int64),
        q_mask=np.asarray(mesh.q_mask, dtype=bool),
        material_id=np.asarray(mesh.material, dtype=np.int64),
        boundary_labels=np.zeros(int(mesh.surf.shape[1]), dtype=np.int64),
        elem_type=str(elem_type).upper(),
    )
