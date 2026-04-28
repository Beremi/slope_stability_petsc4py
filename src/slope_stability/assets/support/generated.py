"""Generic generated-geometry helpers."""

from __future__ import annotations

import numpy as np

from ..api import MeshBuildResult, ResolvedVariant


def generated_variant(generator: str, *, parameters: dict) -> dict:
    return {
        "source": {
            "generator": str(generator),
            "parameters": dict(parameters),
        }
    }


def build_generated_mesh(variant: ResolvedVariant, *, elem_type: str) -> MeshBuildResult:
    source = dict(variant.source)
    generator = str(source.get("generator", "")).strip().lower()
    params = dict(source.get("parameters", {}))
    if generator == "homogeneous_slope_2d":
        from ...mesh.slope_2d import generate_homogeneous_slope_mesh_2d

        if "x2" not in params:
            beta_deg = float(params.get("beta_deg", 45.0))
            y2 = float(params.get("y2", 10.0))
            params["x2"] = float(y2 / np.tan(np.deg2rad(beta_deg)))
        params.pop("beta_deg", None)
        mesh = generate_homogeneous_slope_mesh_2d(elem_type=elem_type, **params)
    elif generator == "sloan2013_2d":
        from ...mesh.sloan2013_2d import generate_sloan2013_mesh_2d

        mesh = generate_sloan2013_mesh_2d(elem_type=elem_type, **params)
    else:
        raise ValueError(f"Unsupported generated asset source {generator!r} for {variant.asset_id!r}.")

    return MeshBuildResult(
        coord=np.asarray(mesh.coord, dtype=np.float64),
        elem=np.asarray(mesh.elem, dtype=np.int64),
        surf=np.asarray(mesh.surf, dtype=np.int64),
        q_mask=np.asarray(mesh.q_mask, dtype=bool),
        material_id=np.asarray(mesh.material, dtype=np.int64),
        boundary_labels=np.zeros(int(mesh.surf.shape[1]), dtype=np.int64),
        elem_type=str(elem_type).upper(),
    )
