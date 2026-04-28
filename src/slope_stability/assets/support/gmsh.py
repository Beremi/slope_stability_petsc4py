"""Generic tagged-Gmsh mesh helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..api import MeshBuildResult, ResolvedVariant


def gmsh_variants_from_dir(asset_dir: Path, pattern: str = "*.msh") -> dict[str, dict]:
    return {path.name: {"source": {"path": path.name}} for path in sorted(asset_dir.glob(pattern))}


def build_generic_gmsh_mesh(variant: ResolvedVariant, *, elem_type: str) -> MeshBuildResult:
    if variant.mesh_path is None:
        raise ValueError(f"Asset {variant.asset_id!r} variant {variant.name!r} does not define a mesh file path.")
    from ...mesh.loader import load_mesh_from_file

    mesh = load_mesh_from_file(variant.mesh_path, boundary_type=int(variant.boundary_type), elem_type=elem_type)
    return MeshBuildResult(
        coord=np.asarray(mesh.coord, dtype=np.float64),
        elem=np.asarray(mesh.elem, dtype=np.int64),
        surf=np.asarray(mesh.surf, dtype=np.int64),
        q_mask=np.asarray(mesh.q_mask, dtype=bool),
        material_id=np.asarray(mesh.material, dtype=np.int64),
        boundary_labels=np.asarray(mesh.boundary, dtype=np.int64),
        elem_type=mesh.elem_type,
    )


def build_waterlevels_gmsh_mesh(variant: ResolvedVariant, *, elem_type: str) -> MeshBuildResult:
    if variant.mesh_path is None:
        raise ValueError(f"Asset {variant.asset_id!r} variant {variant.name!r} does not define a mesh file path.")
    from ...mesh.gmsh_waterlevels import load_mesh_gmsh_waterlevels

    mesh = load_mesh_gmsh_waterlevels(variant.mesh_path, elem_type=elem_type)
    return MeshBuildResult(
        coord=np.asarray(mesh.coord, dtype=np.float64),
        elem=np.asarray(mesh.elem, dtype=np.int64),
        surf=np.asarray(mesh.surf, dtype=np.int64),
        q_mask=np.asarray(mesh.q_mask, dtype=bool),
        material_id=np.asarray(mesh.material, dtype=np.int64),
        boundary_labels=np.asarray(mesh.triangle_labels, dtype=np.int64),
        elem_type=str(elem_type).upper(),
    )


def build_comsol_p2_mesh(variant: ResolvedVariant, *, elem_type: str) -> MeshBuildResult:
    if variant.mesh_path is None:
        raise ValueError(f"Asset {variant.asset_id!r} variant {variant.name!r} does not define a mesh file path.")
    from ...mesh.comsol_p2 import load_mesh_p2_comsol

    mesh = load_mesh_p2_comsol(variant.mesh_path, boundary_type=int(variant.boundary_type))
    return MeshBuildResult(
        coord=np.asarray(mesh.coord, dtype=np.float64),
        elem=np.asarray(mesh.elem, dtype=np.int64),
        surf=np.asarray(mesh.surf, dtype=np.int64),
        q_mask=np.asarray(mesh.q_mask, dtype=bool),
        material_id=np.asarray(mesh.material, dtype=np.int64),
        boundary_labels=np.asarray(mesh.triangle_labels, dtype=np.int64),
        elem_type=str(elem_type).upper(),
    )
