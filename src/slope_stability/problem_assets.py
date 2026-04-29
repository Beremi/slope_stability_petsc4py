"""Generic facade over executable mesh assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .assets import (
    MeshBuildResult,
    ProblemAssetAPI,
    available_problem_assets,
    load_problem_asset,
    load_problem_asset_for_path,
    meshes_root,
    repository_root,
)
from .assets.api import DirichletBCSpec, SeepageSpec


ProblemAssetDefinition = ProblemAssetAPI
MechanicalDirichletRule = DirichletBCSpec


def load_problem_asset_definition(name: str) -> ProblemAssetAPI:
    return load_problem_asset(name)


def load_problem_asset_definition_for_path(path: str | Path) -> ProblemAssetAPI | None:
    return load_problem_asset_for_path(path)


def _resolved_variant_for_asset(asset: ProblemAssetAPI, *, path: str | Path | None = None):
    mesh_path = None if path is None else Path(path).resolve()
    mesh_variant = None if mesh_path is None else mesh_path.name
    return asset.resolve_variant(mesh_variant=mesh_variant, mesh_path=mesh_path)


def _infer_elem_type_from_surface(dim: int, surf: np.ndarray) -> str:
    width = int(np.asarray(surf, dtype=np.int64).shape[0])
    if int(dim) == 2:
        mapping = {2: "P1", 3: "P2", 5: "P4"}
    else:
        mapping = {3: "P1", 6: "P2", 10: "P3", 15: "P4"}
    if width not in mapping:
        raise ValueError(f"Cannot infer element type from surface width {width} in {dim}D.")
    return mapping[width]


def _target_to_reference_permutation(reference_coord: np.ndarray, target_coord: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference_coord, dtype=np.float64)
    tgt = np.asarray(target_coord, dtype=np.float64)
    if ref.shape != tgt.shape:
        raise ValueError(f"Coordinate shapes do not match: reference {ref.shape}, target {tgt.shape}.")

    def _key(column: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(column, dtype=np.float64), 12))

    lookup: dict[tuple[float, ...], int] = {}
    for idx in range(ref.shape[1]):
        key = _key(ref[:, idx])
        if key in lookup:
            raise ValueError("Reference mesh contains duplicate node coordinates; cannot infer permutation robustly.")
        lookup[key] = idx

    out = np.empty(tgt.shape[1], dtype=np.int64)
    for idx in range(tgt.shape[1]):
        key = _key(tgt[:, idx])
        if key not in lookup:
            raise KeyError(f"Target coordinate {key} was not found in the reference mesh.")
        out[idx] = lookup[key]
    return out


def _build_asset_mesh_for_surface(asset: ProblemAssetAPI, *, path: str | Path | None, dim: int, surf: np.ndarray) -> tuple[MeshBuildResult, object]:
    if int(dim) != int(asset.dimension):
        raise ValueError(f"Requested dimension {dim} does not match asset dimension {asset.dimension} for {asset.asset_id!r}.")
    variant = _resolved_variant_for_asset(asset, path=path)
    elem_type = _infer_elem_type_from_surface(int(dim), np.asarray(surf, dtype=np.int64))
    return asset.build_mesh(variant, elem_type=elem_type), variant


def load_material_rows_for_asset(name: str) -> list[list[float]] | None:
    return load_problem_asset(name).material_rows()


def load_material_rows_for_path(path: str | Path) -> list[list[float]] | None:
    asset = load_problem_asset_for_path(path)
    if asset is None:
        raise ValueError(f"No registered problem asset owns mesh path {Path(path).resolve()}.")
    return asset.material_rows()


def load_default_boundary_type_for_asset(name: str) -> int | None:
    return load_default_boundary_type_for_asset_definition(load_problem_asset(name))


def load_default_boundary_type_for_asset_definition(asset: ProblemAssetAPI) -> int | None:
    return int(asset.resolve_variant(mesh_variant=None).boundary_type)


def load_mechanical_dirichlet_rules_for_asset_definition(
    asset: ProblemAssetAPI,
    *,
    dim: int,
    boundary_type: int = 0,
) -> tuple[MechanicalDirichletRule, ...]:
    if int(dim) != int(asset.dimension):
        raise ValueError(f"Requested dimension {dim} does not match asset dimension {asset.dimension} for {asset.asset_id!r}.")
    mechanics = asset.mechanics_spec()
    if mechanics is None:
        return ()
    profile = mechanics.profiles[str(asset.resolve_variant(mesh_variant=None).profile)]
    return tuple(profile.dirichlet)


def load_mechanical_dirichlet_rules_for_path(
    path: str | Path,
    *,
    dim: int,
    boundary_type: int = 0,
) -> tuple[MechanicalDirichletRule, ...]:
    asset = load_problem_asset_for_path(path)
    if asset is None:
        raise ValueError(f"No registered problem asset owns mesh path {Path(path).resolve()}.")
    return load_mechanical_dirichlet_rules_for_asset_definition(asset, dim=dim, boundary_type=boundary_type)


def build_dirichlet_mask_for_asset_definition(
    asset: ProblemAssetAPI,
    *,
    dim: int,
    n_nodes: int,
    surf: np.ndarray,
    boundary: np.ndarray,
    coord: np.ndarray | None = None,
    boundary_type: int = 0,
) -> np.ndarray:
    if coord is None:
        raise ValueError(f"Cannot build asset-backed Dirichlet mask for {asset.asset_id!r} without target mesh coordinates.")
    built, _variant = _build_asset_mesh_for_surface(asset, path=None, dim=dim, surf=surf)
    target_to_ref = _target_to_reference_permutation(np.asarray(built.coord, dtype=np.float64), np.asarray(coord, dtype=np.float64))
    return np.asarray(built.q_mask[:, target_to_ref], dtype=bool)


def build_dirichlet_mask_for_path(
    path: str | Path,
    *,
    dim: int,
    n_nodes: int,
    surf: np.ndarray,
    boundary: np.ndarray,
    coord: np.ndarray | None = None,
    boundary_type: int = 0,
) -> np.ndarray:
    asset = load_problem_asset_for_path(path)
    if asset is None:
        raise ValueError(f"No registered problem asset owns mesh path {Path(path).resolve()}.")
    if coord is None:
        raise ValueError(f"Cannot build asset-backed Dirichlet mask for {Path(path)} without target mesh coordinates.")
    built, _variant = _build_asset_mesh_for_surface(asset, path=path, dim=dim, surf=surf)
    target_to_ref = _target_to_reference_permutation(np.asarray(built.coord, dtype=np.float64), np.asarray(coord, dtype=np.float64))
    return np.asarray(built.q_mask[:, target_to_ref], dtype=bool)


def load_seepage_spec_for_asset_definition(asset: ProblemAssetAPI, *, required: bool = False) -> SeepageSpec | None:
    spec = asset.seepage_spec()
    if spec is None and required:
        raise ValueError(f"Mesh asset {asset.asset_id!r} does not define seepage physics.")
    return spec


def load_seepage_spec_for_path(path: str | Path, *, required: bool = False) -> SeepageSpec | None:
    asset = load_problem_asset_for_path(path)
    if asset is None:
        if required:
            raise ValueError(f"No registered problem asset owns mesh path {Path(path).resolve()}.")
        return None
    return load_seepage_spec_for_asset_definition(asset, required=required)


def load_water_unit_weight_for_asset(name: str, *, required: bool = False) -> float | None:
    spec = load_seepage_spec_for_asset_definition(load_problem_asset(name), required=required)
    return None if spec is None else float(spec.water_unit_weight)


def load_water_unit_weight_for_path(path: str | Path, *, required: bool = False) -> float | None:
    spec = load_seepage_spec_for_path(path, required=required)
    return None if spec is None else float(spec.water_unit_weight)


def load_hydraulic_conductivity_for_asset(name: str, *, required: bool = False) -> np.ndarray | None:
    return load_hydraulic_conductivity_for_asset_definition(load_problem_asset(name), required=required)


def load_hydraulic_conductivity_for_asset_definition(
    asset: ProblemAssetAPI,
    *,
    required: bool = False,
) -> np.ndarray | None:
    conductivity = asset.hydraulic_conductivity()
    if conductivity is None and required:
        raise ValueError(f"Mesh asset {asset.asset_id!r} does not define seepage conductivity.")
    return None if conductivity is None else np.asarray(conductivity, dtype=np.float64)


def load_hydraulic_conductivity_for_path(path: str | Path, *, required: bool = False) -> np.ndarray | None:
    asset = load_problem_asset_for_path(path)
    if asset is None:
        if required:
            raise ValueError(f"No registered problem asset owns mesh path {Path(path).resolve()}.")
        return None
    return load_hydraulic_conductivity_for_asset_definition(asset, required=required)


def infer_3d_seepage_boundary_mode_for_asset(name: str) -> str | None:
    return None


def infer_3d_seepage_boundary_mode_for_asset_definition(asset: ProblemAssetAPI) -> str | None:
    return None


def infer_3d_seepage_boundary_mode_for_path(path: str | Path) -> str | None:
    return None


def build_seepage_boundary_for_asset_definition(
    asset: ProblemAssetAPI,
    coord: np.ndarray,
    surf: np.ndarray,
    triangle_labels: np.ndarray,
    *,
    grho: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    built, variant = _build_asset_mesh_for_surface(asset, path=None, dim=int(asset.dimension), surf=surf)
    seepage = asset.build_seepage(built, variant)
    if seepage is None:
        raise ValueError(f"Mesh asset {asset.asset_id!r} does not define seepage physics.")
    target_to_ref = _target_to_reference_permutation(np.asarray(built.coord, dtype=np.float64), np.asarray(coord, dtype=np.float64))
    q_w = np.asarray(seepage.q_w, dtype=bool)[target_to_ref]
    pw_d = np.asarray(seepage.pw_d, dtype=np.float64)[target_to_ref]
    if grho is None or float(seepage.water_unit_weight) == 0.0:
        return q_w, pw_d
    return q_w, pw_d * (float(grho) / float(seepage.water_unit_weight))


def build_seepage_boundary_for_path(
    path: str | Path,
    coord: np.ndarray,
    surf: np.ndarray,
    triangle_labels: np.ndarray,
    *,
    grho: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    asset = load_problem_asset_for_path(path)
    if asset is None:
        raise ValueError(f"No registered problem asset owns mesh path {Path(path).resolve()}.")
    built, variant = _build_asset_mesh_for_surface(asset, path=path, dim=int(asset.dimension), surf=surf)
    seepage = asset.build_seepage(built, variant)
    if seepage is None:
        raise ValueError(f"Mesh asset {asset.asset_id!r} does not define seepage physics.")
    target_to_ref = _target_to_reference_permutation(np.asarray(built.coord, dtype=np.float64), np.asarray(coord, dtype=np.float64))
    q_w = np.asarray(seepage.q_w, dtype=bool)[target_to_ref]
    pw_d = np.asarray(seepage.pw_d, dtype=np.float64)[target_to_ref]
    if grho is None or float(seepage.water_unit_weight) == 0.0:
        return q_w, pw_d
    return q_w, pw_d * (float(grho) / float(seepage.water_unit_weight))
