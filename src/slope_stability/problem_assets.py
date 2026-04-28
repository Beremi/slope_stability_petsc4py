"""Compatibility helpers over executable mesh assets."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np

from .assets.api import MeshBuildResult, ProblemAssetAPI
from .assets.support.selectors import DirichletRule, SeepageDefinition


ProblemAssetDefinition = ProblemAssetAPI
MechanicalDirichletRule = DirichletRule
SeepageSpec = SeepageDefinition

DEFAULT_DIRICHLET_LABELS: dict[int, dict[str, tuple[int, ...]]] = {
    2: {"x": (1, 2), "y": (3,)},
    3: {"x": (1, 2), "y": (5,), "z": (3, 4)},
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def meshes_root() -> Path:
    return repository_root() / "meshes"


def available_problem_assets() -> list[str]:
    root = meshes_root()
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "definition.py").exists())


def _load_module(path: Path) -> ModuleType:
    spec = spec_from_file_location(f"mesh_definition_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load mesh definition module from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_asset_from_definition_path(definition_path: Path, fallback_name: str) -> ProblemAssetAPI:
    module = _load_module(definition_path)
    asset = getattr(module, "ASSET", None)
    if asset is not None:
        return asset
    payload = getattr(module, "DEFINITION", None)
    if isinstance(payload, dict):
        from .assets.compat import asset_from_definition_dict

        return asset_from_definition_dict(payload, asset_dir=definition_path.parent, fallback_name=fallback_name)
    raise ValueError(f"Mesh definition {definition_path} must expose an ASSET object or legacy DEFINITION dictionary.")


def load_problem_asset_definition(name: str) -> ProblemAssetAPI:
    definition_path = meshes_root() / name / "definition.py"
    if not definition_path.exists():
        raise FileNotFoundError(f"No mesh definition registered for {name!r} at {definition_path}")
    return _load_asset_from_definition_path(definition_path, name)


def load_problem_asset_definition_for_path(path: str | Path) -> ProblemAssetAPI | None:
    mesh_path = Path(path).resolve()
    root = meshes_root().resolve()
    try:
        mesh_path.relative_to(root)
    except ValueError:
        return None

    search_roots = (mesh_path, *mesh_path.parents) if mesh_path.is_dir() else (mesh_path.parent, *mesh_path.parents)
    for parent in search_roots:
        if parent == parent.parent:
            break
        definition_path = parent / "definition.py"
        if definition_path.exists():
            return _load_asset_from_definition_path(definition_path, parent.name)
        if parent == root:
            break
    return None


def _resolved_variant_for_asset(asset: ProblemAssetAPI, *, path: str | Path | None = None):
    mesh_path = None if path is None else Path(path).resolve()
    mesh_variant = None if mesh_path is None else mesh_path.name
    return asset.resolve_variant(mesh_variant=mesh_variant, mesh_path=mesh_path)


def _stub_mesh(
    *,
    dim: int,
    n_nodes: int,
    surf: np.ndarray,
    boundary: np.ndarray,
    coord: np.ndarray | None = None,
) -> MeshBuildResult:
    coord_arr = np.zeros((int(dim), int(n_nodes)), dtype=np.float64) if coord is None else np.asarray(coord, dtype=np.float64)
    return MeshBuildResult(
        coord=coord_arr,
        elem=np.empty((0, 0), dtype=np.int64),
        surf=np.asarray(surf, dtype=np.int64),
        q_mask=np.ones((int(dim), int(n_nodes)), dtype=bool),
        material_id=np.empty(0, dtype=np.int64),
        boundary_labels=np.asarray(boundary, dtype=np.int64).ravel(),
        elem_type=None,
    )


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


def load_material_rows_for_asset(name: str) -> list[list[float]] | None:
    asset = load_problem_asset_definition(name)
    if hasattr(asset, "material_rows"):
        return asset.material_rows()
    return None


def load_material_rows_for_path(path: str | Path) -> list[list[float]] | None:
    asset = load_problem_asset_definition_for_path(path)
    if asset is None:
        return None
    if hasattr(asset, "material_rows"):
        return asset.material_rows()
    return None


def load_default_boundary_type_for_asset(name: str) -> int | None:
    return load_default_boundary_type_for_asset_definition(load_problem_asset_definition(name))


def load_default_boundary_type_for_asset_definition(asset: ProblemAssetAPI) -> int | None:
    return int(getattr(asset, "default_boundary_type", 0))


def load_mechanical_dirichlet_rules_for_asset_definition(
    asset: ProblemAssetAPI,
    *,
    dim: int,
    boundary_type: int = 0,
) -> tuple[MechanicalDirichletRule, ...]:
    if int(dim) != int(asset.dimension):
        raise ValueError(f"Requested dimension {dim} does not match asset dimension {asset.dimension} for {asset.asset_id!r}.")
    if hasattr(asset, "dirichlet_rules"):
        return tuple(asset.dirichlet_rules(boundary_type=boundary_type))
    return ()


def load_mechanical_dirichlet_rules_for_path(
    path: str | Path,
    *,
    dim: int,
    boundary_type: int = 0,
) -> tuple[MechanicalDirichletRule, ...]:
    asset = load_problem_asset_definition_for_path(path)
    if asset is None:
        labels = DEFAULT_DIRICHLET_LABELS.get(int(dim), {})
        rules = [DirichletRule(components=(axis,), labels=tuple(axis_labels)) for axis, axis_labels in labels.items() if axis_labels]
        if int(dim) == 3 and int(boundary_type):
            glued = tuple(labels.get("y", ()))
            if glued:
                rules.append(DirichletRule(components=("x", "y", "z"), labels=glued, boundary_types=(int(boundary_type),)))
        return tuple(rules)
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
        raise ValueError(
            f"Cannot build asset-backed Dirichlet mask for {asset.asset_id!r} without target mesh coordinates."
        )
    variant = _resolved_variant_for_asset(asset)
    elem_type = _infer_elem_type_from_surface(int(dim), np.asarray(surf, dtype=np.int64))
    built = asset.build_mesh(variant, elem_type=elem_type)
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
    asset = load_problem_asset_definition_for_path(path)
    if asset is None:
        q = np.ones((int(dim), int(n_nodes)), dtype=bool)
        face = np.asarray(surf, dtype=np.int64)
        labels = np.asarray(boundary, dtype=np.int64).ravel()
        component_to_idx = {name: idx for idx, name in enumerate(("x", "y") if int(dim) == 2 else ("x", "y", "z"))}
        for rule in load_mechanical_dirichlet_rules_for_path(path, dim=dim, boundary_type=boundary_type):
            if not rule.labels:
                continue
            mask = np.isin(labels, np.asarray(rule.labels, dtype=np.int64))
            if not np.any(mask):
                continue
            nodes = face[:, mask].ravel()
            for component in rule.components:
                q[component_to_idx[component], nodes] = False
        return q
    if coord is None:
        raise ValueError(f"Cannot build asset-backed Dirichlet mask for {Path(path)} without target mesh coordinates.")
    variant = _resolved_variant_for_asset(asset, path=path)
    elem_type = _infer_elem_type_from_surface(int(dim), np.asarray(surf, dtype=np.int64))
    built = asset.build_mesh(variant, elem_type=elem_type)
    target_to_ref = _target_to_reference_permutation(np.asarray(built.coord, dtype=np.float64), np.asarray(coord, dtype=np.float64))
    return np.asarray(built.q_mask[:, target_to_ref], dtype=bool)


def load_seepage_spec_for_asset_definition(asset: ProblemAssetAPI, *, required: bool = False) -> SeepageSpec | None:
    if hasattr(asset, "seepage_spec"):
        spec = asset.seepage_spec()
        if spec is None and required:
            raise ValueError(f"Mesh-family seepage definition missing in {asset.asset_dir}.")
        return spec
    if hasattr(asset, "seepage_definition"):
        spec = asset.seepage_definition()
        if spec is None and required:
            raise ValueError(f"Mesh-family seepage definition missing in {asset.asset_dir}.")
        return spec
    if required:
        raise ValueError(f"Mesh-family seepage definition missing in {asset.asset_dir}.")
    return None


def load_seepage_spec_for_path(path: str | Path, *, required: bool = False) -> SeepageSpec | None:
    asset = load_problem_asset_definition_for_path(path)
    if asset is None:
        if required:
            raise ValueError(f"No mesh-family definition found for seepage mesh {Path(path)}.")
        return None
    return load_seepage_spec_for_asset_definition(asset, required=required)


def load_water_unit_weight_for_asset(name: str, *, required: bool = False) -> float | None:
    spec = load_seepage_spec_for_asset_definition(load_problem_asset_definition(name), required=required)
    return None if spec is None else float(spec.water_unit_weight)


def load_water_unit_weight_for_path(path: str | Path, *, required: bool = False) -> float | None:
    spec = load_seepage_spec_for_path(path, required=required)
    return None if spec is None else float(spec.water_unit_weight)


def load_hydraulic_conductivity_for_asset(name: str, *, required: bool = False) -> np.ndarray | None:
    return load_hydraulic_conductivity_for_asset_definition(load_problem_asset_definition(name), required=required)


def load_hydraulic_conductivity_for_asset_definition(
    asset: ProblemAssetAPI,
    *,
    required: bool = False,
) -> np.ndarray | None:
    if hasattr(asset, "hydraulic_conductivity"):
        conductivity = asset.hydraulic_conductivity()
        if conductivity is None and required:
            raise ValueError(f"Mesh-family seepage definition missing in {asset.asset_dir}.")
        return None if conductivity is None else np.asarray(conductivity, dtype=np.float64)
    if required:
        raise ValueError(f"Mesh-family seepage definition missing in {asset.asset_dir}.")
    return None


def load_hydraulic_conductivity_for_path(path: str | Path, *, required: bool = False) -> np.ndarray | None:
    asset = load_problem_asset_definition_for_path(path)
    if asset is None:
        if required:
            raise ValueError(f"No mesh-family definition found for seepage mesh {Path(path)}.")
        return None
    return load_hydraulic_conductivity_for_asset_definition(asset, required=required)


def infer_3d_seepage_boundary_mode_for_asset(name: str) -> str | None:
    return infer_3d_seepage_boundary_mode_for_asset_definition(load_problem_asset_definition(name))


def infer_3d_seepage_boundary_mode_for_asset_definition(asset: ProblemAssetAPI) -> str | None:
    spec = load_seepage_spec_for_asset_definition(asset, required=False)
    if spec is None:
        return None
    mode = str(spec.hydraulic_boundaries.get("mode", "")).strip().lower()
    if mode == "hybrid_transition":
        return "comsol"
    if mode == "label_sets":
        return "waterlevels"
    return None


def infer_3d_seepage_boundary_mode_for_path(path: str | Path) -> str | None:
    asset = load_problem_asset_definition_for_path(path)
    if asset is None:
        return None
    return infer_3d_seepage_boundary_mode_for_asset_definition(asset)


def build_seepage_boundary_for_asset_definition(
    asset: ProblemAssetAPI,
    coord: np.ndarray,
    surf: np.ndarray,
    triangle_labels: np.ndarray,
    *,
    grho: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    variant = _resolved_variant_for_asset(asset)
    elem_type = _infer_elem_type_from_surface(int(asset.dimension), np.asarray(surf, dtype=np.int64))
    built = asset.build_mesh(variant, elem_type=elem_type)
    seepage = asset.build_seepage(built, variant)
    if seepage is None:
        raise ValueError(f"Mesh-family seepage definition missing in {asset.asset_dir}.")
    target_to_ref = _target_to_reference_permutation(np.asarray(built.coord, dtype=np.float64), np.asarray(coord, dtype=np.float64))
    q_w = np.asarray(seepage.q_w, dtype=bool)[target_to_ref]
    pw_d = np.asarray(seepage.pw_d, dtype=np.float64)[target_to_ref]
    if grho is None or float(seepage.water_unit_weight) == 0.0:
        return q_w, pw_d
    scale = float(grho) / float(seepage.water_unit_weight)
    return q_w, pw_d * scale


def build_seepage_boundary_for_path(
    path: str | Path,
    coord: np.ndarray,
    surf: np.ndarray,
    triangle_labels: np.ndarray,
    *,
    grho: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    asset = load_problem_asset_definition_for_path(path)
    if asset is None:
        raise ValueError(f"No mesh-family definition found for seepage mesh {Path(path)}.")
    variant = _resolved_variant_for_asset(asset, path=path)
    elem_type = _infer_elem_type_from_surface(int(asset.dimension), np.asarray(surf, dtype=np.int64))
    built = asset.build_mesh(variant, elem_type=elem_type)
    seepage = asset.build_seepage(built, variant)
    if seepage is None:
        raise ValueError(f"Mesh-family seepage definition missing in {asset.asset_dir}.")
    target_to_ref = _target_to_reference_permutation(np.asarray(built.coord, dtype=np.float64), np.asarray(coord, dtype=np.float64))
    q_w = np.asarray(seepage.q_w, dtype=bool)[target_to_ref]
    pw_d = np.asarray(seepage.pw_d, dtype=np.float64)[target_to_ref]
    if grho is None or float(seepage.water_unit_weight) == 0.0:
        return q_w, pw_d
    scale = float(grho) / float(seepage.water_unit_weight)
    return q_w, pw_d * scale
