"""Executable mesh asset loading and discovery."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

from .api import MeshBuildResult, MechanicalProblemSpec, ProblemAssetAPI, ResolvedVariant, SeepageProblemSpec, VariantSpec


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
    raise ValueError(f"Mesh definition {definition_path} must expose an ASSET object.")


def load_problem_asset(name: str) -> ProblemAssetAPI:
    definition_path = meshes_root() / name / "definition.py"
    if not definition_path.exists():
        raise FileNotFoundError(f"No mesh definition registered for {name!r} at {definition_path}")
    return _load_asset_from_definition_path(definition_path, name)


def load_problem_asset_for_path(path: str | Path) -> ProblemAssetAPI | None:
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


__all__ = [
    "MeshBuildResult",
    "MechanicalProblemSpec",
    "ProblemAssetAPI",
    "ResolvedVariant",
    "SeepageProblemSpec",
    "VariantSpec",
    "available_problem_assets",
    "load_problem_asset",
    "load_problem_asset_for_path",
    "meshes_root",
    "repository_root",
]
