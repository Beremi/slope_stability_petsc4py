"""Discovery and loading for executable mesh assets."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

from .api import ProblemAssetAPI


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def meshes_root() -> Path:
    return repository_root() / "meshes"


def available_problem_assets() -> list[str]:
    root = meshes_root()
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "definition.py").exists())


def load_problem_asset(name: str) -> ProblemAssetAPI:
    definition_path = meshes_root() / name / "definition.py"
    if not definition_path.exists():
        raise FileNotFoundError(f"No mesh definition registered for {name!r} at {definition_path}")
    return load_asset_from_definition_path(definition_path, name)


def load_asset_from_definition_path(definition_path: str | Path, fallback_name: str | None = None) -> ProblemAssetAPI:
    path = Path(definition_path)
    module = _load_module(path)
    asset = getattr(module, "ASSET", None)
    if asset is not None:
        return asset
    name = fallback_name or path.parent.name
    raise ValueError(f"Mesh definition for {name!r} at {path} must expose an ASSET object.")


def _load_module(path: Path) -> ModuleType:
    spec = spec_from_file_location(f"mesh_definition_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load mesh definition module from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


__all__ = [
    "available_problem_assets",
    "load_asset_from_definition_path",
    "load_problem_asset",
    "meshes_root",
    "repository_root",
]
