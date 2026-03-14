"""Temporary loader for mesh-family definitions under the repository ``meshes/`` tree."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class ProblemAssetDefinition:
    name: str
    asset_dir: Path
    payload: dict[str, Any]


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


def load_problem_asset_definition(name: str) -> ProblemAssetDefinition:
    definition_path = meshes_root() / name / "definition.py"
    if not definition_path.exists():
        raise FileNotFoundError(f"No mesh definition registered for {name!r} at {definition_path}")
    module = _load_module(definition_path)
    payload = getattr(module, "DEFINITION", None)
    if not isinstance(payload, dict):
        raise ValueError(f"Mesh definition {definition_path} must expose a DEFINITION dictionary")
    return ProblemAssetDefinition(
        name=str(payload.get("name", name)),
        asset_dir=definition_path.parent,
        payload=dict(payload),
    )
