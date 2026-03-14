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


def _definition_from_path(definition_path: Path, fallback_name: str) -> ProblemAssetDefinition:
    module = _load_module(definition_path)
    payload = getattr(module, "DEFINITION", None)
    if not isinstance(payload, dict):
        raise ValueError(f"Mesh definition {definition_path} must expose a DEFINITION dictionary")
    return ProblemAssetDefinition(
        name=str(payload.get("name", fallback_name)),
        asset_dir=definition_path.parent,
        payload=dict(payload),
    )


def load_problem_asset_definition(name: str) -> ProblemAssetDefinition:
    definition_path = meshes_root() / name / "definition.py"
    if not definition_path.exists():
        raise FileNotFoundError(f"No mesh definition registered for {name!r} at {definition_path}")
    return _definition_from_path(definition_path, name)


def load_problem_asset_definition_for_path(path: str | Path) -> ProblemAssetDefinition | None:
    mesh_path = Path(path).resolve()
    root = meshes_root().resolve()
    try:
        mesh_path.relative_to(root)
    except ValueError:
        return None

    for parent in (mesh_path.parent, *mesh_path.parents):
        if parent == parent.parent:
            break
        definition_path = parent / "definition.py"
        if definition_path.exists():
            return _definition_from_path(definition_path, parent.name)
        if parent == root:
            break
    return None


def load_material_rows_for_path(path: str | Path) -> list[list[float]] | None:
    asset = load_problem_asset_definition_for_path(path)
    if asset is None:
        return None
    raw = asset.payload.get("materials")
    if not isinstance(raw, list):
        return None

    ordered: dict[int, list[float]] = {}
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Mesh-family materials in {asset.asset_dir} must be dictionaries, got {type(item)!r}.")
        mid = int(item.get("id", idx))
        ordered[mid] = [
            float(item["c0"]),
            float(item["phi"]),
            float(item["psi"]),
            float(item["young"]),
            float(item["poisson"]),
            float(item["gamma_sat"]),
            float(item["gamma_unsat"]),
        ]

    if not ordered:
        return None
    expected = list(range(max(ordered) + 1))
    if sorted(ordered) != expected:
        raise ValueError(f"Mesh-family materials in {asset.asset_dir} must provide contiguous ids {expected}, got {sorted(ordered)}.")
    return [ordered[idx] for idx in expected]
