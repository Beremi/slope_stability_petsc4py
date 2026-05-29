"""Executable mesh asset public API."""

from __future__ import annotations

from .api import MeshBuildResult, MechanicalProblemSpec, ProblemAssetAPI, ResolvedVariant, SeepageProblemSpec, VariantSpec
from .registry import available_problem_assets, load_asset_from_definition_path, load_problem_asset, meshes_root, repository_root


__all__ = [
    "MeshBuildResult",
    "MechanicalProblemSpec",
    "ProblemAssetAPI",
    "ResolvedVariant",
    "SeepageProblemSpec",
    "VariantSpec",
    "available_problem_assets",
    "load_asset_from_definition_path",
    "load_problem_asset",
    "meshes_root",
    "repository_root",
]
