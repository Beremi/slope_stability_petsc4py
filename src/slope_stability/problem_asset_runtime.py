"""Runtime asset resolution and mesh construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .assets import MeshBuildResult, ProblemAssetAPI, ResolvedVariant, load_problem_asset, load_problem_asset_for_path
from .problem_assets import (
    load_hydraulic_conductivity_for_asset_definition,
    load_material_rows_for_asset,
    load_mechanical_dirichlet_rules_for_asset_definition,
    load_seepage_spec_for_asset_definition,
)


@dataclass(frozen=True)
class ResolvedAsset:
    definition: ProblemAssetAPI
    variant_name: str
    variant: dict[str, Any]
    resolved_variant: ResolvedVariant
    mesh_path: Path | None
    boundary_type: int

    @property
    def asset_name(self) -> str:
        return str(self.definition.asset_id)

    @property
    def dimension(self) -> int:
        return int(self.definition.dimension)

    @property
    def source_kind(self) -> str:
        return str(getattr(self.definition, "source_kind", ""))


@dataclass(frozen=True)
class MechanicalProblemSpec:
    material_rows: list[list[float]]
    dirichlet_rules: tuple[Any, ...]
    boundary_type: int


@dataclass(frozen=True)
class SeepageProblemSpec:
    seepage: Any
    conductivity: np.ndarray


def resolve_problem_asset(
    *,
    asset_name: str,
    mesh_variant: str | None = None,
    mesh_path: str | Path | None = None,
    profile: str | None = None,
    boundary_type: int | None = None,
) -> ResolvedAsset:
    asset = load_problem_asset(asset_name)
    resolved_variant = asset.resolve_variant(
        mesh_variant=None if mesh_variant is None else str(mesh_variant),
        mesh_path=None if mesh_path is None else Path(mesh_path).resolve(),
        profile=profile,
    )
    return ResolvedAsset(
        definition=asset,
        variant_name=str(resolved_variant.name),
        variant=resolved_variant.as_dict(),
        resolved_variant=resolved_variant,
        mesh_path=resolved_variant.mesh_path,
        boundary_type=int(resolved_variant.boundary_type),
    )


def resolve_problem_asset_from_config(cfg) -> ResolvedAsset:
    problem = cfg.problem
    asset_name = getattr(problem, "asset", None)
    mesh_variant = getattr(problem, "mesh_variant", None)
    profile = getattr(problem, "profile", None)

    if asset_name:
        return resolve_problem_asset(
            asset_name=str(asset_name),
            mesh_variant=None if mesh_variant is None else str(mesh_variant),
            profile=None if profile is None else str(profile),
        )

    raise KeyError("Could not resolve a problem asset from config; set [problem].asset.")


def build_mesh_for_resolved_asset(resolved: ResolvedAsset, *, elem_type: str) -> MeshBuildResult:
    return resolved.definition.build_mesh(resolved.resolved_variant, elem_type=str(elem_type))


def build_mesh_for_path(
    path: str | Path,
    *,
    elem_type: str,
    profile: str | None = None,
    boundary_type: int = 0,
) -> MeshBuildResult:
    path = Path(path).resolve()
    asset = load_problem_asset_for_path(path)
    if asset is None:
        raise ValueError(f"No registered problem asset owns mesh path {path}.")
    resolved = resolve_problem_asset(
        asset_name=str(asset.asset_id),
        mesh_variant=path.name,
        mesh_path=path,
        profile=profile,
        boundary_type=boundary_type,
    )
    return build_mesh_for_resolved_asset(resolved, elem_type=elem_type)


def load_mechanical_problem_spec(resolved: ResolvedAsset) -> MechanicalProblemSpec:
    rows = load_material_rows_for_asset(resolved.asset_name)
    if rows is None:
        raise ValueError(f"No mechanical materials registered for asset {resolved.asset_name!r}.")
    return MechanicalProblemSpec(
        material_rows=rows,
        dirichlet_rules=load_mechanical_dirichlet_rules_for_asset_definition(
            resolved.definition,
            dim=resolved.dimension,
            boundary_type=resolved.boundary_type,
        ),
        boundary_type=resolved.boundary_type,
    )


def load_seepage_problem_spec(resolved: ResolvedAsset) -> SeepageProblemSpec:
    spec = load_seepage_spec_for_asset_definition(resolved.definition, required=True)
    if spec is None:
        raise ValueError(f"No seepage definition registered for asset {resolved.asset_name!r}.")
    conductivity = load_hydraulic_conductivity_for_asset_definition(resolved.definition, required=True)
    assert conductivity is not None
    return SeepageProblemSpec(seepage=spec, conductivity=np.asarray(conductivity, dtype=np.float64))


def build_seepage_boundary_for_resolved_asset(
    resolved: ResolvedAsset,
    coord: np.ndarray,
    surf: np.ndarray,
    boundary_labels: np.ndarray | None,
    *,
    grho: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    surf_arr = np.asarray(surf, dtype=np.int64)
    width = int(surf_arr.shape[0])
    if resolved.dimension == 2:
        elem_type = {2: "P1", 3: "P2", 5: "P4"}.get(width)
    else:
        elem_type = {3: "P1", 6: "P2", 10: "P3", 15: "P4"}.get(width)
    if elem_type is None:
        raise ValueError(f"Cannot infer element type from surface width {width} for dimension {resolved.dimension}.")
    built = resolved.definition.build_mesh(resolved.resolved_variant, elem_type=elem_type)
    seepage = resolved.definition.build_seepage(built, resolved.resolved_variant)
    if seepage is None:
        raise ValueError(f"No seepage definition registered for asset {resolved.asset_name!r}.")
    ref = np.asarray(built.coord, dtype=np.float64)
    tgt = np.asarray(coord, dtype=np.float64)
    if ref.shape != tgt.shape:
        raise ValueError(f"Coordinate shapes do not match: reference {ref.shape}, target {tgt.shape}.")

    def _key(column: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(column, dtype=np.float64), 12))

    lookup = {_key(ref[:, idx]): idx for idx in range(ref.shape[1])}
    target_to_ref = np.asarray([lookup[_key(tgt[:, idx])] for idx in range(tgt.shape[1])], dtype=np.int64)
    q_w = np.asarray(seepage.q_w, dtype=bool)[target_to_ref]
    pw_d = np.asarray(seepage.pw_d, dtype=np.float64)[target_to_ref]
    if grho is None:
        return q_w, pw_d
    scale = float(grho) / float(seepage.water_unit_weight) if seepage.water_unit_weight else 1.0
    return q_w, pw_d * scale
