"""Core types for canonical executable problem assets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


MECHANICAL_MATERIAL_FIELDS: tuple[str, ...] = (
    "c0",
    "phi",
    "psi",
    "young",
    "poisson",
    "gamma_sat",
    "gamma_unsat",
)


@dataclass(frozen=True)
class MeshVariant:
    name: str
    source: dict[str, Any]
    mesh_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload["source"] = dict(self.source)
        return payload


VariantSpec = MeshVariant


@dataclass(frozen=True)
class ResolvedVariant:
    asset_id: str
    name: str
    source: dict[str, Any]
    mesh_path: Path | None
    metadata: dict[str, Any] = field(default_factory=dict)
    profile: str = "default"
    boundary_type: int = 0

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload["source"] = dict(self.source)
        payload["profile"] = str(self.profile)
        return payload


@dataclass(frozen=True)
class MaterialModelSpec:
    name: str
    parameters: dict[str, float]
    hydraulic_conductivity: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def mechanical_row(self) -> list[float] | None:
        if not any(field in self.parameters for field in MECHANICAL_MATERIAL_FIELDS):
            return None
        missing = [field for field in MECHANICAL_MATERIAL_FIELDS if field not in self.parameters]
        if missing:
            raise ValueError(
                f"Material model {self.name!r} mixes mechanical and non-mechanical fields; missing {missing}."
            )
        return [float(self.parameters[field]) for field in MECHANICAL_MATERIAL_FIELDS]


@dataclass(frozen=True)
class DirichletBCSpec:
    target: str
    components: tuple[str, ...]
    values: tuple[float, ...] | None = None


@dataclass(frozen=True)
class NeumannBCSpec:
    target: str
    kind: str
    value_model: dict[str, Any]
    geometry: str | None = None


@dataclass(frozen=True)
class HeadBCSpec:
    target: str
    kind: str
    value_model: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HydraulicStateSpec:
    kind: str
    value_model: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    dirichlet: tuple[DirichletBCSpec, ...] = ()
    neumann: tuple[NeumannBCSpec, ...] = ()


@dataclass(frozen=True)
class MechanicsSpec:
    materials: dict[str, MaterialModelSpec]
    region_assignment: dict[str, str]
    profiles: dict[str, ProfileSpec]
    default_profile: str = "default"
    hydraulic_state: HydraulicStateSpec | None = None


@dataclass(frozen=True)
class SeepageSpec:
    water_unit_weight: float
    conductivity_mode: str
    conductivity: tuple[float, ...] | None = None
    region_conductivity: dict[str, float] = field(default_factory=dict)
    head_bcs: tuple[HeadBCSpec, ...] = ()
    flux_bcs: tuple[NeumannBCSpec, ...] = ()


@dataclass(frozen=True)
class BoundaryGeometrySpec:
    name: str
    support_boundary: str
    element_family: str = "boundary_simplex"
    geometry_order: int = 2
    coupling_mode: str = "shared_corner_simplex"


@dataclass(frozen=True)
class BoundaryGeometryPatch:
    name: str
    cell_type: str
    corner_nodes: np.ndarray
    control_points: np.ndarray


@dataclass(frozen=True)
class CanonicalMesh:
    coord: np.ndarray
    elem: np.ndarray
    surf: np.ndarray
    region_name_by_elem: tuple[str, ...]
    boundary_name_by_entity: tuple[str, ...]
    region_groups: dict[str, np.ndarray]
    boundary_groups: dict[str, np.ndarray]
    nodesets: dict[str, np.ndarray]
    boundary_geometry: dict[str, BoundaryGeometryPatch]


@dataclass(frozen=True)
class SolverMesh:
    coord: np.ndarray
    elem: np.ndarray
    surf: np.ndarray
    q_mask: np.ndarray
    material_id: np.ndarray
    boundary_labels: np.ndarray
    elem_type: str | None
    region_id_by_name: dict[str, int] = field(default_factory=dict)
    boundary_id_by_name: dict[str, int] = field(default_factory=dict)
    boundary_groups: dict[str, np.ndarray] = field(default_factory=dict)
    nodesets: dict[str, np.ndarray] = field(default_factory=dict)
    boundary_geometry: dict[str, BoundaryGeometryPatch] = field(default_factory=dict)

    @property
    def material(self) -> np.ndarray:
        return self.material_id

    @property
    def boundary(self) -> np.ndarray:
        return self.boundary_labels


MeshBuildResult = SolverMesh


@dataclass(frozen=True)
class MechanicalProblemSpec:
    materials: list[list[float]]
    q_mask: np.ndarray
    profile: str = "default"
    boundary_type: int = 0


@dataclass(frozen=True)
class SeepageProblemSpec:
    water_unit_weight: float
    conductivity: np.ndarray
    q_w: np.ndarray
    pw_d: np.ndarray


class ProblemAssetAPI(ABC):
    asset_id: str
    dimension: int
    capabilities: frozenset[str]
    default_variant: str
    default_profile: str
    source_kind: str
    asset_dir: Path

    @abstractmethod
    def list_variants(self) -> dict[str, MeshVariant]:
        raise NotImplementedError

    @abstractmethod
    def resolve_variant(
        self,
        mesh_variant: str | None,
        mesh_path: Path | None = None,
        *,
        profile: str | None = None,
    ) -> ResolvedVariant:
        raise NotImplementedError

    @abstractmethod
    def build_mesh(self, variant: ResolvedVariant, *, elem_type: str) -> SolverMesh:
        raise NotImplementedError

    @abstractmethod
    def build_mechanics(
        self,
        mesh: SolverMesh,
        variant: ResolvedVariant,
    ) -> MechanicalProblemSpec | None:
        raise NotImplementedError

    @abstractmethod
    def build_seepage(
        self,
        mesh: SolverMesh,
        variant: ResolvedVariant,
    ) -> SeepageProblemSpec | None:
        raise NotImplementedError

    @abstractmethod
    def material_rows(self) -> list[list[float]] | None:
        raise NotImplementedError

    @abstractmethod
    def mechanics_spec(self) -> MechanicsSpec | None:
        raise NotImplementedError

    @abstractmethod
    def seepage_spec(self) -> SeepageSpec | None:
        raise NotImplementedError
