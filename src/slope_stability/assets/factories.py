"""Factories for canonical `.msh`-only problem assets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .api import (
    DirichletBCSpec,
    HeadBCSpec,
    HydraulicStateSpec,
    MaterialModelSpec,
    MechanicalProblemSpec,
    MechanicsSpec,
    MeshVariant,
    NeumannBCSpec,
    ProblemAssetAPI,
    ProfileSpec,
    ResolvedVariant,
    SeepageProblemSpec,
    SeepageSpec,
    SolverMesh,
)
from .evaluators import head_values
from .support.canonical_gmsh import build_solver_mesh, gmsh_variants_from_dir, load_canonical_gmsh_mesh


AXES_BY_DIM: dict[int, tuple[str, ...]] = {
    2: ("x", "y"),
    3: ("x", "y", "z"),
}


def _logical_name(value: str, *, prefixes: tuple[str, ...] = ("region", "boundary", "nodeset", "boundary_geom")) -> str:
    text = str(value).strip()
    for prefix in prefixes:
        head = f"{prefix}:"
        if text.lower().startswith(head):
            return text.split(":", 1)[1]
    return text


def _normalize_components(value: Any, *, dim: int, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field_name} must be a non-empty list of components.")
    allowed = set(AXES_BY_DIM[int(dim)])
    components = tuple(str(item).strip().lower() for item in value)
    invalid = sorted(set(components) - allowed)
    if invalid:
        raise ValueError(f"{field_name} contains invalid components {invalid}; expected subset of {sorted(allowed)}.")
    return tuple(dict.fromkeys(components))


def _normalize_material_models(materials: dict[str, Any] | list[dict[str, Any]]) -> dict[str, MaterialModelSpec]:
    entries: list[tuple[str, dict[str, Any]]] = []
    if isinstance(materials, dict):
        entries = [(str(name), dict(payload)) for name, payload in materials.items()]
    elif isinstance(materials, list):
        for idx, item in enumerate(materials):
            if not isinstance(item, dict):
                raise ValueError(f"materials[{idx}] must be a dictionary.")
            name = str(item.get("name", f"material_{idx}"))
            entries.append((name, dict(item)))
    else:
        raise ValueError("materials must be a dictionary or list of dictionaries.")

    out: dict[str, MaterialModelSpec] = {}
    for name, payload in entries:
        parameters = {
            key: float(value)
            for key, value in payload.items()
            if key not in {"name", "hydraulic_conductivity"}
        }
        out[name] = MaterialModelSpec(
            name=name,
            parameters=parameters,
            hydraulic_conductivity=(
                None if payload.get("hydraulic_conductivity") is None else float(payload["hydraulic_conductivity"])
            ),
        )
    if not out:
        raise ValueError("At least one material model is required.")
    return out


def _normalize_dirichlet(raw: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None, *, dim: int) -> tuple[DirichletBCSpec, ...]:
    if raw is None:
        return ()
    out: list[DirichletBCSpec] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"dirichlet[{idx}] must be a dictionary.")
        target = str(item.get("target") or item.get("boundary") or item.get("nodeset") or "")
        if not target:
            raise ValueError(f"dirichlet[{idx}] must define a target.")
        components = _normalize_components(item.get("components"), dim=dim, field_name=f"dirichlet[{idx}].components")
        values = item.get("values")
        if values is not None:
            if not isinstance(values, (list, tuple)) or len(values) != len(components):
                raise ValueError(f"dirichlet[{idx}].values must match the number of components.")
            values = tuple(float(v) for v in values)
        out.append(DirichletBCSpec(target=_logical_name(target), components=components, values=values))
    return tuple(out)


def _normalize_neumann(raw: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> tuple[NeumannBCSpec, ...]:
    if raw is None:
        return ()
    out: list[NeumannBCSpec] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"neumann[{idx}] must be a dictionary.")
        target = str(item.get("target") or item.get("boundary") or "")
        kind = str(item.get("kind") or "").strip()
        if not target or not kind:
            raise ValueError(f"neumann[{idx}] must define target and kind.")
        out.append(
            NeumannBCSpec(
                target=_logical_name(target),
                kind=kind,
                value_model=dict(item.get("value_model", {})),
                geometry=None if item.get("geometry") is None else _logical_name(str(item["geometry"]), prefixes=("boundary_geom",)),
            )
        )
    return tuple(out)


def _normalize_profile_map(
    raw: dict[str, dict[str, Any]] | None,
    *,
    dim: int,
    default_dirichlet: tuple[DirichletBCSpec, ...],
    default_neumann: tuple[NeumannBCSpec, ...],
) -> dict[str, ProfileSpec]:
    if not raw:
        return {"default": ProfileSpec(name="default", dirichlet=default_dirichlet, neumann=default_neumann)}
    out: dict[str, ProfileSpec] = {}
    for name, payload in raw.items():
        data = dict(payload or {})
        dirichlet = _normalize_dirichlet(data.get("dirichlet"), dim=dim) if "dirichlet" in data else default_dirichlet
        neumann = _normalize_neumann(data.get("neumann")) if "neumann" in data else default_neumann
        out[str(name)] = ProfileSpec(name=str(name), dirichlet=dirichlet, neumann=neumann)
    return out


def _normalize_hydraulic_state(raw: dict[str, Any] | None) -> HydraulicStateSpec | None:
    if raw is None:
        return None
    data = dict(raw)
    kind = str(data.pop("kind", "")).strip()
    if not kind:
        raise ValueError("hydraulic_state.kind is required.")
    return HydraulicStateSpec(kind=kind, value_model=data)


def _normalize_mechanics(
    *,
    materials: dict[str, MaterialModelSpec],
    region_assignment: dict[str, str],
    mechanics: dict[str, Any] | None,
    dim: int,
) -> MechanicsSpec | None:
    if mechanics is None:
        return None
    data = dict(mechanics)
    for region_name, material_name in region_assignment.items():
        if material_name not in materials:
            raise ValueError(f"Region {region_name!r} references unknown material model {material_name!r}.")
    default_dirichlet = _normalize_dirichlet(data.get("dirichlet"), dim=dim)
    default_neumann = _normalize_neumann(data.get("neumann"))
    profiles = _normalize_profile_map(
        data.get("profiles"),
        dim=dim,
        default_dirichlet=default_dirichlet,
        default_neumann=default_neumann,
    )
    default_profile = str(data.get("default_profile", "default"))
    if default_profile not in profiles:
        raise ValueError(f"Unknown mechanics default_profile {default_profile!r}.")
    return MechanicsSpec(
        materials=dict(materials),
        region_assignment={_logical_name(name, prefixes=("region",)): str(value) for name, value in region_assignment.items()},
        profiles=profiles,
        default_profile=default_profile,
        hydraulic_state=_normalize_hydraulic_state(data.get("hydraulic_state")),
    )


def _normalize_head_bcs(raw: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> tuple[HeadBCSpec, ...]:
    if raw is None:
        return ()
    out: list[HeadBCSpec] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"head_bcs[{idx}] must be a dictionary.")
        target = str(item.get("target") or item.get("boundary") or item.get("nodeset") or "")
        kind = str(item.get("kind") or "").strip()
        if not target or not kind:
            raise ValueError(f"head_bcs[{idx}] must define target and kind.")
        payload = {key: value for key, value in dict(item).items() if key not in {"target", "boundary", "nodeset", "kind"}}
        out.append(HeadBCSpec(target=_logical_name(target), kind=kind, value_model=payload))
    return tuple(out)


def _normalize_seepage(seepage: SeepageSpec | dict[str, Any] | None) -> SeepageSpec | None:
    if seepage is None:
        return None
    if isinstance(seepage, SeepageSpec):
        return seepage
    data = dict(seepage)
    mode = str(data.get("conductivity_mode", "by_material")).strip().lower()
    conductivity = data.get("conductivity")
    conductivity_tuple = None
    if conductivity is not None:
        if isinstance(conductivity, (int, float, np.integer, np.floating)):
            conductivity_tuple = (float(conductivity),)
        else:
            conductivity_tuple = tuple(float(value) for value in conductivity)
    region_cond = { _logical_name(name, prefixes=("region",)): float(value) for name, value in dict(data.get("region_conductivity", {})).items() }
    return SeepageSpec(
        water_unit_weight=float(data["water_unit_weight"]),
        conductivity_mode=mode,
        conductivity=conductivity_tuple,
        region_conductivity=region_cond,
        head_bcs=_normalize_head_bcs(data.get("head_bcs")),
        flux_bcs=_normalize_neumann(data.get("flux_bcs")),
    )


def build_seepage_spec(
    *,
    water_unit_weight: float,
    conductivity_mode: str,
    conductivity: list[float] | tuple[float, ...] | float | None = None,
    region_conductivity: dict[str, float] | None = None,
    head_bcs: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    flux_bcs: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> SeepageSpec:
    payload: dict[str, Any] = {
        "water_unit_weight": float(water_unit_weight),
        "conductivity_mode": str(conductivity_mode),
        "head_bcs": list(head_bcs or ()),
        "flux_bcs": list(flux_bcs or ()),
    }
    if conductivity is not None:
        payload["conductivity"] = conductivity
    if region_conductivity is not None:
        payload["region_conductivity"] = dict(region_conductivity)
    spec = _normalize_seepage(payload)
    assert spec is not None
    return spec


def _surface_edge_groups(elem_type: str) -> dict[tuple[int, int], tuple[int, ...]]:
    elem_key = str(elem_type).strip().upper()
    if elem_key == "P1":
        return {}
    if elem_key == "P2":
        return {(0, 1): (2,),}
    if elem_key == "P4":
        return {(0, 1): (2, 3, 4)}
    raise ValueError(f"Unsupported 2D boundary element type {elem_type!r}.")


def _surface_face_layout(elem_type: str) -> tuple[dict[tuple[int, int], tuple[int, ...]], tuple[int, ...]]:
    elem_key = str(elem_type).strip().upper()
    if elem_key == "P1":
        return {}, ()
    if elem_key == "P2":
        return {(0, 1): (3,), (1, 2): (4,), (0, 2): (5,)}, ()
    if elem_key == "P3":
        return {(0, 1): (3, 4), (1, 2): (5, 6), (0, 2): (7, 8)}, (9,)
    if elem_key == "P4":
        return {(0, 1): (3, 4, 5), (1, 2): (6, 7, 8), (0, 2): (9, 10, 11)}, (12, 13, 14)
    raise ValueError(f"Unsupported 3D boundary element type {elem_type!r}.")


def _expand_nodeset(mesh: SolverMesh, base_nodes: np.ndarray) -> np.ndarray:
    selected = np.zeros(mesh.coord.shape[1], dtype=bool)
    selected[np.asarray(base_nodes, dtype=np.int64)] = True
    surf = np.asarray(mesh.surf, dtype=np.int64)
    if surf.size == 0:
        return np.flatnonzero(selected).astype(np.int64)

    if mesh.coord.shape[0] == 2:
        edge_map = _surface_edge_groups(str(mesh.elem_type))
        for col in range(surf.shape[1]):
            corners = surf[:2, col]
            if np.all(selected[corners]):
                selected[surf[:, col]] = True
            else:
                for edge_nodes in edge_map.values():
                    if np.all(selected[corners]):
                        selected[surf[list(edge_nodes), col]] = True
        return np.flatnonzero(selected).astype(np.int64)

    edge_map, face_nodes = _surface_face_layout(str(mesh.elem_type))
    for col in range(surf.shape[1]):
        corners = surf[:3, col]
        if np.all(selected[corners]):
            selected[surf[:, col]] = True
            continue
        for edge, local_nodes in edge_map.items():
            if np.all(selected[corners[list(edge)]]):
                selected[surf[list(local_nodes), col]] = True
        if face_nodes and np.all(selected[corners]):
            selected[surf[list(face_nodes), col]] = True
    return np.flatnonzero(selected).astype(np.int64)


def _resolve_target_nodes(mesh: SolverMesh, target: str) -> np.ndarray:
    text = _logical_name(target)
    if text in mesh.nodesets:
        return _expand_nodeset(mesh, np.asarray(mesh.nodesets[text], dtype=np.int64))
    if text in mesh.boundary_groups:
        return np.unique(mesh.surf[:, np.asarray(mesh.boundary_groups[text], dtype=np.int64)].ravel()).astype(np.int64)
    raise KeyError(f"Unknown support target {target!r}.")


def _material_rows_from_models(materials: dict[str, MaterialModelSpec]) -> list[list[float]] | None:
    rows: list[list[float]] = []
    any_mechanical = False
    for model in materials.values():
        row = model.mechanical_row()
        if row is None:
            rows.append([])
            continue
        rows.append(row)
        any_mechanical = True
    if not any_mechanical:
        return None
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Material models mix mechanical and non-mechanical definitions.")
    return rows


@dataclass(frozen=True)
class CanonicalProblemAsset(ProblemAssetAPI):
    asset_id: str
    asset_dir: Path
    dimension: int
    default_variant: str
    default_profile: str
    _variants: dict[str, MeshVariant]
    _materials: dict[str, MaterialModelSpec]
    _region_assignment: dict[str, str]
    _mechanics: MechanicsSpec | None = None
    _seepage: SeepageSpec | None = None
    _boundary_geometry_specs: dict[str, tuple[str, int]] = field(default_factory=dict)
    source_kind: str = "gmsh_problem_asset"

    @property
    def capabilities(self) -> frozenset[str]:
        out: list[str] = []
        if self._mechanics is not None:
            out.append("mechanics")
        if self._seepage is not None:
            out.append("seepage")
        return frozenset(out)

    def list_variants(self) -> dict[str, MeshVariant]:
        return dict(self._variants)

    def resolve_variant(
        self,
        mesh_variant: str | None,
        *,
        profile: str | None = None,
    ) -> ResolvedVariant:
        name = str(mesh_variant or self.default_variant or next(iter(self._variants)))
        if name not in self._variants:
            raise KeyError(f"Unknown mesh variant {name!r} for asset {self.asset_id!r}.")
        variant = self._variants[name]
        resolved_profile = str(profile or self.default_profile)
        if self._mechanics is not None and resolved_profile not in self._mechanics.profiles:
            raise KeyError(f"Unknown mechanics profile {resolved_profile!r} for asset {self.asset_id!r}.")
        return ResolvedVariant(
            asset_id=self.asset_id,
            name=variant.name,
            source=dict(variant.source),
            mesh_path=variant.mesh_path,
            metadata=dict(variant.metadata),
            profile=resolved_profile,
        )

    def build_mesh(self, variant: ResolvedVariant, *, elem_type: str) -> SolverMesh:
        if variant.mesh_path is None:
            raise ValueError(f"Variant {variant.name!r} for asset {self.asset_id!r} does not define a mesh path.")
        canonical = load_canonical_gmsh_mesh(variant.mesh_path, dimension=int(self.dimension))
        region_to_material = self._region_material_index()
        boundary_id_by_name = {name: idx for idx, name in enumerate(sorted(canonical.boundary_groups))}
        mesh = build_solver_mesh(
            canonical,
            elem_type=str(elem_type),
            boundary_geometry_specs=dict(self._boundary_geometry_specs),
            region_id_by_name=region_to_material,
            boundary_id_by_name=boundary_id_by_name,
        )
        mechanics = self.build_mechanics(mesh, variant)
        if mechanics is None:
            return mesh
        return replace(mesh, q_mask=np.asarray(mechanics.q_mask, dtype=bool))

    def build_mechanics(self, mesh: SolverMesh, variant: ResolvedVariant) -> MechanicalProblemSpec | None:
        if self._mechanics is None:
            return None
        profile = self._mechanics.profiles[str(variant.profile)]
        q_mask = np.ones((int(self.dimension), mesh.coord.shape[1]), dtype=bool)
        component_to_idx = {name: idx for idx, name in enumerate(AXES_BY_DIM[int(self.dimension)])}
        for bc in profile.dirichlet:
            if bc.values is not None and any(abs(float(value)) > 0.0 for value in bc.values):
                raise NotImplementedError("Non-zero prescribed displacements are not wired into the solver stack yet.")
            nodes = _resolve_target_nodes(mesh, bc.target)
            for component in bc.components:
                q_mask[component_to_idx[component], nodes] = False
        rows = self.material_rows()
        return MechanicalProblemSpec(materials=[] if rows is None else rows, q_mask=q_mask, profile=str(variant.profile))

    def build_seepage(self, mesh: SolverMesh, variant: ResolvedVariant) -> SeepageProblemSpec | None:
        if self._seepage is None:
            return None
        q_w = np.ones(mesh.coord.shape[1], dtype=bool)
        pw_d = np.zeros(mesh.coord.shape[1], dtype=np.float64)
        y = np.asarray(mesh.coord[1, :], dtype=np.float64)
        rho_g = float(self._seepage.water_unit_weight)
        for bc in self._seepage.head_bcs:
            nodes = _resolve_target_nodes(mesh, bc.target)
            q_w[nodes] = False
            kind = str(bc.kind).strip().lower()
            scope = str(bc.value_model.get("scope", "support_only")).strip().lower()
            if kind == "dry":
                continue
            if kind in {"constant_level", "piecewise_linear_level"}:
                head = head_values(mesh.coord, bc.value_model, kind=kind)
                values = rho_g * np.maximum(head - y, 0.0)
                if scope == "domain_below_head":
                    pw_d = np.maximum(pw_d, values)
                else:
                    pw_d[nodes] = np.maximum(pw_d[nodes], values[nodes])
                continue
            raise ValueError(f"Unsupported seepage head BC kind {bc.kind!r}.")
        conductivity = self.hydraulic_conductivity()
        if conductivity is None:
            conductivity = np.empty(0, dtype=np.float64)
        return SeepageProblemSpec(
            water_unit_weight=rho_g,
            conductivity=np.asarray(conductivity, dtype=np.float64),
            q_w=q_w,
            pw_d=pw_d,
        )

    def material_rows(self) -> list[list[float]] | None:
        return _material_rows_from_models(self._materials)

    def mechanics_spec(self) -> MechanicsSpec | None:
        return self._mechanics

    def seepage_spec(self) -> SeepageSpec | None:
        return self._seepage

    def hydraulic_conductivity(self) -> np.ndarray | None:
        if self._seepage is None:
            return None
        mode = str(self._seepage.conductivity_mode).strip().lower()
        if mode == "uniform":
            if self._seepage.conductivity is None:
                raise ValueError(f"Asset {self.asset_id!r} requires seepage.conductivity for conductivity_mode='uniform'.")
            return np.asarray(self._seepage.conductivity, dtype=np.float64)
        if self._mechanics is None:
            materials = self._materials
        else:
            materials = self._mechanics.materials
        if mode == "by_material":
            missing = [name for name, model in materials.items() if model.hydraulic_conductivity is None]
            if missing:
                raise ValueError(f"Materials {missing} in asset {self.asset_id!r} are missing hydraulic_conductivity.")
            return np.asarray(
                [float(model.hydraulic_conductivity) for model in materials.values()],
                dtype=np.float64,
            )
        if mode == "by_region":
            by_material: dict[str, float] = {}
            for region_name, material_name in self._region_assignment.items():
                if region_name not in self._seepage.region_conductivity:
                    raise ValueError(f"Region {region_name!r} is missing a seepage conductivity override.")
                value = float(self._seepage.region_conductivity[region_name])
                previous = by_material.get(material_name)
                if previous is not None and not np.isclose(previous, value):
                    raise ValueError(
                        f"Material {material_name!r} in asset {self.asset_id!r} maps to multiple region conductivities; "
                        "use distinct material models instead."
                    )
                by_material[material_name] = value
            return np.asarray([by_material[name] for name in materials], dtype=np.float64)
        raise ValueError(f"Unsupported seepage conductivity_mode {self._seepage.conductivity_mode!r}.")

    def hydraulic_state(self) -> HydraulicStateSpec | None:
        return None if self._mechanics is None else self._mechanics.hydraulic_state

    def _region_material_index(self) -> dict[str, int]:
        material_order = {name: idx for idx, name in enumerate(self._materials)}
        return {region_name: material_order[material_name] for region_name, material_name in self._region_assignment.items()}


def _variant_specs(asset_dir: Path, mesh_variants: dict[str, dict[str, Any]]) -> dict[str, MeshVariant]:
    variants: dict[str, MeshVariant] = {}
    for name, payload in mesh_variants.items():
        item = dict(payload)
        source = dict(item.get("source", {}))
        mesh_path = None
        if source.get("path") is not None:
            mesh_path = (asset_dir / str(source["path"])).resolve()
        metadata = {key: value for key, value in item.items() if key != "source"}
        variants[str(name)] = MeshVariant(name=str(name), source=source, mesh_path=mesh_path, metadata=metadata)
    return variants


def _normalize_boundary_geometry(raw: dict[str, dict[str, Any]] | None) -> dict[str, tuple[str, int]]:
    if not raw:
        return {}
    out: dict[str, tuple[str, int]] = {}
    for name, payload in raw.items():
        data = dict(payload or {})
        support = _logical_name(str(data["support_boundary"]), prefixes=("boundary",))
        order = int(data.get("geometry_order", 2))
        out[_logical_name(str(name), prefixes=("boundary_geom",))] = (support, order)
    return out


def _build_problem_asset(
    *,
    asset_id: str,
    asset_dir: Path,
    dimension: int,
    default_variant: str,
    mesh_variants: dict[str, dict[str, Any]],
    materials: dict[str, Any] | list[dict[str, Any]],
    region_assignment: dict[str, str],
    mechanics: dict[str, Any] | None = None,
    seepage: SeepageSpec | dict[str, Any] | None = None,
    boundary_geometry: dict[str, dict[str, Any]] | None = None,
) -> CanonicalProblemAsset:
    material_models = _normalize_material_models(materials)
    mechanics_spec = _normalize_mechanics(
        materials=material_models,
        region_assignment=region_assignment,
        mechanics=mechanics,
        dim=int(dimension),
    ) if mechanics is not None else None
    seepage_spec = _normalize_seepage(seepage)
    default_profile = "default" if mechanics_spec is None else mechanics_spec.default_profile
    return CanonicalProblemAsset(
        asset_id=str(asset_id),
        asset_dir=asset_dir.resolve(),
        dimension=int(dimension),
        default_variant=str(default_variant),
        default_profile=str(default_profile),
        _variants=_variant_specs(asset_dir.resolve(), mesh_variants),
        _materials=dict(material_models),
        _region_assignment={_logical_name(name, prefixes=("region",)): str(value) for name, value in region_assignment.items()},
        _mechanics=mechanics_spec,
        _seepage=seepage_spec,
        _boundary_geometry_specs=_normalize_boundary_geometry(boundary_geometry),
    )


def build_problem_asset_2d(**kwargs) -> CanonicalProblemAsset:
    return _build_problem_asset(dimension=2, **kwargs)


def build_problem_asset_3d(**kwargs) -> CanonicalProblemAsset:
    return _build_problem_asset(dimension=3, **kwargs)


def build_asset(**kwargs) -> CanonicalProblemAsset:
    """Compatibility alias while the repo migrates to canonical helpers only."""

    return _build_problem_asset(**kwargs)


__all__ = [
    "CanonicalProblemAsset",
    "build_asset",
    "build_problem_asset_2d",
    "build_problem_asset_3d",
    "build_seepage_spec",
    "gmsh_variants_from_dir",
]
