"""Runtime asset resolution and mesh construction helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from .assets import MeshBuildResult, ProblemAssetAPI, ResolvedVariant, load_problem_asset
from .assets.support.physical_names import parse_gmsh_physical_names
from .problem_assets import (
    load_hydraulic_conductivity_for_asset_definition,
    load_material_rows_for_asset,
    load_mechanical_dirichlet_rules_for_asset_definition,
    load_seepage_spec_for_asset_definition,
)


MECHANICS_LABEL_CONSTRAINT_COLUMNS: tuple[str, ...] = (
    "support_kind",
    "support_name",
    "dm_label",
    "tag",
    "components",
    "values",
    "native_status",
)

MECHANICS_NEUMANN_LABEL_COLUMNS: tuple[str, ...] = (
    "support_kind",
    "support_name",
    "dm_label",
    "tag",
    "kind",
    "geometry",
    "geometry_order",
    "value_model",
    "native_status",
)

SEEPAGE_LABEL_BC_COLUMNS: tuple[str, ...] = (
    "field",
    "support_kind",
    "support_name",
    "dm_label",
    "tag",
    "kind",
    "geometry",
    "geometry_order",
    "value_model",
    "native_status",
)


@dataclass(frozen=True)
class ResolvedAsset:
    definition: ProblemAssetAPI
    variant_name: str
    variant: dict[str, Any]
    resolved_variant: ResolvedVariant
    mesh_path: Path | None

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
    neumann_rules: tuple[Any, ...] = ()


@dataclass(frozen=True)
class SeepageProblemSpec:
    seepage: Any
    conductivity: np.ndarray


def resolve_problem_asset(
    *,
    asset_name: str,
    mesh_variant: str | None = None,
    profile: str | None = None,
) -> ResolvedAsset:
    asset = load_problem_asset(asset_name)
    resolved_variant = asset.resolve_variant(
        mesh_variant=None if mesh_variant is None else str(mesh_variant),
        profile=profile,
    )
    return ResolvedAsset(
        definition=asset,
        variant_name=str(resolved_variant.name),
        variant=resolved_variant.as_dict(),
        resolved_variant=resolved_variant,
        mesh_path=resolved_variant.mesh_path,
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


def load_mechanical_problem_spec(resolved: ResolvedAsset) -> MechanicalProblemSpec:
    rows = load_material_rows_for_asset(resolved.asset_name)
    if rows is None:
        raise ValueError(f"No mechanical materials registered for asset {resolved.asset_name!r}.")
    mechanics = resolved.definition.mechanics_spec()
    if mechanics is None:
        raise ValueError(f"No mechanics definition registered for asset {resolved.asset_name!r}.")
    profile = mechanics.profiles[str(resolved.resolved_variant.profile)]
    return MechanicalProblemSpec(
        material_rows=rows,
        dirichlet_rules=tuple(profile.dirichlet),
        neumann_rules=tuple(profile.neumann),
    )


def build_mechanics_label_constraint_rows(resolved: ResolvedAsset) -> list[dict[str, Any]]:
    """Return coordinate-free Dirichlet constraints keyed by asset supports."""

    supports = parse_gmsh_physical_names(resolved.mesh_path)
    mechanics = resolved.definition.mechanics_spec()
    if mechanics is None:
        raise ValueError(f"No mechanics definition registered for asset {resolved.asset_name!r}.")
    profile_name = str(resolved.resolved_variant.profile)
    profile = mechanics.profiles[profile_name]
    rows: list[dict[str, Any]] = []
    for rule in profile.dirichlet:
        support = _required_support_ref(
            f"Dirichlet target {rule.target!r} in asset {resolved.asset_name!r}",
            rule.target,
            supports,
            allowed=("boundary", "nodeset"),
        )
        rows.append(
            {
                "support_kind": support["kind"],
                "support_name": support["name"],
                "dm_label": support["dm_label"],
                "tag": int(support["tag"]),
                "components": " ".join(str(component) for component in rule.components),
                "values": "" if rule.values is None else " ".join(f"{float(value):.17g}" for value in rule.values),
                "native_status": "label_table_native_preferred",
            }
        )
    return rows


def build_mechanics_neumann_label_rows(resolved: ResolvedAsset) -> list[dict[str, Any]]:
    """Return coordinate-free Neumann rules keyed by asset boundary supports."""

    supports = parse_gmsh_physical_names(resolved.mesh_path)
    mechanics = resolved.definition.mechanics_spec()
    if mechanics is None:
        raise ValueError(f"No mechanics definition registered for asset {resolved.asset_name!r}.")
    profile_name = str(resolved.resolved_variant.profile)
    profile = mechanics.profiles[profile_name]
    boundary_geometry = _boundary_geometry_manifest(resolved.definition, supports)
    rows: list[dict[str, Any]] = []
    for rule in profile.neumann:
        support = _required_support_ref(
            f"Neumann target {rule.target!r} in asset {resolved.asset_name!r}",
            rule.target,
            supports,
            allowed=("boundary",),
        )
        geometry = "" if rule.geometry is None else str(rule.geometry)
        geometry_spec = _geometry_support_for_rule(
            "Neumann",
            target=str(rule.target),
            geometry=rule.geometry,
            boundary_geometry=boundary_geometry,
        )
        rows.append(
            {
                "support_kind": support["kind"],
                "support_name": support["name"],
                "dm_label": support["dm_label"],
                "tag": int(support["tag"]),
                "kind": str(rule.kind),
                "geometry": geometry,
                "geometry_order": "" if geometry_spec is None else int(geometry_spec["geometry_order"]),
                "value_model": _json_dumps_compact(_jsonable(rule.value_model)),
                "native_status": _mechanics_neumann_native_status(geometry_spec),
            }
        )
    return rows


def build_seepage_label_bc_rows(resolved: ResolvedAsset) -> list[dict[str, Any]]:
    """Return coordinate-free seepage head/flux rules keyed by asset supports."""

    supports = parse_gmsh_physical_names(resolved.mesh_path)
    seepage = resolved.definition.seepage_spec()
    if seepage is None:
        return []
    boundary_geometry = _boundary_geometry_manifest(resolved.definition, supports)
    rows: list[dict[str, Any]] = []
    for rule in seepage.head_bcs:
        support = _required_support_ref(
            f"Seepage head target {rule.target!r} in asset {resolved.asset_name!r}",
            rule.target,
            supports,
            allowed=("boundary", "nodeset"),
        )
        rows.append(
            {
                "field": "head",
                "support_kind": support["kind"],
                "support_name": support["name"],
                "dm_label": support["dm_label"],
                "tag": int(support["tag"]),
                "kind": str(rule.kind),
                "geometry": "",
                "geometry_order": "",
                "value_model": _json_dumps_compact(_jsonable(rule.value_model)),
                "native_status": "label_ready_coordinate_pressure_bridge_active",
            }
        )
    for rule in seepage.flux_bcs:
        support = _required_support_ref(
            f"Seepage flux target {rule.target!r} in asset {resolved.asset_name!r}",
            rule.target,
            supports,
            allowed=("boundary",),
        )
        geometry = "" if rule.geometry is None else str(rule.geometry)
        geometry_spec = _geometry_support_for_rule(
            "Seepage flux",
            target=str(rule.target),
            geometry=rule.geometry,
            boundary_geometry=boundary_geometry,
        )
        rows.append(
            {
                "field": "flux",
                "support_kind": support["kind"],
                "support_name": support["name"],
                "dm_label": support["dm_label"],
                "tag": int(support["tag"]),
                "kind": str(rule.kind),
                "geometry": geometry,
                "geometry_order": "" if geometry_spec is None else int(geometry_spec["geometry_order"]),
                "value_model": _json_dumps_compact(_jsonable(rule.value_model)),
                "native_status": "pending_native_face_quadrature",
            }
        )
    return rows


def validate_native_problem_artifact_contract(manifest_path: str | Path) -> dict[str, Any]:
    """Validate that native manifest rule declarations match emitted label tables."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    native_inputs = manifest.get("native_inputs") or {}
    rule_counts = manifest.get("rule_counts") or {}
    checks = {
        "mechanics_dirichlet": _validate_manifest_rows(
            path,
            native_inputs.get("mechanics_label_constraints_csv"),
            native_inputs.get("mechanics_label_constraints_row_fingerprint"),
            MECHANICS_LABEL_CONSTRAINT_COLUMNS,
            _manifest_dirichlet_label_rows(manifest),
            required_count=int(rule_counts.get("mechanics_dirichlet", 0) or 0),
        ),
        "mechanics_neumann": _validate_manifest_rows(
            path,
            native_inputs.get("mechanics_neumann_labels_csv"),
            native_inputs.get("mechanics_neumann_labels_row_fingerprint"),
            MECHANICS_NEUMANN_LABEL_COLUMNS,
            _manifest_neumann_label_rows(manifest),
            required_count=int(rule_counts.get("mechanics_neumann", 0) or 0),
        ),
        "seepage_boundary": _validate_manifest_rows(
            path,
            native_inputs.get("seepage_boundary_labels_csv"),
            native_inputs.get("seepage_boundary_labels_row_fingerprint"),
            SEEPAGE_LABEL_BC_COLUMNS,
            _manifest_seepage_label_rows(manifest),
            required_count=int(rule_counts.get("seepage_head", 0) or 0) + int(rule_counts.get("seepage_flux", 0) or 0),
        ),
    }
    return {"ok": True, "manifest": str(path), "checks": checks}


def build_native_label_table_contracts(resolved: ResolvedAsset) -> dict[str, dict[str, Any]]:
    """Return coordinate-free label-table contracts without writing run artifacts."""

    mechanics = resolved.definition.mechanics_spec()
    seepage = resolved.definition.seepage_spec()
    mechanics_dirichlet_rows = build_mechanics_label_constraint_rows(resolved) if mechanics is not None else []
    mechanics_neumann_rows = build_mechanics_neumann_label_rows(resolved) if mechanics is not None else []
    seepage_rows = build_seepage_label_bc_rows(resolved) if seepage is not None else []
    specs = {
        "mechanics_dirichlet": (
            MECHANICS_LABEL_CONSTRAINT_COLUMNS,
            mechanics_dirichlet_rows,
            "mechanics_label_constraints_csv",
            "mechanics_label_constraints_row_fingerprint",
        ),
        "mechanics_neumann": (
            MECHANICS_NEUMANN_LABEL_COLUMNS,
            mechanics_neumann_rows,
            "mechanics_neumann_labels_csv",
            "mechanics_neumann_labels_row_fingerprint",
        ),
        "seepage_boundary": (
            SEEPAGE_LABEL_BC_COLUMNS,
            seepage_rows,
            "seepage_boundary_labels_csv",
            "seepage_boundary_labels_row_fingerprint",
        ),
    }
    contracts: dict[str, dict[str, Any]] = {}
    for name, (columns, rows, path_key, fingerprint_key) in specs.items():
        contracts[name] = {
            "native_input_key": path_key,
            "fingerprint_key": fingerprint_key,
            "columns": list(columns),
            "rows": len(rows),
            "row_fingerprint": _fingerprint_label_rows(columns, rows) if rows else None,
            "native_statuses": sorted({str(row.get("native_status", "")) for row in rows if row.get("native_status")}),
        }
    return contracts


def build_native_problem_manifest(
    resolved: ResolvedAsset,
    *,
    case_id: str | None = None,
    case_path: str | Path | None = None,
    analysis: str | None = None,
    elem_type: str | None = None,
    solver_profile: str | None = None,
    world_size: int | None = None,
    compatibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a coordinate-free asset manifest for the native PETSc boundary path."""

    supports = parse_gmsh_physical_names(resolved.mesh_path)
    mechanics = resolved.definition.mechanics_spec()
    seepage = resolved.definition.seepage_spec()
    profile_name = str(resolved.resolved_variant.profile)
    boundary_geometry = _boundary_geometry_manifest(resolved.definition, supports)
    compatibility_data = dict(compatibility or {})
    native_inputs = _native_inputs_manifest(compatibility_data)
    mechanics_profile = mechanics.profiles[profile_name] if mechanics is not None else None
    support_counts = {
        "regions": len(supports.get("regions", {})),
        "boundaries": len(supports.get("boundaries", {})),
        "nodesets": len(supports.get("nodesets", {})),
        "boundary_geometry": len(boundary_geometry),
    }
    rule_counts = {
        "mechanics_dirichlet": 0 if mechanics_profile is None else len(mechanics_profile.dirichlet),
        "mechanics_neumann": 0 if mechanics_profile is None else len(mechanics_profile.neumann),
        "seepage_head": 0 if seepage is None else len(seepage.head_bcs),
        "seepage_flux": 0 if seepage is None else len(seepage.flux_bcs),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "petsc_ssr_native_problem_manifest",
        "case": {
            "id": case_id,
            "path": None if case_path is None else str(Path(case_path).resolve()),
            "analysis": analysis,
            "element": elem_type,
            "solver_profile": solver_profile,
            "resolved_world_size": world_size,
        },
        "asset": {
            "id": resolved.asset_name,
            "dimension": resolved.dimension,
            "source_kind": resolved.source_kind,
            "capabilities": sorted(resolved.definition.capabilities),
            "variant": resolved.variant_name,
            "profile": profile_name,
            "mesh_path": None if resolved.mesh_path is None else str(resolved.mesh_path),
        },
        "dmplex": {
            "region_label": "Cell Sets",
            "boundary_label": "Face Sets",
            "native_boundary_marker_label": "boundary_marker",
            "support_counts": support_counts,
            "supports": _support_payload(supports),
        },
        "rule_counts": rule_counts,
        "materials": _materials_manifest(mechanics),
        "boundary_geometry": boundary_geometry,
        "native_inputs": native_inputs,
        "mechanics": None,
        "seepage": None,
        "compatibility": {
            "label_constraint_table_available": bool(mechanics and mechanics.profiles[profile_name].dirichlet),
            "label_constraint_table_required": bool(mechanics and mechanics.profiles[profile_name].dirichlet),
            "coordinate_constraint_table_required": False,
            "coordinate_constraint_table_fallback_available": bool(compatibility_data.get("mechanics_coordinate_constraint_table")),
            "coordinate_seepage_pressure_table_required": bool(compatibility_data.get("seepage_coupled", False)),
            "note": (
                "Native manifest-driven mechanics constraints require the DMPlex label table; coordinate "
                "CSVs are debug compatibility artifacts only. Coupled seepage pressure still uses a "
                "coordinate CSV bridge until native field ingestion lands."
            ),
        },
    }
    if compatibility_data:
        payload["compatibility"].update(_jsonable(compatibility_data))
    if mechanics is not None:
        profile = mechanics_profile
        hydraulic_state = None
        if mechanics.hydraulic_state is not None:
            hydraulic_state = _jsonable({"kind": mechanics.hydraulic_state.kind, **mechanics.hydraulic_state.value_model})
        payload["mechanics"] = {
            "profile": profile_name,
            "hydraulic_state": hydraulic_state,
            "dirichlet": [
                {
                    "target": rule.target,
                    "components": list(rule.components),
                    "values": None if rule.values is None else [float(value) for value in rule.values],
                    "support": _required_support_ref(
                        f"Dirichlet target {rule.target!r} in asset {resolved.asset_name!r}",
                        rule.target,
                        supports,
                        allowed=("boundary", "nodeset"),
                    ),
                    "native_status": "label_table_native_preferred",
                }
                for rule in profile.dirichlet
            ],
            "neumann": [
                _neumann_rule_manifest(
                    rule,
                    supports=supports,
                    boundary_geometry=boundary_geometry,
                    owner="Neumann",
                )
                for rule in profile.neumann
            ],
        }
    if seepage is not None:
        payload["seepage"] = {
            "water_unit_weight": float(seepage.water_unit_weight),
            "conductivity_mode": seepage.conductivity_mode,
            "conductivity": None if seepage.conductivity is None else [float(value) for value in seepage.conductivity],
            "region_conductivity": {name: float(value) for name, value in seepage.region_conductivity.items()},
            "head_bcs": [
                {
                    "target": rule.target,
                    "kind": rule.kind,
                    "value_model": _jsonable(rule.value_model),
                    "support": _required_support_ref(
                        f"Seepage head target {rule.target!r} in asset {resolved.asset_name!r}",
                        rule.target,
                        supports,
                        allowed=("boundary", "nodeset"),
                    ),
                    "native_status": "label_ready_coordinate_pressure_bridge_active",
                }
                for rule in seepage.head_bcs
            ],
            "flux_bcs": [
                _neumann_rule_manifest(
                    rule,
                    supports=supports,
                    boundary_geometry=boundary_geometry,
                    owner="Seepage flux",
                )
                for rule in seepage.flux_bcs
            ],
        }
    _attach_native_input_row_fingerprints(payload)
    return payload


def _validate_manifest_rows(
    manifest_path: Path,
    csv_path_value: Any,
    fingerprint_value: Any,
    columns: tuple[str, ...],
    expected_rows: list[dict[str, Any]],
    *,
    required_count: int,
) -> dict[str, Any]:
    if len(expected_rows) != required_count:
        raise ValueError(
            f"Native problem manifest {manifest_path} declares {required_count} rows for {columns[0]} table, "
            f"but its rule arrays contain {len(expected_rows)} rows."
        )
    if not csv_path_value:
        if required_count:
            raise ValueError(
                f"Native problem manifest {manifest_path} declares {required_count} rows for {columns[0]} table, "
                "but no label-table path is available."
            )
        return {"path": None, "rows": 0}

    csv_path = Path(str(csv_path_value))
    if not csv_path.is_absolute():
        csv_path = manifest_path.parent / csv_path
    if not csv_path.exists():
        if required_count:
            raise ValueError(f"Native problem manifest {manifest_path} references missing label table {csv_path}.")
        return {"path": str(csv_path), "rows": 0}

    actual_rows = _read_label_table_rows(csv_path, columns)
    expected = _normalised_row_set(expected_rows, columns)
    actual = _normalised_row_set(actual_rows, columns)
    if actual != expected:
        raise ValueError(
            f"Native problem manifest {manifest_path} does not match label table {csv_path}: "
            f"expected {len(expected_rows)} rows, found {len(actual_rows)} rows."
        )
    expected_fingerprint = _fingerprint_label_rows(columns, expected_rows)
    actual_fingerprint = _fingerprint_label_rows(columns, actual_rows)
    if fingerprint_value and str(fingerprint_value) != expected_fingerprint:
        raise ValueError(
            f"Native problem manifest {manifest_path} declares label-table fingerprint {fingerprint_value}, "
            f"but expected manifest rows fingerprint to be {expected_fingerprint}."
        )
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            f"Native problem manifest {manifest_path} label table {csv_path} has fingerprint "
            f"{actual_fingerprint}, expected {expected_fingerprint}."
        )
    return {"path": str(csv_path), "rows": len(actual_rows)}


def _read_label_table_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"Label table {path} has columns {reader.fieldnames}; expected {list(columns)}.")
        return [dict(row) for row in reader]


def _normalised_row_set(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> list[tuple[str, ...]]:
    return sorted(tuple("" if row.get(column) is None else str(row.get(column)) for column in columns) for row in rows)


def _attach_native_input_row_fingerprints(manifest: dict[str, Any]) -> None:
    native_inputs = manifest.setdefault("native_inputs", {})
    specs = (
        (
            "mechanics_label_constraints_csv",
            "mechanics_label_constraints_row_fingerprint",
            MECHANICS_LABEL_CONSTRAINT_COLUMNS,
            _manifest_dirichlet_label_rows(manifest),
        ),
        (
            "mechanics_neumann_labels_csv",
            "mechanics_neumann_labels_row_fingerprint",
            MECHANICS_NEUMANN_LABEL_COLUMNS,
            _manifest_neumann_label_rows(manifest),
        ),
        (
            "seepage_boundary_labels_csv",
            "seepage_boundary_labels_row_fingerprint",
            SEEPAGE_LABEL_BC_COLUMNS,
            _manifest_seepage_label_rows(manifest),
        ),
    )
    for path_key, fingerprint_key, columns, rows in specs:
        if native_inputs.get(path_key) and rows:
            native_inputs[fingerprint_key] = _fingerprint_label_rows(columns, rows)


def _fingerprint_label_rows(columns: tuple[str, ...], rows: list[dict[str, Any]]) -> str:
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {
            column: "" if row.get(column) is None else str(row.get(column))
            for column in columns
        }
        for row in rows
    )
    value = 0xCBF29CE484222325
    for byte in handle.getvalue().encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{value:016x}"


def _manifest_dirichlet_label_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in ((manifest.get("mechanics") or {}).get("dirichlet") or []):
        support = rule["support"]
        values = rule.get("values")
        rows.append(
            {
                "support_kind": support["kind"],
                "support_name": support["name"],
                "dm_label": support["dm_label"],
                "tag": int(support["tag"]),
                "components": " ".join(str(component) for component in rule.get("components") or []),
                "values": "" if values is None else " ".join(f"{float(value):.17g}" for value in values),
                "native_status": rule.get("native_status", ""),
            }
        )
    return rows


def _manifest_neumann_label_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in ((manifest.get("mechanics") or {}).get("neumann") or []):
        rows.append(_manifest_boundary_load_row(rule, field=None))
    return rows


def _manifest_seepage_label_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    seepage = manifest.get("seepage") or {}
    rows: list[dict[str, Any]] = []
    for rule in seepage.get("head_bcs") or []:
        rows.append(_manifest_boundary_load_row(rule, field="head"))
    for rule in seepage.get("flux_bcs") or []:
        rows.append(_manifest_boundary_load_row(rule, field="flux"))
    return rows


def _manifest_boundary_load_row(rule: dict[str, Any], *, field: str | None) -> dict[str, Any]:
    support = rule["support"]
    geometry_support = rule.get("geometry_support")
    row = {
        "support_kind": support["kind"],
        "support_name": support["name"],
        "dm_label": support["dm_label"],
        "tag": int(support["tag"]),
        "kind": str(rule.get("kind", "")),
        "geometry": "" if rule.get("geometry") is None else str(rule.get("geometry")),
        "geometry_order": "" if geometry_support is None else int(geometry_support["geometry_order"]),
        "value_model": _json_dumps_compact(_jsonable(rule.get("value_model") or {})),
        "native_status": rule.get("native_status", ""),
    }
    if field is not None:
        return {"field": field, **row}
    return row


def _support_payload(supports: dict[str, dict[str, int]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        kind: {
            name: {
                "tag": int(tag),
                "dm_label": _label_for_support_kind(kind),
                "support_kind": _singular_support_kind(kind),
            }
            for name, tag in sorted(values.items(), key=lambda item: item[0])
        }
        for kind, values in supports.items()
    }


def _native_inputs_manifest(compatibility_data: dict[str, Any]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    label_table = compatibility_data.get("mechanics_label_constraint_table")
    coordinate_table = compatibility_data.get("mechanics_coordinate_constraint_table")
    neumann_table = compatibility_data.get("mechanics_neumann_label_table")
    seepage_boundary_table = compatibility_data.get("seepage_boundary_label_table")
    seepage_pressure = compatibility_data.get("seepage_pressure_table")
    seepage_pressure_source = str(compatibility_data.get("seepage_pressure_source", "") or "").strip()
    if label_table:
        inputs["mechanics_label_constraints_csv"] = str(label_table)
    if coordinate_table:
        inputs["debug_coordinate_bc_table"] = bool(compatibility_data.get("debug_coordinate_bc_table", True))
        inputs["mechanics_coordinate_constraints_csv"] = str(coordinate_table)
    if neumann_table:
        inputs["mechanics_neumann_labels_csv"] = str(neumann_table)
    if seepage_boundary_table:
        inputs["seepage_boundary_labels_csv"] = str(seepage_boundary_table)
    if seepage_pressure:
        if seepage_pressure_source != "hydro_prepass_coordinate_bridge":
            raise ValueError(
                "seepage_pressure_table is a coordinate bridge and requires "
                "seepage_pressure_source='hydro_prepass_coordinate_bridge'"
            )
        inputs["seepage_pressure_source"] = seepage_pressure_source
        inputs["seepage_pressure_csv"] = str(seepage_pressure)
    return inputs


def _support_ref(target: str, supports: dict[str, dict[str, int]]) -> dict[str, Any]:
    for kind in ("boundaries", "nodesets", "regions"):
        tag = supports.get(kind, {}).get(str(target))
        if tag is not None:
            return {
                "name": str(target),
                "kind": _singular_support_kind(kind),
                "tag": int(tag),
                "dm_label": _label_for_support_kind(kind),
            }
    return {"name": str(target), "kind": "unresolved", "tag": None, "dm_label": None}


def _required_support_ref(
    owner: str,
    target: str,
    supports: dict[str, dict[str, int]],
    *,
    allowed: tuple[str, ...],
) -> dict[str, Any]:
    support = _support_ref(target, supports)
    if support["tag"] is None or support["dm_label"] is None:
        raise ValueError(f"{owner} does not resolve to a Gmsh physical support.")
    if support["kind"] not in allowed:
        allowed_text = "/".join(allowed)
        raise ValueError(f"{owner} must resolve to a {allowed_text} support, got {support['kind']!r}.")
    return support


def _label_for_support_kind(kind: str) -> str:
    if kind == "regions":
        return "Cell Sets"
    if kind in {"boundaries", "nodesets"}:
        return "Face Sets" if kind == "boundaries" else "Vertex Sets"
    return str(kind)


def _singular_support_kind(kind: str) -> str:
    return {
        "regions": "region",
        "boundaries": "boundary",
        "nodesets": "nodeset",
    }.get(kind, kind)


def _materials_manifest(mechanics: Any | None) -> dict[str, Any] | None:
    if mechanics is None:
        return None
    return {
        "models": {
            name: {
                "parameters": {key: float(value) for key, value in model.parameters.items()},
                "hydraulic_conductivity": (
                    None if model.hydraulic_conductivity is None else float(model.hydraulic_conductivity)
                ),
            }
            for name, model in mechanics.materials.items()
        },
        "region_assignment": dict(mechanics.region_assignment),
    }


def _neumann_rule_manifest(
    rule: Any,
    *,
    supports: dict[str, dict[str, int]],
    boundary_geometry: dict[str, dict[str, Any]],
    owner: str,
) -> dict[str, Any]:
    support = _required_support_ref(
        f"{owner} target {rule.target!r}",
        rule.target,
        supports,
        allowed=("boundary",),
    )
    geometry_support = _geometry_support_for_rule(
        owner,
        target=str(rule.target),
        geometry=rule.geometry,
        boundary_geometry=boundary_geometry,
    )
    return {
        "target": rule.target,
        "kind": rule.kind,
        "geometry": rule.geometry,
        "value_model": _jsonable(rule.value_model),
        "support": support,
        "geometry_support": geometry_support,
        "native_status": _face_rule_native_status(owner, geometry_support),
    }


def _mechanics_neumann_native_status(geometry_support: dict[str, Any] | None) -> str:
    if geometry_support is None:
        return "native_face_quadrature_affine"
    return "pending_native_curved_face_quadrature"


def _face_rule_native_status(owner: str, geometry_support: dict[str, Any] | None) -> str:
    if owner == "Neumann":
        return _mechanics_neumann_native_status(geometry_support)
    return "pending_native_face_quadrature"


def _boundary_geometry_manifest(
    definition: ProblemAssetAPI,
    supports: dict[str, dict[str, int]] | None = None,
) -> dict[str, dict[str, Any]]:
    getter = getattr(definition, "boundary_geometry_specs", None)
    specs = getter() if callable(getter) else {}
    out: dict[str, dict[str, Any]] = {}
    for name, (support, order) in specs.items():
        support_ref = (
            {"name": support, "kind": "boundary", "dm_label": "Face Sets", "tag": None}
            if supports is None
            else _support_ref(support, supports)
        )
        if support_ref["kind"] != "boundary" or support_ref["tag"] is None:
            raise ValueError(
                f"Boundary geometry {name!r} references support_boundary {support!r}, "
                "but that support is not a declared boundary physical name."
            )
        out[name] = {
            "support_boundary": support,
            "support": support_ref,
            "geometry_order": int(order),
        }
    return out


def _geometry_support_for_rule(
    owner: str,
    *,
    target: str,
    geometry: str | None,
    boundary_geometry: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if geometry is None:
        return None
    geometry_name = str(geometry)
    geometry_spec = boundary_geometry.get(geometry_name)
    if geometry_spec is None:
        raise ValueError(f"{owner} target {target!r} references unknown boundary geometry {geometry_name!r}.")
    geometry_support = dict(geometry_spec.get("support") or {})
    if geometry_support.get("kind") != "boundary" or geometry_support.get("tag") is None:
        raise ValueError(f"Boundary geometry {geometry_name!r} does not resolve to a boundary physical support.")
    if str(geometry_support.get("name")) != str(target):
        raise ValueError(
            f"{owner} target {target!r} references boundary geometry {geometry_name!r}, "
            f"but that geometry is attached to boundary {geometry_support.get('name')!r}."
        )
    return geometry_spec


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_dumps_compact(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


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
