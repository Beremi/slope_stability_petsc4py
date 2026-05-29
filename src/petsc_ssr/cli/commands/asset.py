"""Asset validation command helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from petsc_ssr.assets.support.physical_names import parse_gmsh_physical_names


def validate_asset_payload(asset: str) -> dict[str, Any]:
    from petsc_ssr.assets import load_problem_asset

    asset_id = Path(asset).name if Path(asset).exists() else str(asset)
    definition = load_problem_asset(asset_id)
    variants = definition.list_variants()
    mechanics = definition.mechanics_spec()
    seepage = definition.seepage_spec()
    region_assignment = dict(getattr(definition, "_region_assignment", getattr(mechanics, "region_assignment", {}) if mechanics is not None else {}))
    materials = dict(getattr(definition, "_materials", getattr(mechanics, "materials", {}) if mechanics is not None else {}))
    geometry_getter = getattr(definition, "boundary_geometry_specs", None)
    boundary_geometry = geometry_getter() if callable(geometry_getter) else {}
    errors: list[str] = []
    supports: dict[str, list[str]] = {"regions": [], "boundaries": [], "nodesets": []}
    supports_by_variant: dict[str, dict[str, list[str]]] = {}
    physical_by_variant: dict[str, dict[str, dict[str, int]]] = {}
    native_manifest_contracts: dict[str, list[dict[str, Any]]] = {}
    if definition.default_variant not in variants:
        errors.append(f"default variant {definition.default_variant!r} is not declared in mesh variants")
    for name, variant in variants.items():
        if variant.mesh_path is not None and not Path(variant.mesh_path).exists():
            errors.append(f"variant {name!r} mesh is missing: {variant.mesh_path}")
            continue
        if variant.mesh_path is None:
            continue
        physical = parse_gmsh_physical_names(variant.mesh_path)
        physical_by_variant[str(name)] = physical
        supports_by_variant[str(name)] = {key: sorted(value) for key, value in physical.items()}
    default_key = definition.default_variant
    if default_key not in supports_by_variant and f"{default_key}.msh" in supports_by_variant:
        default_key = f"{default_key}.msh"
    if default_key in supports_by_variant:
        supports = supports_by_variant[default_key]
    elif supports_by_variant:
        supports = supports_by_variant[sorted(supports_by_variant)[0]]
    for variant_name, physical in physical_by_variant.items():
        _validate_asset_supports_for_variant(
            errors,
            variant_name=variant_name,
            physical=physical,
            mechanics=mechanics,
            seepage=seepage,
            region_assignment=region_assignment,
            boundary_geometry=boundary_geometry,
        )
        native_manifest_contracts[variant_name] = _native_manifest_contracts_for_variant(
            errors,
            definition=definition,
            variant_name=variant_name,
            mechanics=mechanics,
        )
    for region, material in region_assignment.items():
        if materials and material not in materials:
            errors.append(f"region {region!r} references unknown material {material!r}")
    return {
        "asset": definition.asset_id,
        "dimension": definition.dimension,
        "source_kind": definition.source_kind,
        "variants": sorted(variants),
        "default_variant": definition.default_variant,
        "default_profile": definition.default_profile,
        "mechanics": mechanics is not None,
        "seepage": seepage is not None,
        "supports": supports,
        "variant_supports": supports_by_variant,
        "native_manifest_contracts": native_manifest_contracts,
        "errors": errors,
    }


def validate_all_assets_payload() -> dict[str, Any]:
    from petsc_ssr.assets import available_problem_assets

    reports = [validate_asset_payload(asset_id) for asset_id in available_problem_assets()]
    error_count = sum(len(report["errors"]) for report in reports)
    return {
        "count": len(reports),
        "errors": error_count,
        "assets": reports,
    }


def _validate_asset_supports_for_variant(
    errors: list[str],
    *,
    variant_name: str,
    physical: dict[str, dict[str, int]],
    mechanics: Any | None,
    seepage: Any | None,
    region_assignment: dict[str, str],
    boundary_geometry: dict[str, tuple[str, int]],
) -> None:
    region_supports = set(physical.get("regions", {}))
    boundary_supports = set(physical.get("boundaries", {})) | set(physical.get("nodesets", {}))
    boundary_only_supports = set(physical.get("boundaries", {}))
    assigned_regions = set(region_assignment)
    for region in sorted(assigned_regions - region_supports):
        errors.append(f"variant {variant_name!r} region {region!r} is not declared by mesh physical names")
    for region in sorted(region_supports - assigned_regions):
        errors.append(f"variant {variant_name!r} physical region {region!r} has no material assignment")
    if mechanics is not None:
        for profile_name, profile in mechanics.profiles.items():
            for rule in profile.dirichlet:
                if not rule.target:
                    errors.append(f"profile {profile_name!r} has an empty Dirichlet target")
                elif rule.target not in boundary_supports:
                    errors.append(f"variant {variant_name!r} profile {profile_name!r} Dirichlet target {rule.target!r} is not a boundary/nodeset support")
            for rule in profile.neumann:
                if not rule.target:
                    errors.append(f"profile {profile_name!r} has an empty Neumann target")
                elif rule.target not in boundary_only_supports:
                    errors.append(f"variant {variant_name!r} profile {profile_name!r} Neumann target {rule.target!r} is not a boundary support")
                _validate_boundary_geometry_link(
                    errors,
                    owner=f"profile {profile_name!r} Neumann target {rule.target!r}",
                    target=rule.target,
                    geometry=rule.geometry,
                    boundary_geometry=boundary_geometry,
                )
    if seepage is not None:
        for rule in seepage.head_bcs:
            if not rule.target:
                errors.append("seepage BC has an empty target")
            elif rule.target not in boundary_supports:
                errors.append(f"variant {variant_name!r} seepage BC target {rule.target!r} is not a boundary/nodeset support")
        for rule in seepage.flux_bcs:
            if not rule.target:
                errors.append("seepage flux BC has an empty target")
            elif rule.target not in boundary_only_supports:
                errors.append(f"variant {variant_name!r} seepage flux target {rule.target!r} is not a boundary support")
            _validate_boundary_geometry_link(
                errors,
                owner=f"seepage flux target {rule.target!r}",
                target=rule.target,
                geometry=rule.geometry,
                boundary_geometry=boundary_geometry,
            )
    for geometry_name, (support_boundary, geometry_order) in boundary_geometry.items():
        if not support_boundary:
            errors.append(f"boundary geometry {geometry_name!r} has an empty support_boundary")
        elif support_boundary not in boundary_only_supports:
            errors.append(f"variant {variant_name!r} boundary geometry {geometry_name!r} support {support_boundary!r} is not a boundary support")
        if int(geometry_order) < 1:
            errors.append(f"boundary geometry {geometry_name!r} has invalid geometry_order {geometry_order!r}")


def _validate_boundary_geometry_link(
    errors: list[str],
    *,
    owner: str,
    target: str,
    geometry: str | None,
    boundary_geometry: dict[str, tuple[str, int]],
) -> None:
    if geometry is None:
        return
    geometry_name = str(geometry)
    if geometry_name not in boundary_geometry:
        errors.append(f"{owner} references unknown boundary geometry {geometry_name!r}")
        return
    support_boundary, _order = boundary_geometry[geometry_name]
    if str(support_boundary) != str(target):
        errors.append(
            f"{owner} references boundary geometry {geometry_name!r}, "
            f"but that geometry is attached to boundary {support_boundary!r}"
        )


def _native_manifest_contracts_for_variant(
    errors: list[str],
    *,
    definition: Any,
    variant_name: str,
    mechanics: Any | None,
) -> list[dict[str, Any]]:
    from petsc_ssr.problem_asset_runtime import ResolvedAsset, build_native_label_table_contracts, build_native_problem_manifest

    profiles = sorted(mechanics.profiles) if mechanics is not None else [str(getattr(definition, "default_profile", "default"))]
    contracts: list[dict[str, Any]] = []
    for profile_name in profiles:
        try:
            resolved_variant = definition.resolve_variant(variant_name, profile=profile_name)
            resolved = ResolvedAsset(
                definition=definition,
                variant_name=resolved_variant.name,
                variant=resolved_variant.as_dict(),
                resolved_variant=resolved_variant,
                mesh_path=resolved_variant.mesh_path,
            )
            manifest = build_native_problem_manifest(
                resolved,
                case_id=f"asset-validate:{definition.asset_id}:{variant_name}:{profile_name}",
                analysis="asset-validation",
                solver_profile="asset-validation",
            )
        except Exception as exc:
            errors.append(f"variant {variant_name!r} profile {profile_name!r} native manifest contract failed: {exc}")
            continue
        contracts.append(
            {
                "profile": profile_name,
                "support_counts": dict(manifest["dmplex"]["support_counts"]),
                "rule_counts": dict(manifest["rule_counts"]),
                "label_tables": build_native_label_table_contracts(resolved),
                "native_inputs": dict(manifest["native_inputs"]),
            }
        )
    return contracts
