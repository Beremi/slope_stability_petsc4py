"""Compatibility adapter for legacy dict-based mesh definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .factories import build_asset


SOURCE_KIND_MAP: dict[str, str] = {
    "gmsh_tet4_physical_groups": "gmsh_tagged_simplex",
    "textmesh": "textmesh_bundle",
    "generated": "generated_geometry",
}


def _asset_meta(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("asset")
    return dict(raw) if isinstance(raw, dict) else {}


def _physics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("physics")
    return dict(raw) if isinstance(raw, dict) else {}


def _infer_source_kind(payload: dict[str, Any], *, fallback_name: str) -> str:
    meta = _asset_meta(payload)
    raw = meta.get("source_kind")
    if raw is not None:
        return str(raw)
    storage = payload.get("storage")
    if storage is not None:
        return SOURCE_KIND_MAP.get(str(storage), str(storage))
    if fallback_name.startswith("3d_"):
        return "gmsh_tagged_simplex"
    if fallback_name.startswith("2d_") and "generated" in fallback_name:
        return "generated_geometry"
    return "textmesh_bundle"


def _normalize_mesh_variants(payload: dict[str, Any], *, source_kind: str) -> tuple[dict[str, dict[str, Any]], str | None]:
    raw_variants = payload.get("mesh_variants")
    variants: dict[str, dict[str, Any]] = {}

    if isinstance(raw_variants, dict):
        for key, value in raw_variants.items():
            item = dict(value)
            if "source" not in item and any(name in item for name in ("path", "coordinates", "elements", "materials", "generator")):
                item = {"source": {k: item.pop(k) for k in list(item) if k in {"path", "coordinates", "elements", "materials", "generator"}}, **item}
            variants[str(key)] = item
    elif isinstance(raw_variants, list):
        for idx, value in enumerate(raw_variants):
            key = str(value.get("name", idx))
            variants[key] = dict(value)
    elif source_kind == "gmsh_tagged_simplex":
        for mesh_name in payload.get("mesh_files", []) or []:
            variants[str(mesh_name)] = {"source": {"path": str(mesh_name)}}
    elif source_kind == "generated_geometry":
        source = payload.get("source")
        if isinstance(source, dict):
            variants["default"] = {"source": dict(source)}

    default_variant = payload.get("default_variant")
    if default_variant is None:
        meta = _asset_meta(payload)
        default_variant = meta.get("default_variant")
    if default_variant is None and payload.get("default_mesh") is not None:
        default_variant = str(payload.get("default_mesh"))
    if default_variant is None and variants:
        default_variant = next(iter(variants))
    return variants, None if default_variant is None else str(default_variant)


def _infer_capabilities(payload: dict[str, Any]) -> tuple[str, ...]:
    physics = _physics_payload(payload)
    out: list[str] = []
    materials = physics.get("materials", payload.get("materials"))
    mechanics = physics.get("mechanics", payload.get("mechanical"))
    seepage = physics.get("seepage", payload.get("seepage"))
    if materials is not None or mechanics is not None or payload.get("dirichlet_labels") is not None:
        out.append("mechanics")
    if seepage:
        out.append("seepage")
    if not out:
        out.append("mechanics")
    return tuple(dict.fromkeys(out))


def asset_from_definition_dict(payload: dict[str, Any], *, asset_dir: Path, fallback_name: str | None = None):
    meta = _asset_meta(payload)
    name = str(meta.get("id", payload.get("name", fallback_name or asset_dir.name)))
    dimension = int(meta.get("dimension", payload.get("dimension", 3)))
    source_kind = _infer_source_kind(payload, fallback_name=name)
    mesh_variants, default_variant = _normalize_mesh_variants(payload, source_kind=source_kind)
    capabilities = _infer_capabilities(payload)
    physics = _physics_payload(payload)
    materials = physics.get("materials", payload.get("materials")) or []
    mechanics = physics.get("mechanics", payload.get("mechanical")) or {}
    seepage = physics.get("seepage", payload.get("seepage")) or None
    return build_asset(
        asset_id=name,
        asset_dir=asset_dir,
        dimension=dimension,
        source_kind=source_kind,
        capabilities=capabilities,
        default_variant=str(default_variant or next(iter(mesh_variants), "default")),
        mesh_variants=mesh_variants,
        materials=[dict(item) for item in materials],
        mechanics=dict(mechanics),
        seepage=None if seepage is None else dict(seepage),
        mesh_builder_kind=source_kind,
    )
