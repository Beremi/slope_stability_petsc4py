"""Shared config validation helpers for cases, profiles, and suites."""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any


OUTPUT_PRESETS = frozenset({"standard", "standard-continuation", "standard-seepage", "performance", "smoke", "none"})


def reject_unknown_fields(section_name: str, data: Mapping[str, Any], allowed: Set[str], message: str = "") -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        suffix = f"; {message}" if message else "."
        raise ValueError(f"{section_name} fields {unknown} are not supported{suffix}")


def reject_profile_default_repeats(section_name: str, data: Mapping[str, Any], profile_data: Mapping[str, Any]) -> None:
    repeated = sorted(key for key, value in data.items() if key in profile_data and profile_data[key] == value)
    if repeated:
        raise ValueError(
            f"{section_name} fields {repeated} duplicate the selected profile defaults; "
            "remove them from the case TOML or create/select a profile for that policy."
        )


def normalize_output_preset(value: object, *, section_name: str = "[output]") -> str:
    preset = str(value).strip().lower()
    if preset not in OUTPUT_PRESETS:
        raise ValueError(f"{section_name}.preset {preset!r} is not supported.")
    return preset


def validate_case_metadata(case: Mapping[str, Any], *, mesh: Mapping[str, Any], physics: Mapping[str, Any]) -> None:
    forbidden = {
        "mpi_ranks",
        "ranks",
        "nodes",
        "node_count",
        "wall_time",
        "time_limit",
        "machine",
        "partition",
        "queue",
        "output",
        "output_dir",
        "run_dir",
    }
    present = sorted(forbidden & set(case))
    if present:
        raise ValueError(f"[case] fields {present} are suite/launcher/artifact concerns, not mathematical case metadata.")
    tags = [str(tag).strip().lower() for tag in case.get("tags", [])]
    structured = _structured_case_tags(mesh=mesh, physics=physics)
    duplicates = sorted(tag for tag in tags if tag in structured or _is_composite_tag(tag))
    if duplicates:
        raise ValueError(
            f"[case].tags {duplicates} duplicate structured state; use tags only for orthogonal labels such as regression, scaling, validation, slow, nightly, or experimental."
        )


def _structured_case_tags(*, mesh: Mapping[str, Any], physics: Mapping[str, Any]) -> set[str]:
    element = str(mesh.get("element", "")).strip().lower()
    mechanics = dict(physics.get("mechanics", {}))
    seepage = dict(physics.get("seepage", {}))
    model = str(mechanics.get("model", "")).strip().lower()
    tags = {"mechanics"} if mechanics else set()
    if seepage:
        tags.add("seepage")
    if element:
        tags.add(element)
    if model:
        if "ssr" in model:
            tags.add("ssr")
        if "limit" in model or "ll" in model:
            tags.update({"ll", "limit-load"})
    asset = str(mesh.get("asset", "")).strip().lower()
    if asset.startswith("2d_"):
        tags.add("2d")
    if asset.startswith("3d_"):
        tags.add("3d")
    return tags


def _is_composite_tag(tag: str) -> bool:
    return any(part in tag for part in ("2d", "3d", "p1", "p2", "p4", "ssr", "limit-load", "mechanics", "seepage"))
