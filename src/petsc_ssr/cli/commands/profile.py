"""Profile inspection command helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from petsc_ssr.config import profiles as profile_module
from petsc_ssr.config.profiles import (
    load_continuation_profile,
    load_newton_profile,
    load_seepage_profile,
    load_solver_profile,
    native_linear_algorithm_selector,
    pc_variant_from_backend,
)


PROFILE_KINDS = ("solver", "continuation", "newton", "seepage")
PROFILE_VALIDATE_KINDS = ("all", *PROFILE_KINDS)


def explain_profile_payload(
    name: str,
    *,
    kind: str = "solver",
    world_size: int | None = None,
    element: str = "P4",
) -> dict[str, Any]:
    profile_kind = _profile_kind(kind)
    if profile_kind == "solver":
        profile = load_solver_profile(name, world_size=world_size)
        resolved = _json_safe(profile.data)
        degree = _element_degree(element)
        pc_policy = pc_variant_from_backend(resolved.get("pc_backend"), element_degree=degree)
        native_algorithm = native_linear_algorithm_selector(
            resolved.get("algorithm"),
            pc_variant=pc_policy.variant,
            deflation=bool(resolved.get("deflation", False)),
        )
        return {
            "kind": profile_kind,
            "profile": profile.name,
            "description": profile.description,
            "world_size": profile.world_size,
            "element": f"P{degree}",
            "linear_algorithm": resolved.get("algorithm"),
            "native_linear_algorithm": native_algorithm,
            "pc": {
                "backend": resolved.get("pc_backend"),
                "variant": pc_policy.variant,
                "requested_variant": pc_policy.requested_variant,
                "fallback_reason": pc_policy.fallback_reason,
            },
            "pmg": _resolved_pmg_payload(resolved),
            "resolved": resolved,
        }

    loader = {
        "continuation": load_continuation_profile,
        "newton": load_newton_profile,
        "seepage": load_seepage_profile,
    }[profile_kind]
    profile = loader(name)
    return {
        "kind": profile_kind,
        "profile": profile.name,
        "description": profile.description,
        "resolved": _json_safe(profile.data),
    }


def validate_profiles_payload(
    *,
    kind: str = "all",
    world_sizes: list[int] | tuple[int, ...] | None = None,
    elements: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    profile_kind = _validate_kind(kind)
    selected_kinds = PROFILE_KINDS if profile_kind == "all" else (profile_kind,)
    resolved_world_sizes = _world_sizes(world_sizes)
    resolved_elements = [f"P{_element_degree(element)}" for element in (elements or ("P1", "P4"))]
    profiles: dict[str, list[dict[str, Any]]] = {item: [] for item in selected_kinds}
    issues: list[dict[str, str]] = []

    for current_kind in selected_kinds:
        for path in _profile_paths(current_kind):
            try:
                profiles[current_kind].append(
                    _validate_profile_path(
                        path,
                        kind=current_kind,
                        world_sizes=resolved_world_sizes,
                        elements=resolved_elements,
                    )
                )
            except Exception as exc:
                issues.append({"kind": current_kind, "path": str(path), "error": str(exc)})

    counts = {current_kind: len(profiles[current_kind]) for current_kind in selected_kinds}
    return {
        "ok": not issues,
        "kind": profile_kind,
        "world_sizes": resolved_world_sizes,
        "elements": resolved_elements,
        "counts": counts,
        "profiles": profiles,
        "issues": issues,
    }


def _validate_profile_path(
    path: Path,
    *,
    kind: str,
    world_sizes: list[int],
    elements: list[str],
) -> dict[str, Any]:
    name = path.stem
    if kind == "solver":
        checks: list[dict[str, Any]] = []
        for world_size in world_sizes:
            profile = load_solver_profile(name, world_size=world_size)
            resolved = _json_safe(profile.data)
            for element in elements:
                degree = _element_degree(element)
                pc_policy = pc_variant_from_backend(resolved.get("pc_backend"), element_degree=degree)
                native_algorithm = native_linear_algorithm_selector(
                    resolved.get("algorithm"),
                    pc_variant=pc_policy.variant,
                    deflation=bool(resolved.get("deflation", False)),
                )
                checks.append(
                    {
                        "world_size": int(world_size),
                        "element": f"P{degree}",
                        "linear_algorithm": resolved.get("algorithm"),
                        "native_linear_algorithm": native_algorithm,
                        "pc_variant": pc_policy.variant,
                        "requested_pc_variant": pc_policy.requested_variant,
                        "pc_variant_fallback_reason": pc_policy.fallback_reason,
                        "pmg_p2_active_ranks": resolved.get("pmg_shell_p2_active_ranks"),
                        "pmg_p1_active_ranks": resolved.get("pmg_shell_p1_active_ranks"),
                        "pmg_apply_backend": resolved.get("pmg_apply_backend"),
                        "pmg_coarse_pc_type": resolved.get("pmg_coarse_pc_type"),
                        "pmg_coarse_telescope_ksp_max_it": resolved.get("pmg_coarse_telescope_ksp_max_it"),
                        "pmg_smoother_max_it": resolved.get("pmg_smoother_max_it"),
                    }
                )
        return {"profile": name, "path": str(path), "checks": checks}

    loader = {
        "continuation": load_continuation_profile,
        "newton": load_newton_profile,
        "seepage": load_seepage_profile,
    }[kind]
    profile = loader(name)
    return {
        "profile": profile.name,
        "path": str(path),
        "algorithm": profile.data.get("algorithm"),
        "resolved": _json_safe(profile.data),
    }


def _profile_paths(kind: str) -> list[Path]:
    roots = {
        "solver": profile_module.SOLVER_PROFILE_DIR,
        "continuation": profile_module.CONTINUATION_PROFILE_DIR,
        "newton": profile_module.NEWTON_PROFILE_DIR,
        "seepage": profile_module.SEEPAGE_PROFILE_DIR,
    }
    return sorted(roots[kind].glob("*.toml"))


def _profile_kind(kind: str) -> str:
    value = str(kind).strip().lower()
    if value not in PROFILE_KINDS:
        allowed = ", ".join(PROFILE_KINDS)
        raise ValueError(f"Unknown profile kind {kind!r}; expected one of {allowed}.")
    return value


def _validate_kind(kind: str) -> str:
    value = str(kind).strip().lower()
    if value not in PROFILE_VALIDATE_KINDS:
        allowed = ", ".join(PROFILE_VALIDATE_KINDS)
        raise ValueError(f"Unknown profile validation kind {kind!r}; expected one of {allowed}.")
    return value


def _world_sizes(values: list[int] | tuple[int, ...] | None) -> list[int]:
    items = [1, 32] if values is None else [int(value) for value in values]
    if not items:
        raise ValueError("profile validate requires at least one world size.")
    if any(value < 1 for value in items):
        raise ValueError("profile validate world sizes must be positive.")
    return sorted(set(items))


def _resolved_pmg_payload(resolved: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank_policy": resolved.get("pmg_rank_policy"),
        "apply_backend": resolved.get("pmg_apply_backend"),
        "p2_active_ranks": resolved.get("pmg_shell_p2_active_ranks"),
        "p1_active_ranks": resolved.get("pmg_shell_p1_active_ranks"),
        "p2_policy": resolved.get("pmg_shell_p2_rank_policy"),
        "p1_policy": resolved.get("pmg_shell_p1_rank_policy"),
        "subcomm_type": resolved.get("pmg_shell_subcomm_type"),
        "fine_ksp_max_it": resolved.get("pmg_shell_fine_ksp_max_it"),
        "p2_ksp_max_it": resolved.get("pmg_shell_p2_ksp_max_it"),
        "smoother_ksp_type": resolved.get("pmg_smoother_ksp_type"),
        "smoother_pc_type": resolved.get("pmg_smoother_pc_type"),
        "smoother_max_it": resolved.get("pmg_smoother_max_it"),
        "coarse_pc_type": resolved.get("pmg_coarse_pc_type"),
        "coarse_lu_max_dofs": resolved.get("pmg_coarse_lu_max_dofs"),
        "coarse_redundant_group_size": resolved.get("pmg_coarse_redundant_group_size"),
        "coarse_gamg_aggressive_square_graph": resolved.get("pmg_coarse_gamg_aggressive_square_graph"),
        "coarse_telescope_active_ranks": resolved.get("pmg_coarse_telescope_active_ranks"),
        "coarse_telescope_subcomm_type": resolved.get("pmg_coarse_telescope_subcomm_type"),
        "coarse_telescope_ksp_type": resolved.get("pmg_coarse_telescope_ksp_type"),
        "coarse_telescope_ksp_rtol": resolved.get("pmg_coarse_telescope_ksp_rtol"),
        "coarse_telescope_ksp_max_it": resolved.get("pmg_coarse_telescope_ksp_max_it"),
        "coarse_telescope_pc_type": resolved.get("pmg_coarse_telescope_pc_type"),
        "p2_telescope_active_ranks": resolved.get("pmg_p2_telescope_active_ranks"),
        "p2_telescope_subcomm_type": resolved.get("pmg_p2_telescope_subcomm_type"),
        "p2_telescope_ksp_type": resolved.get("pmg_p2_telescope_ksp_type"),
        "p2_telescope_ksp_rtol": resolved.get("pmg_p2_telescope_ksp_rtol"),
        "p2_telescope_ksp_max_it": resolved.get("pmg_p2_telescope_ksp_max_it"),
        "p2_telescope_pc_type": resolved.get("pmg_p2_telescope_pc_type"),
    }


def _element_degree(element: str) -> int:
    text = str(element).strip().upper()
    if text.startswith("P"):
        text = text[1:]
    degree = int(text)
    if degree < 1:
        raise ValueError(f"Element degree must be positive, got {element!r}.")
    return degree


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
