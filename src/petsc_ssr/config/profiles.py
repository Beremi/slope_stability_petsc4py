"""Solver profile loading and rank-adaptive policy resolution."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .validators import reject_unknown_fields as _reject_unknown_fields


ENGINE_ROOT = Path(__file__).resolve().parents[3]
SOLVER_PROFILE_DIR = ENGINE_ROOT / "configs" / "solver_profiles"
CONTINUATION_PROFILE_DIR = ENGINE_ROOT / "configs" / "continuation_profiles"
NEWTON_PROFILE_DIR = ENGINE_ROOT / "configs" / "newton_profiles"
SEEPAGE_PROFILE_DIR = ENGINE_ROOT / "configs" / "seepage_profiles"
SOLVER_PROFILE_ALIASES = {
    "baseline-pmg-deflated": "pmg-deflated-baseline",
}


@dataclass(frozen=True, slots=True)
class ResolvedSolverProfile:
    name: str
    description: str
    world_size: int
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResolvedControlProfile:
    name: str
    description: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResolvedPcVariant:
    """Concrete native PC variant selected from profile policy."""

    variant: str
    requested_variant: str
    fallback_reason: str | None = None


def load_solver_profile(name: str | None, *, world_size: int | None = None) -> ResolvedSolverProfile:
    """Load a named solver profile and resolve policy fields to concrete options."""

    requested_name = (name or "").strip()
    profile_name = SOLVER_PROFILE_ALIASES.get(requested_name, requested_name)
    if not profile_name:
        return ResolvedSolverProfile("", "", _world_size(world_size), {})
    profile_path = SOLVER_PROFILE_DIR / f"{profile_name}.toml"
    if not profile_path.exists():
        raise ValueError(f"Unknown solver profile {profile_name!r}; expected {profile_path}.")
    payload = tomllib.loads(profile_path.read_text(encoding="utf-8"))
    _reject_unknown_fields(
        f"solver profile {profile_name!r}",
        payload,
        {"description", "linear", "deflation", "pc", "pmg", "petsc", "petsc_options"},
        "solver profiles may describe algorithm policy, PMG policy, and PETSc option defaults only.",
    )

    world = _world_size(world_size)
    data: dict[str, Any] = {
        "profile": profile_name,
        "profile_description": str(payload.get("description", "")),
        "resolved_world_size": world,
    }
    if requested_name and requested_name != profile_name:
        data["profile_alias"] = requested_name
    data.update(_linear_data(payload))
    data.update(_deflation_data(payload.get("deflation", {})))
    data.update(_pc_data(payload.get("pc", {})))
    data.update(_pmg_data(payload.get("pmg", {}), world_size=world))
    data.update(_petsc_data(payload))
    return ResolvedSolverProfile(
        profile_name,
        str(payload.get("description", "")),
        world,
        data,
    )


def load_continuation_profile(name: str | None) -> ResolvedControlProfile:
    """Load a named continuation policy profile."""

    return _load_control_profile(
        name,
        profile_dir=CONTINUATION_PROFILE_DIR,
        section="continuation",
        allowed_fields={
            "method",
            "algorithm",
            "predictor",
            "omega_step_controller",
            "secant_correction_mode",
            "first_newton_warm_start_mode",
            "lambda_init",
            "d_lambda_init",
            "d_lambda_min",
            "d_lambda_diff_scaled_min",
            "lambda_ell",
            "omega_max",
            "step_max",
            "d_omega_ini_scale",
            "d_t_min",
            "omega_no_increase_newton_threshold",
            "omega_half_newton_threshold",
            "omega_target_newton_iterations",
            "omega_adapt_min_scale",
            "omega_adapt_max_scale",
            "omega_hard_newton_threshold",
            "omega_hard_linear_threshold",
            "omega_efficiency_floor",
            "omega_efficiency_drop_ratio",
            "omega_efficiency_window",
            "omega_hard_shrink_scale",
            "step_length_cap_mode",
            "step_length_cap_factor",
            "init_newton_stopping_criterion",
            "init_newton_stopping_tol",
            "fine_newton_stopping_criterion",
            "fine_newton_stopping_tol",
            "fine_switch_mode",
            "fine_switch_distance_factor",
        },
        message="continuation profiles own reusable continuation algorithm policy.",
    )


def load_newton_profile(name: str | None) -> ResolvedControlProfile:
    """Load a named Newton/nonlinear policy profile."""

    return _load_control_profile(
        name,
        profile_dir=NEWTON_PROFILE_DIR,
        section="newton",
        allowed_fields={
            "it_max",
            "algorithm",
            "it_damp_max",
            "tol",
            "r_min",
            "stopping_criterion",
            "stopping_tol",
            "line_search",
            "armijo_alpha0",
            "armijo_c1",
            "armijo_shrink",
            "armijo_max_ls",
            "armijo_rescale_trial_to_omega",
            "armijo_fallback_to_alg5",
        },
        message="Newton profiles own reusable nonlinear solve and line-search policy.",
    )


def load_seepage_profile(name: str | None) -> ResolvedControlProfile:
    """Load a named seepage runtime policy profile."""

    return _load_control_profile(
        name,
        profile_dir=SEEPAGE_PROFILE_DIR,
        section="seepage",
        allowed_fields={
            "linear_tolerance",
            "linear_max_iter",
            "nonlinear_max_iter",
        },
        message="seepage profiles own reusable hydro runtime policy.",
    )


def pc_variant_from_backend(
    pc_backend: str | None,
    *,
    element_degree: int,
    supported: tuple[str, ...] = ("gamg", "bddc", "fetidp", "pmg", "none"),
) -> ResolvedPcVariant:
    """Resolve a profile PC backend to the concrete native ``pc_variant``.

    Solver profiles own the requested PC backend. P1 elements have no
    p-hierarchy, so a PMG request resolves to GAMG and records the fallback.
    """

    degree = int(element_degree)
    backend = str(pc_backend or "").strip().lower()
    aliases = {
        "pmg": "pmg",
        "pmg_shell": "pmg",
        "gamg": "gamg",
        "bddc": "bddc",
        "fetidp": "fetidp",
        "none": "none",
    }
    if backend in {"", "hypre"}:
        requested = "pmg" if degree > 1 else "gamg"
    elif backend in aliases:
        requested = aliases[backend]
    else:
        raise ValueError(f"Unsupported solver profile PC backend {pc_backend!r}.")

    supported_set = {str(value).strip().lower() for value in supported}
    if requested not in supported_set:
        allowed = ", ".join(supported)
        raise ValueError(f"Solver profile PC backend {pc_backend!r} resolves to {requested!r}, but this path supports only {allowed}.")
    if degree == 1 and requested == "pmg":
        if "gamg" not in supported_set:
            allowed = ", ".join(supported)
            raise ValueError(f"Solver profile PC backend {pc_backend!r} requests PMG for P1, but this path supports only {allowed}.")
        return ResolvedPcVariant("gamg", requested, "p1_has_no_p_hierarchy")
    return ResolvedPcVariant(requested, requested, None)


def native_linear_algorithm_selector(
    profile_algorithm: object,
    *,
    pc_variant: str,
    deflation: bool,
) -> str:
    """Map profile-level linear policy to the current native registry selector.

    Solver profiles expose stable, user-facing policy names such as
    ``ksp_deflated``.  The current native registry is still organized around the
    concrete preconditioner/deflation family selected for the resolved element
    and PC policy.  Keeping this mapping in the profile layer makes the
    translation visible to case, suite, and manifest code instead of hiding it
    in one runner path.
    """

    algorithm = str(profile_algorithm or "").strip().lower().replace("_", "-")
    variant = str(pc_variant).strip().lower()
    if algorithm in {"direct-debug", "directdebug"}:
        return "debug-direct"
    if variant == "pmg":
        return "pmg-deflated" if bool(deflation) else "pmg"
    if variant in {"gamg", "bddc", "fetidp"}:
        return variant
    if variant == "none":
        return "none"
    if algorithm in {"ksp-deflated", "fgmres"}:
        return "pmg-deflated" if bool(deflation) else "pmg"
    return algorithm


def _load_control_profile(
    name: str | None,
    *,
    profile_dir: Path,
    section: str,
    allowed_fields: set[str],
    message: str,
) -> ResolvedControlProfile:
    profile_name = (name or "").strip()
    if not profile_name:
        return ResolvedControlProfile("", "", {})
    profile_path = profile_dir / f"{profile_name}.toml"
    if not profile_path.exists():
        raise ValueError(f"Unknown {section} profile {profile_name!r}; expected {profile_path}.")
    payload = tomllib.loads(profile_path.read_text(encoding="utf-8"))
    _reject_unknown_fields(
        f"{section} profile {profile_name!r}",
        payload,
        {"description", section},
        message,
    )
    data = dict(payload.get(section, {}))
    _reject_unknown_fields(f"[{section}]", data, allowed_fields, message)
    if section == "continuation":
        _normalize_continuation_profile_data(profile_name, data)
    if section == "newton":
        _normalize_newton_profile_data(profile_name, data)
    resolved = {
        "profile": profile_name,
        "profile_description": str(payload.get("description", "")),
    }
    resolved.update(data)
    return ResolvedControlProfile(profile_name, str(payload.get("description", "")), resolved)


def _normalize_continuation_profile_data(profile_name: str, data: dict[str, Any]) -> None:
    method = str(data.get("method", "indirect")).strip().lower().replace("_", "-")
    if method not in {"indirect", "direct"}:
        raise ValueError(
            f"Continuation profile {profile_name!r} uses unsupported method "
            f"{data.get('method')!r}; expected 'indirect' or 'direct'."
        )
    data["method"] = method
    algorithm = str(data.get("algorithm", method)).strip().lower().replace("_", "-")
    aliases = {
        "indirect": "indirect",
        "indirect-ssr": "indirect",
        "direct": "direct",
        "direct-limit-load": "direct",
    }
    if algorithm not in aliases:
        raise ValueError(
            f"Continuation profile {profile_name!r} uses unsupported algorithm "
            f"{data.get('algorithm')!r}; expected 'indirect' or 'direct'."
        )
    data["algorithm"] = aliases[algorithm]
    if data["algorithm"] != method:
        raise ValueError(
            f"Continuation profile {profile_name!r} algorithm {data['algorithm']!r} "
            f"does not match method {method!r}."
        )
    controller = str(data.get("omega_step_controller", "classic")).strip().lower()
    aliases = {
        "classic": "classic",
        "legacy": "classic",
    }
    if controller not in aliases:
        raise ValueError(
            f"Continuation profile {profile_name!r} uses unsupported omega_step_controller "
            f"{data.get('omega_step_controller')!r}; expected 'classic'."
        )
    data["omega_step_controller"] = aliases[controller]


def _normalize_newton_profile_data(profile_name: str, data: dict[str, Any]) -> None:
    default_algorithm = "fixed-load" if "limit-load" in profile_name else "indirect-ssr"
    algorithm = str(data.get("algorithm", default_algorithm)).strip().lower().replace("_", "-")
    aliases = {
        "fixed": "fixed-load",
        "fixed-load": "fixed-load",
        "regularized-fixed-load": "fixed-load",
        "indirect": "indirect-ssr",
        "indirect-ssr": "indirect-ssr",
        "regularized-newton": "indirect-ssr",
    }
    if algorithm not in aliases:
        raise ValueError(
            f"Newton profile {profile_name!r} uses unsupported algorithm "
            f"{data.get('algorithm')!r}; expected 'fixed-load' or 'indirect-ssr'."
        )
    data["algorithm"] = aliases[algorithm]


def _world_size(value: int | None) -> int:
    if value is not None:
        return max(1, int(value))
    for env_name in ("PETSC_SSR_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS"):
        raw = os.environ.get(env_name)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    try:
        from mpi4py import MPI

        return max(1, int(MPI.COMM_WORLD.Get_size()))
    except Exception:
        return 1


def _linear_data(payload: dict[str, Any]) -> dict[str, Any]:
    modern = dict(payload.get("linear", {}))
    _reject_unknown_fields(
        "[linear]",
        modern,
        {
            "algorithm",
            "solver_type",
            "ksp_type",
            "norm_type",
            "tolerance",
            "rtol",
            "max_iterations",
            "max_it",
            "reuse_preconditioner",
            "recycle_preconditioner",
        },
        "linear profiles own reusable KSP/outer-solver policy.",
    )
    data: dict[str, Any] = {}
    if "algorithm" in modern and "solver_type" not in data:
        algorithm = str(modern["algorithm"]).strip().lower()
        if algorithm not in {"ksp_deflated", "fgmres", "direct_debug"}:
            raise ValueError(
                f"[linear].algorithm {modern['algorithm']!r} is not supported; "
                "expected 'ksp_deflated', 'fgmres', or 'direct_debug'."
            )
        data["algorithm"] = algorithm
        data["solver_type"] = "PETSC_DMPLEX_C_FGMRES" if algorithm in {"ksp_deflated", "fgmres"} else algorithm
    if "solver_type" in modern:
        data["solver_type"] = modern["solver_type"]
    if "ksp_type" in modern:
        data["ksp_type"] = modern["ksp_type"]
    if "norm_type" in modern:
        data["norm_type"] = modern["norm_type"]
    if "tolerance" in modern:
        data["tolerance"] = modern["tolerance"]
    if "rtol" in modern:
        data["tolerance"] = modern["rtol"]
    if "max_iterations" in modern:
        data["max_iterations"] = modern["max_iterations"]
    if "max_it" in modern:
        data["max_iterations"] = modern["max_it"]
    if "reuse_preconditioner" in modern:
        data["recycle_preconditioner"] = modern["reuse_preconditioner"]
    if "recycle_preconditioner" in modern:
        data["recycle_preconditioner"] = modern["recycle_preconditioner"]
    return data


def _deflation_data(raw: object) -> dict[str, Any]:
    data = dict(raw or {})
    _reject_unknown_fields(
        "[deflation]",
        data,
        {"enabled", "basis_tolerance", "max_vectors", "recycle", "solver"},
        "deflation profiles expose stable deflation policy only.",
    )
    out: dict[str, Any] = {}
    if "enabled" in data:
        out["deflation"] = bool(data["enabled"])
    if "solver" in data:
        out["deflation_solver"] = str(data["solver"])
    if "basis_tolerance" in data:
        out["deflation_basis_tolerance"] = data["basis_tolerance"]
    if "max_vectors" in data:
        out["max_deflation_basis_vectors"] = data["max_vectors"]
    return out


def _pc_data(raw: object) -> dict[str, Any]:
    data = dict(raw or {})
    _reject_unknown_fields("[pc]", data, {"type"}, "PC profile policy uses a named backend.")
    if not data:
        return {}
    pc_type = str(data.get("type", "")).strip()
    return {"pc_backend": pc_type} if pc_type else {}


def _pmg_data(raw: object, *, world_size: int) -> dict[str, Any]:
    data = dict(raw or {})
    p2 = dict(data.pop("p2", {}))
    p1 = dict(data.pop("p1", {}))
    coarse = dict(data.pop("coarse", {}))
    p2_telescope = dict(p2.pop("telescope", {}))
    coarse_telescope = dict(coarse.pop("telescope", {}))
    _reject_unknown_fields(
        "[pmg]",
        data,
        {
            "rank_policy",
            "apply_backend",
            "coarse_pc_type",
            "coarse_lu_max_dofs",
            "coarse_redundant_group_size",
            "coarse_gamg_aggressive_square_graph",
            "coarse_telescope_active_ranks",
            "coarse_telescope_subcomm_type",
            "coarse_telescope_ksp_type",
            "coarse_telescope_ksp_rtol",
            "coarse_telescope_ksp_max_it",
            "coarse_telescope_pc_type",
            "p2_telescope_active_ranks",
            "p2_telescope_subcomm_type",
            "p2_telescope_ksp_type",
            "p2_telescope_ksp_rtol",
            "p2_telescope_ksp_max_it",
            "p2_telescope_pc_type",
            "smoother_ksp_type",
            "smoother_pc_type",
            "smoother_max_it",
            "pmg_apply_backend",
            "pmg_coarse_pc_type",
            "pmg_coarse_lu_max_dofs",
            "pmg_coarse_redundant_group_size",
            "pmg_coarse_gamg_aggressive_square_graph",
            "pmg_coarse_telescope_active_ranks",
            "pmg_coarse_telescope_subcomm_type",
            "pmg_coarse_telescope_ksp_type",
            "pmg_coarse_telescope_ksp_rtol",
            "pmg_coarse_telescope_ksp_max_it",
            "pmg_coarse_telescope_pc_type",
            "pmg_p2_telescope_active_ranks",
            "pmg_p2_telescope_subcomm_type",
            "pmg_p2_telescope_ksp_type",
            "pmg_p2_telescope_ksp_rtol",
            "pmg_p2_telescope_ksp_max_it",
            "pmg_p2_telescope_pc_type",
            "pmg_smoother_ksp_type",
            "pmg_smoother_pc_type",
            "pmg_smoother_max_it",
            "pmg_shell_p2_active_ranks",
            "pmg_shell_p1_active_ranks",
            "pmg_shell_subcomm_type",
            "pmg_shell_fine_ksp_max_it",
            "pmg_shell_p2_ksp_max_it",
            "subcomm_type",
            "fine_ksp_max_it",
            "p2_ksp_max_it",
        },
        "PMG profiles own rank-adaptive shell policy and smoother limits.",
    )
    out: dict[str, Any] = {}
    key_map = {
        "apply_backend": "pmg_apply_backend",
        "coarse_pc_type": "pmg_coarse_pc_type",
        "coarse_lu_max_dofs": "pmg_coarse_lu_max_dofs",
        "coarse_redundant_group_size": "pmg_coarse_redundant_group_size",
        "coarse_gamg_aggressive_square_graph": "pmg_coarse_gamg_aggressive_square_graph",
        "coarse_telescope_active_ranks": "pmg_coarse_telescope_active_ranks",
        "coarse_telescope_subcomm_type": "pmg_coarse_telescope_subcomm_type",
        "coarse_telescope_ksp_type": "pmg_coarse_telescope_ksp_type",
        "coarse_telescope_ksp_rtol": "pmg_coarse_telescope_ksp_rtol",
        "coarse_telescope_ksp_max_it": "pmg_coarse_telescope_ksp_max_it",
        "coarse_telescope_pc_type": "pmg_coarse_telescope_pc_type",
        "p2_telescope_active_ranks": "pmg_p2_telescope_active_ranks",
        "p2_telescope_subcomm_type": "pmg_p2_telescope_subcomm_type",
        "p2_telescope_ksp_type": "pmg_p2_telescope_ksp_type",
        "p2_telescope_ksp_rtol": "pmg_p2_telescope_ksp_rtol",
        "p2_telescope_ksp_max_it": "pmg_p2_telescope_ksp_max_it",
        "p2_telescope_pc_type": "pmg_p2_telescope_pc_type",
        "smoother_ksp_type": "pmg_smoother_ksp_type",
        "smoother_pc_type": "pmg_smoother_pc_type",
        "smoother_max_it": "pmg_smoother_max_it",
        "subcomm_type": "pmg_shell_subcomm_type",
        "fine_ksp_max_it": "pmg_shell_fine_ksp_max_it",
        "p2_ksp_max_it": "pmg_shell_p2_ksp_max_it",
    }
    for key, value in data.items():
        if key == "rank_policy":
            out["pmg_rank_policy"] = value
        elif key in key_map:
            out[key_map[key]] = value
        else:
            out[key] = value
    if coarse:
        _reject_unknown_fields(
            "[pmg.coarse]",
            coarse,
            {"pc_type", "lu_max_dofs", "redundant_group_size", "gamg_aggressive_square_graph"},
            "PMG coarse profile policy owns P1 coarse-solve PETSc policy.",
        )
        _copy_present(
            coarse,
            out,
            {
                "pc_type": "pmg_coarse_pc_type",
                "lu_max_dofs": "pmg_coarse_lu_max_dofs",
                "redundant_group_size": "pmg_coarse_redundant_group_size",
                "gamg_aggressive_square_graph": "pmg_coarse_gamg_aggressive_square_graph",
            },
        )
    if coarse_telescope:
        _copy_telescope_data("[pmg.coarse.telescope]", coarse_telescope, out, prefix="pmg_coarse_telescope", world_size=world_size)
    if p2_telescope:
        _copy_telescope_data("[pmg.p2.telescope]", p2_telescope, out, prefix="pmg_p2_telescope", world_size=world_size)
    if p2:
        out["pmg_shell_p2_active_ranks"] = _resolve_active_ranks("[pmg.p2]", p2, world_size=world_size)
        out["pmg_shell_p2_rank_policy"] = _policy_name(p2)
        if "ksp_max_it" in p2:
            out["pmg_shell_p2_ksp_max_it"] = p2["ksp_max_it"]
    if p1:
        out["pmg_shell_p1_active_ranks"] = _resolve_active_ranks("[pmg.p1]", p1, world_size=world_size)
        out["pmg_shell_p1_rank_policy"] = _policy_name(p1)
        if "pc_type" in p1:
            out["pmg_shell_p1_pc_type"] = p1["pc_type"]
    return out


def _copy_present(source: dict[str, Any], target: dict[str, Any], mapping: dict[str, str]) -> None:
    for source_key, target_key in mapping.items():
        if source_key in source:
            target[target_key] = source[source_key]


def _copy_telescope_data(
    section: str,
    data: dict[str, Any],
    target: dict[str, Any],
    *,
    prefix: str,
    world_size: int,
) -> None:
    _reject_unknown_fields(
        section,
        data,
        {"active_ranks", "subcomm_type", "ksp_type", "ksp_rtol", "ksp_max_it", "pc_type"},
        "PMG telescope policy must resolve to concrete PETSc options before launch.",
    )
    if "active_ranks" in data:
        target[f"{prefix}_active_ranks"] = _resolve_nonnegative_rank_count(
            data["active_ranks"],
            world_size=world_size,
        )
    _copy_present(
        data,
        target,
        {
            "subcomm_type": f"{prefix}_subcomm_type",
            "ksp_type": f"{prefix}_ksp_type",
            "ksp_rtol": f"{prefix}_ksp_rtol",
            "ksp_max_it": f"{prefix}_ksp_max_it",
            "pc_type": f"{prefix}_pc_type",
        },
    )


def _resolve_active_ranks(section: str, data: dict[str, Any], *, world_size: int) -> int:
    _reject_unknown_fields(
        section,
        data,
        {"active_ranks", "active_rank_policy", "max_active_ranks", "fraction", "ksp_max_it", "pc_type"},
        "active rank policy must resolve to a concrete integer before launch.",
    )
    if "active_ranks" in data:
        value = data["active_ranks"]
        if isinstance(value, str):
            return _resolve_rank_expression(value, world_size=world_size)
        return max(1, min(world_size, int(value)))
    policy = str(data.get("active_rank_policy", "all")).strip().lower()
    cap = int(data.get("max_active_ranks", world_size))
    if policy in {"all", "world"}:
        return world_size
    if policy == "cap":
        return max(1, min(world_size, cap))
    if policy == "fraction":
        fraction = float(data.get("fraction", 1.0))
        return max(1, min(world_size, cap, int(world_size * fraction)))
    raise ValueError(f"{section} active_rank_policy {policy!r} is not supported.")


def _resolve_nonnegative_rank_count(value: object, *, world_size: int) -> int:
    if isinstance(value, str):
        text = value.strip()
        if text.isdecimal():
            return max(0, min(world_size, int(text)))
        return max(0, min(world_size, _resolve_rank_expression(text, world_size=world_size)))
    return max(0, min(world_size, int(value)))


def _resolve_rank_expression(value: str, *, world_size: int) -> int:
    text = value.strip().lower()
    if text in {"world", "all"}:
        return world_size
    if text.startswith("min(world,") and text.endswith(")"):
        rhs = text.removeprefix("min(world,").removesuffix(")").strip()
        if rhs.startswith("max(1,") and rhs.endswith(")"):
            inner = rhs.removeprefix("max(1,").removesuffix(")").strip()
            if inner == "world/2":
                return max(1, world_size // 2)
        return max(1, min(world_size, int(rhs)))
    raise ValueError(f"Unsupported active_ranks expression {value!r}.")


def _policy_name(data: dict[str, Any]) -> str:
    if "active_ranks" in data:
        return f"expression:{data['active_ranks']}" if isinstance(data["active_ranks"], str) else "fixed"
    return str(data.get("active_rank_policy", "all"))


def _petsc_data(payload: dict[str, Any]) -> dict[str, Any]:
    petsc = dict(payload.get("petsc", {}))
    _reject_unknown_fields("[petsc]", petsc, {"options_file", "extra"}, "PETSc profiles use an options file plus extra tokens.")
    tokens: list[str] = []
    if "petsc_options" in payload:
        tokens.extend(str(value) for value in payload["petsc_options"])
    tokens.extend(str(value) for value in petsc.get("extra", []))
    out: dict[str, Any] = {}
    if tokens:
        out["petsc_opt"] = tokens
    if petsc.get("options_file"):
        path = Path(str(petsc["options_file"]))
        out["pmg_options_file"] = path if path.is_absolute() else ENGINE_ROOT / path
    return out
