"""Case schemas, profile resolution, and configuration helpers."""

from .case_schema import RunCaseConfig, load_run_case_config
from .manifest import (
    build_environment_manifest,
    build_resolved_config,
    build_resolved_run_manifest,
    dumps_resolved_config_toml,
)
from .profiles import (
    ResolvedControlProfile,
    ResolvedPcVariant,
    ResolvedSolverProfile,
    load_continuation_profile,
    load_newton_profile,
    load_seepage_profile,
    load_solver_profile,
    native_linear_algorithm_selector,
    pc_variant_from_backend,
)
from .resolver import (
    ResolvedCaseModel,
    explain_case_payload,
    resolve_case_model,
    resolved_pc_policy,
    validate_case_payload,
)
from .validators import (
    normalize_output_preset,
    reject_profile_default_repeats,
    reject_unknown_fields,
    validate_case_metadata,
)

__all__ = [
    "ResolvedControlProfile",
    "ResolvedCaseModel",
    "ResolvedPcVariant",
    "ResolvedSolverProfile",
    "RunCaseConfig",
    "build_environment_manifest",
    "build_resolved_config",
    "build_resolved_run_manifest",
    "dumps_resolved_config_toml",
    "explain_case_payload",
    "load_continuation_profile",
    "load_newton_profile",
    "load_run_case_config",
    "load_seepage_profile",
    "load_solver_profile",
    "native_linear_algorithm_selector",
    "normalize_output_preset",
    "pc_variant_from_backend",
    "reject_profile_default_repeats",
    "reject_unknown_fields",
    "resolve_case_model",
    "resolved_pc_policy",
    "validate_case_payload",
    "validate_case_metadata",
]
