"""Lightweight runtime orchestration and artifact helpers."""

from .options import (
    LinearOptions,
    PmgOptions,
    SsrOptions,
    flatten_tokens,
    quote_option_tokens,
    read_options_file,
    resolve_run_option_tokens,
    resolve_run_options_string,
)
from .results import (
    RunArtifacts,
    load_environment_manifest,
    load_resolved_run_manifest,
    load_run_summary,
    run_artifact_manifest,
    run_artifacts,
)


def build_environment_manifest(*args, **kwargs):
    """Build an environment manifest without importing config at package import time."""

    from petsc_ssr.config.manifest import build_environment_manifest as _build_environment_manifest

    return _build_environment_manifest(*args, **kwargs)

__all__ = [
    "LinearOptions",
    "PmgOptions",
    "RunArtifacts",
    "SsrOptions",
    "build_environment_manifest",
    "flatten_tokens",
    "load_environment_manifest",
    "load_resolved_run_manifest",
    "load_run_summary",
    "quote_option_tokens",
    "read_options_file",
    "resolve_run_option_tokens",
    "resolve_run_options_string",
    "run_artifact_manifest",
    "run_artifacts",
]
