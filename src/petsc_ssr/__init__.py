"""Scriptable PETSc-owned slope-stability SSR engine."""

from .options import LinearOptions, PmgOptions, SsrOptions
from .problem import BoundarySpec, MaterialSpec, ProblemSpec

__all__ = [
    "BoundarySpec",
    "ContinuationCurve",
    "HydroMesh",
    "HydroResult",
    "LinearOptions",
    "MaterialSpec",
    "PmgOptions",
    "ProblemSpec",
    "CaseTranslation",
    "SsrContext",
    "SsrOptions",
    "benchmark_capability_rows",
    "load_comsol_seepage_mesh",
    "run_indirect_ssr",
    "run_limit_load_continuation",
    "solve_comsol_seepage",
    "translate_case_toml",
]


def __getattr__(name: str):
    from importlib import import_module

    modules = {
        "CaseTranslation": ".case_config",
        "ContinuationCurve": ".continuation",
        "HydroMesh": ".hydro",
        "HydroResult": ".hydro",
        "SsrContext": ".context",
        "benchmark_capability_rows": ".case_config",
        "load_comsol_seepage_mesh": ".hydro",
        "run_indirect_ssr": ".continuation",
        "run_limit_load_continuation": ".limit_load",
        "solve_comsol_seepage": ".hydro",
        "translate_case_toml": ".case_config",
    }
    if name in modules:
        module = import_module(modules[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
