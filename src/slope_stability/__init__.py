"""Slope-stability package."""

from .version import __version__

_LAZY_EXPORTS = {
    "NewtonConfig": "slope_stability.core.run_config",
    "ContinuationConfig": "slope_stability.core.run_config",
    "LinearSolverConfig": "slope_stability.core.run_config",
    "ExecutionConfig": "slope_stability.core.run_config",
    "ProblemConfig": "slope_stability.core.run_config",
    "SeepageConfig": "slope_stability.core.run_config",
    "ExportConfig": "slope_stability.core.run_config",
    "RunCaseConfig": "slope_stability.core.run_config",
    "load_run_case_config": "slope_stability.core.run_config",
    "write_debug_bundle_h5": "slope_stability.export",
    "write_history_json": "slope_stability.export",
    "write_vtu": "slope_stability.export",
    "MPIContext": "slope_stability.mpi.context",
}

__all__ = ["__version__", *_LAZY_EXPORTS]


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
