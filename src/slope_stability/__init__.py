"""PETSc-native reimplementation of the slope-stability workflow.

The package mirrors the MATLAB workflow from :mod:`slope_stability` while keeping
implementation blocks separated by concern:

- :mod:`slope_stability.fem` for discretization and assembly,
- :mod:`slope_stability.constitutive` for Mohr–Coulomb operators,
- :mod:`slope_stability.linear` for custom deflated FGMRES solvers,
- :mod:`slope_stability.nonlinear` for Newton + continuation schemes.
"""

from .version import __version__

from .core.config import (
    NewtonConfig,
    ContinuationConfig,
    LinearSolverConfig,
    MaterialConfig,
    Problem3DConfig,
    ExecutionConfig,
    Run3DSSRConfig,
    load_run_3d_ssr_config,
)
from .core.run_config import (
    ExportConfig,
    ProblemConfig,
    RunCaseConfig,
    SeepageConfig,
    load_run_case_config,
)
from .export import write_debug_bundle_h5, write_history_json, write_vtu
try:  # pragma: no cover - PETSc is optional in some unit-test environments
    from .mpi.context import MPIContext
except Exception:  # pragma: no cover
    MPIContext = None

__all__ = [
    "__version__",
    "NewtonConfig",
    "ContinuationConfig",
    "LinearSolverConfig",
    "MaterialConfig",
    "Problem3DConfig",
    "ExecutionConfig",
    "Run3DSSRConfig",
    "load_run_3d_ssr_config",
    "ProblemConfig",
    "SeepageConfig",
    "ExportConfig",
    "RunCaseConfig",
    "load_run_case_config",
    "write_debug_bundle_h5",
    "write_history_json",
    "write_vtu",
]

if MPIContext is not None:
    __all__.append("MPIContext")
