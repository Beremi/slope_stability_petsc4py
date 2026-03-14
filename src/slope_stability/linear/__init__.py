"""Custom Krylov and direct solvers with PETSc-aware preconditioners.

This module exposes the high-level linear solver objects used by the rest of
the package.  The implementation intentionally mirrors MATLAB naming used by the
original code base while adding a PETSc-native execution path.
"""

from .collector import IterationCollector
from .orthogonalize import a_orthogonalize, a_orthogonalize_with_metadata
from .deflated_fgmres import FGMRESCore, dfgmres
from .elasticity import attach_rigid_body_near_nullspace, create_rigid_body_near_nullspace, impose_zero_dirichlet_full_system
from .preconditioners import GAMGPreconditioner, JacobiPreconditioner, make_near_nullspace_elasticity
from .solver import (
    DirectSolver,
    FGMRESSolver,
    PetscMatlabExactDFGMRESSolver,
    PetscKSPFGMRESSolver,
    PetscKSPGMRESDeflationSolver,
    PetscKSPMatlabDeflatedFGMRESSolver,
    PetscKSPMatlabDeflatedFGMRESReorthSolver,
    SolverFactory,
)

__all__ = [
    "IterationCollector",
    "a_orthogonalize",
    "a_orthogonalize_with_metadata",
    "FGMRESCore",
    "dfgmres",
    "attach_rigid_body_near_nullspace",
    "create_rigid_body_near_nullspace",
    "impose_zero_dirichlet_full_system",
    "GAMGPreconditioner",
    "JacobiPreconditioner",
    "make_near_nullspace_elasticity",
    "DirectSolver",
    "FGMRESSolver",
    "PetscMatlabExactDFGMRESSolver",
    "PetscKSPFGMRESSolver",
    "PetscKSPGMRESDeflationSolver",
    "PetscKSPMatlabDeflatedFGMRESSolver",
    "PetscKSPMatlabDeflatedFGMRESReorthSolver",
    "SolverFactory",
]
