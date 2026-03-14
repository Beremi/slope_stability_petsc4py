"""MPI helper context utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from petsc4py import PETSc


@dataclass
class MPIContext:
    comm: PETSc.Comm = field(default_factory=lambda: PETSc.COMM_WORLD)

    @property
    def rank(self) -> int:
        return self.comm.Get_rank()

    @property
    def size(self) -> int:
        return self.comm.Get_size()

    @property
    def is_root(self) -> bool:
        return self.rank == 0
