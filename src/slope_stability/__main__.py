"""Module entrypoint for ``python -m slope_stability``."""

from __future__ import annotations

from . import __version__
from .mpi.context import MPIContext


def main() -> None:
    ctx = MPIContext()
    print(
        f"slope_stability v{__version__} running with "
        f"{ctx.size} MPI rank{'s' if ctx.size != 1 else ''}"
    )


if __name__ == "__main__":
    main()

