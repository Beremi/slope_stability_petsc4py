# cython: language_level=3
"""Thin Cython bridge to the pure C DMPlex indirect SSR hot path."""

ctypedef int PetscErrorCode

cdef extern from *:
    """
    #include <petscsys.h>
    PetscErrorCode P4IndirectSSRRunOptionsString(const char options[]);
    """
    PetscErrorCode P4IndirectSSRRunOptionsString(const char options[])


def run_options(str options):
    """Run the C DMPlex SSR solver with PETSc options already serialized.

    The caller is responsible for ensuring PETSc has been initialized by
    petsc4py and for passing a complete option string.  The C side writes the
    usual parseable solver diagnostics plus any configured CSV/JSON artifacts.
    """

    cdef bytes encoded = options.encode("utf-8")
    cdef PetscErrorCode ierr = P4IndirectSSRRunOptionsString(encoded)
    if ierr != 0:
        raise RuntimeError(f"P4IndirectSSRRunOptionsString failed with PETSc error code {ierr}")
    return None
