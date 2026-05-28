"""Finite-element helpers needed by the standalone benchmark notebooks.

The PETSc SSR engine owns the mechanics solve in C. This compatibility package
only keeps lightweight mesh/post-processing utilities from the earlier Python implementation.
"""

from .assembly import Assembly, assemble_strain_operator, build_elastic_stiffness_matrix, assemble_from_mesh, vector_volume
from .quadrature import available_tetra_quadrature_rules, quadrature_volume_2d, quadrature_volume_3d
from .basis import local_basis_volume_2d, local_basis_volume_3d

__all__ = [
    "Assembly",
    "assemble_strain_operator",
    "build_elastic_stiffness_matrix",
    "assemble_from_mesh",
    "vector_volume",
    "available_tetra_quadrature_rules",
    "quadrature_volume_2d",
    "quadrature_volume_3d",
    "local_basis_volume_2d",
    "local_basis_volume_3d",
]
