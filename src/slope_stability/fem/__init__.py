"""Finite-element building blocks."""

from .assembly import Assembly, assemble_strain_operator, build_elastic_stiffness_matrix, assemble_from_mesh, vector_volume
from .distributed_elastic import OwnedElasticRows, assemble_owned_elastic_rows, assemble_owned_elastic_rows_for_comm, find_overlap_partition
from .distributed_tangent import (
    OwnedTangentPattern,
    assemble_owned_regularized_matrix,
    assemble_owned_tangent_matrix,
    assemble_owned_tangent_values,
    build_global_tangent_matrix,
    prepare_owned_tangent_pattern,
)
from .quadrature import quadrature_volume_2d, quadrature_volume_3d
from .basis import local_basis_volume_2d, local_basis_volume_3d

__all__ = [
    "Assembly",
    "assemble_strain_operator",
    "build_elastic_stiffness_matrix",
    "assemble_from_mesh",
    "OwnedElasticRows",
    "assemble_owned_elastic_rows",
    "assemble_owned_elastic_rows_for_comm",
    "find_overlap_partition",
    "OwnedTangentPattern",
    "prepare_owned_tangent_pattern",
    "assemble_owned_tangent_values",
    "assemble_owned_tangent_matrix",
    "assemble_owned_regularized_matrix",
    "build_global_tangent_matrix",
    "vector_volume",
    "quadrature_volume_2d",
    "quadrature_volume_3d",
    "local_basis_volume_2d",
    "local_basis_volume_3d",
]
