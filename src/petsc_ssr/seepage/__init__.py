"""Scalar seepage problem assembly and Newton-flow solves."""

from .flow import (
    SeepageAssembly,
    assemble_auxiliary_matrices,
    heter_conduct,
    penalty_parameters_2d,
    penalty_parameters_3d,
    seepage_problem_2d,
    seepage_problem_3d,
)

__all__ = [
    "SeepageAssembly",
    "assemble_auxiliary_matrices",
    "heter_conduct",
    "penalty_parameters_2d",
    "penalty_parameters_3d",
    "seepage_problem_2d",
    "seepage_problem_3d",
]
