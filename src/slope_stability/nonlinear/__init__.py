"""Nonlinear solvers: Newton and continuation schemes."""

from .damping import damping, damping_alg5
from .newton import newton, newton_ind_ssr, newton_ind_ll

__all__ = [
    "damping",
    "damping_alg5",
    "newton",
    "newton_ind_ssr",
    "newton_ind_ll",
]
