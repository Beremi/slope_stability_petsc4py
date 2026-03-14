"""Material reduction helpers.

These utilities mirror :mod:`slope_stability.CONSTITUTIVE_PROBLEM.reduction`.

The function returns reduced cohesion ``c_bar`` and ``sin(phi)`` used by the
constitutive operators.
"""

from __future__ import annotations

import numpy as np


def reduction(c0: np.ndarray, phi: np.ndarray, psi: np.ndarray, lam: float, Davis_type: str = "A"):
    """Return reduced Mohr-Coulomb parameters.

    Parameters
    ----------
    c0:
        Effective cohesion per integration point.
    phi:
        Friction angle in radians.
    psi:
        Dilatancy angle in radians.
    lam:
        Reduction factor ``lambda``.
    Davis_type:
        One of ``"A"``, ``"B"`` or ``"C"``.
    """

    c0_a = np.asarray(c0, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)

    if c0_a.shape != phi.shape or c0_a.shape != psi.shape:
        raise ValueError("c0, phi and psi must have the same shape")

    if lam <= 0:
        raise ValueError("Reduction parameter lambda must be positive")

    c_bar = np.empty_like(c0_a)
    sin_phi = np.empty_like(c0_a)
    typ = str(Davis_type).upper()

    if typ == "A":
        beta = np.cos(phi) * np.cos(psi) / (1.0 - np.sin(phi) * np.sin(psi))
        c0_lambda = beta * c0_a / lam
        phi_lambda = np.arctan(beta * np.tan(phi) / lam)
        c_bar = 2.0 * c0_lambda * np.cos(phi_lambda)
        sin_phi = np.sin(phi_lambda)
        return c_bar, sin_phi

    if typ == "B":
        c01 = c0_a / lam
        phi1 = np.arctan(np.tan(phi) / lam)
        psi1 = np.arctan(np.tan(psi) / lam)
        beta = np.cos(phi1) * np.cos(psi1) / (1.0 - np.sin(phi1) * np.sin(psi1))
        c0_lambda = beta * c01
        phi_lambda = np.arctan(beta * np.tan(phi1))
        c_bar = 2.0 * c0_lambda * np.cos(phi_lambda)
        sin_phi = np.sin(phi_lambda)
        return c_bar, sin_phi

    if typ == "C":
        c01 = c0_a / lam
        phi1 = np.arctan(np.tan(phi) / lam)
        beta = np.where(
            phi1 > psi,
            np.cos(phi1) * np.cos(psi) / (1.0 - np.sin(phi1) * np.sin(psi)),
            1.0,
        )
        c0_lambda = beta * c01
        phi_lambda = np.arctan(beta * np.tan(phi1))
        c_bar = 2.0 * c0_lambda * np.cos(phi_lambda)
        sin_phi = np.sin(phi_lambda)
        return c_bar, sin_phi

    raise ValueError("Incorrect choice of the Davis approach")


def reduction_parameters(c0: np.ndarray, phi: np.ndarray, psi: np.ndarray, lam: float, Davis_type: str = "A"):
    """Backward-compatible wrapper.

    The historical code exposes both ``reduction`` and ``reduction_parameters``.
    """

    return reduction(c0, phi, psi, lam, Davis_type)
