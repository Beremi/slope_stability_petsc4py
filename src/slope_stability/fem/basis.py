"""Local basis functions and derivatives for standard elements."""

from __future__ import annotations
import numpy as np

from ..core.simplex_lagrange import evaluate_tetra_lagrange_basis


def local_basis_volume_2d(elem_type: str, xi: np.ndarray):
    elem_type = elem_type.upper()
    xi1 = xi[0, :]
    xi2 = xi[1, :]
    n_q = xi.shape[1]

    if elem_type == "P1":
        hatp = np.array([1 - xi1 - xi2, xi1, xi2], dtype=np.float64)
        dhat1 = np.array([-1.0, 1.0, 0.0], dtype=np.float64)[:, None]
        dhat2 = np.array([-1.0, 0.0, 1.0], dtype=np.float64)[:, None]
        return hatp, dhat1, dhat2

    if elem_type == "P2":
        xi0 = 1 - xi1 - xi2
        hatp = np.array(
            [
                xi0 * (2 * xi0 - 1),
                xi1 * (2 * xi1 - 1),
                xi2 * (2 * xi2 - 1),
                4 * xi1 * xi2,
                4 * xi0 * xi2,
                4 * xi0 * xi1,
            ],
            dtype=np.float64,
        )
        dhat1 = np.array(
            [
                -4 * xi0 + 1,
                4 * xi1 - 1,
                np.zeros(n_q),
                4 * xi2,
                -4 * xi2,
                4 * (xi0 - xi1),
            ],
            dtype=np.float64,
        )
        dhat2 = np.array(
            [
                -4 * xi0 + 1,
                np.zeros(n_q),
                4 * xi2 - 1,
                4 * xi1,
                4 * (xi0 - xi2),
                -4 * xi1,
            ],
            dtype=np.float64,
        )
        return hatp, dhat1, dhat2

    if elem_type == "P4":
        xi0 = 1 - xi1 - xi2
        # Reference implementation keeps the full 15-basis P4 form.
        hatp = np.array(
            [
                xi0 * (4 * xi0 - 1) * (4 * xi0 - 2) * (4 * xi0 - 3) / 6,
                xi1 * (4 * xi1 - 1) * (4 * xi1 - 2) * (4 * xi1 - 3) / 6,
                xi2 * (4 * xi2 - 1) * (4 * xi2 - 2) * (4 * xi2 - 3) / 6,
                4 * xi0 * xi1 * (4 * xi0 - 1) * (4 * xi1 - 1),
                4 * xi1 * xi2 * (4 * xi1 - 1) * (4 * xi2 - 1),
                4 * xi0 * xi2 * (4 * xi0 - 1) * (4 * xi2 - 1),
                8 * xi0 * xi1 * (4 * xi0 - 1) * (4 * xi0 - 2) / 3,
                8 * xi0 * xi1 * (4 * xi1 - 1) * (4 * xi1 - 2) / 3,
                8 * xi1 * xi2 * (4 * xi1 - 1) * (4 * xi1 - 2) / 3,
                8 * xi1 * xi2 * (4 * xi2 - 1) * (4 * xi2 - 2) / 3,
                8 * xi0 * xi2 * (4 * xi2 - 1) * (4 * xi2 - 2) / 3,
                8 * xi0 * xi2 * (4 * xi0 - 1) * (4 * xi0 - 2) / 3,
                32 * xi0 * xi1 * xi2 * (4 * xi0 - 1),
                32 * xi0 * xi1 * xi2 * (4 * xi1 - 1),
                32 * xi0 * xi1 * xi2 * (4 * xi2 - 1),
            ],
            dtype=np.float64,
        )
        # Keep derivatives explicit to preserve equivalence to MATLAB formulas.
        dhat1 = np.array(
            [
                -((4 * xi0 - 1) * (4 * xi0 - 2) * (4 * xi0 - 3) + 4 * xi0 * (4 * xi0 - 2) * (4 * xi0 - 3) + 4 * xi0 * (4 * xi0 - 1) * (4 * xi0 - 3) + 4 * xi0 * (4 * xi0 - 1) * (4 * xi0 - 2)) / 6,
                ((4 * xi1 - 1) * (4 * xi1 - 2) * (4 * xi1 - 3) + 4 * xi1 * (4 * xi1 - 2) * (4 * xi1 - 3) + 4 * xi1 * (4 * xi1 - 1) * (4 * xi1 - 3) + 4 * xi1 * (4 * xi1 - 1) * (4 * xi1 - 2)) / 6,
                np.zeros(n_q),
                4 * (-xi1 * (4 * xi0 - 1) * (4 * xi1 - 1) + xi0 * (4 * xi0 - 1) * (4 * xi1 - 1) - 4 * xi0 * xi1 * (4 * xi1 - 1) + 4 * xi0 * xi1 * (4 * xi0 - 1)),
                4 * (xi2 * (4 * xi1 - 1) * (4 * xi2 - 1) + 4 * xi1 * xi2 * (4 * xi2 - 1)),
                4 * (-xi2 * (4 * xi0 - 1) * (4 * xi2 - 1) - 4 * xi0 * xi2 * (4 * xi2 - 1)),
                8 * (-xi1 * (4 * xi0 - 1) * (4 * xi0 - 2) + xi0 * (4 * xi0 - 1) * (4 * xi0 - 2) - 4 * xi0 * xi1 * (4 * xi0 - 2) - 4 * xi0 * xi1 * (4 * xi0 - 1)) / 3,
                8 * (-xi1 * (4 * xi1 - 1) * (4 * xi1 - 2) + xi0 * (4 * xi1 - 1) * (4 * xi1 - 2) + 4 * xi0 * xi1 * (4 * xi1 - 2) + 4 * xi0 * xi1 * (4 * xi1 - 1)) / 3,
                8 * (xi2 * (4 * xi1 - 1) * (4 * xi1 - 2) + 4 * xi1 * xi2 * (4 * xi1 - 2) + 4 * xi1 * xi2 * (4 * xi1 - 1)) / 3,
                8 * xi2 * (4 * xi2 - 1) * (4 * xi2 - 2) / 3,
                -8 * xi2 * (4 * xi2 - 1) * (4 * xi2 - 2) / 3,
                8 * (-xi2 * (4 * xi0 - 1) * (4 * xi0 - 2) - 4 * xi0 * xi2 * (4 * xi0 - 2) - 4 * xi0 * xi2 * (4 * xi0 - 1)) / 3,
                32 * (-xi1 * xi2 * (4 * xi0 - 1) + xi0 * xi2 * (4 * xi0 - 1) - 4 * xi0 * xi1 * xi2),
                32 * (-xi1 * xi2 * (4 * xi1 - 1) + xi0 * xi2 * (4 * xi1 - 1) + 4 * xi0 * xi1 * xi2),
                32 * (-xi1 * xi2 * (4 * xi2 - 1) + xi0 * xi2 * (4 * xi2 - 1)),
            ],
            dtype=np.float64,
        )
        dhat2 = np.array(
            [
                -((4 * xi0 - 1) * (4 * xi0 - 2) * (4 * xi0 - 3) + 4 * xi0 * (4 * xi0 - 2) * (4 * xi0 - 3) + 4 * xi0 * (4 * xi0 - 1) * (4 * xi0 - 3) + 4 * xi0 * (4 * xi0 - 1) * (4 * xi0 - 2)) / 6,
                np.zeros(n_q),
                ((4 * xi2 - 1) * (4 * xi2 - 2) * (4 * xi2 - 3) + 4 * xi2 * (4 * xi2 - 2) * (4 * xi2 - 3) + 4 * xi2 * (4 * xi2 - 1) * (4 * xi2 - 3) + 4 * xi2 * (4 * xi2 - 1) * (4 * xi2 - 2)) / 6,
                4 * (-xi1 * (4 * xi0 - 1) * (4 * xi1 - 1) - 4 * xi0 * xi1 * (4 * xi1 - 1)),
                4 * (xi1 * (4 * xi1 - 1) * (4 * xi2 - 1) + 4 * xi1 * xi2 * (4 * xi1 - 1)),
                4 * (-xi2 * (4 * xi0 - 1) * (4 * xi2 - 1) + xi0 * (4 * xi0 - 1) * (4 * xi2 - 1) - 4 * xi0 * xi2 * (4 * xi2 - 1) + 4 * xi0 * xi2 * (4 * xi0 - 1)),
                8 * (-xi1 * (4 * xi0 - 1) * (4 * xi0 - 2) - 4 * xi0 * xi1 * (4 * xi0 - 2) - 4 * xi0 * xi1 * (4 * xi0 - 1)) / 3,
                -8 * xi1 * (4 * xi1 - 1) * (4 * xi1 - 2) / 3,
                8 * (xi1 * (4 * xi1 - 1) * (4 * xi1 - 2) + 4 * xi1 * xi2 * (4 * xi1 - 1)) / 3,
                8 * (xi1 * (4 * xi1 - 1) * (4 * xi2 - 1) + 4 * xi1 * xi2 * (4 * xi2 - 1)) / 3,
                -8 * xi2 * (4 * xi2 - 1) * (4 * xi2 - 2) / 3,
                8 * (xi2 * (4 * xi2 - 1) * (4 * xi2 - 2) + 4 * xi0 * xi2 * (4 * xi2 - 2) + 4 * xi0 * xi2 * (4 * xi2 - 1)) / 3,
                32 * (-xi1 * xi2 * (4 * xi0 - 1) + xi0 * xi1 * (4 * xi0 - 1) - 4 * xi0 * xi1 * xi2),
                32 * (-xi1 * xi2 * (4 * xi1 - 1) + xi0 * xi1 * (4 * xi1 - 1)),
                32 * (-xi1 * xi2 * (4 * xi2 - 1) + xi0 * xi1 * (4 * xi2 - 1) + 4 * xi0 * xi1 * xi2),
            ],
            dtype=np.float64,
        )
        return hatp, dhat1, dhat2

    raise ValueError(f"Unsupported elem_type={elem_type}")


def local_basis_volume_3d(elem_type: str, xi: np.ndarray):
    elem_type = elem_type.upper()
    xi1 = xi[0, :]
    xi2 = xi[1, :]
    xi3 = xi[2, :]
    n_q = xi.shape[1]

    if elem_type == "P1":
        hatp = np.array([1 - xi1 - xi2 - xi3, xi1, xi2, xi3], dtype=np.float64)
        dhat1 = np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float64)[:, None]
        dhat2 = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float64)[:, None]
        dhat3 = np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float64)[:, None]
        return hatp, dhat1, dhat2, dhat3

    if elem_type == "P2":
        xi0 = 1 - xi1 - xi2 - xi3
        hatp = np.array(
            [
                xi0 * (2 * xi0 - 1),
                xi1 * (2 * xi1 - 1),
                xi2 * (2 * xi2 - 1),
                xi3 * (2 * xi3 - 1),
                4 * xi0 * xi1,
                4 * xi1 * xi2,
                4 * xi0 * xi2,
                4 * xi1 * xi3,
                4 * xi2 * xi3,
                4 * xi0 * xi3,
            ],
            dtype=np.float64,
        )
        dhat1 = np.array(
            [
                -4 * xi0 + 1,
                4 * xi1 - 1,
                np.zeros(n_q),
                np.zeros(n_q),
                4 * (xi0 - xi1),
                4 * xi2,
                -4 * xi2,
                4 * xi3,
                np.zeros(n_q),
                -4 * xi3,
            ],
            dtype=np.float64,
        )
        dhat2 = np.array(
            [
                -4 * xi0 + 1,
                np.zeros(n_q),
                4 * xi2 - 1,
                np.zeros(n_q),
                -4 * xi1,
                4 * xi1,
                4 * (xi0 - xi2),
                np.zeros(n_q),
                4 * xi3,
                -4 * xi3,
            ],
            dtype=np.float64,
        )
        dhat3 = np.array(
            [
                -4 * xi0 + 1,
                np.zeros(n_q),
                np.zeros(n_q),
                4 * xi3 - 1,
                -4 * xi1,
                np.zeros(n_q),
                -4 * xi2,
                4 * xi1,
                4 * xi2,
                4 * (xi0 - xi3),
            ],
            dtype=np.float64,
        )
        return hatp, dhat1, dhat2, dhat3

    if elem_type == "P4":
        return evaluate_tetra_lagrange_basis(4, xi)

    if elem_type == "Q1":
        x = xi
        hatp = np.array(
            [
                (1 - x[0, :]) * (1 - x[1, :]) * (1 - x[2, :]) / 8,
                (1 + x[0, :]) * (1 - x[1, :]) * (1 - x[2, :]) / 8,
                (1 + x[0, :]) * (1 + x[1, :]) * (1 - x[2, :]) / 8,
                (1 - x[0, :]) * (1 + x[1, :]) * (1 - x[2, :]) / 8,
                (1 - x[0, :]) * (1 - x[1, :]) * (1 + x[2, :]) / 8,
                (1 + x[0, :]) * (1 - x[1, :]) * (1 + x[2, :]) / 8,
                (1 + x[0, :]) * (1 + x[1, :]) * (1 + x[2, :]) / 8,
                (1 - x[0, :]) * (1 + x[1, :]) * (1 + x[2, :]) / 8,
            ],
            dtype=np.float64,
        )
        dh1 = np.array(
            [
                -((1 - x[1, :]) * (1 - x[2, :]) / 8),
                ((1 - x[1, :]) * (1 - x[2, :]) / 8),
                ((1 + x[1, :]) * (1 - x[2, :]) / 8),
                -((1 + x[1, :]) * (1 - x[2, :]) / 8),
                -((1 - x[1, :]) * (1 + x[2, :]) / 8),
                ((1 - x[1, :]) * (1 + x[2, :]) / 8),
                ((1 + x[1, :]) * (1 + x[2, :]) / 8),
                -((1 + x[1, :]) * (1 + x[2, :]) / 8),
            ],
            dtype=np.float64,
        )
        dh2 = np.array(
            [
                -((1 - x[0, :]) * (1 - x[2, :]) / 8),
                -((1 + x[0, :]) * (1 - x[2, :]) / 8),
                ((1 + x[0, :]) * (1 - x[2, :]) / 8),
                ((1 - x[0, :]) * (1 - x[2, :]) / 8),
                -((1 - x[0, :]) * (1 + x[2, :]) / 8),
                -((1 + x[0, :]) * (1 + x[2, :]) / 8),
                ((1 + x[0, :]) * (1 + x[2, :]) / 8),
                ((1 - x[0, :]) * (1 + x[2, :]) / 8),
            ],
            dtype=np.float64,
        )
        dh3 = np.array(
            [
                -((1 - x[0, :]) * (1 - x[1, :]) / 8),
                -((1 + x[0, :]) * (1 - x[1, :]) / 8),
                -((1 + x[0, :]) * (1 + x[1, :]) / 8),
                -((1 - x[0, :]) * (1 + x[1, :]) / 8),
                ((1 - x[0, :]) * (1 - x[1, :]) / 8),
                ((1 + x[0, :]) * (1 - x[1, :]) / 8),
                ((1 + x[0, :]) * (1 + x[1, :]) / 8),
                ((1 - x[0, :]) * (1 + x[1, :]) / 8),
            ],
            dtype=np.float64,
        )
        return hatp, dh1, dh2, dh3

    raise ValueError(f"Unsupported elem_type={elem_type}")
