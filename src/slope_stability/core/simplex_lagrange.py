"""Shared simplex Lagrange ordering/evaluation helpers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


TRI_EDGE_ORDER: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (0, 2))
TET_EDGE_ORDER: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (0, 2), (1, 3), (2, 3), (0, 3))
TET_FACE_ORDER: tuple[tuple[int, int, int], ...] = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


@lru_cache(maxsize=None)
def triangle_lagrange_node_tuples(order: int) -> tuple[tuple[int, int, int], ...]:
    p = int(order)
    if p < 1:
        raise ValueError("Triangle Lagrange order must be >= 1.")

    tuples: list[tuple[int, int, int]] = [
        (p, 0, 0),
        (0, p, 0),
        (0, 0, p),
    ]
    for i, j in TRI_EDGE_ORDER:
        for k in range(1, p):
            counts = [0, 0, 0]
            counts[i] = p - k
            counts[j] = k
            tuples.append(tuple(counts))
    for a in range(p - 2, 0, -1):
        for b in range(p - a - 1, 0, -1):
            c = p - a - b
            if c <= 0:
                continue
            tuples.append((a, b, c))
    return tuple(tuples)


@lru_cache(maxsize=None)
def triangle_lagrange_interior_tuples(order: int) -> tuple[tuple[int, int, int], ...]:
    p = int(order)
    if p < 3:
        return ()
    return triangle_lagrange_node_tuples(p)[3 + 3 * (p - 1) :]


@lru_cache(maxsize=None)
def tetra_lagrange_node_tuples(order: int) -> tuple[tuple[int, int, int, int], ...]:
    p = int(order)
    if p < 1:
        raise ValueError("Tetrahedral Lagrange order must be >= 1.")

    tuples: list[tuple[int, int, int, int]] = [
        (p, 0, 0, 0),
        (0, p, 0, 0),
        (0, 0, p, 0),
        (0, 0, 0, p),
    ]
    for i, j in TET_EDGE_ORDER:
        for k in range(1, p):
            counts = [0, 0, 0, 0]
            counts[i] = p - k
            counts[j] = k
            tuples.append(tuple(counts))

    face_interior = triangle_lagrange_interior_tuples(p)
    for face in TET_FACE_ORDER:
        for tri_counts in face_interior:
            counts = [0, 0, 0, 0]
            for local_idx, global_idx in enumerate(face):
                counts[global_idx] = int(tri_counts[local_idx])
            tuples.append(tuple(counts))

    for a in range(p - 3, 0, -1):
        for b in range(p - a - 2, 0, -1):
            for c in range(p - a - b - 1, 0, -1):
                d = p - a - b - c
                if d <= 0:
                    continue
                tuples.append((a, b, c, d))
    return tuple(tuples)


def tetra_reference_nodes(order: int) -> np.ndarray:
    p = float(order)
    tuples = tetra_lagrange_node_tuples(int(order))
    out = np.empty((3, len(tuples)), dtype=np.float64)
    for idx, counts in enumerate(tuples):
        out[:, idx] = np.array([counts[1], counts[2], counts[3]], dtype=np.float64) / p
    return out


def _lagrange_factor_and_derivative(order: int, degree: int, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(lam, dtype=np.float64)
    if int(degree) == 0:
        return np.ones_like(values), np.zeros_like(values)

    factors = [((float(order) * values) - float(m)) / float(m + 1) for m in range(int(degree))]
    prod = np.ones_like(values)
    for factor in factors:
        prod *= factor

    deriv = np.zeros_like(values)
    for m in range(int(degree)):
        term = np.full_like(values, float(order) / float(m + 1))
        for n, factor in enumerate(factors):
            if n == m:
                continue
            term *= factor
        deriv += term
    return prod, deriv


def evaluate_tetra_lagrange_basis(order: int, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xi_arr = np.asarray(xi, dtype=np.float64)
    if xi_arr.ndim != 2 or xi_arr.shape[0] != 3:
        raise ValueError(f"Expected tetrahedral quadrature points with shape (3, n_q), got {xi_arr.shape}.")

    xi1 = xi_arr[0, :]
    xi2 = xi_arr[1, :]
    xi3 = xi_arr[2, :]
    bary = (
        1.0 - xi1 - xi2 - xi3,
        xi1,
        xi2,
        xi3,
    )
    tuples = tetra_lagrange_node_tuples(int(order))
    n_q = xi_arr.shape[1]
    n_p = len(tuples)

    hatp = np.empty((n_p, n_q), dtype=np.float64)
    dhat1 = np.empty((n_p, n_q), dtype=np.float64)
    dhat2 = np.empty((n_p, n_q), dtype=np.float64)
    dhat3 = np.empty((n_p, n_q), dtype=np.float64)

    for idx, counts in enumerate(tuples):
        factors: list[np.ndarray] = []
        derivs: list[np.ndarray] = []
        for lam, degree in zip(bary, counts, strict=False):
            value, deriv = _lagrange_factor_and_derivative(int(order), int(degree), lam)
            factors.append(value)
            derivs.append(deriv)

        dlam: list[np.ndarray] = []
        for d_idx in range(4):
            term = derivs[d_idx]
            for f_idx in range(4):
                if f_idx == d_idx:
                    continue
                term = term * factors[f_idx]
            dlam.append(term)

        hatp[idx, :] = factors[0] * factors[1] * factors[2] * factors[3]
        dhat1[idx, :] = dlam[1] - dlam[0]
        dhat2[idx, :] = dlam[2] - dlam[0]
        dhat3[idx, :] = dlam[3] - dlam[0]

    return hatp, dhat1, dhat2, dhat3
