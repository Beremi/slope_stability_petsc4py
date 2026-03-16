"""Constitutive relations and constitutive operators."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.sparse import block_diag, csc_matrix, csr_matrix, coo_matrix

from .reduction import reduction
from ..fem.distributed_tangent import (
    DEFAULT_TANGENT_KERNEL,
    OwnedTangentPattern,
    assemble_overlap_strain,
    assemble_owned_force_from_local_stress,
    assemble_owned_tangent_values,
)
from ..utils import (
    IterationHistory,
    local_csr_to_petsc_aij_matrix,
    q_to_free_indices,
    to_numpy_vector,
    update_petsc_aij_matrix_csr,
    release_petsc_aij_matrix,
)

try:  # pragma: no cover - PETSc is optional in some tests
    from petsc4py import PETSc
except Exception:  # pragma: no cover
    PETSc = None
try:  # pragma: no cover - mpi4py is optional in some tests
    from mpi4py import MPI as PYMPI
except Exception:  # pragma: no cover
    PYMPI = None

try:  # pragma: no cover - compiled extension is optional during tests
    from slope_stability import _kernels
except Exception:  # pragma: no cover
    _kernels = None


_OWNED_UNIQUE_EXCHANGE_S_TAG = 42117
_OWNED_UNIQUE_EXCHANGE_DS_TAG = 42118



def _eigh_2d(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = values[0]
    b = values[1]
    c = values[2]
    tr = a + b
    d = a - b
    s = np.sqrt(np.maximum(0.0, d * d + 4.0 * c * c))

    eig0_1 = 0.5 * (tr + s)
    eig0_2 = 0.5 * (tr - s)
    eig0_3 = values[3]

    return eig0_1, eig0_2, eig0_3, d, s


def _sorted_eigs_2d(eig0_1, eig0_2, eig0_3):
    eig_1 = eig0_1.copy()
    eig_2 = eig0_2.copy()
    eig_3 = eig0_3.copy()

    test2 = (eig0_1 >= eig0_3) & (eig0_3 > eig0_2)
    test3 = eig0_3 > eig0_1

    eig_2[test2] = eig0_3[test2]
    eig_3[test2] = eig0_2[test2]

    eig_1[test3] = eig0_3[test3]
    eig_2[test3] = eig0_1[test3]
    eig_3[test3] = eig0_2[test3]
    return eig_1, eig_2, eig_3


def _eigen_projection_2d(E_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # MATLAB-compatible eigenprojections in strain representation [e11,e22,e12,e33].
    n_int = E_tr.shape[1]
    e11, e22, e12, e33 = E_tr

    I1 = e11 + e22
    I2 = np.sqrt(np.maximum(0.0, (e11 - e22) ** 2 + e12 * e12))
    eig0_1 = (I1 + I2) / 2.0
    eig0_2 = (I1 - I2) / 2.0
    eig0_3 = e33

    test1 = I2 == 0

    Eig0_1 = np.zeros((4, n_int))
    active = ~test1
    if np.any(active):
        Eig0_1[0, active] = (e11[active] - eig0_2[active]) / I2[active]
        Eig0_1[1, active] = (e22[active] - eig0_2[active]) / I2[active]
        Eig0_1[2, active] = e12[active] / (2.0 * I2[active])

    if np.any(test1):
        Eig0_1[0:2, test1] = 1.0

    Eig0_2 = np.vstack((np.ones((2, n_int)), np.zeros((2, n_int)))) - Eig0_1
    Eig0_3 = np.vstack((np.zeros((3, n_int)), np.ones((1, n_int))))

    eig_1, eig_2, eig_3 = _sorted_eigs_2d(eig0_1, eig0_2, eig0_3)

    Eig_1 = Eig0_1.copy()
    Eig_2 = Eig0_2.copy()
    Eig_3 = Eig0_3.copy()

    test2 = (eig0_1 >= eig0_3) & (eig0_3 > eig0_2)
    Eig_2[:, test2] = Eig0_3[:, test2]
    Eig_3[:, test2] = Eig0_2[:, test2]

    test3 = eig0_3 > eig0_1
    Eig_1[:, test3] = Eig0_3[:, test3]
    Eig_2[:, test3] = Eig0_1[:, test3]
    Eig_3[:, test3] = Eig0_2[:, test3]

    return eig_1, eig_2, eig_3, Eig_1, Eig_2, Eig_3


def _vec_col_major(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64).reshape(-1, 1, order="F")


def _outer_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Column-wise outer products require matching shapes")
    return (a[:, None, :] * b[None, :, :]).reshape(a.shape[0] * b.shape[0], a.shape[1], order="F")


def constitutive_problem_2D(E: np.ndarray, c_bar: np.ndarray, sin_phi: np.ndarray, shear: np.ndarray, bulk: np.ndarray, lame: np.ndarray, return_tangent: bool = False):
    E = np.asarray(E, dtype=np.float64)
    if E.shape[0] != 3:
        raise ValueError("E must have 3 rows for 2D")

    c_bar = np.asarray(c_bar, dtype=np.float64).ravel()
    sin_phi = np.asarray(sin_phi, dtype=np.float64).ravel()
    shear = np.asarray(shear, dtype=np.float64).ravel()
    bulk = np.asarray(bulk, dtype=np.float64).ravel()
    lame = np.asarray(lame, dtype=np.float64).ravel()

    n_int = E.shape[1]
    if not (c_bar.size == shear.size == sin_phi.size == bulk.size == lame.size == n_int):
        raise ValueError("Material arrays must be of size n_int")

    ident4 = np.diag([1.0, 1.0, 0.5, 1.0])
    iota4 = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float64)
    vol4 = np.outer(iota4, iota4)
    vol3 = vol4[:3, :3]
    ident3 = np.diag([1.0, 1.0, 0.5])
    elast = vol3.reshape(-1, 1, order="F") * lame[None, :] + 2.0 * ident3.reshape(-1, 1, order="F") * shear[None, :]

    E_tr = np.vstack((E, np.zeros((1, n_int), dtype=np.float64)))
    I1 = E_tr[0, :] + E_tr[1, :]
    I2 = np.sqrt((E_tr[0, :] - E_tr[1, :]) ** 2 + E_tr[2, :] ** 2)
    eig0_1 = 0.5 * (I1 + I2)
    eig0_2 = 0.5 * (I1 - I2)
    eig0_3 = E_tr[3, :]

    Eig0_1 = np.zeros((4, n_int), dtype=np.float64)
    test1 = I2 == 0.0
    active = ~test1
    if np.any(active):
        Eig0_1[0:3, active] = np.vstack(
            (
                E_tr[0, active] - eig0_2[active],
                E_tr[1, active] - eig0_2[active],
                E_tr[2, active] / 2.0,
            )
        ) / I2[active]
    if np.any(test1):
        Eig0_1[0:2, test1] = 1.0
    Eig0_2 = np.vstack((np.ones((2, n_int), dtype=np.float64), np.zeros((2, n_int), dtype=np.float64))) - Eig0_1
    Eig0_3 = np.vstack((np.zeros((3, n_int), dtype=np.float64), np.ones((1, n_int), dtype=np.float64)))

    EIG0_1 = np.zeros((9, n_int), dtype=np.float64)
    EIG0_2 = np.zeros((9, n_int), dtype=np.float64)
    EIG0_3 = np.zeros((9, n_int), dtype=np.float64)
    if np.any(active):
        a1 = Eig0_1[0, active]
        a2 = Eig0_1[1, active]
        a3 = Eig0_1[2, active]
        b1 = Eig0_2[0, active]
        b2 = Eig0_2[1, active]
        b3 = Eig0_2[2, active]
        denom = I2[active]
        EIG0_1[:, active] = np.vstack(
            (
                1.0 - a1 * a1 - b1 * b1,
                -(a2 * a1 + b2 * b1),
                -(a3 * a1 + b3 * b1),
                -(a1 * a2 + b1 * b2),
                1.0 - a2 * a2 - b2 * b2,
                -(a3 * a2 + b3 * b2),
                -(a1 * a3 + b1 * b3),
                -(a2 * a3 + b2 * b3),
                0.5 - a3 * a3 - b3 * b3,
            )
        ) / denom
        EIG0_2[:, active] = -EIG0_1[:, active]

    eig_1 = eig0_1.copy()
    eig_2 = eig0_2.copy()
    eig_3 = eig0_3.copy()
    Eig_1 = Eig0_1.copy()
    Eig_2 = Eig0_2.copy()
    Eig_3 = Eig0_3.copy()
    EIG_1 = EIG0_1.copy()
    EIG_2 = EIG0_2.copy()
    EIG_3 = EIG0_3.copy()

    test2 = (eig0_1 >= eig0_3) & (eig0_3 > eig0_2)
    eig_2[test2] = eig0_3[test2]
    eig_3[test2] = eig0_2[test2]
    Eig_2[:, test2] = Eig0_3[:, test2]
    Eig_3[:, test2] = Eig0_2[:, test2]
    EIG_2[:, test2] = EIG0_3[:, test2]
    EIG_3[:, test2] = EIG0_2[:, test2]

    test3 = eig0_3 > eig0_1
    eig_1[test3] = eig0_3[test3]
    eig_2[test3] = eig0_1[test3]
    eig_3[test3] = eig0_2[test3]
    Eig_1[:, test3] = Eig0_3[:, test3]
    Eig_2[:, test3] = Eig0_1[:, test3]
    Eig_3[:, test3] = Eig0_2[:, test3]
    EIG_1[:, test3] = EIG0_3[:, test3]
    EIG_2[:, test3] = EIG0_1[:, test3]
    EIG_3[:, test3] = EIG0_2[:, test3]

    trace_E = eig_1 + eig_2 + eig_3
    f_tr = 2.0 * shear * ((1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * eig_3) + 2.0 * lame * sin_phi * trace_E - c_bar
    gamma_sl = (eig_1 - eig_2) / (1.0 + sin_phi)
    gamma_sr = (eig_2 - eig_3) / (1.0 - sin_phi)
    gamma_la = (eig_1 + eig_2 - 2.0 * eig_3) / (3.0 - sin_phi)
    gamma_ra = (2.0 * eig_1 - eig_2 - eig_3) / (3.0 + sin_phi)

    denom_s = 4.0 * lame * sin_phi**2 + 2.0 * shear * (1.0 + sin_phi) ** 2 + 2.0 * shear * (1.0 - sin_phi) ** 2
    denom_l = 4.0 * lame * sin_phi**2 + shear * (1.0 + sin_phi) ** 2 + 2.0 * shear * (1.0 - sin_phi) ** 2
    denom_r = 4.0 * lame * sin_phi**2 + 2.0 * shear * (1.0 + sin_phi) ** 2 + shear * (1.0 - sin_phi) ** 2
    denom_a = 4.0 * bulk * sin_phi**2

    lambda_s = f_tr / denom_s
    lambda_l = (shear * ((1.0 + sin_phi) * (eig_1 + eig_2) - 2.0 * (1.0 - sin_phi) * eig_3) + 2.0 * lame * sin_phi * trace_E - c_bar) / denom_l
    lambda_r = (shear * (2.0 * (1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * (eig_2 + eig_3)) + 2.0 * lame * sin_phi * trace_E - c_bar) / denom_r
    lambda_a = (2.0 * bulk * sin_phi * trace_E - c_bar) / denom_a

    S = np.zeros((4, n_int), dtype=np.float64)
    DS = np.zeros((9, n_int), dtype=np.float64) if return_tangent else None

    test_el = f_tr <= 0.0
    if np.any(test_el):
        lame_el = lame[test_el]
        shear_el = shear[test_el]
        S[:, test_el] = lame_el[None, :] * (vol4 @ E_tr[:, test_el]) + 2.0 * shear_el[None, :] * (ident4 @ E_tr[:, test_el])
        if return_tangent and DS is not None:
            DS[:, test_el] = elast[:, test_el]

    test_s = (lambda_s <= np.minimum(gamma_sl, gamma_sr)) & (~test_el)
    if np.any(test_s):
        lambda_s_t = lambda_s[test_s]
        lame_s = lame[test_s]
        shear_s = shear[test_s]
        sin_phi_s = sin_phi[test_s]
        denom_s_t = denom_s[test_s]
        Eig_1_s = Eig_1[:, test_s]
        Eig_2_s = Eig_2[:, test_s]
        Eig_3_s = Eig_3[:, test_s]
        trace_E_s = trace_E[test_s]
        sigma_1_s = lame_s * trace_E_s + 2.0 * shear_s * eig_1[test_s] - lambda_s_t * (2.0 * lame_s * sin_phi_s + 2.0 * shear_s * (1.0 + sin_phi_s))
        sigma_2_s = lame_s * trace_E_s + 2.0 * shear_s * eig_2[test_s] - lambda_s_t * (2.0 * lame_s * sin_phi_s)
        sigma_3_s = lame_s * trace_E_s + 2.0 * shear_s * eig_3[test_s] - lambda_s_t * (2.0 * lame_s * sin_phi_s - 2.0 * shear_s * (1.0 - sin_phi_s))
        S[:, test_s] = sigma_1_s[None, :] * Eig_1_s + sigma_2_s[None, :] * Eig_2_s + sigma_3_s[None, :] * Eig_3_s
        if return_tangent and DS is not None:
            mat1_s = sigma_1_s[None, :] * EIG_1[:, test_s] + sigma_2_s[None, :] * EIG_2[:, test_s] + sigma_3_s[None, :] * EIG_3[:, test_s]
            mat2_s = vol3.reshape(-1, 1, order="F") * lame_s[None, :]
            mat3_s = 2.0 * np.vstack(
                (
                    shear_s * Eig_1_s[0, :] * Eig_1_s[0, :],
                    shear_s * Eig_1_s[1, :] * Eig_1_s[0, :],
                    shear_s * Eig_1_s[2, :] * Eig_1_s[0, :],
                    shear_s * Eig_1_s[0, :] * Eig_1_s[1, :],
                    shear_s * Eig_1_s[1, :] * Eig_1_s[1, :],
                    shear_s * Eig_1_s[2, :] * Eig_1_s[1, :],
                    shear_s * Eig_1_s[0, :] * Eig_1_s[2, :],
                    shear_s * Eig_1_s[1, :] * Eig_1_s[2, :],
                    shear_s * Eig_1_s[2, :] * Eig_1_s[2, :],
                )
            )
            mat4_s = 2.0 * np.vstack(
                (
                    shear_s * Eig_2_s[0, :] * Eig_2_s[0, :],
                    shear_s * Eig_2_s[1, :] * Eig_2_s[0, :],
                    shear_s * Eig_2_s[2, :] * Eig_2_s[0, :],
                    shear_s * Eig_2_s[0, :] * Eig_2_s[1, :],
                    shear_s * Eig_2_s[1, :] * Eig_2_s[1, :],
                    shear_s * Eig_2_s[2, :] * Eig_2_s[1, :],
                    shear_s * Eig_2_s[0, :] * Eig_2_s[2, :],
                    shear_s * Eig_2_s[1, :] * Eig_2_s[2, :],
                    shear_s * Eig_2_s[2, :] * Eig_2_s[2, :],
                )
            )
            mat5_s = 2.0 * np.vstack(
                (
                    shear_s * Eig_3_s[0, :] * Eig_3_s[0, :],
                    shear_s * Eig_3_s[1, :] * Eig_3_s[0, :],
                    shear_s * Eig_3_s[2, :] * Eig_3_s[0, :],
                    shear_s * Eig_3_s[0, :] * Eig_3_s[1, :],
                    shear_s * Eig_3_s[1, :] * Eig_3_s[1, :],
                    shear_s * Eig_3_s[2, :] * Eig_3_s[1, :],
                    shear_s * Eig_3_s[0, :] * Eig_3_s[2, :],
                    shear_s * Eig_3_s[1, :] * Eig_3_s[2, :],
                    shear_s * Eig_3_s[2, :] * Eig_3_s[2, :],
                )
            )
            Eig_6 = 2.0 * shear_s[None, :] * (1.0 + sin_phi_s)[None, :] * Eig_1_s[:3, :] - 2.0 * shear_s[None, :] * (1.0 - sin_phi_s)[None, :] * Eig_3_s[:3, :] + np.array([[1.0], [1.0], [0.0]]) * (2.0 * lame_s * sin_phi_s)[None, :]
            mat6_s = np.vstack(
                (
                    Eig_6[0, :] * Eig_6[0, :],
                    Eig_6[1, :] * Eig_6[0, :],
                    Eig_6[2, :] * Eig_6[0, :],
                    Eig_6[0, :] * Eig_6[1, :],
                    Eig_6[1, :] * Eig_6[1, :],
                    Eig_6[2, :] * Eig_6[1, :],
                    Eig_6[0, :] * Eig_6[2, :],
                    Eig_6[1, :] * Eig_6[2, :],
                    Eig_6[2, :] * Eig_6[2, :],
                )
            ) / denom_s_t[None, :]
            DS[:, test_s] = mat1_s + mat2_s + mat3_s + mat4_s + mat5_s - mat6_s

    test_l = (gamma_sl < gamma_sr) & (lambda_l >= gamma_sl) & (lambda_l <= gamma_la) & (~(test_el | test_s))
    if np.any(test_l):
        lambda_l_t = lambda_l[test_l]
        lame_l = lame[test_l]
        shear_l = shear[test_l]
        sin_phi_l = sin_phi[test_l]
        denom_l_t = denom_l[test_l]
        Eig_1_l = Eig_1[:, test_l]
        Eig_2_l = Eig_2[:, test_l]
        Eig_3_l = Eig_3[:, test_l]
        trace_E_l = trace_E[test_l]
        sigma_1_l = lame_l * trace_E_l + shear_l * (eig_1[test_l] + eig_2[test_l]) - lambda_l_t * (2.0 * lame_l * sin_phi_l + shear_l * (1.0 + sin_phi_l))
        sigma_3_l = lame_l * trace_E_l + 2.0 * shear_l * eig_3[test_l] - lambda_l_t * (2.0 * lame_l * sin_phi_l - 2.0 * shear_l * (1.0 - sin_phi_l))
        S[:, test_l] = sigma_1_l[None, :] * (Eig_1_l + Eig_2_l) + sigma_3_l[None, :] * Eig_3_l
        if return_tangent and DS is not None:
            Eig_12_l = Eig_1_l[:3, :] + Eig_2_l[:3, :]
            EIG_12 = EIG_1[:, test_l] + EIG_2[:, test_l]
            mat1_l = sigma_1_l[None, :] * EIG_12 + sigma_3_l[None, :] * EIG_3[:, test_l]
            mat2_l = vol3.reshape(-1, 1, order="F") * lame_l[None, :]
            mat3_l = np.vstack(
                (
                    shear_l * Eig_12_l[0, :] * Eig_12_l[0, :],
                    shear_l * Eig_12_l[1, :] * Eig_12_l[0, :],
                    shear_l * Eig_12_l[2, :] * Eig_12_l[0, :],
                    shear_l * Eig_12_l[0, :] * Eig_12_l[1, :],
                    shear_l * Eig_12_l[1, :] * Eig_12_l[1, :],
                    shear_l * Eig_12_l[2, :] * Eig_12_l[1, :],
                    shear_l * Eig_12_l[0, :] * Eig_12_l[2, :],
                    shear_l * Eig_12_l[1, :] * Eig_12_l[2, :],
                    shear_l * Eig_12_l[2, :] * Eig_12_l[2, :],
                )
            )
            mat5_l = 2.0 * np.vstack(
                (
                    shear_l * Eig_3_l[0, :] * Eig_3_l[0, :],
                    shear_l * Eig_3_l[1, :] * Eig_3_l[0, :],
                    shear_l * Eig_3_l[2, :] * Eig_3_l[0, :],
                    shear_l * Eig_3_l[0, :] * Eig_3_l[1, :],
                    shear_l * Eig_3_l[1, :] * Eig_3_l[1, :],
                    shear_l * Eig_3_l[2, :] * Eig_3_l[1, :],
                    shear_l * Eig_3_l[0, :] * Eig_3_l[2, :],
                    shear_l * Eig_3_l[1, :] * Eig_3_l[2, :],
                    shear_l * Eig_3_l[2, :] * Eig_3_l[2, :],
                )
            )
            Eig_6 = shear_l[None, :] * (1.0 + sin_phi_l)[None, :] * Eig_12_l - 2.0 * shear_l[None, :] * (1.0 - sin_phi_l)[None, :] * Eig_3_l[:3, :] + np.array([[1.0], [1.0], [0.0]]) * (2.0 * lame_l * sin_phi_l)[None, :]
            mat6_l = np.vstack(
                (
                    Eig_6[0, :] * Eig_6[0, :],
                    Eig_6[1, :] * Eig_6[0, :],
                    Eig_6[2, :] * Eig_6[0, :],
                    Eig_6[0, :] * Eig_6[1, :],
                    Eig_6[1, :] * Eig_6[1, :],
                    Eig_6[2, :] * Eig_6[1, :],
                    Eig_6[0, :] * Eig_6[2, :],
                    Eig_6[1, :] * Eig_6[2, :],
                    Eig_6[2, :] * Eig_6[2, :],
                )
            ) / denom_l_t[None, :]
            DS[:, test_l] = mat1_l + mat2_l + mat3_l + mat5_l - mat6_l

    test_r = (gamma_sl > gamma_sr) & (lambda_r >= gamma_sr) & (lambda_r <= gamma_ra) & (~(test_el | test_s))
    if np.any(test_r):
        lambda_r_t = lambda_r[test_r]
        lame_r = lame[test_r]
        shear_r = shear[test_r]
        sin_phi_r = sin_phi[test_r]
        denom_r_t = denom_r[test_r]
        Eig_1_r = Eig_1[:, test_r]
        Eig_2_r = Eig_2[:, test_r]
        Eig_3_r = Eig_3[:, test_r]
        trace_E_r = trace_E[test_r]
        sigma_1_r = lame_r * trace_E_r + 2.0 * shear_r * eig_1[test_r] - lambda_r_t * (2.0 * lame_r * sin_phi_r + 2.0 * shear_r * (1.0 + sin_phi_r))
        sigma_3_r = lame_r * trace_E_r + shear_r * (eig_2[test_r] + eig_3[test_r]) - lambda_r_t * (2.0 * lame_r * sin_phi_r - shear_r * (1.0 - sin_phi_r))
        S[:, test_r] = sigma_1_r[None, :] * Eig_1_r + sigma_3_r[None, :] * (Eig_2_r + Eig_3_r)
        if return_tangent and DS is not None:
            Eig_23_r = Eig_2_r[:3, :] + Eig_3_r[:3, :]
            EIG_23 = EIG_2[:, test_r] + EIG_3[:, test_r]
            mat1_r = sigma_1_r[None, :] * EIG_1[:, test_r] + sigma_3_r[None, :] * EIG_23
            mat2_r = vol3.reshape(-1, 1, order="F") * lame_r[None, :]
            mat3_r = 2.0 * np.vstack(
                (
                    shear_r * Eig_1_r[0, :] * Eig_1_r[0, :],
                    shear_r * Eig_1_r[1, :] * Eig_1_r[0, :],
                    shear_r * Eig_1_r[2, :] * Eig_1_r[0, :],
                    shear_r * Eig_1_r[0, :] * Eig_1_r[1, :],
                    shear_r * Eig_1_r[1, :] * Eig_1_r[1, :],
                    shear_r * Eig_1_r[2, :] * Eig_1_r[1, :],
                    shear_r * Eig_1_r[0, :] * Eig_1_r[2, :],
                    shear_r * Eig_1_r[1, :] * Eig_1_r[2, :],
                    shear_r * Eig_1_r[2, :] * Eig_1_r[2, :],
                )
            )
            mat5_r = np.vstack(
                (
                    shear_r * Eig_23_r[0, :] * Eig_23_r[0, :],
                    shear_r * Eig_23_r[1, :] * Eig_23_r[0, :],
                    shear_r * Eig_23_r[2, :] * Eig_23_r[0, :],
                    shear_r * Eig_23_r[0, :] * Eig_23_r[1, :],
                    shear_r * Eig_23_r[1, :] * Eig_23_r[1, :],
                    shear_r * Eig_23_r[2, :] * Eig_23_r[1, :],
                    shear_r * Eig_23_r[0, :] * Eig_23_r[2, :],
                    shear_r * Eig_23_r[1, :] * Eig_23_r[2, :],
                    shear_r * Eig_23_r[2, :] * Eig_23_r[2, :],
                )
            )
            Eig_6 = 2.0 * shear_r[None, :] * (1.0 + sin_phi_r)[None, :] * Eig_1_r[:3, :] - shear_r[None, :] * (1.0 - sin_phi_r)[None, :] * Eig_23_r + np.array([[1.0], [1.0], [0.0]]) * (2.0 * lame_r * sin_phi_r)[None, :]
            mat6_r = np.vstack(
                (
                    Eig_6[0, :] * Eig_6[0, :],
                    Eig_6[1, :] * Eig_6[0, :],
                    Eig_6[2, :] * Eig_6[0, :],
                    Eig_6[0, :] * Eig_6[1, :],
                    Eig_6[1, :] * Eig_6[1, :],
                    Eig_6[2, :] * Eig_6[1, :],
                    Eig_6[0, :] * Eig_6[2, :],
                    Eig_6[1, :] * Eig_6[2, :],
                    Eig_6[2, :] * Eig_6[2, :],
                )
            ) / denom_r_t[None, :]
            DS[:, test_r] = mat1_r + mat2_r + mat3_r + mat5_r - mat6_r

    test_a = ~(test_el | test_s | test_l | test_r)
    if np.any(test_a):
        sigma_1_a = c_bar[test_a] / (2.0 * sin_phi[test_a])
        S[:, test_a] = iota4[:, None] * sigma_1_a[None, :]
        if return_tangent and DS is not None:
            DS[:, test_a] = 0.0

    if not return_tangent:
        return S
    return S, DS


def constitutive_problem_3D(E: np.ndarray, c_bar: np.ndarray, sin_phi: np.ndarray, shear: np.ndarray, bulk: np.ndarray, lame: np.ndarray, return_tangent: bool = False):
    E = np.asarray(E, dtype=np.float64)
    if E.shape[0] != 6:
        raise ValueError("E must have 6 rows for 3D")

    c_bar = np.asarray(c_bar, dtype=np.float64).ravel()
    sin_phi = np.asarray(sin_phi, dtype=np.float64).ravel()
    shear = np.asarray(shear, dtype=np.float64).ravel()
    bulk = np.asarray(bulk, dtype=np.float64).ravel()
    lame = np.asarray(lame, dtype=np.float64).ravel()

    n_int = E.shape[1]
    if not (c_bar.size == shear.size == sin_phi.size == bulk.size == lame.size == n_int):
        raise ValueError("Material arrays must be of size n_int")

    sin_psi = sin_phi

    ident = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    iota = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    vol = np.outer(iota, iota)
    dev = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - vol / 3.0
    ident_vec = _vec_col_major(ident)
    vol_vec = _vec_col_major(vol)
    elast = 2.0 * _vec_col_major(dev) * shear + vol_vec * bulk

    E_trial = E
    E_tr = ident @ E_trial
    # MATLAB uses shear ordering [12, 23, 13].
    e11, e22, e33, e12, e23, e13 = E_tr
    E_square = np.vstack(
        (
            e11 * e11 + e12 * e12 + e13 * e13,
            e22 * e22 + e12 * e12 + e23 * e23,
            e33 * e33 + e23 * e23 + e13 * e13,
            e11 * e12 + e22 * e12 + e23 * e13,
            e12 * e13 + e22 * e23 + e33 * e23,
            e11 * e13 + e12 * e23 + e33 * e13,
        )
    )

    I1 = e11 + e22 + e33
    I2 = e11 * e22 + e11 * e33 + e22 * e33 - e12 * e12 - e23 * e23 - e13 * e13
    I3 = e11 * e22 * e33 - e33 * e12 * e12 - e22 * e13 * e13 - e11 * e23 * e23 + 2.0 * e12 * e13 * e23

    Q = np.maximum(0.0, (I1 * I1 - 3.0 * I2) / 9.0)
    R = (-2.0 * (I1**3) + 9.0 * (I1 * I2) - 27.0 * I3) / 54.0
    theta0 = np.zeros(n_int, dtype=np.float64)
    test1 = Q == 0.0
    active = ~test1
    theta0[active] = R[active] / np.sqrt(Q[active] ** 3)
    theta = np.arccos(np.clip(theta0, -1.0, 1.0)) / 3.0

    eig_1 = -2.0 * np.sqrt(Q) * np.cos(theta + 2.0 * np.pi / 3.0) + I1 / 3.0
    eig_2 = -2.0 * np.sqrt(Q) * np.cos(theta - 2.0 * np.pi / 3.0) + I1 / 3.0
    eig_3 = -2.0 * np.sqrt(Q) * np.cos(theta) + I1 / 3.0

    f_tr = 2.0 * shear * ((1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * eig_3) + 2.0 * (lame * sin_phi) * I1 - c_bar
    gamma_sl = (eig_1 - eig_2) / (1.0 + sin_psi)
    gamma_sr = (eig_2 - eig_3) / (1.0 - sin_psi)
    gamma_la = (eig_1 + eig_2 - 2.0 * eig_3) / (3.0 - sin_psi)
    gamma_ra = (2.0 * eig_1 - eig_2 - eig_3) / (3.0 + sin_psi)

    denom_s = 4.0 * lame * sin_phi * sin_psi + 4.0 * shear * (1.0 + sin_phi * sin_psi)
    denom_l = 4.0 * lame * sin_phi * sin_psi + shear * (1.0 + sin_phi) * (1.0 + sin_psi) + 2.0 * shear * (1.0 - sin_phi) * (1.0 - sin_psi)
    denom_r = 4.0 * lame * sin_phi * sin_psi + 2.0 * shear * (1.0 + sin_phi) * (1.0 + sin_psi) + shear * (1.0 - sin_phi) * (1.0 - sin_psi)
    denom_a = 4.0 * bulk * sin_phi * sin_psi

    lambda_s = f_tr / denom_s
    lambda_l = (
        shear * ((1.0 + sin_phi) * (eig_1 + eig_2) - 2.0 * (1.0 - sin_phi) * eig_3)
        + 2.0 * lame * sin_phi * I1
        - c_bar
    ) / denom_l
    lambda_r = (
        shear * (2.0 * (1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * (eig_2 + eig_3))
        + 2.0 * lame * sin_phi * I1
        - c_bar
    ) / denom_r
    lambda_a = (2.0 * bulk * sin_phi * I1 - c_bar) / denom_a

    S = np.zeros((6, n_int), dtype=np.float64)

    test_el = f_tr <= 0.0
    if np.any(test_el):
        S[:, test_el] = lame[test_el][None, :] * (vol @ E_trial[:, test_el]) + 2.0 * shear[test_el][None, :] * (ident @ E_trial[:, test_el])

    test_s = (lambda_s <= np.minimum(gamma_sl, gamma_sr)) & (~test_el)
    if np.any(test_s):
        lame_s = lame[test_s]
        shear_s = shear[test_s]
        sin_phi_s = sin_phi[test_s]
        sin_psi_s = sin_psi[test_s]
        eig_1_s = eig_1[test_s]
        eig_2_s = eig_2[test_s]
        eig_3_s = eig_3[test_s]
        I1_s = I1[test_s]
        E_square_s = E_square[:, test_s]
        E_tr_s = E_tr[:, test_s]
        lambda_s_sel = lambda_s[test_s]
        denom_s1 = (eig_1_s - eig_2_s) * (eig_1_s - eig_3_s)
        denom_s2 = (eig_2_s - eig_1_s) * (eig_2_s - eig_3_s)
        denom_s3 = (eig_3_s - eig_1_s) * (eig_3_s - eig_2_s)
        Eig_1_s = (1.0 / denom_s1)[None, :] * (
            E_square_s - (eig_2_s + eig_3_s)[None, :] * E_tr_s + iota[:, None] * (eig_2_s * eig_3_s)[None, :]
        )
        Eig_2_s = (1.0 / denom_s2)[None, :] * (
            E_square_s - (eig_1_s + eig_3_s)[None, :] * E_tr_s + iota[:, None] * (eig_1_s * eig_3_s)[None, :]
        )
        Eig_3_s = (1.0 / denom_s3)[None, :] * (
            E_square_s - (eig_1_s + eig_2_s)[None, :] * E_tr_s + iota[:, None] * (eig_1_s * eig_2_s)[None, :]
        )
        sigma_1_s = lame_s * I1_s + 2.0 * shear_s * eig_1_s - lambda_s_sel * (2.0 * lame_s * sin_psi_s + 2.0 * shear_s * (1.0 + sin_psi_s))
        sigma_2_s = lame_s * I1_s + 2.0 * shear_s * eig_2_s - lambda_s_sel * (2.0 * lame_s * sin_psi_s)
        sigma_3_s = lame_s * I1_s + 2.0 * shear_s * eig_3_s - lambda_s_sel * (2.0 * lame_s * sin_psi_s - 2.0 * shear_s * (1.0 - sin_psi_s))
        S[:, test_s] = sigma_1_s[None, :] * Eig_1_s + sigma_2_s[None, :] * Eig_2_s + sigma_3_s[None, :] * Eig_3_s

    test_l = (gamma_sl < gamma_sr) & (lambda_l >= gamma_sl) & (lambda_l <= gamma_la) & (~(test_el | test_s))
    if np.any(test_l):
        lame_l = lame[test_l]
        shear_l = shear[test_l]
        sin_phi_l = sin_phi[test_l]
        sin_psi_l = sin_psi[test_l]
        nt_l = int(np.count_nonzero(test_l))
        eig_1_l = eig_1[test_l]
        eig_2_l = eig_2[test_l]
        eig_3_l = eig_3[test_l]
        I1_l = I1[test_l]
        lambda_l_sel = lambda_l[test_l]
        E_square_l = E_square[:, test_l]
        E_tr_l = E_tr[:, test_l]
        denom_l3 = (eig_3_l - eig_1_l) * (eig_3_l - eig_2_l)
        Eig_3_l = (1.0 / denom_l3)[None, :] * (
            E_square_l - (eig_1_l + eig_2_l)[None, :] * E_tr_l + iota[:, None] * (eig_1_l * eig_2_l)[None, :]
        )
        Eig_12_l = np.vstack((np.ones((3, nt_l), dtype=np.float64), np.zeros((3, nt_l), dtype=np.float64))) - Eig_3_l
        sigma_1_l = lame_l * I1_l + shear_l * (eig_1_l + eig_2_l) - lambda_l_sel * (2.0 * lame_l * sin_psi_l + shear_l * (1.0 + sin_psi_l))
        sigma_3_l = lame_l * I1_l + 2.0 * shear_l * eig_3_l - lambda_l_sel * (2.0 * lame_l * sin_psi_l - 2.0 * shear_l * (1.0 - sin_psi_l))
        S[:, test_l] = sigma_1_l[None, :] * Eig_12_l + sigma_3_l[None, :] * Eig_3_l

    test_r = (gamma_sl > gamma_sr) & (lambda_r >= gamma_sr) & (lambda_r <= gamma_ra) & (~(test_el | test_s))
    if np.any(test_r):
        lame_r = lame[test_r]
        shear_r = shear[test_r]
        sin_phi_r = sin_phi[test_r]
        sin_psi_r = sin_psi[test_r]
        nt_r = int(np.count_nonzero(test_r))
        eig_1_r = eig_1[test_r]
        eig_2_r = eig_2[test_r]
        eig_3_r = eig_3[test_r]
        I1_r = I1[test_r]
        lambda_r_sel = lambda_r[test_r]
        E_square_r = E_square[:, test_r]
        E_tr_r = E_tr[:, test_r]
        denom_r1 = (eig_1_r - eig_2_r) * (eig_1_r - eig_3_r)
        Eig_1_r = (1.0 / denom_r1)[None, :] * (
            E_square_r - (eig_2_r + eig_3_r)[None, :] * E_tr_r + iota[:, None] * (eig_2_r * eig_3_r)[None, :]
        )
        Eig_23_r = np.vstack((np.ones((3, nt_r), dtype=np.float64), np.zeros((3, nt_r), dtype=np.float64))) - Eig_1_r
        sigma_1_r = lame_r * I1_r + 2.0 * shear_r * eig_1_r - lambda_r_sel * (2.0 * lame_r * sin_psi_r + 2.0 * shear_r * (1.0 + sin_psi_r))
        sigma_3_r = lame_r * I1_r + shear_r * (eig_2_r + eig_3_r) - lambda_r_sel * (2.0 * lame_r * sin_psi_r - shear_r * (1.0 - sin_psi_r))
        S[:, test_r] = sigma_1_r[None, :] * Eig_1_r + sigma_3_r[None, :] * Eig_23_r

    test_a = ~(test_el | test_s | test_l | test_r)
    if np.any(test_a):
        sigma_1_a = c_bar[test_a] / (2.0 * sin_phi[test_a])
        S[:, test_a] = iota[:, None] * sigma_1_a[None, :]

    if not return_tangent:
        return S

    DS = np.zeros((36, n_int), dtype=np.float64)
    DER_E_square = np.vstack(
        (
            2.0 * e11,
            np.zeros(n_int),
            np.zeros(n_int),
            e12,
            np.zeros(n_int),
            e13,
            np.zeros(n_int),
            2.0 * e22,
            np.zeros(n_int),
            e12,
            e23,
            np.zeros(n_int),
            np.zeros(n_int),
            np.zeros(n_int),
            2.0 * e33,
            np.zeros(n_int),
            e23,
            e13,
            e12,
            e12,
            np.zeros(n_int),
            0.5 * (e11 + e22),
            0.5 * e13,
            0.5 * e23,
            np.zeros(n_int),
            e23,
            e23,
            0.5 * e13,
            0.5 * (e22 + e33),
            0.5 * e12,
            e13,
            np.zeros(n_int),
            e13,
            0.5 * e23,
            0.5 * e12,
            0.5 * (e11 + e33),
        )
    )

    if np.any(test_el):
        DS[:, test_el] = elast[:, test_el]

    if np.any(test_s):
        lame_s = lame[test_s]
        shear_s = shear[test_s]
        sin_phi_s = sin_phi[test_s]
        sin_psi_s = sin_psi[test_s]
        eig_1_s = eig_1[test_s]
        eig_2_s = eig_2[test_s]
        eig_3_s = eig_3[test_s]
        sigma_1_s = lame_s * I1_s + 2.0 * shear_s * eig_1_s - lambda_s_sel * (2.0 * lame_s * sin_psi_s + 2.0 * shear_s * (1.0 + sin_psi_s))
        sigma_2_s = lame_s * I1_s + 2.0 * shear_s * eig_2_s - lambda_s_sel * (2.0 * lame_s * sin_psi_s)
        sigma_3_s = lame_s * I1_s + 2.0 * shear_s * eig_3_s - lambda_s_sel * (2.0 * lame_s * sin_psi_s - 2.0 * shear_s * (1.0 - sin_psi_s))
        DER_E_square_s = DER_E_square[:, test_s]
        E1_x_E1 = _outer_columns(Eig_1_s, Eig_1_s)
        E2_x_E2 = _outer_columns(Eig_2_s, Eig_2_s)
        E3_x_E3 = _outer_columns(Eig_3_s, Eig_3_s)
        EIG_1_s = (1.0 / denom_s1)[None, :] * (
            DER_E_square_s
            - ident_vec * (eig_2_s + eig_3_s)[None, :]
            - (2.0 * eig_1_s - eig_2_s - eig_3_s)[None, :] * E1_x_E1
            - (eig_2_s - eig_3_s)[None, :] * (E2_x_E2 - E3_x_E3)
        )
        EIG_2_s = (1.0 / denom_s2)[None, :] * (
            DER_E_square_s
            - ident_vec * (eig_1_s + eig_3_s)[None, :]
            - (2.0 * eig_2_s - eig_1_s - eig_3_s)[None, :] * E2_x_E2
            - (eig_1_s - eig_3_s)[None, :] * (E1_x_E1 - E3_x_E3)
        )
        EIG_3_s = (1.0 / denom_s3)[None, :] * (
            DER_E_square_s
            - ident_vec * (eig_1_s + eig_2_s)[None, :]
            - (2.0 * eig_3_s - eig_1_s - eig_2_s)[None, :] * E3_x_E3
            - (eig_1_s - eig_2_s)[None, :] * (E1_x_E1 - E2_x_E2)
        )
        Sder1_s = sigma_1_s[None, :] * EIG_1_s + sigma_2_s[None, :] * EIG_2_s + sigma_3_s[None, :] * EIG_3_s
        Sder2_s = vol_vec * lame_s[None, :]
        Sder3_s = 2.0 * shear_s[None, :] * (E1_x_E1 + E2_x_E2 + E3_x_E3)
        D_phi_s = 2.0 * shear_s[None, :] * ((1.0 + sin_phi_s)[None, :] * Eig_1_s - (1.0 - sin_phi_s)[None, :] * Eig_3_s) + 2.0 * iota[:, None] * (lame_s * sin_phi_s)[None, :]
        D_psi_s = 2.0 * shear_s[None, :] * ((1.0 + sin_psi_s)[None, :] * Eig_1_s - (1.0 - sin_psi_s)[None, :] * Eig_3_s) + 2.0 * iota[:, None] * (lame_s * sin_psi_s)[None, :]
        Sder4_s = _outer_columns(D_psi_s, D_phi_s) / denom_s[test_s][None, :]
        DS[:, test_s] = Sder1_s + Sder2_s + Sder3_s - Sder4_s

    if np.any(test_l):
        lame_l = lame[test_l]
        shear_l = shear[test_l]
        sin_phi_l = sin_phi[test_l]
        sin_psi_l = sin_psi[test_l]
        eig_1_l = eig_1[test_l]
        eig_2_l = eig_2[test_l]
        eig_3_l = eig_3[test_l]
        sigma_1_l = lame_l * I1_l + shear_l * (eig_1_l + eig_2_l) - lambda_l_sel * (2.0 * lame_l * sin_psi_l + shear_l * (1.0 + sin_psi_l))
        sigma_3_l = lame_l * I1_l + 2.0 * shear_l * eig_3_l - lambda_l_sel * (2.0 * lame_l * sin_psi_l - 2.0 * shear_l * (1.0 - sin_psi_l))
        E3_x_E3 = _outer_columns(Eig_3_l, Eig_3_l)
        E12_x_E12 = _outer_columns(Eig_12_l, Eig_12_l)
        E_tr_l = E_tr[:, test_l]
        E12_x_Etr = _outer_columns(Eig_12_l, E_tr_l)
        Etr_x_E12 = _outer_columns(E_tr_l, Eig_12_l)
        E12_x_E3 = _outer_columns(Eig_12_l, Eig_3_l)
        E3_x_E12 = _outer_columns(Eig_3_l, Eig_12_l)
        EIG_3_l = (1.0 / denom_l3)[None, :] * (
            DER_E_square[:, test_l]
            - ident_vec * (eig_1_l + eig_2_l)[None, :]
            - (Etr_x_E12 + E12_x_Etr)
            + (eig_1_l + eig_2_l)[None, :] * E12_x_E12
            + (eig_1_l + eig_2_l - 2.0 * eig_3_l)[None, :] * E3_x_E3
            + eig_3_l[None, :] * (E12_x_E3 + E3_x_E12)
        )
        Sder1_l = (sigma_3_l - sigma_1_l)[None, :] * EIG_3_l
        Sder2_l = vol_vec * lame_l[None, :]
        Sder3_l = shear_l[None, :] * (E12_x_E12 + 2.0 * E3_x_E3)
        D_phi_l = shear_l[None, :] * ((1.0 + sin_phi_l)[None, :] * Eig_12_l - 2.0 * (1.0 - sin_phi_l)[None, :] * Eig_3_l) + 2.0 * iota[:, None] * (lame_l * sin_phi_l)[None, :]
        D_psi_l = shear_l[None, :] * ((1.0 + sin_psi_l)[None, :] * Eig_12_l - 2.0 * (1.0 - sin_psi_l)[None, :] * Eig_3_l) + 2.0 * iota[:, None] * (lame_l * sin_psi_l)[None, :]
        Sder4_l = _outer_columns(D_psi_l, D_phi_l) / denom_l[test_l][None, :]
        DS[:, test_l] = Sder1_l + Sder2_l + Sder3_l - Sder4_l

    if np.any(test_r):
        lame_r = lame[test_r]
        shear_r = shear[test_r]
        sin_phi_r = sin_phi[test_r]
        sin_psi_r = sin_psi[test_r]
        eig_1_r = eig_1[test_r]
        eig_2_r = eig_2[test_r]
        eig_3_r = eig_3[test_r]
        sigma_1_r = lame_r * I1_r + 2.0 * shear_r * eig_1_r - lambda_r_sel * (2.0 * lame_r * sin_psi_r + 2.0 * shear_r * (1.0 + sin_psi_r))
        sigma_3_r = lame_r * I1_r + shear_r * (eig_2_r + eig_3_r) - lambda_r_sel * (2.0 * lame_r * sin_psi_r - shear_r * (1.0 - sin_psi_r))
        E1_x_E1 = _outer_columns(Eig_1_r, Eig_1_r)
        E23_x_E23 = _outer_columns(Eig_23_r, Eig_23_r)
        E_tr_r = E_tr[:, test_r]
        E23_x_Etr = _outer_columns(Eig_23_r, E_tr_r)
        Etr_x_E23 = _outer_columns(E_tr_r, Eig_23_r)
        E23_x_E1 = _outer_columns(Eig_23_r, Eig_1_r)
        E1_x_E23 = _outer_columns(Eig_1_r, Eig_23_r)
        EIG_1_r = (1.0 / denom_r1)[None, :] * (
            DER_E_square[:, test_r]
            - ident_vec * (eig_2_r + eig_3_r)[None, :]
            - Etr_x_E23
            - E23_x_Etr
            + (eig_2_r + eig_3_r)[None, :] * E23_x_E23
            + (eig_2_r + eig_3_r - 2.0 * eig_1_r)[None, :] * E1_x_E1
            + eig_1_r[None, :] * (E23_x_E1 + E1_x_E23)
        )
        Sder1_r = (sigma_1_r - sigma_3_r)[None, :] * EIG_1_r
        Sder2_r = vol_vec * lame_r[None, :]
        Sder3_r = shear_r[None, :] * (2.0 * E1_x_E1 + E23_x_E23)
        D_phi_r = shear_r[None, :] * (2.0 * (1.0 + sin_phi_r)[None, :] * Eig_1_r - (1.0 - sin_phi_r)[None, :] * Eig_23_r) + 2.0 * iota[:, None] * (lame_r * sin_phi_r)[None, :]
        D_psi_r = shear_r[None, :] * (2.0 * (1.0 + sin_psi_r)[None, :] * Eig_1_r - (1.0 - sin_psi_r)[None, :] * Eig_23_r) + 2.0 * iota[:, None] * (lame_r * sin_psi_r)[None, :]
        Sder4_r = _outer_columns(D_psi_r, D_phi_r) / denom_r[test_r][None, :]
        DS[:, test_r] = Sder1_r + Sder2_r + Sder3_r - Sder4_r

    return S, DS


def _batch_constitutive_problem_3d(
    E: np.ndarray,
    c_bar: np.ndarray,
    sin_phi: np.ndarray,
    shear: np.ndarray,
    bulk: np.ndarray,
    lame: np.ndarray,
    *,
    return_tangent: bool,
    use_compiled: bool,
):
    E = np.asarray(E, dtype=np.float64)
    if E.shape[0] != 6:
        raise ValueError("E must have 6 rows for 3D constitutive evaluation")

    c_bar = np.asarray(c_bar, dtype=np.float64).reshape(-1)
    sin_phi = np.asarray(sin_phi, dtype=np.float64).reshape(-1)
    shear = np.asarray(shear, dtype=np.float64).reshape(-1)
    bulk = np.asarray(bulk, dtype=np.float64).reshape(-1)
    lame = np.asarray(lame, dtype=np.float64).reshape(-1)
    n_int = E.shape[1]
    if not (c_bar.size == sin_phi.size == shear.size == bulk.size == lame.size == n_int):
        raise ValueError("Material arrays must all have length n_int")

    if use_compiled and _kernels is not None:
        e_c = np.ascontiguousarray(E.T, dtype=np.float64)
        c_c = np.ascontiguousarray(c_bar, dtype=np.float64)
        sin_c = np.ascontiguousarray(sin_phi, dtype=np.float64)
        shear_c = np.ascontiguousarray(shear, dtype=np.float64)
        bulk_c = np.ascontiguousarray(bulk, dtype=np.float64)
        lame_c = np.ascontiguousarray(lame, dtype=np.float64)
        if return_tangent:
            S_c, DS_c = _kernels.constitutive_problem_3d_sds(e_c, c_c, sin_c, shear_c, bulk_c, lame_c)
            return np.asarray(S_c, dtype=np.float64).T, np.asarray(DS_c, dtype=np.float64).T
        S_c = _kernels.constitutive_problem_3d_s(e_c, c_c, sin_c, shear_c, bulk_c, lame_c)
        return np.asarray(S_c, dtype=np.float64).T

    return constitutive_problem_3D(E, c_bar, sin_phi, shear, bulk, lame, return_tangent=return_tangent)


def _batch_constitutive_problem_local(
    E: np.ndarray,
    c_bar: np.ndarray,
    sin_phi: np.ndarray,
    shear: np.ndarray,
    bulk: np.ndarray,
    lame: np.ndarray,
    *,
    dim: int,
    return_tangent: bool,
    use_compiled: bool,
):
    if int(dim) == 3:
        return _batch_constitutive_problem_3d(
            E,
            c_bar,
            sin_phi,
            shear,
            bulk,
            lame,
            return_tangent=return_tangent,
            use_compiled=use_compiled,
        )
    if int(dim) == 2:
        return constitutive_problem_2D(E, c_bar, sin_phi, shear, bulk, lame, return_tangent=return_tangent)
    raise ValueError(f"Unsupported dimension {dim}")


def _gather_owned_rows(local_rows: np.ndarray, global_size: int, comm) -> np.ndarray:
    local = np.asarray(local_rows, dtype=np.float64).reshape(-1)
    mpi_comm = comm.tompi4py() if comm is not None and hasattr(comm, "tompi4py") else comm
    if mpi_comm is None:
        return local
    size = int(mpi_comm.getSize()) if hasattr(mpi_comm, "getSize") else int(mpi_comm.Get_size())
    if size == 1:
        return local
    parts = mpi_comm.allgather(local)
    if not parts:
        return local
    gathered = np.concatenate(parts)
    if gathered.size != int(global_size):
        raise RuntimeError("Gathered owned-row vector has unexpected size")
    return gathered


def _gather_owned_free_rows(local_rows: np.ndarray, global_size: int, comm) -> np.ndarray:
    local = np.asarray(local_rows, dtype=np.float64).reshape(-1)
    mpi_comm = comm.tompi4py() if comm is not None and hasattr(comm, "tompi4py") else comm
    if mpi_comm is None:
        return local
    size = int(mpi_comm.getSize()) if hasattr(mpi_comm, "getSize") else int(mpi_comm.Get_size())
    if size == 1:
        return local
    parts = mpi_comm.allgather(local)
    if not parts:
        return local
    gathered = np.concatenate(parts)
    if gathered.size != int(global_size):
        raise RuntimeError("Gathered owned free-row vector has unexpected size")
    return gathered


def _owned_force_from_local_stress(
    pattern: OwnedTangentPattern,
    stress_local: np.ndarray,
    *,
    use_compiled: bool,
) -> np.ndarray:
    return assemble_owned_force_from_local_stress(pattern, stress_local, use_compiled=use_compiled)


def _exchange_owned_constitutive_overlap(
    pattern: OwnedTangentPattern,
    S_unique: np.ndarray,
    DS_unique: np.ndarray | None,
    *,
    return_tangent: bool,
    comm,
) -> tuple[np.ndarray, np.ndarray | None]:
    n_overlap = int(np.asarray(pattern.local_int_indices, dtype=np.int64).size)
    S_local = np.empty((int(pattern.n_strain), n_overlap), dtype=np.float64)
    DS_local = np.empty((int(pattern.n_strain * pattern.n_strain), n_overlap), dtype=np.float64) if return_tangent else None
    filled = np.zeros(n_overlap, dtype=bool)

    owner_mask = np.asarray(pattern.local_overlap_owner_mask, dtype=bool)
    owner_pos = np.asarray(pattern.local_overlap_to_unique_pos, dtype=np.int32)
    if np.any(owner_mask):
        unique_pos = owner_pos[owner_mask]
        S_local[:, owner_mask] = np.asarray(S_unique[:, unique_pos], dtype=np.float64)
        if return_tangent and DS_local is not None and DS_unique is not None:
            DS_local[:, owner_mask] = np.asarray(DS_unique[:, unique_pos], dtype=np.float64)
        filled[owner_mask] = True

    mpi_comm = comm.tompi4py() if comm is not None and hasattr(comm, "tompi4py") else comm
    if mpi_comm is None:
        if not np.all(filled):
            raise RuntimeError("Owned constitutive exchange left overlap integration points unfilled in serial mode")
        return S_local, DS_local

    size = int(mpi_comm.getSize()) if hasattr(mpi_comm, "getSize") else int(mpi_comm.Get_size())
    if size == 1:
        if not np.all(filled):
            raise RuntimeError("Owned constitutive exchange left overlap integration points unfilled for single-rank communicator")
        return S_local, DS_local

    send_neighbor_ranks = np.asarray(pattern.send_neighbor_ranks, dtype=np.int32)
    send_ptr = np.asarray(pattern.send_ptr, dtype=np.int32)
    send_unique_pos = np.asarray(pattern.send_unique_pos, dtype=np.int32)
    recv_neighbor_ranks = np.asarray(pattern.recv_neighbor_ranks, dtype=np.int32)
    recv_ptr = np.asarray(pattern.recv_ptr, dtype=np.int32)
    recv_overlap_pos = np.asarray(pattern.recv_overlap_pos, dtype=np.int32)

    requests = []
    recv_s_buffers: list[np.ndarray] = []
    recv_ds_buffers: list[np.ndarray] = []

    for idx, neighbor_rank in enumerate(recv_neighbor_ranks.tolist()):
        r0 = int(recv_ptr[idx])
        r1 = int(recv_ptr[idx + 1])
        count = r1 - r0
        recv_s = np.empty((count, int(pattern.n_strain)), dtype=np.float64)
        recv_s_buffers.append(recv_s)
        requests.append(mpi_comm.Irecv(recv_s, source=int(neighbor_rank), tag=_OWNED_UNIQUE_EXCHANGE_S_TAG))
        if return_tangent and DS_local is not None:
            recv_ds = np.empty((count, int(pattern.n_strain * pattern.n_strain)), dtype=np.float64)
            recv_ds_buffers.append(recv_ds)
            requests.append(mpi_comm.Irecv(recv_ds, source=int(neighbor_rank), tag=_OWNED_UNIQUE_EXCHANGE_DS_TAG))

    send_s_buffers: list[np.ndarray] = []
    send_ds_buffers: list[np.ndarray] = []
    for idx, neighbor_rank in enumerate(send_neighbor_ranks.tolist()):
        s0 = int(send_ptr[idx])
        s1 = int(send_ptr[idx + 1])
        unique_pos = send_unique_pos[s0:s1]
        send_s = np.ascontiguousarray(np.asarray(S_unique[:, unique_pos], dtype=np.float64).T, dtype=np.float64)
        send_s_buffers.append(send_s)
        requests.append(mpi_comm.Isend(send_s, dest=int(neighbor_rank), tag=_OWNED_UNIQUE_EXCHANGE_S_TAG))
        if return_tangent and DS_local is not None and DS_unique is not None:
            send_ds = np.ascontiguousarray(np.asarray(DS_unique[:, unique_pos], dtype=np.float64).T, dtype=np.float64)
            send_ds_buffers.append(send_ds)
            requests.append(mpi_comm.Isend(send_ds, dest=int(neighbor_rank), tag=_OWNED_UNIQUE_EXCHANGE_DS_TAG))

    if requests:
        if hasattr(PYMPI, "Request"):
            PYMPI.Request.Waitall(requests)
        else:  # pragma: no cover - mpi4py import is guarded above
            for req in requests:
                req.Wait()

    for idx, _neighbor_rank in enumerate(recv_neighbor_ranks.tolist()):
        r0 = int(recv_ptr[idx])
        r1 = int(recv_ptr[idx + 1])
        overlap_pos = recv_overlap_pos[r0:r1]
        S_local[:, overlap_pos] = recv_s_buffers[idx].T
        filled[overlap_pos] = True
        if return_tangent and DS_local is not None:
            DS_local[:, overlap_pos] = recv_ds_buffers[idx].T

    if not np.all(filled):
        missing = np.flatnonzero(~filled)
        raise RuntimeError(f"Owned constitutive exchange left {missing.size} overlap integration points unfilled")
    return S_local, DS_local


def _owned_free_force_from_local_stress(
    pattern: OwnedTangentPattern,
    stress_local: np.ndarray,
    *,
    use_compiled: bool,
) -> np.ndarray:
    owned_force = _owned_force_from_local_stress(pattern, stress_local, use_compiled=use_compiled)
    return np.asarray(owned_force[np.asarray(pattern.owned_free_local_rows, dtype=np.int64)], dtype=np.float64)


def _local_owned_rows_from_field(pattern: OwnedTangentPattern, field: np.ndarray) -> np.ndarray:
    row0, row1 = pattern.owned_row_range
    flat = np.asarray(field, dtype=np.float64).reshape(-1, order="F")
    return np.asarray(flat[row0:row1], dtype=np.float64)


def _local_owned_free_rows_from_field(pattern: OwnedTangentPattern, field: np.ndarray) -> np.ndarray:
    local = _local_owned_rows_from_field(pattern, field)
    return np.asarray(local[np.asarray(pattern.owned_free_mask, dtype=bool)], dtype=np.float64)


def potential_2D(E: np.ndarray, c_bar: np.ndarray, sin_phi: np.ndarray, shear: np.ndarray, bulk: np.ndarray, lame: np.ndarray):
    E = np.asarray(E, dtype=np.float64)
    if E.shape[0] != 3:
        raise ValueError("E must have 3 rows for 2D")

    c_bar = np.asarray(c_bar, dtype=np.float64).ravel()
    sin_phi = np.asarray(sin_phi, dtype=np.float64).ravel()
    shear = np.asarray(shear, dtype=np.float64).ravel()
    bulk = np.asarray(bulk, dtype=np.float64).ravel()
    lame = np.asarray(lame, dtype=np.float64).ravel()

    n_int = E.shape[1]
    if not (c_bar.size == shear.size == sin_phi.size == bulk.size == lame.size == n_int):
        raise ValueError("Material arrays must be of size n_int")

    e11, e22, e12 = E
    e33 = np.zeros(n_int)
    I1 = e11 + e22
    I2 = np.sqrt((e11 - e22) ** 2 + e12 * e12)
    eig0_1 = (I1 + I2) / 2.0
    eig0_2 = (I1 - I2) / 2.0
    eig0_3 = e33
    eig_1 = eig0_1
    eig_2 = eig0_2
    eig_3 = eig0_3

    test2 = (eig0_1 >= eig0_3) & (eig0_3 > eig0_2)
    test3 = eig0_3 > eig0_1
    eig_2[test2] = eig0_3[test2]
    eig_3[test2] = eig0_2[test2]
    eig_1[test3] = eig0_3[test3]
    eig_2[test3] = eig0_1[test3]
    eig_3[test3] = eig0_2[test3]

    trace_E = eig_1 + eig_2 + eig_3
    f_tr = 2.0 * shear * ((1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * eig_3) + 2.0 * lame * sin_phi * trace_E - c_bar
    gamma_sl = (eig_1 - eig_2) / np.maximum(1.0e-15, (1.0 + sin_phi))
    gamma_sr = (eig_2 - eig_3) / np.maximum(1.0e-15, (1.0 - sin_phi))
    gamma_la = (eig_1 + eig_2 - 2.0 * eig_3) / np.maximum(1.0e-15, (3.0 - sin_phi))
    gamma_ra = (2.0 * eig_1 - eig_2 - eig_3) / np.maximum(1.0e-15, (3.0 + sin_phi))

    denom_s = 4.0 * lame * sin_phi**2 + 2.0 * shear * (1.0 + sin_phi) ** 2 + 2.0 * shear * (1.0 - sin_phi) ** 2
    denom_l = 4.0 * lame * sin_phi**2 + shear * (1.0 + sin_phi) ** 2 + 2.0 * shear * (1.0 - sin_phi) ** 2
    denom_r = 4.0 * lame * sin_phi**2 + 2.0 * shear * (1.0 + sin_phi) ** 2 + shear * (1.0 - sin_phi) ** 2
    denom_a = 4.0 * bulk * sin_phi**2

    lambda_s = f_tr / np.where(np.abs(denom_s) < 1.0e-15, np.sign(denom_s + 1.0e-15) * 1.0e-15, denom_s)
    lambda_l = (shear * ((1.0 + sin_phi) * (eig_1 + eig_2) - 2.0 * (1.0 - sin_phi) * eig_3) + 2.0 * lame * sin_phi * trace_E - c_bar) / np.where(np.abs(denom_l) < 1.0e-15, np.sign(denom_l + 1.0e-15) * 1.0e-15, denom_l)
    lambda_r = (shear * (2.0 * (1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * (eig_2 + eig_3)) + 2.0 * lame * sin_phi * trace_E - c_bar) / np.where(np.abs(denom_r) < 1.0e-15, np.sign(denom_r + 1.0e-15) * 1.0e-15, denom_r)
    lambda_a = (2.0 * bulk * sin_phi * trace_E - c_bar) / np.where(np.abs(denom_a) < 1.0e-15, np.sign(denom_a + 1.0e-15) * 1.0e-15, denom_a)

    Psi = np.zeros(n_int, dtype=np.float64)

    test_el = f_tr <= 0.0
    Psi[test_el] = 0.5 * lame[test_el] * trace_E[test_el] ** 2 + shear[test_el] * (eig_1[test_el] ** 2 + eig_2[test_el] ** 2 + eig_3[test_el] ** 2)

    test_s = (lambda_s <= np.minimum(gamma_sl, gamma_sr)) & (~test_el)
    Psi[test_s] = (
        0.5 * lame[test_s] * trace_E[test_s] ** 2
        + shear[test_s] * (eig_1[test_s] ** 2 + eig_2[test_s] ** 2 + eig_3[test_s] ** 2)
        - 0.5 * (denom_s[test_s]) * (lambda_s[test_s] ** 2)
    )

    test_l = (gamma_sl < gamma_sr) & (lambda_l >= gamma_sl) & (lambda_l <= gamma_la) & (~(test_el | test_s))
    Psi[test_l] = (
        0.5 * lame[test_l] * trace_E[test_l] ** 2
        + shear[test_l] * (eig_3[test_l] ** 2 + 0.5 * (eig_1[test_l] + eig_2[test_l]) ** 2)
        - 0.5 * (denom_l[test_l]) * (lambda_l[test_l] ** 2)
    )

    test_r = (gamma_sl > gamma_sr) & (lambda_r >= gamma_sr) & (lambda_r <= gamma_ra) & (~(test_el | test_s))
    Psi[test_r] = (
        0.5 * lame[test_r] * trace_E[test_r] ** 2
        + shear[test_r] * (eig_1[test_r] ** 2 + 0.5 * (eig_2[test_r] + eig_3[test_r]) ** 2)
        - 0.5 * (denom_r[test_r]) * (lambda_r[test_r] ** 2)
    )

    test_a = ~(test_el | test_s | test_l | test_r)
    Psi[test_a] = 0.5 * bulk[test_a] * trace_E[test_a] ** 2 - 0.5 * (denom_a[test_a]) * (lambda_a[test_a] ** 2)
    return Psi


def potential_3D(E: np.ndarray, c_bar: np.ndarray, sin_phi: np.ndarray, shear: np.ndarray, bulk: np.ndarray, lame: np.ndarray):
    E = np.asarray(E, dtype=np.float64)
    if E.shape[0] != 6:
        raise ValueError("E must have 6 rows for 3D")

    c_bar = np.asarray(c_bar, dtype=np.float64).ravel()
    sin_phi = np.asarray(sin_phi, dtype=np.float64).ravel()
    shear = np.asarray(shear, dtype=np.float64).ravel()
    bulk = np.asarray(bulk, dtype=np.float64).ravel()
    lame = np.asarray(lame, dtype=np.float64).ravel()

    n_int = E.shape[1]
    if not (c_bar.size == shear.size == sin_phi.size == bulk.size == lame.size == n_int):
        raise ValueError("Material arrays must be of size n_int")

    e11, e22, e33, e12, e13, e23 = E
    I1 = e11 + e22 + e33
    I2 = e11 * e22 + e11 * e33 + e22 * e33 - e12 * e12 - e13 * e13 - e23 * e23
    I3 = e11 * e22 * e33 - e33 * e12 * e12 - e22 * e13 * e13 - e11 * e23 * e23 + 2.0 * e12 * e13 * e23

    Q = np.maximum(0.0, (I1 * I1 - 3.0 * I2) / 9.0)
    R = (-2.0 * I1 ** 3 + 9.0 * I1 * I2 - 27.0 * I3) / 54.0
    theta = np.zeros(n_int, dtype=np.float64)
    active = Q > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        theta0 = R[active] / np.maximum(1.0e-15, np.sqrt(Q[active] ** 3))
    theta0 = np.clip(theta0, -1.0, 1.0)
    theta[active] = np.arccos(theta0) / 3.0

    sqrtQ = np.sqrt(Q)
    eig_1 = -2.0 * sqrtQ * np.cos(theta + 2.0 * np.pi / 3.0) + I1 / 3.0
    eig_2 = -2.0 * sqrtQ * np.cos(theta - 2.0 * np.pi / 3.0) + I1 / 3.0
    eig_3 = -2.0 * sqrtQ * np.cos(theta) + I1 / 3.0

    f_tr = 2.0 * shear * ((1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * eig_3) + 2.0 * (lame * sin_phi) * I1 - c_bar
    gamma_sl = (eig_1 - eig_2) / np.maximum(1.0e-15, (1.0 + sin_phi))
    gamma_sr = (eig_2 - eig_3) / np.maximum(1.0e-15, (1.0 - sin_phi))
    gamma_la = (eig_1 + eig_2 - 2.0 * eig_3) / np.maximum(1.0e-15, (3.0 - sin_phi))
    gamma_ra = (2.0 * eig_1 - eig_2 - eig_3) / np.maximum(1.0e-15, (3.0 + sin_phi))

    denom_s = 4.0 * lame * sin_phi**2 + 4.0 * shear * (1.0 + sin_phi**2)
    denom_l = 4.0 * lame * sin_phi**2 + shear * (1.0 + sin_phi) * (1.0 + sin_phi) + 2.0 * shear * (1.0 - sin_phi) * (1.0 - sin_phi)
    denom_r = 4.0 * lame * sin_phi**2 + 2.0 * shear * (1.0 + sin_phi) * (1.0 + sin_phi) + shear * (1.0 - sin_phi) * (1.0 - sin_phi)
    denom_a = 4.0 * bulk * sin_phi**2

    lambda_s = f_tr / np.where(np.abs(denom_s) < 1.0e-15, np.sign(denom_s + 1.0e-15) * 1.0e-15, denom_s)
    lambda_l = (shear * ((1.0 + sin_phi) * (eig_1 + eig_2) - 2.0 * (1.0 - sin_phi) * eig_3) + 2.0 * lame * sin_phi * I1 - c_bar) / np.where(np.abs(denom_l) < 1.0e-15, np.sign(denom_l + 1.0e-15) * 1.0e-15, denom_l)
    lambda_r = (shear * (2.0 * (1.0 + sin_phi) * eig_1 - (1.0 - sin_phi) * (eig_2 + eig_3)) + 2.0 * lame * sin_phi * I1 - c_bar) / np.where(np.abs(denom_r) < 1.0e-15, np.sign(denom_r + 1.0e-15) * 1.0e-15, denom_r)
    lambda_a = (2.0 * bulk * sin_phi * I1 - c_bar) / np.where(np.abs(denom_a) < 1.0e-15, np.sign(denom_a + 1.0e-15) * 1.0e-15, denom_a)

    Psi = np.zeros(n_int, dtype=np.float64)

    test_el = f_tr <= 0
    Psi[test_el] = 0.5 * lame[test_el] * I1[test_el] ** 2 + shear[test_el] * (eig_1[test_el] ** 2 + eig_2[test_el] ** 2 + eig_3[test_el] ** 2)

    test_s = (lambda_s <= np.minimum(gamma_sl, gamma_sr)) & (~test_el)
    Psi[test_s] = (
        0.5 * lame[test_s] * I1[test_s] ** 2
        + shear[test_s] * (eig_1[test_s] ** 2 + eig_2[test_s] ** 2 + eig_3[test_s] ** 2)
        - 0.5 * denom_s[test_s] * (lambda_s[test_s] ** 2)
    )

    test_l = (gamma_sl < gamma_sr) & (lambda_l >= gamma_sl) & (lambda_l <= gamma_la) & (~(test_el | test_s))
    Psi[test_l] = (
        0.5 * lame[test_l] * I1[test_l] ** 2
        + shear[test_l] * (eig_3[test_l] ** 2 + 0.5 * (eig_1[test_l] + eig_2[test_l]) ** 2)
        - 0.5 * denom_l[test_l] * (lambda_l[test_l] ** 2)
    )

    test_r = (gamma_sl > gamma_sr) & (lambda_r >= gamma_sr) & (lambda_r <= gamma_ra) & (~(test_el | test_s))
    Psi[test_r] = (
        0.5 * lame[test_r] * I1[test_r] ** 2
        + shear[test_r] * (eig_1[test_r] ** 2 + 0.5 * (eig_2[test_r] + eig_3[test_r]) ** 2)
        - 0.5 * denom_r[test_r] * (lambda_r[test_r] ** 2)
    )

    test_a = ~(test_el | test_s | test_l | test_r)
    Psi[test_a] = 0.5 * bulk[test_a] * I1[test_a] ** 2 - 0.5 * denom_a[test_a] * (lambda_a[test_a] ** 2)
    return Psi


@dataclass
class ConstitutiveOperator:
    B: Any
    c0: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    Davis_type: str
    shear: np.ndarray
    bulk: np.ndarray
    lame: np.ndarray
    WEIGHT: np.ndarray
    n_strain: int
    n_int: int
    dim: int
    q_mask: np.ndarray | None = None

    def __post_init__(self):
        self.c0 = np.asarray(self.c0, dtype=np.float64)
        self.phi = np.asarray(self.phi, dtype=np.float64)
        self.psi = np.asarray(self.psi, dtype=np.float64)
        self.shear = np.asarray(self.shear, dtype=np.float64)
        self.bulk = np.asarray(self.bulk, dtype=np.float64)
        self.lame = np.asarray(self.lame, dtype=np.float64)
        self.WEIGHT = np.asarray(self.WEIGHT, dtype=np.float64)
        self.q_mask = None if self.q_mask is None else np.asarray(self.q_mask, dtype=bool)
        self.S = None
        self.DS = None
        self.c_bar = None
        self.sin_phi = None

        # runtime records
        self.time_reduction = []
        self.time_stress = []
        self.time_stress_tangent = []
        self.time_build_F = []
        self.time_build_F_K_tangent = []
        self.time_potential = []
        self.time_build_tangent_local = []
        self.time_local_strain = []
        self.time_local_constitutive = []
        self.time_local_constitutive_comm = []
        self.time_local_force_assembly = []
        self.time_local_force_gather = []
        self.owned_tangent_pattern: OwnedTangentPattern | None = None
        self.use_compiled_owned_tangent = True
        self.owned_tangent_kernel = DEFAULT_TANGENT_KERNEL
        self.use_compiled_owned_constitutive = True
        self.owned_constitutive_mode = "global"
        self._owned_local_S = None
        self._owned_local_DS = None
        self._owned_tangent_mat = None
        self._owned_tangent_indptr = None
        self._owned_tangent_indices = None
        self._owned_tangent_values = None
        self._owned_regularized_mat = None
        self._owned_regularized_indptr = None
        self._owned_regularized_indices = None
        self._owned_regularized_values = None
        self._owned_overlap_c_bar = None
        self._owned_overlap_sin_phi = None
        self._owned_unique_c_bar = None
        self._owned_unique_sin_phi = None

        # Precompute sparse assembly helpers.
        aux = np.arange(self.n_strain * self.n_int).reshape(self.n_strain, self.n_int, order="F")
        self.AUX = aux
        self.iD = np.tile(aux, (self.n_strain, 1))
        self.jD = np.kron(aux, np.ones((self.n_strain, 1), dtype=np.int64))
        self.vD_pre = np.repeat(self.WEIGHT[np.newaxis, :], self.n_strain * self.n_strain, axis=0).ravel(order="F")

    def _extract_free_rows(self, field: np.ndarray) -> np.ndarray:
        flat = np.asarray(field, dtype=np.float64).reshape(-1, order="F")
        if self.q_mask is None:
            return flat
        return flat[q_to_free_indices(self.q_mask)]

    def set_owned_tangent_pattern(
        self,
        pattern: OwnedTangentPattern,
        *,
        use_compiled: bool = True,
        tangent_kernel: str = DEFAULT_TANGENT_KERNEL,
        constitutive_mode: str = "global",
        use_compiled_constitutive: bool = True,
    ) -> None:
        self.owned_tangent_pattern = pattern
        self.use_compiled_owned_tangent = bool(use_compiled)
        self.owned_tangent_kernel = str(tangent_kernel).lower()
        self.use_compiled_owned_constitutive = bool(use_compiled_constitutive)
        self.owned_constitutive_mode = str(constitutive_mode).lower()
        self.release_petsc_caches()

    def _local_comm(self):
        if PETSc is not None:
            return PETSc.COMM_WORLD
        if PYMPI is not None:
            return PYMPI.COMM_WORLD
        return None

    def _clear_owned_local_cache(self) -> None:
        self._owned_local_S = None
        self._owned_local_DS = None

    def _clear_owned_reduction_cache(self) -> None:
        self._owned_overlap_c_bar = None
        self._owned_overlap_sin_phi = None
        self._owned_unique_c_bar = None
        self._owned_unique_sin_phi = None

    def _use_owned_constitutive(self) -> bool:
        return (
            self.owned_tangent_pattern is not None
            and str(self.owned_constitutive_mode).lower() != "global"
        )

    def _evaluate_owned_overlap_constitutive(self, U, *, return_tangent: bool) -> None:
        pattern = self.owned_tangent_pattern
        if pattern is None:
            raise ValueError("Owned tangent pattern not configured")

        t0 = perf_counter()
        E_local = assemble_overlap_strain(
            pattern,
            U,
            use_compiled=self.use_compiled_owned_constitutive,
        )
        self.time_local_strain.append(perf_counter() - t0)

        local_idx = np.asarray(pattern.local_int_indices, dtype=np.int64)
        t1 = perf_counter()
        c_bar_local = self._owned_overlap_c_bar
        sin_phi_local = self._owned_overlap_sin_phi
        if c_bar_local is None or sin_phi_local is None:
            if self.c_bar is None or self.sin_phi is None:
                raise ValueError("Material reduction not set. Call reduction(lambda) first.")
            c_bar_local = np.asarray(self.c_bar[local_idx], dtype=np.float64)
            sin_phi_local = np.asarray(self.sin_phi[local_idx], dtype=np.float64)
        if return_tangent:
            S_local, DS_local = _batch_constitutive_problem_local(
                E_local,
                c_bar_local,
                sin_phi_local,
                self.shear[local_idx],
                self.bulk[local_idx],
                self.lame[local_idx],
                dim=self.dim,
                return_tangent=True,
                use_compiled=self.use_compiled_owned_constitutive,
            )
            self._owned_local_DS = np.asarray(DS_local, dtype=np.float64)
        else:
            S_local = _batch_constitutive_problem_local(
                E_local,
                c_bar_local,
                sin_phi_local,
                self.shear[local_idx],
                self.bulk[local_idx],
                self.lame[local_idx],
                dim=self.dim,
                return_tangent=False,
                use_compiled=self.use_compiled_owned_constitutive,
            )
            self._owned_local_DS = None
        self._owned_local_S = np.asarray(S_local, dtype=np.float64)
        self.time_local_constitutive.append(perf_counter() - t1)
        self.S = None
        self.DS = None

    def _evaluate_owned_unique_gather_constitutive(self, U, *, return_tangent: bool) -> None:
        pattern = self.owned_tangent_pattern
        if pattern is None:
            raise ValueError("Owned tangent pattern not configured")

        n_local = int(np.asarray(pattern.unique_local_int_indices, dtype=np.int64).size)
        t0 = perf_counter()
        if n_local:
            u_flat = np.asarray(U, dtype=np.float64).reshape(-1, order="F")
            u_unique = u_flat[np.asarray(pattern.unique_global_dofs, dtype=np.int64)]
            E_unique = pattern.unique_B @ u_unique
            E_unique = np.asarray(E_unique, dtype=np.float64).reshape(self.n_strain, -1, order="F")
        else:
            E_unique = np.empty((self.n_strain, 0), dtype=np.float64)
        self.time_local_strain.append(perf_counter() - t0)

        unique_idx = np.asarray(pattern.unique_local_int_indices, dtype=np.int64)
        t1 = perf_counter()
        c_bar_unique = self._owned_unique_c_bar
        sin_phi_unique = self._owned_unique_sin_phi
        if c_bar_unique is None or sin_phi_unique is None:
            if self.c_bar is None or self.sin_phi is None:
                raise ValueError("Material reduction not set. Call reduction(lambda) first.")
            c_bar_unique = np.asarray(self.c_bar[unique_idx], dtype=np.float64)
            sin_phi_unique = np.asarray(self.sin_phi[unique_idx], dtype=np.float64)
        if return_tangent:
            S_unique, DS_unique = _batch_constitutive_problem_local(
                E_unique,
                c_bar_unique,
                sin_phi_unique,
                self.shear[unique_idx],
                self.bulk[unique_idx],
                self.lame[unique_idx],
                dim=self.dim,
                return_tangent=True,
                use_compiled=self.use_compiled_owned_constitutive,
            )
        else:
            S_unique = _batch_constitutive_problem_local(
                E_unique,
                c_bar_unique,
                sin_phi_unique,
                self.shear[unique_idx],
                self.bulk[unique_idx],
                self.lame[unique_idx],
                dim=self.dim,
                return_tangent=False,
                use_compiled=self.use_compiled_owned_constitutive,
            )
            DS_unique = None
        self.time_local_constitutive.append(perf_counter() - t1)

        t2 = perf_counter()
        comm = self._local_comm()
        parts = [(unique_idx, np.asarray(S_unique, dtype=np.float64))]
        if return_tangent:
            parts = [(unique_idx, np.asarray(S_unique, dtype=np.float64), np.asarray(DS_unique, dtype=np.float64))]
        mpi_comm = comm.tompi4py() if comm is not None and hasattr(comm, "tompi4py") else comm
        if mpi_comm is None:
            gathered_parts = parts
        else:
            size = int(mpi_comm.getSize()) if hasattr(mpi_comm, "getSize") else int(mpi_comm.Get_size())
            if size == 1:
                gathered_parts = parts
            else:
                gathered_parts = mpi_comm.allgather(parts[0])

        S_global = np.zeros((self.n_strain, self.n_int), dtype=np.float64)
        DS_global = np.zeros((self.n_strain * self.n_strain, self.n_int), dtype=np.float64) if return_tangent else None
        for part in gathered_parts:
            idx = np.asarray(part[0], dtype=np.int64)
            if idx.size == 0:
                continue
            S_global[:, idx] = np.asarray(part[1], dtype=np.float64)
            if return_tangent and DS_global is not None:
                DS_global[:, idx] = np.asarray(part[2], dtype=np.float64)
        self.time_local_constitutive_comm.append(perf_counter() - t2)

        local_idx = np.asarray(pattern.local_int_indices, dtype=np.int64)
        self._owned_local_S = S_global[:, local_idx]
        self._owned_local_DS = DS_global[:, local_idx] if DS_global is not None else None
        self.S = S_global
        self.DS = DS_global

    def _evaluate_owned_unique_exchange_constitutive(self, U, *, return_tangent: bool) -> None:
        pattern = self.owned_tangent_pattern
        if pattern is None:
            raise ValueError("Owned tangent pattern not configured")

        n_local = int(np.asarray(pattern.unique_local_int_indices, dtype=np.int64).size)
        t0 = perf_counter()
        if n_local:
            u_flat = np.asarray(U, dtype=np.float64).reshape(-1, order="F")
            u_unique = u_flat[np.asarray(pattern.unique_global_dofs, dtype=np.int64)]
            E_unique = pattern.unique_B @ u_unique
            E_unique = np.asarray(E_unique, dtype=np.float64).reshape(self.n_strain, -1, order="F")
        else:
            E_unique = np.empty((self.n_strain, 0), dtype=np.float64)
        self.time_local_strain.append(perf_counter() - t0)

        unique_idx = np.asarray(pattern.unique_local_int_indices, dtype=np.int64)
        t1 = perf_counter()
        c_bar_unique = self._owned_unique_c_bar
        sin_phi_unique = self._owned_unique_sin_phi
        if c_bar_unique is None or sin_phi_unique is None:
            if self.c_bar is None or self.sin_phi is None:
                raise ValueError("Material reduction not set. Call reduction(lambda) first.")
            c_bar_unique = np.asarray(self.c_bar[unique_idx], dtype=np.float64)
            sin_phi_unique = np.asarray(self.sin_phi[unique_idx], dtype=np.float64)
        if return_tangent:
            S_unique, DS_unique = _batch_constitutive_problem_local(
                E_unique,
                c_bar_unique,
                sin_phi_unique,
                self.shear[unique_idx],
                self.bulk[unique_idx],
                self.lame[unique_idx],
                dim=self.dim,
                return_tangent=True,
                use_compiled=self.use_compiled_owned_constitutive,
            )
            DS_unique = np.asarray(DS_unique, dtype=np.float64)
        else:
            S_unique = _batch_constitutive_problem_local(
                E_unique,
                c_bar_unique,
                sin_phi_unique,
                self.shear[unique_idx],
                self.bulk[unique_idx],
                self.lame[unique_idx],
                dim=self.dim,
                return_tangent=False,
                use_compiled=self.use_compiled_owned_constitutive,
            )
            DS_unique = None
        self.time_local_constitutive.append(perf_counter() - t1)

        t2 = perf_counter()
        S_local, DS_local = _exchange_owned_constitutive_overlap(
            pattern,
            np.asarray(S_unique, dtype=np.float64),
            DS_unique,
            return_tangent=return_tangent,
            comm=self._local_comm(),
        )
        self.time_local_constitutive_comm.append(perf_counter() - t2)

        self._owned_local_S = np.asarray(S_local, dtype=np.float64)
        self._owned_local_DS = np.asarray(DS_local, dtype=np.float64) if DS_local is not None else None
        self.S = None
        self.DS = None

    def _evaluate_owned_constitutive(self, U, *, return_tangent: bool) -> None:
        mode = str(self.owned_constitutive_mode).lower()
        self._clear_owned_local_cache()
        if mode == "overlap":
            self._evaluate_owned_overlap_constitutive(U, return_tangent=return_tangent)
            return
        if mode == "unique_exchange":
            self._evaluate_owned_unique_exchange_constitutive(U, return_tangent=return_tangent)
            return
        if mode in {"unique", "unique_gather", "no_overlap", "partitioned"}:
            self._evaluate_owned_unique_gather_constitutive(U, return_tangent=return_tangent)
            return
        raise ValueError(f"Unsupported owned constitutive mode {self.owned_constitutive_mode!r}")

    def _build_owned_tangent_matrix(self):
        if self.owned_tangent_pattern is None:
            raise ValueError("Owned tangent pattern not configured")
        if PETSc is None:
            raise RuntimeError("PETSc is required for the owned tangent matrix path")

        t0 = perf_counter()
        pattern = self.owned_tangent_pattern
        tang = assemble_owned_tangent_values(
            pattern,
            self._owned_local_DS if self._owned_local_DS is not None else self.DS,
            use_compiled=self.use_compiled_owned_tangent,
            kernel=self.owned_tangent_kernel,
        )
        tang = np.asarray(tang, dtype=np.float64)
        if self._owned_tangent_values is None or self._owned_tangent_values.shape != tang.shape:
            self._owned_tangent_values = np.empty_like(tang)
        values = self._owned_tangent_values
        np.copyto(values, tang)
        self.time_build_tangent_local.append(perf_counter() - t0)

        if self._owned_tangent_indptr is None:
            self._owned_tangent_indptr = np.array(pattern.local_matrix_pattern.indptr, dtype=PETSc.IntType, copy=True)
        if self._owned_tangent_indices is None:
            self._owned_tangent_indices = np.array(pattern.local_matrix_pattern.indices, dtype=PETSc.IntType, copy=True)

        if self._owned_tangent_mat is None:
            global_size = int(pattern.local_matrix_pattern.shape[1])
            from scipy.sparse import csr_matrix

            local_matrix = csr_matrix(
                (np.array(values, dtype=np.float64, copy=True), self._owned_tangent_indices, self._owned_tangent_indptr),
                shape=pattern.local_matrix_pattern.shape,
            )
            self._owned_tangent_mat = local_csr_to_petsc_aij_matrix(
                local_matrix,
                global_shape=(global_size, global_size),
                comm=PETSc.COMM_WORLD,
                block_size=self.dim,
            )
        else:
            update_petsc_aij_matrix_csr(
                self._owned_tangent_mat,
                indptr=self._owned_tangent_indptr,
                indices=self._owned_tangent_indices,
                data=values,
            )
        return self._owned_tangent_mat

    def _build_owned_regularized_matrix(self, r: float):
        if self.owned_tangent_pattern is None:
            raise ValueError("Owned tangent pattern not configured")
        if PETSc is None:
            raise RuntimeError("PETSc is required for the owned regularized matrix path")

        t0 = perf_counter()
        pattern = self.owned_tangent_pattern
        tang = assemble_owned_tangent_values(
            pattern,
            self._owned_local_DS if self._owned_local_DS is not None else self.DS,
            use_compiled=self.use_compiled_owned_tangent,
            kernel=self.owned_tangent_kernel,
        )
        tang = np.asarray(tang, dtype=np.float64)
        if self._owned_regularized_values is None or self._owned_regularized_values.shape != tang.shape:
            self._owned_regularized_values = np.empty_like(tang)
        values = self._owned_regularized_values
        np.copyto(values, tang)
        values *= 1.0 - float(r)
        values += float(r) * np.asarray(pattern.elastic_values, dtype=np.float64)
        self.time_build_tangent_local.append(perf_counter() - t0)

        if self._owned_regularized_indptr is None:
            self._owned_regularized_indptr = np.array(pattern.local_matrix_pattern.indptr, dtype=PETSc.IntType, copy=True)
        if self._owned_regularized_indices is None:
            self._owned_regularized_indices = np.array(pattern.local_matrix_pattern.indices, dtype=PETSc.IntType, copy=True)

        if self._owned_regularized_mat is None:
            global_size = int(pattern.local_matrix_pattern.shape[1])
            from scipy.sparse import csr_matrix

            local_matrix = csr_matrix(
                (np.array(values, dtype=np.float64, copy=True), self._owned_regularized_indices, self._owned_regularized_indptr),
                shape=pattern.local_matrix_pattern.shape,
            )
            self._owned_regularized_mat = local_csr_to_petsc_aij_matrix(
                local_matrix,
                global_shape=(global_size, global_size),
                comm=PETSc.COMM_WORLD,
                block_size=self.dim,
            )
        else:
            update_petsc_aij_matrix_csr(
                self._owned_regularized_mat,
                indptr=self._owned_regularized_indptr,
                indices=self._owned_regularized_indices,
                data=values,
            )
        return self._owned_regularized_mat

    def _strain(self, U):
        if self.B is None:
            raise RuntimeError("Global strain operator B is not available for this constitutive path")
        U = np.asarray(U, dtype=np.float64)
        if U.shape[0] != self.dim:
            raise ValueError("U shape first dimension must match dim")
        return (self.B @ U.ravel(order="F"))[: self.n_strain * self.n_int].reshape(self.n_strain, self.n_int, order="F")

    def reduction(self, lam: float):
        t0 = perf_counter()
        self._clear_owned_reduction_cache()
        if self._use_owned_constitutive():
            pattern = self.owned_tangent_pattern
            if pattern is None:
                raise ValueError("Owned tangent pattern not configured")
            local_idx = np.asarray(pattern.local_int_indices, dtype=np.int64)
            self._owned_overlap_c_bar, self._owned_overlap_sin_phi = reduction(
                self.c0[local_idx],
                self.phi[local_idx],
                self.psi[local_idx],
                lam,
                self.Davis_type,
            )
            unique_idx = np.asarray(pattern.unique_local_int_indices, dtype=np.int64)
            if unique_idx.size:
                self._owned_unique_c_bar, self._owned_unique_sin_phi = reduction(
                    self.c0[unique_idx],
                    self.phi[unique_idx],
                    self.psi[unique_idx],
                    lam,
                    self.Davis_type,
                )
            else:
                self._owned_unique_c_bar = np.empty(0, dtype=np.float64)
                self._owned_unique_sin_phi = np.empty(0, dtype=np.float64)
            self.c_bar = None
            self.sin_phi = None
        else:
            self.c_bar, self.sin_phi = reduction(self.c0, self.phi, self.psi, lam, self.Davis_type)
        self.time_reduction.append(perf_counter() - t0)

    def constitutive_problem_stress(self, U):
        t0 = perf_counter()
        if (self.c_bar is None or self.sin_phi is None) and not self._use_owned_constitutive():
            raise ValueError("Material reduction not set. Call reduction(lambda) first.")
        if self._use_owned_constitutive():
            self._evaluate_owned_constitutive(U, return_tangent=False)
        else:
            E = self._strain(U)
            if self.dim == 2:
                self.S = constitutive_problem_2D(E, self.c_bar, self.sin_phi, self.shear, self.bulk, self.lame, return_tangent=False)
            elif self.dim == 3:
                self.S = constitutive_problem_3D(E, self.c_bar, self.sin_phi, self.shear, self.bulk, self.lame, return_tangent=False)
            else:
                raise ValueError("wrong dimension")
            self._clear_owned_local_cache()
        self.time_stress.append(perf_counter() - t0)
        return self.S if self.S is not None else self._owned_local_S

    def constitutive_problem_stress_tangent(self, U):
        t0 = perf_counter()
        if (self.c_bar is None or self.sin_phi is None) and not self._use_owned_constitutive():
            raise ValueError("Material reduction not set. Call reduction(lambda) first.")
        if self._use_owned_constitutive():
            self._evaluate_owned_constitutive(U, return_tangent=True)
        elif self.dim == 2:
            E = self._strain(U)
            self.S, self.DS = constitutive_problem_2D(E, self.c_bar, self.sin_phi, self.shear, self.bulk, self.lame, return_tangent=True)
            self._clear_owned_local_cache()
        elif self.dim == 3:
            E = self._strain(U)
            self.S, self.DS = constitutive_problem_3D(E, self.c_bar, self.sin_phi, self.shear, self.bulk, self.lame, return_tangent=True)
            self._clear_owned_local_cache()
        else:
            raise ValueError("wrong dimension")
        self.time_stress_tangent.append(perf_counter() - t0)
        return self.S if self.S is not None else self._owned_local_S

    def build_F(self):
        if self.S is None and self._owned_local_S is None:
            raise ValueError("Stress tensor not computed")
        t0 = perf_counter()
        if self._owned_local_S is not None and self.owned_tangent_pattern is not None:
            t_force = perf_counter()
            owned_force = _owned_force_from_local_stress(
                self.owned_tangent_pattern,
                self._owned_local_S[: self.n_strain],
                use_compiled=self.use_compiled_owned_tangent,
            )
            self.time_local_force_assembly.append(perf_counter() - t_force)
            t_gather = perf_counter()
            global_size = int(self.owned_tangent_pattern.local_matrix_pattern.shape[1])
            Ft = _gather_owned_rows(owned_force, global_size, self._local_comm())
            self.time_local_force_gather.append(perf_counter() - t_gather)
        else:
            Ft = self.B.T.dot((self.WEIGHT * self.S[: self.n_strain]).reshape(-1, order="F"))
        F = np.asarray(Ft, dtype=np.float64).reshape(self.dim, -1, order="F")
        self.time_build_F.append(perf_counter() - t0)
        return F

    def build_F_local(self):
        if self.S is None and self._owned_local_S is None:
            raise ValueError("Stress tensor not computed")
        if self._owned_local_S is not None and self.owned_tangent_pattern is not None:
            t0 = perf_counter()
            owned_force = _owned_force_from_local_stress(
                self.owned_tangent_pattern,
                self._owned_local_S[: self.n_strain],
                use_compiled=self.use_compiled_owned_tangent,
            )
            self.time_local_force_assembly.append(perf_counter() - t0)
            self.time_build_F.append(perf_counter() - t0)
            return np.asarray(owned_force, dtype=np.float64)
        return self.build_F().reshape(-1, order="F")

    def build_F_free(self):
        if self.S is None and self._owned_local_S is None:
            raise ValueError("Stress tensor not computed")
        t0 = perf_counter()
        if self._owned_local_S is not None and self.owned_tangent_pattern is not None:
            t_force = perf_counter()
            owned_force_free = _owned_free_force_from_local_stress(
                self.owned_tangent_pattern,
                self._owned_local_S[: self.n_strain],
                use_compiled=self.use_compiled_owned_tangent,
            )
            self.time_local_force_assembly.append(perf_counter() - t_force)
            t_gather = perf_counter()
            F_free = _gather_owned_free_rows(
                owned_force_free,
                int(self.owned_tangent_pattern.global_free_size),
                self._local_comm(),
            )
            self.time_local_force_gather.append(perf_counter() - t_gather)
            self.time_build_F.append(perf_counter() - t0)
            return np.asarray(F_free, dtype=np.float64)
        return self._extract_free_rows(self.build_F())

    def build_F_free_local(self):
        if self.S is None and self._owned_local_S is None:
            raise ValueError("Stress tensor not computed")
        if self._owned_local_S is not None and self.owned_tangent_pattern is not None:
            t0 = perf_counter()
            owned_force_free = _owned_free_force_from_local_stress(
                self.owned_tangent_pattern,
                self._owned_local_S[: self.n_strain],
                use_compiled=self.use_compiled_owned_tangent,
            )
            self.time_local_force_assembly.append(perf_counter() - t0)
            self.time_build_F.append(perf_counter() - t0)
            return np.asarray(owned_force_free, dtype=np.float64)
        return self.build_F_free()

    def build_F_K_tangent(self):
        if self.DS is None and self._owned_local_DS is None:
            raise ValueError("Tangent DS not computed")
        t0 = perf_counter()
        F = self.build_F()
        if self.owned_tangent_pattern is not None:
            K_tangent = self._build_owned_tangent_matrix()
        else:
            vD = self.vD_pre * self.DS.ravel(order="F")
            D = csc_matrix(
                (vD, (self.iD.ravel(order="F"), self.jD.ravel(order="F"))),
                shape=(self.n_strain * self.n_int, self.n_strain * self.n_int),
            )
            K_tangent = self.B.T @ D @ self.B
            K_tangent = (K_tangent + K_tangent.T) * 0.5
        self.time_build_F_K_tangent.append(perf_counter() - t0)
        return F, K_tangent

    def build_F_all(self, lam: float, U):
        self.reduction(lam)
        self.constitutive_problem_stress(U)
        return self.build_F()

    def build_F_all_local(self, lam: float, U):
        self.reduction(lam)
        self.constitutive_problem_stress(U)
        return self.build_F_local()

    def build_F_all_free(self, lam: float, U):
        self.reduction(lam)
        self.constitutive_problem_stress(U)
        return self.build_F_free()

    def build_F_all_free_local(self, lam: float, U):
        self.reduction(lam)
        self.constitutive_problem_stress(U)
        return self.build_F_free_local()

    def build_F_K_tangent_all(self, lam: float, U):
        self.reduction(lam)
        self.constitutive_problem_stress_tangent(U)
        return self.build_F_K_tangent()

    def build_F_K_tangent_all_free(self, lam: float, U):
        self.reduction(lam)
        self.constitutive_problem_stress_tangent(U)
        F_free = self.build_F_free()
        if self.owned_tangent_pattern is not None:
            K_tangent = self._build_owned_tangent_matrix()
        else:
            _F_full, K_tangent = self.build_F_K_tangent()
        return F_free, K_tangent

    def build_F_reduced(self, U):
        self.constitutive_problem_stress(U)
        return self.build_F()

    def build_F_reduced_local(self, U):
        self.constitutive_problem_stress(U)
        return self.build_F_local()

    def build_F_reduced_free(self, U):
        self.constitutive_problem_stress(U)
        return self.build_F_free()

    def build_F_reduced_free_local(self, U):
        self.constitutive_problem_stress(U)
        return self.build_F_free_local()

    def build_F_K_tangent_reduced(self, U):
        self.constitutive_problem_stress_tangent(U)
        return self.build_F_K_tangent()

    def build_F_K_tangent_reduced_free(self, U):
        self.constitutive_problem_stress_tangent(U)
        F_free = self.build_F_free()
        if self.owned_tangent_pattern is not None:
            K_tangent = self._build_owned_tangent_matrix()
        else:
            _F_full, K_tangent = self.build_F_K_tangent()
        return F_free, K_tangent

    def build_K_regularized(self, r: float):
        if self.owned_tangent_pattern is None:
            raise RuntimeError("Regularized in-place matrix path requires owned_tangent_pattern")
        return self._build_owned_regularized_matrix(r)

    def build_F_K_regularized_all(self, lam: float, U, r: float):
        self.reduction(lam)
        self.constitutive_problem_stress_tangent(U)
        t0 = perf_counter()
        F = self.build_F()
        K_r = self.build_K_regularized(r)
        self.time_build_F_K_tangent.append(perf_counter() - t0)
        return F, K_r

    def build_F_K_regularized_all_free(self, lam: float, U, r: float):
        self.reduction(lam)
        self.constitutive_problem_stress_tangent(U)
        t0 = perf_counter()
        F_free = self.build_F_free()
        K_r = self.build_K_regularized(r)
        self.time_build_F_K_tangent.append(perf_counter() - t0)
        return F_free, K_r

    def build_F_K_regularized_reduced(self, U, r: float):
        self.constitutive_problem_stress_tangent(U)
        t0 = perf_counter()
        F = self.build_F()
        K_r = self.build_K_regularized(r)
        self.time_build_F_K_tangent.append(perf_counter() - t0)
        return F, K_r

    def build_F_K_regularized_reduced_free(self, U, r: float):
        self.constitutive_problem_stress_tangent(U)
        t0 = perf_counter()
        F_free = self.build_F_free()
        K_r = self.build_K_regularized(r)
        self.time_build_F_K_tangent.append(perf_counter() - t0)
        return F_free, K_r

    def potential_energy(self, U):
        t0 = perf_counter()
        E = self._strain(U)
        if self.dim == 2:
            Psi = potential_2D(E, self.c_bar, self.sin_phi, self.shear, self.bulk, self.lame)
        else:
            Psi = potential_3D(E, self.c_bar, self.sin_phi, self.shear, self.bulk, self.lame)
        Psi_integrated = float(np.dot(self.WEIGHT, Psi))
        self.time_potential.append(perf_counter() - t0)
        return Psi_integrated

    def get_total_time(self):
        return {
            "reduction": sum(self.time_reduction),
            "stress": sum(self.time_stress),
            "stress_tangent": sum(self.time_stress_tangent),
            "build_F": sum(self.time_build_F),
            "build_F_K_tangent": sum(self.time_build_F_K_tangent),
            "build_tangent_local": sum(self.time_build_tangent_local),
            "local_strain": sum(self.time_local_strain),
            "local_constitutive": sum(self.time_local_constitutive),
            "local_constitutive_comm": sum(self.time_local_constitutive_comm),
            "local_force_assembly": sum(self.time_local_force_assembly),
            "local_force_gather": sum(self.time_local_force_gather),
            "potential": sum(self.time_potential),
        }

    def release_petsc_caches(self) -> None:
        if PETSc is not None and self._owned_tangent_mat is not None:
            release_petsc_aij_matrix(self._owned_tangent_mat)
            self._owned_tangent_mat.destroy()
        self._owned_tangent_mat = None
        self._owned_tangent_indptr = None
        self._owned_tangent_indices = None
        self._owned_tangent_values = None
        if PETSc is not None and self._owned_regularized_mat is not None:
            release_petsc_aij_matrix(self._owned_regularized_mat)
            self._owned_regularized_mat.destroy()
        self._owned_regularized_mat = None
        self._owned_regularized_indptr = None
        self._owned_regularized_indices = None
        self._owned_regularized_values = None
