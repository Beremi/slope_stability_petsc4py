# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

"""Small Cython kernel primitives for inner-loop vector ops."""

from libc.math cimport sqrt
from libc.stdint cimport int64_t
import numpy as np
cimport numpy as cnp
from mpi4py import MPI as PYMPI

cdef extern from "assemble_tangent_values_3d.h":
    void assemble_tangent_values_3d_p2_c(
        const double *dphi1,
        const double *dphi2,
        const double *dphi3,
        const double *ds,
        const double *weight,
        const int64_t *scatter_map,
        int n_int,
        int n_elem,
        int n_p,
        int n_q,
        int nnz_out,
        double *out_values
    )

cdef extern from "constitutive_3d_batch.h":
    void constitutive_problem_3d_s_batch_c(
        const double *E,
        const double *c_bar,
        const double *sin_phi,
        const double *shear,
        const double *bulk,
        const double *lame,
        int n_int,
        double *S_out
    )
    void constitutive_problem_3d_sds_batch_c(
        const double *E,
        const double *c_bar,
        const double *sin_phi,
        const double *shear,
        const double *bulk,
        const double *lame,
        int n_int,
        double *S_out,
        double *DS_out
    )


cpdef double dot(double[:] x, double[:] y):
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i
    cdef double acc = 0.0
    for i in range(n):
        acc += x[i] * y[i]
    return acc


cpdef double norm2(double[:] x):
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i
    cdef double acc = 0.0
    for i in range(n):
        acc += x[i] * x[i]
    return sqrt(acc)


cdef double _dot_view(double[:] x, double[:] y) noexcept:
    cdef Py_ssize_t i
    cdef double acc = 0.0
    for i in range(x.shape[0]):
        acc += x[i] * y[i]
    return acc


def dfgmres_matlab_exact_distributed_compiled(
    object matvec_local,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] b_local,
    object prec_local,
    object W_local_obj,
    int maxits,
    double tol,
    object comm,
    dict stats=None,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] W_local
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] x_local
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] r0_local
    cdef cnp.ndarray[cnp.float64_t, ndim=2] V
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Wbasis
    cdef cnp.ndarray[cnp.float64_t, ndim=2] H
    cdef cnp.ndarray[cnp.float64_t, ndim=1] g
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y
    cdef cnp.ndarray[cnp.float64_t, ndim=1] w_local
    cdef cnp.ndarray[cnp.float64_t, ndim=1] u_local
    cdef cnp.ndarray[cnp.float64_t, ndim=1] coeff_local
    cdef cnp.ndarray[cnp.float64_t, ndim=1] coeff
    cdef cnp.ndarray[cnp.float64_t, ndim=1] Avec_local
    cdef cnp.ndarray[cnp.float64_t, ndim=2] subH
    cdef cnp.ndarray[cnp.float64_t, ndim=1] subg
    cdef Py_ssize_t i, j, k, n_local, n_basis
    cdef double b_norm, res_norm, t0, acc
    cdef int iters = 0

    if maxits <= 0:
        return np.zeros(0, dtype=np.float64), 0, np.zeros(0, dtype=np.float64)

    n_local = b_local.shape[0]
    x_local = np.zeros(n_local, dtype=np.float64)

    if W_local_obj is None:
        W_local = np.zeros((n_local, 0), dtype=np.float64)
    else:
        W_local = np.asarray(W_local_obj, dtype=np.float64)
        if W_local.ndim == 1:
            W_local = W_local.reshape((W_local.shape[0], 1))
        if W_local.shape[0] != n_local:
            raise ValueError("Local deflation basis row count must match local rhs size")
    n_basis = W_local.shape[1]

    if n_basis > 0:
        coeff_local = np.zeros(n_basis, dtype=np.float64)
        for j in range(n_basis):
            coeff_local[j] = _dot_view(W_local[:, j], b_local)
        t0 = PYMPI.Wtime() if stats is not None else 0.0
        coeff = np.asarray(comm.allreduce(np.asarray(coeff_local), op=PYMPI.SUM), dtype=np.float64)
        if stats is not None:
            stats["coarse_solve_s"] = stats.get("coarse_solve_s", 0.0) + (PYMPI.Wtime() - t0)
        for i in range(n_local):
            acc = 0.0
            for j in range(n_basis):
                acc += W_local[i, j] * coeff[j]
            x_local[i] = acc

    t0 = PYMPI.Wtime() if stats is not None else 0.0
    r0_local = b_local - np.asarray(matvec_local(np.asarray(x_local)), dtype=np.float64)
    if stats is not None:
        stats["matvec_s"] = stats.get("matvec_s", 0.0) + (PYMPI.Wtime() - t0)

    b_norm = <double> comm.allreduce(_dot_view(b_local, b_local), op=PYMPI.SUM)
    b_norm = sqrt(b_norm) if b_norm > 0.0 else 1.0
    res_norm = <double> comm.allreduce(_dot_view(r0_local, r0_local), op=PYMPI.SUM)
    res_norm = sqrt(res_norm)

    cdef cnp.ndarray[cnp.float64_t, ndim=1] res_hist = np.zeros(maxits + 1, dtype=np.float64)
    res_hist[0] = res_norm / b_norm
    if res_hist[0] < tol:
        return np.asarray(x_local), 0, np.asarray(res_hist[:1])

    V = np.zeros((n_local, maxits + 1), dtype=np.float64)
    H = np.zeros((maxits + 1, maxits), dtype=np.float64)
    Wbasis = np.zeros((n_local, maxits), dtype=np.float64)
    g = np.zeros(maxits + 1, dtype=np.float64)
    y = np.zeros(1, dtype=np.float64)

    for i in range(n_local):
        V[i, 0] = r0_local[i] / res_norm
    g[0] = res_norm

    for j in range(maxits):
        t0 = PYMPI.Wtime() if stats is not None else 0.0
        w_local = np.asarray(prec_local(np.asarray(V[:, j])), dtype=np.float64)
        if stats is not None:
            stats["preconditioner_apply_s"] = stats.get("preconditioner_apply_s", 0.0) + (PYMPI.Wtime() - t0)

        if n_basis > 0:
            t0 = PYMPI.Wtime() if stats is not None else 0.0
            Avec_local = np.asarray(matvec_local(np.asarray(w_local)), dtype=np.float64)
            if stats is not None:
                stats["matvec_s"] = stats.get("matvec_s", 0.0) + (PYMPI.Wtime() - t0)
            coeff_local = np.zeros(n_basis, dtype=np.float64)
            for k in range(n_basis):
                coeff_local[k] = _dot_view(W_local[:, k], Avec_local)
            t0 = PYMPI.Wtime() if stats is not None else 0.0
            coeff = np.asarray(comm.allreduce(np.asarray(coeff_local), op=PYMPI.SUM), dtype=np.float64)
            if stats is not None:
                stats["projection_s"] = stats.get("projection_s", 0.0) + (PYMPI.Wtime() - t0)
            for i in range(n_local):
                acc = 0.0
                for k in range(n_basis):
                    acc += W_local[i, k] * coeff[k]
                w_local[i] -= acc

        t0 = PYMPI.Wtime() if stats is not None else 0.0
        u_local = np.asarray(matvec_local(np.asarray(w_local)), dtype=np.float64)
        if stats is not None:
            stats["matvec_s"] = stats.get("matvec_s", 0.0) + (PYMPI.Wtime() - t0)

        for i in range(j + 1):
            H[i, j] = <double> comm.allreduce(_dot_view(V[:, i], u_local), op=PYMPI.SUM)
            u_local = u_local - H[i, j] * np.asarray(V[:, i])

        H[j + 1, j] = sqrt(max(<double> comm.allreduce(_dot_view(u_local, u_local), op=PYMPI.SUM), 0.0))
        Wbasis[:, j] = w_local

        if H[j + 1, j] < 1.0e-14:
            iters = j + 1
            break

        for i in range(n_local):
            V[i, j + 1] = u_local[i] / H[j + 1, j]

        subH = np.asarray(H[: j + 2, : j + 1])
        subg = np.asarray(g[: j + 2])
        t0 = PYMPI.Wtime() if stats is not None else 0.0
        y, *_ = np.linalg.lstsq(subH, subg, rcond=None)
        if stats is not None:
            stats["least_squares_s"] = stats.get("least_squares_s", 0.0) + (PYMPI.Wtime() - t0)
        res_norm = norm2(subg - subH @ y)
        res_hist[j + 1] = res_norm / b_norm
        iters = j + 1
        if res_hist[j + 1] < tol:
            break

    if iters > 0 and y.shape[0] != iters:
        subH = np.asarray(H[: iters + 1, :iters])
        subg = np.asarray(g[: iters + 1])
        t0 = PYMPI.Wtime() if stats is not None else 0.0
        y, *_ = np.linalg.lstsq(subH, subg, rcond=None)
        if stats is not None:
            stats["least_squares_s"] = stats.get("least_squares_s", 0.0) + (PYMPI.Wtime() - t0)

    if iters > 0:
        x_local = x_local + np.asarray(Wbasis[:, :iters]) @ np.asarray(y)
    return np.asarray(x_local), iters, np.asarray(res_hist[: iters + 1])


def assemble_tangent_values_3d(
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] dphi1,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] dphi2,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] dphi3,
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] ds,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] weight,
    cnp.ndarray[cnp.int64_t, ndim=2, mode="c"] scatter_map,
    int nnz_out,
):
    cdef int n_int = <int>dphi1.shape[0]
    cdef int n_p = <int>dphi1.shape[1]
    cdef int n_elem = <int>scatter_map.shape[0]
    cdef int n_q
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out

    if dphi2.shape[0] != n_int or dphi2.shape[1] != n_p or dphi3.shape[0] != n_int or dphi3.shape[1] != n_p:
        raise ValueError("dphi arrays must have matching shapes")
    if ds.shape[0] != n_int or ds.shape[1] != 36:
        raise ValueError("ds must have shape (n_int, 36)")
    if weight.shape[0] != n_int:
        raise ValueError("weight length must match n_int")
    if n_elem == 0:
        return np.zeros(nnz_out, dtype=np.float64)
    if n_int % n_elem != 0:
        raise ValueError("n_int must be divisible by n_elem")

    n_q = n_int // n_elem
    out = np.zeros(nnz_out, dtype=np.float64)
    assemble_tangent_values_3d_p2_c(
        &dphi1[0, 0],
        &dphi2[0, 0],
        &dphi3[0, 0],
        &ds[0, 0],
        &weight[0],
        <const int64_t *>&scatter_map[0, 0],
        n_int,
        n_elem,
        n_p,
        n_q,
        nnz_out,
        &out[0],
    )
    return out


def constitutive_problem_3d_s(
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] E,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] c_bar,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sin_phi,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] shear,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] bulk,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] lame,
):
    cdef int n_int = <int>E.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] S

    if E.shape[1] != 6:
        raise ValueError("E must have shape (n_int, 6)")
    if c_bar.shape[0] != n_int or sin_phi.shape[0] != n_int or shear.shape[0] != n_int or bulk.shape[0] != n_int or lame.shape[0] != n_int:
        raise ValueError("Material arrays must all have length n_int")

    S = np.empty((n_int, 6), dtype=np.float64)
    constitutive_problem_3d_s_batch_c(
        &E[0, 0],
        &c_bar[0],
        &sin_phi[0],
        &shear[0],
        &bulk[0],
        &lame[0],
        n_int,
        &S[0, 0],
    )
    return S


def constitutive_problem_3d_sds(
    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] E,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] c_bar,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sin_phi,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] shear,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] bulk,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] lame,
):
    cdef int n_int = <int>E.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] S
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] DS

    if E.shape[1] != 6:
        raise ValueError("E must have shape (n_int, 6)")
    if c_bar.shape[0] != n_int or sin_phi.shape[0] != n_int or shear.shape[0] != n_int or bulk.shape[0] != n_int or lame.shape[0] != n_int:
        raise ValueError("Material arrays must all have length n_int")

    S = np.empty((n_int, 6), dtype=np.float64)
    DS = np.empty((n_int, 36), dtype=np.float64)
    constitutive_problem_3d_sds_batch_c(
        &E[0, 0],
        &c_bar[0],
        &sin_phi[0],
        &shear[0],
        &bulk[0],
        &lame[0],
        n_int,
        &S[0, 0],
        &DS[0, 0],
    )
    return S, DS
