"""Deflated flexible GMRES implemented in pure Python/Cython kernels."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional

import numpy as np
try:
    from mpi4py import MPI
except Exception:  # pragma: no cover - optional when MPI is unavailable
    MPI = None

try:
    from petsc4py import PETSc
except Exception:  # pragma: no cover - optional when PETSc is unavailable
    PETSc = None

from scipy.sparse import issparse

try:
    from .. import _kernels as kernels
except Exception:  # pragma: no cover - optional extension
    kernels = None

from .orthogonalize import a_orthogonalize


Vec = np.ndarray
VecOp = Callable[[Vec], Vec]
MatLike = object


def _dot(x: np.ndarray, y: np.ndarray) -> float:
    if kernels is None:
        return float(np.dot(x, y))
    return float(kernels.dot(x, y))


def _norm(x: np.ndarray) -> float:
    if kernels is None:
        return float(np.linalg.norm(x))
    return float(kernels.norm2(x))


def _matvec(A, x: Vec) -> Vec:
    if A is None:
        raise ValueError("Linear operator is None")

    if callable(A):
        return np.asarray(A(x), dtype=np.float64)

    if issparse(A):
        return A @ x

    if isinstance(A, np.ndarray):
        return A @ x

    if PETSc is not None and isinstance(A, PETSc.Mat):
        v = PETSc.Vec().createWithArray(np.asarray(x, dtype=np.float64), comm=A.getComm())
        y = A.createVecRight()
        A.mult(v, y)
        return np.asarray(y.getArray(readonly=False)).copy()

    if hasattr(A, "dot"):
        return np.asarray(A.dot(x), dtype=np.float64)

    arr = np.asarray(A)
    return arr @ x


def _to_vec(x) -> Vec:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Expected vector-like input.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("Vector contains non-finite values")
    return arr


def dfgmres(
    A: MatLike,
    b: Vec,
    M: VecOp,
    W: Optional[np.ndarray],
    maxits: int,
    tol: float,
    x0: Optional[Vec] = None,
) -> tuple[Vec, int, np.ndarray]:
    """Solve ``A x = b`` by deflated flexible GMRES."""

    b = _to_vec(b)
    x = np.zeros_like(b) if x0 is None else _to_vec(x0)

    b_norm = _norm(b)
    if b_norm == 0.0:
        b_norm = 1.0

    if W is None:
        W = np.empty((b.size, 0), dtype=np.float64)
    else:
        W = np.asarray(W, dtype=np.float64)
        if W.ndim == 1:
            W = W[:, None]
        if W.ndim != 2:
            raise ValueError("Deflation basis must be a vector or matrix")

    if W.size > 0:
        W = a_orthogonalize(W, A, 0.0)
        x = W @ (W.T @ b)

    def projection(v: Vec) -> Vec:
        if W.size == 0:
            return v
        av = _matvec(A, v)
        return v - W @ (W.T @ av)

    r = b - _matvec(A, x)
    res0 = _norm(r)
    res_hist = np.zeros(maxits + 1, dtype=np.float64)
    res_hist[0] = res0 / b_norm
    if res_hist[0] < tol:
        return x, 0, res_hist[:1]

    n = b.size
    V = np.zeros((n, maxits + 1), dtype=np.float64)
    H = np.zeros((maxits + 1, maxits), dtype=np.float64)
    Wbasis = np.zeros((n, maxits), dtype=np.float64)

    V[:, 0] = r / res0
    g = np.zeros(maxits + 1, dtype=np.float64)
    g[0] = res0

    iters = 0
    y = np.zeros(1, dtype=np.float64)

    for j in range(maxits):
        w = M(V[:, j])
        w = projection(w)
        u = _matvec(A, w)

        for i in range(j + 1):
            H[i, j] = _dot(V[:, i], u)
            u = u - H[i, j] * V[:, i]

        H[j + 1, j] = _norm(u)
        Wbasis[:, j] = w

        if H[j + 1, j] < 1e-14:
            iters = j + 1
            break

        V[:, j + 1] = u / H[j + 1, j]

        subH = H[: j + 2, : j + 1]
        subg = g[: j + 2]
        y, *_ = np.linalg.lstsq(subH, subg, rcond=None)
        residual = _norm(subg - subH @ y)
        res_hist[j + 1] = residual / b_norm
        iters = j + 1

        if res_hist[j + 1] < tol:
            break

    if iters > 0 and y.shape[0] != iters:
        subH = H[: iters + 1, :iters]
        subg = g[: iters + 1]
        y, *_ = np.linalg.lstsq(subH, subg, rcond=None)
    x = x + Wbasis[:, :iters] @ y
    return x, iters, res_hist[: iters + 1]


def dfgmres_matlab_exact(
    A: MatLike,
    b: Vec,
    M: VecOp,
    W: Optional[np.ndarray],
    maxits: int,
    tol: float,
    x0: Optional[Vec] = None,
    stats: Optional[dict[str, float]] = None,
) -> tuple[Vec, int, np.ndarray]:
    """Mirror the repository MATLAB ``dfgmres_solver`` as closely as possible.

    Differences from :func:`dfgmres` are deliberate:
    - no implicit re-A-orthogonalization of ``W`` inside the solve
    - initial guess and projection operators follow the MATLAB code directly
    - residual history is the GMRES least-squares residual history, matching MATLAB
    """

    if maxits is None:
        maxits = 1000
    if tol is None:
        tol = 1.0e-6

    b = _to_vec(b)
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = _to_vec(x0)

    if W is None:
        W = np.empty((b.size, 0), dtype=np.float64)
    else:
        W = np.asarray(W, dtype=np.float64)
        if W.ndim == 1:
            W = W[:, None]
        if W.ndim != 2:
            raise ValueError("Deflation basis must be a vector or matrix")

    if W.size == 0:
        def proj_fct(v: Vec) -> Vec:
            return np.asarray(v, dtype=np.float64)
    else:
        def q_coarse_solve(v: Vec) -> Vec:
            t0 = perf_counter() if stats is not None else None
            vec = np.asarray(v, dtype=np.float64)
            out = W @ (W.T @ vec)
            if t0 is not None:
                stats["coarse_solve_s"] = stats.get("coarse_solve_s", 0.0) + (perf_counter() - t0)
            return out

        def proj_fct(v: Vec) -> Vec:
            t0 = perf_counter() if stats is not None else None
            vec = np.asarray(v, dtype=np.float64)
            out = vec - W @ (W.T @ _matvec(A, vec))
            if t0 is not None:
                stats["projection_s"] = stats.get("projection_s", 0.0) + (perf_counter() - t0)
            return out

        x = q_coarse_solve(b)

    n = b.size
    r0 = b - _matvec(A, x)
    b_norm = _norm(b)
    if b_norm == 0.0:
        b_norm = 1.0
    res_norm = _norm(r0)

    res_hist = np.zeros(maxits + 1, dtype=np.float64)
    res_hist[0] = res_norm / b_norm
    if res_hist[0] < tol:
        return x, 0, res_hist[:1]

    V = np.zeros((n, maxits + 1), dtype=np.float64)
    H = np.zeros((maxits + 1, maxits), dtype=np.float64)
    Wbasis = np.zeros((n, maxits), dtype=np.float64)

    V[:, 0] = r0 / res_norm
    g = np.zeros(maxits + 1, dtype=np.float64)
    g[0] = res_norm

    iters = 0
    y = np.zeros(1, dtype=np.float64)
    for j in range(maxits):
        w = _to_vec(M(V[:, j]))
        w = proj_fct(w)
        u = _matvec(A, w)

        for i in range(j + 1):
            H[i, j] = _dot(V[:, i], u)
            u = u - H[i, j] * V[:, i]

        H[j + 1, j] = _norm(u)
        Wbasis[:, j] = w

        if H[j + 1, j] < 1.0e-14:
            iters = j + 1
            break

        V[:, j + 1] = u / H[j + 1, j]
        subH = H[: j + 2, : j + 1]
        subg = g[: j + 2]
        t_ls = perf_counter() if stats is not None else None
        y, *_ = np.linalg.lstsq(subH, subg, rcond=None)
        if t_ls is not None:
            stats["least_squares_s"] = stats.get("least_squares_s", 0.0) + (perf_counter() - t_ls)
        res_norm = _norm(subg - subH @ y)
        res_hist[j + 1] = res_norm / b_norm
        iters = j + 1

        if res_hist[j + 1] < tol:
            break

    if iters > 0 and y.shape[0] != iters:
        subH = H[: iters + 1, :iters]
        subg = g[: iters + 1]
        t_ls = perf_counter() if stats is not None else None
        y, *_ = np.linalg.lstsq(subH, subg, rcond=None)
        if t_ls is not None:
            stats["least_squares_s"] = stats.get("least_squares_s", 0.0) + (perf_counter() - t_ls)
    res_hist = res_hist[: iters + 1]
    x = x + Wbasis[:, :iters] @ y
    return x, iters, res_hist


def dfgmres_matlab_exact_distributed(
    A_local: VecOp,
    b_local: Vec,
    M_local: VecOp,
    W_local: Optional[np.ndarray],
    maxits: int,
    tol: float,
    comm,
    x0_local: Optional[Vec] = None,
    stats: Optional[dict[str, float]] = None,
) -> tuple[Vec, int, np.ndarray]:
    """Distributed MATLAB-style DFGMRES using owned local vector slices.

    The global Krylov basis is represented implicitly by identical Hessenberg
    data on every rank and local owned slices of the basis vectors. Global
    inner products and norms are handled by MPI reductions only.
    """

    if MPI is None:
        raise RuntimeError("mpi4py is required for distributed explicit DFGMRES")
    if maxits is None:
        maxits = 1000
    if tol is None:
        tol = 1.0e-6

    def _dist_dot(x_local: np.ndarray, y_local: np.ndarray) -> float:
        return float(comm.allreduce(float(np.dot(x_local, y_local)), op=MPI.SUM))

    def _dist_norm(x_local: np.ndarray) -> float:
        return float(np.sqrt(max(_dist_dot(x_local, x_local), 0.0)))

    b_local = _to_vec(b_local)
    if x0_local is None:
        x_local = np.zeros_like(b_local)
    else:
        x_local = _to_vec(x0_local)

    if W_local is None:
        W_local = np.empty((b_local.size, 0), dtype=np.float64)
    else:
        W_local = np.asarray(W_local, dtype=np.float64)
        if W_local.ndim == 1:
            W_local = W_local[:, None]
        if W_local.ndim != 2:
            raise ValueError("Deflation basis must be a vector or matrix")

    if W_local.size == 0:
        def q_coarse_solve(v_local: Vec) -> Vec:
            return np.zeros_like(np.asarray(v_local, dtype=np.float64))

        def proj_fct(v_local: Vec) -> Vec:
            return np.asarray(v_local, dtype=np.float64)
    else:
        def q_coarse_solve(v_local: Vec) -> Vec:
            t0 = perf_counter() if stats is not None else None
            vec = np.asarray(v_local, dtype=np.float64)
            coeff_local = W_local.T @ vec
            coeff = comm.allreduce(coeff_local, op=MPI.SUM)
            out = W_local @ coeff
            if t0 is not None:
                stats["coarse_solve_s"] = stats.get("coarse_solve_s", 0.0) + (perf_counter() - t0)
            return out

        def proj_fct(v_local: Vec) -> Vec:
            t0 = perf_counter() if stats is not None else None
            vec = np.asarray(v_local, dtype=np.float64)
            Avec_local = A_local(vec)
            coeff_local = W_local.T @ Avec_local
            coeff = comm.allreduce(coeff_local, op=MPI.SUM)
            out = vec - W_local @ coeff
            if t0 is not None:
                stats["projection_s"] = stats.get("projection_s", 0.0) + (perf_counter() - t0)
            return out

        x_local = q_coarse_solve(b_local)

    n_local = b_local.size
    r0_local = b_local - A_local(x_local)
    b_norm = _dist_norm(b_local)
    if b_norm == 0.0:
        b_norm = 1.0
    res_norm = _dist_norm(r0_local)

    res_hist = np.zeros(maxits + 1, dtype=np.float64)
    res_hist[0] = res_norm / b_norm
    if res_hist[0] < tol:
        return x_local, 0, res_hist[:1]

    V = np.zeros((n_local, maxits + 1), dtype=np.float64)
    H = np.zeros((maxits + 1, maxits), dtype=np.float64)
    Wbasis = np.zeros((n_local, maxits), dtype=np.float64)

    V[:, 0] = r0_local / res_norm
    g = np.zeros(maxits + 1, dtype=np.float64)
    g[0] = res_norm

    iters = 0
    y = np.zeros(1, dtype=np.float64)
    for j in range(maxits):
        w_local = _to_vec(M_local(V[:, j]))
        w_local = proj_fct(w_local)
        u_local = A_local(w_local)

        for i in range(j + 1):
            H[i, j] = _dist_dot(V[:, i], u_local)
            u_local = u_local - H[i, j] * V[:, i]

        H[j + 1, j] = _dist_norm(u_local)
        Wbasis[:, j] = w_local

        if H[j + 1, j] < 1.0e-14:
            iters = j + 1
            break

        V[:, j + 1] = u_local / H[j + 1, j]
        subH = H[: j + 2, : j + 1]
        subg = g[: j + 2]
        t_ls = perf_counter() if stats is not None else None
        y, *_ = np.linalg.lstsq(subH, subg, rcond=None)
        if t_ls is not None:
            stats["least_squares_s"] = stats.get("least_squares_s", 0.0) + (perf_counter() - t_ls)
        res_norm = _norm(subg - subH @ y)
        res_hist[j + 1] = res_norm / b_norm
        iters = j + 1

        if res_hist[j + 1] < tol:
            break

    if iters > 0 and y.shape[0] != iters:
        subH = H[: iters + 1, :iters]
        subg = g[: iters + 1]
        t_ls = perf_counter() if stats is not None else None
        y, *_ = np.linalg.lstsq(subH, subg, rcond=None)
        if t_ls is not None:
            stats["least_squares_s"] = stats.get("least_squares_s", 0.0) + (perf_counter() - t_ls)
    res_hist = res_hist[: iters + 1]
    x_local = x_local + Wbasis[:, :iters] @ y
    return x_local, iters, res_hist


def dfgmres_matlab_exact_distributed_compiled(
    A_local: VecOp,
    b_local: Vec,
    M_local: VecOp,
    W_local: Optional[np.ndarray],
    maxits: int,
    tol: float,
    comm,
    x0_local: Optional[Vec] = None,
    stats: Optional[dict[str, float]] = None,
) -> tuple[Vec, int, np.ndarray]:
    """Compiled local-vector variant when the Cython extension is available."""

    if kernels is None or not hasattr(kernels, "dfgmres_matlab_exact_distributed_compiled"):
        return dfgmres_matlab_exact_distributed(A_local, b_local, M_local, W_local, maxits, tol, comm, x0_local, stats)
    if x0_local is not None:
        raise NotImplementedError("Compiled distributed DFGMRES currently expects zero initial guess")
    return kernels.dfgmres_matlab_exact_distributed_compiled(
        A_local,
        np.asarray(b_local, dtype=np.float64),
        M_local,
        None if W_local is None else np.asarray(W_local, dtype=np.float64),
        int(maxits),
        float(tol),
        comm,
        stats,
    )


def _identity_prec(x: Vec) -> Vec:
    return np.asarray(x, dtype=np.float64)


@dataclass
class FGMRESCore:
    """Small helper holding pure-NumPy FGMRES core settings."""

    max_iterations: int = 100
    tolerance: float = 1e-6
    use_deflation: bool = True
    preconditioner: VecOp = _identity_prec

    def solve(
        self,
        A,
        b: Vec,
        W: Optional[np.ndarray] = None,
        x0: Optional[Vec] = None,
    ) -> tuple[Vec, int, np.ndarray]:
        return dfgmres(A, b, self.preconditioner, W, self.max_iterations, self.tolerance, x0)
