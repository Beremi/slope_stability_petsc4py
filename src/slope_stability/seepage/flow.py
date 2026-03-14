"""MATLAB-compatible seepage assembly and scalar Newton-flow solve."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, vstack
from scipy.sparse.linalg import spsolve

from ..fem.basis import local_basis_volume_2d, local_basis_volume_3d
from ..fem.quadrature import quadrature_volume_2d, quadrature_volume_3d


@dataclass(frozen=True)
class SeepageAssembly:
    dim: int
    n_nodes: int
    n_elem: int
    n_q: int
    n_int: int
    elem: np.ndarray
    weight: np.ndarray
    B: csr_matrix
    C: csr_matrix
    hatp: np.ndarray
    dphi: dict[str, np.ndarray]

    @property
    def hatphi_tiled(self) -> np.ndarray:
        return np.tile(self.hatp, (1, self.n_elem))


def heter_conduct(mater: np.ndarray, n_q: int, k: np.ndarray | list[float]) -> np.ndarray:
    """Replicate MATLAB ``SEEPAGE.heter_conduct``."""

    mat_id = np.asarray(mater, dtype=np.int64).ravel()
    k = np.asarray(k, dtype=np.float64).ravel()
    conduct_elem = np.zeros(mat_id.size, dtype=np.float64)
    for i, mid in enumerate(mat_id):
        conduct_elem[i] = float(k[int(mid)])
    return np.kron(conduct_elem, np.ones(n_q, dtype=np.float64))


def penalty_parameters_2d(coord: np.ndarray, elem: np.ndarray) -> np.ndarray:
    coord = np.asarray(coord, dtype=np.float64)
    elem = np.asarray(elem, dtype=np.int64)
    grho = 9.81
    base = elem[:3, :]
    out = np.zeros(base.shape[1], dtype=np.float64)
    for i in range(base.shape[1]):
        w1, w2, w3 = base[:, i]
        p1 = coord[:, w1]
        p2 = coord[:, w2]
        p3 = coord[:, w3]
        l12 = float(np.linalg.norm(p1 - p2))
        l13 = float(np.linalg.norm(p1 - p3))
        l23 = float(np.linalg.norm(p2 - p3))
        out[i] = grho * min(l12, l13, l23) / 2.0
    return out


def penalty_parameters_3d(coord: np.ndarray, elem: np.ndarray) -> np.ndarray:
    coord = np.asarray(coord, dtype=np.float64)
    elem = np.asarray(elem, dtype=np.int64)
    grho = 9.81
    base = elem[:4, :]
    out = np.zeros(base.shape[1], dtype=np.float64)
    for i in range(base.shape[1]):
        ids = base[:, i]
        pts = coord[:, ids].T
        # Match MATLAB ``SEEPAGE.penalty_parameters_3D`` exactly, including
        # its historical L34 formula that uses the z-coordinate of W2.
        p1, p2, p3, p4 = pts
        l12 = float(np.linalg.norm(p1 - p2))
        l13 = float(np.linalg.norm(p1 - p3))
        l23 = float(np.linalg.norm(p2 - p3))
        l14 = float(np.linalg.norm(p1 - p4))
        l24 = float(np.linalg.norm(p2 - p4))
        l34 = float(
            np.sqrt(
                (p3[0] - p4[0]) ** 2
                + (p3[1] - p4[1]) ** 2
                + (p2[2] - p4[2]) ** 2
            )
        )
        out[i] = grho * min((l12, l13, l23, l14, l24, l34)) / 2.0
    return out


def assemble_auxiliary_matrices(coord: np.ndarray, elem: np.ndarray, elem_type: str) -> SeepageAssembly:
    coord = np.asarray(coord, dtype=np.float64)
    elem = np.asarray(elem, dtype=np.int64)
    dim = int(coord.shape[0])
    if dim == 2:
        return _assemble_auxiliary_matrices_2d(coord, elem, elem_type)
    if dim == 3:
        return _assemble_auxiliary_matrices_3d(coord, elem, elem_type)
    raise ValueError("Only 2D and 3D seepage are supported.")


def _assemble_auxiliary_matrices_2d(coord: np.ndarray, elem: np.ndarray, elem_type: str) -> SeepageAssembly:
    xi, wf = quadrature_volume_2d(elem_type)
    hatp, dhat1, dhat2 = local_basis_volume_2d(elem_type, xi)
    n_p = int(elem.shape[0])
    n_elem = int(elem.shape[1])
    n_q = int(xi.shape[1])
    n_int = n_elem * n_q
    n_nodes = int(coord.shape[1])

    dphi1 = np.empty((n_p, n_int), dtype=np.float64)
    dphi2 = np.empty((n_p, n_int), dtype=np.float64)
    weight = np.empty(n_int, dtype=np.float64)
    for e in range(n_elem):
        nodes = elem[:, e]
        xe = coord[0, nodes]
        ye = coord[1, nodes]
        for q in range(n_q):
            col = e * n_q + q
            dh1 = dhat1[:, q]
            dh2 = dhat2[:, q]
            j11 = float(np.dot(xe, dh1))
            j12 = float(np.dot(ye, dh1))
            j21 = float(np.dot(xe, dh2))
            j22 = float(np.dot(ye, dh2))
            det_j = j11 * j22 - j12 * j21
            inv = np.array([[j22, -j12], [-j21, j11]], dtype=np.float64) / det_j
            grads = inv @ np.vstack((dh1, dh2))
            dphi1[:, col] = grads[0, :]
            dphi2[:, col] = grads[1, :]
            weight[col] = abs(det_j) * float(wf[q])

    hatphi = np.tile(hatp, (1, n_elem))
    int_ids = np.arange(n_int, dtype=np.int64)
    rows0 = 2 * int_ids
    rows1 = rows0 + 1

    b_rows: list[np.ndarray] = []
    b_cols: list[np.ndarray] = []
    b_vals: list[np.ndarray] = []
    c_rows: list[np.ndarray] = []
    c_cols: list[np.ndarray] = []
    c_vals: list[np.ndarray] = []

    for a in range(n_p):
        node_rep = np.repeat(elem[a, :], n_q)
        b_rows.extend((rows0, rows1))
        b_cols.extend((node_rep, node_rep))
        b_vals.extend((dphi1[a, :], dphi2[a, :]))
        c_rows.append(int_ids)
        c_cols.append(node_rep)
        c_vals.append(hatphi[a, :])

    B = coo_matrix(
        (np.concatenate(b_vals), (np.concatenate(b_rows), np.concatenate(b_cols))),
        shape=(2 * n_int, n_nodes),
    ).tocsr()
    B.eliminate_zeros()

    C = coo_matrix(
        (np.concatenate(c_vals), (np.concatenate(c_rows), np.concatenate(c_cols))),
        shape=(n_int, n_nodes),
    ).tocsr()
    C.eliminate_zeros()

    return SeepageAssembly(
        dim=2,
        n_nodes=n_nodes,
        n_elem=n_elem,
        n_q=n_q,
        n_int=n_int,
        elem=elem,
        weight=weight,
        B=B,
        C=C,
        hatp=hatp,
        dphi={"dphi1": dphi1, "dphi2": dphi2},
    )


def _assemble_auxiliary_matrices_3d(coord: np.ndarray, elem: np.ndarray, elem_type: str) -> SeepageAssembly:
    xi, wf = quadrature_volume_3d(elem_type)
    hatp, dhat1, dhat2, dhat3 = local_basis_volume_3d(elem_type, xi)
    n_p = int(elem.shape[0])
    n_elem = int(elem.shape[1])
    n_q = int(xi.shape[1])
    n_int = n_elem * n_q
    n_nodes = int(coord.shape[1])

    coord_x = coord[0, elem]
    coord_y = coord[1, elem]
    coord_z = coord[2, elem]

    dhat1_t = np.tile(dhat1, (1, n_elem))
    dhat2_t = np.tile(dhat2, (1, n_elem))
    dhat3_t = np.tile(dhat3, (1, n_elem))
    cx = np.repeat(coord_x, n_q, axis=1)
    cy = np.repeat(coord_y, n_q, axis=1)
    cz = np.repeat(coord_z, n_q, axis=1)

    j11 = np.sum(cx * dhat1_t, axis=0)
    j12 = np.sum(cy * dhat1_t, axis=0)
    j13 = np.sum(cz * dhat1_t, axis=0)
    j21 = np.sum(cx * dhat2_t, axis=0)
    j22 = np.sum(cy * dhat2_t, axis=0)
    j23 = np.sum(cz * dhat2_t, axis=0)
    j31 = np.sum(cx * dhat3_t, axis=0)
    j32 = np.sum(cy * dhat3_t, axis=0)
    j33 = np.sum(cz * dhat3_t, axis=0)

    det_j = j11 * (j22 * j33 - j23 * j32) - j12 * (j21 * j33 - j23 * j31) + j13 * (j21 * j32 - j22 * j31)
    inv_det = 1.0 / det_j

    dphi1 = ((j22 * j33 - j23 * j32) * dhat1_t - (j12 * j33 - j13 * j32) * dhat2_t + (j12 * j23 - j13 * j22) * dhat3_t) * inv_det
    dphi2 = (-(j21 * j33 - j23 * j31) * dhat1_t + (j11 * j33 - j13 * j31) * dhat2_t - (j11 * j23 - j13 * j21) * dhat3_t) * inv_det
    dphi3 = ((j21 * j32 - j22 * j31) * dhat1_t - (j11 * j32 - j12 * j31) * dhat2_t + (j11 * j22 - j12 * j21) * dhat3_t) * inv_det
    weight = np.tile(np.asarray(wf, dtype=np.float64), n_elem) * np.abs(det_j)

    hatphi = np.tile(hatp, (1, n_elem))
    int_ids = np.arange(n_int, dtype=np.int64)
    rows0 = 3 * int_ids
    rows1 = rows0 + 1
    rows2 = rows0 + 2

    b_rows: list[np.ndarray] = []
    b_cols: list[np.ndarray] = []
    b_vals: list[np.ndarray] = []
    c_rows: list[np.ndarray] = []
    c_cols: list[np.ndarray] = []
    c_vals: list[np.ndarray] = []

    for a in range(n_p):
        node_rep = np.repeat(elem[a, :], n_q)
        b_rows.extend((rows0, rows1, rows2))
        b_cols.extend((node_rep, node_rep, node_rep))
        b_vals.extend((dphi1[a, :], dphi2[a, :], dphi3[a, :]))
        c_rows.append(int_ids)
        c_cols.append(node_rep)
        c_vals.append(hatphi[a, :])

    B = coo_matrix(
        (np.concatenate(b_vals), (np.concatenate(b_rows), np.concatenate(b_cols))),
        shape=(3 * n_int, n_nodes),
    ).tocsr()
    B.eliminate_zeros()

    C = coo_matrix(
        (np.concatenate(c_vals), (np.concatenate(c_rows), np.concatenate(c_cols))),
        shape=(n_int, n_nodes),
    ).tocsr()
    C.eliminate_zeros()

    return SeepageAssembly(
        dim=3,
        n_nodes=n_nodes,
        n_elem=n_elem,
        n_q=n_q,
        n_int=n_int,
        elem=elem,
        weight=weight,
        B=B,
        C=C,
        hatp=hatp,
        dphi={"dphi1": dphi1, "dphi2": dphi2, "dphi3": dphi3},
    )


def _compute_pressure_at_integration_points(assembly: SeepageAssembly, pw: np.ndarray) -> np.ndarray:
    pw_e = np.asarray(pw[assembly.elem], dtype=np.float64)
    return np.sum(assembly.hatphi_tiled * np.kron(pw_e, np.ones((1, assembly.n_q), dtype=np.float64)), axis=0)


def _compute_gradient(assembly: SeepageAssembly, nodal_values: np.ndarray) -> np.ndarray:
    grad = assembly.B @ np.asarray(nodal_values, dtype=np.float64)
    return grad.reshape(assembly.dim, assembly.n_int, order="F")


def _build_flow_stiffness(assembly: SeepageAssembly, conduct0: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
    wc = assembly.weight * np.asarray(conduct0, dtype=np.float64)
    diag_vals = np.repeat(wc, assembly.dim)
    D = diags(diag_vals, format="csr")
    K_D = (assembly.B.T @ (D @ assembly.B)).tocsr()
    return K_D, wc


def _rhs_from_dirichlet(assembly: SeepageAssembly, wc: np.ndarray, pw_D: np.ndarray, grho: float) -> np.ndarray:
    grad_pw_D = assembly.B @ np.asarray(pw_D, dtype=np.float64)
    q3 = grad_pw_D
    if assembly.dim == 3:
        grad_y = assembly.B @ np.asarray(assembly_coordinate_y(assembly), dtype=np.float64)
        q3 = q3 + grho * grad_y
    return -assembly.B.T @ (np.repeat(wc, assembly.dim) * q3)


def assembly_coordinate_y(assembly: SeepageAssembly) -> np.ndarray:
    if "_coord_y" in assembly.dphi:
        return np.asarray(assembly.dphi["_coord_y"], dtype=np.float64)
    raise KeyError("assembly does not carry coordinate y cache")


def _build_newton_matrix(
    assembly: SeepageAssembly,
    conduct0: np.ndarray,
    wc: np.ndarray,
    K_D: csr_matrix,
    coord_y: np.ndarray,
    perm_r_der: np.ndarray,
    grho: float,
) -> csr_matrix:
    coeff = np.asarray(conduct0, dtype=np.float64) * np.asarray(perm_r_der, dtype=np.float64) * assembly.weight
    grad_y = _compute_gradient(assembly, coord_y)
    c_coo = assembly.C.tocoo()
    row_base = np.asarray(c_coo.row, dtype=np.int64)
    rows_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    for c in range(assembly.dim):
        rows_parts.append(assembly.dim * row_base + c)
        data_parts.append(np.asarray(c_coo.data, dtype=np.float64) * (coeff * grho * grad_y[c, :])[row_base])
    EC = coo_matrix(
        (np.concatenate(data_parts), (np.concatenate(rows_parts), np.tile(np.asarray(c_coo.col, dtype=np.int64), assembly.dim))),
        shape=(assembly.dim * assembly.n_int, assembly.n_nodes),
    ).tocsr()
    return (K_D + assembly.B.T @ EC).tocsr()


def _build_newton_rhs(
    assembly: SeepageAssembly,
    wc: np.ndarray,
    pw: np.ndarray,
    coord_y: np.ndarray,
    grho: float,
    perm_r: np.ndarray,
) -> np.ndarray:
    grad_pw = _compute_gradient(assembly, pw)
    grad_y = _compute_gradient(assembly, coord_y)
    q3 = grad_pw + grho * np.asarray(perm_r, dtype=np.float64)[None, :] * grad_y
    return -assembly.B.T @ (np.repeat(wc, assembly.dim) * q3.reshape(-1, order="F"))


def newton_flow(
    pw_init: np.ndarray,
    conduct0: np.ndarray,
    Q_w: np.ndarray,
    assembly: SeepageAssembly,
    K_D: csr_matrix,
    wc: np.ndarray,
    eps_int: np.ndarray,
    grho: float,
    *,
    linear_system_solver=None,
    it_max: int = 50,
    tol: float = 1e-10,
) -> tuple[np.ndarray, dict[str, object]]:
    """Replicate MATLAB ``SEEPAGE.newton_flow`` with optional PETSc linear solve."""

    pw = np.asarray(pw_init, dtype=np.float64).copy()
    Q_w = np.asarray(Q_w, dtype=bool).ravel()
    eps_int = np.asarray(eps_int, dtype=np.float64).ravel()
    conduct0 = np.asarray(conduct0, dtype=np.float64).ravel()
    coord_y = np.asarray(assembly.dphi["_coord_y"], dtype=np.float64)
    denom = max(float(np.linalg.norm(pw_init)), 1.0e-14)

    history: dict[str, object] = {
        "criterion": [],
        "linear_iterations": [],
        "linear_solve_time": [],
        "linear_preconditioner_time": [],
        "linear_orthogonalization_time": [],
        "linear_solve_info": [],
        "converged": False,
    }

    it = 0
    while True:
        it += 1
        pw_int = _compute_pressure_at_integration_points(assembly, pw)
        perm_r = np.ones(assembly.n_int, dtype=np.float64)
        perm_r_der = np.zeros(assembly.n_int, dtype=np.float64)
        part1 = (pw_int < eps_int) & (pw_int > 0.0)
        part2 = pw_int <= 0.0
        perm_r[part1] = pw_int[part1] / eps_int[part1]
        perm_r[part2] = 0.0
        perm_r_der[part1] = 1.0 / eps_int[part1]

        K = _build_newton_matrix(assembly, conduct0, wc, K_D, coord_y, perm_r_der, grho)
        f = _build_newton_rhs(assembly, wc, pw, coord_y, grho, perm_r)

        dp = np.zeros_like(pw)
        K_free = K[Q_w][:, Q_w]
        f_free = np.asarray(f[Q_w], dtype=np.float64)

        if linear_system_solver is None:
            dp[Q_w] = spsolve(K_free, f_free)
            nit = 1
            solve_time = 0.0
            precond_time = 0.0
            orth_time = 0.0
            solve_info = {"iterations": 1, "true_residual_history": []}
        else:
            collector = linear_system_solver.iteration_collector
            idx = int(linear_system_solver.instance_id) - 1
            pre_n = len(collector.iterations[idx])
            pre_p = len(collector.preconditioner_times[idx])
            pre_o = len(collector.orthogonalization_times[idx])
            linear_system_solver.setup_preconditioner(K_free)
            linear_system_solver.A_orthogonalize(K_free)
            dp_free = linear_system_solver.solve(K_free, f_free)
            dp[Q_w] = np.asarray(dp_free, dtype=np.float64)
            post_n = len(collector.iterations[idx])
            post_p = len(collector.preconditioner_times[idx])
            post_o = len(collector.orthogonalization_times[idx])
            nit = collector.iterations[idx][post_n - 1] if post_n > pre_n else 0
            solve_time = collector.solve_times[idx][post_n - 1] if post_n > pre_n else 0.0
            precond_time = collector.preconditioner_times[idx][post_p - 1] if post_p > pre_p else 0.0
            orth_time = collector.orthogonalization_times[idx][post_o - 1] if post_o > pre_o else 0.0
            solve_info = linear_system_solver.get_last_solve_info() if hasattr(linear_system_solver, "get_last_solve_info") else {}

        pw = pw + dp
        crit = float(np.linalg.norm(dp) / denom)
        history["criterion"].append(crit)
        history["linear_iterations"].append(int(nit))
        history["linear_solve_time"].append(float(solve_time))
        history["linear_preconditioner_time"].append(float(precond_time))
        history["linear_orthogonalization_time"].append(float(orth_time))
        history["linear_solve_info"].append(dict(solve_info))
        if crit < tol:
            history["converged"] = True
            break
        if it > int(it_max):
            break

    history["iterations"] = len(history["criterion"])
    return pw, history


def _finalize_seepage_outputs(
    assembly: SeepageAssembly,
    pw: np.ndarray,
    eps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grad_p = _compute_gradient(assembly, pw)
    pw_int = _compute_pressure_at_integration_points(assembly, pw)
    if assembly.n_q > 1:
        int_pw_e = np.sum((pw_int * assembly.weight).reshape(assembly.n_elem, assembly.n_q), axis=1)
        int_e = np.sum(assembly.weight.reshape(assembly.n_elem, assembly.n_q), axis=1)
        pw_aver_e = int_pw_e / int_e
    else:
        pw_aver_e = pw_int
    mater_sat = pw_aver_e >= 0.1 * np.asarray(eps, dtype=np.float64)
    return pw, grad_p, mater_sat


def _solve_reduced_system(A_free: csr_matrix, rhs_free: np.ndarray, linear_system_solver):
    if linear_system_solver is None:
        return spsolve(A_free, rhs_free), {
            "iterations": 1,
            "solve_time": 0.0,
            "preconditioner_time": 0.0,
            "orthogonalization_time": 0.0,
            "solve_info": {"iterations": 1, "true_residual_history": []},
        }

    collector = linear_system_solver.iteration_collector
    idx = int(linear_system_solver.instance_id) - 1
    pre_n = len(collector.iterations[idx])
    pre_p = len(collector.preconditioner_times[idx])
    pre_o = len(collector.orthogonalization_times[idx])
    linear_system_solver.setup_preconditioner(A_free)
    linear_system_solver.A_orthogonalize(A_free)
    sol = linear_system_solver.solve(A_free, rhs_free)
    post_n = len(collector.iterations[idx])
    post_p = len(collector.preconditioner_times[idx])
    post_o = len(collector.orthogonalization_times[idx])
    return np.asarray(sol, dtype=np.float64), {
        "iterations": collector.iterations[idx][post_n - 1] if post_n > pre_n else 0,
        "solve_time": collector.solve_times[idx][post_n - 1] if post_n > pre_n else 0.0,
        "preconditioner_time": collector.preconditioner_times[idx][post_p - 1] if post_p > pre_p else 0.0,
        "orthogonalization_time": collector.orthogonalization_times[idx][post_o - 1] if post_o > pre_o else 0.0,
        "solve_info": linear_system_solver.get_last_solve_info() if hasattr(linear_system_solver, "get_last_solve_info") else {},
    }


def seepage_problem_2d(
    coord: np.ndarray,
    elem: np.ndarray,
    Q_w: np.ndarray,
    pw_D: np.ndarray,
    grho: float,
    conduct0: np.ndarray,
    *,
    elem_type: str = "P1",
    linear_system_solver=None,
    it_max: int = 50,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], SeepageAssembly]:
    assembly = assemble_auxiliary_matrices(coord, elem, elem_type)
    assembly.dphi["_coord_y"] = np.asarray(coord[1, :], dtype=np.float64)  # type: ignore[index]
    eps = penalty_parameters_2d(coord, elem)
    eps_int = np.kron(eps, np.ones(assembly.n_q, dtype=np.float64))
    K_D, wc = _build_flow_stiffness(assembly, conduct0)
    f = _rhs_from_dirichlet(assembly, wc, pw_D, grho)
    pw_0 = np.zeros(assembly.n_nodes, dtype=np.float64)
    q_free = np.asarray(Q_w, dtype=bool)
    init_sol, init_info = _solve_reduced_system(K_D[q_free][:, q_free], np.asarray(f[q_free], dtype=np.float64), linear_system_solver)
    pw_0[q_free] = init_sol
    pw_init = pw_0 + np.asarray(pw_D, dtype=np.float64)
    pw, history = newton_flow(
        pw_init,
        conduct0,
        Q_w,
        assembly,
        K_D,
        wc,
        eps_int,
        grho,
        linear_system_solver=linear_system_solver,
        it_max=it_max,
        tol=tol,
    )
    pw, grad_p, mater_sat = _finalize_seepage_outputs(assembly, pw, eps)
    history["K_D_nnz"] = int(K_D.nnz)
    history["init_linear"] = init_info
    return pw, grad_p, mater_sat, history, assembly


def seepage_problem_3d(
    coord: np.ndarray,
    elem: np.ndarray,
    Q_w: np.ndarray,
    pw_D: np.ndarray,
    grho: float,
    conduct0: np.ndarray,
    *,
    elem_type: str = "P2",
    linear_system_solver=None,
    it_max: int = 50,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], SeepageAssembly]:
    assembly = assemble_auxiliary_matrices(coord, elem, elem_type)
    assembly.dphi["_coord_y"] = np.asarray(coord[1, :], dtype=np.float64)  # type: ignore[index]
    eps = penalty_parameters_3d(coord, elem)
    eps_int = np.kron(eps, np.ones(assembly.n_q, dtype=np.float64))
    K_D, wc = _build_flow_stiffness(assembly, conduct0)
    f = _rhs_from_dirichlet(assembly, wc, pw_D, grho)
    q_free = np.asarray(Q_w, dtype=bool)
    pw_0 = np.zeros(assembly.n_nodes, dtype=np.float64)
    init_sol, init_info = _solve_reduced_system(K_D[q_free][:, q_free], np.asarray(f[q_free], dtype=np.float64), linear_system_solver)
    pw_0[q_free] = init_sol
    pw_init = pw_0 + np.asarray(pw_D, dtype=np.float64)
    pw, history = newton_flow(
        pw_init,
        conduct0,
        Q_w,
        assembly,
        K_D,
        wc,
        eps_int,
        grho,
        linear_system_solver=linear_system_solver,
        it_max=it_max,
        tol=tol,
    )
    pw, grad_p, mater_sat = _finalize_seepage_outputs(assembly, pw, eps)
    history["K_D_nnz"] = int(K_D.nnz)
    history["init_linear"] = init_info
    return pw, grad_p, mater_sat, history, assembly
