"""Finite-element assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, block_diag

from .quadrature import quadrature_volume_2d, quadrature_volume_3d
from .basis import local_basis_volume_2d, local_basis_volume_3d


def _as_float(a):
    return np.asarray(a, dtype=np.float64)


@dataclass
class Assembly:
    dim: int
    n_nodes: int
    n_strain: int
    n_int: int
    n_q: int
    n_elem: int
    elem: np.ndarray
    weight: np.ndarray
    B: csr_matrix | None
    dphi: dict[str, np.ndarray]


def assemble_strain_operator(coord: np.ndarray, elem: np.ndarray, elem_type: str, dim: int) -> Assembly:
    if dim == 2:
        return _assemble_2d(coord, elem, elem_type)
    if dim == 3:
        return _assemble_3d(coord, elem, elem_type, build_matrix=True)
    raise ValueError("dim must be 2 or 3")


def assemble_strain_geometry(coord: np.ndarray, elem: np.ndarray, elem_type: str, dim: int) -> Assembly:
    if dim == 3:
        return _assemble_3d(coord, elem, elem_type, build_matrix=False)
    return assemble_strain_operator(coord, elem, elem_type, dim)


def _point_ids(n_nodes_per_elem: int, n_elem: int, n_q: int) -> np.ndarray:
    return (np.tile(np.arange(n_q), n_elem) + np.repeat(np.arange(n_elem) * n_q, n_q)).astype(np.int64)


def _assemble_2d(coord: np.ndarray, elem: np.ndarray, elem_type: str) -> Assembly:
    coord = _as_float(coord)
    elem = np.asarray(elem, dtype=np.int64)

    if coord.ndim != 2:
        raise ValueError("coord must be (dim, n_nodes)")
    n_nodes = coord.shape[1]
    n_elem = elem.shape[1]

    xi, wf = quadrature_volume_2d(elem_type)
    n_q = xi.shape[1]
    hatp, dhat1, dhat2 = local_basis_volume_2d(elem_type, xi)

    if elem_type == "P1":
        n_p = 3
    elif elem_type == "P2":
        n_p = 6
    elif elem_type == "P4":
        n_p = 15
    else:
        raise ValueError(f"Unsupported 2D elem_type={elem_type}")

    n_strain = 3
    n_int = n_elem * n_q
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

    int_ids = np.arange(n_int, dtype=np.int64)
    row0 = 3 * int_ids
    row1 = row0 + 1
    row2 = row0 + 2

    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    vals_parts: list[np.ndarray] = []
    for a in range(n_p):
        node_rep = np.repeat(elem[a, :], n_q)
        col_x = 2 * node_rep
        col_y = col_x + 1

        d1 = np.asarray(dphi1[a, :], dtype=np.float64)
        d2 = np.asarray(dphi2[a, :], dtype=np.float64)

        # Engineering shear row ordering matches MATLAB:
        # [e11, e22, gamma12] with local block [[dN/dx, 0], [0, dN/dy], [dN/dy, dN/dx]].
        rows_parts.extend((row0, row2, row1, row2))
        cols_parts.extend((col_x, col_x, col_y, col_y))
        vals_parts.extend((d1, d2, d2, d1))

    rows = np.concatenate(rows_parts)
    cols = np.concatenate(cols_parts)
    vals = np.concatenate(vals_parts)

    B = coo_matrix((vals, (rows, cols)), shape=(3 * n_int, 2 * n_nodes)).tocsr()
    B.eliminate_zeros()

    return Assembly(
        dim=2,
        n_nodes=n_nodes,
        n_strain=n_strain,
        n_int=n_int,
        n_q=n_q,
        n_elem=n_elem,
        elem=elem,
        weight=weight,
        B=B,
        dphi={"dphi1": dphi1, "dphi2": dphi2, "hatp": hatp},
    )


def _assemble_3d(coord: np.ndarray, elem: np.ndarray, elem_type: str, *, build_matrix: bool) -> Assembly:
    coord = _as_float(coord)
    elem = np.asarray(elem, dtype=np.int64)

    if coord.ndim != 2:
        raise ValueError("coord must be (dim, n_nodes)")
    n_nodes = coord.shape[1]
    n_elem = elem.shape[1]

    xi, wf = quadrature_volume_3d(elem_type)
    n_q = xi.shape[1]

    if elem_type == "P1":
        hatp, dhat1, dhat2, dhat3 = local_basis_volume_3d(elem_type, xi)
        n_p = 4
    elif elem_type == "P2":
        hatp, dhat1, dhat2, dhat3 = local_basis_volume_3d(elem_type, xi)
        n_p = 10
    elif elem_type == "P4":
        hatp, dhat1, dhat2, dhat3 = local_basis_volume_3d(elem_type, xi)
        n_p = 35
    elif elem_type == "Q1":
        hatp, dhat1, dhat2, dhat3 = local_basis_volume_3d(elem_type, xi)
        n_p = 8
    else:
        raise ValueError(f"Unsupported 3D elem_type={elem_type}")

    n_int = n_elem * n_q
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

    dphi1 = np.empty_like(dhat1_t)
    dphi2 = np.empty_like(dhat2_t)
    dphi3 = np.empty_like(dhat3_t)

    dphi1[:] = ((j22 * j33 - j23 * j32) * dhat1_t - (j12 * j33 - j13 * j32) * dhat2_t + (j12 * j23 - j13 * j22) * dhat3_t) * inv_det
    dphi2[:] = (-(j21 * j33 - j23 * j31) * dhat1_t + (j11 * j33 - j13 * j31) * dhat2_t - (j11 * j23 - j13 * j21) * dhat3_t) * inv_det
    dphi3[:] = ((j21 * j32 - j22 * j31) * dhat1_t - (j11 * j32 - j12 * j31) * dhat2_t + (j11 * j22 - j12 * j21) * dhat3_t) * inv_det

    n_strain = 6

    B = None
    if build_matrix:
        n_b = 18 * n_p
        vB = np.zeros((n_b, n_int), dtype=np.float64)
        vB[0:n_b:18, :] = dphi1
        vB[9:n_b:18, :] = dphi1
        vB[17:n_b:18, :] = dphi1
        vB[3:n_b:18, :] = dphi2
        vB[7:n_b:18, :] = dphi2
        vB[16:n_b:18, :] = dphi2
        vB[5:n_b:18, :] = dphi3
        vB[10:n_b:18, :] = dphi3
        vB[14:n_b:18, :] = dphi3

        aux = np.arange(1, 6 * n_int + 1, dtype=np.int64).reshape(6, n_int, order="F")
        iB = np.tile(aux, (3 * n_p, 1))

        aux1 = np.tile(np.arange(1, n_p + 1, dtype=np.int64), (3, 1))
        aux2 = np.tile(np.array([[2], [1], [0]], dtype=np.int64), (1, n_p))
        elem_sel = elem[aux1.reshape(-1, order="F") - 1, :]
        offsets = np.kron(np.ones((1, n_elem), dtype=np.int64), (2 - aux2).reshape(-1, order="F")[:, None])
        aux3 = 3 * elem_sel + offsets
        jB = np.kron(aux3, np.ones((6, n_q), dtype=np.int64))

        B = coo_matrix(
            (vB.reshape(-1, order="F"), (iB.reshape(-1, order="F") - 1, jB.reshape(-1, order="F"))),
            shape=(6 * n_int, 3 * n_nodes),
        ).tocsr()
        B.eliminate_zeros()

    weight = np.tile(np.asarray(wf, dtype=np.float64), n_elem) * np.abs(det_j)

    return Assembly(
        dim=3,
        n_nodes=n_nodes,
        n_strain=n_strain,
        n_int=n_int,
        n_q=n_q,
        n_elem=n_elem,
        elem=elem,
        weight=weight,
        B=B,
        dphi={"dphi1": dphi1, "dphi2": dphi2, "dphi3": dphi3, "hatp": hatp},
    )


def build_elastic_stiffness_matrix(
    assembly: Assembly,
    shear: np.ndarray,
    lame: np.ndarray,
    bulk: np.ndarray | None = None,
) -> tuple[csr_matrix, np.ndarray, csr_matrix]:
    """Build elastic stiffness ``K_elast = B^T D B`` and return local blocks."""

    n_int = assembly.n_int
    n_strain = assembly.n_strain

    shear = np.asarray(shear, dtype=np.float64).ravel()
    lame = np.asarray(lame, dtype=np.float64).ravel()

    if shear.size == 1:
        shear = np.full(n_int, float(shear[0]))
    elif shear.size != n_int:
        raise ValueError("shear size must be 1 or number of integration points")

    if lame.size == 1:
        lame = np.full(n_int, float(lame[0]))
    elif lame.size != n_int:
        raise ValueError("lame size must be 1 or number of integration points")

    if bulk is not None:
        bulk = np.asarray(bulk, dtype=np.float64).ravel()
        if bulk.size == 1:
            bulk = np.full(n_int, float(bulk[0]))
        elif bulk.size != n_int:
            raise ValueError("bulk size must be 1 or number of integration points")

    if n_strain == 3:
        iota = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        vol = np.outer(iota, iota)
        ident = np.diag([1.0, 1.0, 0.5])
    elif n_strain == 6:
        iota = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        vol = np.outer(iota, iota)
        if bulk is None:
            # Keep same shape as legacy MATLAB formula for mixed elasticity usage in some calls.
            bulk = np.zeros(n_int, dtype=np.float64)
        ident = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - vol / 3.0
    else:
        raise ValueError(f"Unsupported n_strain={n_strain}")

    vol_flat = vol.reshape(-1, order="F")
    ident_flat = ident.reshape(-1, order="F")

    d_blocks: list[np.ndarray] = []
    for q in range(n_int):
        vol_coeff = lame[q] if n_strain == 3 else bulk[q]
        block = vol_flat * vol_coeff + 2.0 * ident_flat * shear[q]
        block = block.reshape(n_strain, n_strain, order="F") * assembly.weight[q]
        d_blocks.append(csr_matrix(block))

    Dp = block_diag(d_blocks, format="csr")
    k_tan = assembly.B.T @ (Dp @ assembly.B)
    if n_strain == 3:
        k_tan = (k_tan + k_tan.T) / 2.0
    return k_tan.tocsr(), assembly.weight, assembly.B


def vector_volume(assembly: Assembly, f_int: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Assemble element/interpolation values to nodal vector field."""

    f_int = np.asarray(f_int, dtype=np.float64)
    if f_int.shape[0] != assembly.dim:
        raise ValueError("f_int shape must be (dim, n_int)")
    if f_int.shape[1] != assembly.n_int:
        raise ValueError("f_int shape second dim must equal number of integration points")

    if weight is None:
        weight = assembly.weight
    hatp = assembly.dphi.get("hatp")
    if hatp is None:
        raise ValueError("assembly must contain quadrature basis in dphi['hatp']")
    n_p = hatp.shape[0]
    if n_p not in {3, 4, 6, 8, 10, 15, 35}:
        raise ValueError(f"Unexpected number of basis functions {n_p}")

    n_nodes = assembly.n_nodes
    n_int = assembly.n_int
    hatphi = np.tile(hatp, (1, assembly.n_elem))
    iF = np.zeros(n_p * n_int, dtype=np.int64)
    jF = np.kron(assembly.elem, np.ones((1, assembly.n_q), dtype=np.int64)).reshape(-1, order="F")

    f_out = np.zeros((assembly.dim, n_nodes), dtype=np.float64)
    for c in range(assembly.dim):
        vals = hatphi * (weight * f_int[c, :])[None, :]
        row = coo_matrix(
            (vals.reshape(-1, order="F"), (iF, jF)),
            shape=(1, n_nodes),
        ).toarray()
        f_out[c, :] = row.ravel()
    return f_out


def assemble_from_mesh(
    coord: np.ndarray,
    elem: np.ndarray,
    elem_type: str,
    material: dict[str, np.ndarray] | None = None,
) -> tuple[Assembly, csr_matrix, csr_matrix]:
    dim = int(coord.shape[0])
    asm = assemble_strain_operator(coord, elem, elem_type, dim)
    if material is None:
        n_int = asm.n_int
        shear = np.ones(n_int)
        lame = np.ones(n_int)
        bulk = np.ones(n_int)
    else:
        shear = np.asarray(material["shear"], dtype=np.float64)
        lame = np.asarray(material["lame"], dtype=np.float64)
        bulk = np.asarray(material.get("bulk", np.ones_like(lame)), dtype=np.float64)

    K_elast, weight, B = build_elastic_stiffness_matrix(asm, shear, lame, bulk)
    return asm, K_elast, B
