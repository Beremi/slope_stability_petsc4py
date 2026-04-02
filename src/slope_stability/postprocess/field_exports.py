"""Field-export helpers shared by VTU exports and notebook tooling."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def build_field_exports(
    arrays: Mapping[str, np.ndarray],
    *,
    n_cells: int,
    coord: np.ndarray | None = None,
    elem: np.ndarray | None = None,
    elem_type: str | None = None,
    dim: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    point_data: dict[str, np.ndarray] = {}
    cell_data: dict[str, np.ndarray] = {}
    displacement_matrix: np.ndarray | None = None

    if "U" in arrays:
        U = np.asarray(arrays["U"], dtype=np.float64)
        displacement_matrix = U
        disp = np.zeros((U.shape[1], 3), dtype=np.float64)
        disp[:, : U.shape[0]] = U.T
        point_data["displacement"] = disp
        point_data["displacement_magnitude"] = np.linalg.norm(disp, axis=1)

    pore_pressure = _first_array(
        arrays,
        "pore_pressure_export",
        "pw_export",
        "seepage_pw_reordered",
        "pw_reordered",
        "pw",
        "seepage_pw",
    )
    if pore_pressure is not None:
        point_data["pore_pressure"] = np.asarray(pore_pressure, dtype=np.float64).reshape(-1)

    grad_p = _first_array(arrays, "grad_p", "seepage_grad_p")
    if grad_p is not None:
        grad = np.asarray(grad_p, dtype=np.float64)
        if grad.ndim == 2 and grad.shape[1] % max(n_cells, 1) == 0:
            n_q = grad.shape[1] // max(n_cells, 1)
            grad_cell = grad.reshape(grad.shape[0], n_q, n_cells, order="F").mean(axis=1).T
            pad = np.zeros((n_cells, 3), dtype=np.float64)
            pad[:, : grad_cell.shape[1]] = grad_cell
            cell_data["pressure_gradient"] = pad

    saturation = _first_array(arrays, "mater_sat", "seepage_mater_sat")
    if saturation is not None:
        sat = np.asarray(saturation, dtype=np.float64).reshape(-1)
        if sat.size == n_cells:
            cell_data["saturation"] = sat

    if (
        displacement_matrix is not None
        and coord is not None
        and elem is not None
        and dim is not None
        and int(dim) == 3
        and elem_type is not None
    ):
        elem_key = str(elem_type).strip().upper()
        coord_arr = np.asarray(coord, dtype=np.float64)
        elem_arr = np.asarray(elem, dtype=np.int64)
        if elem_key == "P4":
            point_data["deviatoric_strain"] = _compute_point_deviatoric_strain_p4(
                coord_arr,
                elem_arr,
                np.asarray(displacement_matrix, dtype=np.float64),
            )
        else:
            cell_data["deviatoric_strain"] = _compute_cell_deviatoric_strain(
                coord_arr,
                elem_arr,
                elem_key,
                np.asarray(displacement_matrix, dtype=np.float64),
                dim=3,
            )

    return point_data, cell_data


def _first_array(arrays: Mapping[str, np.ndarray], *names: str) -> np.ndarray | None:
    for name in names:
        if name in arrays:
            return np.asarray(arrays[name])
    return None


def _compute_cell_deviatoric_strain(
    coord: np.ndarray,
    elem: np.ndarray,
    elem_type: str,
    displacement: np.ndarray,
    *,
    dim: int,
) -> np.ndarray:
    from slope_stability.fem import assemble_strain_operator

    assembly = assemble_strain_operator(coord, elem, elem_type, dim=dim)
    strain = assembly.B @ np.asarray(displacement, dtype=np.float64).reshape(-1, order="F")
    strain = strain.reshape(assembly.n_strain, -1, order="F")
    dev_norm = _deviatoric_strain_norm(strain, dim=dim)
    n_q = max(dev_norm.size // elem.shape[1], 1)
    return np.mean(dev_norm.reshape(n_q, elem.shape[1], order="F"), axis=0)


def _compute_point_deviatoric_strain_p4(
    coord: np.ndarray,
    elem: np.ndarray,
    displacement: np.ndarray,
) -> np.ndarray:
    from slope_stability.core.simplex_lagrange import tetra_reference_nodes
    from slope_stability.fem.basis import local_basis_volume_3d

    xi = tetra_reference_nodes(4)
    _, dhat1, dhat2, dhat3 = local_basis_volume_3d("P4", xi)

    elem_arr = np.asarray(elem, dtype=np.int64)
    coord_arr = np.asarray(coord, dtype=np.float64)
    disp_arr = np.asarray(displacement, dtype=np.float64)
    if coord_arr.shape[0] != 3 or disp_arr.shape[0] != 3:
        raise ValueError("P4 3D deviatoric export expects 3D coordinates and 3D displacement.")

    x = coord_arr[0, elem_arr]
    y = coord_arr[1, elem_arr]
    z = coord_arr[2, elem_arr]
    ux = disp_arr[0, elem_arr]
    uy = disp_arr[1, elem_arr]
    uz = disp_arr[2, elem_arr]

    accum = np.zeros(coord_arr.shape[1], dtype=np.float64)
    counts = np.zeros(coord_arr.shape[1], dtype=np.int64)

    for q in range(elem_arr.shape[0]):
        dh1 = np.asarray(dhat1[:, q], dtype=np.float64)
        dh2 = np.asarray(dhat2[:, q], dtype=np.float64)
        dh3 = np.asarray(dhat3[:, q], dtype=np.float64)

        j11 = dh1 @ x
        j12 = dh1 @ y
        j13 = dh1 @ z
        j21 = dh2 @ x
        j22 = dh2 @ y
        j23 = dh2 @ z
        j31 = dh3 @ x
        j32 = dh3 @ y
        j33 = dh3 @ z

        det_j = j11 * (j22 * j33 - j23 * j32) - j12 * (j21 * j33 - j23 * j31) + j13 * (j21 * j32 - j22 * j31)
        inv_det = 1.0 / det_j

        c11 = (j22 * j33 - j23 * j32) * inv_det
        c12 = -(j12 * j33 - j13 * j32) * inv_det
        c13 = (j12 * j23 - j13 * j22) * inv_det
        c21 = -(j21 * j33 - j23 * j31) * inv_det
        c22 = (j11 * j33 - j13 * j31) * inv_det
        c23 = -(j11 * j23 - j13 * j21) * inv_det
        c31 = (j21 * j32 - j22 * j31) * inv_det
        c32 = -(j11 * j32 - j12 * j31) * inv_det
        c33 = (j11 * j22 - j12 * j21) * inv_det

        dphi1 = (
            dh1[:, None] * c11[None, :]
            + dh2[:, None] * c12[None, :]
            + dh3[:, None] * c13[None, :]
        )
        dphi2 = (
            dh1[:, None] * c21[None, :]
            + dh2[:, None] * c22[None, :]
            + dh3[:, None] * c23[None, :]
        )
        dphi3 = (
            dh1[:, None] * c31[None, :]
            + dh2[:, None] * c32[None, :]
            + dh3[:, None] * c33[None, :]
        )

        e11 = np.sum(dphi1 * ux, axis=0)
        e22 = np.sum(dphi2 * uy, axis=0)
        e33 = np.sum(dphi3 * uz, axis=0)
        g12 = np.sum(dphi2 * ux + dphi1 * uy, axis=0)
        g23 = np.sum(dphi3 * uy + dphi2 * uz, axis=0)
        g13 = np.sum(dphi3 * ux + dphi1 * uz, axis=0)
        dev_norm = _deviatoric_strain_norm(
            np.vstack((e11, e22, e33, g12, g23, g13)),
            dim=3,
        )
        np.add.at(accum, elem_arr[q, :], dev_norm)
        np.add.at(counts, elem_arr[q, :], 1)

    values = np.zeros_like(accum)
    mask = counts > 0
    values[mask] = accum[mask] / counts[mask]
    return values


def _deviatoric_strain_norm(strain: np.ndarray, *, dim: int) -> np.ndarray:
    strain_arr = np.asarray(strain, dtype=np.float64)
    if dim == 3:
        iota = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        dev = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - np.outer(iota, iota) / 3.0
    elif dim == 2:
        iota = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        dev = np.diag([1.0, 1.0, 0.5]) - np.outer(iota, iota) / 2.0
    else:
        raise ValueError(f"Unsupported dim {dim}")
    dev_e = dev @ strain_arr
    return np.sqrt(np.maximum(0.0, np.sum(strain_arr * dev_e, axis=0)))
