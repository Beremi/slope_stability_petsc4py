"""Field-export helpers shared by VTU exports and notebook tooling."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def build_field_exports(arrays: Mapping[str, np.ndarray], *, n_cells: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    point_data: dict[str, np.ndarray] = {}
    cell_data: dict[str, np.ndarray] = {}

    if "U" in arrays:
        U = np.asarray(arrays["U"], dtype=np.float64)
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

    return point_data, cell_data


def _first_array(arrays: Mapping[str, np.ndarray], *names: str) -> np.ndarray | None:
    for name in names:
        if name in arrays:
            return np.asarray(arrays[name])
    return None
