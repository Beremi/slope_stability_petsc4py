"""Material expansion utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaterialSpec:
    c0: float
    phi: float
    psi: float
    young: float
    poisson: float
    gamma_sat: float
    gamma_unsat: float

    @property
    def shear(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def bulk(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    @property
    def lame(self) -> float:
        return self.bulk - 2.0 * self.shear / 3.0


def heterogenous_materials(
    mat_identifier: np.ndarray,
    saturation: np.ndarray,
    n_q: int,
    materials: list[MaterialSpec] | list[dict] | dict,
):
    """Replicate MATLAB :func:`ASSEMEBLY.heterogenous_materials`.

    Returns arrays at integration points ``(n_e * n_q,)``.
    """

    if isinstance(materials, dict):
        materials = [materials]

    mat_list: list[MaterialSpec] = []
    for entry in materials:
        if isinstance(entry, MaterialSpec):
            mat_list.append(entry)
        else:
            mat_list.append(MaterialSpec(**entry))

    mat_id = np.asarray(mat_identifier, dtype=np.int64).ravel()
    n_e = len(mat_id)
    n_int = n_e * n_q

    max_mid = int(np.max(mat_id)) if mat_id.size else -1
    if len(mat_list) == 1 and max_mid > 0:
        mat_list = mat_list * (max_mid + 1)
    if max_mid >= len(mat_list):
        raise IndexError(
            f"Material identifier {max_mid} requires at least {max_mid + 1} material rows, got {len(mat_list)}."
        )

    props = np.asarray(
        [
            [
                spec.c0,
                np.deg2rad(spec.phi),
                np.deg2rad(spec.psi),
                spec.shear,
                spec.bulk,
                spec.lame,
                spec.gamma_sat,
                spec.gamma_unsat,
            ]
            for spec in mat_list
        ],
        dtype=np.float64,
    )
    elem_props = props[mat_id, :]
    c0, phi, psi, shear, bulk, lame = (
        np.repeat(elem_props[:, col], n_q).astype(np.float64, copy=False)
        for col in range(6)
    )

    sat = np.asarray(saturation, dtype=bool).ravel()
    if sat.size == 1:
        gamma_col = 6 if bool(sat[0]) else 7
        gamma = np.repeat(elem_props[:, gamma_col], n_q).astype(np.float64, copy=False)
    elif sat.size != n_int:
        raise ValueError("saturation must have size n_e * n_q")
    else:
        gamma_sat = np.repeat(elem_props[:, 6], n_q).astype(np.float64, copy=False)
        gamma_unsat = np.repeat(elem_props[:, 7], n_q).astype(np.float64, copy=False)
        gamma = np.where(sat, gamma_sat, gamma_unsat)

    return c0, phi, psi, shear, bulk, lame, gamma
