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

    c0 = np.zeros(n_e, dtype=np.float64)
    phi = np.zeros(n_e, dtype=np.float64)
    psi = np.zeros(n_e, dtype=np.float64)
    shear = np.zeros(n_e, dtype=np.float64)
    bulk = np.zeros(n_e, dtype=np.float64)
    lame = np.zeros(n_e, dtype=np.float64)
    gamma_sat = np.zeros(n_e, dtype=np.float64)
    gamma_unsat = np.zeros(n_e, dtype=np.float64)

    for i, mid in enumerate(mat_id):
        spec = mat_list[int(mid)]
        c0[i] = spec.c0
        phi[i] = np.deg2rad(spec.phi)
        psi[i] = np.deg2rad(spec.psi)
        shear[i] = spec.shear
        bulk[i] = spec.bulk
        lame[i] = spec.lame
        gamma_sat[i] = spec.gamma_sat
        gamma_unsat[i] = spec.gamma_unsat

    reps = np.repeat(np.arange(n_e), n_q)
    c0 = c0[reps]
    phi = phi[reps]
    psi = psi[reps]
    shear = shear[reps]
    bulk = bulk[reps]
    lame = lame[reps]
    gamma_sat = gamma_sat[reps]
    gamma_unsat = gamma_unsat[reps]

    sat = np.asarray(saturation, dtype=bool).ravel()
    if sat.size != n_int:
        raise ValueError("saturation must have size n_e * n_q")

    gamma = np.where(sat, gamma_sat, gamma_unsat)
    return c0, phi, psi, shear, bulk, lame, gamma
