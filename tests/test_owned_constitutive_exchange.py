from __future__ import annotations

import numpy as np

from slope_stability.constitutive.problem import ConstitutiveOperator
from slope_stability.core.simplex_lagrange import tetra_lagrange_node_tuples
from slope_stability.fem.distributed_tangent import assemble_owned_tangent_values, prepare_owned_tangent_pattern


MATERIAL = {
    "c0": 15.0,
    "phi": 30.0,
    "psi": 0.0,
    "young": 10000.0,
    "poisson": 0.33,
    "gamma_sat": 19.0,
    "gamma_unsat": 19.0,
}


def _elevated_tetra_mesh(order: int, tet4: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tuples = tetra_lagrange_node_tuples(int(order))
    coord_map: dict[tuple[float, float, float], int] = {}
    coord_list: list[np.ndarray] = []
    elem = np.empty((len(tuples), tet4.shape[1]), dtype=np.int64)

    for e in range(int(tet4.shape[1])):
        verts = np.asarray(vertices[:, tet4[:, e]], dtype=np.float64)
        for local_idx, counts in enumerate(tuples):
            bary = np.asarray(counts, dtype=np.float64) / float(order)
            point = verts @ bary
            key = tuple(np.round(point, 12).tolist())
            global_idx = coord_map.get(key)
            if global_idx is None:
                global_idx = len(coord_list)
                coord_map[key] = global_idx
                coord_list.append(point)
            elem[local_idx, e] = int(global_idx)

    coord = np.column_stack(coord_list) if coord_list else np.empty((3, 0), dtype=np.float64)
    return coord, elem


def _build_chain_pattern(elem_type: str = "P2", *, n_elem: int = 4):
    order = {"P1": 1, "P2": 2, "P4": 4}[str(elem_type).upper()]
    n_vertices = int(n_elem) + 3
    t = np.arange(1, n_vertices + 1, dtype=np.float64)
    vertices = np.vstack((t, 0.1 * t * t, 0.01 * t * t * t))
    tet4 = np.vstack([np.arange(i, i + n_elem, dtype=np.int64) for i in range(4)])
    coord, elem = _elevated_tetra_mesh(order, tet4, vertices)
    q_mask = np.ones((3, coord.shape[1]), dtype=bool)
    material_identifier = np.zeros(elem.shape[1], dtype=np.int64)
    pattern = prepare_owned_tangent_pattern(
        coord,
        elem,
        q_mask,
        material_identifier,
        [MATERIAL],
        (0, coord.shape[1]),
        elem_type=elem_type,
        include_unique=True,
        include_legacy_scatter=False,
        include_overlap_B=False,
    )
    return coord, elem, q_mask, material_identifier, pattern


def _material_fields(n_int: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    young = float(MATERIAL["young"])
    poisson = float(MATERIAL["poisson"])
    shear = young / (2.0 * (1.0 + poisson))
    bulk = young / (3.0 * (1.0 - 2.0 * poisson))
    lame = bulk - 2.0 * shear / 3.0
    return (
        np.full(n_int, float(MATERIAL["c0"]), dtype=np.float64),
        np.full(n_int, np.deg2rad(float(MATERIAL["phi"])), dtype=np.float64),
        np.full(n_int, np.deg2rad(float(MATERIAL["psi"])), dtype=np.float64),
        np.full(n_int, shear, dtype=np.float64),
        np.full(n_int, bulk, dtype=np.float64),
        np.full(n_int, lame, dtype=np.float64),
        np.ones(n_int, dtype=np.float64),
    )


def _evaluate_mode(pattern, q_mask, n_elem: int, n_q: int, n_nodes: int, mode: str):
    c0, phi, psi, shear, bulk, lame, weight = _material_fields(int(n_elem * n_q))
    builder = ConstitutiveOperator(
        B=None,
        c0=c0,
        phi=phi,
        psi=psi,
        Davis_type="B",
        shear=shear,
        bulk=bulk,
        lame=lame,
        WEIGHT=weight,
        n_strain=int(pattern.n_strain),
        n_int=int(n_elem * n_q),
        dim=int(pattern.dim),
        q_mask=q_mask,
    )
    builder.set_owned_tangent_pattern(
        pattern,
        use_compiled=False,
        tangent_kernel="rows",
        constitutive_mode=mode,
        use_compiled_constitutive=False,
    )
    rng = np.random.default_rng(12345)
    U = rng.standard_normal((int(pattern.dim), int(n_nodes)))
    builder.reduction(1.0)
    builder.constitutive_problem_stress_tangent(U)
    return {
        "S_local": np.asarray(builder._owned_local_S, dtype=np.float64).copy(),
        "DS_local": np.asarray(builder._owned_local_DS, dtype=np.float64).copy(),
        "F_local": np.asarray(builder.build_F_local(), dtype=np.float64).copy(),
        "tangent_values": np.asarray(
            assemble_owned_tangent_values(pattern, builder._owned_local_DS, use_compiled=False, kernel="rows"),
            dtype=np.float64,
        ).copy(),
    }


def test_owned_constitutive_modes_match_in_rank1() -> None:
    coord, elem, q_mask, _material_identifier, pattern = _build_chain_pattern("P2", n_elem=4)
    n_elem = int(elem.shape[1])
    modes = {
        mode: _evaluate_mode(pattern, q_mask, n_elem, int(pattern.n_q), int(coord.shape[1]), mode)
        for mode in ("overlap", "unique_gather", "unique_exchange")
    }

    baseline = modes["overlap"]
    for mode in ("unique_gather", "unique_exchange"):
        candidate = modes[mode]
        assert np.allclose(candidate["S_local"], baseline["S_local"], rtol=1.0e-11, atol=1.0e-11)
        assert np.allclose(candidate["DS_local"], baseline["DS_local"], rtol=1.0e-11, atol=1.0e-11)
        assert np.allclose(candidate["F_local"], baseline["F_local"], rtol=1.0e-11, atol=1.0e-11)
        assert np.allclose(candidate["tangent_values"], baseline["tangent_values"], rtol=1.0e-11, atol=1.0e-11)
