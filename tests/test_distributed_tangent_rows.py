from __future__ import annotations

import numpy as np
import pytest

from slope_stability.core.simplex_lagrange import tetra_lagrange_node_tuples
from slope_stability.fem.distributed_tangent import (
    _ensure_int32_capacity,
    assemble_overlap_strain,
    assemble_owned_force_from_local_stress,
    assemble_owned_tangent_matrix,
    assemble_owned_tangent_values,
    prepare_owned_tangent_pattern,
)
from slope_stability.fem.distributed_elastic import assemble_owned_elastic_rows


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


def _build_pattern(
    elem_type: str,
    *,
    n_elem: int = 1,
    constrain_first_node: bool = False,
):
    order = {"P1": 1, "P2": 2, "P4": 4}[str(elem_type).upper()]
    vertices = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    tet4 = np.array([[0], [1], [2], [3]], dtype=np.int64)
    if n_elem == 2:
        tet4 = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)

    coord, elem = _elevated_tetra_mesh(order, tet4, vertices)
    q_mask = np.ones((3, coord.shape[1]), dtype=bool)
    if constrain_first_node:
        q_mask[:, 0] = False

    pattern = prepare_owned_tangent_pattern(
        coord,
        elem,
        q_mask,
        np.zeros(elem.shape[1], dtype=np.int64),
        [MATERIAL],
        (0, coord.shape[1]),
        elem_type=elem_type,
        include_unique=True,
    )
    return coord, elem, q_mask, pattern


def _make_ds(pattern, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal((pattern.n_strain * pattern.n_strain, pattern.local_int_indices.size))


@pytest.mark.parametrize("elem_type", ["P1", "P2", "P4"])
def test_owned_tangent_rows_match_legacy_single_tet(elem_type: str) -> None:
    _coord, _elem, _q_mask, pattern = _build_pattern(elem_type, n_elem=1)
    ds = _make_ds(pattern, seed=101)

    legacy = assemble_owned_tangent_values(pattern, ds, use_compiled=False, kernel="legacy")
    rows = assemble_owned_tangent_values(pattern, ds, use_compiled=False, kernel="rows")

    assert legacy.shape == rows.shape == (pattern.local_matrix_pattern.nnz,)
    assert np.allclose(rows, legacy, rtol=1.0e-11, atol=1.0e-11)


def test_owned_tangent_rows_match_legacy_on_shared_rows() -> None:
    _coord, _elem, _q_mask, pattern = _build_pattern("P4", n_elem=2)
    ds = _make_ds(pattern, seed=202)

    row_slots = np.diff(pattern.row_slot_ptr.astype(np.int64))
    assert int(row_slots.max(initial=0)) >= 2

    legacy = assemble_owned_tangent_values(pattern, ds, use_compiled=False, kernel="legacy")
    rows = assemble_owned_tangent_values(pattern, ds, use_compiled=False, kernel="rows")

    assert np.allclose(rows, legacy, rtol=1.0e-11, atol=1.0e-11)


def test_owned_tangent_rows_preserve_constrained_diagonals() -> None:
    _coord, _elem, q_mask, pattern = _build_pattern("P4", n_elem=1, constrain_first_node=True)
    ds = _make_ds(pattern, seed=303)

    legacy = assemble_owned_tangent_values(pattern, ds, use_compiled=False, kernel="legacy")
    rows = assemble_owned_tangent_values(pattern, ds, use_compiled=False, kernel="rows")

    assert np.allclose(rows, legacy, rtol=1.0e-11, atol=1.0e-11)
    assert pattern.constrained_diag_positions.size == 3
    assert np.all(rows[pattern.constrained_diag_positions] == 1.0)

    constrained_rows = np.flatnonzero(~q_mask.reshape(-1, order="F")[: pattern.local_matrix_pattern.shape[0]])
    assert constrained_rows.size == 3
    assert np.all(np.diff(pattern.row_slot_ptr.astype(np.int64))[constrained_rows] == 0)


def test_row_slot_metadata_matches_legacy_scatter_map() -> None:
    _coord, _elem, _q_mask, pattern = _build_pattern("P4", n_elem=2)
    n_local_dof = int(pattern.dim * pattern.n_p)
    legacy_rows = pattern.scatter_map.reshape(pattern.scatter_map.shape[0], n_local_dof, n_local_dof)

    seen: set[tuple[int, int]] = set()
    for local_row in range(int(pattern.local_matrix_pattern.shape[0])):
        for slot_idx in range(int(pattern.row_slot_ptr[local_row]), int(pattern.row_slot_ptr[local_row + 1])):
            elem_id = int(pattern.slot_elem[slot_idx])
            alpha = int(pattern.slot_lrow[slot_idx])
            seen.add((elem_id, alpha))
            assert np.array_equal(pattern.slot_pos[slot_idx], legacy_rows[elem_id, alpha, :])

    active_legacy_rows = int(np.count_nonzero(np.any(legacy_rows >= 0, axis=2)))
    assert len(seen) == active_legacy_rows == int(pattern.slot_elem.size)
    assert pattern.stats["scatter_bytes"] > 0.0
    assert pattern.stats["row_slot_bytes"] > 0.0
    assert pattern.stats["avg_active_rows_per_overlap_element"] > 0.0
    assert pattern.stats["max_active_rows_per_overlap_element"] >= pattern.stats["avg_active_rows_per_overlap_element"]


@pytest.mark.parametrize("elem_type,n_elem", [("P1", 1), ("P2", 2), ("P4", 2)])
def test_direct_overlap_strain_matches_sparse_reference(elem_type: str, n_elem: int) -> None:
    coord, elem, q_mask, pattern = _build_pattern(elem_type, n_elem=n_elem)
    pattern_rows_only = prepare_owned_tangent_pattern(
        coord,
        elem,
        q_mask,
        np.zeros(elem.shape[1], dtype=np.int64),
        [MATERIAL],
        (0, coord.shape[1]),
        elem_type=elem_type,
        include_unique=True,
        include_legacy_scatter=False,
        include_overlap_B=False,
    )
    rng = np.random.default_rng(606)
    U = rng.standard_normal((3, coord.shape[1]))
    u_overlap = np.asarray(U, dtype=np.float64).reshape(-1, order="F")[np.asarray(pattern.overlap_global_dofs, dtype=np.int64)]
    reference = np.asarray(pattern.overlap_B @ u_overlap, dtype=np.float64).reshape(pattern.n_strain, -1, order="F")
    direct = assemble_overlap_strain(pattern_rows_only, U, use_compiled=False)

    assert pattern_rows_only.stats["overlap_B_bytes"] == 0.0
    assert np.allclose(direct, reference, rtol=1.0e-11, atol=1.0e-11)


@pytest.mark.parametrize("elem_type,n_elem", [("P1", 1), ("P2", 2), ("P4", 2)])
def test_direct_owned_force_matches_sparse_reference(elem_type: str, n_elem: int) -> None:
    coord, elem, q_mask, pattern = _build_pattern(elem_type, n_elem=n_elem, constrain_first_node=True)
    pattern_rows_only = prepare_owned_tangent_pattern(
        coord,
        elem,
        q_mask,
        np.zeros(elem.shape[1], dtype=np.int64),
        [MATERIAL],
        (0, coord.shape[1]),
        elem_type=elem_type,
        include_unique=True,
        include_legacy_scatter=False,
        include_overlap_B=False,
    )
    rng = np.random.default_rng(909)
    stress_local = rng.standard_normal((int(pattern.n_strain), int(pattern.local_int_indices.size)))
    load = (np.asarray(pattern.overlap_assembly_weight, dtype=np.float64)[None, :] * stress_local).reshape(-1, order="F")
    overlap_force = pattern.overlap_B.T.dot(load)
    reference = np.asarray(overlap_force[np.asarray(pattern.owned_local_overlap_dofs, dtype=np.int64)], dtype=np.float64).copy()
    reference[~np.asarray(pattern.owned_free_mask, dtype=bool)] = 0.0
    direct = assemble_owned_force_from_local_stress(pattern_rows_only, stress_local, use_compiled=False)

    assert np.allclose(direct, reference, rtol=1.0e-11, atol=1.0e-11)


def test_rows_pattern_can_skip_legacy_scatter_metadata() -> None:
    coord, elem, q_mask, pattern_full = _build_pattern("P4", n_elem=2)
    ds = _make_ds(pattern_full, seed=707)

    pattern_rows_only = prepare_owned_tangent_pattern(
        coord,
        elem,
        q_mask,
        np.zeros(elem.shape[1], dtype=np.int64),
        [MATERIAL],
        (0, coord.shape[1]),
        elem_type="P4",
        include_unique=True,
        include_legacy_scatter=False,
        include_overlap_B=False,
    )

    rows_full = assemble_owned_tangent_values(pattern_full, ds, use_compiled=False, kernel="rows")
    rows_only = assemble_owned_tangent_values(pattern_rows_only, ds, use_compiled=False, kernel="rows")

    assert np.allclose(rows_only, rows_full, rtol=1.0e-11, atol=1.0e-11)
    assert pattern_rows_only.stats["scatter_bytes"] == 0.0
    assert pattern_rows_only.stats["legacy_scatter_enabled"] == 0.0
    assert pattern_rows_only.stats["overlap_B_bytes"] == 0.0
    with pytest.raises(ValueError):
        assemble_owned_tangent_values(pattern_rows_only, ds, use_compiled=False, kernel="legacy")


def test_prepare_owned_tangent_pattern_reuses_prebuilt_elastic_rows() -> None:
    coord, elem, q_mask, pattern_fresh = _build_pattern("P2", n_elem=2)
    material_identifier = np.zeros(elem.shape[1], dtype=np.int64)
    elastic_rows = assemble_owned_elastic_rows(
        coord,
        elem,
        q_mask,
        material_identifier,
        [MATERIAL],
        (0, coord.shape[1]),
        elem_type="P2",
    )
    pattern_reused = prepare_owned_tangent_pattern(
        coord,
        elem,
        q_mask,
        material_identifier,
        [MATERIAL],
        (0, coord.shape[1]),
        elem_type="P2",
        include_unique=True,
        include_legacy_scatter=False,
        include_overlap_B=False,
        elastic_rows=elastic_rows,
    )

    ds = _make_ds(pattern_fresh, seed=808)
    rows_fresh = assemble_owned_tangent_values(pattern_fresh, ds, use_compiled=False, kernel="rows")
    rows_reused = assemble_owned_tangent_values(pattern_reused, ds, use_compiled=False, kernel="rows")

    assert np.allclose(rows_reused, rows_fresh, rtol=1.0e-11, atol=1.0e-11)
    assert pattern_reused.timings["elastic_pattern_reused"] == 1.0
    assert pattern_reused.stats["overlap_B_bytes"] == 0.0
    assert pattern_reused.stats["dphi_bytes"] > 0.0


def test_int32_capacity_guard_raises() -> None:
    with pytest.raises(OverflowError):
        _ensure_int32_capacity("test", np.iinfo(np.int32).max + 1)


def test_plain_owned_tangent_matrix_reuses_petsc_handle() -> None:
    pytest.importorskip("petsc4py")
    from slope_stability.constitutive.problem import ConstitutiveOperator

    _coord, _elem, q_mask, pattern = _build_pattern("P2", n_elem=2)
    ds1 = _make_ds(pattern, seed=404)
    ds2 = _make_ds(pattern, seed=405)
    n_int = int(pattern.local_int_indices.size)

    builder = ConstitutiveOperator(
        B=None,
        c0=np.zeros(n_int, dtype=np.float64),
        phi=np.zeros(n_int, dtype=np.float64),
        psi=np.zeros(n_int, dtype=np.float64),
        Davis_type="B",
        shear=np.ones(n_int, dtype=np.float64),
        bulk=np.ones(n_int, dtype=np.float64),
        lame=np.ones(n_int, dtype=np.float64),
        WEIGHT=np.ones(n_int, dtype=np.float64),
        n_strain=int(pattern.n_strain),
        n_int=n_int,
        dim=int(pattern.dim),
        q_mask=q_mask,
    )
    builder.set_owned_tangent_pattern(
        pattern,
        use_compiled=False,
        tangent_kernel="rows",
        constitutive_mode="overlap",
        use_compiled_constitutive=False,
    )

    builder._owned_local_DS = ds1
    mat1 = builder._build_owned_tangent_matrix()
    handle1 = int(mat1.handle)

    builder._owned_local_DS = ds2
    mat2 = builder._build_owned_tangent_matrix()
    handle2 = int(mat2.handle)
    assert handle2 == handle1

    rows, cols, vals = mat2.getValuesCSR()
    updated = assemble_owned_tangent_matrix(pattern, ds2, use_compiled=False, kernel="rows")
    assert np.array_equal(np.asarray(rows), np.asarray(updated.indptr))
    assert np.array_equal(np.asarray(cols), np.asarray(updated.indices))
    assert np.allclose(np.asarray(vals), np.asarray(updated.data), rtol=1.0e-11, atol=1.0e-11)

    builder.release_petsc_caches()
