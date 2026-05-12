from __future__ import annotations

import json

import numpy as np
from mpi4py import MPI

from slope_stability.constitutive.problem import ConstitutiveOperator
from slope_stability.core.simplex_lagrange import tetra_lagrange_node_tuples
from slope_stability.fem.distributed_tangent import assemble_owned_tangent_matrix, prepare_owned_tangent_pattern


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


def _build_p4_two_tet_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    tet4 = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    coord, elem = _elevated_tetra_mesh(4, tet4, vertices)
    q_mask = np.ones((3, coord.shape[1]), dtype=bool)
    q_mask[:, 0] = False
    return coord, elem, q_mask


def _builder(pattern, q_mask: np.ndarray, use_compiled: bool, *, n_int_global: int) -> ConstitutiveOperator:
    builder = ConstitutiveOperator(
        B=None,
        c0=np.zeros(n_int_global, dtype=np.float64),
        phi=np.zeros(n_int_global, dtype=np.float64),
        psi=np.zeros(n_int_global, dtype=np.float64),
        Davis_type="B",
        shear=np.ones(n_int_global, dtype=np.float64),
        bulk=np.ones(n_int_global, dtype=np.float64),
        lame=np.ones(n_int_global, dtype=np.float64),
        WEIGHT=np.ones(n_int_global, dtype=np.float64),
        n_strain=int(pattern.n_strain),
        n_int=int(n_int_global),
        dim=int(pattern.dim),
        q_mask=q_mask,
    )
    builder.set_owned_tangent_pattern(
        pattern,
        use_compiled=use_compiled,
        tangent_kernel="rows",
        constitutive_mode="overlap",
        tangent_matrix_backend="petsc_aij_element",
        use_compiled_constitutive=False,
    )
    return builder


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    if size != 4:
        raise RuntimeError("This PETSc-native tangent check expects exactly 4 ranks")

    coord, elem, q_mask = _build_p4_two_tet_mesh()
    owned_node_range = ((rank * coord.shape[1]) // size, ((rank + 1) * coord.shape[1]) // size)
    pattern = prepare_owned_tangent_pattern(
        coord,
        elem,
        q_mask,
        np.zeros(elem.shape[1], dtype=np.int64),
        [MATERIAL],
        owned_node_range,
        elem_type="P4",
        include_unique=True,
    )

    n_int_global = int(elem.shape[1] * pattern.n_q)
    rng = np.random.default_rng(1205)
    ds1 = rng.standard_normal((int(pattern.n_strain * pattern.n_strain), n_int_global))
    ds2 = rng.standard_normal((int(pattern.n_strain * pattern.n_strain), n_int_global))

    builder = _builder(pattern, q_mask, use_compiled=True, n_int_global=n_int_global)
    builder.DS = ds1
    mat1 = builder._build_owned_tangent_matrix()
    handle1 = int(mat1.handle)

    builder.DS = ds2
    mat2 = builder._build_owned_tangent_matrix()
    handle_reused = int(mat2.handle) == handle1
    mat_range = tuple(int(v) for v in mat2.getOwnershipRange())

    expected = assemble_owned_tangent_matrix(pattern, ds2, use_compiled=True, kernel="rows")
    x_global = np.linspace(-0.5, 0.75, int(expected.shape[1]), dtype=np.float64)
    x = mat2.createVecRight()
    xr0, xr1 = x.getOwnershipRange()
    x.setArray(np.asarray(x_global[int(xr0) : int(xr1)], dtype=np.float64).copy())
    y = mat2.createVecLeft()
    mat2.mult(x, y)
    expected_y = np.asarray(expected.dot(x_global), dtype=np.float64)
    actual_y = np.asarray(y.getArray(readonly=True), dtype=np.float64)
    local_max_abs = float(np.max(np.abs(actual_y - expected_y))) if actual_y.size else 0.0
    row_ids = np.arange(mat_range[0], mat_range[1], dtype=np.int32)
    col_ids = np.arange(int(expected.shape[1]), dtype=np.int32)
    actual_dense = np.asarray(mat2.getValues(row_ids, col_ids), dtype=np.float64)
    expected_dense = np.asarray(expected.toarray(), dtype=np.float64)
    dense_diff = np.abs(actual_dense - expected_dense)
    local_dense_max_abs = float(dense_diff.max(initial=0.0))
    local_ok = (
        handle_reused
        and np.allclose(actual_y, expected_y, rtol=1.0e-11, atol=1.0e-11)
        and local_dense_max_abs <= 1.0e-11
    )
    all_ok = bool(comm.allreduce(bool(local_ok), op=MPI.LAND))
    nnz_total = int(comm.allreduce(int(expected.nnz), op=MPI.SUM))
    max_abs_by_rank = comm.gather(local_max_abs, root=0)
    dense_max_abs_by_rank = comm.gather(local_dense_max_abs, root=0)
    ranges_by_rank = comm.gather(
        {
            "pattern": tuple(int(v) for v in pattern.owned_row_range),
            "mat": mat_range,
            "x": (int(xr0), int(xr1)),
            "y_size": int(actual_y.size),
            "expected_size": int(expected_y.size),
        },
        root=0,
    )

    builder.release_petsc_caches()

    if rank == 0:
        print(
            json.dumps(
                {
                    "size": size,
                    "elem_type": "P4",
                    "backend": str(builder.owned_tangent_matrix_backend),
                    "handle_reused": bool(handle_reused),
                    "max_abs_by_rank": max_abs_by_rank,
                    "dense_max_abs_by_rank": dense_max_abs_by_rank,
                    "nnz_total": nnz_total,
                    "ok": all_ok,
                    "ranges_by_rank": ranges_by_rank,
                },
                sort_keys=True,
            )
        )
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
