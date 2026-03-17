from __future__ import annotations

import numpy as np
import pytest
from petsc4py import PETSc
from scipy.sparse import csr_matrix, diags

from slope_stability.core.simplex_lagrange import tetra_lagrange_node_tuples
from slope_stability.fem.assembly import assemble_strain_operator, build_elastic_stiffness_matrix
from slope_stability.fem.distributed_tangent import _project_values_onto_pattern, prepare_bddc_subdomain_pattern
from slope_stability.mesh.materials import heterogenous_materials
from slope_stability.linear.solver import PetscMatlabExactDFGMRESSolver
from slope_stability.utils import (
    get_petsc_is_local_mat,
    get_petsc_matrix_metadata,
    local_csr_to_petsc_matis_matrix,
    update_petsc_aij_matrix_csr,
)


MATERIAL = {
    "c0": 15.0,
    "phi": 30.0,
    "psi": 0.0,
    "young": 10000.0,
    "poisson": 0.33,
    "gamma_sat": 19.0,
    "gamma_unsat": 19.0,
}


def _local_block_matrix() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [4.0, 1.0, 0.0, 0.0],
                [1.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 1.0],
                [0.0, 0.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )
    )


def _local_field_is() -> tuple[PETSc.IS, ...]:
    return tuple(
        PETSc.IS().createGeneral(np.asarray([comp, comp + 2], dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
        for comp in range(2)
    )


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


def _legacy_projected_bddc_elastic_values(
    coord: np.ndarray,
    elem: np.ndarray,
    q_mask: np.ndarray,
    material_identifier: np.ndarray,
    materials,
    owned_node_range: tuple[int, int],
    *,
    elem_type: str,
    local_matrix_pattern: csr_matrix,
):
    dim = int(coord.shape[0])
    node0, node1 = (int(owned_node_range[0]), int(owned_node_range[1]))
    elem_owner_nodes = np.min(np.asarray(elem, dtype=np.int64), axis=0)
    local_elements = np.flatnonzero((elem_owner_nodes >= node0) & (elem_owner_nodes < node1)).astype(np.int64)
    local_nodes = np.unique(np.asarray(elem[:, local_elements], dtype=np.int64).reshape(-1, order="F")).astype(np.int64)
    node_lids = np.full(int(coord.shape[1]), -1, dtype=np.int64)
    node_lids[local_nodes] = np.arange(local_nodes.size, dtype=np.int64)
    coord_local = np.asarray(coord, dtype=np.float64)[:, local_nodes]
    elem_local = node_lids[np.asarray(elem, dtype=np.int64)[:, local_elements]]
    asm = assemble_strain_operator(coord_local, elem_local, elem_type, dim=dim)
    local_global_dofs = (
        dim * np.repeat(local_nodes, dim) + np.tile(np.arange(dim, dtype=np.int64), local_nodes.size)
    ).astype(np.int64)
    free_mask_local = np.asarray(q_mask, dtype=bool).reshape(-1, order="F")[local_global_dofs]
    _c0, _phi, _psi, shear, bulk, lame, _gamma = heterogenous_materials(
        np.asarray(material_identifier, dtype=np.int64)[local_elements],
        np.ones(asm.n_int, dtype=bool),
        asm.n_q,
        materials,
    )
    K_elast_local, _weight, _B = build_elastic_stiffness_matrix(asm, shear, lame, bulk)
    K_elast_local = (K_elast_local @ diags(np.asarray(free_mask_local, dtype=np.float64), format="csr")).tocsr()
    constrained = np.flatnonzero(~np.asarray(free_mask_local, dtype=bool))
    if constrained.size:
        K_elast_local = K_elast_local.tolil()
        for row_local in constrained.tolist():
            K_elast_local.rows[row_local] = [int(row_local)]
            K_elast_local.data[row_local] = [1.0]
        K_elast_local = K_elast_local.tocsr()
    return _project_values_onto_pattern(K_elast_local, local_matrix_pattern)


def test_local_csr_to_petsc_matis_matrix_keeps_metadata_and_local_matrix() -> None:
    A_local = _local_block_matrix()
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    mat = local_csr_to_petsc_matis_matrix(
        A_local,
        global_size=4,
        local_to_global=np.array([0, 1, 2, 3], dtype=np.int64),
        comm=PETSc.COMM_WORLD,
        block_size=2,
        metadata={
            "bddc_field_is_local": _local_field_is(),
            "bddc_dirichlet_local": np.array([], dtype=np.int32),
            "bddc_local_adjacency": (
                np.asarray(A_local.indptr, dtype=PETSc.IntType),
                np.asarray(A_local.indices, dtype=PETSc.IntType),
            ),
            "bddc_local_coordinates": np.repeat(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64), 2, axis=0),
            "bddc_local_nullspace_basis": basis,
            "bddc_local_near_nullspace_basis": basis,
        },
    )
    metadata = get_petsc_matrix_metadata(mat)
    local_mat = get_petsc_is_local_mat(mat)

    assert str(mat.getType()).lower() == "is"
    assert local_mat is not None
    assert metadata["bddc_field_is_local"]
    assert metadata["bddc_local_adjacency"][0].shape == A_local.indptr.shape
    assert np.asarray(metadata["bddc_local_coordinates"]).shape == (4, 2)
    assert local_mat.getNullSpace() is not None
    assert local_mat.getNearNullSpace() is not None

    new_data = np.asarray(A_local.data, dtype=np.float64) * 1.5
    update_petsc_aij_matrix_csr(
        local_mat,
        indptr=np.asarray(A_local.indptr, dtype=PETSc.IntType),
        indices=np.asarray(A_local.indices, dtype=PETSc.IntType),
        data=new_data,
    )
    mat.assemble()
    mat.destroy()


def test_matlab_exact_solver_accepts_explicit_bddc_preconditioner() -> None:
    A_local = _local_block_matrix()
    P = local_csr_to_petsc_matis_matrix(
        A_local,
        global_size=4,
        local_to_global=np.array([0, 1, 2, 3], dtype=np.int64),
        comm=PETSc.COMM_WORLD,
        block_size=2,
        metadata={
            "bddc_field_is_local": _local_field_is(),
            "bddc_dirichlet_local": np.array([], dtype=np.int32),
            "bddc_local_adjacency": (
                np.asarray(A_local.indptr, dtype=PETSc.IntType),
                np.asarray(A_local.indices, dtype=PETSc.IntType),
            ),
        },
    )
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="JACOBI",
        q_mask=np.array([[True, True], [True, True]], dtype=bool),
        coord=np.zeros((2, 2), dtype=np.float64),
        preconditioner_options={
            "pc_backend": "bddc",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
        },
    )

    solver.setup_preconditioner(A_local, full_matrix=A_local, preconditioning_matrix=P)
    x = solver.solve(
        A_local,
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        full_rhs=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )
    diagnostics = solver.get_preconditioner_diagnostics()

    assert x.shape == (4,)
    assert np.all(np.isfinite(x))
    assert diagnostics["pc_backend"] == "bddc"
    assert diagnostics["preconditioner_rebuild_count"] == 1

    solver.release_iteration_resources()
    P.destroy()


def test_bddc_refresh_gate_tracks_lagged_policy() -> None:
    A_local = _local_block_matrix()
    P = local_csr_to_petsc_matis_matrix(
        A_local,
        global_size=4,
        local_to_global=np.array([0, 1, 2, 3], dtype=np.int64),
        comm=PETSc.COMM_WORLD,
        block_size=2,
        metadata={
            "bddc_field_is_local": _local_field_is(),
            "bddc_dirichlet_local": np.array([], dtype=np.int32),
            "bddc_local_adjacency": (
                np.asarray(A_local.indptr, dtype=PETSc.IntType),
                np.asarray(A_local.indices, dtype=PETSc.IntType),
            ),
        },
    )
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="JACOBI",
        q_mask=np.array([[True, True], [True, True]], dtype=bool),
        coord=np.zeros((2, 2), dtype=np.float64),
        preconditioner_options={
            "pc_backend": "bddc",
            "preconditioner_matrix_policy": "lagged",
            "preconditioner_rebuild_policy": "accepted_step",
        },
    )

    assert solver.needs_preconditioning_matrix_refresh() is True
    solver.setup_preconditioner(A_local, full_matrix=A_local, preconditioning_matrix=P)
    solver.release_iteration_resources()
    assert solver.needs_preconditioning_matrix_refresh() is False
    solver.notify_continuation_attempt(success=True)
    assert solver.needs_preconditioning_matrix_refresh() is True

    P.destroy()


def test_bddc_elastic_source_reuses_static_pmat_under_current_policy() -> None:
    A_local = _local_block_matrix()
    P = local_csr_to_petsc_matis_matrix(
        A_local,
        global_size=4,
        local_to_global=np.array([0, 1, 2, 3], dtype=np.int64),
        comm=PETSc.COMM_WORLD,
        block_size=2,
        metadata={
            "bddc_field_is_local": _local_field_is(),
            "bddc_dirichlet_local": np.array([], dtype=np.int32),
            "bddc_local_adjacency": (
                np.asarray(A_local.indptr, dtype=PETSc.IntType),
                np.asarray(A_local.indices, dtype=PETSc.IntType),
            ),
        },
    )
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="JACOBI",
        q_mask=np.array([[True, True], [True, True]], dtype=bool),
        coord=np.zeros((2, 2), dtype=np.float64),
        preconditioner_options={
            "pc_backend": "bddc",
            "preconditioner_matrix_source": "elastic",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
        },
    )

    solver.setup_preconditioner(A_local, full_matrix=A_local, preconditioning_matrix=P)
    solver.release_iteration_resources()
    solver.setup_preconditioner(A_local, full_matrix=A_local, preconditioning_matrix=P)
    diagnostics = solver.get_preconditioner_diagnostics()

    assert diagnostics["preconditioner_rebuild_count"] == 1
    assert diagnostics["preconditioner_reuse_count"] == 1
    assert diagnostics["preconditioner_last_rebuild_reason"] == "initial"

    solver.release_iteration_resources()
    P.destroy()


def test_prepare_bddc_subdomain_pattern_marks_interface_primal_vertices() -> None:
    vertices = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 0.1, 0.4, 0.9, 1.6],
            [0.0, 0.01, 0.08, 0.27, 0.64],
        ],
        dtype=np.float64,
    )
    tet4 = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    coord, elem = _elevated_tetra_mesh(1, tet4, vertices)
    q_mask = np.ones((3, coord.shape[1]), dtype=bool)
    pattern = prepare_bddc_subdomain_pattern(
        coord,
        elem,
        q_mask,
        np.zeros(elem.shape[1], dtype=np.int64),
        [MATERIAL],
        (0, 1),
        elem_type="P1",
    )

    assert pattern.local_primal_vertices.size > 0
    assert int(pattern.stats["local_interface_nodes_count"]) > 0
    assert int(pattern.stats["local_primal_vertices_count"]) == int(pattern.local_primal_vertices.size)
    assert bool(pattern.stats["explicit_primal_vertices_used"]) is True


@pytest.mark.parametrize("elem_type,order", [("P1", 1), ("P2", 2), ("P4", 4)])
def test_prepare_bddc_subdomain_pattern_elastic_values_match_projected_reference(elem_type: str, order: int) -> None:
    vertices = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 0.1, 0.4, 0.9, 1.6],
            [0.0, 0.01, 0.08, 0.27, 0.64],
        ],
        dtype=np.float64,
    )
    tet4 = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    coord, elem = _elevated_tetra_mesh(order, tet4, vertices)
    q_mask = np.ones((3, coord.shape[1]), dtype=bool)
    owned_node_range = (0, int(coord.shape[1]))
    pattern = prepare_bddc_subdomain_pattern(
        coord,
        elem,
        q_mask,
        np.zeros(elem.shape[1], dtype=np.int64),
        [MATERIAL],
        owned_node_range,
        elem_type=elem_type,
    )

    ref = _legacy_projected_bddc_elastic_values(
        coord,
        elem,
        q_mask,
        np.zeros(elem.shape[1], dtype=np.int64),
        [MATERIAL],
        owned_node_range,
        elem_type=elem_type,
        local_matrix_pattern=pattern.local_matrix_pattern,
    )

    np.testing.assert_allclose(np.asarray(pattern.elastic_values), np.asarray(ref), rtol=1e-11, atol=1e-11)


def test_prepare_bddc_subdomain_pattern_elastic_values_keep_constrained_diagonals() -> None:
    vertices = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 0.1, 0.4, 0.9, 1.6],
            [0.0, 0.01, 0.08, 0.27, 0.64],
        ],
        dtype=np.float64,
    )
    tet4 = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    coord, elem = _elevated_tetra_mesh(4, tet4, vertices)
    q_mask = np.ones((3, coord.shape[1]), dtype=bool)
    q_mask[:, 0] = False
    owned_node_range = (0, int(coord.shape[1]))
    pattern = prepare_bddc_subdomain_pattern(
        coord,
        elem,
        q_mask,
        np.zeros(elem.shape[1], dtype=np.int64),
        [MATERIAL],
        owned_node_range,
        elem_type="P4",
    )

    ref = _legacy_projected_bddc_elastic_values(
        coord,
        elem,
        q_mask,
        np.zeros(elem.shape[1], dtype=np.int64),
        [MATERIAL],
        owned_node_range,
        elem_type="P4",
        local_matrix_pattern=pattern.local_matrix_pattern,
    )

    np.testing.assert_allclose(np.asarray(pattern.elastic_values), np.asarray(ref), rtol=1e-11, atol=1e-11)
    assert np.all(np.asarray(pattern.elastic_values)[np.asarray(pattern.constrained_diag_positions, dtype=np.int64)] == 1.0)
