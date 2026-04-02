from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from petsc4py import PETSc
from scipy.sparse import identity

from slope_stability.cli.run_3D_hetero_SSR_capture import run_capture
import slope_stability.cli.run_3D_hetero_SSR_capture as run_capture_mod
from slope_stability.core.config import ContinuationConfig, LinearSolverConfig, MaterialConfig, Problem3DConfig, Run3DSSRConfig
from slope_stability.core.simplex_lagrange import tetra_lagrange_node_tuples, tetra_reference_nodes
from slope_stability.fem.basis import local_basis_volume_3d
from slope_stability.linear.pmg import (
    ElasticPMGHierarchy,
    GeneralPMGHierarchy,
    PMGLevel,
    _adjacent_level_prolongation,
    _cross_mesh_p1_to_p1_prolongation,
    build_3d_mixed_pmg_chain_hierarchy,
    build_3d_mixed_pmg_hierarchy,
    build_3d_mixed_pmg_hierarchy_with_intermediate_p2,
    build_3d_same_mesh_pmg_hierarchy,
)
from slope_stability.linear.solver import PetscMatlabExactDFGMRESSolver, _ManualPMGShellPC
from slope_stability.utils import q_to_free_indices, to_petsc_aij_matrix


ROOT = Path(__file__).resolve().parents[1]
MESH_PATH = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
MESH_PATH_L2 = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L2.msh"
MESH_PATH_L3 = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L3.msh"


def _single_tetra_mesh(order: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    tuples = tetra_lagrange_node_tuples(int(order))
    coord = np.empty((3, len(tuples)), dtype=np.float64)
    for idx, counts in enumerate(tuples):
        bary = np.asarray(counts, dtype=np.float64) / float(order)
        coord[:, idx] = vertices @ bary
    elem = np.arange(len(tuples), dtype=np.int64)[:, None]
    return coord, elem


def _dummy_level(order: int, *, q_mask: np.ndarray | None = None, owned_nodes: tuple[int, int] | None = None) -> PMGLevel:
    coord, elem = _single_tetra_mesh(order)
    n_nodes = coord.shape[1]
    dim = coord.shape[0]
    q = np.ones((dim, n_nodes), dtype=bool) if q_mask is None else np.asarray(q_mask, dtype=bool)
    free_total = q_to_free_indices(q)
    total_to_free_orig = np.full(dim * n_nodes, -1, dtype=np.int64)
    total_to_free_orig[free_total] = np.arange(free_total.size, dtype=np.int64)
    perm = np.arange(free_total.size, dtype=np.int64)
    iperm = perm.copy()
    node0, node1 = owned_nodes if owned_nodes is not None else (0, n_nodes)
    total_range = (dim * node0, dim * node1)
    lo = int(np.searchsorted(free_total, total_range[0], side="left"))
    hi = int(np.searchsorted(free_total, total_range[1], side="left"))
    surf_nodes = 3 if order == 1 else (6 if order == 2 else 15)
    surf = np.zeros((surf_nodes, 0), dtype=np.int64)
    return PMGLevel(
        order=order,
        elem_type=f"P{order}",
        coord=coord,
        elem=elem,
        surf=surf,
        q_mask=q,
        material_identifier=np.array([0], dtype=np.int64),
        freedofs=free_total.copy(),
        total_to_free_orig=total_to_free_orig,
        perm=perm,
        iperm=iperm,
        owned_node_range=(int(node0), int(node1)),
        owned_total_range=(int(total_range[0]), int(total_range[1])),
        owned_free_range=(lo, hi),
    )


def _custom_level(
    *,
    order: int,
    coord: np.ndarray,
    elem: np.ndarray,
    q_mask: np.ndarray | None = None,
    owned_nodes: tuple[int, int] | None = None,
) -> PMGLevel:
    coord_arr = np.asarray(coord, dtype=np.float64)
    elem_arr = np.asarray(elem, dtype=np.int64)
    n_nodes = int(coord_arr.shape[1])
    dim = int(coord_arr.shape[0])
    q = np.ones((dim, n_nodes), dtype=bool) if q_mask is None else np.asarray(q_mask, dtype=bool)
    free_total = q_to_free_indices(q)
    total_to_free_orig = np.full(dim * n_nodes, -1, dtype=np.int64)
    total_to_free_orig[free_total] = np.arange(free_total.size, dtype=np.int64)
    perm = np.arange(free_total.size, dtype=np.int64)
    iperm = perm.copy()
    node0, node1 = owned_nodes if owned_nodes is not None else (0, n_nodes)
    total_range = (dim * node0, dim * node1)
    lo = int(np.searchsorted(free_total, total_range[0], side="left"))
    hi = int(np.searchsorted(free_total, total_range[1], side="left"))
    surf = np.zeros((3 if order == 1 else 6, 0), dtype=np.int64)
    return PMGLevel(
        order=int(order),
        elem_type=f"P{order}",
        coord=coord_arr,
        elem=elem_arr,
        surf=surf,
        q_mask=q,
        material_identifier=np.zeros(max(1, elem_arr.shape[1]), dtype=np.int64),
        freedofs=free_total.copy(),
        total_to_free_orig=total_to_free_orig,
        perm=perm,
        iperm=iperm,
        owned_node_range=(int(node0), int(node1)),
        owned_total_range=(int(total_range[0]), int(total_range[1])),
        owned_free_range=(lo, hi),
    )


def _tiny_pmg_hierarchy() -> ElasticPMGHierarchy:
    level_p1 = _dummy_level(1)
    level_p2 = _dummy_level(2)
    level_p4 = _dummy_level(4)
    return ElasticPMGHierarchy(
        level_p1=level_p1,
        level_p2=level_p2,
        level_p4=level_p4,
        prolongation_p21=_adjacent_level_prolongation(level_p1, level_p2, coarse_order=1, fine_order=2),
        prolongation_p42=_adjacent_level_prolongation(level_p2, level_p4, coarse_order=2, fine_order=4),
        materials=(),
        mesh_path=MESH_PATH,
        node_ordering="original",
    )


def _tiny_chain_pmg_hierarchy() -> GeneralPMGHierarchy:
    coarse = _dummy_level(1)
    mid0 = _dummy_level(1)
    mid1 = _dummy_level(1)
    fine = _dummy_level(2)
    return GeneralPMGHierarchy(
        levels_tuple=(coarse, mid0, mid1, fine),
        prolongations_tuple=(
            _cross_mesh_p1_to_p1_prolongation(coarse, mid0),
            _cross_mesh_p1_to_p1_prolongation(mid0, mid1),
            _adjacent_level_prolongation(mid1, fine, coarse_order=1, fine_order=2),
        ),
        materials=(),
        mesh_path=MESH_PATH_L2,
        node_ordering="original",
    )


def test_legacy_run_3d_config_accepts_p1() -> None:
    cfg = Run3DSSRConfig(
        problem=Problem3DConfig(
            elem_type="P1",
            mesh_path=MESH_PATH,
            materials=(
                MaterialConfig(
                    name="soil",
                    c0=15.0,
                    phi=30.0,
                    psi=0.0,
                    young=10000.0,
                    poisson=0.33,
                    gamma_sat=19.0,
                    gamma_unsat=19.0,
                ),
            ),
        ),
        continuation=ContinuationConfig(step_max=1),
        linear_solver=LinearSolverConfig(),
    )

    assert cfg.validate().problem.elem_type == "P1"


def test_level_metadata_tracks_free_dofs_and_owned_range() -> None:
    q_mask = np.array(
        [
            [False, True, True, True],
            [True, False, True, True],
            [True, True, True, False],
        ],
        dtype=bool,
    )
    level = _dummy_level(1, q_mask=q_mask, owned_nodes=(1, 4))

    assert np.array_equal(level.freedofs, np.array([1, 2, 3, 5, 6, 7, 8, 9, 10], dtype=np.int64))
    assert level.total_to_free_orig[0] == -1
    assert level.total_to_free_orig[1] == 0
    assert level.total_to_free_orig[10] == 8
    assert np.array_equal(level.perm, np.arange(level.free_size, dtype=np.int64))
    assert np.array_equal(level.iperm, np.arange(level.free_size, dtype=np.int64))
    assert level.owned_total_range == (3, 12)
    assert level.owned_free_range == (2, 9)


def test_p21_scalar_prolongation_matches_p1_basis_on_p2_nodes() -> None:
    level_p1 = _dummy_level(1)
    level_p2 = _dummy_level(2)
    transfer = _adjacent_level_prolongation(level_p1, level_p2, coarse_order=1, fine_order=2)

    scalar = transfer.local_matrix[0::3, 0::3].toarray()
    expected = local_basis_volume_3d("P1", tetra_reference_nodes(2))[0].T
    assert scalar.shape == expected.shape
    assert np.allclose(scalar, expected, atol=1.0e-12)


def test_p42_scalar_prolongation_matches_p2_basis_on_p4_nodes() -> None:
    level_p2 = _dummy_level(2)
    level_p4 = _dummy_level(4)
    transfer = _adjacent_level_prolongation(level_p2, level_p4, coarse_order=2, fine_order=4)

    scalar = transfer.local_matrix[0::3, 0::3].toarray()
    expected = local_basis_volume_3d("P2", tetra_reference_nodes(4))[0].T
    assert scalar.shape == expected.shape
    assert np.allclose(scalar, expected, atol=1.0e-12)


def test_vector_prolongation_uses_componentwise_identity_blocks() -> None:
    level_p1 = _dummy_level(1)
    level_p2 = _dummy_level(2)
    transfer = _adjacent_level_prolongation(level_p1, level_p2, coarse_order=1, fine_order=2)

    xx = transfer.local_matrix[0::3, 0::3].toarray()
    yy = transfer.local_matrix[1::3, 1::3].toarray()
    zz = transfer.local_matrix[2::3, 2::3].toarray()
    xy = transfer.local_matrix[0::3, 1::3].toarray()
    yz = transfer.local_matrix[1::3, 2::3].toarray()

    assert np.allclose(xx, yy, atol=1.0e-12)
    assert np.allclose(xx, zz, atol=1.0e-12)
    assert np.allclose(xy, 0.0, atol=1.0e-12)
    assert np.allclose(yz, 0.0, atol=1.0e-12)


def test_transfer_omits_constrained_dofs_in_free_space() -> None:
    coarse_q = np.array(
        [
            [True, False, True, True],
            [True, True, True, False],
            [True, True, False, True],
        ],
        dtype=bool,
    )
    fine_coord, _ = _single_tetra_mesh(2)
    fine_q = np.ones((3, fine_coord.shape[1]), dtype=bool)
    fine_q[0, 0] = False
    fine_q[1, 1] = False
    fine_q[2, 2] = False

    coarse = _dummy_level(1, q_mask=coarse_q)
    fine = _dummy_level(2, q_mask=fine_q)
    transfer = _adjacent_level_prolongation(coarse, fine, coarse_order=1, fine_order=2)

    assert transfer.global_shape == (fine.free_size, coarse.free_size)
    assert np.all(transfer.coo_rows >= fine.lo)
    assert np.all(transfer.coo_rows < fine.hi)
    assert np.all(fine.total_to_free_orig[fine.freedofs] >= 0)
    assert np.all(coarse.total_to_free_orig[coarse.freedofs] >= 0)


def test_cross_mesh_p11_prolongation_matches_p1_basis_at_offgrid_nodes() -> None:
    coarse_coord = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    fine_coord = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.25],
            [0.0, 0.0, 1.0, 0.0, 0.25],
            [0.0, 0.0, 0.0, 1.0, 0.25],
        ],
        dtype=np.float64,
    )
    coarse = _custom_level(order=1, coord=coarse_coord, elem=np.array([[0], [1], [2], [3]], dtype=np.int64))
    fine = _custom_level(order=1, coord=fine_coord, elem=np.array([[0], [1], [2], [3]], dtype=np.int64))

    transfer = _cross_mesh_p1_to_p1_prolongation(coarse, fine)

    scalar = transfer.local_matrix[0::3, 0::3].toarray()
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=np.float64,
    )
    assert scalar.shape == expected.shape
    assert np.allclose(scalar, expected, atol=1.0e-12)


def test_build_3d_mixed_pmg_hierarchy_smoke_on_real_l1_l2_meshes() -> None:
    hierarchy = build_3d_mixed_pmg_hierarchy(
        MESH_PATH_L2,
        MESH_PATH,
        fine_elem_type="P2",
        node_ordering="original",
        comm=PETSc.COMM_SELF,
    )

    levels = hierarchy.levels
    assert [level.order for level in levels] == [1, 1, 2]
    assert hierarchy.coarse_level.elem_type == "P1"
    assert hierarchy.mid_level.elem_type == "P1"
    assert hierarchy.fine_level.elem_type == "P2"
    assert hierarchy.prolongation_p21.global_shape == (hierarchy.mid_level.free_size, hierarchy.coarse_level.free_size)
    assert hierarchy.prolongation_p42.global_shape == (hierarchy.fine_level.free_size, hierarchy.mid_level.free_size)
    assert hierarchy.mid_level.n_nodes == 6795
    assert hierarchy.fine_level.n_nodes == 49797
    assert not np.any(np.diff(hierarchy.prolongation_p21.local_matrix.tocsc().indptr) == 0)


def test_build_3d_mixed_pmg_hierarchy_with_intermediate_p2_smoke_on_real_l1_l2_meshes() -> None:
    hierarchy = build_3d_mixed_pmg_hierarchy_with_intermediate_p2(
        MESH_PATH_L2,
        MESH_PATH,
        node_ordering="block_metis",
        comm=PETSc.COMM_SELF,
    )

    levels = hierarchy.levels
    assert isinstance(hierarchy, GeneralPMGHierarchy)
    assert [level.order for level in levels] == [1, 1, 2, 4]
    assert levels[0].elem_type == "P1"
    assert levels[1].elem_type == "P1"
    assert levels[2].elem_type == "P2"
    assert levels[3].elem_type == "P4"
    assert len(hierarchy.prolongations) == 3
    assert hierarchy.prolongations[0].global_shape == (levels[1].free_size, levels[0].free_size)
    assert hierarchy.prolongations[1].global_shape == (levels[2].free_size, levels[1].free_size)
    assert hierarchy.prolongations[2].global_shape == (levels[3].free_size, levels[2].free_size)
    assert levels[0].free_size < levels[1].free_size < levels[2].free_size < levels[3].free_size


def test_mixed_pmg_shell_vector_hypre_uses_direct_elastic_coarse_operator_on_real_meshes() -> None:
    hierarchy = build_3d_mixed_pmg_hierarchy(
        MESH_PATH_L2,
        MESH_PATH,
        fine_elem_type="P2",
        node_ordering="original",
        comm=PETSc.COMM_SELF,
    )
    fine = hierarchy.level_p4
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="GAMG",
        q_mask=fine.q_mask,
        coord=fine.coord,
        preconditioner_options={
            "pc_backend": "pmg_shell",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "mpi_distribute_by_nodes": False,
            "full_system_preconditioner": False,
            "pmg_hierarchy": hierarchy,
            "mg_coarse_ksp_type": "cg",
            "pc_hypre_boomeramg_numfunctions": 3,
            "pc_hypre_boomeramg_nodal_coarsen": 6,
            "pc_hypre_boomeramg_vec_interp_variant": 3,
            "pc_hypre_boomeramg_max_iter": 4,
            "pc_hypre_boomeramg_tol": 0.0,
        },
    )
    A = identity(fine.free_size, format="csr", dtype=np.float64)

    solver.setup_preconditioner(A)
    diagnostics = solver.get_preconditioner_diagnostics()

    assert diagnostics["manualmg_coarse_full_system"] is True
    assert diagnostics["manualmg_coarse_operator_source"] == "direct_elastic_full_system"
    assert diagnostics["manualmg_coarse_solve_global_size"] == hierarchy.level_p1.total_size
    assert diagnostics["manualmg_coarse_free_global_size"] == hierarchy.level_p1.free_size
    solver._reset_petsc_objects()


def test_mixed_parallel_pmg_shell_defaults_to_chebyshev_jacobi_smoothers() -> None:
    hierarchy = build_3d_mixed_pmg_hierarchy(
        MESH_PATH_L2,
        MESH_PATH,
        fine_elem_type="P2",
        node_ordering="original",
        comm=PETSc.COMM_SELF,
    )
    fine = hierarchy.level_p4
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="GAMG",
        q_mask=fine.q_mask,
        coord=fine.coord,
        preconditioner_options={
            "pc_backend": "pmg_shell",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "mpi_distribute_by_nodes": False,
            "full_system_preconditioner": False,
            "pmg_hierarchy": hierarchy,
        },
    )

    pc = _ManualPMGShellPC(solver)
    assert pc._default_smoother_options(hierarchy, comm_size=8) == (
        PETSc.KSP.Type.CHEBYSHEV,
        PETSc.PC.Type.JACOBI,
        3,
    )
    assert pc._default_smoother_options(hierarchy, comm_size=1) == (
        PETSc.KSP.Type.RICHARDSON,
        PETSc.PC.Type.SOR,
        3,
    )


def test_pmg_backend_builds_three_level_petsc_mg_with_reference_defaults() -> None:
    hierarchy = _tiny_pmg_hierarchy()
    fine = hierarchy.level_p4
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="GAMG",
        q_mask=fine.q_mask,
        coord=fine.coord,
        preconditioner_options={
            "pc_backend": "pmg",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "mpi_distribute_by_nodes": True,
            "full_system_preconditioner": False,
            "pc_mg_galerkin": "both",
            "pmg_hierarchy": hierarchy,
        },
    )
    A = identity(fine.free_size, format="csr", dtype=np.float64)

    solver.setup_preconditioner(A, full_matrix=A)

    pc = solver._inner_ksp.getPC()
    assert pc.getType() == PETSc.PC.Type.MG
    assert int(pc.getMGLevels()) == 3
    assert solver._pmg_state is not None
    assert solver._pmg_state.restrictions[0] is not None
    assert solver._pmg_state.restrictions[1] is not None
    assert tuple(int(v) for v in pc.getMGRestriction(1).getSize()) == tuple(
        int(v) for v in solver._pmg_state.restrictions[0].getSize()
    )
    assert tuple(int(v) for v in pc.getMGRestriction(2).getSize()) == tuple(
        int(v) for v in solver._pmg_state.restrictions[1].getSize()
    )
    coarse = pc.getMGCoarseSolve()
    assert coarse.getType() == PETSc.KSP.Type.PREONLY
    assert coarse.getPC().getType() == PETSc.PC.Type.LU
    for level_idx in (1, 2):
        smoother = pc.getMGSmoother(level_idx)
        assert smoother.getType() == PETSc.KSP.Type.RICHARDSON
        assert smoother.getPC().getType() == PETSc.PC.Type.SOR
    opts = PETSc.Options()
    assert opts.getString(f"{solver._options_prefix}inner_pc_mg_galerkin") == "both"
    diagnostics = solver.get_preconditioner_diagnostics()
    assert diagnostics["pmg_coarse_pc_type"] == PETSc.PC.Type.LU
    assert diagnostics["pmg_level_global_sizes"] == [level.free_size for level in (hierarchy.level_p1, hierarchy.level_p2, hierarchy.level_p4)]
    solver._reset_petsc_objects()


def test_pmg_nested_options_reach_coarse_and_smoothers_after_setup() -> None:
    hierarchy = _tiny_pmg_hierarchy()
    fine = hierarchy.level_p4
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="GAMG",
        q_mask=fine.q_mask,
        coord=fine.coord,
        preconditioner_options={
            "pc_backend": "pmg",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "mpi_distribute_by_nodes": True,
            "full_system_preconditioner": False,
            "pc_mg_galerkin": "both",
            "mg_levels_ksp_type": "gmres",
            "mg_levels_pc_type": "jacobi",
            "mg_levels_ksp_max_it": 4,
            "mg_coarse_ksp_type": "cg",
            "mg_coarse_pc_type": "jacobi",
            "pmg_hierarchy": hierarchy,
        },
    )
    A = identity(fine.free_size, format="csr", dtype=np.float64)

    solver.setup_preconditioner(A, full_matrix=A)

    pc = solver._inner_ksp.getPC()
    coarse = pc.getMGCoarseSolve()
    assert coarse.getType() == PETSc.KSP.Type.CG
    assert coarse.getPC().getType() == PETSc.PC.Type.JACOBI
    for level_idx in (1, 2):
        smoother = pc.getMGSmoother(level_idx)
        assert smoother.getType() == PETSc.KSP.Type.GMRES
        assert smoother.getPC().getType() == PETSc.PC.Type.JACOBI
    solver._reset_petsc_objects()


def test_pmg_shell_backend_builds_manual_vcycle_with_hypre_coarse() -> None:
    hierarchy = _tiny_pmg_hierarchy()
    fine = hierarchy.level_p4
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="GAMG",
        q_mask=fine.q_mask,
        coord=fine.coord,
        preconditioner_options={
            "pc_backend": "pmg_shell",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "mpi_distribute_by_nodes": False,
            "full_system_preconditioner": False,
            "pmg_hierarchy": hierarchy,
        },
    )
    A = identity(fine.free_size, format="csr", dtype=np.float64)

    solver.setup_preconditioner(A)

    diagnostics = solver.get_preconditioner_diagnostics()
    assert diagnostics["manualmg_levels"] == 3
    assert diagnostics["manualmg_coarse_pc_type"] == PETSc.PC.Type.HYPRE
    assert diagnostics["manualmg_coarse_ksp_type"] == PETSc.KSP.Type.PREONLY
    assert diagnostics["manualmg_coarse_hypre_type"] == "boomeramg"

    out = solver._apply_inner_preconditioner_local(np.ones(fine.free_size, dtype=np.float64))
    assert out.shape == (fine.free_size,)
    assert np.all(np.isfinite(out))
    diagnostics = solver.get_preconditioner_diagnostics()
    assert diagnostics["manualmg_apply_count"] == 1
    assert diagnostics["manualmg_coarse_solve_count"] == 1
    assert diagnostics["manualmg_coarse_hypre_time_total_s"] >= 0.0
    assert diagnostics["manualmg_coarse_iteration_count_mode"] == "coarse_ksp_iterations_only"
    solver._reset_petsc_objects()


def test_pmg_shell_backend_accepts_multilevel_p1_tail_hierarchy() -> None:
    hierarchy = _tiny_chain_pmg_hierarchy()
    fine = hierarchy.fine_level
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="GAMG",
        q_mask=fine.q_mask,
        coord=fine.coord,
        preconditioner_options={
            "pc_backend": "pmg_shell",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "mpi_distribute_by_nodes": False,
            "full_system_preconditioner": False,
            "pmg_hierarchy": hierarchy,
        },
    )
    A = identity(fine.free_size, format="csr", dtype=np.float64)

    solver.setup_preconditioner(A)

    diagnostics = solver.get_preconditioner_diagnostics()
    assert diagnostics["manualmg_levels"] == 4
    assert diagnostics["manualmg_level_orders"] == [1, 1, 1, 2]
    assert len(diagnostics["manualmg_transfer_shapes"]) == 3
    out = solver._apply_inner_preconditioner_local(np.ones(fine.free_size, dtype=np.float64))
    assert out.shape == (fine.free_size,)
    assert np.all(np.isfinite(out))
    solver._reset_petsc_objects()


def test_pmg_backends_accept_mixed_p1_p1_p2_hierarchy() -> None:
    coarse = _dummy_level(1)
    mid = _dummy_level(1)
    fine = _dummy_level(2)
    hierarchy = ElasticPMGHierarchy(
        level_p1=coarse,
        level_p2=mid,
        level_p4=fine,
        prolongation_p21=_cross_mesh_p1_to_p1_prolongation(coarse, mid),
        prolongation_p42=_adjacent_level_prolongation(mid, fine, coarse_order=1, fine_order=2),
        materials=(),
        mesh_path=MESH_PATH_L2,
        node_ordering="original",
    )
    A = identity(fine.free_size, format="csr", dtype=np.float64)

    for backend in ("pmg", "pmg_shell"):
        solver = PetscMatlabExactDFGMRESSolver(
            pc_type="GAMG",
            q_mask=fine.q_mask,
            coord=fine.coord,
            preconditioner_options={
                "pc_backend": backend,
                "preconditioner_matrix_source": "tangent",
                "preconditioner_matrix_policy": "current",
                "preconditioner_rebuild_policy": "every_newton",
                "preconditioner_rebuild_interval": 1,
                "mpi_distribute_by_nodes": True,
                "full_system_preconditioner": False,
                "pc_mg_galerkin": "both",
                "pmg_hierarchy": hierarchy,
            },
        )
        solver.setup_preconditioner(A, full_matrix=A)
        diagnostics = solver.get_preconditioner_diagnostics()
        level_key = "pmg_level_orders" if backend == "pmg" else "manualmg_level_orders"
        assert diagnostics[level_key] == [1, 1, 2]
    solver._reset_petsc_objects()


def test_build_3d_mixed_pmg_chain_hierarchy_smoke_on_real_meshes() -> None:
    hierarchy = build_3d_mixed_pmg_chain_hierarchy(
        MESH_PATH_L3,
        [MESH_PATH_L2, MESH_PATH],
        fine_elem_type="P2",
        boundary_type=0,
        node_ordering="original",
        material_rows=None,
        comm=PETSc.COMM_WORLD,
    )

    assert [level.order for level in hierarchy.levels] == [1, 1, 1, 2]
    assert len(hierarchy.prolongations) == 3
    assert hierarchy.fine_level.elem_type == "P2"
    assert hierarchy.coarse_level.elem_type == "P1"


def test_pmg_shell_richardson_coarse_hypre_reports_inner_iterations() -> None:
    hierarchy = _tiny_pmg_hierarchy()
    fine = hierarchy.level_p4
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="GAMG",
        q_mask=fine.q_mask,
        coord=fine.coord,
        preconditioner_options={
            "pc_backend": "pmg_shell",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "mpi_distribute_by_nodes": False,
            "full_system_preconditioner": False,
            "pmg_hierarchy": hierarchy,
            "mg_coarse_ksp_type": "richardson",
            "mg_coarse_max_it": 1,
            "pc_hypre_boomeramg_max_iter": 2,
            "pc_hypre_boomeramg_tol": 0.0,
        },
    )
    A = identity(fine.free_size, format="csr", dtype=np.float64)

    solver.setup_preconditioner(A)
    out = solver._apply_inner_preconditioner_local(np.ones(fine.free_size, dtype=np.float64))

    assert out.shape == (fine.free_size,)
    diagnostics = solver.get_preconditioner_diagnostics()
    assert diagnostics["manualmg_coarse_ksp_type"] == PETSc.KSP.Type.RICHARDSON
    assert diagnostics["manualmg_coarse_iteration_count_mode"] == "hypre_inner_v_cycles_via_pcapplyrichardson"
    assert diagnostics["manualmg_coarse_hypre_inner_iterations_total"] >= 1
    solver._reset_petsc_objects()


def test_pmg_shell_can_opt_in_to_vector_hypre_coarse_options() -> None:
    hierarchy = _tiny_pmg_hierarchy()
    fine = hierarchy.level_p4
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="GAMG",
        q_mask=fine.q_mask,
        coord=fine.coord,
        preconditioner_options={
            "pc_backend": "pmg_shell",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "mpi_distribute_by_nodes": False,
            "full_system_preconditioner": False,
            "pmg_hierarchy": hierarchy,
            "allow_unsafe_hypre_vector_options": True,
            "pc_hypre_boomeramg_numfunctions": 3,
            "pc_hypre_boomeramg_nodal_coarsen": 4,
            "pc_hypre_boomeramg_nodal_coarsen_diag": 1,
            "pc_hypre_boomeramg_vec_interp_variant": 2,
            "pc_hypre_boomeramg_vec_interp_qmax": 4,
            "pc_hypre_boomeramg_vec_interp_smooth": True,
        },
    )
    A = identity(fine.free_size, format="csr", dtype=np.float64)

    solver.setup_preconditioner(A)
    diagnostics = solver.get_preconditioner_diagnostics()

    assert diagnostics["manualmg_coarse_full_system"] is True
    assert diagnostics["manualmg_coarse_block_size"] == 3
    assert diagnostics["manualmg_coarse_pc_hypre_boomeramg_numfunctions"] == "3"
    assert diagnostics["manualmg_coarse_pc_hypre_boomeramg_nodal_coarsen"] == "4"
    assert diagnostics["manualmg_coarse_pc_hypre_boomeramg_vec_interp_variant"] == "2"
    assert diagnostics["manualmg_coarse_pc_hypre_boomeramg_vec_interp_qmax"] == "4"
    assert diagnostics["manualmg_coarse_pc_hypre_boomeramg_vec_interp_smooth"] == "true"
    solver._reset_petsc_objects()


def test_pmg_shell_full_system_coarse_handles_constrained_dofs_with_vector_hypre() -> None:
    coarse = _dummy_level(
        1,
        q_mask=np.array(
            [
                [True, False, True, True],
                [True, True, True, False],
                [True, True, False, True],
            ],
            dtype=bool,
        ),
    )
    mid = _dummy_level(1)
    fine = _dummy_level(2)
    hierarchy = ElasticPMGHierarchy(
        level_p1=coarse,
        level_p2=mid,
        level_p4=fine,
        prolongation_p21=_cross_mesh_p1_to_p1_prolongation(coarse, mid),
        prolongation_p42=_adjacent_level_prolongation(mid, fine, coarse_order=1, fine_order=2),
        materials=(),
        mesh_path=MESH_PATH_L2,
        node_ordering="original",
    )
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="GAMG",
        q_mask=fine.q_mask,
        coord=fine.coord,
        preconditioner_options={
            "pc_backend": "pmg_shell",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "mpi_distribute_by_nodes": False,
            "full_system_preconditioner": False,
            "pmg_hierarchy": hierarchy,
            "mg_coarse_ksp_type": "cg",
            "pc_hypre_boomeramg_numfunctions": 3,
            "pc_hypre_boomeramg_nodal_coarsen": 6,
            "pc_hypre_boomeramg_vec_interp_variant": 3,
            "pc_hypre_boomeramg_max_iter": 4,
            "pc_hypre_boomeramg_tol": 0.0,
        },
    )
    A = identity(fine.free_size, format="csr", dtype=np.float64)

    solver.setup_preconditioner(A)
    out = solver._apply_inner_preconditioner_local(np.ones(fine.free_size, dtype=np.float64))

    diagnostics = solver.get_preconditioner_diagnostics()
    assert out.shape == (fine.free_size,)
    assert np.all(np.isfinite(out))
    assert diagnostics["manualmg_coarse_full_system"] is True
    assert diagnostics["manualmg_coarse_free_global_size"] < diagnostics["manualmg_coarse_solve_global_size"]
    assert diagnostics["manualmg_coarse_solve_global_size"] == hierarchy.level_p1.total_size
    solver._reset_petsc_objects()


def test_pmg_rebuild_preserves_explicit_coarse_hypre_override() -> None:
    hierarchy = _tiny_pmg_hierarchy()
    fine = hierarchy.level_p4
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="GAMG",
        q_mask=fine.q_mask,
        coord=fine.coord,
        preconditioner_options={
            "pc_backend": "pmg",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "mpi_distribute_by_nodes": True,
            "full_system_preconditioner": False,
            "pc_mg_galerkin": "both",
            "mg_coarse_ksp_type": "preonly",
            "mg_coarse_pc_type": "hypre",
            "mg_coarse_pc_hypre_type": "boomeramg",
            "pmg_hierarchy": hierarchy,
        },
    )
    A = identity(fine.free_size, format="csr", dtype=np.float64)

    solver.setup_preconditioner(A, full_matrix=A)
    solver.setup_preconditioner(A, full_matrix=A)

    coarse = solver._inner_ksp.getPC().getMGCoarseSolve()
    assert coarse.getType() == PETSc.KSP.Type.PREONLY
    assert coarse.getPC().getType() == PETSc.PC.Type.HYPRE
    assert coarse.getPC().getHYPREType() == "boomeramg"
    solver._reset_petsc_objects()


def test_pmg_microbenchmark_reports_apply_and_level_metadata() -> None:
    hierarchy = _tiny_pmg_hierarchy()
    fine = hierarchy.level_p4
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="GAMG",
        q_mask=fine.q_mask,
        coord=fine.coord,
        preconditioner_options={
            "pc_backend": "pmg",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "mpi_distribute_by_nodes": True,
            "full_system_preconditioner": False,
            "pc_mg_galerkin": "both",
            "pmg_hierarchy": hierarchy,
        },
    )
    A = identity(fine.free_size, format="csr", dtype=np.float64)

    solver.setup_preconditioner(A, full_matrix=A)
    info = solver.run_pmg_microbenchmark(np.ones(fine.free_size, dtype=np.float64))

    assert info["status"] == "completed"
    assert float(info["pc_apply_elapsed_s"]) >= 0.0
    diagnostics = solver.get_preconditioner_diagnostics()
    assert "pmg_microbenchmark" in diagnostics
    assert diagnostics["pmg_microbenchmark"]["phase"] == "microbenchmark"
    assert diagnostics["pmg_microbenchmark"]["pmg_coarse_subcomm_size"] == 1
    solver._reset_petsc_objects()


def test_pmg_release_preserves_ksp_for_next_newton_setup() -> None:
    hierarchy = _tiny_pmg_hierarchy()
    fine = hierarchy.level_p4
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="GAMG",
        q_mask=fine.q_mask,
        coord=fine.coord,
        preconditioner_options={
            "pc_backend": "pmg",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "mpi_distribute_by_nodes": True,
            "full_system_preconditioner": False,
            "pc_mg_galerkin": "both",
            "pmg_hierarchy": hierarchy,
        },
    )
    A1 = to_petsc_aij_matrix(identity(fine.free_size, format="csr", dtype=np.float64), comm=PETSc.COMM_SELF)
    A2 = to_petsc_aij_matrix(2.0 * identity(fine.free_size, format="csr", dtype=np.float64), comm=PETSc.COMM_SELF)

    try:
        solver.setup_preconditioner(A1, full_matrix=A1)
        first_ksp = int(solver._inner_ksp.handle)
        solver.release_iteration_resources()
        A1.destroy()
        solver.setup_preconditioner(A2, full_matrix=A2)

        assert solver._inner_ksp is not None
        assert int(solver._inner_ksp.handle) == first_ksp
        diagnostics = solver.get_preconditioner_diagnostics()
        assert diagnostics["preconditioner_rebuild_count"] == 2
    finally:
        solver.close()
        A2.destroy()


def test_solver_rejects_non_tangent_pmg_source() -> None:
    hierarchy = _tiny_pmg_hierarchy()
    with pytest.raises(ValueError, match="preconditioner_matrix_source"):
        PetscMatlabExactDFGMRESSolver(
            pc_type="GAMG",
            q_mask=hierarchy.level_p4.q_mask,
            coord=hierarchy.level_p4.coord,
            preconditioner_options={
                "pc_backend": "pmg",
                "preconditioner_matrix_source": "elastic",
                "full_system_preconditioner": False,
                "pmg_hierarchy": hierarchy,
            },
        )


def test_run_capture_rejects_pmg_on_non_p4(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="P4"):
        run_capture(
            tmp_path / "pmg_invalid_elem",
            mesh_path=MESH_PATH,
            elem_type="P2",
            pc_backend="pmg",
            preconditioner_matrix_source="tangent",
            step_max=1,
        )


def test_run_capture_rejects_non_tangent_pmg_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="preconditioner_matrix_source"):
        run_capture(
            tmp_path / "pmg_invalid_source",
            mesh_path=MESH_PATH,
            elem_type="P4",
            pc_backend="pmg",
            preconditioner_matrix_source="elastic",
            step_max=1,
        )


def test_run_capture_accepts_mixed_p2_pmg_shell_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _sentinel_builder(*args, **kwargs):
        raise RuntimeError("mixed-pmg-sentinel")

    monkeypatch.setattr(run_capture_mod, "build_3d_mixed_pmg_hierarchy", _sentinel_builder)

    with pytest.raises(RuntimeError, match="mixed-pmg-sentinel"):
        run_capture(
            tmp_path / "pmg_shell_mixed_p2",
            mesh_path=MESH_PATH_L2,
            elem_type="P2",
            pc_backend="pmg_shell",
            pmg_coarse_mesh_path=MESH_PATH,
            preconditioner_matrix_source="tangent",
            node_ordering="original",
            step_max=1,
        )


def test_run_capture_accepts_mixed_p4_with_intermediate_p2_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _sentinel_builder(*args, **kwargs):
        raise RuntimeError("mixed-pmg-p4-p2-sentinel")

    monkeypatch.setattr(run_capture_mod, "build_3d_mixed_pmg_hierarchy_with_intermediate_p2", _sentinel_builder)

    with pytest.raises(RuntimeError, match="mixed-pmg-p4-p2-sentinel"):
        run_capture(
            tmp_path / "pmg_shell_mixed_p4_p2",
            mesh_path=MESH_PATH_L2,
            elem_type="P4",
            pc_backend="pmg_shell",
            pmg_coarse_mesh_path=MESH_PATH,
            pmg_fine_hierarchy_mode="p4_p2_intermediate",
            preconditioner_matrix_source="tangent",
            node_ordering="block_metis",
            step_max=1,
        )


def test_build_3d_same_mesh_p2_pmg_hierarchy_smoke_on_real_mesh() -> None:
    hierarchy = build_3d_same_mesh_pmg_hierarchy(
        MESH_PATH,
        fine_elem_type="P2",
        boundary_type=0,
        node_ordering="original",
        reorder_parts=1,
        comm=PETSc.COMM_SELF,
    )
    assert isinstance(hierarchy, GeneralPMGHierarchy)
    assert tuple(level.order for level in hierarchy.levels) == (1, 2)
    assert len(hierarchy.prolongations) == 1
    assert hierarchy.fine_level.free_size > hierarchy.coarse_level.free_size > 0


def test_run_capture_accepts_same_mesh_p2_pmg_shell_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _sentinel_builder(*args, **kwargs):
        raise RuntimeError("same-mesh-pmg-sentinel")

    monkeypatch.setattr(run_capture_mod, "build_3d_same_mesh_pmg_hierarchy", _sentinel_builder)

    with pytest.raises(RuntimeError, match="same-mesh-pmg-sentinel"):
        run_capture(
            tmp_path / "pmg_shell_same_mesh_p2",
            mesh_path=MESH_PATH,
            elem_type="P2",
            pc_backend="pmg_shell",
            preconditioner_matrix_source="tangent",
            node_ordering="original",
            step_max=1,
        )

