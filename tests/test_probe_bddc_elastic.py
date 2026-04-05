from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from petsc4py import PETSc
from scipy.sparse import csr_matrix

from slope_stability.utils import (
    local_csr_to_petsc_aij_matrix,
    local_csr_to_petsc_matis_matrix,
    release_petsc_aij_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_probe_module():
    probe_path = ROOT / "benchmarks" / "slope_stability_3D_hetero_SSR_default" / "archive" / "probe_bddc_elastic.py"
    spec = importlib.util.spec_from_file_location("probe_bddc_elastic", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_native_bddc_probe_helpers_solve_small_problem() -> None:
    module = _load_probe_module()
    A_local = _local_block_matrix()
    A = local_csr_to_petsc_aij_matrix(
        A_local,
        global_shape=A_local.shape,
        comm=PETSc.COMM_WORLD,
        block_size=2,
    )
    P = local_csr_to_petsc_matis_matrix(
        A_local,
        global_size=4,
        local_to_global=np.array([0, 1, 2, 3], dtype=np.int64),
        comm=PETSc.COMM_WORLD,
        block_size=2,
        metadata={
            "bddc_field_is_local": tuple(
                PETSc.IS().createGeneral(np.asarray([comp, comp + 2], dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
                for comp in range(2)
            ),
            "bddc_dirichlet_local": np.array([], dtype=np.int32),
            "bddc_local_adjacency": (
                np.asarray(A_local.indptr, dtype=PETSc.IntType),
                np.asarray(A_local.indices, dtype=PETSc.IntType),
            ),
            "bddc_local_coordinates": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        },
    )
    args = SimpleNamespace(
        pc_backend="bddc",
        native_ksp_type="cg",
        native_ksp_norm_type="unpreconditioned",
        linear_tolerance=1e-10,
        linear_max_iter=50,
        pc_bddc_symmetric=True,
        pc_bddc_dirichlet_ksp_type="preonly",
        pc_bddc_dirichlet_pc_type="lu",
        pc_bddc_neumann_ksp_type="preonly",
        pc_bddc_neumann_pc_type="lu",
        pc_bddc_coarse_ksp_type="preonly",
        pc_bddc_coarse_pc_type="lu",
        pc_bddc_dirichlet_approximate=None,
        pc_bddc_neumann_approximate=None,
        pc_bddc_monolithic=True,
        pc_bddc_coarse_redundant_pc_type="svd",
        pc_bddc_switch_static=True,
        pc_bddc_use_deluxe_scaling=False,
        pc_bddc_use_vertices=True,
        pc_bddc_use_edges=False,
        pc_bddc_use_faces=False,
        pc_bddc_use_change_of_basis=False,
        pc_bddc_use_change_on_faces=False,
        pc_bddc_check_level=None,
        pc_hypre_coarsen_type=None,
        pc_hypre_interp_type=None,
        use_coordinates=True,
        petsc_opt_map={},
    )

    A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    A.setOption(PETSc.Mat.Option.SPD, True)
    ksp, setup_elapsed = module._build_native_petsc_ksp(args, operator_matrix=A, preconditioning_matrix=P)
    x, solve_elapsed, iterations, reason, residual_history, relative_residual_history = module._native_petsc_ksp_solve_once(
        ksp,
        A,
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )

    assert setup_elapsed >= 0.0
    assert solve_elapsed >= 0.0
    assert iterations >= 1
    assert reason > 0
    assert x.shape == (4,)
    assert np.all(np.isfinite(x))
    assert residual_history
    assert relative_residual_history
    assert len(residual_history) == len(relative_residual_history)

    ksp.destroy()
    release_petsc_aij_matrix(P)
    P.destroy()
    release_petsc_aij_matrix(A)
    A.destroy()


def test_parse_petsc_opt_entries_accepts_key_value_pairs() -> None:
    module = _load_probe_module()
    parsed = module._parse_petsc_opt_entries(
        [
            "pc_bddc_dirichlet_pc_gamg_threshold=0.05",
            "pc_bddc_use_deluxe_scaling=true",
        ]
    )

    assert parsed == {
        "pc_bddc_dirichlet_pc_gamg_threshold": "0.05",
        "pc_bddc_use_deluxe_scaling": "true",
    }


def test_is_global_petsc_option_detects_log_view_family() -> None:
    module = _load_probe_module()

    assert module._is_global_petsc_option("log_view") is True
    assert module._is_global_petsc_option("log_view_memory") is True
    assert module._is_global_petsc_option("options_left") is True
    assert module._is_global_petsc_option("pc_view") is True
    assert module._is_global_petsc_option("pc_bddc_dirichlet_pc_gamg_threshold") is False
