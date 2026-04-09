from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    probe_path = ROOT / "benchmarks" / "slope_stability_3D_hetero_SSR_default" / "archive" / "probe_hypre_frozen.py"
    spec = importlib.util.spec_from_file_location("probe_hypre_frozen", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_petsc_opt_entries_accepts_key_value_pairs() -> None:
    module = _load_module()
    parsed = module._parse_petsc_opt_entries(
        [
            "pc_hypre_boomeramg_max_iter=2",
            "ksp_view=true",
        ]
    )

    assert parsed == {
        "pc_hypre_boomeramg_max_iter": "2",
        "ksp_view": "true",
    }


def test_build_preconditioner_options_coerces_bool_and_numeric_petsc_opts() -> None:
    module = _load_module()
    args = SimpleNamespace(
        pc_backend="pmg_shell",
        pmat_source="tangent",
        max_deflation_basis_vectors=16,
        pc_hypre_coarsen_type=None,
        pc_hypre_interp_type=None,
        pc_hypre_strong_threshold=None,
        pc_hypre_boomeramg_max_iter=None,
        pc_hypre_P_max=None,
        pc_hypre_agg_nl=None,
        pc_hypre_nongalerkin_tol=None,
        petsc_opt=[
            "full_system_preconditioner=false",
            "mg_levels_ksp_max_it=5",
            "pc_hypre_boomeramg_tol=0.0",
            "mg_levels_ksp_type=chebyshev",
        ],
    )
    problem = {"pmg_hierarchy": object()}

    options = module._build_preconditioner_options(args, problem)

    assert options["full_system_preconditioner"] is False
    assert options["mg_levels_ksp_max_it"] == 5
    assert options["pc_hypre_boomeramg_tol"] == 0.0
    assert options["mg_levels_ksp_type"] == "chebyshev"


def test_native_pc_type_accepts_hmg_and_gamg() -> None:
    module = _load_module()

    assert module._native_pc_type("hmg") is not None
    assert module._native_pc_type("gamg") is not None


def test_repo_pmg_solver_type_guard_accepts_plain_kspfgmres() -> None:
    module = _load_module()

    assert module._supports_repo_pmg_solver_type("PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE") is True
    assert module._supports_repo_pmg_solver_type("KSPFGMRES") is True
    assert module._supports_repo_pmg_solver_type("KSPFGMRES_hypre") is True
    assert module._supports_repo_pmg_solver_type("GMRES") is False


def test_rank_hint_from_path_reads_rank_component() -> None:
    module = _load_module()

    assert module._rank_hint_from_path(Path("artifacts/foo/rank8_bar/data/petsc_run.npz")) == 8
    assert module._rank_hint_from_path(Path("artifacts/foo/no_rank_here/data/petsc_run.npz")) is None


def test_select_state_uses_step_history_and_run_info_defaults(tmp_path: Path) -> None:
    module = _load_module()
    state_npz = tmp_path / "petsc_run.npz"
    state_run_info = tmp_path / "run_info.json"

    np.savez(
        state_npz,
        U=np.full((3, 5), 9.0, dtype=np.float64),
        step_U=np.stack(
            [
                np.full((3, 5), 1.0, dtype=np.float64),
                np.full((3, 5), 2.0, dtype=np.float64),
                np.full((3, 5), 3.0, dtype=np.float64),
            ],
            axis=0,
        ),
        lambda_hist=np.array([1.0, 1.1, 1.25], dtype=np.float64),
        omega_hist=np.array([10.0, 11.0, 12.5], dtype=np.float64),
    )
    state_run_info.write_text(
        json.dumps(
            {
                "params": {
                    "node_ordering": "block_metis",
                    "elem_type": "P4",
                    "r_min": 5e-4,
                    "material_rows": [[1, 2, 3, 4, 5, 6, 7]],
                }
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        state_npz=state_npz,
        state_run_info=state_run_info,
        state_selector="hard",
        state_index=None,
        node_ordering="block_metis",
        elem_type="P4",
        regularization_r=None,
        reorder_parts=None,
    )

    run_info = module._resolve_state_run_info(args)
    selected = module._select_state(args, run_info)

    assert selected["selected_index"] == 2
    assert selected["selected_label"] == "step_2"
    assert np.allclose(selected["state_u"], np.full((3, 5), 3.0, dtype=np.float64))
    assert selected["lambda_value"] == 1.25
    assert selected["omega_value"] == 12.5
    assert selected["regularization_r"] == 5e-4
    assert selected["source_node_ordering"] == "block_metis"
    assert selected["source_elem_type"] == "P4"
    assert int(selected["reorder_parts"]) >= 1
