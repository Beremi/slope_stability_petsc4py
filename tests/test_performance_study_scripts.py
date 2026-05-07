from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tomllib

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "docs" / "petsc_matlab_performance_comparison" / "scripts"


def _load_script(name: str):
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"performance_study_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_no_resume_invalidates_stale_completion_files(tmp_path: Path) -> None:
    run_study = _load_script("run_study")

    output_dir = tmp_path / "run"
    (output_dir / "data").mkdir(parents=True)
    for rel in ("data/run_info.json", "data/petsc_run.npz", "summary.json", "summary.h5", "matlab_run.mat", "study_run.json"):
        path = output_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale", encoding="utf-8")

    run_study.invalidate_completion_files(output_dir, "petsc")

    assert not (output_dir / "data" / "run_info.json").exists()
    assert not (output_dir / "data" / "petsc_run.npz").exists()
    assert (output_dir / "study_run.json").exists()

    run_study.invalidate_completion_files(output_dir, "matlab")

    assert not (output_dir / "summary.json").exists()
    assert not (output_dir / "summary.h5").exists()
    assert not (output_dir / "matlab_run.mat").exists()
    assert (output_dir / "study_run.json").exists()


def test_smoke_horizon_uses_one_step_headroom(tmp_path: Path) -> None:
    run_study = _load_script("run_study")

    data_dir = tmp_path / "run" / "data"
    data_dir.mkdir(parents=True)
    np.savez(data_dir / "petsc_run.npz", omega_hist=np.array([2.0, 3.5, 4.0], dtype=np.float64))

    assert run_study.choose_main_horizon_from_smoke(tmp_path / "run", 4.0) == 4.5


def test_collector_rejects_stale_run_settings() -> None:
    collect_results = _load_script("collect_results")

    meta = {
        "case_id": "seepage_3d",
        "d_lambda_diff_scaled_min": 0.0,
        "shared_omega_max_stop": 4_939_008.747926297,
    }
    case_settings = {
        "seepage_3d": {
            "lambda_init": 0.7,
            "d_lambda_init": 0.1,
            "d_lambda_min": 1.0e-5,
            "tol": 5.0e-2,
        }
    }
    params = {
        "lambda_init": 0.7,
        "d_lambda_init": 0.1,
        "d_lambda_min": 1.0e-5,
        "d_lambda_diff_scaled_min": 0.0,
        "omega_max_stop": 4_939_008.747926297,
        "tol": 5.0e-2,
    }

    assert collect_results._matches_current_run_settings(meta, params, case_settings)  # noqa: SLF001

    stale_seed = dict(params, lambda_init=0.005)
    assert not collect_results._matches_current_run_settings(meta, stale_seed, case_settings)  # noqa: SLF001

    stale_horizon = dict(params, omega_max_stop=62_369_288.95910192)
    assert not collect_results._matches_current_run_settings(meta, stale_horizon, case_settings)  # noqa: SLF001


def test_seepage_study_levels_match_waterlevels_mesh_ladder() -> None:
    study = tomllib.loads((ROOT / "docs" / "petsc_matlab_performance_comparison" / "study.toml").read_text(encoding="utf-8"))
    seepage_case = next(case for case in study["cases"] if case["id"] == "seepage_3d")
    levels = {level["id"]: level for level in seepage_case["levels"]}

    assert levels["concave_L2"]["mesh_variant"] == "concave_family_b.msh"
    assert levels["concave_L2"]["matlab_mesh_key"] == "seepage_concave_l2"
    assert "pmg_coarse_mesh_variant" not in levels["concave_L2"]

    assert levels["concave"]["mesh_variant"] == "concave_family_a.msh"
    assert levels["concave"]["matlab_mesh_key"] == "seepage_concave"
    assert levels["concave"]["pmg_coarse_mesh_variant"] == "concave_family_b.msh"
