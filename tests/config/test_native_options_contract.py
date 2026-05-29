from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from petsc_ssr.case_config import (
    planned_mechanics_neumann_label_table,
    planned_seepage_label_table,
    translate_case_toml,
)
from petsc_ssr.hydro_cases import hydro_option_tokens, translate_hydro_case_toml
from petsc_ssr.options import PmgOptions, SsrOptions
from petsc_ssr.problem import ProblemSpec
from petsc_ssr.runtime.options import read_options_file


ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = ROOT / "benchmarks" / "cases"
NATIVE_ROOT = ROOT / "src" / "petsc_ssr" / "native"


NATIVE_OPTION_RE = re.compile(
    r"PetscOptions(?:Get[A-Za-z0-9_]+|HasName|String|Real|Int|Bool)\s*\([^;]*?\"(-[A-Za-z0-9_]+)\"",
    re.DOTALL,
)
MATERIAL_REGION_RE = re.compile(r"^-material_region_[0-9]+$")
PMG_SHELL_PREFIXED_PETSC_RE = re.compile(
    r"^-pmg_shell_(?:fine|p2|p1)_(?:ksp|pc|redundant_ksp|redundant_pc)_"
)

PETSC_OWNED_EXACT = {
    "-dm_plex_partition_balance",
    "-ksp_converged_reason",
    "-ksp_max_it",
    "-ksp_norm_type",
    "-ksp_type",
    "-log_view",
    "-options_left",
    "-options_view",
}


def test_representative_case_options_are_consumed_by_native_or_petsc(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PETSC_SSR_WORLD_SIZE", "32")
    native_options = _native_option_names()

    cases = (
        CASE_ROOT / "3d-heterogeneous-ssr-p4" / "case.toml",
        CASE_ROOT / "3d-homogeneous-seepage-ssr-concave" / "case.toml",
    )
    unknown: dict[str, list[str]] = {}
    for case_toml in cases:
        tokens = _normal_run_tokens(case_toml, tmp_path / case_toml.parent.name)
        assert "-continuation_algorithm" in tokens
        assert "-newton_algorithm" in tokens
        assert "-linear_algorithm" in tokens
        missing = _unknown_project_options(tokens, native_options)
        if missing:
            unknown[case_toml.parent.name] = missing

    assert unknown == {}


def test_artifact_bridge_options_are_consumed_by_native(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PETSC_SSR_WORLD_SIZE", "32")
    native_options = _native_option_names()
    translation = translate_case_toml(CASE_ROOT / "3d-heterogeneous-ssr-p4" / "case.toml")
    assert translation.supported, translation.reason
    assert translation.problem is not None

    data_dir = tmp_path / "bridge" / "data"
    metadata = {
        **translation.problem.metadata,
        "native_problem_manifest": str(data_dir / "native_problem_manifest.json"),
        "mechanics_bc_labels_csv": str(data_dir / "mechanics_bc_labels.csv"),
        "mechanics_bc_nodes_csv": str(data_dir / "mechanics_bc_nodes.csv"),
        "debug_coordinate_bc_table": True,
        "mechanics_neumann_labels_csv": str(data_dir / "mechanics_neumann_labels.csv"),
        "seepage_boundary_labels_csv": str(data_dir / "seepage_boundary_labels.csv"),
        "seepage_pressure_source": "hydro_prepass_coordinate_bridge",
        "seepage_pressure_csv": str(data_dir / "seepage_pressure.csv"),
        "seepage_grho": 9.81,
    }
    problem = replace(translation.problem, metadata=metadata)

    assert _unknown_project_options(problem.option_tokens(), native_options) == []


def test_default_problem_and_all_pmg_profile_tokens_have_consumers() -> None:
    native_options = _native_option_names()
    pmg = PmgOptions(
        p2_active_ranks=8,
        p1_active_ranks=4,
        subcomm_type="interlaced",
        fine_ksp_max_it=5,
        p2_ksp_max_it=10,
        p1_pc_type="redundant",
        p1_redundant_number=1,
        p1_redundant_ksp_type="fgmres",
        p1_redundant_ksp_rtol=1.0e-3,
        p1_redundant_ksp_max_it=5,
        p1_redundant_pc_type="gamg",
    )
    tokens = [
        *ProblemSpec.tiny_box().option_tokens(),
        *replace(SsrOptions.current_baseline(), pmg=pmg).option_tokens(),
    ]

    assert "-continuation_algorithm" in tokens
    assert "-newton_algorithm" in tokens
    assert "-linear_algorithm" in tokens
    assert "-pmg_apply_backend" in tokens
    assert "-pmg_coarse_telescope_ksp_max_it" in tokens
    assert "-pmg_p2_telescope_active_ranks" in tokens
    assert "-pmg_smoother_max_it" in tokens
    assert _unknown_project_options(tokens, native_options) == []


def test_raw_pmg_options_defaults_do_not_pin_active_rank_caps() -> None:
    pmg = PmgOptions()

    assert pmg.p2_active_ranks == 0
    assert pmg.p1_active_ranks == 0


def test_pmg_baseline_options_file_keeps_solver_policy_in_profile() -> None:
    tokens = read_options_file(PmgOptions().options_file)

    assert tokens == ["-dm_plex_partition_balance", "true", "-ksp_converged_reason"]


def test_hydro_case_options_are_consumed_by_native_or_petsc(tmp_path: Path) -> None:
    native_options = _native_option_names()
    cases = (
        CASE_ROOT / "2d-sloan2013-seepage" / "case.toml",
        CASE_ROOT / "3d-heterogeneous-seepage" / "case.toml",
    )

    unknown: dict[str, list[str]] = {}
    for case_toml in cases:
        translation = translate_hydro_case_toml(case_toml)
        assert translation.supported, translation.reason
        tokens = hydro_option_tokens(translation, tmp_path / case_toml.parent.name)
        missing = _unknown_project_options(tokens, native_options)
        if missing:
            unknown[case_toml.parent.name] = missing

    assert unknown == {}


def _normal_run_tokens(case_toml: Path, output_dir: Path) -> list[str]:
    translation = translate_case_toml(case_toml)
    assert translation.supported, translation.reason
    assert translation.problem is not None
    assert translation.options is not None

    data_dir = output_dir / "data"
    metadata = {
        **translation.problem.metadata,
        "native_problem_manifest": str(data_dir / "native_problem_manifest.json"),
        "mechanics_bc_labels_csv": str(data_dir / "mechanics_bc_labels.csv"),
    }
    neumann_labels_csv = planned_mechanics_neumann_label_table(translation, output_dir)
    seepage_labels_csv = planned_seepage_label_table(translation, output_dir)
    if neumann_labels_csv is not None:
        metadata["mechanics_neumann_labels_csv"] = str(neumann_labels_csv)
    if seepage_labels_csv is not None:
        metadata["seepage_boundary_labels_csv"] = str(seepage_labels_csv)
    if bool(metadata.get("seepage_coupled", False)):
        metadata["seepage_pressure_source"] = "hydro_prepass_coordinate_bridge"
        metadata["seepage_pressure_csv"] = str(data_dir / "seepage_pressure.csv")
        metadata["seepage_grho"] = 9.81
    problem = replace(translation.problem, metadata=metadata)

    return [
        *read_options_file(Path(translation.options.pmg.options_file)),
        *problem.option_tokens(),
        *translation.options.option_tokens(),
        "-curve_csv",
        str(data_dir / "continuation_curve.csv"),
        "-summary_json",
        str(data_dir / "summary.json"),
        "-solution_binary",
        str(data_dir / "final_displacement.petscbin"),
        "-solution_points_csv",
        str(data_dir / "final_displacement_points.csv"),
        "-solution_vtk",
        str(output_dir / "exports" / "final_solution.vtu"),
        "-log_view",
        f":{output_dir / 'logs' / 'petsc_log.txt'}",
        "-options_view",
        f":{output_dir / 'logs' / 'options_view.txt'}",
        "-options_left",
    ]


def _native_option_names() -> set[str]:
    names: set[str] = set()
    for path in NATIVE_ROOT.rglob("*"):
        if path.suffix not in {".c", ".h", ".inc"}:
            continue
        names.update(NATIVE_OPTION_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
    return names


def _unknown_project_options(tokens: list[str], native_options: set[str]) -> list[str]:
    keys = sorted({token for token in tokens if token.startswith("-")})
    return [key for key in keys if not _is_known_option(key, native_options)]


def _is_known_option(option: str, native_options: set[str]) -> bool:
    if option in native_options:
        return True
    if option in PETSC_OWNED_EXACT:
        return True
    if MATERIAL_REGION_RE.match(option):
        return True
    if PMG_SHELL_PREFIXED_PETSC_RE.match(option):
        return True
    return False
