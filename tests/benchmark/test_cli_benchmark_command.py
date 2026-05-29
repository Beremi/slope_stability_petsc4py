from __future__ import annotations

import argparse
from pathlib import Path

from petsc_ssr.benchmarks.generators import create_case_skeleton
from petsc_ssr.cli.commands.benchmark import benchmark_check_payload, benchmark_init_result, benchmark_list_payload


ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_command_check_payload_accepts_slug_and_path(tmp_path: Path) -> None:
    case_toml = create_case_skeleton(
        "3d-command-slope-ssr-p4",
        asset="3d_hetero_slope",
        cases_root=tmp_path,
        element="P4",
        analysis="ssr",
        generate_notebooks=False,
    )

    slug_payload = benchmark_check_payload("3d-command-slope-ssr-p4", cases_root=tmp_path, check_notebooks=False)
    path_payload = benchmark_check_payload(str(case_toml), cases_root=tmp_path, check_notebooks=False)

    assert slug_payload["ok"] is True
    assert slug_payload["issues"] == []
    assert path_payload["ok"] is True
    assert path_payload["issues"] == []


def test_benchmark_check_payload_validates_suite_and_target_registries(tmp_path: Path) -> None:
    cases_root = tmp_path / "cases"
    suites_root = tmp_path / "suites"
    targets_root = tmp_path / "targets"
    cases_root.mkdir()
    suites_root.mkdir()
    targets_root.mkdir()
    (targets_root / "demo.json").write_text(
        '{"case": "demo", "metrics": {"wall_time": {"maximum": 1.0}}}\n',
        encoding="utf-8",
    )

    payload = benchmark_check_payload(
        None,
        cases_root=cases_root,
        suites_root=suites_root,
        targets_root=targets_root,
        check_notebooks=False,
    )

    assert payload["ok"] is False
    assert payload["suites_root"] == str(suites_root)
    assert payload["targets_root"] == str(targets_root)
    assert any("invalid benchmark target registry" in issue and "maximum" in issue for issue in payload["issues"])


def test_benchmark_check_payload_validates_profile_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from petsc_ssr.config import profiles as profile_module

    cases_root = tmp_path / "cases"
    suites_root = tmp_path / "suites"
    targets_root = tmp_path / "targets"
    solver_root = tmp_path / "solver_profiles"
    for path in (cases_root, suites_root, targets_root, solver_root):
        path.mkdir()
    (solver_root / "bad.toml").write_text(
        """
description = "bad"

[linear]
algorithm = "python_array_solver"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(profile_module, "SOLVER_PROFILE_DIR", solver_root)

    payload = benchmark_check_payload(
        None,
        cases_root=cases_root,
        suites_root=suites_root,
        targets_root=targets_root,
        check_notebooks=False,
    )

    assert payload["ok"] is False
    assert any("invalid profile registry" in issue and "python_array_solver" in issue for issue in payload["issues"])


def test_benchmark_init_result_reports_created_case_path(tmp_path: Path) -> None:
    args = argparse.Namespace(
        check=False,
        case="2d-command-case-ll-p2",
        asset="2d_homo_slope",
        cases_root=tmp_path,
        variant=None,
        element="P2",
        analysis="ll",
        title=None,
        linear_profile="pmg-deflated-baseline",
        overwrite=False,
        no_notebooks=True,
    )

    result = benchmark_init_result(args)

    assert result.status == 0
    assert result.payload is None
    assert result.path == tmp_path / "2d-command-case-ll-p2" / "case.toml"
    assert result.path.exists()
    raw = result.path.read_text(encoding="utf-8")
    assert 'profile = "direct-limit-load"' in raw
    assert 'profile = "limit-load-regularized"' in raw


def test_benchmark_list_payload_filters_registry_sections() -> None:
    payload = benchmark_list_payload(
        kind="suites",
        cases_root=ROOT / "benchmarks" / "cases",
        suites_root=ROOT / "benchmarks" / "suites",
        targets_root=ROOT / "benchmarks" / "targets",
    )

    assert sorted(payload) == ["suites"]
    assert any(entry["id"] == "local-32-smoke" for entry in payload["suites"])
