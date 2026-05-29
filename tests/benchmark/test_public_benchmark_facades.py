from __future__ import annotations

import json
from pathlib import Path

import petsc_ssr.benchmarks.suites as suites_mod
from petsc_ssr.benchmarks.compare import compare_targets
from petsc_ssr.benchmarks.generate import check_case_artifacts, create_case_skeleton
from petsc_ssr.benchmarks.generators import check_case_artifacts as legacy_check_case_artifacts
from petsc_ssr.benchmarks.generators import create_case_skeleton as legacy_create_case_skeleton
from petsc_ssr.benchmarks.report import write_report


def test_public_benchmark_facades_reexport_or_delegate_existing_implementations(tmp_path: Path) -> None:
    assert create_case_skeleton is legacy_create_case_skeleton
    assert check_case_artifacts is legacy_check_case_artifacts

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "suite": {"id": "facade-suite", "title": "Facade suite", "ranks": [1]},
                "runs": [
                    {
                        "case": "demo-case",
                        "profile": "pmg-deflated-baseline",
                        "ranks": 1,
                        "repeat": 1,
                        "output_dir": str(tmp_path / "runs" / "demo-case"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    target_root = tmp_path / "targets"
    target_root.mkdir()

    direct = compare_targets(tmp_path, target_root, output=tmp_path / "direct.json")
    delegated = suites_mod.compare_targets(tmp_path, target_root, output=tmp_path / "delegated.json")

    assert json.loads(direct.read_text(encoding="utf-8")) == json.loads(delegated.read_text(encoding="utf-8"))

    direct_report = write_report(tmp_path, output=tmp_path / "direct-report.md")
    delegated_report = suites_mod.write_report(tmp_path, output=tmp_path / "delegated-report.md")

    assert direct_report.read_text(encoding="utf-8") == delegated_report.read_text(encoding="utf-8")
