from __future__ import annotations

import json
from pathlib import Path

import pytest

from petsc_ssr.benchmarks.compare import compare_targets
from petsc_ssr.benchmarks.targets import load_target, validate_target_payload


def test_target_schema_accepts_case_bound_metrics_and_groups(tmp_path: Path) -> None:
    target = tmp_path / "3d-heterogeneous-ssr-p4.json"
    target.write_text(
        json.dumps(
            {
                "case": "3d-heterogeneous-ssr-p4",
                "profile": "pmg-deflated-baseline",
                "suite": "local-32-strong-scaling",
                "status": "measured",
                "notes": "Fixture target.",
                "metrics": {"lambda_last": {"expected": 1.0, "abs_tol": 1.0e-6}},
                "rank_metrics": {"32": {"metrics": {"wall_time": {"max": 2.0}}, "source": "fixture"}},
                "groups": [{"ranks": 64, "metrics": {"total_linear_its": {"max": 1000}}}],
            }
        ),
        encoding="utf-8",
    )

    payload = load_target(target)

    assert payload["case"] == "3d-heterogeneous-ssr-p4"


def test_target_schema_rejects_commands_and_invalid_metric_specs() -> None:
    with pytest.raises(ValueError, match=r"command"):
        validate_target_payload({"case": "demo", "command": ["mpiexec"], "metrics": {}})

    with pytest.raises(ValueError, match=r"wall_time.*max, expected, or value"):
        validate_target_payload({"case": "demo", "metrics": {"wall_time": {"abs_tol": 1.0}}})

    with pytest.raises(ValueError, match=r"rank_metrics.*positive integer rank"):
        validate_target_payload({"case": "demo", "metrics": {}, "rank_metrics": {"zero": {"metrics": {}}}})

    with pytest.raises(ValueError, match=r"case.*string"):
        validate_target_payload({"case": 123, "metrics": {}})

    with pytest.raises(ValueError, match=r"wall_time.max.*numeric"):
        validate_target_payload({"case": "demo", "metrics": {"wall_time": {"max": True}}})


def test_target_compare_validates_direct_target_files(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "suite": {"id": "target-validation", "title": "Target validation"},
                "runs": [
                    {
                        "case": "demo",
                        "profile": "pmg-deflated-baseline",
                        "ranks": 1,
                        "repeat": 1,
                        "output_dir": str(tmp_path / "runs" / "demo"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    target_root = tmp_path / "targets"
    target_root.mkdir()
    (target_root / "demo.json").write_text(
        json.dumps({"case": "demo", "metrics": {"wall_time": {"maximum": 1.0}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"maximum"):
        compare_targets(tmp_path, target_root)
