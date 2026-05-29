from __future__ import annotations

import json
from pathlib import Path

from petsc_ssr.benchmarks.registry import discover_benchmark_registry, discover_cases, discover_suites, discover_targets


ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_registry_discovers_cases_suites_and_targets() -> None:
    registry = discover_benchmark_registry()

    cases = {entry["slug"]: entry for entry in registry["cases"]}
    suites = {entry["id"]: entry for entry in registry["suites"]}
    targets = {(entry["target_set"], entry["case"]): entry for entry in registry["targets"]}

    assert "3d-heterogeneous-ssr-p4" in cases
    assert cases["3d-heterogeneous-ssr-p4"]["asset"] == "3d_hetero_slope"
    assert cases["3d-heterogeneous-ssr-p4"]["profiles"]["linear"] == "pmg-deflated-baseline"
    assert "3d-heterogeneous-seepage" in cases
    assert cases["3d-heterogeneous-seepage"]["analysis"] == "seepage"

    assert "local-32-smoke" in suites
    assert suites["local-32-smoke"]["ranks"] == [1, 2, 4, 8, 16, 32]
    assert suites["local-32-smoke"]["resources"]["local"]["cores"] == 32
    assert suites["local-32-smoke"]["environment"] == {"OMP_NUM_THREADS": "1"}
    assert "ssr-scaling" in suites
    assert suites["ssr-scaling"]["profiles"] == ["pmg-deflated-baseline"]
    assert suites["ssr-scaling"]["ranks"] == [32, 64, 128, 256]
    assert suites["ssr-scaling"]["resources"]["karolina"]["ranks_per_node"] == 128
    assert "hpc-strong-scaling" in suites
    assert suites["hpc-strong-scaling"]["resources"]["karolina"]["max_ranks"] == 256
    assert suites["hpc-strong-scaling"]["ranks"] == [32, 64, 128, 256]
    assert "validation" in suites
    assert suites["validation"]["ranks"] == [1]
    assert "2d-sloan2013-seepage" in suites["validation"]["cases"]

    local_target = targets[("local-32", "3d-heterogeneous-ssr-p4")]
    numerical_target = targets[("numerical", "3d-heterogeneous-ssr-p4")]
    assert local_target["suite"] == "local-32-strong-scaling"
    assert local_target["rank_metric_groups"] >= 1
    assert "wall_time" in local_target["metrics"]
    assert numerical_target["suite"] == "numerical"
    assert "lambda_last" in numerical_target["metrics"]
    assert "wall_time" not in numerical_target["metrics"]
    assert (".", "l1-unrefined-c-target") not in targets


def test_benchmark_registry_helpers_accept_custom_roots(tmp_path: Path) -> None:
    case_dir = tmp_path / "cases" / "tiny"
    case_dir.mkdir(parents=True)
    (case_dir / "case.toml").write_text(
        """
[case]
id = "tiny"
title = "Tiny"
tags = ["experimental"]

[mesh]
asset = "tiny_asset"
variant = "default"
element = "P1"

[physics.seepage]
model = "darcy"

[linear]
profile = "pmg-deflated-baseline"
""",
        encoding="utf-8",
    )
    suites_root = tmp_path / "suites"
    suites_root.mkdir()
    (suites_root / "tiny-suite.toml").write_text(
        """
[suite]
id = "tiny-suite"
cases = ["tiny"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]
""",
        encoding="utf-8",
    )
    targets_root = tmp_path / "targets"
    targets_root.mkdir()
    (targets_root / "tiny.json").write_text(json.dumps({"case": "tiny", "metrics": {"wall_time": {"max": 1.0}}}), encoding="utf-8")

    assert discover_cases(case_dir.parent)[0]["analysis"] == "seepage"
    assert discover_suites(suites_root)[0]["id"] == "tiny-suite"
    assert discover_targets(targets_root)[0]["metrics"] == ["wall_time"]


def test_benchmark_list_cli_outputs_registry(capsys) -> None:
    from petsc_ssr.cli import main as cli_main

    status = cli_main.main(["benchmark", "list", "--kind", "suites"])

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert sorted(payload) == ["suites"]
    assert any(entry["id"] == "local-32-smoke" for entry in payload["suites"])
