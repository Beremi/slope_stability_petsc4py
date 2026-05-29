from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from petsc_ssr.benchmarks.generators import check_case_artifacts, create_case_skeleton
from petsc_ssr.config import load_run_case_config


def test_benchmark_init_creates_schema_valid_asset_backed_case(tmp_path: Path) -> None:
    case_toml = create_case_skeleton(
        "3d-new-slope-ssr-p4",
        asset="3d_hetero_slope",
        cases_root=tmp_path,
        element="P4",
        analysis="ssr",
        generate_notebooks=False,
    )

    raw = tomllib.loads(case_toml.read_text(encoding="utf-8"))
    assert sorted(raw) == ["case", "continuation", "linear", "mesh", "newton", "output", "physics"]
    assert raw["case"]["id"] == "3d-new-slope-ssr-p4"
    assert raw["case"]["tags"] == ["experimental"]
    assert raw["mesh"] == {
        "asset": "3d_hetero_slope",
        "variant": "adaptive_family_a_l1",
        "element": "P4",
    }
    assert raw["linear"] == {"profile": "pmg-deflated-baseline"}
    assert raw["newton"] == {"profile": "indirect-regularized-dlambda-stop"}
    assert not ({"ranks", "nodes", "wall_time", "output_dir"} & set(raw["case"]))

    cfg = load_run_case_config(case_toml).validate()
    assert cfg.problem.asset == "3d_hetero_slope"
    assert cfg.problem.elem_type == "P4"
    assert cfg.linear_solver.profile == "pmg-deflated-baseline"
    assert (case_toml.parent / "README.md").exists()
    assert (case_toml.parent / "run.sh").exists()
    assert (case_toml.parent / "notebook.toml").read_text(encoding="utf-8") == '[notebook]\nfamily = "3d_continuation"\n'


def test_benchmark_init_cli_can_create_case_without_notebook_extra(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from petsc_ssr.cli import main as cli_main

    status = cli_main.main(
        [
            "benchmark",
            "init",
            "2d-new-case-ll-p2",
            "--asset",
            "2d_homo_slope",
            "--analysis",
            "ll",
            "--element",
            "P2",
            "--cases-root",
            str(tmp_path),
            "--no-notebooks",
        ]
    )

    assert status == 0
    created = Path(capsys.readouterr().out.strip())
    assert created == tmp_path / "2d-new-case-ll-p2" / "case.toml"
    raw = tomllib.loads(created.read_text(encoding="utf-8"))
    assert raw["physics"]["mechanics"]["model"] == "mohr_coulomb_limit_load"
    assert raw["continuation"]["profile"] == "direct-limit-load"
    assert raw["newton"]["profile"] == "limit-load-regularized"
    assert raw["output"]["preset"] == "standard-continuation"
    assert load_run_case_config(created).validate().problem.analysis == "ll"


def test_benchmark_init_check_validates_generated_scaffold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from petsc_ssr.cli import main as cli_main

    case_toml = create_case_skeleton(
        "3d-check-slope-ssr-p4",
        asset="3d_hetero_slope",
        cases_root=tmp_path,
        element="P4",
        analysis="ssr",
        generate_notebooks=False,
    )

    assert check_case_artifacts(case_toml, check_notebooks=False) == []
    status = cli_main.main(
        [
            "benchmark",
            "init",
            "--check",
            "3d-check-slope-ssr-p4",
            "--cases-root",
            str(tmp_path),
            "--no-notebooks",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["ok"] is True
    assert payload["issues"] == []

    (case_toml.parent / "README.md").write_text("stale\n", encoding="utf-8")
    issues = check_case_artifacts(case_toml, check_notebooks=False)
    assert any("stale generated README" in issue for issue in issues)


def test_generated_notebooks_do_not_expose_inline_material_tables(tmp_path: Path) -> None:
    case_toml = create_case_skeleton(
        "3d-notebook-slope-ssr-p4",
        asset="3d_hetero_slope",
        cases_root=tmp_path,
        element="P4",
        analysis="ssr",
        generate_notebooks=True,
    )

    for notebook_path in (case_toml.parent / "simulation.ipynb", case_toml.parent / "visualisation.ipynb"):
        text = notebook_path.read_text(encoding="utf-8")
        assert "materials = nb.load_case_materials" not in text
        assert "Modify values directly in `sections` or `materials`" not in text
        assert "[[materials]]" not in text


def test_benchmark_init_rejects_non_orthogonal_slug_and_missing_asset_support(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lower-kebab"):
        create_case_skeleton("Bad_Slug", asset="3d_hetero_slope", cases_root=tmp_path, generate_notebooks=False)

    with pytest.raises(ValueError, match="does not declare seepage"):
        create_case_skeleton(
            "3d-no-seepage",
            asset="3d_hetero_slope",
            cases_root=tmp_path,
            analysis="seepage",
            generate_notebooks=False,
        )
