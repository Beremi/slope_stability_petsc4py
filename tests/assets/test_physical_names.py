from __future__ import annotations

import json
from pathlib import Path

from petsc_ssr.assets.factories import build_problem_asset_3d
from petsc_ssr.assets import available_problem_assets
from petsc_ssr.assets.support.physical_names import parse_gmsh_physical_names
from petsc_ssr.cli.commands.asset import validate_asset_payload
from petsc_ssr.cli.main import _asset_validate


def _write_mesh(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(["$MeshFormat", "2.2 0 8", "$EndMeshFormat", *lines]) + "\n", encoding="utf-8")
    return path


def test_physical_name_parser_prefers_explicit_support_prefixes(tmp_path: Path) -> None:
    mesh = _write_mesh(
        tmp_path / "prefixed.msh",
        [
            "$PhysicalNames",
            "3",
            '3 1 "region:body"',
            '2 2 "boundary:base"',
            '0 3 "nodeset:toe"',
            "$EndPhysicalNames",
        ],
    )

    assert parse_gmsh_physical_names(mesh) == {
        "regions": {"body": 1},
        "boundaries": {"base": 2},
        "nodesets": {"toe": 3},
    }


def test_physical_name_parser_classifies_unprefixed_2d_and_3d_supports(tmp_path: Path) -> None:
    mesh2d = _write_mesh(
        tmp_path / "legacy_2d.msh",
        [
            "$PhysicalNames",
            "3",
            '2 11 "soil"',
            '1 12 "base"',
            '0 13 "toe"',
            "$EndPhysicalNames",
        ],
    )
    mesh3d = _write_mesh(
        tmp_path / "legacy_3d.msh",
        [
            "$PhysicalNames",
            "3",
            '3 21 "soil"',
            '2 22 "base"',
            '0 23 "toe"',
            "$EndPhysicalNames",
        ],
    )

    assert parse_gmsh_physical_names(mesh2d) == {
        "regions": {"soil": 11},
        "boundaries": {"base": 12},
        "nodesets": {"toe": 13},
    }
    assert parse_gmsh_physical_names(mesh3d) == {
        "regions": {"soil": 21},
        "boundaries": {"base": 22},
        "nodesets": {"toe": 23},
    }


def test_asset_validate_accepts_declared_supports_for_registered_assets() -> None:
    for asset_id in available_problem_assets():
        assert _asset_validate(asset_id) == 0


def test_asset_validate_reports_native_manifest_contracts_for_registered_assets() -> None:
    payload = validate_asset_payload("3d_hetero_slope")

    assert payload["errors"] == []
    contract = payload["native_manifest_contracts"]["adaptive_family_a_l1.msh"][0]
    assert contract["profile"] == "default"
    assert contract["support_counts"] == {
        "regions": 4,
        "boundaries": 7,
        "nodesets": 3,
        "boundary_geometry": 0,
    }
    assert contract["rule_counts"] == {
        "mechanics_dirichlet": 3,
        "mechanics_neumann": 0,
        "seepage_head": 0,
        "seepage_flux": 0,
    }
    assert contract["label_tables"]["mechanics_dirichlet"]["rows"] == 3
    assert contract["label_tables"]["mechanics_dirichlet"]["row_fingerprint"].startswith("fnv1a64:")
    assert contract["label_tables"]["mechanics_dirichlet"]["native_statuses"] == ["label_table_native_preferred"]
    assert contract["label_tables"]["mechanics_neumann"]["row_fingerprint"] is None
    assert contract["native_inputs"] == {}


def test_asset_validate_all_uses_registered_assets(capsys) -> None:
    from petsc_ssr.cli import main as cli_main

    assert cli_main.main(["asset", "validate", "--all"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["count"] == len(available_problem_assets())
    assert payload["errors"] == 0
    assert {row["asset"] for row in payload["assets"]} == set(available_problem_assets())
    assert all("native_manifest_contracts" in row for row in payload["assets"])


def test_asset_validate_rejects_bad_boundary_geometry_links(monkeypatch, tmp_path: Path) -> None:
    mesh = _write_mesh(
        tmp_path / "mesh.msh",
        [
            "$PhysicalNames",
            "3",
            '3 1 "region:body"',
            '2 2 "boundary:base"',
            '2 3 "boundary:slope_face"',
            "$EndPhysicalNames",
        ],
    )
    asset = build_problem_asset_3d(
        asset_id="synthetic_bad_geometry_link",
        asset_dir=tmp_path,
        default_variant=mesh.name,
        mesh_variants={mesh.name: {"source": {"path": mesh.name}}},
        materials={
            "soil": {
                "c0": 1.0,
                "phi": 30.0,
                "psi": 0.0,
                "young": 1000.0,
                "poisson": 0.3,
                "gamma_sat": 20.0,
                "gamma_unsat": 20.0,
            }
        },
        region_assignment={"body": "soil"},
        boundary_geometry={"base_curve": {"support_boundary": "base", "geometry_order": 2}},
        mechanics={
            "neumann": [
                {
                    "target": "slope_face",
                    "kind": "traction",
                    "geometry": "base_curve",
                    "value_model": {"type": "constant", "value": [0.0, -10.0, 0.0]},
                }
            ],
        },
    )

    import petsc_ssr.assets as assets

    monkeypatch.setattr(assets, "load_problem_asset", lambda _asset_id: asset)

    assert _asset_validate("synthetic_bad_geometry_link") == 2


def test_asset_validate_requires_face_supports_for_neumann_and_flux(monkeypatch, tmp_path: Path, capsys) -> None:
    mesh = _write_mesh(
        tmp_path / "mesh.msh",
        [
            "$PhysicalNames",
            "3",
            '3 1 "region:body"',
            '2 2 "boundary:slope_face"',
            '0 3 "nodeset:crest"',
            "$EndPhysicalNames",
        ],
    )
    asset = build_problem_asset_3d(
        asset_id="synthetic_bad_face_support",
        asset_dir=tmp_path,
        default_variant=mesh.name,
        mesh_variants={mesh.name: {"source": {"path": mesh.name}}},
        materials={
            "soil": {
                "c0": 1.0,
                "phi": 30.0,
                "psi": 0.0,
                "young": 1000.0,
                "poisson": 0.3,
                "gamma_sat": 20.0,
                "gamma_unsat": 20.0,
            }
        },
        region_assignment={"body": "soil"},
        mechanics={
            "neumann": [
                {
                    "target": "crest",
                    "kind": "traction",
                    "value_model": {"type": "constant", "value": [0.0, -10.0, 0.0]},
                }
            ],
        },
        seepage={
            "water_unit_weight": 9.81,
            "conductivity_mode": "isotropic",
            "conductivity": 1.0e-6,
            "head_bcs": [{"target": "crest", "kind": "fixed", "value_model": {"head": 0.0}}],
            "flux_bcs": [{"target": "crest", "kind": "zero", "value_model": {"type": "constant", "value": 0.0}}],
        },
    )

    import petsc_ssr.assets as assets

    monkeypatch.setattr(assets, "load_problem_asset", lambda _asset_id: asset)

    assert _asset_validate("synthetic_bad_face_support") == 2
    payload = capsys.readouterr().out
    assert "Neumann target" in payload
    assert "seepage flux target" in payload
    assert "not a boundary support" in payload


def test_asset_validate_checks_every_mesh_variant(monkeypatch, tmp_path: Path, capsys) -> None:
    good = _write_mesh(
        tmp_path / "good.msh",
        [
            "$PhysicalNames",
            "2",
            '3 1 "region:body"',
            '2 2 "boundary:base"',
            "$EndPhysicalNames",
        ],
    )
    bad = _write_mesh(
        tmp_path / "bad.msh",
        [
            "$PhysicalNames",
            "1",
            '3 1 "region:body"',
            "$EndPhysicalNames",
        ],
    )
    asset = build_problem_asset_3d(
        asset_id="synthetic_bad_second_variant",
        asset_dir=tmp_path,
        default_variant=good.name,
        mesh_variants={
            good.name: {"source": {"path": good.name}},
            bad.name: {"source": {"path": bad.name}},
        },
        materials={
            "soil": {
                "c0": 1.0,
                "phi": 30.0,
                "psi": 0.0,
                "young": 1000.0,
                "poisson": 0.3,
                "gamma_sat": 20.0,
                "gamma_unsat": 20.0,
            }
        },
        region_assignment={"body": "soil"},
        mechanics={"dirichlet": [{"target": "base", "components": ["y"]}]},
    )

    import petsc_ssr.assets as assets

    monkeypatch.setattr(assets, "load_problem_asset", lambda _asset_id: asset)

    assert _asset_validate("synthetic_bad_second_variant") == 2
    payload = capsys.readouterr().out
    assert "bad.msh" in payload
    assert "Dirichlet target" in payload
    assert "variant_supports" in payload
