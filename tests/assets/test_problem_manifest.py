from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from petsc_ssr.assets.factories import build_problem_asset_3d
from petsc_ssr.config import load_run_case_config
from petsc_ssr.problem import ProblemSpec
from petsc_ssr.problem_asset_runtime import (
    MECHANICS_LABEL_CONSTRAINT_COLUMNS,
    MECHANICS_NEUMANN_LABEL_COLUMNS,
    SEEPAGE_LABEL_BC_COLUMNS,
    ResolvedAsset,
    build_mechanics_label_constraint_rows,
    build_mechanics_neumann_label_rows,
    build_native_problem_manifest,
    build_seepage_label_bc_rows,
    resolve_problem_asset_from_config,
    validate_native_problem_artifact_contract,
)


ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = ROOT / "benchmarks" / "cases"


def test_native_problem_manifest_records_case_asset_supports(monkeypatch) -> None:
    monkeypatch.setenv("PETSC_SSR_WORLD_SIZE", "32")
    cfg = load_run_case_config(CASE_ROOT / "3d-heterogeneous-ssr-p4" / "case.toml").validate()
    resolved = resolve_problem_asset_from_config(cfg)

    manifest = build_native_problem_manifest(
        resolved,
        case_id=cfg.problem.name,
        analysis=cfg.problem.analysis,
        elem_type=cfg.problem.elem_type,
        solver_profile=cfg.linear_solver.profile,
        world_size=cfg.linear_solver.resolved_world_size,
        compatibility={"seepage_coupled": False},
    )

    assert manifest["kind"] == "petsc_ssr_native_problem_manifest"
    assert manifest["schema_version"] == 1
    assert manifest["case"]["analysis"] == "ssr"
    assert manifest["case"]["element"] == "P4"
    assert manifest["case"]["solver_profile"] == "pmg-deflated-baseline"
    assert manifest["case"]["resolved_world_size"] == 32
    assert manifest["asset"]["id"] == "3d_hetero_slope"
    assert manifest["dmplex"]["region_label"] == "Cell Sets"
    assert manifest["dmplex"]["boundary_label"] == "Face Sets"
    assert manifest["dmplex"]["support_counts"] == {
        "regions": 4,
        "boundaries": 7,
        "nodesets": 3,
        "boundary_geometry": 0,
    }
    assert manifest["dmplex"]["supports"]["regions"]["slope_mass"]["tag"] > 0
    assert manifest["dmplex"]["supports"]["boundaries"]["base"]["tag"] > 0
    assert manifest["dmplex"]["supports"]["nodesets"]["x_lock"]["tag"] > 0
    assert manifest["rule_counts"] == {
        "mechanics_dirichlet": 3,
        "mechanics_neumann": 0,
        "seepage_head": 0,
        "seepage_flux": 0,
    }

    dirichlet = manifest["mechanics"]["dirichlet"]
    assert {rule["target"] for rule in dirichlet} == {"base", "x_lock", "z_lock"}
    assert {rule["support"]["kind"] for rule in dirichlet} == {"boundary", "nodeset"}
    assert all(rule["native_status"] == "label_table_native_preferred" for rule in dirichlet)
    assert manifest["compatibility"]["label_constraint_table_available"] is True
    assert manifest["compatibility"]["label_constraint_table_required"] is True
    assert manifest["compatibility"]["coordinate_constraint_table_required"] is False
    assert manifest["compatibility"]["coordinate_constraint_table_fallback_available"] is False
    assert manifest["compatibility"]["coordinate_seepage_pressure_table_required"] is False
    assert "coordinates" not in _all_keys(manifest)
    assert "node_coordinates" not in _all_keys(manifest)


def test_mechanics_label_constraint_rows_are_coordinate_free(monkeypatch) -> None:
    monkeypatch.setenv("PETSC_SSR_WORLD_SIZE", "32")
    cfg = load_run_case_config(CASE_ROOT / "3d-heterogeneous-ssr-p4" / "case.toml").validate()
    resolved = resolve_problem_asset_from_config(cfg)

    rows = build_mechanics_label_constraint_rows(resolved)

    assert len(rows) == 3
    assert set(rows[0]) == set(MECHANICS_LABEL_CONSTRAINT_COLUMNS)
    assert {row["support_name"] for row in rows} == {"base", "x_lock", "z_lock"}
    assert {row["support_kind"] for row in rows} == {"boundary", "nodeset"}
    assert {row["dm_label"] for row in rows} == {"Face Sets", "Vertex Sets"}
    assert all(int(row["tag"]) > 0 for row in rows)
    assert {row["components"] for row in rows} == {"x", "y", "z"}
    assert {row["native_status"] for row in rows} == {"label_table_native_preferred"}
    assert {"x", "y", "z", "node", "coordinate"}.isdisjoint(rows[0])


def test_case_artifacts_write_label_table_and_manifest_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PETSC_SSR_WORLD_SIZE", "32")
    from petsc_ssr.case_config import (
        translate_case_toml,
        write_mechanics_label_constraint_table,
        write_native_problem_manifest,
    )

    translation = translate_case_toml(CASE_ROOT / "3d-heterogeneous-ssr-p4" / "case.toml")
    assert translation.supported, translation.reason

    label_path = write_mechanics_label_constraint_table(translation, tmp_path)
    manifest_path = write_native_problem_manifest(translation, tmp_path)

    assert label_path.name == "mechanics_bc_labels.csv"
    assert label_path.read_text(encoding="utf-8").splitlines()[0] == ",".join(MECHANICS_LABEL_CONSTRAINT_COLUMNS)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["compatibility"]["mechanics_label_constraint_table"] == str(label_path)
    assert manifest["compatibility"]["label_constraint_table_required"] is True
    assert "mechanics_coordinate_constraint_table" not in manifest["compatibility"]
    assert manifest["compatibility"]["coordinate_constraint_table_fallback_available"] is False
    assert manifest["native_inputs"]["mechanics_label_constraints_csv"] == str(label_path)
    assert manifest["native_inputs"]["mechanics_label_constraints_row_fingerprint"].startswith("fnv1a64:")
    assert "mechanics_coordinate_constraints_csv" not in manifest["native_inputs"]

    coordinate_path = tmp_path / "data" / "mechanics_bc_nodes.csv"
    debug_manifest_path = write_native_problem_manifest(
        translation,
        tmp_path,
        mechanics_coordinate_constraint_table=coordinate_path,
    )
    debug_manifest = json.loads(debug_manifest_path.read_text(encoding="utf-8"))
    assert debug_manifest["compatibility"]["mechanics_coordinate_constraint_table"] == str(coordinate_path)
    assert debug_manifest["compatibility"]["debug_coordinate_bc_table"] is True
    assert debug_manifest["compatibility"]["coordinate_constraint_table_fallback_available"] is True
    assert debug_manifest["native_inputs"]["debug_coordinate_bc_table"] is True
    assert debug_manifest["native_inputs"]["mechanics_coordinate_constraints_csv"] == str(coordinate_path)

    contract = validate_native_problem_artifact_contract(manifest_path)
    assert contract["checks"]["mechanics_dirichlet"]["rows"] == 3


def test_native_problem_artifact_contract_rejects_manifest_fingerprint_drift(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PETSC_SSR_WORLD_SIZE", "32")
    from petsc_ssr.case_config import (
        translate_case_toml,
        write_mechanics_label_constraint_table,
        write_native_problem_manifest,
    )

    translation = translate_case_toml(CASE_ROOT / "3d-heterogeneous-ssr-p4" / "case.toml")
    assert translation.supported, translation.reason

    write_mechanics_label_constraint_table(translation, tmp_path)
    manifest_path = write_native_problem_manifest(translation, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["native_inputs"]["mechanics_label_constraints_row_fingerprint"] = "fnv1a64:0000000000000000"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="declares label-table fingerprint"):
        validate_native_problem_artifact_contract(manifest_path)


def test_native_problem_artifact_contract_rejects_label_table_drift(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PETSC_SSR_WORLD_SIZE", "32")
    from petsc_ssr.case_config import (
        translate_case_toml,
        write_mechanics_label_constraint_table,
        write_native_problem_manifest,
    )

    translation = translate_case_toml(CASE_ROOT / "3d-heterogeneous-ssr-p4" / "case.toml")
    assert translation.supported, translation.reason

    label_path = write_mechanics_label_constraint_table(translation, tmp_path)
    manifest_path = write_native_problem_manifest(translation, tmp_path)
    assert validate_native_problem_artifact_contract(manifest_path)["ok"] is True

    rows = list(csv.DictReader(label_path.open("r", encoding="utf-8", newline="")))
    with label_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MECHANICS_LABEL_CONSTRAINT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[:-1])

    with pytest.raises(ValueError, match="does not match label table"):
        validate_native_problem_artifact_contract(manifest_path)


def test_seepage_label_bc_rows_are_coordinate_free(monkeypatch) -> None:
    monkeypatch.setenv("PETSC_SSR_WORLD_SIZE", "32")
    cfg = load_run_case_config(CASE_ROOT / "3d-heterogeneous-seepage" / "case.toml").validate()
    resolved = resolve_problem_asset_from_config(cfg)

    rows = build_seepage_label_bc_rows(resolved)
    manifest = build_native_problem_manifest(
        resolved,
        case_id=cfg.problem.name,
        analysis=cfg.problem.analysis,
        elem_type=cfg.problem.elem_type,
        solver_profile=cfg.linear_solver.profile,
        world_size=cfg.linear_solver.resolved_world_size,
        compatibility={
            "seepage_coupled": False,
            "seepage_boundary_label_table": ".local/run/data/seepage_boundary_labels.csv",
        },
    )

    assert len(rows) == 3
    assert set(rows[0]) == set(SEEPAGE_LABEL_BC_COLUMNS)
    assert {row["field"] for row in rows} == {"head"}
    assert {row["support_name"] for row in rows} == {"head_dry", "head_porous", "head_free"}
    assert {row["support_kind"] for row in rows} == {"nodeset"}
    assert {row["dm_label"] for row in rows} == {"Vertex Sets"}
    assert all(int(row["tag"]) > 0 for row in rows)
    assert {row["native_status"] for row in rows} == {"label_ready_coordinate_pressure_bridge_active"}
    assert {"x", "y", "z", "node", "coordinate"}.isdisjoint(rows[0])
    assert {bc["native_status"] for bc in manifest["seepage"]["head_bcs"]} == {"label_ready_coordinate_pressure_bridge_active"}
    assert manifest["native_inputs"]["seepage_boundary_labels_csv"] == ".local/run/data/seepage_boundary_labels.csv"
    assert manifest["native_inputs"]["seepage_boundary_labels_row_fingerprint"].startswith("fnv1a64:")
    assert "coordinates" not in _all_keys(manifest)


def test_seepage_pressure_manifest_input_requires_source_contract(monkeypatch) -> None:
    monkeypatch.setenv("PETSC_SSR_WORLD_SIZE", "32")
    cfg = load_run_case_config(CASE_ROOT / "3d-homogeneous-seepage-ssr-concave" / "case.toml").validate()
    resolved = resolve_problem_asset_from_config(cfg)

    with pytest.raises(ValueError, match="seepage_pressure_source"):
        build_native_problem_manifest(
            resolved,
            case_id=cfg.problem.name,
            analysis=cfg.problem.analysis,
            elem_type=cfg.problem.elem_type,
            solver_profile=cfg.linear_solver.profile,
            world_size=cfg.linear_solver.resolved_world_size,
            compatibility={"seepage_pressure_table": ".local/run/data/seepage_pressure.csv"},
        )

    manifest = build_native_problem_manifest(
        resolved,
        case_id=cfg.problem.name,
        analysis=cfg.problem.analysis,
        elem_type=cfg.problem.elem_type,
        solver_profile=cfg.linear_solver.profile,
        world_size=cfg.linear_solver.resolved_world_size,
        compatibility={
            "seepage_pressure_table": ".local/run/data/seepage_pressure.csv",
            "seepage_pressure_source": "hydro_prepass_coordinate_bridge",
        },
    )

    assert manifest["native_inputs"]["seepage_pressure_source"] == "hydro_prepass_coordinate_bridge"
    assert manifest["native_inputs"]["seepage_pressure_csv"] == ".local/run/data/seepage_pressure.csv"


def test_native_problem_manifest_preserves_neumann_and_boundary_geometry(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.msh"
    mesh_path.write_text(
        "\n".join(
            [
                "$MeshFormat",
                "2.2 0 8",
                "$EndMeshFormat",
                "$PhysicalNames",
                "4",
                '3 1 "region:body"',
                '2 2 "boundary:base"',
                '2 3 "boundary:slope_face"',
                '0 4 "nodeset:x_lock"',
                "$EndPhysicalNames",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    asset = build_problem_asset_3d(
        asset_id="synthetic_neumann",
        asset_dir=tmp_path,
        default_variant="mesh.msh",
        mesh_variants={"mesh.msh": {"source": {"path": "mesh.msh"}}},
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
        boundary_geometry={"slope_curve": {"support_boundary": "slope_face", "geometry_order": 2}},
        mechanics={
            "dirichlet": [
                {"target": "base", "components": ["y"]},
                {"target": "x_lock", "components": ["x"]},
            ],
            "neumann": [
                {
                    "target": "slope_face",
                    "kind": "traction",
                    "geometry": "slope_curve",
                    "value_model": {"type": "constant", "value": [0.0, -10.0, 0.0]},
                }
            ],
        },
    )
    variant = asset.resolve_variant(None)
    resolved = ResolvedAsset(
        definition=asset,
        variant_name=variant.name,
        variant=variant.as_dict(),
        resolved_variant=variant,
        mesh_path=variant.mesh_path,
    )

    manifest = build_native_problem_manifest(resolved, case_id="synthetic", analysis="ssr", elem_type="P4")
    neumann_rows = build_mechanics_neumann_label_rows(resolved)

    assert manifest["dmplex"]["support_counts"] == {
        "regions": 1,
        "boundaries": 2,
        "nodesets": 1,
        "boundary_geometry": 1,
    }
    assert manifest["rule_counts"]["mechanics_neumann"] == 1
    assert manifest["boundary_geometry"]["slope_curve"]["support_boundary"] == "slope_face"
    assert manifest["boundary_geometry"]["slope_curve"]["support"] == {
        "name": "slope_face",
        "kind": "boundary",
        "tag": 3,
        "dm_label": "Face Sets",
    }
    neumann = manifest["mechanics"]["neumann"][0]
    assert neumann["target"] == "slope_face"
    assert neumann["support"]["tag"] == 3
    assert neumann["geometry_support"]["geometry_order"] == 2
    assert neumann["geometry_support"]["support"]["tag"] == 3
    assert neumann["native_status"] == "pending_native_curved_face_quadrature"
    assert len(neumann_rows) == 1
    assert set(neumann_rows[0]) == set(MECHANICS_NEUMANN_LABEL_COLUMNS)
    assert neumann_rows[0]["support_name"] == "slope_face"
    assert neumann_rows[0]["dm_label"] == "Face Sets"
    assert neumann_rows[0]["value_model"] == '{"type":"constant","value":[0.0,-10.0,0.0]}'
    assert neumann_rows[0]["native_status"] == "pending_native_curved_face_quadrature"
    assert json.loads(json.dumps(manifest)) == manifest


def test_native_problem_manifest_marks_affine_neumann_as_native_quadrature_ready(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.msh"
    mesh_path.write_text(
        "\n".join(
            [
                "$MeshFormat",
                "2.2 0 8",
                "$EndMeshFormat",
                "$PhysicalNames",
                "2",
                '3 1 "region:body"',
                '2 3 "boundary:slope_face"',
                "$EndPhysicalNames",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    asset = build_problem_asset_3d(
        asset_id="synthetic_affine_neumann",
        asset_dir=tmp_path,
        default_variant="mesh.msh",
        mesh_variants={"mesh.msh": {"source": {"path": "mesh.msh"}}},
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
                    "target": "slope_face",
                    "kind": "traction",
                    "value_model": {"type": "constant", "value": [0.0, -10.0, 0.0]},
                }
            ],
        },
    )
    variant = asset.resolve_variant(None)
    resolved = ResolvedAsset(
        definition=asset,
        variant_name=variant.name,
        variant=variant.as_dict(),
        resolved_variant=variant,
        mesh_path=variant.mesh_path,
    )

    manifest = build_native_problem_manifest(resolved, case_id="synthetic-affine", analysis="ssr", elem_type="P4")
    rows = build_mechanics_neumann_label_rows(resolved)

    assert manifest["mechanics"]["neumann"][0]["native_status"] == "native_face_quadrature_affine"
    assert rows[0]["geometry"] == ""
    assert rows[0]["geometry_order"] == ""
    assert rows[0]["native_status"] == "native_face_quadrature_affine"


def test_case_artifacts_write_neumann_label_table_for_neumann_assets(tmp_path: Path) -> None:
    from petsc_ssr.case_config import (
        CaseTranslation,
        write_mechanics_neumann_label_table,
        write_native_problem_manifest,
    )

    mesh_path = tmp_path / "mesh.msh"
    mesh_path.write_text(
        "\n".join(
            [
                "$MeshFormat",
                "2.2 0 8",
                "$EndMeshFormat",
                "$PhysicalNames",
                "2",
                '3 1 "region:body"',
                '2 3 "boundary:slope_face"',
                "$EndPhysicalNames",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    asset = build_problem_asset_3d(
        asset_id="synthetic_neumann_artifact",
        asset_dir=tmp_path,
        default_variant="mesh.msh",
        mesh_variants={"mesh.msh": {"source": {"path": "mesh.msh"}}},
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
        boundary_geometry={"slope_curve": {"support_boundary": "slope_face", "geometry_order": 2}},
        mechanics={
            "neumann": [
                {
                    "target": "slope_face",
                    "kind": "traction",
                    "geometry": "slope_curve",
                    "value_model": {"type": "constant", "value": [0.0, -10.0, 0.0]},
                }
            ],
        },
    )
    variant = asset.resolve_variant(None)
    resolved = ResolvedAsset(
        definition=asset,
        variant_name=variant.name,
        variant=variant.as_dict(),
        resolved_variant=variant,
        mesh_path=variant.mesh_path,
    )

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.problem = _Cfg()
    cfg.problem.seepage_coupled = False
    cfg.problem.name = "synthetic_neumann_artifact"
    cfg.problem.analysis = "ssr"
    cfg.problem.elem_type = "P4"
    cfg.linear_solver = _Cfg()
    cfg.linear_solver.profile = "pmg-deflated-baseline"
    cfg.linear_solver.resolved_world_size = 1

    translation = CaseTranslation(True, "synthetic", config=cfg)

    import petsc_ssr.case_config as case_config

    original = case_config.resolve_problem_asset_from_config if hasattr(case_config, "resolve_problem_asset_from_config") else None

    def _resolved(_cfg):
        return resolved

    # write_* imports the resolver into function scope; monkeypatch the module import target.
    import petsc_ssr.problem_asset_runtime as runtime

    old_runtime_resolver = runtime.resolve_problem_asset_from_config
    runtime.resolve_problem_asset_from_config = _resolved
    try:
        path = write_mechanics_neumann_label_table(translation, tmp_path / "out")
        manifest_path = write_native_problem_manifest(translation, tmp_path / "out")
    finally:
        runtime.resolve_problem_asset_from_config = old_runtime_resolver
        if original is not None:
            case_config.resolve_problem_asset_from_config = original

    assert path is not None
    table_text = path.read_text(encoding="utf-8")
    assert table_text.splitlines()[0] == ",".join(MECHANICS_NEUMANN_LABEL_COLUMNS)
    assert '"{""type"":""constant"",""value"":[0.0,-10.0,0.0]}"' in table_text
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["value_model"] == '{"type":"constant","value":[0.0,-10.0,0.0]}'
    assert rows[0]["native_status"] == "pending_native_curved_face_quadrature"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["native_inputs"]["mechanics_neumann_labels_csv"] == str(path)
    assert manifest["native_inputs"]["mechanics_neumann_labels_row_fingerprint"].startswith("fnv1a64:")


def test_boundary_geometry_requires_declared_boundary_support(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.msh"
    mesh_path.write_text(
        "\n".join(
            [
                "$MeshFormat",
                "2.2 0 8",
                "$EndMeshFormat",
                "$PhysicalNames",
                "2",
                '3 1 "region:body"',
                '2 2 "boundary:slope_face"',
                "$EndPhysicalNames",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    asset = build_problem_asset_3d(
        asset_id="synthetic_missing_geometry_support",
        asset_dir=tmp_path,
        default_variant="mesh.msh",
        mesh_variants={"mesh.msh": {"source": {"path": "mesh.msh"}}},
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
        boundary_geometry={"missing_curve": {"support_boundary": "missing_face", "geometry_order": 2}},
        mechanics={"neumann": []},
    )
    variant = asset.resolve_variant(None)
    resolved = ResolvedAsset(
        definition=asset,
        variant_name=variant.name,
        variant=variant.as_dict(),
        resolved_variant=variant,
        mesh_path=variant.mesh_path,
    )

    with pytest.raises(ValueError, match="missing_curve.*missing_face.*boundary physical name"):
        build_native_problem_manifest(resolved, case_id="synthetic", analysis="ssr", elem_type="P4")


def test_native_problem_manifest_rejects_unresolved_rule_supports(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.msh"
    mesh_path.write_text(
        "\n".join(
            [
                "$MeshFormat",
                "2.2 0 8",
                "$EndMeshFormat",
                "$PhysicalNames",
                "2",
                '3 1 "region:body"',
                '2 2 "boundary:base"',
                "$EndPhysicalNames",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    asset = build_problem_asset_3d(
        asset_id="synthetic_missing_rule_support",
        asset_dir=tmp_path,
        default_variant="mesh.msh",
        mesh_variants={"mesh.msh": {"source": {"path": "mesh.msh"}}},
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
        mechanics={"dirichlet": [{"target": "missing_nodeset", "components": ["x"]}]},
    )
    variant = asset.resolve_variant(None)
    resolved = ResolvedAsset(
        definition=asset,
        variant_name=variant.name,
        variant=variant.as_dict(),
        resolved_variant=variant,
        mesh_path=variant.mesh_path,
    )

    with pytest.raises(ValueError, match="Dirichlet target 'missing_nodeset'.*Gmsh physical support"):
        build_native_problem_manifest(resolved, case_id="synthetic", analysis="ssr", elem_type="P4")


def test_neumann_geometry_must_reference_target_boundary(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.msh"
    mesh_path.write_text(
        "\n".join(
            [
                "$MeshFormat",
                "2.2 0 8",
                "$EndMeshFormat",
                "$PhysicalNames",
                "3",
                '3 1 "region:body"',
                '2 2 "boundary:base"',
                '2 3 "boundary:slope_face"',
                "$EndPhysicalNames",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    asset = build_problem_asset_3d(
        asset_id="synthetic_mismatched_geometry",
        asset_dir=tmp_path,
        default_variant="mesh.msh",
        mesh_variants={"mesh.msh": {"source": {"path": "mesh.msh"}}},
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
    variant = asset.resolve_variant(None)
    resolved = ResolvedAsset(
        definition=asset,
        variant_name=variant.name,
        variant=variant.as_dict(),
        resolved_variant=variant,
        mesh_path=variant.mesh_path,
    )

    with pytest.raises(ValueError, match="slope_face.*base_curve.*base"):
        build_mechanics_neumann_label_rows(resolved)


def test_problem_options_pass_native_manifest_to_petsc_options() -> None:
    problem = ProblemSpec.l1_slope()
    problem.metadata["native_problem_manifest"] = ".local/run/data/native_problem_manifest.json"
    problem.metadata["mechanics_bc_labels_csv"] = ".local/run/data/mechanics_bc_labels.csv"
    problem.metadata["mechanics_neumann_labels_csv"] = ".local/run/data/mechanics_neumann_labels.csv"
    problem.metadata["seepage_boundary_labels_csv"] = ".local/run/data/seepage_boundary_labels.csv"

    tokens = problem.option_tokens()

    idx = tokens.index("-native_problem_manifest")
    assert tokens[idx + 1] == ".local/run/data/native_problem_manifest.json"
    idx = tokens.index("-mechanics_bc_labels_csv")
    assert tokens[idx + 1] == ".local/run/data/mechanics_bc_labels.csv"
    idx = tokens.index("-mechanics_neumann_labels_csv")
    assert tokens[idx + 1] == ".local/run/data/mechanics_neumann_labels.csv"
    idx = tokens.index("-seepage_boundary_labels_csv")
    assert tokens[idx + 1] == ".local/run/data/seepage_boundary_labels.csv"


def test_coordinate_constraint_problem_option_requires_debug_flag() -> None:
    problem = ProblemSpec.l1_slope()
    problem.metadata["mechanics_bc_nodes_csv"] = ".local/run/data/mechanics_bc_nodes.csv"

    with pytest.raises(ValueError, match="debug compatibility input"):
        problem.option_tokens()

    problem.metadata["debug_coordinate_bc_table"] = True
    tokens = problem.option_tokens()

    idx = tokens.index("-debug_coordinate_bc_table")
    assert tokens[idx + 1] == "true"
    idx = tokens.index("-mechanics_bc_nodes_csv")
    assert tokens[idx + 1] == ".local/run/data/mechanics_bc_nodes.csv"


def test_seepage_pressure_problem_option_requires_source_contract() -> None:
    problem = ProblemSpec.l1_slope()
    problem.metadata["seepage_pressure_csv"] = ".local/run/data/seepage_pressure.csv"

    with pytest.raises(ValueError, match="seepage_pressure_source"):
        problem.option_tokens()

    problem.metadata["seepage_pressure_source"] = "hydro_prepass_coordinate_bridge"
    tokens = problem.option_tokens()

    idx = tokens.index("-seepage_pressure_source")
    assert tokens[idx + 1] == "hydro_prepass_coordinate_bridge"
    idx = tokens.index("-seepage_pressure_csv")
    assert tokens[idx + 1] == ".local/run/data/seepage_pressure.csv"


def test_native_c_consumes_manifest_option_for_options_left_cleanliness() -> None:
    context_source = (ROOT / "src" / "petsc_ssr" / "native" / "core" / "context.c.inc").read_text(encoding="utf-8")
    engine_source = (ROOT / "src" / "petsc_ssr" / "native" / "core" / "engine_main.c").read_text(encoding="utf-8")
    manifest_source = (ROOT / "src" / "petsc_ssr" / "native" / "io" / "problem_manifest.c.inc").read_text(encoding="utf-8")
    cli_source = (ROOT / "src" / "petsc_ssr" / "native" / "core" / "cli_runner.c.inc").read_text(encoding="utf-8")
    cython_source = (ROOT / "src" / "petsc_ssr" / "native" / "cython" / "cython_api.c.inc").read_text(encoding="utf-8")
    assembly_header = (ROOT / "src" / "petsc_ssr" / "native" / "assembly" / "assembly.h").read_text(encoding="utf-8")
    assembly_source = (ROOT / "src" / "petsc_ssr" / "native" / "assembly" / "assembly.c").read_text(encoding="utf-8")
    neumann_source = (ROOT / "src" / "petsc_ssr" / "native" / "assembly" / "neumann.c").read_text(encoding="utf-8")
    reporting_source = (ROOT / "src" / "petsc_ssr" / "native" / "reporting" / "reporting.c.inc").read_text(encoding="utf-8")

    assert 'PetscOptionsString("-native_problem_manifest"' in context_source
    assert 'PetscOptionsString("-mechanics_bc_labels_csv"' in context_source
    assert 'PetscOptionsString("-mechanics_neumann_labels_csv"' in context_source
    assert 'PetscOptionsString("-seepage_boundary_labels_csv"' in context_source
    assert 'PetscOptionsString("-seepage_pressure_source"' in context_source
    assert 'PetscOptionsBool("-debug_coordinate_bc_table"' in context_source
    assert "../io/problem_manifest.c.inc" in engine_source
    assert "NativeProblemManifestApply" in manifest_source
    assert "NativeProblemManifestValidateMetadata" in manifest_source
    assert "NativeProblemManifestValidateTopology" in manifest_source
    assert "NativeProblemManifestValidateBoundaryRules" in manifest_source
    assert "NativeManifestFindInt" in manifest_source
    assert "NativeManifestFindIntBounded" in manifest_source
    assert "NativeManifestFindBool" in manifest_source
    assert "NativeManifestCountArrayInObject" in manifest_source
    assert "NativeManifestCountObjectInObject" in manifest_source
    assert "NativeManifestFindMemberValueBounded" in manifest_source
    assert "NativeManifestFindObjectMember" in manifest_source
    assert "NativeProblemManifestValidateDeclaredCount" in manifest_source
    assert '"support_counts"' in manifest_source
    assert '"rule_counts"' in manifest_source
    assert "dmplex.support_counts" in manifest_source
    assert 'NativeManifestFindObjectMember(json, "boundary_geometry"' in manifest_source
    assert "NativeProblemManifestApplyNativeInputPath" in manifest_source
    assert "NativeProblemManifestApplyNativeInputBool" in manifest_source
    assert "NativeProblemManifestValidateLabelTableFingerprint" in manifest_source
    assert "NativeManifestFileFingerprint" in manifest_source
    assert "NATIVE_PROBLEM_MANIFEST_ROW_FINGERPRINT" in manifest_source
    assert "fnv1a64:" in manifest_source
    assert "mechanics_label_constraints_row_fingerprint" in manifest_source
    assert "mechanics_neumann_labels_row_fingerprint" in manifest_source
    assert "seepage_boundary_labels_row_fingerprint" in manifest_source
    assert "native_problem_manifest_loaded" in context_source
    assert "native_manifest_mechanics_dirichlet" in context_source
    assert "native_manifest_mechanics_neumann" in context_source
    assert "native_manifest_seepage_head" in context_source
    assert "native_manifest_seepage_flux" in context_source
    assert "rule_stats.mechanics_dirichlet" in manifest_source
    assert "rule_stats.mechanics_neumann" in manifest_source
    assert "rule_stats.seepage_head" in manifest_source
    assert "rule_stats.seepage_flux" in manifest_source
    assert "conflicts with already configured value" in manifest_source
    assert 'NativeManifestFindObjectMember(json, "native_inputs"' in manifest_source
    assert "NativeManifestFindStringBounded(native_inputs_begin" in manifest_source
    assert '"debug_coordinate_bc_table", &app->debug_coordinate_bc_table' in manifest_source
    assert '"seepage_pressure_source", app->seepage_pressure_source' in manifest_source
    assert "petsc_ssr_native_problem_manifest" in manifest_source
    assert "schema_version" in manifest_source
    assert "resolved_world_size" in manifest_source
    assert "does not match MPI size" in manifest_source
    assert "does not match option -element_degree" in manifest_source
    assert "NATIVE_PROBLEM_MANIFEST_METADATA" in manifest_source
    assert "NATIVE_PROBLEM_MANIFEST_LABELS" in manifest_source
    assert "NATIVE_PROBLEM_MANIFEST_RULES" in manifest_source
    assert "dmplex.supports.regions" in manifest_source
    assert "mechanics_dirichlet" in manifest_source
    assert "seepage_head" in manifest_source
    assert "declares mechanics Dirichlet rules but no mechanics label constraint table" in manifest_source
    assert "no mechanics label or coordinate constraint table" not in manifest_source
    assert "declares mechanics Neumann rules but no mechanics Neumann label table" in manifest_source
    assert "declares seepage boundary rules but no seepage boundary label table" in manifest_source
    assert "mechanics_label_constraints_csv" in manifest_source
    assert "mechanics_neumann_labels_csv" in manifest_source
    assert "seepage_boundary_labels_csv" in manifest_source
    assert "NativeProblemManifestApply(&app)" in cli_source
    assert "NativeProblemManifestApply(&ctx->app)" in cython_source
    assert "AssemblyCtxLoadLabelConstraintsCSV" in assembly_header
    assert "AssemblyCtxLoadLabelConstraintsCSV" in assembly_source
    assert "AssemblyLabelConstraintStats" in assembly_header
    assert "PetscSectionGetConstraintDof(gsec, point" not in assembly_source
    assert "AppendPoint(raw_points[i], &expanded" in assembly_source
    assert "DMPlexGetTransitiveClosure(ctx->dm, raw_points[i]" in assembly_source
    assert "AppendExistingSectionConstraintDofsForComponents" in assembly_source
    assert "missing_vertex_label_using_section_constraints" in assembly_source
    assert "DMGetLocalSection(ctx->dm, &lsec)" in assembly_source
    assert "constraint_indices ? constraint_indices[k] : k" in assembly_source
    assert "label_stats.rows == app.native_manifest_mechanics_dirichlet" in cli_source
    assert "label_stats.rows == ctx->app.native_manifest_mechanics_dirichlet" in cython_source
    assert "AssemblyCtxLoadNeumannLabelsCSV" in assembly_header
    assert "AssemblyCtxLoadNeumannLabelsCSV" in neumann_source
    assert "AssemblyNeumannRule" in assembly_header
    assert "neumann_rule_count" in assembly_header
    assert "neumann_rules" in assembly_header
    assert "AssemblySeepageBoundaryRule" in assembly_header
    assert "seepage_boundary_rule_count" in assembly_header
    assert "seepage_boundary_rules" in assembly_header
    assert "PetscFree(ctx->neumann_rules)" in assembly_source
    assert "PetscFree(ctx->seepage_boundary_rules)" in assembly_source
    assert "AssemblyCtxAppendNeumannRule" in neumann_source
    assert "AssemblyCtxClearNeumannRules" in neumann_source
    assert "AssemblyCtxAppendSeepageBoundaryRule" in assembly_source
    assert "AssemblyCtxClearSeepageBoundaryRules" in assembly_source
    assert "missing_vertex_label_coordinate_pressure_bridge_active" in assembly_source
    assert "SplitQuotedCsvFields" in assembly_source
    assert "expected exactly 10" in assembly_source
    assert "label_ready_coordinate_pressure_bridge_active" in assembly_source
    assert "validate_native_problem_artifact_contract" in (ROOT / "src" / "petsc_ssr" / "problem_asset_runtime.py").read_text(encoding="utf-8")
    assert "value_model_name" in assembly_header
    assert "PetscBool in_quotes = PETSC_FALSE" in neumann_source
    assert "if (in_quotes && *src == '\"')" in neumann_source
    assert "!in_quotes && (c == ','" in neumann_source
    assert "nfields == 9" in neumann_source
    assert "expected exactly 9" in neumann_source
    assert "NeumannParseGeometryOrder" in neumann_source
    assert "NeumannValidateNativeStatus" in neumann_source
    assert "SsrNeumannValueRegistryFind(model_name" in neumann_source
    assert "last_value_model" in assembly_header
    assert "affine_rows" in assembly_header
    assert "curved_rows" in assembly_header
    assert "last_geometry_order" in assembly_header
    assert "last_native_status" in assembly_header
    assert "constant-traction" in neumann_source
    assert "staged_rules=%\" PetscInt_FMT" in neumann_source
    assert "AssemblyCtxValidateNeumannLabelsCSV" in assembly_header
    assert "AssemblyCtxValidateNeumannLabelsCSV" in neumann_source
    assert "expected_rows" in neumann_source
    assert "mechanics Neumann row(s), but label table" in neumann_source
    assert "AssemblyCtxAssembleNeumannResidual" in assembly_header
    assert "AssemblyCtxAssembleNeumannResidual" in neumann_source
    assert "ctx->neumann_rule_count == 0" in neumann_source
    assert "SsrNeumannStats neumann_stats" in assembly_header
    assert "NeumannParseConstantTraction" in neumann_source
    assert "NeumannFaceQuadrature" in neumann_source
    assert "NeumannBuildBasisAlphas" in neumann_source
    assert "DMPlexVecSetClosure(dm, lsec, rhs_loc, cell, elem_vec, ADD_VALUES)" in neumann_source
    assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_NEUMANN" in neumann_source
    assert "SsrStatsAddNeumannAssembly(&ctx->neumann_stats" in neumann_source
    assert "MECHANICS_NEUMANN_ASSEMBLY" in neumann_source
    assert "native_face_quadrature_affine" in neumann_source
    assert "pending_native_curved_face_quadrature" in neumann_source
    assert "native curved face quadrature is not implemented yet" in neumann_source
    assert "AssemblyCtxAssembleNeumannResidual(ctx, f_ext)" in assembly_source
    assert "AssemblyCtxValidateSeepageBoundaryLabelsCSV" in assembly_header
    assert "AssemblyCtxValidateSeepageBoundaryLabelsCSV" in assembly_source
    assert "expected_head_rows" in assembly_source
    assert "expected_flux_rows" in assembly_source
    assert "seepage head row(s), but label table" in assembly_source
    assert "seepage flux row(s), but label table" in assembly_source
    assert "metadata_only_pressure_csv_active" in assembly_source
    assert "native_status=%s" in neumann_source
    assert "-seepage_pressure_csv requires -seepage_pressure_source hydro_prepass_coordinate_bridge" in cli_source
    assert "-seepage_pressure_csv requires -seepage_pressure_source hydro_prepass_coordinate_bridge" in cython_source
    assert cli_source.index("AssemblyCtxLoadLabelConstraintsCSV") < cli_source.index("AssemblyCtxLoadCoordinateConstraintsCSV")
    assert cython_source.index("AssemblyCtxLoadLabelConstraintsCSV") < cython_source.index("AssemblyCtxLoadCoordinateConstraintsCSV")
    assert "-mechanics_bc_nodes_csv requires -debug_coordinate_bc_table true" in cli_source
    assert "-mechanics_bc_nodes_csv requires -debug_coordinate_bc_table true" in cython_source
    assert "status=debug_coordinate_override" in cli_source
    assert "status=debug_coordinate_override" in cython_source
    assert "native_problem_manifest" in reporting_source
    assert "mechanics_bc_labels_csv" in reporting_source
    assert "mechanics_neumann_labels_csv" in reporting_source
    assert "seepage_boundary_labels_csv" in reporting_source


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_all_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_all_keys(item))
        return keys
    return set()
