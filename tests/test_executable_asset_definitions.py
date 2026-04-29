from __future__ import annotations

from pathlib import Path

import pytest
import meshio

from slope_stability.assets import ProblemAssetAPI, available_problem_assets, load_problem_asset


ROOT = Path(__file__).resolve().parents[1]

CANONICAL_ASSETS = {
    "2d_franz_dam",
    "2d_homo_slope",
    "2d_kozinec",
    "2d_luzec",
    "2d_sloan2013",
    "3d_hetero_seepage",
    "3d_hetero_seepage_transition",
    "3d_hetero_slope",
    "3d_homo_slope",
    "3d_siopt",
}


def test_mesh_root_exposes_only_canonical_assets() -> None:
    assert set(available_problem_assets()) == CANONICAL_ASSETS


def test_canonical_assets_expose_executable_api() -> None:
    for asset_name in sorted(CANONICAL_ASSETS):
        asset = load_problem_asset(asset_name)
        assert isinstance(asset, ProblemAssetAPI)
        assert asset.asset_id == asset_name
        assert asset.source_kind == "gmsh_problem_asset"
        assert asset.default_variant in asset.list_variants()
        variant = asset.resolve_variant(asset.default_variant)
        assert variant.name == asset.default_variant
        assert callable(asset.build_mesh)
        assert callable(asset.build_mechanics)
        assert callable(asset.build_seepage)


def test_invalid_variant_resolution_raises() -> None:
    asset = load_problem_asset("3d_homo_slope")
    with pytest.raises(KeyError, match="Unknown mesh variant"):
        asset.resolve_variant("missing.msh")


def test_seepage_and_mechanics_capabilities_match_asset_families() -> None:
    assert load_problem_asset("2d_sloan2013").capabilities == frozenset({"seepage"})
    assert load_problem_asset("2d_homo_slope").capabilities == frozenset({"mechanics"})
    assert load_problem_asset("3d_hetero_seepage").capabilities == frozenset({"mechanics", "seepage"})


def test_asset_definitions_own_materials_and_hydraulics() -> None:
    for asset_name in sorted(CANONICAL_ASSETS):
        asset = load_problem_asset(asset_name)
        if "mechanics" in asset.capabilities:
            rows = asset.material_rows()
            assert rows is not None
            assert len(rows) > 0
        if "seepage" in asset.capabilities:
            spec = asset.seepage_spec()
            assert spec is not None
            assert spec.water_unit_weight > 0.0
            assert spec.head_bcs
            assert asset.hydraulic_conductivity() is not None


def test_removed_legacy_asset_aliases_are_not_loadable() -> None:
    for legacy_name in (
        "2d_generated_homo",
        "3d_homo_ssr",
        "3d_homo_ll",
        "3d_hetero_ssr",
        "3d_hetero_ll",
        "3d_hetero_seepage_ssr_comsol",
    ):
        with pytest.raises(FileNotFoundError):
            load_problem_asset(legacy_name)


def test_all_canonical_mesh_variants_use_canonical_msh41_contract() -> None:
    allowed_cell_types = {"vertex", "line", "line3", "triangle", "triangle6", "tetra"}
    allowed_name_prefixes = {"region", "boundary", "nodeset", "boundary_geom"}

    for asset_name in sorted(CANONICAL_ASSETS):
        asset = load_problem_asset(asset_name)
        for variant in asset.list_variants().values():
            assert variant.mesh_path is not None
            header = variant.mesh_path.read_text(encoding="utf-8").splitlines()[:3]
            assert header[:3] == ["$MeshFormat", "4.1 0 8", "$EndMeshFormat"]

            mesh = meshio.read(variant.mesh_path)
            cell_types = {block.type for block in mesh.cells}
            assert cell_types <= allowed_cell_types
            assert ("triangle" if asset.dimension == 2 else "tetra") in cell_types
            for name in mesh.field_data:
                prefix = str(name).split(":", 1)[0]
                assert prefix in allowed_name_prefixes
