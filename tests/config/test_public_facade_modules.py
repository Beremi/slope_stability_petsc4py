from __future__ import annotations

from pathlib import Path

import petsc_ssr.assets as assets
from petsc_ssr.assets.bcs import DirichletBCSpec, NeumannBCSpec, build_seepage_spec
from petsc_ssr.assets.curved import BoundaryGeometrySpec, boundary_geometry_specs
from petsc_ssr.assets.gmsh import parse_gmsh_physical_names
from petsc_ssr.assets.registry import available_problem_assets, load_problem_asset
from petsc_ssr.config.resolver import resolve_case_model, validate_case_payload
from petsc_ssr.config.schema import RunCaseConfig, load_run_case_config


ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = ROOT / "benchmarks" / "cases"


def test_config_schema_and_resolver_are_public_import_surfaces() -> None:
    case_toml = CASE_ROOT / "3d-heterogeneous-ssr-p4" / "case.toml"

    cfg = load_run_case_config(case_toml).validate()
    model = resolve_case_model(case_toml)
    payload = validate_case_payload(case_toml)

    assert isinstance(cfg, RunCaseConfig)
    assert model.config == cfg
    assert model.asset.asset_name == "3d_hetero_slope"
    assert model.asset.variant_name == "adaptive_family_a_l1.msh"
    assert model.pc_policy.variant == payload["pc_variant"]
    assert payload["resolved_pmg"]["p2_policy"] == "cap"


def test_asset_registry_and_gmsh_facades_reexport_existing_behavior() -> None:
    assert assets.available_problem_assets is available_problem_assets
    assert assets.load_problem_asset is load_problem_asset
    assert "3d_hetero_slope" in available_problem_assets()
    assert load_problem_asset("3d_hetero_slope").asset_id == "3d_hetero_slope"

    physical = parse_gmsh_physical_names(ROOT / "meshes" / "2d_homo_slope" / "h1.0.msh")
    assert "slope_mass" in physical["regions"]
    assert "base" in physical["boundaries"]


def test_asset_bc_and_curved_facades_expose_typed_contracts() -> None:
    seepage = build_seepage_spec(
        water_unit_weight=9.81,
        conductivity_mode="uniform",
        conductivity=1e-5,
        head_bcs=[{"target": "crest", "kind": "dry"}],
        flux_bcs=[{"target": "face", "kind": "zero", "value_model": {}}],
    )

    class _Asset:
        def boundary_geometry_specs(self):
            return {"curved_face": ("face", 2)}

    geometry = boundary_geometry_specs(_Asset())

    assert isinstance(DirichletBCSpec("base", ("x", "y")), DirichletBCSpec)
    assert isinstance(NeumannBCSpec("face", "constant", {}), NeumannBCSpec)
    assert seepage.conductivity == (1e-5,)
    assert geometry == {"curved_face": BoundaryGeometrySpec(name="curved_face", support_boundary="face", geometry_order=2)}
