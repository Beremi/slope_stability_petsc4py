from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "slope_stability"

FORBIDDEN_SUBSTRINGS = (
    "2d_franz_dam",
    "2d_homo_slope",
    "2d_kozinec",
    "2d_luzec",
    "2d_sloan2013",
    "3d_hetero_seepage",
    "3d_hetero_slope",
    "3d_homo_slope",
    "3d_siopt",
    "adaptive_family",
    "build_mesh_for_path",
    "concave_family",
    "comsol",
    "franz",
    "kozinec",
    "load_hydraulic_conductivity_for_path",
    "load_material_rows_for_path",
    "load_problem_asset_for_path",
    "load_seepage_spec_for_path",
    "load_water_unit_weight_for_path",
    "luzec",
    "mesh_boundary_type",
    "pmg_coarse_mesh_path",
    "run_2d_mechanics_capture",
    "run_2d_seepage_capture",
    "run_2d_homo",
    "run_2d_sloan",
    "run_2d_textmesh",
    "run_3d_mechanics_capture",
    "run_3d_seepage_capture",
    "run_3d_seepage_ssr_capture",
    "run_3d_hetero",
    "run_3d_homo",
    "siopt",
    "sloan",
    "textmesh",
    "transition_default",
    "waterlevels",
)


def test_src_does_not_commit_problem_specific_assets_or_defaults() -> None:
    hits: list[str] = []
    for path in sorted((SRC / "cli").glob("run_[23]d_*_capture.py")):
        hits.append(f"{path.relative_to(ROOT)}: route-specific CLI module")
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN_SUBSTRINGS:
            if token in text:
                hits.append(f"{relative}: {token}")
    assert hits == []
