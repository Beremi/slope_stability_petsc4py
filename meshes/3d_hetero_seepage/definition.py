from pathlib import Path

from petsc_ssr.assets.factories import build_problem_asset_3d, build_seepage_spec


ASSET_DIR = Path(__file__).resolve().parent

ASSET = build_problem_asset_3d(
    asset_id="3d_hetero_seepage",
    asset_dir=ASSET_DIR,
    default_variant="concave_family_b.msh",
    mesh_variants={
        name: {"source": {"path": name}}
        for name in (
            "family_a.msh",
            "family_b.msh",
            "family_c.msh",
            "family_d.msh",
            "family_e.msh",
            "family_f.msh",
            "concave_family_a.msh",
            "concave_family_b.msh",
            "concave_family_c.msh",
            "concave_family_d.msh",
        )
    },
    materials={
        "cover_layer": {"c0": 15.0, "phi": 30.0, "psi": 0.0, "young": 10000.0, "poisson": 0.33, "gamma_sat": 19.0, "gamma_unsat": 19.0, "hydraulic_conductivity": 1.0},
        "general_foundation": {"c0": 15.0, "phi": 38.0, "psi": 0.0, "young": 50000.0, "poisson": 0.30, "gamma_sat": 22.0, "gamma_unsat": 22.0, "hydraulic_conductivity": 1.0},
        "weak_foundation": {"c0": 10.0, "phi": 35.0, "psi": 0.0, "young": 50000.0, "poisson": 0.30, "gamma_sat": 21.0, "gamma_unsat": 21.0, "hydraulic_conductivity": 1.0},
        "slope_mass": {"c0": 18.0, "phi": 32.0, "psi": 0.0, "young": 20000.0, "poisson": 0.33, "gamma_sat": 20.0, "gamma_unsat": 20.0, "hydraulic_conductivity": 1.0},
    },
    region_assignment={
        "cover_layer": "cover_layer",
        "general_foundation": "general_foundation",
        "weak_foundation": "weak_foundation",
        "slope_mass": "slope_mass",
    },
    mechanics={
        "dirichlet": [
            {"target": "x_lock", "components": ["x"]},
            {"target": "y_lateral_lock", "components": ["z"]},
            {"target": "base", "components": ["x", "y", "z"]},
        ],
    },
    seepage=build_seepage_spec(
        water_unit_weight=9.81,
        conductivity_mode="by_material",
        head_bcs=[
            {"target": "head_dry", "kind": "dry"},
            {"target": "head_porous", "kind": "constant_level", "level": 55.0},
            {"target": "head_free", "kind": "constant_level", "level": 35.0},
        ],
    ),
)
