from pathlib import Path

from petsc_ssr.assets.factories import build_problem_asset_3d, build_seepage_spec


ASSET_DIR = Path(__file__).resolve().parent

ASSET = build_problem_asset_3d(
    asset_id="3d_hetero_seepage_transition",
    asset_dir=ASSET_DIR,
    default_variant="transition_default.msh",
    mesh_variants={
        "transition_default.msh": {"source": {"path": "transition_default.msh"}},
    },
    materials={
        "cover_layer": {"c0": 15.0, "phi": 30.0, "psi": 0.0, "young": 10000.0, "poisson": 0.33, "gamma_sat": 19.0, "gamma_unsat": 19.0},
        "general_foundation": {"c0": 15.0, "phi": 38.0, "psi": 0.0, "young": 50000.0, "poisson": 0.30, "gamma_sat": 22.0, "gamma_unsat": 22.0},
        "weak_foundation": {"c0": 10.0, "phi": 35.0, "psi": 0.0, "young": 50000.0, "poisson": 0.30, "gamma_sat": 21.0, "gamma_unsat": 21.0},
        "slope_mass": {"c0": 18.0, "phi": 32.0, "psi": 0.0, "young": 20000.0, "poisson": 0.33, "gamma_sat": 20.0, "gamma_unsat": 20.0},
    },
    region_assignment={
        "cover_layer": "cover_layer",
        "general_foundation": "general_foundation",
        "weak_foundation": "weak_foundation",
        "slope_mass": "slope_mass",
    },
    mechanics={
        "default_profile": "fixed_base",
        "dirichlet": [
            {"target": "x_lock", "components": ["x"]},
            {"target": "z_lock", "components": ["z"]},
            {"target": "base", "components": ["y"]},
        ],
        "profiles": {
            "roller_base": {
                "dirichlet": [
                    {"target": "x_lock", "components": ["x"]},
                    {"target": "z_lock", "components": ["z"]},
                    {"target": "base", "components": ["y"]},
                ],
            },
            "fixed_base": {
                "dirichlet": [
                    {"target": "x_lock", "components": ["x"]},
                    {"target": "z_lock", "components": ["z"]},
                    {"target": "base", "components": ["x", "y", "z"]},
                ],
            },
        },
    },
    seepage=build_seepage_spec(
        water_unit_weight=9.81,
        conductivity_mode="uniform",
        conductivity=[1.0],
        head_bcs=[
            {"target": "head_dry", "kind": "dry"},
            {"target": "head_porous", "kind": "constant_level", "level": 55.0},
            {"target": "head_free", "kind": "constant_level", "level": 35.0},
        ],
    ),
)
