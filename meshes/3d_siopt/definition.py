from pathlib import Path

from slope_stability.assets.factories import build_problem_asset_3d


ASSET_DIR = Path(__file__).resolve().parent

ASSET = build_problem_asset_3d(
    asset_id="3d_siopt",
    asset_dir=ASSET_DIR,
    default_variant="reference_l0.msh",
    mesh_variants={
        name: {"source": {"path": name}}
        for name in (
            "reference_l0.msh",
            "reference_l1.msh",
            "reference_l5.msh",
        )
    },
    materials={
        "siopt_reference": {"c0": 15.0, "phi": 20.0, "psi": 20.0, "young": 40000.0, "poisson": 0.30, "gamma_sat": 20.0, "gamma_unsat": 20.0},
    },
    region_assignment={
        "reference_mass": "siopt_reference",
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
)
