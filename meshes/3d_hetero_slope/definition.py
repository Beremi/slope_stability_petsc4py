from pathlib import Path

from petsc_ssr.assets.factories import build_problem_asset_3d


ASSET_DIR = Path(__file__).resolve().parent

GLUED_BOTTOM_DIRICHLET = [
    {"target": "x_lock", "components": ["x"]},
    {"target": "base", "components": ["x", "y", "z"]},
    {"target": "z_lock", "components": ["z"]},
]

LEGACY_C_ROLLER_DIRICHLET = [
    {"target": "base", "components": ["y"]},
    {"target": "x_min", "components": ["x"]},
    {"target": "x_max", "components": ["x"]},
    {"target": "z_min", "components": ["z"]},
    {"target": "z_max", "components": ["z"]},
]

ASSET = build_problem_asset_3d(
    asset_id="3d_hetero_slope",
    asset_dir=ASSET_DIR,
    default_variant="adaptive_family_a_l1.msh",
    mesh_variants={
        name: {"source": {"path": name}}
        for name in (
            "adaptive_family_a_l1.msh",
            "adaptive_family_a_l2.msh",
            "adaptive_family_a_l3.msh",
            "adaptive_family_a_l4.msh",
            "adaptive_family_a_l5.msh",
            "uniform_family_a.msh",
            "adaptive_family_b_l1.msh",
            "adaptive_family_b_l2.msh",
            "adaptive_family_b_l3.msh",
            "adaptive_family_b_l4.msh",
            "adaptive_family_b_l5.msh",
            "uniform_family_b.msh",
        )
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
        "dirichlet": GLUED_BOTTOM_DIRICHLET,
        "profiles": {
            "default": {"dirichlet": GLUED_BOTTOM_DIRICHLET},
            "glued_bottom": {"dirichlet": GLUED_BOTTOM_DIRICHLET},
            "legacy_c_rollers": {"dirichlet": LEGACY_C_ROLLER_DIRICHLET},
        },
    },
)
