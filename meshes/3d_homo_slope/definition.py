from pathlib import Path

from petsc_ssr.assets.factories import build_problem_asset_3d


ASSET_DIR = Path(__file__).resolve().parent

ASSET = build_problem_asset_3d(
    asset_id="3d_homo_slope",
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
        "homogeneous_slope": {"c0": 6.0, "phi": 45.0, "psi": 0.0, "young": 40000.0, "poisson": 0.30, "gamma_sat": 20.0, "gamma_unsat": 20.0},
    },
    region_assignment={
        "slope_mass": "homogeneous_slope",
    },
    mechanics={
        "dirichlet": [
            {"target": "x_lock", "components": ["x"]},
            {"target": "base", "components": ["x", "y", "z"]},
            {"target": "z_lock", "components": ["z"]},
        ],
    },
)
