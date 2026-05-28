from pathlib import Path

from petsc_ssr.assets.factories import build_problem_asset_2d


ASSET_DIR = Path(__file__).resolve().parent

ASSET = build_problem_asset_2d(
    asset_id="2d_homo_slope",
    asset_dir=ASSET_DIR,
    default_variant="h1.0.msh",
    mesh_variants={
        "h1.0.msh": {"source": {"path": "h1.0.msh"}},
        "h0.5.msh": {"source": {"path": "h0.5.msh"}},
    },
    materials={
        "homogeneous_slope": {"c0": 6.0, "phi": 45.0, "psi": 0.0, "young": 40000.0, "poisson": 0.30, "gamma_sat": 20.0, "gamma_unsat": 20.0},
    },
    region_assignment={
        "slope_mass": "homogeneous_slope",
    },
    mechanics={
        "dirichlet": [
            {"target": "left", "components": ["x"]},
            {"target": "right", "components": ["x"]},
            {"target": "base", "components": ["x", "y"]},
        ],
    },
)
