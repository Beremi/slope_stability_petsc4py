from pathlib import Path

from slope_stability.assets.factories import build_asset
from slope_stability.assets.support.gmsh import gmsh_variants_from_dir


ASSET_DIR = Path(__file__).resolve().parent

ASSET = build_asset(
    asset_id="3d_hetero_ll",
    asset_dir=ASSET_DIR,
    dimension=3,
    source_kind="gmsh_tagged_simplex",
    capabilities=["mechanics"],
    default_variant="LL_hetero_ada_L1.msh",
    mesh_variants=gmsh_variants_from_dir(ASSET_DIR),
    materials=[
        {"id": 0, "name": "cover_layer", "c0": 15.0, "phi": 30.0, "psi": 0.0, "young": 10000.0, "poisson": 0.33, "gamma_sat": 19.0, "gamma_unsat": 19.0},
        {"id": 1, "name": "general_foundation", "c0": 15.0, "phi": 38.0, "psi": 0.0, "young": 50000.0, "poisson": 0.30, "gamma_sat": 22.0, "gamma_unsat": 22.0},
        {"id": 2, "name": "weak_foundation", "c0": 10.0, "phi": 35.0, "psi": 0.0, "young": 50000.0, "poisson": 0.30, "gamma_sat": 21.0, "gamma_unsat": 21.0},
        {"id": 3, "name": "slope_mass", "c0": 18.0, "phi": 32.0, "psi": 0.0, "young": 20000.0, "poisson": 0.33, "gamma_sat": 20.0, "gamma_unsat": 20.0},
    ],
    mechanics={"dirichlet": [{"labels": [1, 2], "components": ["x"]}, {"labels": [5], "components": ["y"]}, {"labels": [3, 4], "components": ["z"]}]},
)
