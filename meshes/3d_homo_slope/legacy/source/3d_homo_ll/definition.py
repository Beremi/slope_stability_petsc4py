from pathlib import Path

from slope_stability.assets.factories import build_asset
from slope_stability.assets.support.gmsh import gmsh_variants_from_dir


ASSET_DIR = Path(__file__).resolve().parent

ASSET = build_asset(
    asset_id="3d_homo_ll",
    asset_dir=ASSET_DIR,
    dimension=3,
    source_kind="gmsh_tagged_simplex",
    capabilities=["mechanics"],
    default_variant="LL_homo_ada_L1.msh",
    mesh_variants=gmsh_variants_from_dir(ASSET_DIR),
    materials=[{"id": 0, "name": "homogeneous_slope", "c0": 6.0, "phi": 45.0, "psi": 0.0, "young": 40000.0, "poisson": 0.30, "gamma_sat": 20.0, "gamma_unsat": 20.0}],
    mechanics={"dirichlet": [{"labels": [1, 2], "components": ["x"]}, {"labels": [5], "components": ["y"]}, {"labels": [3, 4], "components": ["z"]}]},
)
