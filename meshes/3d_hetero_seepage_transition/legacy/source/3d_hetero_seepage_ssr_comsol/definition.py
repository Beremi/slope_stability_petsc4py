from pathlib import Path

from slope_stability.assets.factories import build_asset
from slope_stability.assets.support.gmsh import gmsh_variants_from_dir


ASSET_DIR = Path(__file__).resolve().parent

ASSET = build_asset(
    asset_id="3d_hetero_seepage_ssr_comsol",
    asset_dir=ASSET_DIR,
    dimension=3,
    source_kind="gmsh_tagged_simplex",
    capabilities=["mechanics", "seepage"],
    default_variant="comsol_mesh.msh",
    mesh_variants=gmsh_variants_from_dir(ASSET_DIR),
    materials=[
        {"id": 0, "name": "cover_layer", "c0": 15.0, "phi": 30.0, "psi": 0.0, "young": 10000.0, "poisson": 0.33, "gamma_sat": 19.0, "gamma_unsat": 19.0},
        {"id": 1, "name": "general_foundation", "c0": 15.0, "phi": 38.0, "psi": 0.0, "young": 50000.0, "poisson": 0.30, "gamma_sat": 22.0, "gamma_unsat": 22.0},
        {"id": 2, "name": "weak_foundation", "c0": 10.0, "phi": 35.0, "psi": 0.0, "young": 50000.0, "poisson": 0.30, "gamma_sat": 21.0, "gamma_unsat": 21.0},
        {"id": 3, "name": "slope_mass", "c0": 18.0, "phi": 32.0, "psi": 0.0, "young": 20000.0, "poisson": 0.33, "gamma_sat": 20.0, "gamma_unsat": 20.0},
    ],
    mechanics={
        "default_boundary_type": 1,
        "dirichlet": [
            {"labels": [1, 2], "components": ["x"]},
            {"labels": [3, 4], "components": ["z"]},
            {"labels": [5], "components": ["y"]},
            {"labels": [5], "components": ["x", "z"], "boundary_type": [1]},
        ],
    },
    seepage={
        "water_unit_weight": 9.81,
        "conductivity_mode": "uniform",
        "conductivity": [1.0],
        "water_levels": {"free": 35.0, "porous": 55.0},
        "hydraulic_boundaries": {
            "mode": "hybrid_transition",
            "dry_labels": [6],
            "porous_labels": [2],
            "free_labels": [1],
            "geometry_recipe": {
                "base_point": [55.0, 30.0, 0.0],
                "toe_point": [115.0, 60.0, 0.0],
                "apex_left": [30.0, 30.0, 43.3],
                "apex_right": [30.0, 30.0, -43.3],
                "bed_y": 30.0,
                "triangle_normal_tolerance": 1.0e-1,
                "plane_distance_tolerance": 1.0e-6,
                "bed_tolerance": 1.0e-1,
                "sector_tolerance": 1.0e-10,
            },
        },
    },
    mesh_builder_kind="comsol_p2",
)
