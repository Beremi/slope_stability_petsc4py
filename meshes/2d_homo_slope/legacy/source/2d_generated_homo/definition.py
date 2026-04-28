from pathlib import Path

from slope_stability.assets.factories import build_asset
from slope_stability.assets.support.generated import generated_variant


ASSET_DIR = Path(__file__).resolve().parent

GEOMETRY = {
    "x1": 15.0,
    "x3": 15.0,
    "y1": 10.0,
    "y2": 10.0,
    "beta_deg": 45.0,
}

ASSET = build_asset(
    asset_id="2d_generated_homo",
    asset_dir=ASSET_DIR,
    dimension=2,
    source_kind="generated_geometry",
    capabilities=["mechanics"],
    default_variant="h1.0",
    mesh_variants={
        "h1.0": generated_variant("homogeneous_slope_2d", parameters={**GEOMETRY, "h": 1.0}),
        "h0.5": generated_variant("homogeneous_slope_2d", parameters={**GEOMETRY, "h": 0.5}),
    },
    materials=[
        {"id": 0, "name": "homogeneous_slope", "c0": 6.0, "phi": 45.0, "psi": 0.0, "young": 40000.0, "poisson": 0.30, "gamma_sat": 20.0, "gamma_unsat": 20.0},
    ],
    mechanics={
        "dirichlet": [
            {"selector": {"kind": "coordinate_plane", "axis": "x", "anchor": "min", "tolerance": 1.0e-9}, "components": ["x"]},
            {"selector": {"kind": "coordinate_plane", "axis": "x", "anchor": "max", "tolerance": 1.0e-9}, "components": ["x"]},
            {"selector": {"kind": "coordinate_plane", "axis": "y", "anchor": "min", "tolerance": 1.0e-9}, "components": ["x", "y"]},
        ],
    },
)
