import math
from pathlib import Path

from slope_stability.assets.factories import build_problem_asset_2d, build_seepage_spec


ASSET_DIR = Path(__file__).resolve().parent

X1 = 15.0
Y11 = 6.75
Y12 = 0.5
Y13 = 0.75
Y21 = 1.0
Y22 = 9.25
Y23 = 2.0
Y1 = Y11 + Y12 + Y13
Y2 = Y21 + Y22 + Y23
X2 = Y2 / math.tan(math.radians(26.6))
X_BAR = X1 + (1.0 - (Y21 / Y2)) * X2

ASSET = build_problem_asset_2d(
    asset_id="2d_sloan2013",
    asset_dir=ASSET_DIR,
    default_variant="default.msh",
    mesh_variants={
        "default.msh": {"source": {"path": "default.msh"}},
    },
    materials={
        "slope_mass": {"hydraulic_conductivity": 1.0},
        "weak_layer": {"hydraulic_conductivity": 1.0},
    },
    region_assignment={
        "slope_mass": "slope_mass",
        "weak_layer": "weak_layer",
    },
    seepage=build_seepage_spec(
        water_unit_weight=9.81,
        conductivity_mode="by_material",
        head_bcs=[
            {
                "target": "head_support",
                "kind": "piecewise_linear_level",
                "axis": "x",
                "points": [
                    [0.0, Y1 + 1.0 + 9.25],
                    [X_BAR, Y1 + 1.0],
                ],
                "scope": "domain_below_head",
                "left_mode": "constant",
                "right_mode": "constant",
            },
        ],
    ),
)
