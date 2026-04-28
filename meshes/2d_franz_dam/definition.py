from pathlib import Path

from slope_stability.assets.factories import build_problem_asset_2d, build_seepage_spec


ASSET_DIR = Path(__file__).resolve().parent

ASSET = build_problem_asset_2d(
    asset_id="2d_franz_dam",
    asset_dir=ASSET_DIR,
    default_variant="default.msh",
    mesh_variants={
        "default.msh": {"source": {"path": "default.msh"}},
    },
    materials={
        "zone_1": {"c0": 50.0, "phi": 42.0, "psi": 42.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 27.0, "gamma_unsat": 27.0, "hydraulic_conductivity": 1.0e-5},
        "zone_2": {"c0": 0.5, "phi": 41.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 22.0, "gamma_unsat": 22.0, "hydraulic_conductivity": 1.0e-4},
        "zone_3": {"c0": 0.5, "phi": 40.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 23.0, "gamma_unsat": 23.0, "hydraulic_conductivity": 5.0e-5},
        "zone_4": {"c0": 0.0, "phi": 38.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 23.0, "gamma_unsat": 23.0, "hydraulic_conductivity": 5.0e-6},
        "zone_5": {"c0": 10.0, "phi": 25.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 22.0, "gamma_unsat": 22.0, "hydraulic_conductivity": 1.0e-9},
        "zone_6": {"c0": 0.5, "phi": 41.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 22.0, "gamma_unsat": 22.0, "hydraulic_conductivity": 1.0e-4},
        "zone_7": {"c0": 75.0, "phi": 42.0, "psi": 42.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 27.0, "gamma_unsat": 27.0, "hydraulic_conductivity": 2.0e-9},
        "zone_8": {"c0": 0.0, "phi": 38.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 23.0, "gamma_unsat": 23.0, "hydraulic_conductivity": 5.0e-6},
        "zone_9": {"c0": 0.5, "phi": 41.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 23.0, "gamma_unsat": 23.0, "hydraulic_conductivity": 5.0e-5},
        "zone_10": {"c0": 0.5, "phi": 41.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 21.0, "gamma_unsat": 21.0, "hydraulic_conductivity": 5.0e-4},
    },
    region_assignment={
        "zone_1": "zone_1",
        "zone_2": "zone_2",
        "zone_3": "zone_3",
        "zone_4": "zone_4",
        "zone_5": "zone_5",
        "zone_6": "zone_6",
        "zone_7": "zone_7",
        "zone_8": "zone_8",
        "zone_9": "zone_9",
        "zone_10": "zone_10",
    },
    mechanics={
        "dirichlet": [
            {"target": "left", "components": ["x"]},
            {"target": "right", "components": ["x"]},
            {"target": "base", "components": ["x", "y"]},
        ],
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
                    [-82.5, -50.0],
                    [172.5, -112.0],
                ],
                "scope": "domain_below_head",
                "left_mode": "constant",
                "right_mode": "constant",
            },
        ],
    ),
)
