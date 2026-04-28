from pathlib import Path

from slope_stability.assets.factories import build_problem_asset_2d, build_seepage_spec


ASSET_DIR = Path(__file__).resolve().parent

ASSET = build_problem_asset_2d(
    asset_id="2d_luzec",
    asset_dir=ASSET_DIR,
    default_variant="default.msh",
    mesh_variants={
        "default.msh": {"source": {"path": "default.msh"}},
    },
    materials={
        "S1": {"c0": 14.0, "phi": 21.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 21.0, "gamma_unsat": 21.0, "hydraulic_conductivity": 0.000864},
        "S2": {"c0": 1.0, "phi": 33.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 21.0, "gamma_unsat": 19.0, "hydraulic_conductivity": 86.4},
        "S3": {"c0": 7.5, "phi": 30.25, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 22.0, "gamma_unsat": 21.0, "hydraulic_conductivity": 0.000864},
        "S4": {"c0": 1.6, "phi": 24.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 21.0, "gamma_unsat": 19.0, "hydraulic_conductivity": 0.86},
        "S5": {"c0": 2.0, "phi": 37.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 21.0, "gamma_unsat": 21.0, "hydraulic_conductivity": 86.4},
        "S6": {"c0": 1.6, "phi": 24.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 21.0, "gamma_unsat": 19.0, "hydraulic_conductivity": 0.86},
        "S7": {"c0": 50.0, "phi": 45.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 19.0, "gamma_unsat": 19.0, "hydraulic_conductivity": 0.000864},
        "S8": {"c0": 1.6, "phi": 24.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 21.0, "gamma_unsat": 19.0, "hydraulic_conductivity": 0.86},
    },
    region_assignment={
        "S1": "S1",
        "S2": "S2",
        "S3": "S3",
        "S4": "S4",
        "S5": "S5",
        "S6": "S6",
        "S7": "S7",
        "S8": "S8",
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
                    [91.12, 15.75],
                    [101.845, 22.40],
                ],
                "scope": "domain_below_head",
                "left_mode": "constant",
                "right_mode": "constant",
            },
        ],
    ),
)
