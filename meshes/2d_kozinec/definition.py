from pathlib import Path

from petsc_ssr.assets.factories import build_problem_asset_2d


ASSET_DIR = Path(__file__).resolve().parent

ASSET = build_problem_asset_2d(
    asset_id="2d_kozinec",
    asset_dir=ASSET_DIR,
    default_variant="default.msh",
    mesh_variants={
        "default.msh": {"source": {"path": "default.msh"}},
    },
    materials={
        "subdomain_1": {"c0": 9.0, "phi": 26.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 20.3, "gamma_unsat": 20.7},
        "subdomain_2": {"c0": 2.0, "phi": 33.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 19.0, "gamma_unsat": 20.5},
        "subdomain_3": {"c0": 5.0, "phi": 27.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 19.4, "gamma_unsat": 21.4},
        "subdomain_4": {"c0": 3.0, "phi": 13.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 20.0, "gamma_unsat": 20.5},
        "subdomain_5": {"c0": 5.0, "phi": 27.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 19.4, "gamma_unsat": 21.4},
        "subdomain_6": {"c0": 3.0, "phi": 13.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 20.0, "gamma_unsat": 20.5},
        "subdomain_7": {"c0": 1.0, "phi": 45.0, "psi": 0.0, "young": 16000.0, "poisson": 0.4, "gamma_sat": 20.5, "gamma_unsat": 20.6},
    },
    region_assignment={
        "subdomain_1": "subdomain_1",
        "subdomain_2": "subdomain_2",
        "subdomain_3": "subdomain_3",
        "subdomain_4": "subdomain_4",
        "subdomain_5": "subdomain_5",
        "subdomain_6": "subdomain_6",
        "subdomain_7": "subdomain_7",
    },
    mechanics={
        "dirichlet": [
            {"target": "left", "components": ["x"]},
            {"target": "right", "components": ["x"]},
            {"target": "base", "components": ["x", "y"]},
        ],
        "hydraulic_state": {
            "kind": "piecewise_linear_level",
            "axis": "x",
            "points": [
                [0.0, 59.0],
                [44.0, 55.0],
                [116.0, 39.0],
                [149.0, 32.0],
                [165.0, 27.0],
                [194.0, 24.0],
                [232.0, 20.0],
            ],
            "left_mode": "constant",
            "right_mode": "constant",
        },
    },
)
