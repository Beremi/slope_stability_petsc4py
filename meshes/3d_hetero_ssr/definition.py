from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent

DEFINITION = {
    "name": "3d_hetero_ssr",
    "dimension": 3,
    "storage": "gmsh_tet4_physical_groups",
    "default_mesh": "SSR_hetero_ada_L1.msh",
    "mesh_files": sorted(path.name for path in ASSET_DIR.glob("*.msh")),
    "dirichlet_labels": {
        "x": [1, 2],
        "y": [5],
        "z": [3, 4],
    },
    "materials": [
        {
            "id": 0,
            "name": "cover_layer",
            "c0": 15.0,
            "phi": 30.0,
            "psi": 0.0,
            "young": 10000.0,
            "poisson": 0.33,
            "gamma_sat": 19.0,
            "gamma_unsat": 19.0,
        },
        {
            "id": 1,
            "name": "general_foundation",
            "c0": 15.0,
            "phi": 38.0,
            "psi": 0.0,
            "young": 50000.0,
            "poisson": 0.30,
            "gamma_sat": 22.0,
            "gamma_unsat": 22.0,
        },
        {
            "id": 2,
            "name": "weak_foundation",
            "c0": 10.0,
            "phi": 35.0,
            "psi": 0.0,
            "young": 50000.0,
            "poisson": 0.30,
            "gamma_sat": 21.0,
            "gamma_unsat": 21.0,
        },
        {
            "id": 3,
            "name": "slope_mass",
            "c0": 18.0,
            "phi": 32.0,
            "psi": 0.0,
            "young": 20000.0,
            "poisson": 0.33,
            "gamma_sat": 20.0,
            "gamma_unsat": 20.0,
        },
    ],
    "notes": "Heterogeneous 3D SSR benchmark family stored as Gmsh tet4 with family-local BC and material metadata.",
}
