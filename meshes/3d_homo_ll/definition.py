from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent

DEFINITION = {
    "name": "3d_homo_ll",
    "dimension": 3,
    "storage": "gmsh_tet4_physical_groups",
    "default_mesh": "LL_homo_ada_L1.msh",
    "mesh_files": sorted(path.name for path in ASSET_DIR.glob("*.msh")),
    "dirichlet_labels": {
        "x": [1, 2],
        "y": [5],
        "z": [3, 4],
    },
    "materials": [
        {
            "name": "homogeneous_slope",
            "c0": 6.0,
            "phi": 45.0,
            "psi": 0.0,
            "young": 40000.0,
            "poisson": 0.30,
            "gamma_sat": 20.0,
            "gamma_unsat": 20.0,
        },
    ],
    "notes": "Homogeneous 3D limit-load benchmark family stored as Gmsh tet4 with family-local BC and material metadata.",
}
