from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent

DEFINITION = {
    "name": "3d_siopt",
    "dimension": 3,
    "storage": "gmsh_tet4_physical_groups",
    "default_mesh": "SIOPT_L0.msh",
    "mesh_files": sorted(path.name for path in ASSET_DIR.glob("*.msh")),
    "dirichlet_labels": {
        "x": [1],
        "y": [5],
        "z": [3],
    },
    "materials": [
        {
            "name": "siopt_reference",
            "c0": 15.0,
            "phi": 20.0,
            "psi": 20.0,
            "young": 40000.0,
            "poisson": 0.30,
            "gamma_sat": 20.0,
            "gamma_unsat": 20.0,
        },
    ],
    "notes": "SIOPT 3D benchmark family stored as Gmsh tet4 with family-local BC and material metadata; boundary label 5 is the glued bottom when boundary_type=1.",
}
