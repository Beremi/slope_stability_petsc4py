from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent

DEFINITION = {
    "name": "2d_sloan2013",
    "dimension": 2,
    "storage": "generated",
    "default_mesh": None,
    "mesh_files": [],
    "notes": "Sloan2013 2D seepage meshes are generated procedurally from benchmark geometry parameters.",
}
