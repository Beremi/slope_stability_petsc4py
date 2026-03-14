from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent

DEFINITION = {
    "name": "2d_generated_homo",
    "dimension": 2,
    "storage": "generated",
    "default_mesh": None,
    "mesh_files": [],
    "notes": "Homogeneous 2D slope meshes are generated procedurally from geometry parameters.",
}
