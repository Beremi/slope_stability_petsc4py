from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent

DEFINITION = {
    "name": "2d_luzec",
    "dimension": 2,
    "storage": "textmesh",
    "default_mesh": "coordinates.txt",
    "mesh_files": sorted(path.name for path in ASSET_DIR.iterdir() if path.is_file()),
    "notes": "Text-mesh family used by the Luzec 2D seepage/SSR case.",
}
