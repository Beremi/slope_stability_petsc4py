from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent

DEFINITION = {
    "name": "2d_kozinec",
    "dimension": 2,
    "storage": "textmesh",
    "default_mesh": "coordinates3.txt",
    "mesh_files": sorted(path.name for path in ASSET_DIR.iterdir() if path.is_file()),
    "notes": "Text-mesh family used by the Kozinec 2D cases; extra setup comes from the textmesh loader.",
}
