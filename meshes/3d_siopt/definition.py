from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent

DEFINITION = {
    "name": "3d_siopt",
    "dimension": 3,
    "storage": "hdf5_tet",
    "default_mesh": "SIOPT_L0.h5",
    "mesh_files": sorted(path.name for path in ASSET_DIR.glob("*.h5")),
    "notes": "SIOPT 3D benchmark family.",
}
