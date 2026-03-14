from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent

DEFINITION = {
    "name": "3d_homo_ll",
    "dimension": 3,
    "storage": "hdf5_tet",
    "default_mesh": "LL_homo_ada_L1.h5",
    "mesh_files": sorted(path.name for path in ASSET_DIR.glob("*.h5")),
    "notes": "Homogeneous 3D limit-load benchmark family.",
}
