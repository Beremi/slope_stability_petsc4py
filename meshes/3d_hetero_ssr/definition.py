from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent

DEFINITION = {
    "name": "3d_hetero_ssr",
    "dimension": 3,
    "storage": "hdf5_tet",
    "default_mesh": "SSR_hetero_ada_L1.h5",
    "mesh_files": sorted(path.name for path in ASSET_DIR.glob("*.h5")),
    "notes": "Heterogeneous 3D SSR benchmark family.",
}
