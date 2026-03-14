from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent

DEFINITION = {
    "name": "3d_hetero_seepage",
    "dimension": 3,
    "storage": "hdf5_tet",
    "default_mesh": "slope_with_waterlevels_concave_L2.h5",
    "mesh_files": sorted(path.name for path in ASSET_DIR.glob("*.h5")),
    "notes": "3D heterogeneous seepage mesh family with water-level boundary metadata encoded in the mesh export.",
}
