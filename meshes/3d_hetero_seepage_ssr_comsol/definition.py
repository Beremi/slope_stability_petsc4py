from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent

DEFINITION = {
    "name": "3d_hetero_seepage_ssr_comsol",
    "dimension": 3,
    "storage": "hdf5_tet",
    "default_mesh": "comsol_mesh.h5",
    "mesh_files": sorted(path.name for path in ASSET_DIR.glob("*.h5")),
    "notes": "COMSOL-exported 3D seepage+SSR family.",
}
