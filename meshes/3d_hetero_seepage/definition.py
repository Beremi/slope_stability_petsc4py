from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent

DEFINITION = {
    "name": "3d_hetero_seepage",
    "dimension": 3,
    "storage": "gmsh_tet4_physical_groups",
    "default_mesh": "slope_with_waterlevels_concave_L2.msh",
    "mesh_files": sorted(path.name for path in ASSET_DIR.glob("*.msh")),
    "notes": "3D heterogeneous seepage mesh family stored as Gmsh tet4 with triangle physical groups preserving water-level boundary labels.",
}
