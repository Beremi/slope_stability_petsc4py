"""Mesh helpers and material preprocessing."""

from .comsol_p2 import ComsolP2Mesh3D, load_mesh_p2_comsol, seepage_boundary_3d_hetero_comsol
from .loader import load_mesh_from_file
from .materials import heterogenous_materials, MaterialSpec
from .reorder import ReorderedMesh, compute_node_permutation, reorder_mesh_nodes
from .gmsh_waterlevels import WaterlevelsMesh3D, load_mesh_gmsh_waterlevels, seepage_boundary_3d_hetero
from .sloan2013_2d import Sloan2013Mesh2D, generate_sloan2013_mesh_2d
from .slope_2d import Slope2DMesh, generate_homogeneous_slope_mesh_2d, mesh_p1_2d, mesh_p2_2d
from .textmesh_2d import (
    TextMesh2D,
    franz_dam_pressure_boundary,
    load_mesh_franz_dam_2d,
    load_mesh_kozinec_2d,
    load_mesh_luzec_2d,
    luzec_pressure_boundary,
)

__all__ = [
    "load_mesh_from_file",
    "heterogenous_materials",
    "MaterialSpec",
    "ComsolP2Mesh3D",
    "load_mesh_p2_comsol",
    "seepage_boundary_3d_hetero_comsol",
    "ReorderedMesh",
    "compute_node_permutation",
    "reorder_mesh_nodes",
    "WaterlevelsMesh3D",
    "load_mesh_gmsh_waterlevels",
    "seepage_boundary_3d_hetero",
    "Sloan2013Mesh2D",
    "generate_sloan2013_mesh_2d",
    "Slope2DMesh",
    "generate_homogeneous_slope_mesh_2d",
    "mesh_p1_2d",
    "mesh_p2_2d",
    "TextMesh2D",
    "load_mesh_kozinec_2d",
    "load_mesh_luzec_2d",
    "load_mesh_franz_dam_2d",
    "luzec_pressure_boundary",
    "franz_dam_pressure_boundary",
]
