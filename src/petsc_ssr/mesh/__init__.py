"""Generic mesh helpers and material preprocessing."""

from .loader import load_mesh_from_file
from .materials import MaterialSpec, heterogenous_materials


def __getattr__(name: str):
    if name in {
        "ReorderedMesh",
        "canonical_node_ordering_strategy",
        "compute_node_permutation",
        "node_ordering_requires_partitions",
        "reorder_mesh_nodes",
    }:
        from . import reorder

        value = getattr(reorder, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MaterialSpec",
    "ReorderedMesh",
    "canonical_node_ordering_strategy",
    "compute_node_permutation",
    "heterogenous_materials",
    "load_mesh_from_file",
    "node_ordering_requires_partitions",
    "reorder_mesh_nodes",
]
