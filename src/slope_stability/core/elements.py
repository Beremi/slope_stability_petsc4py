"""Element-order helpers shared by config, mesh loading, and exports."""

from __future__ import annotations

import numpy as np


SUPPORTED_ELEM_TYPES_BY_DIM: dict[int, tuple[str, ...]] = {
    2: ("P1", "P2", "P4"),
    3: ("P1", "P2", "P4"),
}

SIMPLEX_NODES_PER_ELEMENT: dict[tuple[int, str], int] = {
    (2, "P1"): 3,
    (2, "P2"): 6,
    (2, "P4"): 15,
    (3, "P1"): 4,
    (3, "P2"): 10,
    (3, "P4"): 35,
}

SIMPLEX_NODES_PER_SURFACE: dict[tuple[int, str], int] = {
    (2, "P1"): 2,
    (2, "P2"): 3,
    (2, "P4"): 5,
    (3, "P1"): 3,
    (3, "P2"): 6,
    (3, "P4"): 15,
}


def normalize_elem_type(elem_type: str) -> str:
    return str(elem_type).strip().upper()


def validate_supported_elem_type(dim: int, elem_type: str) -> str:
    elem_key = normalize_elem_type(elem_type)
    allowed = SUPPORTED_ELEM_TYPES_BY_DIM.get(int(dim), ())
    if elem_key not in allowed:
        raise ValueError(f"Unsupported {dim}D elem_type {elem_type!r}; expected one of {allowed}.")
    return elem_key


def infer_simplex_elem_type(dim: int, n_nodes_per_elem: int) -> str:
    for (elem_dim, elem_type), count in SIMPLEX_NODES_PER_ELEMENT.items():
        if int(elem_dim) == int(dim) and int(count) == int(n_nodes_per_elem):
            return elem_type
    raise ValueError(f"Cannot infer {dim}D simplex elem_type from {n_nodes_per_elem} nodes per element.")


def simplex_vtk_cell_block(dim: int, elem: np.ndarray, elem_type: str | None = None) -> tuple[str, np.ndarray]:
    elem_arr = np.asarray(elem, dtype=np.int64)
    if elem_arr.ndim != 2:
        raise ValueError(f"Expected 2D element connectivity array, got shape {elem_arr.shape}.")
    elem_key = infer_simplex_elem_type(dim, elem_arr.shape[0]) if elem_type is None else validate_supported_elem_type(dim, elem_type)

    if int(dim) == 2:
        if elem_key == "P2":
            return "triangle6", elem_arr[:6, :].T
        return "triangle", elem_arr[:3, :].T

    if int(dim) == 3:
        if elem_key == "P2":
            return "tetra10", elem_arr[:10, :].T
        return "tetra", elem_arr[:4, :].T

    raise ValueError(f"Unsupported simplex dimension {dim}.")

