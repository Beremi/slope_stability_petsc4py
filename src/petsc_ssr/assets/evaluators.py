"""Generic evaluators for asset-declared hydraulic models."""

from __future__ import annotations

from typing import Any

import numpy as np


def piecewise_linear_level(coords: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    axis = str(spec.get("axis", "x")).strip().lower()
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis)
    if axis_idx is None:
        raise ValueError(f"Unsupported piecewise-linear axis {axis!r}.")
    points = np.asarray(spec.get("points"), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("piecewise_linear_level requires at least two [position, level] points.")
    positions = np.asarray(coords[axis_idx, :], dtype=np.float64)
    level = np.interp(positions, points[:, 0], points[:, 1])
    if str(spec.get("left_mode", "constant")).strip().lower() == "constant":
        level[positions < points[0, 0]] = float(points[0, 1])
    if str(spec.get("right_mode", "constant")).strip().lower() == "constant":
        level[positions > points[-1, 0]] = float(points[-1, 1])
    return level


def constant_level(coords: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    return np.full(coords.shape[1], float(spec["level"]), dtype=np.float64)


def head_values(coords: np.ndarray, spec: dict[str, Any], *, kind: str | None = None) -> np.ndarray:
    model = str(kind or spec.get("kind", "piecewise_linear_level")).strip().lower()
    if model == "constant_level":
        return constant_level(coords, spec)
    if model == "piecewise_linear_level":
        return piecewise_linear_level(coords, spec)
    raise ValueError(f"Unsupported hydraulic head model {model!r}.")


def saturation_from_hydraulic_state(
    coord: np.ndarray,
    elem: np.ndarray,
    hatp: np.ndarray,
    state: Any,
) -> np.ndarray:
    n_p = int(elem.shape[0])
    n_e = int(elem.shape[1])
    n_q = int(hatp.shape[1])
    hatphi = np.tile(np.asarray(hatp, dtype=np.float64), (1, n_e))
    coord_int = []
    for axis in range(coord.shape[0]):
        axis_values = np.reshape(coord[axis, elem.reshape(-1, order="F")], (n_p, n_e), order="F")
        coord_int.append(np.sum(np.kron(axis_values, np.ones((1, n_q), dtype=np.float64)) * hatphi, axis=0))
    integration_coords = np.vstack(coord_int)

    kind = str(state.kind).strip().lower()
    if kind in {"constant_level", "piecewise_linear_level"}:
        level = head_values(integration_coords, dict(state.value_model), kind=kind)
        return integration_coords[1, :] <= level
    raise ValueError(f"Unsupported hydraulic_state kind {state.kind!r}.")
