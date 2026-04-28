"""Generic selector, mechanics, and seepage helpers for executable assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..api import MeshBuildResult, MechanicalProblemSpec, SeepageProblemSpec


MECHANICAL_MATERIAL_FIELDS: tuple[str, ...] = (
    "c0",
    "phi",
    "psi",
    "young",
    "poisson",
    "gamma_sat",
    "gamma_unsat",
)

AXES_BY_DIM: dict[int, tuple[str, ...]] = {
    2: ("x", "y"),
    3: ("x", "y", "z"),
}


@dataclass(frozen=True)
class DirichletRule:
    components: tuple[str, ...]
    labels: tuple[int, ...] | None = None
    selector: dict[str, Any] | None = None
    boundary_types: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SeepageDefinition:
    water_unit_weight: float
    conductivity_mode: str
    conductivity: tuple[float, ...] | None
    water_levels: dict[str, float] | None
    hydraulic_boundaries: dict[str, Any]


def ordered_material_entries(materials: list[dict[str, Any]], *, asset_dir: Path) -> list[dict[str, Any]]:
    ordered: dict[int, dict[str, Any]] = {}
    for idx, item in enumerate(materials):
        if not isinstance(item, dict):
            raise ValueError(f"materials[{idx}] in {asset_dir} must be a dictionary.")
        mid = int(item.get("id", idx))
        ordered[mid] = dict(item)
    if not ordered:
        return []
    expected = list(range(max(ordered) + 1))
    if sorted(ordered) != expected:
        raise ValueError(f"materials in {asset_dir} must provide contiguous ids {expected}, got {sorted(ordered)}.")
    return [ordered[idx] for idx in expected]


def material_rows_from_entries(materials: list[dict[str, Any]], *, asset_dir: Path) -> list[list[float]] | None:
    ordered = ordered_material_entries(materials, asset_dir=asset_dir)
    if not ordered:
        return None
    any_mechanical = any(any(field in item for field in MECHANICAL_MATERIAL_FIELDS) for item in ordered)
    if not any_mechanical:
        return None
    missing = [
        idx
        for idx, item in enumerate(ordered)
        if any(field not in item for field in MECHANICAL_MATERIAL_FIELDS)
    ]
    if missing:
        raise ValueError(
            f"materials in {asset_dir} mix mechanical and non-mechanical entries; missing full rows for ids {missing}."
        )
    return [[float(item[field]) for field in MECHANICAL_MATERIAL_FIELDS] for item in ordered]


def hydraulic_conductivity_from_entries(
    materials: list[dict[str, Any]],
    seepage: SeepageDefinition | None,
    *,
    asset_dir: Path,
) -> np.ndarray | None:
    if seepage is None:
        return None
    if seepage.conductivity_mode == "uniform":
        return np.asarray(seepage.conductivity or (), dtype=np.float64)
    ordered = ordered_material_entries(materials, asset_dir=asset_dir)
    if not ordered:
        raise ValueError(f"Seepage conductivity mode 'by_material' in {asset_dir} requires materials[].")
    missing = [idx for idx, item in enumerate(ordered) if item.get("hydraulic_conductivity") is None]
    if missing:
        raise ValueError(
            f"materials[{missing[0]}].hydraulic_conductivity is required for seepage conductivity mode 'by_material' in {asset_dir}."
        )
    return np.asarray([float(item["hydraulic_conductivity"]) for item in ordered], dtype=np.float64)


def normalize_labels(value: Any, *, field_name: str, asset_dir: Path) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} in {asset_dir} must be a list of integers.")
    return tuple(dict.fromkeys(int(v) for v in value))


def normalize_components(value: Any, *, dim: int, field_name: str, asset_dir: Path) -> tuple[str, ...]:
    allowed = set(AXES_BY_DIM[int(dim)])
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field_name} in {asset_dir} must be a non-empty list of components.")
    components = tuple(str(v).strip().lower() for v in value)
    invalid = sorted(set(components) - allowed)
    if invalid:
        raise ValueError(
            f"{field_name} in {asset_dir} contains invalid components {invalid}; expected subset of {sorted(allowed)}."
        )
    return tuple(dict.fromkeys(components))


def normalize_boundary_types(value: Any, *, field_name: str, asset_dir: Path) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return (int(value),)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} in {asset_dir} must be an int or list of ints.")
    return tuple(dict.fromkeys(int(v) for v in value))


def normalize_float_sequence(value: Any, *, field_name: str, asset_dir: Path) -> tuple[float, ...]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (float(value),)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} in {asset_dir} must be a float or list of floats.")
    return tuple(float(v) for v in value)


def normalize_float_mapping(value: Any, *, field_name: str, asset_dir: Path) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field_name} in {asset_dir} must be a non-empty dictionary of numeric values.")
    return {str(key): float(item) for key, item in value.items()}


def normalize_point3(value: Any, *, field_name: str, asset_dir: Path) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} in {asset_dir} must be a 3-vector.")
    return (float(value[0]), float(value[1]), float(value[2]))


def normalize_point(value: Any, *, field_name: str, asset_dir: Path) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field_name} in {asset_dir} must be a non-empty coordinate vector.")
    return tuple(float(v) for v in value)


def normalize_dirichlet_rules(
    raw_rules: list[dict[str, Any]] | None,
    *,
    dim: int,
    asset_dir: Path,
) -> tuple[DirichletRule, ...]:
    if raw_rules is None:
        return ()
    if not isinstance(raw_rules, list):
        raise ValueError(f"mechanics.dirichlet in {asset_dir} must be a list of rule dictionaries.")
    rules: list[DirichletRule] = []
    for idx, item in enumerate(raw_rules):
        if not isinstance(item, dict):
            raise ValueError(f"mechanics.dirichlet[{idx}] in {asset_dir} must be a dictionary.")
        components = normalize_components(
            item.get("components"),
            dim=dim,
            field_name=f"mechanics.dirichlet[{idx}].components",
            asset_dir=asset_dir,
        )
        labels = None
        selector = None
        if item.get("labels") is not None:
            labels = normalize_labels(
                item.get("labels"),
                field_name=f"mechanics.dirichlet[{idx}].labels",
                asset_dir=asset_dir,
            )
        if item.get("selector") is not None:
            if not isinstance(item.get("selector"), dict):
                raise ValueError(f"mechanics.dirichlet[{idx}].selector in {asset_dir} must be a dictionary.")
            selector = dict(item["selector"])
        if labels is None and selector is None:
            raise ValueError(f"mechanics.dirichlet[{idx}] in {asset_dir} must define either labels or selector.")
        rules.append(
            DirichletRule(
                components=components,
                labels=labels,
                selector=selector,
                boundary_types=normalize_boundary_types(
                    item.get("boundary_type"),
                    field_name=f"mechanics.dirichlet[{idx}].boundary_type",
                    asset_dir=asset_dir,
                ),
            )
        )
    return tuple(rules)


def normalize_seepage_definition(raw: dict[str, Any] | None, *, materials: list[dict[str, Any]], asset_dir: Path) -> SeepageDefinition | None:
    if not raw:
        return None
    if "water_unit_weight" not in raw:
        raise ValueError(f"seepage.water_unit_weight is required in {asset_dir}.")
    if "conductivity_mode" not in raw:
        raise ValueError(f"seepage.conductivity_mode is required in {asset_dir}.")
    if "hydraulic_boundaries" not in raw:
        raise ValueError(f"seepage.hydraulic_boundaries is required in {asset_dir}.")

    conductivity_mode = str(raw["conductivity_mode"]).strip().lower()
    conductivity: tuple[float, ...] | None
    if conductivity_mode == "uniform":
        if raw.get("conductivity") is None:
            raise ValueError(f"Uniform seepage conductivity is required in {asset_dir}.")
        conductivity = normalize_float_sequence(raw["conductivity"], field_name="seepage.conductivity", asset_dir=asset_dir)
    elif conductivity_mode == "by_material":
        ordered = ordered_material_entries(materials, asset_dir=asset_dir)
        if not ordered:
            raise ValueError(f"Seepage conductivity mode 'by_material' in {asset_dir} requires materials[].")
        missing = [idx for idx, item in enumerate(ordered) if item.get("hydraulic_conductivity") is None]
        if missing:
            raise ValueError(
                f"materials[{missing[0]}].hydraulic_conductivity is required for seepage conductivity mode 'by_material' in {asset_dir}."
            )
        conductivity = None
    else:
        raise ValueError(
            f"Unsupported seepage.conductivity_mode {conductivity_mode!r} in {asset_dir}; expected 'uniform' or 'by_material'."
        )

    hydraulic_boundaries = raw["hydraulic_boundaries"]
    if not isinstance(hydraulic_boundaries, dict):
        raise ValueError(f"seepage.hydraulic_boundaries in {asset_dir} must be a dictionary.")
    mode = str(hydraulic_boundaries.get("mode", "")).strip().lower()
    water_levels: dict[str, float] | None = None
    if mode in {"label_sets", "hybrid_transition"}:
        water_levels = normalize_float_mapping(raw.get("water_levels"), field_name="seepage.water_levels", asset_dir=asset_dir)
        for key in ("free", "porous"):
            if key not in water_levels:
                raise ValueError(f"seepage.water_levels.{key} is required in {asset_dir}.")
        for field_name in ("dry_labels", "porous_labels", "free_labels"):
            normalize_labels(hydraulic_boundaries.get(field_name), field_name=f"seepage.hydraulic_boundaries.{field_name}", asset_dir=asset_dir)
        if mode == "hybrid_transition":
            recipe = hydraulic_boundaries.get("geometry_recipe")
            if not isinstance(recipe, dict):
                raise ValueError(f"seepage.hydraulic_boundaries.geometry_recipe in {asset_dir} must be a dictionary.")
            for field_name in ("base_point", "toe_point", "apex_left", "apex_right"):
                normalize_point3(recipe.get(field_name), field_name=f"seepage.hydraulic_boundaries.geometry_recipe.{field_name}", asset_dir=asset_dir)
            for field_name in ("bed_y", "triangle_normal_tolerance", "plane_distance_tolerance", "bed_tolerance", "sector_tolerance"):
                if recipe.get(field_name) is None:
                    raise ValueError(f"seepage.hydraulic_boundaries.geometry_recipe.{field_name} is required in {asset_dir}.")
                float(recipe[field_name])
    elif mode == "selector_polyline_head_2d":
        if not isinstance(hydraulic_boundaries.get("dirichlet_selector"), dict):
            raise ValueError(f"seepage.hydraulic_boundaries.dirichlet_selector in {asset_dir} must be a selector dictionary.")
        profile = hydraulic_boundaries.get("head_profile")
        if not isinstance(profile, dict):
            raise ValueError(f"seepage.hydraulic_boundaries.head_profile in {asset_dir} must be a dictionary.")
        _profile_levels_from_points(np.asarray([0.0, 1.0], dtype=np.float64), profile, asset_dir=asset_dir, field_name="seepage.hydraulic_boundaries.head_profile")
        extra = hydraulic_boundaries.get("extra_dirichlet_selectors")
        if extra is not None:
            if not isinstance(extra, list):
                raise ValueError(f"seepage.hydraulic_boundaries.extra_dirichlet_selectors in {asset_dir} must be a list.")
            for idx, item in enumerate(extra):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"seepage.hydraulic_boundaries.extra_dirichlet_selectors[{idx}] in {asset_dir} must be a dictionary."
                    )
    else:
        raise ValueError(
            f"Unsupported seepage.hydraulic_boundaries.mode {mode!r} in {asset_dir}; "
            "expected 'label_sets', 'hybrid_transition', or 'selector_polyline_head_2d'."
        )

    return SeepageDefinition(
        water_unit_weight=float(raw["water_unit_weight"]),
        conductivity_mode=conductivity_mode,
        conductivity=conductivity,
        water_levels=water_levels,
        hydraulic_boundaries=dict(hydraulic_boundaries),
    )


def _axis_index(dim: int, axis: str, *, asset_dir: Path) -> int:
    axes = AXES_BY_DIM[int(dim)]
    try:
        return axes.index(str(axis).strip().lower())
    except ValueError as exc:
        raise ValueError(f"Unknown axis {axis!r} for dimension {dim} in {asset_dir}.") from exc


def _compare(values: np.ndarray, operator: str, reference: float) -> np.ndarray:
    op = str(operator).strip().lower()
    if op in {"<", "lt"}:
        return values < reference
    if op in {"<=", "le"}:
        return values <= reference
    if op in {">", "gt"}:
        return values > reference
    if op in {">=", "ge"}:
        return values >= reference
    if op in {"==", "eq"}:
        return values == reference
    raise ValueError(f"Unsupported selector operator {operator!r}.")


def _resolve_reference(values: np.ndarray, selector: dict[str, Any], *, asset_dir: Path, field_name: str) -> float:
    if selector.get("value") is not None:
        return float(selector["value"])
    anchor = selector.get("anchor")
    offset = float(selector.get("offset", 0.0))
    if anchor is None:
        raise ValueError(f"{field_name} in {asset_dir} must define either value or anchor.")
    anchor_text = str(anchor).strip().lower()
    if anchor_text == "min":
        return float(np.min(values)) + offset
    if anchor_text == "max":
        return float(np.max(values)) + offset
    try:
        return float(anchor) + offset
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}.anchor in {asset_dir} must be 'min', 'max', or numeric.") from exc


def _profile_levels_from_points(
    x_values: np.ndarray,
    profile: dict[str, Any],
    *,
    asset_dir: Path,
    field_name: str,
) -> np.ndarray:
    points = profile.get("points")
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        raise ValueError(f"{field_name}.points in {asset_dir} must be a list of at least two [x, y] points.")
    pts = np.asarray([normalize_point(item, field_name=f"{field_name}.points", asset_dir=asset_dir) for item in points], dtype=np.float64)
    if pts.shape[1] != 2:
        raise ValueError(f"{field_name}.points in {asset_dir} must contain 2D points.")
    order = np.argsort(pts[:, 0], kind="mergesort")
    pts = pts[order]
    x_nodes = pts[:, 0]
    y_nodes = pts[:, 1]
    left_mode = str(profile.get("left_mode", "constant")).strip().lower()
    right_mode = str(profile.get("right_mode", "constant")).strip().lower()
    levels = np.interp(x_values, x_nodes, y_nodes, left=y_nodes[0], right=y_nodes[-1])

    if left_mode == "extend":
        slope = (y_nodes[1] - y_nodes[0]) / (x_nodes[1] - x_nodes[0])
        mask = x_values < x_nodes[0]
        levels[mask] = y_nodes[0] + slope * (x_values[mask] - x_nodes[0])
    elif left_mode != "constant":
        raise ValueError(f"{field_name}.left_mode in {asset_dir} must be 'constant' or 'extend'.")

    if right_mode == "extend":
        slope = (y_nodes[-1] - y_nodes[-2]) / (x_nodes[-1] - x_nodes[-2])
        mask = x_values > x_nodes[-1]
        levels[mask] = y_nodes[-1] + slope * (x_values[mask] - x_nodes[-1])
    elif right_mode != "constant":
        raise ValueError(f"{field_name}.right_mode in {asset_dir} must be 'constant' or 'extend'.")

    return np.asarray(levels, dtype=np.float64)


def evaluate_node_selector(
    coord: np.ndarray,
    surf: np.ndarray,
    boundary_labels: np.ndarray,
    selector: dict[str, Any],
    *,
    asset_dir: Path,
) -> np.ndarray:
    selector_kind = str(selector.get("kind", "")).strip().lower()
    dim = int(coord.shape[0])
    n_nodes = int(coord.shape[1])

    if selector_kind in {"union", "or"}:
        selectors = selector.get("selectors")
        if not isinstance(selectors, list) or not selectors:
            raise ValueError(f"selector.selectors for kind {selector_kind!r} in {asset_dir} must be a non-empty list.")
        out = np.zeros(n_nodes, dtype=bool)
        for item in selectors:
            if not isinstance(item, dict):
                raise ValueError(f"selector.selectors entries in {asset_dir} must be dictionaries.")
            out |= evaluate_node_selector(coord, surf, boundary_labels, item, asset_dir=asset_dir)
        return out

    if selector_kind in {"intersection", "and"}:
        selectors = selector.get("selectors")
        if not isinstance(selectors, list) or not selectors:
            raise ValueError(f"selector.selectors for kind {selector_kind!r} in {asset_dir} must be a non-empty list.")
        out = np.ones(n_nodes, dtype=bool)
        for item in selectors:
            if not isinstance(item, dict):
                raise ValueError(f"selector.selectors entries in {asset_dir} must be dictionaries.")
            out &= evaluate_node_selector(coord, surf, boundary_labels, item, asset_dir=asset_dir)
        return out

    if selector_kind == "difference":
        base = selector.get("base")
        subtract = selector.get("subtract")
        if not isinstance(base, dict) or not isinstance(subtract, dict):
            raise ValueError(f"selector kind 'difference' in {asset_dir} must define base and subtract selectors.")
        return evaluate_node_selector(coord, surf, boundary_labels, base, asset_dir=asset_dir) & ~evaluate_node_selector(
            coord,
            surf,
            boundary_labels,
            subtract,
            asset_dir=asset_dir,
        )

    if selector_kind == "boundary_labels":
        labels = normalize_labels(selector.get("labels"), field_name="selector.labels", asset_dir=asset_dir)
        out = np.zeros(n_nodes, dtype=bool)
        if surf.size == 0 or boundary_labels.size == 0:
            return out
        mask = np.isin(np.asarray(boundary_labels, dtype=np.int64), np.asarray(labels, dtype=np.int64))
        if np.any(mask):
            out[np.unique(np.asarray(surf[:, mask], dtype=np.int64).ravel())] = True
        return out

    if selector_kind == "all_surface_nodes":
        out = np.zeros(n_nodes, dtype=bool)
        if surf.size:
            out[np.unique(np.asarray(surf, dtype=np.int64).ravel())] = True
        return out

    if selector_kind == "coordinate_plane":
        axis_idx = _axis_index(dim, str(selector.get("axis")), asset_dir=asset_dir)
        values = np.asarray(coord[axis_idx, :], dtype=np.float64)
        reference = _resolve_reference(values, selector, asset_dir=asset_dir, field_name="selector")
        tolerance = float(selector.get("tolerance", 1.0e-9))
        return np.abs(values - reference) <= tolerance

    if selector_kind == "coordinate_threshold":
        axis_idx = _axis_index(dim, str(selector.get("axis")), asset_dir=asset_dir)
        values = np.asarray(coord[axis_idx, :], dtype=np.float64)
        reference = _resolve_reference(values, selector, asset_dir=asset_dir, field_name="selector")
        return _compare(values, str(selector.get("operator", "<=")), reference)

    if selector_kind == "point_proximity":
        point = np.asarray(normalize_point(selector.get("point"), field_name="selector.point", asset_dir=asset_dir), dtype=np.float64)
        if point.size != dim:
            raise ValueError(f"selector.point in {asset_dir} must have dimension {dim}, got {point.size}.")
        tolerance = selector.get("tolerance", 1.0e-9)
        if isinstance(tolerance, (int, float, np.integer, np.floating)):
            tol = np.full(dim, float(tolerance), dtype=np.float64)
        elif isinstance(tolerance, (list, tuple)):
            tol = np.asarray([float(item) for item in tolerance], dtype=np.float64)
            if tol.size != dim:
                raise ValueError(f"selector.tolerance in {asset_dir} must have dimension {dim}, got {tol.size}.")
        else:
            raise ValueError(f"selector.tolerance in {asset_dir} must be numeric or a vector.")
        return np.all(np.abs(coord - point[:, None]) <= tol[:, None], axis=0)

    if selector_kind == "line_halfspace":
        if dim != 2:
            raise ValueError(f"selector kind 'line_halfspace' in {asset_dir} is only supported in 2D.")
        p1 = np.asarray(normalize_point(selector.get("point1"), field_name="selector.point1", asset_dir=asset_dir), dtype=np.float64)
        p2 = np.asarray(normalize_point(selector.get("point2"), field_name="selector.point2", asset_dir=asset_dir), dtype=np.float64)
        if p1.size != 2 or p2.size != 2:
            raise ValueError(f"selector kind 'line_halfspace' in {asset_dir} expects 2D points.")
        tolerance = float(selector.get("tolerance", 0.0))
        side = str(selector.get("side", "below")).strip().lower()
        x = np.asarray(coord[0, :], dtype=np.float64)
        y = np.asarray(coord[1, :], dtype=np.float64)
        if abs(float(p2[0] - p1[0])) <= 1.0e-12:
            reference = float(p1[0])
            if side in {"left", "below"}:
                return x <= reference + tolerance
            if side in {"right", "above"}:
                return x >= reference - tolerance
            raise ValueError(f"selector.side {side!r} in {asset_dir} is not supported for vertical lines.")
        slope = float((p2[1] - p1[1]) / (p2[0] - p1[0]))
        y_line = slope * (x - p1[0]) + p1[1]
        if side in {"below", "le"}:
            return y <= y_line + tolerance
        if side in {"above", "ge"}:
            return y >= y_line - tolerance
        raise ValueError(f"selector.side {side!r} in {asset_dir} must be 'below' or 'above'.")

    if selector_kind == "polyline_below":
        if dim != 2:
            raise ValueError(f"selector kind 'polyline_below' in {asset_dir} is only supported in 2D.")
        levels = _profile_levels_from_points(np.asarray(coord[0, :], dtype=np.float64), selector, asset_dir=asset_dir, field_name="selector")
        tolerance = float(selector.get("tolerance", 0.0))
        return np.asarray(coord[1, :], dtype=np.float64) <= levels + tolerance

    if selector_kind == "surface_midpoint_threshold":
        out = np.zeros(n_nodes, dtype=bool)
        if surf.size == 0:
            return out
        axis_idx = _axis_index(dim, str(selector.get("axis")), asset_dir=asset_dir)
        reference = _resolve_reference(np.asarray(coord[axis_idx, :], dtype=np.float64), selector, asset_dir=asset_dir, field_name="selector")
        corner_count = int(selector.get("corner_count", 2 if dim == 2 else 3))
        if int(surf.shape[0]) < corner_count:
            raise ValueError(f"selector.corner_count={corner_count} exceeds surface node count {surf.shape[0]} in {asset_dir}.")
        face_mid = np.mean(np.asarray(coord[axis_idx, surf[:corner_count, :]], dtype=np.float64), axis=0)
        face_mask = _compare(face_mid, str(selector.get("operator", ">=")), reference)
        if np.any(face_mask):
            out[np.unique(np.asarray(surf[:, face_mask], dtype=np.int64).ravel())] = True
        return out

    raise ValueError(f"Unsupported selector kind {selector_kind!r} in {asset_dir}.")


def build_dirichlet_mask(
    *,
    dim: int,
    n_nodes: int,
    surf: np.ndarray,
    boundary: np.ndarray,
    coord: np.ndarray,
    rules: tuple[DirichletRule, ...],
    boundary_type: int,
    asset_dir: Path,
) -> np.ndarray:
    q = np.ones((int(dim), int(n_nodes)), dtype=bool)
    face = np.asarray(surf, dtype=np.int64)
    labels = np.asarray(boundary, dtype=np.int64).ravel()
    component_to_idx = {name: idx for idx, name in enumerate(AXES_BY_DIM[int(dim)])}

    for rule in rules:
        if rule.boundary_types is not None and int(boundary_type) not in rule.boundary_types:
            continue
        node_mask = np.zeros(int(n_nodes), dtype=bool)
        if rule.labels:
            mask = np.isin(labels, np.asarray(rule.labels, dtype=np.int64))
            if np.any(mask) and face.size:
                node_mask[np.unique(face[:, mask].ravel())] = True
        if rule.selector is not None:
            node_mask |= evaluate_node_selector(
                np.asarray(coord, dtype=np.float64),
                face,
                labels,
                dict(rule.selector),
                asset_dir=asset_dir,
            )
        if not np.any(node_mask):
            continue
        nodes = np.flatnonzero(node_mask)
        for component in rule.components:
            q[component_to_idx[component], nodes] = False
    return q


def build_mechanical_problem(
    *,
    materials: list[dict[str, Any]],
    rules: tuple[DirichletRule, ...],
    mesh: MeshBuildResult,
    dim: int,
    boundary_type: int,
    asset_dir: Path,
) -> MechanicalProblemSpec | None:
    material_rows = material_rows_from_entries(materials, asset_dir=asset_dir)
    if material_rows is None:
        return None
    q_mask = build_dirichlet_mask(
        dim=dim,
        n_nodes=int(mesh.coord.shape[1]),
        surf=mesh.surf,
        boundary=mesh.boundary_labels,
        coord=mesh.coord,
        rules=rules,
        boundary_type=boundary_type,
        asset_dir=asset_dir,
    )
    return MechanicalProblemSpec(materials=material_rows, q_mask=q_mask, boundary_type=int(boundary_type))


def build_seepage_problem(
    *,
    materials: list[dict[str, Any]],
    seepage: SeepageDefinition | None,
    mesh: MeshBuildResult,
    asset_dir: Path,
    grho: float | None = None,
) -> SeepageProblemSpec | None:
    if seepage is None:
        return None
    conductivity = hydraulic_conductivity_from_entries(materials, seepage, asset_dir=asset_dir)
    assert conductivity is not None
    q_w, pw_d = build_seepage_boundary(
        coord=mesh.coord,
        surf=mesh.surf,
        triangle_labels=mesh.boundary_labels,
        seepage=seepage,
        asset_dir=asset_dir,
        grho=grho,
    )
    return SeepageProblemSpec(
        water_unit_weight=float(seepage.water_unit_weight if grho is None else grho),
        conductivity=np.asarray(conductivity, dtype=np.float64),
        q_w=np.asarray(q_w, dtype=bool),
        pw_d=np.asarray(pw_d, dtype=np.float64),
    )


def build_seepage_boundary(
    *,
    coord: np.ndarray,
    surf: np.ndarray,
    triangle_labels: np.ndarray,
    seepage: SeepageDefinition,
    asset_dir: Path,
    grho: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    coord = np.asarray(coord, dtype=np.float64)
    surf = np.asarray(surf, dtype=np.int64)
    triangle_labels = np.asarray(triangle_labels if triangle_labels is not None else np.empty(0, dtype=np.int64), dtype=np.int64).ravel()

    n_nodes = int(coord.shape[1])
    q_w = np.ones(n_nodes, dtype=bool)
    pw_d = np.zeros(n_nodes, dtype=np.float64)
    hydraulic = seepage.hydraulic_boundaries
    mode = str(hydraulic["mode"]).strip().lower()
    rho_g = float(seepage.water_unit_weight if grho is None else grho)

    if mode in {"label_sets", "hybrid_transition"}:
        assert seepage.water_levels is not None
        y = coord[1, :]
        dry_labels = normalize_labels(hydraulic.get("dry_labels"), field_name="seepage.hydraulic_boundaries.dry_labels", asset_dir=asset_dir)
        porous_labels = normalize_labels(hydraulic.get("porous_labels"), field_name="seepage.hydraulic_boundaries.porous_labels", asset_dir=asset_dir)
        free_labels = normalize_labels(hydraulic.get("free_labels"), field_name="seepage.hydraulic_boundaries.free_labels", asset_dir=asset_dir)

        q_dry = np.zeros(n_nodes, dtype=bool)
        for label in dry_labels:
            tmp = surf[:, triangle_labels == label]
            q_dry[tmp.ravel()] = True

        porous_level = float(seepage.water_levels["porous"])
        q_wet1 = np.zeros(n_nodes, dtype=bool)
        if porous_labels:
            tmp = surf[:, np.isin(triangle_labels, np.asarray(porous_labels, dtype=np.int64))]
            if mode == "label_sets":
                q_wet1[tmp.ravel()] = True
            else:
                nodes_tmp = np.unique(tmp.ravel())
                selected_nodes = nodes_tmp[y[nodes_tmp] < porous_level]
                q_wet1[selected_nodes] = True
                dry_nodes = nodes_tmp[y[nodes_tmp] >= porous_level]
                q_dry[dry_nodes] = True
            pw_d[q_wet1] = rho_g * (porous_level - coord[1, q_wet1])

        free_level = float(seepage.water_levels["free"])
        q_wet2 = np.zeros(n_nodes, dtype=bool)
        if free_labels:
            tmp = surf[:, np.isin(triangle_labels, np.asarray(free_labels, dtype=np.int64))]
            q_wet2[tmp.ravel()] = True

        if mode == "label_sets":
            if np.any(q_wet2):
                pw_d[q_wet2] = rho_g * (free_level - coord[1, q_wet2])
        else:
            recipe = hydraulic["geometry_recipe"]
            triangles = surf[:3, :]
            v1 = coord[:, triangles[0, :]]
            v2 = coord[:, triangles[1, :]]
            v3 = coord[:, triangles[2, :]]
            e1 = v2 - v1
            e2 = v3 - v1
            normals = np.cross(e1.T, e2.T).T
            normal_tol = float(recipe["triangle_normal_tolerance"])
            condition = np.all(np.abs(normals) > normal_tol, axis=0)
            selected_triangles = surf[:, condition]
            nodes_tmp = np.unique(selected_triangles.ravel())

            c = np.asarray(normalize_point3(recipe["base_point"], field_name="base_point", asset_dir=asset_dir), dtype=np.float64)
            t = np.asarray(normalize_point3(recipe["toe_point"], field_name="toe_point", asset_dir=asset_dir), dtype=np.float64)
            a_left = np.asarray(normalize_point3(recipe["apex_left"], field_name="apex_left", asset_dir=asset_dir), dtype=np.float64)
            a_right = np.asarray(normalize_point3(recipe["apex_right"], field_name="apex_right", asset_dir=asset_dir), dtype=np.float64)
            plane_tol = float(recipe["plane_distance_tolerance"])
            normal_left = np.cross(t - c, a_left - c)
            normal_right = np.cross(t - c, a_right - c)
            X = coord[:, nodes_tmp].T
            V = X - c
            d_left = np.abs(V @ normal_left)
            d_right = np.abs(V @ normal_right)
            nodes_tmp = nodes_tmp[(d_left < plane_tol) | (d_right < plane_tol)]

            selected_nodes = nodes_tmp[y[nodes_tmp] < free_level]
            q_wet2[selected_nodes] = True
            selected_nodes_dry = nodes_tmp[y[nodes_tmp] >= free_level]
            q_dry[selected_nodes_dry] = True

            boundary_nodes = np.unique(surf.ravel())
            bed_tol = float(recipe["bed_tolerance"])
            bed_y = float(recipe["bed_y"])
            selected = boundary_nodes[np.abs(y[boundary_nodes] - bed_y) < bed_tol]
            n = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            X = coord[:, selected].T
            dL = a_left - c
            dR = a_right - c
            VL = X - c
            kL = np.cross(n, dL)
            kR = np.cross(n, dR)
            kL = kL / np.linalg.norm(kL)
            kR = kR / np.linalg.norm(kR)
            sL = (VL @ kL) / np.linalg.norm(VL @ kL)
            sR = (VL @ kR) / np.linalg.norm(VL @ kR)
            sector_tol = float(recipe["sector_tolerance"])
            selected = selected[(sL < sector_tol) & (sR > sector_tol)]
            q_wet2[selected] = True
            if np.any(q_wet2):
                pw_d[q_wet2] = rho_g * (free_level - coord[1, q_wet2])

        q_w[q_dry] = False
        q_w[q_wet1] = False
        q_w[q_wet2] = False
        return q_w, pw_d

    if mode == "selector_polyline_head_2d":
        q_d = evaluate_node_selector(coord, surf, triangle_labels, hydraulic["dirichlet_selector"], asset_dir=asset_dir)
        for item in hydraulic.get("extra_dirichlet_selectors", []) or []:
            q_d |= evaluate_node_selector(coord, surf, triangle_labels, item, asset_dir=asset_dir)
        q_w[q_d] = False
        head = _profile_levels_from_points(
            np.asarray(coord[0, :], dtype=np.float64),
            dict(hydraulic["head_profile"]),
            asset_dir=asset_dir,
            field_name="seepage.hydraulic_boundaries.head_profile",
        )
        wet = np.asarray(coord[1, :], dtype=np.float64) < head
        pw_d[wet] = rho_g * (head[wet] - coord[1, wet])
        return q_w, pw_d

    raise ValueError(f"Unsupported seepage.hydraulic_boundaries.mode {mode!r} in {asset_dir}.")
