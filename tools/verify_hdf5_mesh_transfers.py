#!/usr/bin/env python3
"""Verify HDF5 -> Gmsh mesh transfers for all supported 3D mesh families."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from slope_stability.mesh import load_mesh_from_file, load_mesh_gmsh_waterlevels, load_mesh_p2_comsol


ROOT = Path(__file__).resolve().parents[1]
MESH_ROOT = ROOT / "meshes"


@dataclass(frozen=True)
class VerificationResult:
    source: Path
    target: Path
    max_coord_drift: float


def _node_maps(old_coord: np.ndarray, new_coord: np.ndarray, *, tol: float) -> tuple[np.ndarray, np.ndarray, float]:
    tree = cKDTree(np.asarray(old_coord, dtype=np.float64).T)
    dist, idx = tree.query(np.asarray(new_coord, dtype=np.float64).T, k=1)
    max_dist = float(np.max(dist)) if dist.size else 0.0
    if max_dist > tol:
        raise AssertionError(f"Coordinate drift {max_dist:.3e} exceeds tolerance {tol:.3e}.")
    if np.unique(idx).size != old_coord.shape[1]:
        raise AssertionError("Coordinate-based node map is not bijective.")
    old_to_new = np.empty_like(idx)
    old_to_new[idx] = np.arange(idx.size, dtype=np.int64)
    return idx.astype(np.int64), old_to_new.astype(np.int64), max_dist


def _compare_connectivity(old_conn: np.ndarray, new_conn: np.ndarray, old_to_new: np.ndarray, label: str) -> None:
    if not np.array_equal(old_to_new[np.asarray(old_conn, dtype=np.int64)], np.asarray(new_conn, dtype=np.int64)):
        raise AssertionError(f"{label} connectivity differs after node remapping.")


def _verify_generic_pair(h5_path: Path, msh_path: Path, *, boundary_type: int) -> VerificationResult:
    old = load_mesh_from_file(h5_path, boundary_type=boundary_type, elem_type="P2")
    new = load_mesh_from_file(msh_path, boundary_type=boundary_type, elem_type="P2")
    new_to_old, old_to_new, drift = _node_maps(old.coord, new.coord, tol=1.0e-10)
    if not np.array_equal(old.material, new.material):
        raise AssertionError("Material identifiers differ.")
    if not np.array_equal(old.boundary, new.boundary):
        raise AssertionError("Boundary labels differ.")
    if not np.array_equal(old.q_mask[:, new_to_old], new.q_mask):
        raise AssertionError("q_mask differs after node remapping.")
    _compare_connectivity(old.elem, new.elem, old_to_new, "Element")
    _compare_connectivity(old.surf, new.surf, old_to_new, "Surface")
    return VerificationResult(source=h5_path, target=msh_path, max_coord_drift=drift)


def _verify_waterlevels_pair(h5_path: Path, msh_path: Path) -> VerificationResult:
    old = load_mesh_gmsh_waterlevels(h5_path)
    new = load_mesh_gmsh_waterlevels(msh_path)
    new_to_old, old_to_new, drift = _node_maps(old.coord, new.coord, tol=1.0e-10)
    if not np.array_equal(old.material, new.material):
        raise AssertionError("Material identifiers differ.")
    if not np.array_equal(old.triangle_labels, new.triangle_labels):
        raise AssertionError("Triangle labels differ.")
    if not np.array_equal(old.q_mask[:, new_to_old], new.q_mask):
        raise AssertionError("q_mask differs after node remapping.")
    _compare_connectivity(old.elem, new.elem, old_to_new, "Element")
    _compare_connectivity(old.surf, new.surf, old_to_new, "Surface")
    return VerificationResult(source=h5_path, target=msh_path, max_coord_drift=drift)


def _verify_comsol_pair(h5_path: Path, msh_path: Path) -> VerificationResult:
    max_drift = 0.0
    for boundary_type in (0, 1):
        old = load_mesh_p2_comsol(h5_path, boundary_type=boundary_type)
        new = load_mesh_p2_comsol(msh_path, boundary_type=boundary_type)
        new_to_old, old_to_new, drift = _node_maps(old.coord, new.coord, tol=1.0e-10)
        max_drift = max(max_drift, drift)
        if not np.array_equal(old.material, new.material):
            raise AssertionError(f"Material identifiers differ for boundary_type={boundary_type}.")
        if not np.array_equal(old.triangle_labels, new.triangle_labels):
            raise AssertionError(f"Triangle labels differ for boundary_type={boundary_type}.")
        if not np.array_equal(old.q_mask[:, new_to_old], new.q_mask):
            raise AssertionError(f"q_mask differs for boundary_type={boundary_type} after node remapping.")
        _compare_connectivity(old.elem, new.elem, old_to_new, f"Element(boundary_type={boundary_type})")
        _compare_connectivity(old.surf, new.surf, old_to_new, f"Surface(boundary_type={boundary_type})")
    return VerificationResult(source=h5_path, target=msh_path, max_coord_drift=max_drift)


def _verify_path(h5_path: Path) -> VerificationResult:
    msh_path = h5_path.with_suffix(".msh")
    if not msh_path.exists():
        raise FileNotFoundError(f"Missing converted mesh {msh_path}")

    family = h5_path.parent.name
    if family == "3d_hetero_seepage":
        return _verify_waterlevels_pair(h5_path, msh_path)
    if family == "3d_hetero_seepage_ssr_comsol":
        return _verify_comsol_pair(h5_path, msh_path)
    boundary_type = 1 if family == "3d_siopt" else 0
    return _verify_generic_pair(h5_path, msh_path, boundary_type=boundary_type)


def main() -> None:
    files = sorted(MESH_ROOT.rglob("*.h5"))
    results = []
    for h5_path in files:
        result = _verify_path(h5_path)
        results.append(result)
        rel = result.source.relative_to(ROOT)
        print(f"OK {rel} -> {result.target.relative_to(ROOT)} drift={result.max_coord_drift:.3e}")

    max_drift = max((r.max_coord_drift for r in results), default=0.0)
    print(f"\nVerified {len(results)} HDF5->MSH transfers. Max coordinate drift: {max_drift:.3e}")


if __name__ == "__main__":
    main()
