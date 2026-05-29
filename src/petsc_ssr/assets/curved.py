"""Boundary-geometry contracts for curved and high-order asset supports."""

from __future__ import annotations

from .api import BoundaryGeometryPatch, BoundaryGeometrySpec, ProblemAssetAPI


def boundary_geometry_specs(asset: ProblemAssetAPI) -> dict[str, BoundaryGeometrySpec]:
    getter = getattr(asset, "boundary_geometry_specs", None)
    raw = getter() if callable(getter) else {}
    return {
        str(name): BoundaryGeometrySpec(
            name=str(name),
            support_boundary=str(support_boundary),
            geometry_order=int(geometry_order),
        )
        for name, (support_boundary, geometry_order) in raw.items()
    }


__all__ = [
    "BoundaryGeometryPatch",
    "BoundaryGeometrySpec",
    "boundary_geometry_specs",
]
