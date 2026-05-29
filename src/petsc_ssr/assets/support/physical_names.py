"""Gmsh physical-name support parsing for mesh assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhysicalName:
    dim: int
    tag: int
    name: str


def parse_gmsh_physical_names(path: str | Path | None) -> dict[str, dict[str, int]]:
    """Read region, boundary, and nodeset supports from a Gmsh v2/v4 text mesh.

    Prefixed names such as ``region:core`` and ``boundary:base`` are canonical
    and avoid ambiguity. For older unprefixed meshes, classify by the highest
    entity dimension present in ``$PhysicalNames`` so both 2D and 3D assets map
    their top-dimensional supports to regions.
    """

    records = _read_physical_name_records(path)
    regions: dict[str, int] = {}
    boundaries: dict[str, int] = {}
    nodesets: dict[str, int] = {}
    if not records:
        return {"regions": regions, "boundaries": boundaries, "nodesets": nodesets}

    mesh_dim = max(record.dim for record in records)
    for record in records:
        name = record.name
        if name.startswith("region:"):
            regions[name.split(":", 1)[1]] = record.tag
        elif name.startswith("boundary:"):
            boundaries[name.split(":", 1)[1]] = record.tag
        elif name.startswith("nodeset:"):
            nodesets[name.split(":", 1)[1]] = record.tag
        elif record.dim == mesh_dim:
            regions[name] = record.tag
        elif record.dim == 0:
            nodesets[name] = record.tag
        else:
            boundaries[name] = record.tag
    return {"regions": regions, "boundaries": boundaries, "nodesets": nodesets}


def _read_physical_name_records(path: str | Path | None) -> list[PhysicalName]:
    if path is None or not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    try:
        start = lines.index("$PhysicalNames")
    except ValueError:
        return []
    try:
        count = int(lines[start + 1].strip())
    except (IndexError, ValueError):
        return []
    out: list[PhysicalName] = []
    for raw in lines[start + 2 : start + 2 + count]:
        parts = raw.split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            dim = int(parts[0])
            tag = int(parts[1])
        except ValueError:
            continue
        out.append(PhysicalName(dim=dim, tag=tag, name=parts[2].strip().strip('"')))
    return out
