"""Best-effort DMPlex probes for the C-compatible petsc4py SSR path."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

try:  # pragma: no cover - petsc4py is not available in lightweight docs jobs
    from petsc4py import PETSc
except Exception:  # pragma: no cover
    PETSc = None


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _point_count(dm, *, height: int | None = None, depth: int | None = None) -> int | None:
    try:
        if height is not None:
            start, end = dm.getHeightStratum(int(height))
        else:
            start, end = dm.getDepthStratum(int(depth or 0))
        return int(end - start)
    except Exception:
        return None


def _create_fe(dim: int, field_dim: int, degree: int, comm):
    fe = PETSc.FE().createLagrange(int(dim), int(field_dim), True, int(degree), comm=comm)
    try:
        fe.setName(f"P{int(degree)}_displacement")
    except Exception:
        pass
    return fe


def _degree_dofs(mesh_path: Path, *, degree: int, dim: int, field_dim: int, comm) -> dict[str, object]:
    dm = None
    try:
        dm = PETSc.DMPlex().createFromFile(str(mesh_path), comm=comm)
        fe = _create_fe(dim, field_dim, degree, comm)
        try:
            dm.setField(0, None, fe)
        except TypeError:
            dm.setField(0, fe)
        dm.createDS()
        vec = dm.createGlobalVec()
        try:
            local_size = int(vec.getLocalSize())
            global_size = int(vec.getSize())
        finally:
            vec.destroy()
        return {
            "degree": int(degree),
            "status": "ok",
            "local_dofs": local_size,
            "global_dofs": global_size,
        }
    except Exception as exc:
        return {
            "degree": int(degree),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if dm is not None:
            try:
                dm.destroy()
            except Exception:
                pass


def probe_dmplex_lagrange_layout(
    mesh_path: str | Path,
    *,
    degrees: Iterable[int] = (1, 2, 4),
    dim: int = 3,
    field_dim: int = 3,
    comm=None,
) -> dict[str, object]:
    """Return a non-fatal DMPlex layout probe for parity diagnostics.

    The first petsc4py rewrite pass still uses the existing array/CSR assembly
    path.  This probe records whether the same mesh can be opened through
    DMPlex and whether PETSc can create Lagrange global vectors for the target
    polynomial degrees.  Later work can make these DMs the source of truth.
    """

    if PETSc is None:
        return {"status": "unavailable", "error": "petsc4py is not importable"}
    if comm is None:
        comm = PETSc.COMM_WORLD

    path = Path(mesh_path)
    payload: dict[str, object] = {
        "status": "ok",
        "mesh_file": str(path),
        "rank": _safe_int(comm.getRank()),
        "size": _safe_int(comm.getSize()),
    }
    dm = None
    try:
        dm = PETSc.DMPlex().createFromFile(str(path), comm=comm)
        payload["cells_local"] = _point_count(dm, height=0)
        payload["vertices_local"] = _point_count(dm, depth=0)
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return payload
    finally:
        if dm is not None:
            try:
                dm.destroy()
            except Exception:
                pass

    level_rows = [
        _degree_dofs(path, degree=int(degree), dim=int(dim), field_dim=int(field_dim), comm=comm)
        for degree in degrees
    ]
    payload["levels"] = level_rows
    if any(row.get("status") != "ok" for row in level_rows):
        payload["status"] = "partial"
    return payload
