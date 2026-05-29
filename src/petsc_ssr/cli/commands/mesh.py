"""Mesh inspection command helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from petsc_ssr.config import load_run_case_config
from petsc_ssr.problem_asset_runtime import build_mesh_for_resolved_asset, load_mechanical_problem_spec, resolve_problem_asset_from_config


class MeshInspectionError(RuntimeError):
    """Raised when mesh inspection cannot build the local mesh report."""


def mesh_report_payload(case_toml: Path) -> dict[str, Any]:
    cfg = load_run_case_config(case_toml).validate()
    resolved = resolve_problem_asset_from_config(cfg)
    try:
        mesh = build_mesh_for_resolved_asset(resolved, elem_type=cfg.problem.elem_type)
    except ImportError as exc:
        raise MeshInspectionError(
            f"{exc} Install the mesh optional extra, for example `pip install .[mesh]`, "
            "before running mesh-only inspection on file-backed mesh assets."
        ) from exc
    mechanical = None
    if cfg.problem.analysis.lower() != "seepage":
        mechanical = load_mechanical_problem_spec(resolved)
    q_mask = np.asarray(getattr(mesh, "q_mask", np.empty((0, 0), dtype=bool)), dtype=bool)
    return {
        "case": cfg.problem.name,
        "asset": resolved.asset_name,
        "mesh_variant": resolved.variant_name,
        "mesh_path": None if resolved.mesh_path is None else str(resolved.mesh_path),
        "dimension": resolved.dimension,
        "element": cfg.problem.elem_type,
        "nodes": int(np.asarray(mesh.coord).shape[1]),
        "cells": int(np.asarray(mesh.elem).shape[1]),
        "boundary_entities": int(np.asarray(mesh.surf).shape[1]),
        "regions": sorted(getattr(mesh, "region_id_by_name", {}).keys()),
        "boundaries": sorted(getattr(mesh, "boundary_id_by_name", {}).keys()),
        "nodesets": sorted(getattr(mesh, "nodesets", {}).keys()),
        "free_component_dofs": int(np.count_nonzero(q_mask)) if q_mask.size else None,
        "constrained_component_dofs": int(q_mask.size - np.count_nonzero(q_mask)) if q_mask.size else None,
        "materials": 0 if mechanical is None else len(mechanical.material_rows),
    }
