#!/usr/bin/env python
"""Local uniform-refinement runner for the 3D heterogeneous SSR asset.

Generated under artifacts on purpose: uniform refinements are local study inputs,
not tracked mesh assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import meshio
import numpy as np


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError("Could not locate repository root.")


ROOT = _repo_root()
STUDY_DIR = Path(__file__).resolve().parent
MESH_DIR = STUDY_DIR / "meshes"
CASE_DIR = STUDY_DIR / "cases"
RUN_DIR = STUDY_DIR / "runs"
BASE_MESH = ROOT / "meshes" / "3d_hetero_slope" / "adaptive_family_a_l1.msh"
BASE_CASE = ROOT / "benchmarks" / "slope_stability_3D_hetero_SSR_default" / "case.toml"


def _split_cells_by_type(mesh: meshio.Mesh) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    cells_by_type: dict[str, list[np.ndarray]] = {}
    tags_by_type: dict[str, list[np.ndarray]] = {}
    for block, physical in zip(mesh.cells, mesh.cell_data["gmsh:physical"], strict=True):
        cells_by_type.setdefault(str(block.type), []).append(np.asarray(block.data, dtype=np.int64))
        tags_by_type.setdefault(str(block.type), []).append(np.asarray(physical, dtype=np.int64).ravel())
    return cells_by_type, tags_by_type


def _uniform_refine_tet_mesh(src: Path, dst: Path) -> None:
    mesh = meshio.read(src)
    cells_by_type, tags_by_type = _split_cells_by_type(mesh)
    points = [np.asarray(p, dtype=np.float64).copy() for p in np.asarray(mesh.points, dtype=np.float64)]
    edge_mid: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = tuple(sorted((int(a), int(b))))
        found = edge_mid.get(key)
        if found is not None:
            return found
        idx = len(points)
        edge_mid[key] = idx
        points.append(0.5 * (points[key[0]] + points[key[1]]))
        return idx

    def volume(tet: list[int]) -> float:
        p = np.asarray([points[i] for i in tet], dtype=np.float64)
        return float(np.linalg.det(np.column_stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]))) / 6.0)

    def orient(tet: list[int]) -> list[int]:
        return [tet[0], tet[2], tet[1], tet[3]] if volume(tet) < 0.0 else tet

    ref_tets: list[list[int]] = []
    ref_tet_tags: list[int] = []
    for arr, tags in zip(cells_by_type.get("tetra", []), tags_by_type.get("tetra", []), strict=True):
        for tet, tag in zip(arr, tags, strict=True):
            a, b, c, d = [int(v) for v in tet]
            ab = midpoint(a, b)
            ac = midpoint(a, c)
            ad = midpoint(a, d)
            bc = midpoint(b, c)
            bd = midpoint(b, d)
            cd = midpoint(c, d)
            children = (
                [a, ab, ac, ad],
                [ab, b, bc, bd],
                [ac, bc, c, cd],
                [ad, bd, cd, d],
                [ab, ac, ad, cd],
                [ab, ac, bc, cd],
                [ab, bc, bd, cd],
                [ab, ad, bd, cd],
            )
            for child in children:
                ref_tets.append(orient(child))
                ref_tet_tags.append(int(tag))

    ref_tris: list[list[int]] = []
    ref_tri_tags: list[int] = []
    for arr, tags in zip(cells_by_type.get("triangle", []), tags_by_type.get("triangle", []), strict=True):
        for tri, tag in zip(arr, tags, strict=True):
            a, b, c = [int(v) for v in tri]
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ac = midpoint(a, c)
            ref_tris.extend(([a, ab, ac], [ab, b, bc], [ac, bc, c], [ab, bc, ac]))
            ref_tri_tags.extend([int(tag)] * 4)

    node_sets: dict[int, set[int]] = {}
    for arr, tags in zip(cells_by_type.get("vertex", []), tags_by_type.get("vertex", []), strict=True):
        for vertex, tag in zip(arr.reshape(-1), tags, strict=True):
            node_sets.setdefault(int(tag), set()).add(int(vertex))
    for tag, selected in list(node_sets.items()):
        expanded = set(selected)
        for (a, b), mid in edge_mid.items():
            if a in selected and b in selected:
                expanded.add(mid)
        node_sets[tag] = expanded

    ref_vertices: list[list[int]] = []
    ref_vertex_tags: list[int] = []
    for tag in sorted(node_sets):
        for node in sorted(node_sets[tag]):
            ref_vertices.append([node])
            ref_vertex_tags.append(tag)

    cell_blocks: list[tuple[str, np.ndarray]] = []
    physical: list[np.ndarray] = []
    geometrical: list[np.ndarray] = []
    if ref_vertices:
        cell_blocks.append(("vertex", np.asarray(ref_vertices, dtype=np.int64)))
        physical.append(np.asarray(ref_vertex_tags, dtype=np.int64))
        geometrical.append(np.asarray(ref_vertex_tags, dtype=np.int64))
    cell_blocks.append(("triangle", np.asarray(ref_tris, dtype=np.int64)))
    physical.append(np.asarray(ref_tri_tags, dtype=np.int64))
    geometrical.append(np.asarray(ref_tri_tags, dtype=np.int64))
    cell_blocks.append(("tetra", np.asarray(ref_tets, dtype=np.int64)))
    physical.append(np.asarray(ref_tet_tags, dtype=np.int64))
    geometrical.append(np.asarray(ref_tet_tags, dtype=np.int64))

    dst.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(
        dst,
        meshio.Mesh(
            np.asarray(points, dtype=np.float64),
            cell_blocks,
            cell_data={"gmsh:physical": physical, "gmsh:geometrical": geometrical},
            field_data=mesh.field_data,
        ),
        file_format="gmsh22",
        binary=False,
    )


def prepare(max_refinement: int) -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASE_MESH, MESH_DIR / "l1.msh")
    current = MESH_DIR / "l1.msh"
    for refinement in range(1, int(max_refinement) + 1):
        dst = MESH_DIR / f"l1_r{refinement}.msh"
        if not dst.exists():
            _uniform_refine_tet_mesh(current, dst)
        current = dst
        mesh = meshio.read(current)
        tet_count = sum(len(block.data) for block in mesh.cells if str(block.type) == "tetra")
        print(f"prepared {current}: points={mesh.points.shape[0]} tetra={tet_count}", flush=True)


def _case_text(
    level: str,
    *,
    omega_max: float,
    step_max: int,
    elem_type: str = "P2",
    store_step_u: bool = True,
    constitutive_mode: str = "overlap",
    pmg_coarse_level: str | None = None,
    pmg_fine_hierarchy_mode: str = "default",
    mg_coarse_pc_type: str | None = None,
    mg_coarse_factor_solver_type: str | None = None,
    petsc_opt: tuple[str, ...] = (),
) -> str:
    text = BASE_CASE.read_text(encoding="utf-8")
    text = text.replace('elem_type = "P4"', f'elem_type = "{str(elem_type).upper()}"')
    text = text.replace('mesh_variant = "adaptive_family_a_l1.msh"', f'mesh_variant = "{level}.msh"')
    text = text.replace("omega_max = 6.7e6", f"omega_max = {float(omega_max):.16g}")
    text = text.replace("step_max = 100", f"step_max = {int(step_max)}")
    text = text.replace(
        "constitutive_mode = \"overlap\"",
        f"constitutive_mode = \"{str(constitutive_mode)}\"\nstore_step_u = {str(bool(store_step_u)).lower()}",
    )
    linear_extra: list[str] = []
    if pmg_coarse_level is not None:
        linear_extra.append(f'pmg_coarse_mesh_variant = "{pmg_coarse_level}.msh"')
    if str(pmg_fine_hierarchy_mode).strip().lower() != "default":
        linear_extra.append(f'pmg_fine_hierarchy_mode = "{str(pmg_fine_hierarchy_mode).strip().lower()}"')
    if mg_coarse_factor_solver_type is not None:
        linear_extra.append(f'factor_solver_type = "{mg_coarse_factor_solver_type}"')
    if linear_extra:
        text = text.replace('pc_backend = "pmg_shell"', 'pc_backend = "pmg_shell"\n' + "\n".join(linear_extra), 1)
    extra_opts: list[str] = []
    if mg_coarse_pc_type is not None:
        extra_opts.append(f"mg_coarse_pc_type={mg_coarse_pc_type}")
    if mg_coarse_factor_solver_type is not None:
        extra_opts.append(f"mg_coarse_factor_solver_type={mg_coarse_factor_solver_type}")
    extra_opts.extend(str(entry) for entry in petsc_opt)
    if extra_opts:
        old = 'petsc_opt = ["pc_hypre_boomeramg_max_iter=4", "pc_hypre_boomeramg_tol=0.0"]'
        extra = ", ".join(f'"{entry}"' for entry in extra_opts)
        new = f'petsc_opt = ["pc_hypre_boomeramg_max_iter=4", "pc_hypre_boomeramg_tol=0.0", {extra}]'
        text = text.replace(old, new)
    return text


def _rss_kib() -> int | None:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None


def _debug_record(debug_dir: Path | None, rank: int, label: str, **payload) -> None:
    if debug_dir is None:
        return
    record = {
        "time": time.time(),
        "label": str(label),
        "rank": int(rank),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "rss_kib": _rss_kib(),
        **payload,
    }
    rank_dir = debug_dir / "ranks"
    rank_dir.mkdir(parents=True, exist_ok=True)
    with (rank_dir / f"rank_{int(rank):04d}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")


def _debug_stage(comm, debug_dir: Path | None, label: str, **payload) -> None:
    rank = int(comm.getRank())
    record = {
        "label": str(label),
        "rank": rank,
        "size": int(comm.getSize()),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "rss_kib": _rss_kib(),
        **payload,
    }
    _debug_record(debug_dir, rank, label, **payload)
    gathered = None
    try:
        gathered = comm.tompi4py().allgather(record)
    except Exception as exc:
        _debug_record(debug_dir, rank, f"{label}_allgather_failed", error=repr(exc))
    if rank == 0:
        print(f"[debug-stage] {label} | {payload}", flush=True)
        if debug_dir is not None and gathered is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            hosts: dict[str, int] = {}
            max_rss = 0
            for item in gathered:
                hosts[str(item.get("host", ""))] = hosts.get(str(item.get("host", "")), 0) + 1
                max_rss = max(max_rss, int(item.get("rss_kib") or 0))
            summary = {
                "time": time.time(),
                "label": str(label),
                "size": int(comm.getSize()),
                "host_counts": hosts,
                "max_rss_kib": max_rss,
                "payload": payload,
            }
            with (debug_dir / "stage_summary.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, default=str, sort_keys=True) + "\n")
            (debug_dir / f"stage_{str(label).replace('/', '_')}.json").write_text(
                json.dumps(gathered, default=str, indent=2, sort_keys=True),
                encoding="utf-8",
            )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _case_probe(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return {"path": str(path), "exists": path.exists(), "ok": False, "error": repr(exc)}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "ok": "[problem]" in text and 'asset = "3d_hetero_slope"' in text,
        "size": len(text.encode("utf-8")),
        "stat_size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "has_problem": "[problem]" in text,
        "has_asset": 'asset = "3d_hetero_slope"' in text,
        "prefix": text[:240],
    }


def _abort_comm(comm, code: int) -> None:
    try:
        comm.tompi4py().Abort(int(code))
    except Exception:
        raise SystemExit(int(code))


def _patch_resolver(level: str, *, extra_levels: tuple[str, ...] = (), pmg_chain_levels: tuple[str, ...] = ()) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from slope_stability.assets import load_problem_asset
    from slope_stability.assets.api import ResolvedVariant
    import slope_stability.problem_asset_runtime as runtime
    import slope_stability.execution.asset_case.mechanics_3d as mechanics_3d
    import slope_stability.execution.asset_case.runner as runner
    import slope_stability.linear.pmg as pmg
    import slope_stability.linear.solver as linear_solver
    import slope_stability.postprocess.case_mesh as case_mesh

    original_resolve = runtime.resolve_problem_asset
    original_from_config = runtime.resolve_problem_asset_from_config
    local_levels = {str(level), *(str(v) for v in extra_levels), *(str(v) for v in pmg_chain_levels)}

    def resolve_problem_asset(*, asset_name: str, mesh_variant: str | None = None, profile: str | None = None):
        variant_name = str(mesh_variant or "")
        if str(asset_name) != "3d_hetero_slope" or not variant_name.endswith(".msh") or variant_name[:-4] not in local_levels:
            return original_resolve(asset_name=asset_name, mesh_variant=mesh_variant, profile=profile)
        local_level = variant_name[:-4]
        mesh_path = (MESH_DIR / f"{local_level}.msh").resolve()
        asset = load_problem_asset("3d_hetero_slope")
        base = asset.resolve_variant("adaptive_family_a_l1.msh", profile=profile)
        variant = ResolvedVariant(
            asset_id="3d_hetero_slope",
            name=variant_name,
            source={"path": str(mesh_path), "generated_from": str(BASE_MESH.resolve()), "uniform_refinement_level": local_level},
            mesh_path=mesh_path,
            metadata={"generated": True, "uniform_refinement_level": local_level},
            profile=base.profile,
        )
        return runtime.ResolvedAsset(
            definition=asset,
            variant_name=variant.name,
            variant=variant.as_dict(),
            resolved_variant=variant,
            mesh_path=mesh_path,
        )

    def resolve_problem_asset_from_config(cfg):
        problem = cfg.problem
        mesh_variant = str(getattr(problem, "mesh_variant", ""))
        if getattr(problem, "asset", None) == "3d_hetero_slope" and mesh_variant.endswith(".msh") and mesh_variant[:-4] in local_levels:
            return resolve_problem_asset(
                asset_name="3d_hetero_slope",
                mesh_variant=mesh_variant,
                profile=getattr(problem, "profile", None),
            )
        return original_from_config(cfg)

    manual_pc_cls = linear_solver._ManualPMGShellPC
    if not hasattr(manual_pc_cls, "_coarse_factor_solver_type") and not getattr(
        manual_pc_cls, "_artifact_coarse_factor_patch", False
    ):
        original_build_coarse_ksp = manual_pc_cls._build_coarse_ksp
        original_diagnostics = manual_pc_cls.diagnostics

        def _build_coarse_ksp_with_factor(self, A, *, prefix: str):
            factor = self.solver.preconditioner_options.get("mg_coarse_factor_solver_type")
            if factor is None:
                factor = self.solver.preconditioner_options.get("factor_solver_type")
            if factor is not None:
                opts = linear_solver.PETSc.Options()
                opts[f"{prefix}pc_factor_mat_solver_type"] = str(factor)
            return original_build_coarse_ksp(self, A, prefix=prefix)

        def _diagnostics_with_factor(self, *args, **kwargs):
            payload = original_diagnostics(self, *args, **kwargs)
            if self.coarse_ksp is not None and str(self.coarse_ksp.getPC().getType()).lower() == "lu":
                payload["manualmg_coarse_factor_solver_type"] = str(self.coarse_ksp.getPC().getFactorSolverType())
            return payload

        manual_pc_cls._build_coarse_ksp = _build_coarse_ksp_with_factor
        manual_pc_cls.diagnostics = _diagnostics_with_factor
        manual_pc_cls._artifact_coarse_factor_patch = True

    normalized_chain = tuple(str(v) for v in pmg_chain_levels if str(v))
    if normalized_chain:
        chain_variants_coarse_to_fine = tuple(f"{v}.msh" for v in normalized_chain)

        def _build_mixed_chain_hierarchy(
            resolved_asset,
            *,
            coarse_mesh_variant: str,
            fine_elem_type: str = "P2",
            node_ordering: str = "original",
            reorder_parts: int | None = None,
            material_rows: list[list[float]] | None = None,
            comm,
        ):
            # build_3d_mixed_pmg_chain_hierarchy expects the coarse variants in
            # fine-adjacent-to-coarsest order; the CLI exposes coarse-to-fine.
            return pmg.build_3d_mixed_pmg_chain_hierarchy(
                resolved_asset,
                coarse_mesh_variants=tuple(reversed(chain_variants_coarse_to_fine)),
                fine_elem_type=fine_elem_type,
                node_ordering=node_ordering,
                reorder_parts=reorder_parts,
                material_rows=material_rows,
                comm=comm,
            )

        mechanics_3d.build_3d_mixed_pmg_hierarchy = _build_mixed_chain_hierarchy

    runtime.resolve_problem_asset = resolve_problem_asset
    runtime.resolve_problem_asset_from_config = resolve_problem_asset_from_config
    mechanics_3d.resolve_problem_asset = resolve_problem_asset
    pmg.resolve_problem_asset = resolve_problem_asset
    runner.resolve_problem_asset_from_config = resolve_problem_asset_from_config
    case_mesh.resolve_problem_asset_from_config = resolve_problem_asset_from_config


def run_level(
    level: str,
    *,
    omega_max: float,
    step_max: int,
    elem_type: str = "P2",
    store_step_u: bool = True,
    constitutive_mode: str = "overlap",
    out_dir: Path,
    pmg_coarse_level: str | None = None,
    pmg_fine_hierarchy_mode: str = "default",
    mg_coarse_pc_type: str | None = None,
    mg_coarse_factor_solver_type: str | None = None,
    petsc_opt: tuple[str, ...] = (),
    pmg_chain_levels: tuple[str, ...] = (),
    case_path: Path | None = None,
    debug_dir: Path | None = None,
) -> None:
    if pmg_chain_levels and pmg_coarse_level is None:
        pmg_coarse_level = str(pmg_chain_levels[0])
    extra_levels = tuple(v for v in (pmg_coarse_level,) if v is not None)
    _patch_resolver(level, extra_levels=extra_levels, pmg_chain_levels=tuple(pmg_chain_levels))
    sys.path.insert(0, str(ROOT / "src"))
    from petsc4py import PETSc
    from slope_stability.core.run_config import load_run_case_config
    from slope_stability.execution.asset_case import run_case_config

    comm = PETSc.COMM_WORLD
    rank = int(comm.getRank())
    if debug_dir is None:
        env_debug_dir = os.environ.get("SSR_DEBUG_DIR")
        debug_dir = Path(env_debug_dir) if env_debug_dir else None
    if case_path is None:
        env_case_path = os.environ.get("SSR_CASE_PATH")
        case_path = Path(env_case_path) if env_case_path else CASE_DIR / f"{level}.toml"
    case_path = Path(case_path)
    out_dir = Path(out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()
    _debug_stage(
        comm,
        debug_dir,
        "run_level_start",
        level=str(level),
        elem_type=str(elem_type).upper(),
        out_dir=str(out_dir),
        case_path=str(case_path),
        omega_max=float(omega_max),
        step_max=int(step_max),
        pmg_fine_hierarchy_mode=str(pmg_fine_hierarchy_mode),
        pmg_chain_levels=list(pmg_chain_levels),
    )

    if rank == 0:
        case_text = _case_text(
            level,
            elem_type=elem_type,
            omega_max=omega_max,
            step_max=step_max,
            store_step_u=store_step_u,
            constitutive_mode=constitutive_mode,
            pmg_coarse_level=pmg_coarse_level,
            pmg_fine_hierarchy_mode=pmg_fine_hierarchy_mode,
            mg_coarse_pc_type=mg_coarse_pc_type,
            mg_coarse_factor_solver_type=mg_coarse_factor_solver_type,
            petsc_opt=tuple(petsc_opt),
        )
        _write_text_atomic(case_path, case_text)
        if debug_dir is not None:
            (debug_dir / "case_written_by_rank0.toml").write_text(case_text, encoding="utf-8")
    comm.barrier()

    probe = _case_probe(case_path)
    _debug_record(debug_dir, rank, "case_probe", **probe)
    probes = comm.tompi4py().allgather(probe)
    if rank == 0:
        bad = [item for item in probes if not bool(item.get("ok"))]
        print(
            f"[debug-stage] case_probe | ranks={len(probes)} bad={len(bad)} "
            f"path={case_path}",
            flush=True,
        )
        if debug_dir is not None:
            (debug_dir / "case_probe_all_ranks.json").write_text(json.dumps(probes, default=str, indent=2, sort_keys=True), encoding="utf-8")
            if bad:
                (debug_dir / "case_probe_bad_ranks.json").write_text(json.dumps(bad, default=str, indent=2, sort_keys=True), encoding="utf-8")
    if any(not bool(item.get("ok")) for item in probes):
        _debug_stage(comm, debug_dir, "case_probe_failed", bad_ranks=[i for i, item in enumerate(probes) if not bool(item.get("ok"))])
        _abort_comm(comm, 91)

    try:
        _debug_stage(comm, debug_dir, "load_config_start", case_path=str(case_path))
        cfg = load_run_case_config(case_path)
        _debug_stage(
            comm,
            debug_dir,
            "load_config_done",
            asset=str(cfg.problem.asset),
            mesh_variant=str(cfg.problem.mesh_variant),
            elem_type=str(cfg.problem.elem_type),
        )
    except Exception:
        _debug_record(debug_dir, rank, "load_config_exception", traceback=traceback.format_exc())
        if debug_dir is not None:
            snapshot = debug_dir / f"case_snapshot_load_failure_rank_{rank:04d}.toml"
            try:
                snapshot.write_text(case_path.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception as exc:
                _debug_record(debug_dir, rank, "case_snapshot_failed", error=repr(exc))
        _abort_comm(comm, 92)
        raise

    try:
        _debug_stage(comm, debug_dir, "run_case_config_start")
        result = run_case_config(cfg, out_dir)
        _debug_stage(comm, debug_dir, "run_case_config_done", result=result if rank == 0 else {})
    except Exception:
        _debug_record(debug_dir, rank, "run_case_config_exception", traceback=traceback.format_exc())
        _abort_comm(comm, 93)
        raise

    if rank == 0:
        print(json.dumps(result, indent=2), flush=True)


def launch(
    levels: list[str],
    *,
    ranks: int,
    omega_max: float,
    step_max: int,
    elem_type: str = "P2",
    pmg_fine_hierarchy_mode: str = "default",
) -> None:
    for level in levels:
        out_dir = RUN_DIR / level
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "solver.log"
        cmd = [
            "mpiexec",
            "-n",
            str(int(ranks)),
            sys.executable,
            str(Path(__file__).resolve()),
            "run-level",
            "--level",
            level,
            "--elem-type",
            str(elem_type).upper(),
            "--omega-max",
            str(float(omega_max)),
            "--step-max",
            str(int(step_max)),
            "--pmg-fine-hierarchy-mode",
            str(pmg_fine_hierarchy_mode),
            "--out-dir",
            str(out_dir / "run"),
        ]
        env = dict(os.environ)
        env.setdefault("PYTHONPATH", str(ROOT / "src"))
        print("launching:", " ".join(cmd), flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"completed {level}; log={log_path}", flush=True)


def summarize(levels: list[str]) -> None:
    for level in levels:
        info_path = RUN_DIR / level / "run" / "data" / "run_info.json"
        npz_path = RUN_DIR / level / "run" / "data" / "petsc_run.npz"
        if not info_path.exists() or not npz_path.exists():
            print(f"{level}: missing run artifacts")
            continue
        data = json.loads(info_path.read_text(encoding="utf-8"))
        z = np.load(npz_path, allow_pickle=True)
        ri = data["run_info"]
        print(
            f"{level}: runtime={float(ri['runtime_seconds']):.3f}s "
            f"steps={int(ri['step_count'])} unknowns={int(ri['unknowns'])} "
            f"omega={float(z['omega_hist'][-1]):.6g} lambda={float(z['lambda_hist'][-1]):.9f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--max-refinement", type=int, default=1)
    p_run = sub.add_parser("run-level")
    p_run.add_argument("--level", required=True)
    p_run.add_argument("--elem-type", choices=["P1", "P2", "P4"], default="P2")
    p_run.add_argument("--omega-max", type=float, default=7_000_000.0)
    p_run.add_argument("--step-max", type=int, default=100)
    p_run.add_argument("--store-step-u", action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument(
        "--constitutive-mode",
        choices=["global", "overlap", "unique_gather", "unique_exchange"],
        default="overlap",
    )
    p_run.add_argument("--out-dir", type=Path, required=True)
    p_run.add_argument("--pmg-coarse-level", type=str, default=None)
    p_run.add_argument(
        "--pmg-fine-hierarchy-mode",
        choices=["default", "p4_p2_intermediate"],
        default="default",
    )
    p_run.add_argument("--mg-coarse-pc-type", type=str, default=None)
    p_run.add_argument("--mg-coarse-factor-solver-type", type=str, default=None)
    p_run.add_argument("--petsc-opt", action="append", default=[])
    p_run.add_argument("--case-path", type=Path, default=None, help="Write/read the generated case TOML at this path.")
    p_run.add_argument("--debug-dir", type=Path, default=None, help="Write rank/stage debug JSONL files into this directory.")
    p_run.add_argument(
        "--pmg-chain-levels",
        nargs="+",
        default=[],
        help=(
            "P1 mesh levels from coarsest to finest, excluding the fine high-order level; "
            "for P4 p4_p2_intermediate, pass the coarse level, e.g. l1 for fine l1_r1."
        ),
    )
    p_launch = sub.add_parser("launch")
    p_launch.add_argument("--levels", nargs="+", required=True)
    p_launch.add_argument("--ranks", type=int, default=8)
    p_launch.add_argument("--elem-type", choices=["P1", "P2", "P4"], default="P2")
    p_launch.add_argument("--omega-max", type=float, default=7_000_000.0)
    p_launch.add_argument("--step-max", type=int, default=100)
    p_launch.add_argument(
        "--pmg-fine-hierarchy-mode",
        choices=["default", "p4_p2_intermediate"],
        default="default",
    )
    p_summary = sub.add_parser("summarize")
    p_summary.add_argument("--levels", nargs="+", required=True)
    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(args.max_refinement)
    elif args.cmd == "run-level":
        run_level(
            args.level,
            elem_type=args.elem_type,
            omega_max=args.omega_max,
            step_max=args.step_max,
            store_step_u=bool(args.store_step_u),
            constitutive_mode=str(args.constitutive_mode),
            out_dir=args.out_dir,
            pmg_coarse_level=args.pmg_coarse_level,
            pmg_fine_hierarchy_mode=args.pmg_fine_hierarchy_mode,
            mg_coarse_pc_type=args.mg_coarse_pc_type,
            mg_coarse_factor_solver_type=args.mg_coarse_factor_solver_type,
            petsc_opt=tuple(args.petsc_opt or ()),
            pmg_chain_levels=tuple(args.pmg_chain_levels or ()),
            case_path=args.case_path,
            debug_dir=args.debug_dir,
        )
    elif args.cmd == "launch":
        launch(
            args.levels,
            ranks=args.ranks,
            elem_type=args.elem_type,
            omega_max=args.omega_max,
            step_max=args.step_max,
            pmg_fine_hierarchy_mode=args.pmg_fine_hierarchy_mode,
        )
    elif args.cmd == "summarize":
        summarize(args.levels)


if __name__ == "__main__":
    main()
