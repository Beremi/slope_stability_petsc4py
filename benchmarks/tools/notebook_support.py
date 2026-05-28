from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
from html import escape as html_escape
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import warnings
from queue import Empty, Queue
from textwrap import dedent
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.tri as mtri
import meshio
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = ROOT / "benchmarks" / "cases"
DEFAULT_PYTHON = Path(os.environ.get("PETSC_SSR_ENGINE_PYTHON", sys.executable))
DEFAULT_MPIEXEC = shutil.which("mpiexec") or "mpiexec"
SRC_DIR = ROOT / "src"
for _path in (ROOT, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

RUNTIME_SECTION_ORDER = (
    "case",
    "mesh",
    "physics",
    "continuation",
    "newton",
    "linear",
    "seepage",
    "output",
    "notebook",
    "problem",
    "geometry",
    "execution",
    "linear_solver",
    "export",
)

PARULA_EQUIV = LinearSegmentedColormap.from_list(
    "parula_equiv",
    [
        (0.2081, 0.1663, 0.5292),
        (0.2116, 0.2743, 0.6887),
        (0.1535, 0.3929, 0.7843),
        (0.1220, 0.5210, 0.7603),
        (0.2394, 0.6600, 0.6203),
        (0.4775, 0.7540, 0.4322),
        (0.7414, 0.8145, 0.2628),
        (0.9932, 0.9061, 0.1439),
    ],
)

MATERIAL_PALETTES: dict[str, dict[int, tuple[float, float, float]]] = {
    "sloan2013": {
        0: (0.0, 1.0, 1.0),
        1: (0.0, 0.0, 1.0),
    },
    "luzec": {
        0: (0.0, 0.0, 0.0),
        1: (0.0, 1.0, 0.0),
        2: (0.0, 1.0, 1.0),
        3: (0.0, 0.0, 1.0),
        4: (1.0, 1.0, 0.0),
        5: (0.0, 0.0, 1.0),
        6: (1.0, 0.0, 1.0),
        7: (0.0, 0.0, 1.0),
    },
    "franz_dam": {
        0: (0.5, 0.5, 0.5),
        1: (0.0, 0.0, 0.0),
        2: (0.0, 1.0, 1.0),
        3: (1.0, 1.0, 0.0),
        4: (0.0, 1.0, 0.0),
        5: (0.0, 0.0, 0.0),
        6: (1.0, 0.0, 1.0),
        7: (1.0, 1.0, 0.0),
        8: (0.0, 0.0, 1.0),
        9: (1.0, 0.0, 0.0),
    },
    "kozinec": {
        1: (0.0, 1.0, 1.0),
        2: (0.0, 0.0, 1.0),
        3: (1.0, 0.0, 0.0),
        4: (0.0, 0.0, 0.0),
        5: (1.0, 0.0, 0.0),
        6: (0.0, 0.0, 0.0),
        7: (1.0, 1.0, 0.0),
    },
}

SATURATION_PALETTE = {
    0: (1.0, 1.0, 0.0),
    1: (0.0, 0.0, 1.0),
}


@dataclass(frozen=True)
class RunArtifacts:
    out_dir: Path
    data_dir: Path
    plots_dir: Path
    exports_dir: Path
    run_info: dict[str, Any]
    history: dict[str, Any]
    npz: dict[str, np.ndarray]
    progress_events: list[dict[str, Any]]
    vtu_path: Path


@dataclass(frozen=True)
class VtuData:
    points: np.ndarray
    cell_blocks: list[tuple[str, np.ndarray]]
    point_data: dict[str, np.ndarray]
    cell_data: dict[str, np.ndarray]


@dataclass(frozen=True)
class NotebookExecution:
    out_dir: Path
    active_config: Path
    generated_config: Path
    reused_existing: bool
    source_label: str
    run_result: dict[str, Any] | None


@dataclass(frozen=True)
class InlineHtml:
    data: str

    def _repr_html_(self) -> str:
        return self.data


def benchmark_case_tomls(root: Path = BENCHMARKS_DIR) -> list[Path]:
    return sorted(path for path in root.glob("*/case.toml") if path.is_file())


def load_case_document(case_toml: Path) -> dict[str, Any]:
    return tomllib.loads(Path(case_toml).read_text(encoding="utf-8"))


def load_case_metadata(case_toml: Path) -> dict[str, Any]:
    case_toml = Path(case_toml).resolve()
    raw = load_case_document(case_toml)
    case = dict(raw.get("case", {}))
    mesh = dict(raw.get("mesh", {}))
    physics = dict(raw.get("physics", {}))
    mechanics = dict(physics.get("mechanics", {}))
    benchmark = dict(raw.get("benchmark", {}))
    notebook = dict(raw.get("notebook", {}))
    problem = dict(raw.get("problem", {}))
    if case or mesh:
        linear = dict(raw.get("linear", {}))
        model = str(mechanics.get("model", ""))
        analysis = "ll" if "limit" in model else ("ssr" if mechanics else "seepage")
        benchmark = {
            "title": str(case.get("title", case_toml.parent.name)),
            "comparison_kind": "continuation" if analysis in {"ssr", "ll"} else "seepage",
            "mpi_ranks": 8,
        }
        problem = {
            "name": str(case.get("name", case_toml.parent.name)),
            "analysis": analysis,
            "asset": str(mesh.get("asset", "")),
            "mesh_variant": str(mesh.get("variant", "")),
            "elem_type": str(mesh.get("element", "")),
            "profile": str(linear.get("profile", "baseline-pmg-deflated") or "baseline-pmg-deflated"),
        }
    elem_type = str(problem.get("elem_type", "")).strip().upper()
    default_surface_subdivision = 2 if elem_type == "P4" else 0
    return {
        "case_toml": case_toml,
        "case_dir": case_toml.parent,
        "benchmark_name": case_toml.parent.name,
        "benchmark": benchmark,
        "notebook": notebook,
        "problem": problem,
        "title": str(benchmark.get("title", case_toml.parent.name)),
        "comparison_kind": str(benchmark.get("comparison_kind", "")).lower(),
        "mpi_ranks": int(benchmark.get("mpi_ranks", 8)),
        "family": str(notebook.get("family", "")),
        "jupyter_backend": str(notebook.get("jupyter_backend", "trame")),
        "nonlinear_surface_subdivision": int(notebook.get("nonlinear_surface_subdivision", default_surface_subdivision)),
        "surface_decimate_reduction": float(notebook.get("surface_decimate_reduction", 0.0)),
        "boundary_edge_overlay": bool(notebook.get("boundary_edge_overlay", False)),
    }


def _codespaces_active() -> bool:
    return str(os.environ.get("CODESPACES", "")).strip().lower() == "true"


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return None
    try:
        parsed = int(str(value).strip())
    except ValueError:
        return None
    return max(parsed, 1)


def _env_flag(name: str, *, default: bool | None = None) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def load_case_sections(case_toml: Path) -> dict[str, dict[str, Any]]:
    case_toml = Path(case_toml).resolve()
    raw = load_case_document(case_toml)
    sections: dict[str, dict[str, Any]] = {}
    if "case" in raw or "mesh" in raw:
        modern_order = (
            "case",
            "mesh",
            "physics",
            "continuation",
            "newton",
            "linear",
            "seepage",
            "output",
            "notebook",
            "geometry",
        )
        for name in modern_order:
            value = raw.get(name, {})
            sections[name] = _resolve_section_paths(case_toml, value) if isinstance(value, dict) else {}
        return sections
    for name in RUNTIME_SECTION_ORDER:
        value = raw.get(name, {})
        if name == "export":
            merged = default_export_section()
            if isinstance(value, dict):
                merged.update(_resolve_section_paths(case_toml, value))
            sections[name] = merged
        else:
            sections[name] = _resolve_section_paths(case_toml, value) if isinstance(value, dict) else {}
    return sections


def load_case_materials(case_toml: Path) -> list[dict[str, Any]]:
    raw = load_case_document(case_toml)
    materials = raw.get("materials", [])
    return [dict(item) for item in materials] if isinstance(materials, list) else []


def default_export_section() -> dict[str, Any]:
    return {
        "write_custom_debug_bundle": True,
        "write_history_json": True,
        "write_solution_vtu": True,
        "custom_debug_name": "run_debug.h5",
        "history_name": "continuation_history.json",
        "solution_name": "final_solution.vtu",
    }


def render_case_toml(sections: dict[str, dict[str, Any]], materials: list[dict[str, Any]] | None = None) -> str:
    lines: list[str] = []
    for section_name in RUNTIME_SECTION_ORDER:
        section = sections.get(section_name, {})
        if not section:
            continue
        _append_toml_section(lines, section_name, section)
        lines.append("")
    for material in materials or []:
        lines.append("[[materials]]")
        for key, value in material.items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _append_toml_section(lines: list[str], name: str, section: dict[str, Any]) -> None:
    scalar_items = [(key, value) for key, value in section.items() if not isinstance(value, dict)]
    nested_items = [(key, value) for key, value in section.items() if isinstance(value, dict)]
    if scalar_items:
        lines.append(f"[{name}]")
        for key, value in scalar_items:
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
    for key, value in nested_items:
        if scalar_items:
            lines.append("")
        _append_toml_section(lines, f"{name}.{key}", value)


def write_generated_case_toml(
    *,
    case_toml: Path,
    sections: dict[str, dict[str, Any]],
    materials: list[dict[str, Any]] | None,
    run_label: str,
    root: Path | None = None,
) -> Path:
    case_toml = Path(case_toml).resolve()
    artifact_root = Path(root).resolve() if root is not None else case_toml.parent
    out_path = (
        artifact_root
        / "artifacts"
        / str(run_label)
        / "generated_case.toml"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_case_toml(sections, materials), encoding="utf-8")
    return out_path


def ensure_notebook_artifacts(
    *,
    case_toml: Path,
    sections: dict[str, dict[str, Any]],
    materials: list[dict[str, Any]] | None,
    run_label: str,
    run_mode: str = "auto",
    execution_profile: str = "smoke",
    mpi_ranks: int | None = None,
    root: Path | None = None,
) -> NotebookExecution:
    case_toml = Path(case_toml).resolve()
    artifact_root = Path(root).resolve() if root is not None else case_toml.parent
    metadata = load_case_metadata(case_toml)
    normalized_mode = str(run_mode).strip().lower()
    if normalized_mode not in {"auto", "reuse", "run"}:
        raise ValueError(f"Unsupported run_mode {run_mode!r}")

    generated_config = write_generated_case_toml(
        case_toml=case_toml,
        sections=_profile_sections(case_toml, sections, execution_profile),
        materials=materials,
        run_label=run_label,
        root=artifact_root,
    )
    reuse_candidates = candidate_artifact_dirs(case_toml=case_toml, run_label=run_label, root=artifact_root)
    if normalized_mode in {"auto", "reuse"}:
        for candidate in reuse_candidates:
            if artifact_dir_complete(candidate):
                active_config = candidate / "generated_case.toml"
                if not active_config.exists():
                    active_config = case_toml
                return NotebookExecution(
                    out_dir=candidate,
                    active_config=active_config,
                    generated_config=generated_config,
                    reused_existing=True,
                    source_label=_display_path(candidate, artifact_root),
                    run_result=None,
                )
    if normalized_mode == "reuse":
        raise FileNotFoundError(f"No reusable notebook artifacts found for {case_toml.parent.name}")

    ranks = int(mpi_ranks if mpi_ranks is not None else _profile_mpi_ranks(metadata, execution_profile))
    out_dir = artifact_root / "artifacts" / str(run_label)
    run_result = run_parallel_case(
        config_path=generated_config,
        out_dir=out_dir,
        mpi_ranks=ranks,
    )
    return NotebookExecution(
        out_dir=out_dir,
        active_config=generated_config,
        generated_config=generated_config,
        reused_existing=False,
        source_label=_display_path(out_dir, artifact_root),
        run_result=run_result,
    )


def candidate_artifact_dirs(*, case_toml: Path, run_label: str, root: Path | None = None) -> list[Path]:
    case_toml = Path(case_toml).resolve()
    benchmark_name = case_toml.parent.name
    artifact_root = Path(root).resolve() if root is not None else case_toml.parent
    candidates = [
        artifact_root / "artifacts" / str(run_label),
        ROOT / "artifacts" / "notebooks" / benchmark_name / str(run_label),
        ROOT / "artifacts" / "cases" / benchmark_name / "latest",
        ROOT / "artifacts" / "benchmarks" / "mpi8" / benchmark_name / "petsc",
    ]
    return candidates


def _display_path(path: Path, root: Path) -> str:
    path = Path(path).resolve()
    root = Path(root).resolve()
    for base in (root, ROOT):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def artifact_dir_complete(path: Path) -> bool:
    path = Path(path)
    required = [
        path / "data" / "run_info.json",
        path / "data" / "petsc_run.npz",
        path / "exports" / "final_solution.vtu",
    ]
    return all(item.exists() for item in required)


def run_parallel_case(
    *,
    config_path: Path,
    out_dir: Path,
    mpi_ranks: int,
    python_executable: Path = DEFAULT_PYTHON,
    mpiexec: str = DEFAULT_MPIEXEC,
    clean_out_dir: bool = True,
    poll_seconds: float = 0.25,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    config_path = Path(config_path).resolve()
    preserved_config_text: str | None = None
    if clean_out_dir and out_dir.exists():
        try:
            config_path.relative_to(out_dir)
        except ValueError:
            preserved_config_text = None
        else:
            if config_path.exists():
                preserved_config_text = config_path.read_text(encoding="utf-8")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if preserved_config_text is not None:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(preserved_config_text, encoding="utf-8")

    cmd = [
        str(mpiexec),
        *(["--oversubscribe"] if _should_use_mpi_oversubscribe(mpi_ranks) else []),
        "-n",
        str(int(mpi_ranks)),
        str(python_executable),
        "-m",
        "petsc_ssr.runners.run_case_from_config",
        str(config_path),
        "--output-dir",
        str(out_dir),
    ]
    env = dict(os.environ)
    paths = [str(ROOT), str(ROOT / "src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if extra_env:
        env.update(extra_env)

    print("Launching solver:", flush=True)
    print("  " + " ".join(cmd), flush=True)
    print(f"Output directory: {out_dir}", flush=True)

    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_queue: Queue[str | None] = Queue()

    def _enqueue_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            stdout_queue.put(line)
        stdout_queue.put(None)

    stdout_thread = threading.Thread(target=_enqueue_stdout, daemon=True)
    stdout_thread.start()

    progress_path = out_dir / "data" / "progress.jsonl"
    progress_position = 0
    stdout_open = True
    while stdout_open or process.poll() is None:
        while True:
            try:
                line = stdout_queue.get_nowait()
            except Empty:
                break
            if line is None:
                stdout_open = False
                break
            print(f"[solver] {line.rstrip()}", flush=True)
        progress_position = _drain_progress(progress_path, progress_position)
        if process.poll() is not None and not stdout_open:
            break
        time.sleep(poll_seconds)

    stdout_thread.join(timeout=1.0)
    _drain_progress(progress_path, progress_position)
    return_code = int(process.wait())
    if return_code != 0 and not _tolerate_sigterm_success(out_dir, return_code):
        raise RuntimeError(f"Parallel solve failed with exit code {return_code}.")

    artifacts = load_run_artifacts(out_dir)
    summary = _run_completion_summary(artifacts)
    step_count = int(summary["step_count"])
    runtime = float(summary["runtime_seconds"])
    lambda_last = summary["lambda_last"]
    omega_last = summary["omega_last"]
    print("", flush=True)
    print("=== Finished ===", flush=True)
    print(f"Accepted steps: {step_count}", flush=True)
    print(f"Final lambda:   {_format_optional_metric(lambda_last)}", flush=True)
    print(f"Final omega:    {_format_optional_metric(omega_last)}", flush=True)
    print(f"Runtime [s]:    {runtime:.3f}", flush=True)
    return {
        "out_dir": str(out_dir),
        "run_info_path": str(artifacts.data_dir / "run_info.json"),
        "npz_path": str(artifacts.data_dir / "petsc_run.npz"),
        "vtu_path": str(artifacts.vtu_path),
        "step_count": step_count,
        "lambda_last": lambda_last,
        "omega_last": omega_last,
        "runtime_seconds": runtime,
    }


def _should_use_mpi_oversubscribe(mpi_ranks: int) -> bool:
    forced = _env_flag("SLOPE_STABILITY_MPI_OVERSUBSCRIBE")
    if forced is not None:
        return forced
    cpu_count = os.cpu_count() or 1
    return _codespaces_active() or int(mpi_ranks) > int(cpu_count)


def _tolerate_sigterm_success(out_dir: Path, return_code: int) -> bool:
    if int(return_code) != 143:
        return False
    out_dir = Path(out_dir)
    if not ((out_dir / "data" / "run_info.json").exists() and (out_dir / "data" / "petsc_run.npz").exists()):
        return False
    warnings.warn(
        (
            f"MPI runner exited with code 143 after writing solver artifacts under {out_dir}. "
            "Treating this as a successful containerized shutdown."
        ),
        RuntimeWarning,
        stacklevel=2,
    )
    return True


def _run_completion_summary(artifacts: RunArtifacts) -> dict[str, Any]:
    run_info = dict(artifacts.run_info.get("run_info", {}))
    lambda_hist = np.asarray(artifacts.npz.get("lambda_hist", []), dtype=np.float64)
    omega_hist = np.asarray(artifacts.npz.get("omega_hist", []), dtype=np.float64)
    return {
        "step_count": int(run_info.get("step_count", lambda_hist.size)),
        "runtime_seconds": float(run_info.get("runtime_seconds", 0.0)),
        "lambda_last": _optional_metric(_last_history_value(lambda_hist), run_info.get("lambda_last")),
        "omega_last": _optional_metric(_last_history_value(omega_hist), run_info.get("omega_last")),
    }


def _last_history_value(history: np.ndarray) -> float | None:
    series = np.asarray(history, dtype=np.float64)
    if series.size == 0:
        return None
    return float(series.reshape(-1)[-1])


def _optional_metric(*candidates: Any) -> float | None:
    for value in candidates:
        if value is None:
            continue
        try:
            metric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(metric):
            return metric
    return None


def _format_optional_metric(value: float | None) -> str:
    return f"{value:.9f}" if value is not None else "n/a"


def load_run_artifacts(out_dir: Path) -> RunArtifacts:
    out_dir = Path(out_dir).resolve()
    data_dir = out_dir / "data"
    plots_dir = out_dir / "plots"
    exports_dir = out_dir / "exports"
    run_info = json.loads((data_dir / "run_info.json").read_text(encoding="utf-8"))
    history_path = exports_dir / "continuation_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    with np.load(data_dir / "petsc_run.npz", allow_pickle=True) as npz_file:
        npz = {name: np.asarray(npz_file[name]) for name in npz_file.files}
    progress_events = list(history.get("progress_events", []))
    vtu_path = exports_dir / "final_solution.vtu"
    return RunArtifacts(
        out_dir=out_dir,
        data_dir=data_dir,
        plots_dir=plots_dir,
        exports_dir=exports_dir,
        run_info=run_info,
        history=history,
        npz=npz,
        progress_events=progress_events,
        vtu_path=vtu_path,
    )


def load_vtu(path: Path) -> VtuData:
    mesh = meshio.read(Path(path))
    cell_blocks = [(block.type, np.asarray(block.data, dtype=np.int64)) for block in mesh.cells]
    point_data = {name: np.asarray(values) for name, values in mesh.point_data.items()}
    cell_data: dict[str, np.ndarray] = {}
    for name, values in mesh.cell_data.items():
        arrays = [np.asarray(block_values) for block_values in values]
        if not arrays:
            continue
        cell_data[name] = np.concatenate(arrays, axis=0)
    return VtuData(
        points=np.asarray(mesh.points, dtype=np.float64),
        cell_blocks=cell_blocks,
        point_data=point_data,
        cell_data=cell_data,
    )


def _vtu_2d_topology(vtu: VtuData) -> tuple[np.ndarray, np.ndarray, str]:
    elem_blocks: list[np.ndarray] = []
    elem_type: str | None = None
    for block_type, block_data in vtu.cell_blocks:
        if block_type == "triangle":
            block_elem = np.asarray(block_data, dtype=np.int64).T
            block_elem_type = "P1"
        elif block_type == "triangle6":
            block_elem = np.asarray(block_data, dtype=np.int64).T
            block_elem_type = "P2"
        else:
            continue
        if elem_type is None:
            elem_type = block_elem_type
        elif elem_type != block_elem_type:
            raise ValueError(f"Mixed 2D VTU triangle cell types are not supported: {elem_type!r} vs {block_elem_type!r}")
        elem_blocks.append(block_elem)
    if not elem_blocks or elem_type is None:
        raise ValueError("No supported 2D triangle cells found in VTU export")
    elem = np.concatenate(elem_blocks, axis=1) if len(elem_blocks) > 1 else elem_blocks[0]
    coord = np.asarray(vtu.points[:, :2].T, dtype=np.float64)
    return coord, elem, elem_type


def _canonicalize_triangle6_vtu_elem(coord: np.ndarray, elem: np.ndarray) -> np.ndarray:
    elem_arr = np.asarray(elem, dtype=np.int64)
    coord_arr = np.asarray(coord, dtype=np.float64)
    if elem_arr.shape[0] != 6:
        raise ValueError(f"Expected triangle6 connectivity with width 6, got {elem_arr.shape[0]}.")

    vertex_pts = coord_arr[:, elem_arr[:3, :]]
    midside_pts = coord_arr[:, elem_arr[3:, :]]
    edge_midpoints = np.stack(
        (
            0.5 * (vertex_pts[:, 0, :] + vertex_pts[:, 1, :]),
            0.5 * (vertex_pts[:, 1, :] + vertex_pts[:, 2, :]),
            0.5 * (vertex_pts[:, 2, :] + vertex_pts[:, 0, :]),
        ),
        axis=1,
    )
    dist2 = np.sum((midside_pts[:, :, None, :] - edge_midpoints[:, None, :, :]) ** 2, axis=0)
    permutations = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ],
        dtype=np.int64,
    )
    errors = np.stack(
        [dist2[perm[0], 0, :] + dist2[perm[1], 1, :] + dist2[perm[2], 2, :] for perm in permutations],
        axis=0,
    )
    best_idx = np.argmin(errors, axis=0)
    ordered_errors = np.sort(errors, axis=0)
    edge_lengths_sq = np.stack(
        (
            np.sum((vertex_pts[:, 1, :] - vertex_pts[:, 0, :]) ** 2, axis=0),
            np.sum((vertex_pts[:, 2, :] - vertex_pts[:, 1, :]) ** 2, axis=0),
            np.sum((vertex_pts[:, 0, :] - vertex_pts[:, 2, :]) ** 2, axis=0),
        ),
        axis=0,
    )
    edge_scale = np.sqrt(np.maximum(np.max(edge_lengths_sq, axis=0), 1.0))
    tol = np.square(1.0e-6 + 1.0e-2 * edge_scale)
    if np.any(ordered_errors[0, :] > tol):
        raise ValueError("triangle6 midside nodes cannot be matched to edge midpoints.")
    if np.any(ordered_errors[1, :] - ordered_errors[0, :] <= tol):
        raise ValueError("triangle6 midside-node roles are ambiguous.")

    role_to_local = permutations[best_idx, :]
    midside_nodes = elem_arr[3:, :].T
    e01 = midside_nodes[np.arange(elem_arr.shape[1]), role_to_local[:, 0]]
    e12 = midside_nodes[np.arange(elem_arr.shape[1]), role_to_local[:, 1]]
    e20 = midside_nodes[np.arange(elem_arr.shape[1]), role_to_local[:, 2]]
    return np.vstack((elem_arr[:3, :], e12[None, :], e20[None, :], e01[None, :]))


def _vtu_internal_elem_2d(elem: np.ndarray, elem_type: str) -> np.ndarray:
    elem_arr = np.asarray(elem, dtype=np.int64)
    if elem_type == "P2":
        return elem_arr
    return elem_arr


def _vtu_linear_triangles_2d(vtu: VtuData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    coord, elem, elem_type = _vtu_2d_topology(vtu)
    if elem_type == "P2":
        elem = _canonicalize_triangle6_vtu_elem(coord, elem)
        n_elem = elem.shape[1]
        triangles = np.empty((n_elem * 4, 3), dtype=np.int64)
        parents = np.repeat(np.arange(n_elem, dtype=np.int64), 4)
        e12 = elem[3, :]
        e20 = elem[4, :]
        e01 = elem[5, :]
        triangles[0::4, :] = np.stack((elem[0, :], e01, e20), axis=1)
        triangles[1::4, :] = np.stack((e01, elem[1, :], e12), axis=1)
        triangles[2::4, :] = np.stack((e20, e12, elem[2, :]), axis=1)
        triangles[3::4, :] = np.stack((e01, e12, e20), axis=1)
        pts = coord.T
        areas = 0.5 * (
            (pts[triangles[:, 1], 0] - pts[triangles[:, 0], 0]) * (pts[triangles[:, 2], 1] - pts[triangles[:, 0], 1])
            - (pts[triangles[:, 1], 1] - pts[triangles[:, 0], 1]) * (pts[triangles[:, 2], 0] - pts[triangles[:, 0], 0])
        )
        if np.any(areas <= 0.0):
            raise ValueError(f"triangle6 subdivision produced {int(np.count_nonzero(areas <= 0.0))} non-positive subtriangles.")
    else:
        triangles, parents = _linear_triangles_2d(elem, elem_type)
    return coord, triangles, parents, elem, elem_type


def list_saved_files(out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir).resolve()
    return sorted(path for path in out_dir.rglob("*") if path.is_file())


def show_run_summary(artifacts: RunArtifacts) -> None:
    print(json.dumps(artifacts.run_info.get("run_info", {}), indent=2))
    mesh = dict(artifacts.run_info.get("mesh", {}))
    if mesh:
        print("")
        print(json.dumps(mesh, indent=2))
    timings = dict(artifacts.run_info.get("timings", {}))
    if timings:
        print("")
        print(json.dumps(timings, indent=2))


def plot_convergence_dashboard(artifacts: RunArtifacts):
    lambda_hist = np.asarray(artifacts.npz.get("lambda_hist", []), dtype=np.float64)
    omega_hist = np.asarray(artifacts.npz.get("omega_hist", []), dtype=np.float64)
    umax_hist = np.asarray(artifacts.npz.get("Umax_hist", []), dtype=np.float64)
    accepted_steps = np.arange(1, max(len(lambda_hist), len(umax_hist), 1) + 1, dtype=np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)
    if lambda_hist.size and omega_hist.size:
        axes[0].plot(omega_hist, lambda_hist, marker="o", linewidth=1.3)
        axes[0].set_xlabel(r"$\omega$")
        axes[0].set_ylabel(r"$\lambda$")
        axes[0].set_title("Continuation curve")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No continuation data", ha="center", va="center")
        axes[0].set_axis_off()

    if umax_hist.size:
        axes[1].plot(accepted_steps[: umax_hist.size], umax_hist, marker="o", linewidth=1.3)
        axes[1].set_xlabel("Accepted step")
        axes[1].set_ylabel(r"$U_{max}$")
        axes[1].set_title("Step displacement history")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No displacement history", ha="center", va="center")
        axes[1].set_axis_off()
    fig.tight_layout()
    return fig


def plot_timing_breakdown(artifacts: RunArtifacts):
    series = _timing_breakdown_series(artifacts)
    nonzero = {key: value for key, value in series.items() if value > 0.0}
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=160)
    if not nonzero:
        ax.text(0.5, 0.5, "No timing breakdown available", ha="center", va="center")
        ax.set_axis_off()
        return fig
    labels = list(nonzero)
    values = [nonzero[label] for label in labels]
    ax.bar(labels, values, color="#3269a8")
    ax.set_ylabel("Seconds")
    ax.set_title("Timing breakdown")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    return fig


def _timing_breakdown_series(artifacts: RunArtifacts) -> dict[str, float]:
    timings = dict(artifacts.run_info.get("timings", {}))
    summary = dict(artifacts.run_info.get("c_hotpath_summary", {}))
    constitutive = dict(timings.get("constitutive", {}))
    linear = dict(timings.get("linear", {}))
    series: dict[str, float] = {}

    constitutive_time = _positive_float(sum(float(v) for v in constitutive.values()))
    if constitutive_time > 0.0:
        series["Constitutive"] = constitutive_time

    assembly_time = _positive_float(summary.get("elastic_assembly_time", timings.get("assembly_time", 0.0)))
    if assembly_time > 0.0:
        series["Elastic assembly"] = assembly_time

    linear_solve_time = _positive_float(linear.get("attempt_linear_solve_time_total", 0.0))
    if linear_solve_time > 0.0:
        series["Linear solve"] = linear_solve_time

    pc_apply_time = _positive_float(
        linear.get(
            "preconditioner_apply_time_total",
            summary.get("deflation_pc_apply_time", linear.get("deflation_pc_apply_time", 0.0)),
        )
    )
    if pc_apply_time > 0.0:
        series["Deflation PC apply"] = pc_apply_time

    projector_time = _positive_float(summary.get("deflation_projector_time", linear.get("deflation_projector_time", 0.0)))
    if projector_time > 0.0:
        series["Deflation projector"] = projector_time

    orthogonalization_time = _positive_float(
        linear.get(
            "attempt_linear_orthogonalization_time_total",
            summary.get("deflation_orthogonalization_time", linear.get("deflation_orthogonalization_time", 0.0)),
        )
    )
    if orthogonalization_time > 0.0:
        series["Orthogonalization"] = orthogonalization_time

    pc_setup_time = _positive_float(linear.get("preconditioner_setup_time_total", 0.0))
    if pc_setup_time > 0.0:
        series["PC setup"] = pc_setup_time

    continuation_time = _positive_float(summary.get("continuation_wall_time", timings.get("continuation_total_wall_time", 0.0)))
    if continuation_time > 0.0 and series:
        accounted = float(sum(series.values()))
        residual = max(continuation_time - accounted, 0.0)
        if residual > 1.0e-9:
            series["Other continuation"] = residual
    elif not series:
        wall_time = _positive_float(summary.get("wall_time", timings.get("wall_time", 0.0)))
        if wall_time > 0.0:
            series["Wall time"] = wall_time
    return series


def _positive_float(value: Any) -> float:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return metric if np.isfinite(metric) and metric > 0.0 else 0.0


def matlab_warp_scale(coord: np.ndarray, displacement: np.ndarray) -> float:
    coord_arr = np.asarray(coord, dtype=np.float64)
    disp_arr = np.asarray(displacement, dtype=np.float64)
    coord_max = float(np.max(np.abs(coord_arr))) if coord_arr.size else 0.0
    disp_max = float(np.max(np.abs(disp_arr))) if disp_arr.size else 0.0
    if coord_max <= 0.0 or disp_max <= 0.0:
        return 1.0
    return 0.05 * coord_max / disp_max


def viz_support_status() -> dict[str, bool]:
    return {
        "pyvista": _module_available("pyvista"),
        "ipywidgets": _module_available("ipywidgets"),
        "trame": _module_available("trame"),
    }


def viz_support_message() -> str:
    status = viz_support_status()
    if all(status.values()):
        return "Interactive 3D notebook extras are available."
    missing = ", ".join(name for name, present in status.items() if not present)
    return f"Interactive 3D views require optional viz extras. Missing: {missing}. Install `.[viz]`."


def get_material_palette(name: str) -> dict[int, tuple[float, float, float]]:
    palette_name = str(name).strip().lower()
    if palette_name not in MATERIAL_PALETTES:
        raise KeyError(f"Unknown material palette {name!r}")
    return dict(MATERIAL_PALETTES[palette_name])


def plot_2d_mesh(case_toml: Path):
    case_mesh = _load_case_mesh(case_toml)
    triangles, _ = _parent_triangles_2d(case_mesh.elem)
    triang = mtri.Triangulation(case_mesh.coord[0], case_mesh.coord[1], triangles)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=160)
    ax.triplot(triang, color="black", linewidth=0.55)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Mesh")
    fig.tight_layout()
    return fig


def plot_2d_heterogeneity(case_toml: Path, *, palette_name: str):
    case_mesh = _load_case_mesh(case_toml)
    triangles, parents = _parent_triangles_2d(case_mesh.elem)
    values = case_mesh.material_id[parents]
    triang = mtri.Triangulation(case_mesh.coord[0], case_mesh.coord[1], triangles)
    cmap, norm = _categorical_cmap(get_material_palette(palette_name), values)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=160)
    artist = ax.tripcolor(triang, facecolors=values, cmap=cmap, norm=norm, edgecolors="k", linewidth=0.15)
    _seal_2d_triangle_artist(artist)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Material zones")
    fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    return fig


def plot_2d_pore_pressure(artifacts: RunArtifacts, case_toml: Path):
    vtu = load_vtu(artifacts.vtu_path)
    coord, triangles, parents, elem, _ = _vtu_linear_triangles_2d(vtu)
    pore_pressure = _pore_pressure_field(artifacts, case_toml, vtu=vtu)
    parent_triangles, parent_ids = _parent_triangles_2d(elem)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    if pore_pressure.size == coord.shape[1]:
        triang = mtri.Triangulation(coord[0], coord[1], triangles)
        artist = ax.tripcolor(triang, pore_pressure, shading="gouraud", cmap=PARULA_EQUIV)
    elif pore_pressure.size == elem.shape[1]:
        triang = mtri.Triangulation(coord[0], coord[1], parent_triangles)
        artist = ax.tripcolor(triang, facecolors=pore_pressure[parent_ids], cmap=PARULA_EQUIV, edgecolors="none")
    else:
        raise ValueError(
            f"Pore-pressure field size {pore_pressure.size} does not match VTU point count {coord.shape[1]} "
            f"or cell count {elem.shape[1]}"
        )
    _seal_2d_triangle_artist(artist)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Pore pressure [kPa]")
    fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    return fig


def plot_2d_saturation(artifacts: RunArtifacts, case_toml: Path):
    vtu = load_vtu(artifacts.vtu_path)
    coord, _triangles, _parents, elem, _ = _vtu_linear_triangles_2d(vtu)
    saturation = _saturation_field(artifacts, vtu=vtu, n_cells=elem.shape[1])
    triangles, parents = _parent_triangles_2d(elem)
    values = saturation[parents]
    triang = mtri.Triangulation(coord[0], coord[1], triangles)
    cmap, norm = _categorical_cmap(SATURATION_PALETTE, values)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    artist = ax.tripcolor(triang, facecolors=values, cmap=cmap, norm=norm, edgecolors="k", linewidth=0.1)
    _seal_2d_triangle_artist(artist)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Saturation")
    fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    return fig


def plot_2d_displacement(artifacts: RunArtifacts, case_toml: Path, *, warp_scale: float | None = None):
    vtu = load_vtu(artifacts.vtu_path)
    coord, triangles, _, _, _ = _vtu_linear_triangles_2d(vtu)
    displacement = _displacement_field(artifacts, vtu, dim=2)
    displacement_mag = _point_field(vtu, "displacement_magnitude", default=np.linalg.norm(displacement, axis=1))
    scale = matlab_warp_scale(coord, displacement[:, :2].T) if warp_scale is None else float(warp_scale)
    deformed = coord[:2].T + scale * displacement[:, :2]
    triang = mtri.Triangulation(deformed[:, 0], deformed[:, 1], triangles)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    artist = ax.tripcolor(triang, displacement_mag, shading="gouraud", cmap=PARULA_EQUIV)
    _seal_2d_triangle_artist(artist)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"Displacement magnitude (warp scale = {scale:.4g})")
    fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    return fig


def plot_2d_deviatoric_strain(artifacts: RunArtifacts, case_toml: Path):
    vtu = load_vtu(artifacts.vtu_path)
    coord, _triangles, _parents, elem, elem_type = _vtu_linear_triangles_2d(vtu)
    displacement = _displacement_field(artifacts, vtu, dim=2)[:, :2].T
    display_coord, display_triangles, display_values = _build_discontinuous_deviatoric_plot_mesh_2d(
        coord,
        _vtu_internal_elem_2d(elem, elem_type),
        elem_type,
        displacement,
    )
    triang = mtri.Triangulation(display_coord[:, 0], display_coord[:, 1], display_triangles)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    artist = ax.tripcolor(triang, display_values, shading="gouraud", cmap=PARULA_EQUIV)
    _seal_2d_triangle_artist(artist)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Deviatoric strain norm")
    fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    return fig


def show_3d_mesh_view(
    artifacts: RunArtifacts,
    case_toml: Path,
    *,
    surface_subdivision: int | None = None,
    surface_decimate_reduction: float | None = None,
    jupyter_backend: str | None = None,
):
    if not _module_available("pyvista"):
        return viz_support_message()
    pv = _import_pyvista()
    grid = _display_source_grid(pv.read(artifacts.vtu_path), artifacts=artifacts, case_toml=case_toml)
    subdivision = _display_nonlinear_surface_subdivision(case_toml, override=surface_subdivision)
    reduction = _display_surface_decimate_reduction(case_toml, override=surface_decimate_reduction)
    surface = _surface_for_display(grid, case_toml=case_toml, artifacts=artifacts, nonlinear_subdivision=subdivision)
    surface = _decimate_display_mesh(surface, reduction=reduction)
    surface = _optimize_display_mesh(surface)
    plotter = _new_plotter(pv, title="Mesh outline")
    plotter.add_mesh(surface, color="white", show_edges=True, edge_color="#2a62d0")
    _apply_matlab_camera(plotter)
    return _show_plotter(plotter, case_toml, jupyter_backend=jupyter_backend)


def show_3d_pore_pressure_view(
    artifacts: RunArtifacts,
    case_toml: Path,
    *,
    surface_subdivision: int | None = None,
    surface_decimate_reduction: float | None = None,
    boundary_edge_overlay: bool | None = None,
    jupyter_backend: str | None = None,
):
    if not _module_available("pyvista"):
        return viz_support_message()
    pv = _import_pyvista()
    grid = pv.read(artifacts.vtu_path)
    if "pore_pressure" not in grid.point_data:
        vtu = load_vtu(artifacts.vtu_path)
        grid.point_data["pore_pressure"] = _pore_pressure_field(artifacts, case_toml, vtu=vtu)
    grid = _display_source_grid(grid, artifacts=artifacts, case_toml=case_toml)
    subdivision = _display_nonlinear_surface_subdivision(case_toml, override=surface_subdivision)
    reduction = _display_surface_decimate_reduction(case_toml, override=surface_decimate_reduction)
    surface = _surface_for_display(
        grid,
        case_toml=case_toml,
        artifacts=artifacts,
        nonlinear_subdivision=subdivision,
        point_array_names=("pore_pressure",),
    )
    surface = _decimate_display_mesh(surface, reduction=reduction)
    surface = _optimize_display_mesh(surface, keep_point_arrays=("pore_pressure",))
    plotter = _new_plotter(pv, title="Pore pressure [kPa]")
    plotter.add_mesh(surface, scalars="pore_pressure", cmap=PARULA_EQUIV, show_edges=False)
    if _display_boundary_edge_overlay(case_toml, override=boundary_edge_overlay):
        _add_boundary_edge_overlay(plotter, grid, case_toml=case_toml, artifacts=artifacts)
    _apply_matlab_camera(plotter)
    return _show_plotter(plotter, case_toml, jupyter_backend=jupyter_backend)


def show_3d_saturation_view(
    artifacts: RunArtifacts,
    case_toml: Path,
    *,
    surface_subdivision: int | None = None,
    surface_decimate_reduction: float | None = None,
    boundary_edge_overlay: bool | None = None,
    jupyter_backend: str | None = None,
):
    if not _module_available("pyvista"):
        return viz_support_message()
    pv = _import_pyvista()
    grid = _display_source_grid(pv.read(artifacts.vtu_path), artifacts=artifacts, case_toml=case_toml)
    saturation = _saturation_field(artifacts, n_cells=int(grid.n_cells))
    plotter = _new_plotter(pv, title="Saturation")
    legend_entries: list[list[str]] = []
    value_labels = {0.0: "unsaturated", 1.0: "saturated"}
    surface = _build_parent_boundary_surface(
        grid,
        case_toml=case_toml,
        artifacts=artifacts,
        cell_data_from_parent={"saturation": saturation},
    )
    surface = _optimize_display_mesh(surface, keep_cell_arrays=("saturation",))
    for value in sorted(float(v) for v in np.unique(surface.cell_data["saturation"])):
        cell_ids = np.flatnonzero(np.isclose(np.asarray(surface.cell_data["saturation"], dtype=np.float64), value))
        if cell_ids.size == 0:
            continue
        region = surface.extract_cells(cell_ids)
        if region.n_cells == 0:
            continue
        color = SATURATION_PALETTE.get(int(round(value)), (0.8, 0.8, 0.8))
        plotter.add_mesh(
            region,
            color=color,
            show_edges=True,
            edge_color="#222222",
            line_width=0.35,
            lighting=False,
            opacity=1.0,
        )
        legend_entries.append([value_labels.get(value, f"saturation={value:g}"), color])
    if not legend_entries:
        return "No saturation field available for 3D rendering."
    plotter.add_legend(legend_entries, bcolor="white", face="rectangle")
    if _display_boundary_edge_overlay(case_toml, override=boundary_edge_overlay):
        _add_boundary_edge_overlay(plotter, grid, case_toml=case_toml, artifacts=artifacts)
    _apply_matlab_camera(plotter)
    return _show_plotter(plotter, case_toml, jupyter_backend=jupyter_backend)


def show_3d_displacement_view(
    artifacts: RunArtifacts,
    case_toml: Path,
    *,
    warp_scale: float | None = None,
    surface_subdivision: int | None = None,
    surface_decimate_reduction: float | None = None,
    boundary_edge_overlay: bool | None = None,
    jupyter_backend: str | None = None,
):
    if not _module_available("pyvista"):
        return viz_support_message()
    pv = _import_pyvista()
    grid = _display_source_grid(pv.read(artifacts.vtu_path), artifacts=artifacts, case_toml=case_toml)
    displacement = np.asarray(artifacts.npz["U"], dtype=np.float64)
    case_mesh = _load_case_mesh(case_toml, artifacts=artifacts)
    scale = matlab_warp_scale(case_mesh.coord, displacement) if warp_scale is None else float(warp_scale)
    subdivision = _display_nonlinear_surface_subdivision(case_toml, override=surface_subdivision)
    reduction = _display_surface_decimate_reduction(case_toml, override=surface_decimate_reduction)
    surface = _surface_for_display(
        grid,
        case_toml=case_toml,
        artifacts=artifacts,
        nonlinear_subdivision=subdivision,
        point_array_names=("displacement", "displacement_magnitude"),
    )
    surface = _decimate_display_mesh(surface, reduction=reduction)
    surface = _optimize_display_mesh(surface, keep_point_arrays=("displacement", "displacement_magnitude"))
    surface = surface.warp_by_vector("displacement", factor=scale)
    surface = _optimize_display_mesh(surface, keep_point_arrays=("displacement", "displacement_magnitude"))
    plotter = _new_plotter(pv, title=f"Displacement magnitude (warp scale = {scale:.4g})")
    plotter.add_mesh(surface, scalars="displacement_magnitude", cmap=PARULA_EQUIV, show_edges=False)
    if _display_boundary_edge_overlay(case_toml, override=boundary_edge_overlay):
        _add_boundary_edge_overlay(
            plotter,
            grid,
            case_toml=case_toml,
            artifacts=artifacts,
            warp_vector_name="displacement",
            warp_factor=scale,
        )
    _apply_matlab_camera(plotter)
    return _show_plotter(plotter, case_toml, jupyter_backend=jupyter_backend)


def show_3d_deviatoric_surface_view(
    artifacts: RunArtifacts,
    case_toml: Path,
    *,
    surface_subdivision: int | None = None,
    surface_decimate_reduction: float | None = None,
    boundary_edge_overlay: bool | None = None,
    jupyter_backend: str | None = None,
):
    if not _module_available("pyvista"):
        return viz_support_message()
    pv = _import_pyvista()
    grid = _display_source_grid(pv.read(artifacts.vtu_path), artifacts=artifacts, case_toml=case_toml)
    cfg = _load_runtime_config(case_toml)
    elem_type = str(cfg.problem.elem_type).strip().upper()
    case_mesh = _load_case_mesh(case_toml, artifacts=artifacts)
    subdivision = _display_nonlinear_surface_subdivision(case_toml, override=surface_subdivision)
    reduction = _display_surface_decimate_reduction(case_toml, override=surface_decimate_reduction)
    if elem_type in {"P1", "P2"} and case_mesh.surf is not None:
        displacement = np.asarray(artifacts.npz["U"], dtype=np.float64)
        surface = _build_discontinuous_deviatoric_surface_3d(
            case_mesh.coord,
            case_mesh.elem,
            case_mesh.surf,
            elem_type,
            displacement,
        )
        surface = _decimate_display_mesh(surface, reduction=reduction)
        surface = _optimize_display_mesh(surface, keep_point_arrays=("deviatoric_strain",))
        plotter = _new_plotter(pv, title="Deviatoric strain (boundary surface)")
        plotter.add_mesh(
            surface,
            scalars="deviatoric_strain",
            cmap="jet",
            preference="point",
            show_edges=False,
            lighting=False,
        )
        if _display_boundary_edge_overlay(case_toml, override=boundary_edge_overlay):
            _add_boundary_edge_overlay(plotter, grid, case_toml=case_toml, artifacts=artifacts)
        _apply_matlab_camera(plotter)
        return _show_plotter(plotter, case_toml, jupyter_backend=jupyter_backend)
    if "deviatoric_strain" in grid.point_data:
        surface = _surface_for_display(
            grid,
            case_toml=case_toml,
            artifacts=artifacts,
            nonlinear_subdivision=subdivision,
            point_array_names=("deviatoric_strain",),
        )
        surface = _decimate_display_mesh(surface, reduction=reduction)
        surface = _optimize_display_mesh(surface, keep_point_arrays=("deviatoric_strain",))
        plotter = _new_plotter(pv, title="Deviatoric strain (boundary surface)")
        plotter.add_mesh(
            surface,
            scalars="deviatoric_strain",
            cmap="jet",
            preference="point",
            show_edges=False,
            lighting=False,
        )
        if _display_boundary_edge_overlay(case_toml, override=boundary_edge_overlay):
            _add_boundary_edge_overlay(plotter, grid, case_toml=case_toml, artifacts=artifacts)
        _apply_matlab_camera(plotter)
        return _show_plotter(plotter, case_toml, jupyter_backend=jupyter_backend)
    if "deviatoric_strain" in grid.cell_data:
        surface = _surface_for_display(
            grid,
            case_toml=case_toml,
            artifacts=artifacts,
            nonlinear_subdivision=subdivision,
            cell_data_from_parent={"deviatoric_strain": np.asarray(grid.cell_data["deviatoric_strain"], dtype=np.float64)},
        )
        surface = _decimate_display_mesh(surface, reduction=reduction)
        surface = _optimize_display_mesh(surface, keep_cell_arrays=("deviatoric_strain",))
        plotter = _new_plotter(pv, title="Deviatoric strain (boundary surface)")
        plotter.add_mesh(
            surface,
            scalars="deviatoric_strain",
            cmap="jet",
            preference="cell",
            show_edges=False,
            lighting=False,
        )
        if _display_boundary_edge_overlay(case_toml, override=boundary_edge_overlay):
            _add_boundary_edge_overlay(plotter, grid, case_toml=case_toml, artifacts=artifacts)
        _apply_matlab_camera(plotter)
        return _show_plotter(plotter, case_toml, jupyter_backend=jupyter_backend)

    if case_mesh.surf is None:
        return "No boundary surface is available for deviatoric strain rendering."
    displacement = np.asarray(artifacts.npz["U"], dtype=np.float64)
    values = compute_element_deviatoric_strain(
        case_mesh.coord,
        case_mesh.elem,
        cfg.problem.elem_type,
        displacement,
        dim=3,
    )
    triangles, face_ids = _build_plotting_mesh_with_face_ids(np.asarray(case_mesh.coord, dtype=np.float64), np.asarray(case_mesh.surf, dtype=np.int64))
    face_parent = _surface_parent_elements(np.asarray(case_mesh.elem, dtype=np.int64), np.asarray(case_mesh.surf, dtype=np.int64))
    tri_vals = np.asarray(values, dtype=np.float64)[face_parent[face_ids]]
    faces = np.column_stack((np.full(triangles.shape[0], 3, dtype=np.int64), triangles)).reshape(-1)
    surface = pv.PolyData(np.asarray(case_mesh.coord.T, dtype=np.float64), faces)
    surface.cell_data["deviatoric_strain"] = tri_vals
    plotter = _new_plotter(pv, title="Deviatoric strain (boundary surface)")
    plotter.add_mesh(
        surface,
        scalars="deviatoric_strain",
        cmap="jet",
        preference="cell",
        show_edges=False,
        lighting=False,
    )
    if _display_boundary_edge_overlay(case_toml, override=boundary_edge_overlay):
        _add_boundary_edge_overlay(plotter, grid, case_toml=case_toml, artifacts=artifacts)
    _apply_matlab_camera(plotter)
    return _show_plotter(plotter, case_toml, jupyter_backend=jupyter_backend)


def show_3d_deviatoric_slices(
    artifacts: RunArtifacts,
    case_toml: Path,
    *,
    slice_planes_x: list[float] | None = None,
    slice_planes_y: list[float] | None = None,
    slice_planes_z: list[float] | None = None,
    clim_scale_max: float | None = None,
    jupyter_backend: str | None = None,
):
    if not _module_available("pyvista"):
        return viz_support_message()
    plane_map = {
        "x": list(slice_planes_x or []),
        "y": list(slice_planes_y or []),
        "z": list(slice_planes_z or []),
    }
    if not any(plane_map.values()):
        return "No MATLAB slice planes are configured for this benchmark."
    pv = _import_pyvista()
    grid = _slice_source_grid(pv.read(artifacts.vtu_path), artifacts=artifacts, case_toml=case_toml)
    if "deviatoric_strain" in grid.point_data:
        point_grid = grid
        values = np.asarray(grid.point_data["deviatoric_strain"], dtype=np.float64)
    elif "deviatoric_strain" in grid.cell_data:
        point_grid = grid.cell_data_to_point_data(pass_cell_data=True)
        point_grid.point_data["deviatoric_strain"] = np.asarray(point_grid.point_data["deviatoric_strain"])
        values = np.asarray(grid.cell_data["deviatoric_strain"], dtype=np.float64)
    else:
        cfg = _load_runtime_config(case_toml)
        case_mesh = _load_case_mesh(case_toml, artifacts=artifacts)
        displacement = np.asarray(artifacts.npz["U"], dtype=np.float64)
        values = compute_element_deviatoric_strain(
            case_mesh.coord,
            case_mesh.elem,
            cfg.problem.elem_type,
            displacement,
            dim=3,
        )
        grid.cell_data["deviatoric_strain"] = values
        point_grid = grid.cell_data_to_point_data(pass_cell_data=True)
        point_grid.point_data["deviatoric_strain"] = np.asarray(point_grid.point_data["deviatoric_strain"])
    clim = (float(np.min(values)), float(np.max(values)))
    if clim_scale_max is not None:
        clim = (clim[0], max(clim[0], float(clim_scale_max) * clim[1]))
    plotter = _new_plotter(pv, title="MATLAB slice views")
    first = True
    for axis, planes in plane_map.items():
        normal = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis]
        for value in planes:
            origin = list(point_grid.center)
            origin[{"x": 0, "y": 1, "z": 2}[axis]] = float(value)
            slc = point_grid.slice(normal=normal, origin=origin, generate_triangles=True)
            if slc.n_points == 0:
                continue
            slc = _refine_slice_for_display(slc, point_grid, case_toml=case_toml)
            slc = _optimize_display_mesh(slc, keep_point_arrays=("deviatoric_strain",))
            plotter.add_mesh(
                slc,
                scalars="deviatoric_strain",
                cmap="jet",
                clim=clim,
                show_edges=False,
                show_scalar_bar=first,
                scalar_bar_args={"title": "deviatoric strain norm"} if first else None,
            )
            first = False
    _apply_matlab_camera(plotter)
    return _show_plotter(plotter, case_toml, jupyter_backend=jupyter_backend)


def compute_element_deviatoric_strain(
    coord: np.ndarray,
    elem: np.ndarray,
    elem_type: str,
    displacement: np.ndarray,
    *,
    dim: int,
) -> np.ndarray:
    from petsc_ssr.fem import assemble_strain_operator

    assembly = assemble_strain_operator(coord, elem, elem_type, dim=dim)
    strain = assembly.B @ np.asarray(displacement, dtype=np.float64).reshape(-1, order="F")
    strain = strain.reshape(assembly.n_strain, -1, order="F")
    dev_norm = deviatoric_strain_norm(strain, dim=dim)
    n_q = max(dev_norm.size // elem.shape[1], 1)
    return np.mean(dev_norm.reshape(n_q, elem.shape[1], order="F"), axis=0)


def _extract_surface_for_display(dataset, *, nonlinear_subdivision: int = 4):
    kwargs = {
        "pass_pointid": True,
        "pass_cellid": True,
        "algorithm": "dataset_surface",
    }
    if nonlinear_subdivision is not None:
        kwargs["nonlinear_subdivision"] = int(nonlinear_subdivision)
    return dataset.extract_surface(**kwargs)


def _display_source_grid(dataset, *, artifacts: RunArtifacts, case_toml: Path):
    high_order = _high_order_case_grid(dataset, artifacts=artifacts, case_toml=case_toml)
    return high_order if high_order is not None else dataset


def _high_order_case_grid(dataset, *, artifacts: RunArtifacts, case_toml: Path):
    if int(_load_runtime_config(case_toml).problem.dimension) != 3 or _elem_type(case_toml) != "P4":
        return None
    try:
        case_mesh = _load_case_mesh(case_toml, artifacts=artifacts)
    except Exception:
        return None
    if not getattr(case_mesh, "cell_blocks", None):
        return None
    block = None
    for cell_type, cells in case_mesh.cell_blocks:
        if str(cell_type) == "VTK_LAGRANGE_TETRAHEDRON":
            block = np.asarray(cells, dtype=np.int64)
            break
    if block is None or block.ndim != 2 or block.shape[1] != 35:
        return None

    pv = _import_pyvista()
    points = np.asarray(dataset.points, dtype=np.float64)
    try:
        node_map = _case_to_dataset_node_map(case_mesh, points)
    except ValueError:
        return None
    mapped_block = node_map[block]
    cell_array = np.column_stack((np.full(mapped_block.shape[0], mapped_block.shape[1], dtype=np.int64), mapped_block)).reshape(-1)
    cell_types = np.full(block.shape[0], int(pv.CellType.LAGRANGE_TETRAHEDRON), dtype=np.uint8)
    high_order = pv.UnstructuredGrid(cell_array, cell_types, points)
    _copy_compatible_grid_arrays(dataset, high_order, points)
    return high_order


def _case_mesh_points(case_mesh) -> np.ndarray:
    points = getattr(case_mesh, "points", None)
    if points is not None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] == 3:
            return pts
    coord = np.asarray(case_mesh.coord, dtype=np.float64)
    pts = np.zeros((coord.shape[1], 3), dtype=np.float64)
    pts[:, : coord.shape[0]] = coord.T
    return pts


def _copy_compatible_grid_arrays(source, target, target_points: np.ndarray) -> None:
    source_points = np.asarray(getattr(source, "points", np.empty((0, 3))), dtype=np.float64)
    point_order_matches = source_points.shape == target_points.shape and np.allclose(source_points, target_points)
    if point_order_matches:
        for name in list(getattr(source, "point_data", {}).keys()):
            values = np.asarray(source.point_data[name])
            if values.shape[0] == target.n_points:
                target.point_data[name] = values

    if int(getattr(source, "n_cells", -1)) == int(target.n_cells):
        for name in list(getattr(source, "cell_data", {}).keys()):
            values = np.asarray(source.cell_data[name])
            if values.shape[0] == target.n_cells:
                target.cell_data[name] = values


def _case_to_dataset_node_map(case_mesh, dataset_points: np.ndarray) -> np.ndarray:
    case_points = _case_mesh_points(case_mesh)
    target_points = np.asarray(dataset_points, dtype=np.float64)
    if case_points.shape != target_points.shape:
        raise ValueError(f"Case mesh points {case_points.shape} do not match dataset points {target_points.shape}.")
    if np.allclose(case_points, target_points):
        return np.arange(case_points.shape[0], dtype=np.int64)

    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(target_points).query(case_points, k=1)
    except Exception:
        indices = _case_to_dataset_node_map_by_rounding(case_points, target_points)
        distances = np.linalg.norm(target_points[indices] - case_points, axis=1)

    coord_scale = max(float(np.ptp(target_points, axis=0).max(initial=0.0)), 1.0)
    tolerance = 1.0e-8 * coord_scale
    max_distance = float(np.max(distances)) if distances.size else 0.0
    if max_distance > tolerance:
        raise ValueError(f"Case-to-dataset node map failed: max coordinate mismatch {max_distance:.3e} > {tolerance:.3e}.")
    unique_count = int(np.unique(indices).size)
    if unique_count != int(indices.size):
        raise ValueError(f"Case-to-dataset node map is not one-to-one: {indices.size - unique_count} duplicate target nodes.")
    return np.asarray(indices, dtype=np.int64)


def _case_to_dataset_node_map_by_rounding(case_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    decimals = 10
    lookup = {tuple(np.round(point, decimals)): idx for idx, point in enumerate(target_points)}
    indices = np.empty(case_points.shape[0], dtype=np.int64)
    for idx, point in enumerate(case_points):
        key = tuple(np.round(point, decimals))
        if key not in lookup:
            raise ValueError(f"Case node {idx} at {point} was not found in the dataset point cloud.")
        indices[idx] = lookup[key]
    return indices


def _slice_refinement_levels(case_toml: Path) -> int:
    return 1 if _elem_type(case_toml) == "P4" else 0


def _slice_source_grid(grid, *, artifacts: RunArtifacts, case_toml: Path):
    elem_type = _elem_type(case_toml)
    if int(_load_runtime_config(case_toml).problem.dimension) != 3:
        return grid
    if elem_type == "P4":
        return _display_source_grid(grid, artifacts=artifacts, case_toml=case_toml)
    if elem_type != "P2":
        return grid
    try:
        case_mesh = _load_case_mesh(case_toml, artifacts=artifacts)
    except Exception:
        return grid
    if not case_mesh.cell_blocks:
        return grid
    cell_type, cells = case_mesh.cell_blocks[0]
    cell_block = np.asarray(cells, dtype=np.int64)
    if cell_type != "tetra10" or cell_block.ndim != 2 or cell_block.shape[1] != 10 or cell_block.shape[0] != grid.n_cells:
        return grid
    if case_mesh.points.shape != np.asarray(grid.points).shape or not np.allclose(case_mesh.points, np.asarray(grid.points)):
        return grid

    # Internal quadratic tetra order is [v0, v1, v2, v3, e01, e12, e02, e13, e23, e03].
    # VTK_QUADRATIC_TETRA expects [v0, v1, v2, v3, e01, e12, e02, e03, e13, e23].
    canonical = cell_block[:, [0, 1, 2, 3, 4, 5, 6, 9, 7, 8]]
    pv = _import_pyvista()
    n_nodes = canonical.shape[1]
    cell_array = np.column_stack((np.full(canonical.shape[0], n_nodes, dtype=np.int64), canonical)).reshape(-1)
    cell_types = np.full(canonical.shape[0], int(pv.CellType.QUADRATIC_TETRA), dtype=np.uint8)
    corrected = pv.UnstructuredGrid(cell_array, cell_types, np.asarray(grid.points, dtype=np.float64))
    for name in list(getattr(grid, "point_data", {}).keys()):
        corrected.point_data[name] = np.asarray(grid.point_data[name])
    for name in list(getattr(grid, "cell_data", {}).keys()):
        corrected.cell_data[name] = np.asarray(grid.cell_data[name])
    return corrected


def _refine_slice_for_display(dataset, source_grid, *, case_toml: Path):
    levels = _slice_refinement_levels(case_toml)
    if levels <= 0 or getattr(dataset, "n_cells", 0) == 0:
        return dataset
    refined = dataset
    for _ in range(levels):
        refined = refined.subdivide(1, subfilter="linear")
    for name in list(getattr(refined, "point_data", {}).keys()):
        del refined.point_data[name]
    for name in list(getattr(refined, "cell_data", {}).keys()):
        del refined.cell_data[name]
    return refined.sample(source_grid)


def _use_explicit_surface_builder(case_toml: Path) -> bool:
    return _elem_type(case_toml) in {"P1", "P2"}


def _build_explicit_boundary_surface(
    dataset,
    *,
    case_toml: Path,
    artifacts: RunArtifacts,
    point_array_names: tuple[str, ...] = (),
    cell_data_from_parent: dict[str, np.ndarray] | None = None,
):
    pv = _import_pyvista()
    case_mesh = _load_case_mesh(case_toml, artifacts=artifacts)
    surf = np.asarray(case_mesh.surf, dtype=np.int64)
    if surf.size == 0:
        return pv.PolyData()

    node_map = _case_to_dataset_node_map(case_mesh, np.asarray(dataset.points, dtype=np.float64))
    triangles_case, face_ids = _build_plotting_mesh_with_face_ids(_case_mesh_points(case_mesh), surf)
    triangles = node_map[triangles_case]
    used_nodes, inverse = np.unique(triangles.reshape(-1), return_inverse=True)
    local_triangles = inverse.reshape(triangles.shape)
    faces = np.column_stack((np.full(local_triangles.shape[0], 3, dtype=np.int64), local_triangles)).reshape(-1)
    surface = pv.PolyData(np.asarray(dataset.points[used_nodes], dtype=np.float64), faces)

    for name in point_array_names:
        if name not in dataset.point_data:
            continue
        surface.point_data[name] = np.asarray(dataset.point_data[name])[used_nodes]

    if cell_data_from_parent:
        parent = _surface_parent_elements(np.asarray(case_mesh.elem, dtype=np.int64), surf)
        for name, values in cell_data_from_parent.items():
            arr = np.asarray(values).reshape(-1)
            surface.cell_data[name] = arr[parent[face_ids]]
    return surface


def _surface_for_display(
    dataset,
    *,
    case_toml: Path,
    artifacts: RunArtifacts,
    nonlinear_subdivision: int = 4,
    point_array_names: tuple[str, ...] = (),
    cell_data_from_parent: dict[str, np.ndarray] | None = None,
):
    if not _use_explicit_surface_builder(case_toml):
        return _extract_surface_for_display(dataset, nonlinear_subdivision=nonlinear_subdivision)

    return _build_explicit_boundary_surface(
        dataset,
        case_toml=case_toml,
        artifacts=artifacts,
        point_array_names=point_array_names,
        cell_data_from_parent=cell_data_from_parent,
    )


def _build_parent_boundary_surface(
    dataset,
    *,
    case_toml: Path,
    artifacts: RunArtifacts,
    point_array_names: tuple[str, ...] = (),
    cell_data_from_parent: dict[str, np.ndarray] | None = None,
):
    pv = _import_pyvista()
    case_mesh = _load_case_mesh(case_toml, artifacts=artifacts)
    surf = np.asarray(case_mesh.surf, dtype=np.int64)
    if surf.size == 0:
        return pv.PolyData()

    node_map = _case_to_dataset_node_map(case_mesh, np.asarray(dataset.points, dtype=np.float64))
    faces_arr = node_map[_surface_faces_by_width(surf)[:, :3]]
    used_nodes, inverse = np.unique(faces_arr.reshape(-1), return_inverse=True)
    local_faces = inverse.reshape(faces_arr.shape)
    faces = np.column_stack((np.full(local_faces.shape[0], 3, dtype=np.int64), local_faces)).reshape(-1)
    surface = pv.PolyData(np.asarray(dataset.points[used_nodes], dtype=np.float64), faces)

    for name in point_array_names:
        if name not in dataset.point_data:
            continue
        surface.point_data[name] = np.asarray(dataset.point_data[name])[used_nodes]

    if cell_data_from_parent:
        parent = _surface_parent_elements(np.asarray(case_mesh.elem, dtype=np.int64), surf)
        for name, values in cell_data_from_parent.items():
            arr = np.asarray(values).reshape(-1)
            surface.cell_data[name] = arr[parent]
    return surface


def _add_boundary_edge_overlay(
    plotter,
    dataset,
    *,
    case_toml: Path,
    artifacts: RunArtifacts,
    warp_vector_name: str | None = None,
    warp_factor: float = 1.0,
    color: str = "#1f1f1f",
    line_width: float = 1.2,
):
    edge_surface = _build_parent_boundary_surface(
        dataset,
        case_toml=case_toml,
        artifacts=artifacts,
        point_array_names=((warp_vector_name,) if warp_vector_name is not None else ()),
    )
    keep_arrays = (warp_vector_name,) if warp_vector_name is not None else ()
    edge_surface = _optimize_display_mesh(edge_surface, keep_point_arrays=keep_arrays)
    if warp_vector_name is not None and warp_vector_name in edge_surface.point_data:
        edge_surface = edge_surface.warp_by_vector(warp_vector_name, factor=float(warp_factor))
        edge_surface = _optimize_display_mesh(edge_surface)
    edge_lines = edge_surface.extract_all_edges()
    edge_lines = _optimize_display_mesh(edge_lines)
    if edge_lines.n_cells > 0:
        plotter.add_mesh(
            edge_lines,
            color=color,
            line_width=line_width,
            lighting=False,
        )


def _decimate_display_mesh(dataset, *, reduction: float | None = None):
    if reduction is None:
        return dataset
    value = float(reduction)
    if value <= 0.0 or dataset.n_cells == 0:
        return dataset
    value = min(value, 0.99)
    triangulated = dataset.triangulate()
    return triangulated.decimate_pro(value)


def _optimize_display_mesh(
    dataset,
    *,
    keep_point_arrays: tuple[str, ...] = (),
    keep_cell_arrays: tuple[str, ...] = (),
):
    if hasattr(dataset, "points"):
        dataset.points = np.asarray(dataset.points, dtype=np.float32)
    for name in list(getattr(dataset, "point_data", {}).keys()):
        if name not in keep_point_arrays:
            del dataset.point_data[name]
            continue
        arr = np.asarray(dataset.point_data[name])
        if np.issubdtype(arr.dtype, np.floating):
            dataset.point_data[name] = arr.astype(np.float32, copy=False)
        elif np.issubdtype(arr.dtype, np.integer):
            dataset.point_data[name] = arr.astype(np.int32, copy=False)
    for name in list(getattr(dataset, "cell_data", {}).keys()):
        if name not in keep_cell_arrays:
            del dataset.cell_data[name]
            continue
        arr = np.asarray(dataset.cell_data[name])
        if np.issubdtype(arr.dtype, np.floating):
            dataset.cell_data[name] = arr.astype(np.float32, copy=False)
        elif np.issubdtype(arr.dtype, np.integer):
            dataset.cell_data[name] = arr.astype(np.int32, copy=False)
    return dataset


def _display_nonlinear_surface_subdivision(case_toml: Path, *, override: int | None = None) -> int:
    if override is not None:
        return max(0, int(override))
    metadata = load_case_metadata(case_toml)
    return max(0, int(metadata.get("nonlinear_surface_subdivision", 0)))


def _display_boundary_edge_overlay(case_toml: Path, *, override: bool | None = None) -> bool:
    if override is not None:
        return bool(override)
    metadata = load_case_metadata(case_toml)
    return bool(metadata.get("boundary_edge_overlay", False))


def _display_surface_decimate_reduction(case_toml: Path, *, override: float | None = None) -> float:
    if override is not None:
        return min(max(float(override), 0.0), 0.99)
    metadata = load_case_metadata(case_toml)
    return min(max(float(metadata.get("surface_decimate_reduction", 0.0)), 0.0), 0.99)


def _display_jupyter_backend(case_toml: Path, *, override: str | None = None) -> str:
    if override is not None:
        return str(override)
    env_override = os.environ.get("SLOPE_STABILITY_JUPYTER_BACKEND")
    if env_override is not None and str(env_override).strip():
        return str(env_override)
    metadata = load_case_metadata(case_toml)
    return str(metadata.get("jupyter_backend", "trame"))


def deviatoric_strain_norm(strain: np.ndarray, *, dim: int) -> np.ndarray:
    strain_arr = np.asarray(strain, dtype=np.float64)
    if dim == 3:
        iota = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        dev = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - np.outer(iota, iota) / 3.0
    elif dim == 2:
        iota = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        dev = np.diag([1.0, 1.0, 0.5]) - np.outer(iota, iota) / 2.0
    else:
        raise ValueError(f"Unsupported dim {dim}")
    dev_e = dev @ strain_arr
    return np.sqrt(np.maximum(0.0, np.sum(strain_arr * dev_e, axis=0)))


def _expand_display_derivatives(derivatives: tuple[np.ndarray, ...], n_q: int) -> tuple[np.ndarray, ...]:
    expanded: list[np.ndarray] = []
    for deriv in derivatives:
        arr = np.asarray(deriv, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected derivative array with shape (n_p, n_q), got {arr.shape}.")
        if arr.shape[1] == 1 and int(n_q) > 1:
            arr = np.tile(arr, (1, int(n_q)))
        expanded.append(arr)
    return tuple(expanded)


def _reference_nodes_internal_2d(elem_type: str) -> np.ndarray:
    elem_key = str(elem_type).strip().upper()
    if elem_key == "P1":
        return np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if elem_key == "P2":
        return np.array(
            [
                [0.0, 1.0, 0.0, 0.5, 0.0, 0.5],
                [0.0, 0.0, 1.0, 0.5, 0.5, 0.0],
            ],
            dtype=np.float64,
        )
    raise NotImplementedError(f"2D deviatoric display is only implemented for P1/P2, got {elem_type!r}.")


def _reference_nodes_internal_3d(elem_type: str) -> np.ndarray:
    from petsc_ssr.core.simplex_lagrange import tetra_reference_nodes

    elem_key = str(elem_type).strip().upper()
    if elem_key == "P1":
        return tetra_reference_nodes(1)
    if elem_key == "P2":
        return tetra_reference_nodes(2)
    raise NotImplementedError(f"3D deviatoric display is only implemented for P1/P2, got {elem_type!r}.")


def _evaluate_deviatoric_strain_at_reference_points(
    coord: np.ndarray,
    elem: np.ndarray,
    elem_type: str,
    displacement: np.ndarray,
    *,
    dim: int,
    xi: np.ndarray,
) -> np.ndarray:
    from petsc_ssr.fem import local_basis_volume_2d, local_basis_volume_3d

    coord_arr = np.asarray(coord, dtype=np.float64)
    elem_arr = np.asarray(elem, dtype=np.int64)
    disp_arr = np.asarray(displacement, dtype=np.float64)
    xi_arr = np.asarray(xi, dtype=np.float64)
    n_elem = elem_arr.shape[1]
    n_q = xi_arr.shape[1]

    if dim == 2:
        _, dhat1, dhat2 = local_basis_volume_2d(elem_type, xi_arr)
        dhat1, dhat2 = _expand_display_derivatives((dhat1, dhat2), n_q)
        x = coord_arr[0, elem_arr]
        y = coord_arr[1, elem_arr]
        ux = disp_arr[0, elem_arr]
        uy = disp_arr[1, elem_arr]

        x_rep = np.repeat(x, n_q, axis=1)
        y_rep = np.repeat(y, n_q, axis=1)
        ux_rep = np.repeat(ux, n_q, axis=1)
        uy_rep = np.repeat(uy, n_q, axis=1)
        dhat1_t = np.tile(dhat1, (1, n_elem))
        dhat2_t = np.tile(dhat2, (1, n_elem))

        j11 = np.sum(x_rep * dhat1_t, axis=0)
        j12 = np.sum(y_rep * dhat1_t, axis=0)
        j21 = np.sum(x_rep * dhat2_t, axis=0)
        j22 = np.sum(y_rep * dhat2_t, axis=0)
        det_j = j11 * j22 - j12 * j21
        inv_det = 1.0 / det_j

        dphi1 = (j22 * dhat1_t - j12 * dhat2_t) * inv_det
        dphi2 = (-j21 * dhat1_t + j11 * dhat2_t) * inv_det
        strain = np.vstack(
            (
                np.sum(ux_rep * dphi1, axis=0),
                np.sum(uy_rep * dphi2, axis=0),
                np.sum(ux_rep * dphi2 + uy_rep * dphi1, axis=0),
            )
        )
    elif dim == 3:
        _, dhat1, dhat2, dhat3 = local_basis_volume_3d(elem_type, xi_arr)
        dhat1, dhat2, dhat3 = _expand_display_derivatives((dhat1, dhat2, dhat3), n_q)
        x = coord_arr[0, elem_arr]
        y = coord_arr[1, elem_arr]
        z = coord_arr[2, elem_arr]
        ux = disp_arr[0, elem_arr]
        uy = disp_arr[1, elem_arr]
        uz = disp_arr[2, elem_arr]

        x_rep = np.repeat(x, n_q, axis=1)
        y_rep = np.repeat(y, n_q, axis=1)
        z_rep = np.repeat(z, n_q, axis=1)
        ux_rep = np.repeat(ux, n_q, axis=1)
        uy_rep = np.repeat(uy, n_q, axis=1)
        uz_rep = np.repeat(uz, n_q, axis=1)
        dhat1_t = np.tile(dhat1, (1, n_elem))
        dhat2_t = np.tile(dhat2, (1, n_elem))
        dhat3_t = np.tile(dhat3, (1, n_elem))

        j11 = np.sum(x_rep * dhat1_t, axis=0)
        j12 = np.sum(y_rep * dhat1_t, axis=0)
        j13 = np.sum(z_rep * dhat1_t, axis=0)
        j21 = np.sum(x_rep * dhat2_t, axis=0)
        j22 = np.sum(y_rep * dhat2_t, axis=0)
        j23 = np.sum(z_rep * dhat2_t, axis=0)
        j31 = np.sum(x_rep * dhat3_t, axis=0)
        j32 = np.sum(y_rep * dhat3_t, axis=0)
        j33 = np.sum(z_rep * dhat3_t, axis=0)

        det_j = j11 * (j22 * j33 - j23 * j32) - j12 * (j21 * j33 - j23 * j31) + j13 * (j21 * j32 - j22 * j31)
        inv_det = 1.0 / det_j

        dphi1 = ((j22 * j33 - j23 * j32) * dhat1_t - (j12 * j33 - j13 * j32) * dhat2_t + (j12 * j23 - j13 * j22) * dhat3_t) * inv_det
        dphi2 = (-(j21 * j33 - j23 * j31) * dhat1_t + (j11 * j33 - j13 * j31) * dhat2_t - (j11 * j23 - j13 * j21) * dhat3_t) * inv_det
        dphi3 = ((j21 * j32 - j22 * j31) * dhat1_t - (j11 * j32 - j12 * j31) * dhat2_t + (j11 * j22 - j12 * j21) * dhat3_t) * inv_det
        strain = np.vstack(
            (
                np.sum(ux_rep * dphi1, axis=0),
                np.sum(uy_rep * dphi2, axis=0),
                np.sum(uz_rep * dphi3, axis=0),
                np.sum(ux_rep * dphi2 + uy_rep * dphi1, axis=0),
                np.sum(uy_rep * dphi3 + uz_rep * dphi2, axis=0),
                np.sum(ux_rep * dphi3 + uz_rep * dphi1, axis=0),
            )
        )
    else:
        raise ValueError(f"Unsupported dim {dim}")

    values = deviatoric_strain_norm(strain, dim=dim)
    return values.reshape(n_elem, n_q, order="C").T


def _triangle_display_local_split(elem_type: str) -> np.ndarray:
    elem_key = str(elem_type).strip().upper()
    if elem_key == "P2":
        return np.array(
            [
                [0, 5, 4],
                [5, 1, 3],
                [4, 3, 2],
                [5, 3, 4],
            ],
            dtype=np.int64,
        )
    return np.array([[0, 1, 2]], dtype=np.int64)


def _build_discontinuous_deviatoric_plot_mesh_2d(
    coord: np.ndarray,
    elem: np.ndarray,
    elem_type: str,
    displacement: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_arr = np.asarray(coord, dtype=np.float64)
    elem_arr = np.asarray(elem, dtype=np.int64)
    values = _evaluate_deviatoric_strain_at_reference_points(
        coord_arr,
        elem_arr,
        elem_type,
        displacement,
        dim=2,
        xi=_reference_nodes_internal_2d(elem_type),
    )
    n_local = elem_arr.shape[0]
    n_elem = elem_arr.shape[1]
    local_split = _triangle_display_local_split(elem_type)
    display_coord = coord_arr[:, elem_arr.T.reshape(-1)].T
    display_values = values.T.reshape(-1)
    base = (np.arange(n_elem, dtype=np.int64) * n_local)[:, None, None]
    triangles = (base + local_split[None, :, :]).reshape(-1, 3)
    return display_coord, triangles, display_values


def _surface_faces_by_width(surf: np.ndarray) -> np.ndarray:
    surf_arr = np.asarray(surf, dtype=np.int64)
    if surf_arr.ndim != 2:
        raise ValueError(f"Expected a 2D surface array, got shape {surf_arr.shape}")
    if surf_arr.shape[0] == 6:
        return surf_arr.T.astype(np.int64)
    if surf_arr.shape[1] == 6:
        return surf_arr.astype(np.int64)
    if surf_arr.shape[0] == 15:
        return surf_arr[:3, :].T.astype(np.int64)
    if surf_arr.shape[1] == 15:
        return surf_arr[:, :3].astype(np.int64)
    if surf_arr.shape[0] == 3:
        return surf_arr.T.astype(np.int64)
    if surf_arr.shape[1] == 3:
        return surf_arr.astype(np.int64)
    raise ValueError(f"Unsupported surface array shape {surf_arr.shape}")


def _surface_display_local_split(n_face_nodes: int) -> np.ndarray:
    if int(n_face_nodes) == 6:
        # Internal quadratic triangle order is [v0, v1, v2, e12, e20, e01].
        return np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2], [5, 3, 4]], dtype=np.int64)
    if int(n_face_nodes) == 3:
        return np.array([[0, 1, 2]], dtype=np.int64)
    raise ValueError(f"Unsupported surface face width {n_face_nodes}")


def _canonical_surface_faces_for_display(coord: np.ndarray, surf: np.ndarray) -> np.ndarray:
    surf_faces = _surface_faces_by_width(surf)
    if surf_faces.shape[1] != 6:
        return surf_faces

    coord_arr = np.asarray(coord, dtype=np.float64)
    if coord_arr.ndim != 2:
        raise ValueError(f"Expected coordinate array with shape (dim, n_nodes) or (n_nodes, dim), got {coord_arr.shape}.")
    if coord_arr.shape[0] > coord_arr.shape[1]:
        coord_arr = coord_arr.T
    if coord_arr.shape[0] not in {2, 3}:
        raise ValueError(f"Unsupported coordinate shape {coord_arr.shape} for quadratic surface canonicalization.")
    try:
        return _canonicalize_triangle6_vtu_elem(coord_arr, surf_faces.T).T
    except ValueError:
        return surf_faces


def _build_plotting_mesh_with_face_ids(
    coord_or_surf: np.ndarray,
    surf: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if surf is None:
        coord = None
        surf_arr = coord_or_surf
    else:
        coord = np.asarray(coord_or_surf, dtype=np.float64)
        surf_arr = surf
    surf_faces = _surface_faces_by_width(surf_arr) if coord is None else _canonical_surface_faces_for_display(coord, surf_arr)
    if surf_faces.shape[1] != 6:
        tri = surf_faces.astype(np.int64)
        face_ids = np.arange(tri.shape[0], dtype=np.int64)
        return tri, face_ids

    split = _surface_display_local_split(surf_faces.shape[1])
    triangles: list[np.ndarray] = []
    face_ids: list[int] = []
    for face_id, face in enumerate(surf_faces):
        for local in split:
            triangles.append(face[local])
            face_ids.append(face_id)
    return np.asarray(triangles, dtype=np.int64), np.asarray(face_ids, dtype=np.int64)


def _surface_parent_elements(elem: np.ndarray, surf: np.ndarray) -> np.ndarray:
    tet = np.asarray(elem, dtype=np.int64)
    faces = _surface_faces_by_width(surf)
    if tet.ndim != 2 or tet.shape[0] < 4:
        raise ValueError(f"Expected tetrahedral connectivity, got shape {tet.shape}")
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise ValueError(f"Expected triangular faces, got shape {faces.shape}")

    lookup: dict[tuple[int, int, int], int] = {}
    local_faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    corner_tet = tet[:4, :]
    for elem_id in range(corner_tet.shape[1]):
        nodes = corner_tet[:, elem_id]
        for local in local_faces:
            key = tuple(sorted(int(nodes[idx]) for idx in local))
            lookup[key] = elem_id

    parent = np.empty(faces.shape[0], dtype=np.int64)
    for face_id, face in enumerate(faces):
        key = tuple(sorted(int(v) for v in face[:3]))
        if key not in lookup:
            raise KeyError(f"Boundary face {key} was not found in any tetrahedron.")
        parent[face_id] = lookup[key]
    return parent


def _surface_parent_local_node_indices(coord: np.ndarray, elem: np.ndarray, surf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    elem_arr = np.asarray(elem, dtype=np.int64)
    surf_faces = _canonical_surface_faces_for_display(coord, surf)
    parents = _surface_parent_elements(elem_arr, surf)
    local_ids = np.empty_like(surf_faces)
    for face_id, parent_id in enumerate(parents):
        parent_nodes = elem_arr[:, int(parent_id)]
        lookup = {int(node): idx for idx, node in enumerate(parent_nodes)}
        for local_face_id, node in enumerate(surf_faces[face_id]):
            local_ids[face_id, local_face_id] = lookup[int(node)]
    return parents, local_ids


def _build_discontinuous_deviatoric_surface_3d(
    coord: np.ndarray,
    elem: np.ndarray,
    surf: np.ndarray,
    elem_type: str,
    displacement: np.ndarray,
):
    pv = _import_pyvista()
    surf_faces = _canonical_surface_faces_for_display(coord, surf)
    if surf_faces.size == 0:
        return pv.PolyData()

    nodal_values = _evaluate_deviatoric_strain_at_reference_points(
        np.asarray(coord, dtype=np.float64),
        np.asarray(elem, dtype=np.int64),
        elem_type,
        np.asarray(displacement, dtype=np.float64),
        dim=3,
        xi=_reference_nodes_internal_3d(elem_type),
    )
    parents, local_ids = _surface_parent_local_node_indices(
        np.asarray(coord, dtype=np.float64),
        np.asarray(elem, dtype=np.int64),
        np.asarray(surf, dtype=np.int64),
    )
    face_values = nodal_values[local_ids, parents[:, None]]

    n_faces, n_face_nodes = surf_faces.shape
    local_split = _surface_display_local_split(n_face_nodes)
    display_points = np.asarray(coord, dtype=np.float64)[:, surf_faces.reshape(-1)].T
    base = (np.arange(n_faces, dtype=np.int64) * n_face_nodes)[:, None, None]
    triangles = (base + local_split[None, :, :]).reshape(-1, 3)
    faces = np.column_stack((np.full(triangles.shape[0], 3, dtype=np.int64), triangles)).reshape(-1)
    surface = pv.PolyData(display_points, faces)
    surface.point_data["deviatoric_strain"] = np.asarray(face_values, dtype=np.float64).reshape(-1)
    return surface


def summarize_sections(sections: dict[str, dict[str, Any]], materials: list[dict[str, Any]]) -> str:
    lines = ["Editable runtime sections:"]
    for name in RUNTIME_SECTION_ORDER:
        section = sections.get(name, {})
        if section:
            lines.append(f"- {name}: {', '.join(sorted(section))}")
    if materials:
        lines.append(f"- materials: {len(materials)} entries")
    return "\n".join(lines)


def _load_runtime_config(case_toml: Path):
    from petsc_ssr.core.run_config import load_run_case_config

    case_toml = Path(case_toml).resolve()
    try:
        return load_run_case_config(case_toml)
    except Exception:
        # Some committed historical artifacts still carry legacy generated configs
        # that predate the canonical asset-based run contract. When a notebook is
        # pointed at such an artifact, fall back to the owning benchmark case.
        if case_toml.name == "generated_case.toml":
            benchmark_case = case_toml.parents[2] / "case.toml"
            if benchmark_case.exists():
                return load_run_case_config(benchmark_case)
        raise


def _load_case_mesh(case_toml: Path, *, artifacts: RunArtifacts | None = None):
    from petsc_ssr.postprocess import rebuild_case_mesh

    cfg = _load_runtime_config(case_toml)
    return rebuild_case_mesh(cfg, mpi_size=_artifacts_mpi_size(artifacts))


def _artifacts_mpi_size(artifacts: RunArtifacts | None) -> int:
    if artifacts is None:
        return 1
    run_info = dict(artifacts.run_info.get("run_info", {}))
    mpi_size = run_info.get("mpi_size", 1)
    try:
        return max(int(mpi_size), 1)
    except (TypeError, ValueError):
        return 1


def _seepage_ssr_node_permutation(case_toml: Path, artifacts: RunArtifacts) -> np.ndarray | None:
    from petsc_ssr.mesh import reorder_mesh_nodes
    from petsc_ssr.problem_asset_runtime import build_mesh_for_resolved_asset, resolve_problem_asset_from_config

    cfg = _load_runtime_config(case_toml)
    if int(getattr(cfg.problem, "dimension", 0)) != 3:
        return None
    if str(getattr(cfg.problem, "analysis", "")).strip().lower() != "ssr":
        return None

    resolved = resolve_problem_asset_from_config(cfg)
    if "seepage" not in resolved.definition.capabilities:
        return None

    part_count = _artifacts_mpi_size(artifacts) if cfg.execution.node_ordering.lower() == "block_metis" else None
    mesh = build_mesh_for_resolved_asset(resolved, elem_type=cfg.problem.elem_type)
    reordered = reorder_mesh_nodes(
        mesh.coord,
        mesh.elem,
        mesh.surf,
        mesh.q_mask,
        strategy=cfg.execution.node_ordering,
        n_parts=part_count,
    )
    return np.asarray(reordered.permutation, dtype=np.int64)


def _linear_triangles_2d(elem: np.ndarray, elem_type: str) -> tuple[np.ndarray, np.ndarray]:
    elem_arr = np.asarray(elem, dtype=np.int64)
    n_elem = elem_arr.shape[1]
    if elem_type == "P2":
        local = _triangle_display_local_split(elem_type)
        base = (np.arange(n_elem, dtype=np.int64) * elem_arr.shape[0])[:, None, None]
        flat_tri = (base + local[None, :, :]).reshape(-1, 3)
        flat_nodes = elem_arr.T.reshape(-1)
        tris = flat_nodes[flat_tri]
        parents = np.repeat(np.arange(n_elem, dtype=np.int64), local.shape[0])
        return tris, parents
    tris = elem_arr[:3, :].T.copy()
    parents = np.arange(n_elem, dtype=np.int64)
    return tris, parents


def _parent_triangles_2d(elem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    elem_arr = np.asarray(elem, dtype=np.int64)
    return elem_arr[:3, :].T.copy(), np.arange(elem_arr.shape[1], dtype=np.int64)


def _point_field(vtu: VtuData, name: str, *, default: np.ndarray | None = None) -> np.ndarray:
    if name in vtu.point_data:
        return np.asarray(vtu.point_data[name])
    if default is not None:
        return np.asarray(default)
    raise KeyError(f"Point field {name!r} not found in VTU export")


def _cell_field(vtu: VtuData, name: str) -> np.ndarray:
    if name not in vtu.cell_data:
        raise KeyError(f"Cell field {name!r} not found in VTU export")
    return np.asarray(vtu.cell_data[name])


def _seal_2d_triangle_artist(artist):
    # Notebook backends can show white seams between adjacent triangles even
    # when the connectivity is correct. Disable collection antialiasing and
    # edge drawing so the 2D fills render watertight.
    for setter in ("set_antialiased", "set_antialiaseds"):
        if hasattr(artist, setter):
            try:
                getattr(artist, setter)(False)
            except TypeError:
                pass
    if hasattr(artist, "set_linewidth"):
        try:
            artist.set_linewidth(0.0)
        except TypeError:
            pass
    if hasattr(artist, "set_edgecolor"):
        try:
            artist.set_edgecolor("none")
        except (TypeError, ValueError):
            pass
    if hasattr(artist, "set_snap"):
        try:
            artist.set_snap(True)
        except TypeError:
            pass
    if hasattr(artist, "set_rasterized"):
        try:
            artist.set_rasterized(True)
        except TypeError:
            pass
    return artist


def _saturation_field(
    artifacts: RunArtifacts,
    *,
    vtu: VtuData | None = None,
    n_cells: int | None = None,
) -> np.ndarray:
    if vtu is not None and "saturation" in vtu.cell_data:
        saturation = np.asarray(vtu.cell_data["saturation"], dtype=np.float64).reshape(-1)
    else:
        saturation = None
        for key in ("saturation", "mater_sat", "seepage_mater_sat"):
            if key in artifacts.npz:
                saturation = np.asarray(artifacts.npz[key], dtype=np.float64).reshape(-1)
                break
        if saturation is None:
            raise KeyError("No saturation field available in artifacts or VTU export")

    if n_cells is not None and saturation.size != int(n_cells):
        raise ValueError(f"Saturation field size {saturation.size} does not match cell count {int(n_cells)}")
    return saturation


def _displacement_field(artifacts: RunArtifacts, vtu: VtuData, *, dim: int) -> np.ndarray:
    if "displacement" in vtu.point_data:
        return np.asarray(vtu.point_data["displacement"], dtype=np.float64)
    U = np.asarray(artifacts.npz["U"], dtype=np.float64)
    disp = np.zeros((U.shape[1], 3), dtype=np.float64)
    disp[:, :dim] = U.T
    return disp


def _pore_pressure_field(
    artifacts: RunArtifacts,
    case_toml: Path,
    *,
    vtu: VtuData | None = None,
) -> np.ndarray:
    for key in ("pore_pressure_export", "pw_export", "seepage_pw_reordered", "pw_reordered"):
        if key in artifacts.npz:
            return np.asarray(artifacts.npz[key], dtype=np.float64).reshape(-1)

    if vtu is not None and "pore_pressure" in vtu.point_data:
        return _point_field(vtu, "pore_pressure")

    raw = None
    for key in ("pw", "seepage_pw"):
        if key in artifacts.npz:
            raw = np.asarray(artifacts.npz[key], dtype=np.float64).reshape(-1)
            break
    if raw is None:
        if vtu is not None:
            return _point_field(vtu, "pore_pressure")
        raise KeyError("No pore-pressure field available in artifacts or VTU export")

    perm = _seepage_ssr_node_permutation(case_toml, artifacts)
    if perm is not None and perm.size == raw.size:
        return raw[perm]
    return raw


def _categorical_cmap(
    palette: dict[int, tuple[float, float, float]],
    values: np.ndarray,
) -> tuple[mcolors.ListedColormap, mcolors.BoundaryNorm]:
    distinct = sorted({int(v) for v in np.asarray(values).reshape(-1)})
    colors = [palette.get(value, (0.8, 0.8, 0.8)) for value in distinct]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.asarray(distinct + [distinct[-1] + 1], dtype=np.float64) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _resolve_section_paths(case_toml: Path, data: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_section_paths(case_toml, value)
        elif isinstance(value, list):
            resolved[key] = list(value)
        elif isinstance(value, str) and (key.endswith("_path") or key.endswith("_dir")):
            path = Path(value)
            resolved[key] = path if path.is_absolute() else (case_toml.parent / path).resolve()
        else:
            resolved[key] = value
    return resolved


def _profile_sections(case_toml: Path, sections: dict[str, dict[str, Any]], execution_profile: str) -> dict[str, dict[str, Any]]:
    profile = str(execution_profile).strip().lower()
    cloned = {name: dict(value) for name, value in sections.items()}
    if profile in {"benchmark", "full"}:
        return cloned
    if profile != "smoke":
        raise ValueError(f"Unsupported execution profile {execution_profile!r}")

    if "case" in cloned or "mesh" in cloned or "linear" in cloned:
        continuation = dict(cloned.get("continuation", {}))
        linear = dict(cloned.get("linear", {}))
        output = dict(cloned.get("output", {}))
        if continuation:
            continuation["step_max"] = min(int(continuation.get("step_max", 100)), 2)
            cloned["continuation"] = continuation
        if linear:
            linear["max_iterations"] = min(int(linear.get("max_iterations", 100)), 120)
            cloned["linear"] = linear
        output["solution"] = ["vtu", "petscbin"]
        output["history"] = ["curve_csv"]
        cloned["output"] = output
        return cloned

    problem = dict(cloned.get("problem", {}))
    continuation = dict(cloned.get("continuation", {}))
    linear_solver = dict(cloned.get("linear_solver", {}))
    seepage = dict(cloned.get("seepage", {}))
    execution = dict(cloned.get("execution", {}))
    export = dict(cloned.get("export", default_export_section()))
    case_id = str(problem.get("case", "")).lower()
    asset_id = str(problem.get("asset", "")).lower()
    analysis = str(problem.get("analysis", "")).lower()
    benchmark_name = case_toml.parent.name.lower()

    if continuation:
        continuation["step_max"] = min(int(continuation.get("step_max", 100)), 2)
        cloned["continuation"] = continuation
    if seepage:
        seepage["linear_max_iter"] = min(int(seepage.get("linear_max_iter", 500)), 300)
        cloned["seepage"] = seepage
    if linear_solver:
        linear_solver["max_iterations"] = min(int(linear_solver.get("max_iterations", 100)), 120)
        linear_solver["threads"] = 1
        cloned["linear_solver"] = linear_solver
    if execution:
        execution["mpi_distribute_by_nodes"] = bool(execution.get("mpi_distribute_by_nodes", True))
        cloned["execution"] = execution
    export["write_custom_debug_bundle"] = False
    export["write_history_json"] = False
    export["write_solution_vtu"] = True
    cloned["export"] = export
    if analysis == "seepage" and "linear_solver" in cloned:
        cloned["linear_solver"]["max_iterations"] = min(int(cloned["linear_solver"].get("max_iterations", 500)), 300)
    if case_id in {
        "2d-franz-dam-ssr",
        "2d-kozinec-ll",
        "2d-kozinec-ssr",
        "2d-luzec-ssr",
    } or (
        asset_id in {"2d_franz_dam", "2d_kozinec", "2d_luzec"}
        and analysis in {"ll", "ssr"}
    ):
        execution = dict(cloned.get("execution", {}))
        execution["mpi_distribute_by_nodes"] = False
        execution["constitutive_mode"] = "global"
        cloned["execution"] = execution
        linear_solver = dict(cloned.get("linear_solver", {}))
        linear_solver["solver_type"] = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE"
        cloned["linear_solver"] = linear_solver
        problem = dict(cloned.get("problem", {}))
        problem["elem_type"] = "P1"
        cloned["problem"] = problem
    if benchmark_name in {
        "3d-heterogeneous-ll",
        "3d-homogeneous-ll",
    }:
        execution = dict(cloned.get("execution", {}))
        execution["mpi_distribute_by_nodes"] = False
        execution["constitutive_mode"] = "global"
        cloned["execution"] = execution
        linear_solver = dict(cloned.get("linear_solver", {}))
        linear_solver["solver_type"] = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE"
        cloned["linear_solver"] = linear_solver
    return cloned


def _profile_mpi_ranks(metadata: dict[str, Any], execution_profile: str) -> int:
    profile = str(execution_profile).strip().lower()
    if profile == "smoke":
        return 1
    env_override = _env_int("SLOPE_STABILITY_MPI_RANKS")
    if env_override is not None:
        return env_override
    configured = int(metadata.get("mpi_ranks", 8))
    if _codespaces_active():
        cpu_count = os.cpu_count() or 1
        return max(1, min(configured, int(cpu_count)))
    return configured


def _import_pyvista():
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
    os.environ.setdefault("PYVISTA_TRAME_SERVER_PROXY_ENABLED", "true")
    if os.environ.get("CODESPACES") == "true":
        os.environ.setdefault("PYVISTA_TRAME_SERVER_PROXY_PREFIX", "/proxy/{port}/")
    import pyvista as pv

    return pv


def _codespaces_backend_autoswitch(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if not _codespaces_active():
        return normalized
    if normalized in {"client", "server", "trame", "auto", ""}:
        return "html"
    return normalized


def _new_plotter(pv, *, title: str):
    plotter = pv.Plotter(notebook=True)
    plotter.add_title(title, font_size=12)
    return plotter


def _show_plotter(plotter, case_toml: Path | None = None, *, jupyter_backend: str | None = None):
    backend = jupyter_backend
    if backend is None and case_toml is not None:
        backend = _display_jupyter_backend(case_toml)
    if backend is None:
        backend = "trame" if _module_available("ipywidgets") and _module_available("trame") else "static"
    backend = _codespaces_backend_autoswitch(str(backend))
    if backend == "html":
        try:
            exported = plotter.export_html(None)
            html_doc = exported.getvalue() if hasattr(exported, "getvalue") else str(exported)
            iframe = (
                '<iframe '
                'style="width: 100%; height: 720px; border: 0;" '
                f'srcdoc="{html_escape(html_doc, quote=True)}"></iframe>'
            )
            return InlineHtml(iframe)
        finally:
            plotter.close()
    return plotter.show(jupyter_backend=backend)


def _apply_matlab_camera(plotter) -> None:
    plotter.camera.parallel_projection = True
    plotter.view_vector((0.5, 1.0, -2.0), viewup=(0.0, 0.0, 1.0))


def _elem_type(case_toml: Path) -> str:
    return str(load_case_document(case_toml).get("problem", {}).get("elem_type", "P2")).upper()


def _drain_progress(progress_path: Path, offset: int) -> int:
    if not progress_path.exists():
        return offset
    with progress_path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("event", "progress"))
            parts = [f"[rank0-progress] {event_type}"]
            for key in ("accepted_steps", "lambda", "omega", "u_max", "wall", "target_step", "success"):
                if key in event:
                    parts.append(f"{key}={event[key]}")
            print(" ".join(parts), flush=True)
        return handle.tell()


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Path):
        return json.dumps(str(value))
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {value!r}")


def notebook_intro_markdown(metadata: dict[str, Any]) -> str:
    title = metadata["title"]
    kind = metadata["comparison_kind"] or "benchmark"
    family = metadata["family"]
    return dedent(
        f"""
        # {title}

        This notebook is generated from the shared benchmark notebook framework.

        - Benchmark folder: `{metadata["benchmark_name"]}`
        - Comparison kind: `{kind}`
        - Notebook family: `{family}`
        """
    ).strip()
