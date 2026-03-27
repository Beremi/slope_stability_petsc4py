from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from queue import Empty, Queue
from textwrap import dedent
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.tri as mtri
import meshio
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"
DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULT_MPIEXEC = shutil.which("mpiexec") or "mpiexec"
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RUNTIME_SECTION_ORDER = (
    "problem",
    "geometry",
    "case_data",
    "execution",
    "continuation",
    "newton",
    "linear_solver",
    "seepage",
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


def benchmark_case_tomls(root: Path = BENCHMARKS_DIR) -> list[Path]:
    return sorted(path for path in root.glob("*/case.toml") if path.is_file())


def load_case_document(case_toml: Path) -> dict[str, Any]:
    return tomllib.loads(Path(case_toml).read_text(encoding="utf-8"))


def load_case_metadata(case_toml: Path) -> dict[str, Any]:
    case_toml = Path(case_toml).resolve()
    raw = load_case_document(case_toml)
    benchmark = dict(raw.get("benchmark", {}))
    notebook = dict(raw.get("notebook", {}))
    problem = dict(raw.get("problem", {}))
    return {
        "case_toml": case_toml,
        "case_dir": case_toml.parent,
        "benchmark_name": case_toml.parent.name,
        "benchmark": benchmark,
        "notebook": notebook,
        "problem": problem,
        "title": str(benchmark.get("title", case_toml.parent.name)),
        "matlab_script": str(benchmark.get("matlab_script", "")),
        "comparison_kind": str(benchmark.get("comparison_kind", "")).lower(),
        "mpi_ranks": int(benchmark.get("mpi_ranks", 8)),
        "family": str(notebook.get("family", "")),
    }


def load_case_sections(case_toml: Path) -> dict[str, dict[str, Any]]:
    case_toml = Path(case_toml).resolve()
    raw = load_case_document(case_toml)
    sections: dict[str, dict[str, Any]] = {}
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
        lines.append(f"[{section_name}]")
        for key, value in section.items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    for material in materials or []:
        lines.append("[[materials]]")
        for key, value in material.items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


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
        "-n",
        str(int(mpi_ranks)),
        str(python_executable),
        "-m",
        "slope_stability.cli.run_case_from_config",
        str(config_path),
        "--out_dir",
        str(out_dir),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
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
    if return_code != 0:
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


def _vtu_internal_elem_2d(elem: np.ndarray, elem_type: str) -> np.ndarray:
    elem_arr = np.asarray(elem, dtype=np.int64)
    if elem_type == "P2":
        # VTU triangle6 order is [v0, v1, v2, e01, e12, e20];
        # the FEM operators expect [v0, v1, v2, e12, e20, e01].
        return elem_arr[[0, 1, 2, 4, 5, 3], :]
    return elem_arr


def _vtu_linear_triangles_2d(vtu: VtuData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    coord, elem, elem_type = _vtu_2d_topology(vtu)
    if elem_type == "P2":
        n_elem = elem.shape[1]
        triangles = np.empty((n_elem * 4, 3), dtype=np.int64)
        parents = np.repeat(np.arange(n_elem, dtype=np.int64), 4)
        e01 = elem[3, :]
        e12 = elem[4, :]
        e20 = elem[5, :]
        triangles[0::4, :] = np.stack((elem[0, :], e01, e20), axis=1)
        triangles[1::4, :] = np.stack((e01, elem[1, :], e12), axis=1)
        triangles[2::4, :] = np.stack((e20, e12, elem[2, :]), axis=1)
        triangles[3::4, :] = np.stack((e01, e12, e20), axis=1)
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
    timings = dict(artifacts.run_info.get("timings", {}))
    constitutive = dict(timings.get("constitutive", {}))
    linear = dict(timings.get("linear", {}))
    series = {
        "Constitutive": float(sum(float(v) for v in constitutive.values())),
        "Linear solve": float(linear.get("attempt_linear_solve_time_total", 0.0)),
        "PC apply": float(linear.get("preconditioner_apply_time_total", 0.0)),
        "PC setup": float(linear.get("preconditioner_setup_time_total", 0.0)),
        "Orthogonalization": float(linear.get("attempt_linear_orthogonalization_time_total", 0.0)),
    }
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
    triangles, _ = _linear_triangles_2d(case_mesh.elem, _elem_type(case_toml))
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
    triangles, parents = _linear_triangles_2d(case_mesh.elem, _elem_type(case_toml))
    values = case_mesh.material_id[parents]
    triang = mtri.Triangulation(case_mesh.coord[0], case_mesh.coord[1], triangles)
    cmap, norm = _categorical_cmap(get_material_palette(palette_name), values)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=160)
    artist = ax.tripcolor(triang, facecolors=values, cmap=cmap, norm=norm, edgecolors="k", linewidth=0.15)
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
    triang = mtri.Triangulation(coord[0], coord[1], triangles)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    if pore_pressure.size == coord.shape[1]:
        artist = ax.tripcolor(triang, pore_pressure, shading="gouraud", cmap=PARULA_EQUIV)
    elif pore_pressure.size == elem.shape[1]:
        artist = ax.tripcolor(triang, facecolors=pore_pressure[parents], cmap=PARULA_EQUIV, edgecolors="none")
    else:
        raise ValueError(
            f"Pore-pressure field size {pore_pressure.size} does not match VTU point count {coord.shape[1]} "
            f"or cell count {elem.shape[1]}"
        )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Pore pressure [kPa]")
    fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    return fig


def plot_2d_saturation(artifacts: RunArtifacts, case_toml: Path):
    vtu = load_vtu(artifacts.vtu_path)
    coord, triangles, parents, elem, _ = _vtu_linear_triangles_2d(vtu)
    saturation = _saturation_field(artifacts, vtu=vtu, n_cells=elem.shape[1])
    values = saturation[parents]
    triang = mtri.Triangulation(coord[0], coord[1], triangles)
    cmap, norm = _categorical_cmap(SATURATION_PALETTE, values)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    artist = ax.tripcolor(triang, facecolors=values, cmap=cmap, norm=norm, edgecolors="k", linewidth=0.1)
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
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"Displacement magnitude (warp scale = {scale:.4g})")
    fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    return fig


def plot_2d_deviatoric_strain(artifacts: RunArtifacts, case_toml: Path):
    vtu = load_vtu(artifacts.vtu_path)
    coord, triangles, parents, elem, elem_type = _vtu_linear_triangles_2d(vtu)
    displacement = _displacement_field(artifacts, vtu, dim=2)[:, :2].T
    values = compute_element_deviatoric_strain(
        coord,
        _vtu_internal_elem_2d(elem, elem_type),
        elem_type,
        displacement,
        dim=2,
    )
    triang = mtri.Triangulation(coord[0], coord[1], triangles)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    artist = ax.tripcolor(triang, facecolors=values[parents], cmap=PARULA_EQUIV, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Deviatoric strain norm")
    fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    return fig


def show_3d_mesh_view(artifacts: RunArtifacts, case_toml: Path):
    if not _module_available("pyvista"):
        return viz_support_message()
    pv = _import_pyvista()
    grid = pv.read(artifacts.vtu_path)
    surface = grid.extract_surface(algorithm="dataset_surface")
    plotter = _new_plotter(pv, title="Mesh outline")
    plotter.add_mesh(surface, color="white", show_edges=True, edge_color="#2a62d0")
    _apply_matlab_camera(plotter)
    return _show_plotter(plotter)


def show_3d_pore_pressure_view(artifacts: RunArtifacts, case_toml: Path):
    if not _module_available("pyvista"):
        return viz_support_message()
    pv = _import_pyvista()
    grid = pv.read(artifacts.vtu_path)
    grid.point_data["pore_pressure"] = _pore_pressure_field(artifacts, case_toml)
    surface = grid.extract_surface(algorithm="dataset_surface")
    plotter = _new_plotter(pv, title="Pore pressure [kPa]")
    plotter.add_mesh(surface, scalars="pore_pressure", cmap=PARULA_EQUIV, show_edges=False)
    _apply_matlab_camera(plotter)
    return _show_plotter(plotter)


def show_3d_saturation_view(artifacts: RunArtifacts, case_toml: Path):
    if not _module_available("pyvista"):
        return viz_support_message()
    pv = _import_pyvista()
    grid = pv.read(artifacts.vtu_path)
    saturation = _saturation_field(artifacts, n_cells=int(grid.n_cells))
    plotter = _new_plotter(pv, title="Saturation")
    legend_entries: list[list[str]] = []
    value_labels = {0.0: "unsaturated", 1.0: "saturated"}
    for value in sorted(float(v) for v in np.unique(saturation)):
        cell_ids = np.flatnonzero(np.isclose(saturation, value))
        if cell_ids.size == 0:
            continue
        region = grid.extract_cells(cell_ids)
        surface = region.extract_surface(pass_cellid=True, algorithm="dataset_surface")
        if surface.n_cells == 0:
            continue
        color = SATURATION_PALETTE.get(int(round(value)), (0.8, 0.8, 0.8))
        plotter.add_mesh(
            surface,
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
    _apply_matlab_camera(plotter)
    return _show_plotter(plotter)


def show_3d_displacement_view(artifacts: RunArtifacts, case_toml: Path, *, warp_scale: float | None = None):
    if not _module_available("pyvista"):
        return viz_support_message()
    pv = _import_pyvista()
    grid = pv.read(artifacts.vtu_path)
    displacement = np.asarray(artifacts.npz["U"], dtype=np.float64)
    case_mesh = _load_case_mesh(case_toml, artifacts=artifacts)
    scale = matlab_warp_scale(case_mesh.coord, displacement) if warp_scale is None else float(warp_scale)
    surface = grid.extract_surface(algorithm="dataset_surface").warp_by_vector("displacement", factor=scale)
    plotter = _new_plotter(pv, title=f"Displacement magnitude (warp scale = {scale:.4g})")
    plotter.add_mesh(surface, scalars="displacement_magnitude", cmap=PARULA_EQUIV, show_edges=False)
    _apply_matlab_camera(plotter)
    return _show_plotter(plotter)


def show_3d_deviatoric_surface_view(artifacts: RunArtifacts, case_toml: Path):
    if not _module_available("pyvista"):
        return viz_support_message()
    pv = _import_pyvista()
    cfg = _load_runtime_config(case_toml)
    case_mesh = _load_case_mesh(case_toml, artifacts=artifacts)
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
    triangles, face_ids = _build_plotting_mesh_with_face_ids(np.asarray(case_mesh.surf, dtype=np.int64))
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
    _apply_matlab_camera(plotter)
    return _show_plotter(plotter)


def show_3d_deviatoric_slices(
    artifacts: RunArtifacts,
    case_toml: Path,
    *,
    slice_planes_x: list[float] | None = None,
    slice_planes_y: list[float] | None = None,
    slice_planes_z: list[float] | None = None,
    clim_scale_max: float | None = None,
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
    grid = pv.read(artifacts.vtu_path)
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
            slc = point_grid.slice(normal=normal, origin=origin)
            if slc.n_points == 0:
                continue
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
    return _show_plotter(plotter)


def compute_element_deviatoric_strain(
    coord: np.ndarray,
    elem: np.ndarray,
    elem_type: str,
    displacement: np.ndarray,
    *,
    dim: int,
) -> np.ndarray:
    from slope_stability.fem import assemble_strain_operator

    assembly = assemble_strain_operator(coord, elem, elem_type, dim=dim)
    strain = assembly.B @ np.asarray(displacement, dtype=np.float64).reshape(-1, order="F")
    strain = strain.reshape(assembly.n_strain, -1, order="F")
    dev_norm = deviatoric_strain_norm(strain, dim=dim)
    n_q = max(dev_norm.size // elem.shape[1], 1)
    return np.mean(dev_norm.reshape(n_q, elem.shape[1], order="F"), axis=0)


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


def _build_plotting_mesh_with_face_ids(surf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    surf_faces = _surface_faces_by_width(surf)
    if surf_faces.shape[1] != 6:
        tri = surf_faces.astype(np.int64)
        face_ids = np.arange(tri.shape[0], dtype=np.int64)
        return tri, face_ids

    split = np.array([[0, 3, 5], [3, 1, 4], [3, 4, 5], [5, 4, 2]], dtype=np.int64)
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
    from slope_stability.core.run_config import load_run_case_config

    return load_run_case_config(case_toml)


def _load_case_mesh(case_toml: Path, *, artifacts: RunArtifacts | None = None):
    from slope_stability.postprocess import rebuild_case_mesh

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


def _comsol_ssr_node_permutation(case_toml: Path, artifacts: RunArtifacts) -> np.ndarray | None:
    from slope_stability.mesh import load_mesh_p2_comsol, reorder_mesh_nodes

    cfg = _load_runtime_config(case_toml)
    if cfg.problem.case not in {"3d_hetero_seepage_ssr_comsol", "3d_homo_seepage_ssr"}:
        return None
    part_count = _artifacts_mpi_size(artifacts) if cfg.execution.node_ordering.lower() == "block_metis" else None
    mesh = load_mesh_p2_comsol(cfg.problem.mesh_path, boundary_type=1)
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
        tris = np.empty((n_elem * 4, 3), dtype=np.int64)
        parents = np.repeat(np.arange(n_elem, dtype=np.int64), 4)
        e01 = elem_arr[5, :]
        e12 = elem_arr[3, :]
        e20 = elem_arr[4, :]
        tris[0::4, :] = np.stack((elem_arr[0, :], e01, e20), axis=1)
        tris[1::4, :] = np.stack((e01, elem_arr[1, :], e12), axis=1)
        tris[2::4, :] = np.stack((e20, e12, elem_arr[2, :]), axis=1)
        tris[3::4, :] = np.stack((e01, e12, e20), axis=1)
        return tris, parents
    tris = elem_arr[:3, :].T.copy()
    parents = np.arange(n_elem, dtype=np.int64)
    return tris, parents


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

    cfg = _load_runtime_config(case_toml)
    if cfg.problem.case in {"3d_hetero_seepage_ssr_comsol", "3d_homo_seepage_ssr"}:
        perm = _comsol_ssr_node_permutation(case_toml, artifacts)
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

    problem = dict(cloned.get("problem", {}))
    continuation = dict(cloned.get("continuation", {}))
    linear_solver = dict(cloned.get("linear_solver", {}))
    seepage = dict(cloned.get("seepage", {}))
    execution = dict(cloned.get("execution", {}))
    case_id = str(problem.get("case", "")).lower()
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
    if str(problem.get("analysis", "")).lower() == "seepage" and "linear_solver" in cloned:
        cloned["linear_solver"]["max_iterations"] = min(int(cloned["linear_solver"].get("max_iterations", 500)), 300)
    if case_id in {
        "2d_franz_dam_ssr",
        "2d_kozinec_ll",
        "2d_kozinec_ssr",
        "2d_luzec_ssr",
    }:
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
        "run_3d_hetero_ll",
        "run_3d_homo_ll",
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
    return int(metadata.get("mpi_ranks", 8))


def _import_pyvista():
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
    import pyvista as pv

    return pv


def _new_plotter(pv, *, title: str):
    plotter = pv.Plotter(notebook=True)
    plotter.add_title(title, font_size=12)
    return plotter


def _show_plotter(plotter):
    backend = "trame" if _module_available("ipywidgets") and _module_available("trame") else "static"
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
    matlab_script = metadata["matlab_script"]
    kind = metadata["comparison_kind"] or "benchmark"
    family = metadata["family"]
    return dedent(
        f"""
        # {title}

        This notebook is generated from the shared benchmark notebook framework.

        - Benchmark folder: `{metadata["benchmark_name"]}`
        - Original MATLAB driver: `{matlab_script}`
        - Comparison kind: `{kind}`
        - Notebook family: `{family}`
        """
    ).strip()
