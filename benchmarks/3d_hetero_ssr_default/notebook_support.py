from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.tri as mtri
from matplotlib import colors as mcolors
import numpy as np

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")

import pyvista as pv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_TOML = CASE_DIR / "case.toml"
DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULT_MPIEXEC = shutil.which("mpiexec") or "mpiexec"
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SECTION_ORDER = (
    "problem",
    "execution",
    "continuation",
    "newton",
    "linear_solver",
    "export",
)

STRAIN_CMAP = "jet"


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
class DeviatoricStrainVisuals:
    volume_grid: pv.DataSet
    surface_grid: pv.PolyData
    element_values: np.ndarray


def load_case_sections(case_toml: Path = DEFAULT_CASE_TOML) -> dict[str, dict[str, Any]]:
    data = tomllib.loads(Path(case_toml).read_text(encoding="utf-8"))
    sections: dict[str, dict[str, Any]] = {}
    for name in SECTION_ORDER:
        value = data.get(name, {})
        sections[name] = dict(value) if isinstance(value, dict) else {}
    if not sections["export"]:
        sections["export"] = default_export_section()
    return sections


def default_export_section() -> dict[str, Any]:
    return {
        "write_custom_debug_bundle": True,
        "write_history_json": True,
        "write_solution_vtu": True,
        "custom_debug_name": "run_debug.h5",
        "history_name": "continuation_history.json",
        "solution_name": "final_solution.vtu",
    }


def render_case_toml(sections: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for section_name in SECTION_ORDER:
        section = sections.get(section_name)
        if not section:
            continue
        lines.append(f"[{section_name}]")
        for key, value in section.items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_case_toml(sections: dict[str, dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_case_toml(sections), encoding="utf-8")
    return path


def run_parallel_case(
    *,
    config_path: Path,
    out_dir: Path,
    mpi_ranks: int = 2,
    python_executable: Path = DEFAULT_PYTHON,
    mpiexec: str = DEFAULT_MPIEXEC,
    clean_out_dir: bool = True,
    poll_seconds: float = 0.25,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    config_path = Path(config_path).resolve()
    if clean_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    progress_position = _drain_progress(progress_path, progress_position)
    return_code = int(process.wait())

    if return_code != 0:
        raise RuntimeError(f"Parallel solve failed with exit code {return_code}.")

    artifacts = load_run_artifacts(out_dir)
    step_count = int(artifacts.run_info["run_info"]["step_count"])
    runtime = float(artifacts.run_info["run_info"]["runtime_seconds"])
    lambda_last = float(artifacts.npz["lambda_hist"][-1])
    omega_last = float(artifacts.npz["omega_hist"][-1])
    print("", flush=True)
    print("=== Finished ===", flush=True)
    print(f"Accepted steps: {step_count}", flush=True)
    print(f"Final lambda:   {lambda_last:.9f}", flush=True)
    print(f"Final omega:    {omega_last:.9f}", flush=True)
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


def show_run_summary(artifacts: RunArtifacts) -> None:
    run_info = artifacts.run_info["run_info"]
    mesh_info = artifacts.run_info["mesh"]
    timings = artifacts.run_info["timings"]
    print(json.dumps(run_info, indent=2))
    print("")
    print(json.dumps(mesh_info, indent=2))
    print("")
    print(json.dumps(timings, indent=2))


def plot_convergence_dashboard(artifacts: RunArtifacts):
    lambda_hist = np.asarray(artifacts.npz.get("lambda_hist", []), dtype=np.float64)
    omega_hist = np.asarray(artifacts.npz.get("omega_hist", []), dtype=np.float64)
    umax_hist = np.asarray(artifacts.npz.get("Umax_hist", []), dtype=np.float64)
    stats = dict(artifacts.history.get("stats", {}))
    step_linear_iterations = np.asarray(stats.get("step_linear_iterations", []), dtype=np.float64)
    step_wall_time = np.asarray(stats.get("step_wall_time", []), dtype=np.float64)
    accepted_steps = np.arange(1, max(len(lambda_hist), len(umax_hist)) + 1, dtype=np.int64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=160)
    ax = axes[0, 0]
    ax.plot(omega_hist, lambda_hist, marker="o", linewidth=1.25)
    ax.set_title("Continuation path")
    ax.set_xlabel("omega")
    ax.set_ylabel("lambda")
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(accepted_steps[: len(umax_hist)], umax_hist, marker="o", linewidth=1.25)
    ax.set_title("Displacement growth")
    ax.set_xlabel("accepted step")
    ax.set_ylabel("max |U|")
    ax.grid(True)

    ax = axes[1, 0]
    ax.bar(np.arange(1, len(step_linear_iterations) + 1), step_linear_iterations)
    ax.set_title("Linear iterations per accepted step")
    ax.set_xlabel("accepted step")
    ax.set_ylabel("iterations")
    ax.grid(True, axis="y")

    ax = axes[1, 1]
    ax.bar(np.arange(1, len(step_wall_time) + 1), step_wall_time)
    ax.set_title("Wall time per accepted step")
    ax.set_xlabel("accepted step")
    ax.set_ylabel("seconds")
    ax.grid(True, axis="y")

    fig.tight_layout()
    return fig


def plot_timing_breakdown(artifacts: RunArtifacts):
    timings = artifacts.run_info.get("timings", {})
    constitutive = dict(timings.get("constitutive", {}))
    linear = dict(timings.get("linear", {}))

    const_items = [(key, float(value)) for key, value in constitutive.items() if float(value) > 0.0]
    linear_items = [(key, float(value)) for key, value in linear.items() if "time" in key and float(value) > 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)

    ax = axes[0]
    if const_items:
        labels, values = zip(*const_items)
        ax.barh(labels, values)
    ax.set_title("Constitutive timings")
    ax.set_xlabel("seconds")

    ax = axes[1]
    if linear_items:
        labels, values = zip(*linear_items)
        ax.barh(labels, values)
    ax.set_title("Linear-solver timings")
    ax.set_xlabel("seconds")

    fig.tight_layout()
    return fig


def display_saved_solver_figures(artifacts: RunArtifacts):
    files = sorted(artifacts.plots_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No solver PNGs found in {artifacts.plots_dir}")

    n_cols = 2
    n_rows = int(np.ceil(len(files) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), dpi=140)
    axes_arr = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for ax in axes_arr.ravel():
        ax.axis("off")

    for ax, path in zip(axes_arr.ravel(), files, strict=False):
        ax.imshow(mpimg.imread(path))
        ax.set_title(path.name)
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_static_slices(
    grid: pv.DataSet,
    *,
    scalar: str = "displacement_magnitude",
):
    bounds = grid.bounds
    center = grid.center
    specs = [
        ("X-mid slice", (1.0, 0.0, 0.0), ((bounds[0] + bounds[1]) / 2.0, center[1], center[2]), (2, 1), ("z", "y")),
        ("Y-mid slice", (0.0, 1.0, 0.0), (center[0], (bounds[2] + bounds[3]) / 2.0, center[2]), (0, 2), ("x", "z")),
        ("Z-mid slice", (0.0, 0.0, 1.0), (center[0], center[1], (bounds[4] + bounds[5]) / 2.0), (0, 1), ("x", "y")),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=160)
    for ax, (title, normal, origin, dims, labels) in zip(axes, specs, strict=False):
        section = grid.slice(normal=normal, origin=origin).triangulate()
        if section.n_cells == 0:
            ax.text(0.5, 0.5, "Empty slice", ha="center", va="center")
            ax.set_title(title)
            ax.axis("off")
            continue

        pts = np.asarray(section.points, dtype=np.float64)
        faces = np.asarray(section.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
        triang = mtri.Triangulation(pts[:, dims[0]], pts[:, dims[1]], faces)
        if scalar in section.point_data:
            values = np.asarray(section.point_data[scalar], dtype=np.float64).reshape(-1)
            artist = ax.tripcolor(triang, values, shading="gouraud", cmap="viridis")
        elif scalar in section.cell_data:
            values = np.asarray(section.cell_data[scalar], dtype=np.float64).reshape(-1)
            artist = ax.tripcolor(triang, facecolors=values, cmap="viridis")
        else:
            raise KeyError(f"Scalar {scalar!r} not found on slice output.")
        ax.set_title(title)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_aspect("equal")
        ax.grid(False)
        fig.colorbar(artist, ax=ax, shrink=0.8, label=scalar)

    fig.tight_layout()
    return fig


def plot_static_deviatoric_slices(grid: pv.DataSet, *, scalar: str = "deviatoric_strain") -> list[plt.Figure]:
    bounds = grid.bounds
    center = grid.center
    vmin, vmax = _scalar_limits(grid, scalar)
    specs = [
        ("Deviatoric strain slice: x = mid", (1.0, 0.0, 0.0), ((bounds[0] + bounds[1]) / 2.0, center[1], center[2]), (2, 1), ("z", "y")),
        ("Deviatoric strain slice: y = mid", (0.0, 1.0, 0.0), (center[0], (bounds[2] + bounds[3]) / 2.0, center[2]), (0, 2), ("x", "z")),
        ("Deviatoric strain slice: z = mid", (0.0, 0.0, 1.0), (center[0], center[1], (bounds[4] + bounds[5]) / 2.0), (0, 1), ("x", "y")),
    ]
    figures: list[plt.Figure] = []
    for title, normal, origin, dims, labels in specs:
        section = grid.slice(normal=normal, origin=origin).triangulate()
        fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
        if section.n_cells == 0:
            ax.text(0.5, 0.5, "Empty slice", ha="center", va="center")
            ax.set_title(title)
            ax.axis("off")
            figures.append(fig)
            continue

        pts = np.asarray(section.points, dtype=np.float64)
        faces = np.asarray(section.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
        triang = mtri.Triangulation(pts[:, dims[0]], pts[:, dims[1]], faces)
        if scalar in section.point_data:
            values = np.asarray(section.point_data[scalar], dtype=np.float64).reshape(-1)
            artist = ax.tripcolor(triang, values, shading="gouraud", cmap=STRAIN_CMAP, vmin=vmin, vmax=vmax)
        elif scalar in section.cell_data:
            values = np.asarray(section.cell_data[scalar], dtype=np.float64).reshape(-1)
            artist = ax.tripcolor(triang, facecolors=values, cmap=STRAIN_CMAP, vmin=vmin, vmax=vmax)
        else:
            raise KeyError(f"Scalar {scalar!r} not found on slice output.")

        ax.set_title(title)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_aspect("equal")
        ax.grid(False)
        fig.colorbar(artist, ax=ax, shrink=0.82, label=scalar)
        fig.tight_layout()
        figures.append(fig)
    return figures


async def ensure_trame_server(**kwargs):
    from pyvista.trame.jupyter import launch_server

    server = launch_server(**kwargs)
    await server.ready
    return server


def configure_pyvista_backend(preferred: tuple[str, ...] = ("client", "html", "static")) -> str | None:
    for backend in preferred:
        try:
            pv.set_jupyter_backend(backend)
            return backend
        except Exception:
            continue
    return None


def load_solution_grid(artifacts: RunArtifacts) -> pv.DataSet:
    if not artifacts.vtu_path.exists():
        raise FileNotFoundError(f"Expected VTU export at {artifacts.vtu_path}")
    return pv.read(artifacts.vtu_path)


def prepare_deviatoric_strain_visuals(
    artifacts: RunArtifacts,
    config_path: Path,
    *,
    grid: pv.DataSet | None = None,
) -> DeviatoricStrainVisuals:
    cfg, coord, elem, surf, U = _load_reordered_solution_state(artifacts, config_path)
    elem_strain = _compute_element_deviatoric_strain(coord, elem, cfg.problem.elem_type, U)

    volume_grid = load_solution_grid(artifacts) if grid is None else grid.copy()
    if volume_grid.n_cells != elem.shape[1]:
        raise ValueError(
            f"VTU cell count {volume_grid.n_cells} does not match reordered mesh element count {elem.shape[1]}."
        )
    volume_grid.cell_data["deviatoric_strain"] = elem_strain
    strain_limits = _nonnegative_limits(elem_strain)
    _store_scalar_limits(volume_grid, "deviatoric_strain", strain_limits)
    strain_point_grid = volume_grid.cell_data_to_point_data(pass_cell_data=False)
    if "deviatoric_strain" in strain_point_grid.point_data:
        volume_grid.point_data["deviatoric_strain_point"] = np.asarray(
            strain_point_grid.point_data["deviatoric_strain"],
            dtype=np.float64,
        )
        _store_scalar_limits(volume_grid, "deviatoric_strain_point", strain_limits)

    tri_surface, tri_face_ids = _build_plotting_mesh_with_face_ids(surf)
    face_parent = _surface_parent_elements(elem, surf)
    tri_vals = elem_strain[face_parent[tri_face_ids]]
    surface_grid = volume_grid.extract_surface(algorithm="dataset_surface").triangulate()
    _store_scalar_limits(surface_grid, "deviatoric_strain", strain_limits)
    if "deviatoric_strain_point" in surface_grid.point_data:
        _store_scalar_limits(surface_grid, "deviatoric_strain_point", strain_limits)

    return DeviatoricStrainVisuals(
        volume_grid=volume_grid,
        surface_grid=surface_grid,
        element_values=elem_strain,
    )


def build_deformed_plotter(
    grid: pv.DataSet,
    *,
    scalar: str = "displacement_magnitude",
    scale_factor: float = 1.0,
    off_screen: bool = False,
) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=off_screen)
    warped = grid.warp_by_vector("displacement", factor=scale_factor)
    plotter.add_mesh(warped, scalars=scalar, cmap="viridis", show_edges=False)
    plotter.add_axes()
    plotter.show_grid()
    return plotter


def build_deformed_notebook_widget(
    grid: pv.DataSet,
    *,
    scalar: str = "displacement_magnitude",
    scale_factor: float = 1.0,
    backend: str | None = None,
):
    import ipywidgets as widgets

    plotter, viewer, _ = _show_existing_plotter_notebook(
        build_deformed_plotter(
            grid,
            scalar=scalar,
            scale_factor=scale_factor,
        ),
        preferred_backend=backend,
    )
    if hasattr(viewer, "layout"):
        viewer.layout.width = "100%"
    container = widgets.VBox([viewer], layout=widgets.Layout(width="100%", align_items="stretch"))
    container._plotter = plotter
    container._viewer = viewer
    return container


def build_deviatoric_surface_plotter(
    surface_grid: pv.PolyData,
    *,
    scalar: str = "deviatoric_strain",
    off_screen: bool = False,
    show_scalar_bar: bool = False,
) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=off_screen)
    clim = _scalar_limits(surface_grid, scalar)
    render_scalar = _surface_render_scalar(surface_grid, scalar)
    plotter.add_mesh(
        surface_grid,
        scalars=render_scalar,
        cmap=_strain_lookup_table(clim),
        clim=clim,
        show_edges=False,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args=_strain_scalar_bar_args(),
        preference="point",
        interpolate_before_map=True,
        n_colors=2048,
    )
    plotter.add_axes()
    plotter.show_grid()
    return plotter


def build_deviatoric_surface_notebook_widget(
    surface_grid: pv.PolyData,
    *,
    scalar: str = "deviatoric_strain",
    backend: str | None = None,
):
    import ipywidgets as widgets

    clim = _scalar_limits(surface_grid, scalar)
    plotter, viewer, used_backend = _show_notebook_plotter(
        lambda show_scalar_bar: build_deviatoric_surface_plotter(
            surface_grid,
            scalar=scalar,
            show_scalar_bar=show_scalar_bar,
        ),
        preferred_backend=backend,
    )
    if hasattr(viewer, "layout"):
        viewer.layout.width = "100%"
    if _uses_builtin_scalar_bar(used_backend):
        container = widgets.VBox([viewer], layout=widgets.Layout(width="100%", align_items="stretch"))
        container._plotter = plotter
        container._viewer = viewer
        container._backend = used_backend
        return container
    colorbar = _build_scalar_bar_widget(label="deviatoric strain norm", cmap_name=STRAIN_CMAP, vmin=clim[0], vmax=clim[1], orientation="horizontal")
    container = widgets.VBox([viewer, colorbar], layout=widgets.Layout(width="100%", align_items="stretch"))
    container._plotter = plotter
    container._viewer = viewer
    container._backend = used_backend
    return container


def build_axis_slice_plotter(
    grid: pv.DataSet,
    *,
    axis: str,
    scalar: str = "deviatoric_strain",
    off_screen: bool = False,
    show_scalar_bar: bool = False,
) -> pv.Plotter:
    center, slice_min, slice_max, slice0 = _axis_slice_parameters(grid, axis)
    clim = _scalar_limits(grid, scalar)
    render_scalar = _slice_render_scalar(grid, scalar)
    axis_name = _axis_name(axis)
    actor_name = f"{axis_name}-slice"

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(grid.outline(), color="black", line_width=1.0)

    scalar_kwargs = {
        "scalars": render_scalar,
        "cmap": _strain_lookup_table(clim),
        "clim": clim,
        "show_edges": False,
        "show_scalar_bar": show_scalar_bar,
        "scalar_bar_args": _strain_scalar_bar_args(),
        "preference": "point",
        "interpolate_before_map": True,
        "n_colors": 2048,
    }
    plotter.add_mesh(_slice_at_axis(grid, center, slice0, axis), name=actor_name, **scalar_kwargs)

    def _update(slice_value: float) -> None:
        plotter.add_mesh(_slice_at_axis(grid, center, slice_value, axis), name=actor_name, reset_camera=False, **scalar_kwargs)

    plotter.add_slider_widget(
        _update,
        rng=(slice_min, slice_max),
        value=slice0,
        title=f"{axis_name} slice",
        pointa=(0.25, 0.08),
        pointb=(0.75, 0.08),
        interaction_event="always",
    )
    plotter.add_axes()
    plotter.show_grid()
    return plotter


def build_axis_slice_notebook_widget(
    grid: pv.DataSet,
    *,
    axis: str,
    scalar: str = "deviatoric_strain",
    backend: str | None = None,
):
    import ipywidgets as widgets

    center, slice_min, slice_max, slice0 = _axis_slice_parameters(grid, axis)
    clim = _scalar_limits(grid, scalar)
    render_scalar = _slice_render_scalar(grid, scalar)
    axis_name = _axis_name(axis)
    actor_name = f"{axis_name}-slice"
    scalar_kwargs = {
        "scalars": render_scalar,
        "cmap": _strain_lookup_table(clim),
        "clim": clim,
        "show_edges": False,
        "show_scalar_bar": False,
        "preference": "point",
        "interpolate_before_map": True,
        "n_colors": 2048,
    }

    def _make_plotter(show_scalar_bar: bool) -> pv.Plotter:
        plotter = pv.Plotter()
        plotter.add_mesh(grid.outline(), color="black", line_width=1.0)
        plotter.add_mesh(
            _slice_at_axis(grid, center, slice0, axis),
            name=actor_name,
            **{**scalar_kwargs, "show_scalar_bar": show_scalar_bar},
        )
        plotter.add_axes()
        plotter.show_grid()
        return plotter

    plotter, viewer, used_backend = _show_notebook_plotter(_make_plotter, preferred_backend=backend)
    if hasattr(viewer, "layout"):
        viewer.layout.width = "100%"

    status = widgets.HTML(f"<b>{axis_name} slice</b>: {slice0:.6g}")
    slider = widgets.FloatSlider(
        value=slice0,
        min=slice_min,
        max=slice_max,
        step=max((slice_max - slice_min) / 200.0, 1e-9),
        description=axis_name,
        readout=True,
        readout_format=".6g",
        continuous_update=True,
        layout=widgets.Layout(width="100%"),
    )

    def _update(change: dict[str, Any]) -> None:
        if change.get("name") != "value":
            return
        slice_value = float(change["new"])
        plotter.add_mesh(
            _slice_at_axis(grid, center, slice_value, axis),
            name=actor_name,
            reset_camera=False,
            render=False,
            remove_existing_actor=True,
            **{
                **scalar_kwargs,
                "show_scalar_bar": _uses_builtin_scalar_bar(used_backend),
                "scalar_bar_args": _strain_scalar_bar_args(),
            },
        )
        status.value = f"<b>{axis_name} slice</b>: {slice_value:.6g}"
        plotter.render()

    slider.observe(_update, names="value")
    if _uses_builtin_scalar_bar(used_backend):
        container = widgets.VBox([status, slider, viewer], layout=widgets.Layout(width="100%", align_items="stretch"))
        container._plotter = plotter
        container._viewer = viewer
        container._backend = used_backend
        return container
    colorbar = _build_scalar_bar_widget(label="deviatoric strain norm", cmap_name=STRAIN_CMAP, vmin=clim[0], vmax=clim[1], orientation="horizontal")
    container = widgets.VBox([status, slider, viewer, colorbar], layout=widgets.Layout(width="100%", align_items="stretch"))
    container._plotter = plotter
    container._viewer = viewer
    container._backend = used_backend
    return container


def build_x_slice_plotter(
    grid: pv.DataSet,
    *,
    scalar: str = "deviatoric_strain",
    off_screen: bool = False,
    show_scalar_bar: bool = False,
) -> pv.Plotter:
    return build_axis_slice_plotter(
        grid,
        axis="x",
        scalar=scalar,
        off_screen=off_screen,
        show_scalar_bar=show_scalar_bar,
    )


def build_y_slice_plotter(
    grid: pv.DataSet,
    *,
    scalar: str = "deviatoric_strain",
    off_screen: bool = False,
    show_scalar_bar: bool = False,
) -> pv.Plotter:
    return build_axis_slice_plotter(
        grid,
        axis="y",
        scalar=scalar,
        off_screen=off_screen,
        show_scalar_bar=show_scalar_bar,
    )


def build_z_slice_plotter(
    grid: pv.DataSet,
    *,
    scalar: str = "deviatoric_strain",
    off_screen: bool = False,
    show_scalar_bar: bool = False,
) -> pv.Plotter:
    return build_axis_slice_plotter(
        grid,
        axis="z",
        scalar=scalar,
        off_screen=off_screen,
        show_scalar_bar=show_scalar_bar,
    )


def build_x_slice_notebook_widget(
    grid: pv.DataSet,
    *,
    scalar: str = "deviatoric_strain",
    backend: str | None = None,
):
    return build_axis_slice_notebook_widget(grid, axis="x", scalar=scalar, backend=backend)


def build_y_slice_notebook_widget(
    grid: pv.DataSet,
    *,
    scalar: str = "deviatoric_strain",
    backend: str | None = None,
):
    return build_axis_slice_notebook_widget(grid, axis="y", scalar=scalar, backend=backend)


def build_z_slice_notebook_widget(
    grid: pv.DataSet,
    *,
    scalar: str = "deviatoric_strain",
    backend: str | None = None,
):
    return build_axis_slice_notebook_widget(grid, axis="z", scalar=scalar, backend=backend)


def build_material_plotter(
    grid: pv.DataSet,
    *,
    material_id: int | None = None,
    scalar: str = "material_id",
    off_screen: bool = False,
) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=off_screen)
    target = grid
    if material_id is not None:
        target = grid.threshold(
            value=(float(material_id), float(material_id)),
            scalars=scalar,
            preference="cell",
        )
    plotter.add_mesh(target, scalars=scalar, cmap="tab10", show_edges=False)
    plotter.add_axes()
    plotter.show_grid()
    return plotter


def save_plotter_screenshot(plotter: pv.Plotter, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(path)
    return path


def plot_matlab_style_strain_surface(
    artifacts: RunArtifacts,
    config_path: Path,
):
    visuals = prepare_deviatoric_strain_visuals(artifacts, config_path)
    cfg, coord, elem, surf, _ = _load_reordered_solution_state(artifacts, config_path)
    tri_surface, tri_face_ids = _build_plotting_mesh_with_face_ids(surf)
    face_parent = _surface_parent_elements(elem, surf)
    tri_vals = visuals.element_values[face_parent[tri_face_ids]]
    vmin, vmax = _scalar_limits(visuals.volume_grid, "deviatoric_strain")

    cmap = plt.get_cmap(STRAIN_CMAP)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    triangles = [coord[:, tri_nodes].T for tri_nodes in tri_surface]
    face_colors = cmap(norm(tri_vals))
    mesh_poly = Poly3DCollection(triangles, facecolors=face_colors, edgecolor="none", alpha=0.95)
    ax.add_collection3d(mesh_poly)
    ax.set_title("Deviatoric strain (undeformed boundary surface)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(np.asarray([vmin, vmax], dtype=np.float64))
    fig.colorbar(mappable, ax=ax, pad=0.1, shrink=0.7, label="deviatoric strain norm")
    _set_axes_equal(ax)
    fig.tight_layout()
    return fig


def _drain_progress(progress_path: Path, position: int) -> int:
    if not progress_path.exists():
        return position
    with progress_path.open("r", encoding="utf-8") as handle:
        handle.seek(position)
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            print(_format_progress_event(event), flush=True)
        return handle.tell()


def _format_progress_event(event: dict[str, Any]) -> str:
    kind = str(event.get("event", "event"))
    if kind == "init_complete":
        lambda_hist = event.get("lambda_hist", [])
        omega_hist = event.get("omega_hist", [])
        return (
            "[rank0-progress] init_complete "
            f"accepted_steps={event.get('accepted_steps')} "
            f"lambda_hist={lambda_hist} omega_hist={omega_hist}"
        )
    if kind == "attempt_complete":
        return (
            "[rank0-progress] attempt_complete "
            f"target_step={event.get('target_step')} "
            f"attempt_in_step={event.get('attempt_in_step')} "
            f"success={event.get('success')} "
            f"lambda_before={event.get('lambda_before')} "
            f"lambda_after={event.get('lambda_after')} "
            f"newton_iterations={event.get('newton_iterations')} "
            f"linear_iterations={event.get('linear_iterations')}"
        )
    if kind == "step_accepted":
        return (
            "[rank0-progress] step_accepted "
            f"accepted_step={event.get('accepted_step')} "
            f"lambda={event.get('lambda_value')} "
            f"omega={event.get('omega_value')} "
            f"u_max={event.get('u_max')} "
            f"wall={event.get('step_wall_time')}"
        )
    if kind == "finished":
        return (
            "[rank0-progress] finished "
            f"accepted_steps={event.get('accepted_steps')} "
            f"lambda_last={event.get('lambda_last')} "
            f"omega_last={event.get('omega_last')} "
            f"total_wall_time={event.get('total_wall_time')}"
        )
    return f"[rank0-progress] {json.dumps(event)}"


def _surface_faces_by_width(surf: np.ndarray) -> np.ndarray:
    surf = np.asarray(surf, dtype=np.int64)
    if surf.ndim != 2:
        raise ValueError(f"Expected a 2D surface array, got shape {surf.shape}")
    if surf.shape[0] == 6:
        return surf.T.astype(np.int64)
    if surf.shape[1] == 6:
        return surf.astype(np.int64)
    if surf.shape[0] == 15:
        return surf[:3, :].T.astype(np.int64)
    if surf.shape[1] == 15:
        return surf[:, :3].astype(np.int64)
    if surf.shape[0] == 3:
        return surf.T.astype(np.int64)
    if surf.shape[1] == 3:
        return surf.astype(np.int64)
    raise ValueError(f"Unsupported surface array shape {surf.shape}")


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


def _deviatoric_strain_norm(strain: np.ndarray) -> np.ndarray:
    iota = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    dev = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) - np.outer(iota, iota) / 3.0
    dev_e = dev @ np.asarray(strain, dtype=np.float64)
    return np.sqrt(np.maximum(0.0, np.sum(strain * dev_e, axis=0)))


def _set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range) * 0.6
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d(x_mid - max_range, x_mid + max_range)
    ax.set_ylim3d(y_mid - max_range, y_mid + max_range)
    ax.set_zlim3d(z_mid - max_range, z_mid + max_range)


def _axis_slice_parameters(grid: pv.DataSet, axis: str) -> tuple[tuple[float, float, float], float, float, float]:
    bounds = grid.bounds
    center = tuple(float(value) for value in grid.center)
    axis_idx = _axis_index(axis)
    slice_min = float(bounds[2 * axis_idx])
    slice_max = float(bounds[2 * axis_idx + 1])
    slice0 = float(center[axis_idx])
    return center, slice_min, slice_max, slice0


def _y_slice_parameters(grid: pv.DataSet) -> tuple[tuple[float, float, float], float, float, float]:
    return _axis_slice_parameters(grid, "y")


def _slice_at_axis(
    grid: pv.DataSet,
    center: tuple[float, float, float],
    slice_value: float,
    axis: str,
) -> pv.DataSet:
    axis_idx = _axis_index(axis)
    normal = [0.0, 0.0, 0.0]
    normal[axis_idx] = 1.0
    origin = [center[0], center[1], center[2]]
    origin[axis_idx] = float(slice_value)
    return grid.slice(normal=tuple(normal), origin=tuple(origin)).triangulate()


def _slice_at_y(
    grid: pv.DataSet,
    center: tuple[float, float, float],
    y_value: float,
) -> pv.DataSet:
    return _slice_at_axis(grid, center, y_value, "y")


def _scalar_limits(grid: pv.DataSet, scalar: str) -> tuple[float, float]:
    stored = _read_scalar_limits(grid, scalar)
    if stored is not None:
        return stored

    if scalar in grid.point_data:
        values = np.asarray(grid.point_data[scalar], dtype=np.float64).reshape(-1)
    elif scalar in grid.cell_data:
        values = np.asarray(grid.cell_data[scalar], dtype=np.float64).reshape(-1)
    else:
        raise KeyError(f"Scalar {scalar!r} was not found on the provided grid.")

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Scalar {scalar!r} does not contain any finite values.")

    vmin = float(finite.min())
    vmax = float(finite.max())
    if np.isclose(vmin, vmax):
        pad = max(abs(vmax) * 1e-9, 1e-12)
        return vmin - pad, vmax + pad
    return vmin, vmax


def _store_scalar_limits(grid: pv.DataSet, scalar: str, limits: tuple[float, float]) -> None:
    grid.field_data[_scalar_limits_key(scalar)] = np.asarray(limits, dtype=np.float64)


def _read_scalar_limits(grid: pv.DataSet, scalar: str) -> tuple[float, float] | None:
    key = _scalar_limits_key(scalar)
    if key not in grid.field_data:
        return None
    values = np.asarray(grid.field_data[key], dtype=np.float64).reshape(-1)
    if values.size != 2:
        return None
    return float(values[0]), float(values[1])


def _scalar_limits_key(scalar: str) -> str:
    return f"{scalar}_limits"


def _nonnegative_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot build scalar limits from an array without finite values.")
    vmax = float(finite.max())
    if np.isclose(vmax, 0.0):
        return 0.0, 1e-12
    return 0.0, vmax


def _build_scalar_bar_widget(
    *,
    label: str,
    cmap_name: str,
    vmin: float,
    vmax: float,
    orientation: str = "vertical",
):
    import ipywidgets as widgets

    if orientation == "horizontal":
        fig = plt.figure(figsize=(5.8, 0.9), dpi=170)
        ax = fig.add_axes([0.08, 0.42, 0.84, 0.28])
    else:
        fig = plt.figure(figsize=(1.3, 4.2), dpi=170)
        ax = fig.add_axes([0.40, 0.08, 0.28, 0.84])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
    mappable.set_array(np.asarray([vmin, vmax], dtype=np.float64))
    colorbar = fig.colorbar(mappable, cax=ax, orientation=orientation)
    colorbar.set_label(label)
    colorbar.ax.tick_params(labelsize=9)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buffer.seek(0)

    title = widgets.HTML(
        f"<div><b>{label}</b><br><span style='font-size: 12px;'>range: {vmin:.6g} .. {vmax:.6g}</span></div>"
    )
    image = widgets.Image(
        value=buffer.getvalue(),
        format="png",
        layout=widgets.Layout(width="100%"),
    )
    return widgets.VBox(
        [title, image],
        layout=widgets.Layout(width="100%", align_items="center"),
    )


def _show_notebook_plotter(plotter_factory, *, preferred_backend: str | None):
    candidates = _strain_backend_candidates(preferred_backend)
    last_error: Exception | None = None
    for backend in candidates:
        show_scalar_bar = _uses_builtin_scalar_bar(backend)
        plotter = plotter_factory(show_scalar_bar)
        try:
            viewer = plotter.show(
                jupyter_backend=backend,
                return_viewer=True,
                interactive_update=True,
            )
        except Exception as exc:
            last_error = exc
            try:
                plotter.close()
            except Exception:
                pass
            continue
        if viewer is not None:
            return plotter, viewer, backend
        try:
            plotter.close()
        except Exception:
            pass
    if last_error is not None:
        raise last_error
    raise RuntimeError("The PyVista notebook viewer must be created inside a live Jupyter notebook cell.")


def _show_existing_plotter_notebook(plotter: pv.Plotter, *, preferred_backend: str | None):
    candidates = _strain_backend_candidates(preferred_backend)
    last_error: Exception | None = None
    for backend in candidates:
        try:
            viewer = plotter.show(
                jupyter_backend=backend,
                return_viewer=True,
                interactive_update=True,
            )
        except Exception as exc:
            last_error = exc
            continue
        if viewer is not None:
            return plotter, viewer, backend
    if last_error is not None:
        raise last_error
    raise RuntimeError("The PyVista notebook viewer must be created inside a live Jupyter notebook cell.")


def _strain_backend_candidates(preferred_backend: str | None) -> tuple[str, ...]:
    candidates = ["server"]
    if preferred_backend:
        candidates.append(preferred_backend)
    candidates.extend(["client", "html", "static"])
    ordered: list[str] = []
    for backend in candidates:
        if backend not in ordered:
            ordered.append(backend)
    return tuple(ordered)


def _uses_builtin_scalar_bar(backend: str | None) -> bool:
    return backend in {"server", "trame"}


def _strain_lookup_table(clim: tuple[float, float]) -> pv.LookupTable:
    lut = pv.LookupTable(cmap=STRAIN_CMAP, n_values=2048, flip=False)
    lut.scalar_range = clim
    return lut


def _strain_scalar_bar_args() -> dict[str, Any]:
    return {
        "title": "deviatoric strain norm",
        "fmt": "%.4g",
        "vertical": False,
        "n_labels": 5,
        "position_x": 0.22,
        "position_y": 0.04,
        "width": 0.56,
        "height": 0.08,
        "title_font_size": 14,
        "label_font_size": 11,
        "unconstrained_font_size": True,
    }


def _slice_render_scalar(grid: pv.DataSet, scalar: str) -> str:
    point_scalar = f"{scalar}_point"
    if point_scalar in grid.point_data:
        return point_scalar
    return scalar


def _surface_render_scalar(grid: pv.DataSet, scalar: str) -> str:
    return _slice_render_scalar(grid, scalar)


def _axis_name(axis: str) -> str:
    axis_name = str(axis).strip().lower()
    if axis_name not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported slice axis {axis!r}. Expected one of 'x', 'y', or 'z'.")
    return axis_name


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[_axis_name(axis)]


def _load_reordered_solution_state(
    artifacts: RunArtifacts,
    config_path: Path,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from slope_stability.core.run_config import load_run_case_config
    from slope_stability.mesh import load_mesh_from_file, reorder_mesh_nodes

    cfg = load_run_case_config(config_path)
    part_count = (
        int(artifacts.run_info["run_info"].get("mpi_size", 1))
        if cfg.execution.node_ordering.lower() == "block_metis"
        else None
    )
    mesh = load_mesh_from_file(
        cfg.problem.mesh_path,
        boundary_type=cfg.problem.mesh_boundary_type,
        elem_type=cfg.problem.elem_type,
    )
    reordered = reorder_mesh_nodes(
        mesh.coord,
        mesh.elem,
        mesh.surf,
        mesh.q_mask,
        strategy=cfg.execution.node_ordering,
        n_parts=part_count,
    )
    coord = np.asarray(reordered.coord, dtype=np.float64)
    elem = np.asarray(reordered.elem, dtype=np.int64)
    surf = np.asarray(reordered.surf, dtype=np.int64)
    U = np.asarray(artifacts.npz["U"], dtype=np.float64)
    if U.shape[1] != coord.shape[1]:
        raise ValueError(f"Displacement field shape {U.shape} does not match reordered mesh nodes {coord.shape[1]}.")
    return cfg, coord, elem, surf, U


def _compute_element_deviatoric_strain(
    coord: np.ndarray,
    elem: np.ndarray,
    elem_type: str,
    U: np.ndarray,
) -> np.ndarray:
    from slope_stability.fem import assemble_strain_operator

    assembly = assemble_strain_operator(coord, elem, elem_type, dim=3)
    strain = assembly.B @ U.reshape(-1, order="F")
    strain = strain.reshape(6, -1, order="F")
    dev_norm = _deviatoric_strain_norm(strain)
    n_q = dev_norm.size // elem.shape[1]
    return np.mean(dev_norm.reshape(n_q, elem.shape[1], order="F"), axis=0)


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
